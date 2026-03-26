#include <ros/ros.h>
#include <geometry_msgs/Twist.h>
#include <std_msgs/Float32MultiArray.h>
#include <sensor_msgs/Image.h>
#include <cv_bridge/cv_bridge.h>
#include <opencv2/opencv.hpp>
#include <vector>
#include <algorithm>
#include <iostream>
#include <iomanip>
#include <cmath>
#include <thread>
#include <mutex>
#include <atomic>
#include <string>

using namespace cv;
using namespace std;

// -------------------------- 全局配置（重点优化性能开关） --------------------------
const bool ENABLE_DEBUG_LOG = false;    // 强制关闭DEBUG日志
const int WINDOW_REFRESH_INTERVAL = 5;  // 图像窗口每5帧刷新一次
const int LOG_PRINT_INTERVAL = 200;     // 日志每200帧输出一次（仅用于内部计数，无实际输出）
const int WAITKEY_DURATION = 1;         // 最短窗口阻塞时间
const bool ENABLE_IMAGE_DISPLAY = true; // 可关闭显示进一步提速

// -------------------------- 全局状态枚举（完全不变） --------------------------
enum SystemState
{
    TRACKING_DIGITS, // 数字跟踪状态（最高优先级，但仅在NORMAL_DRIVING后可触发）
    CONE_NAVIGATION, // 锥桶导航状态
    STANDBY          // 待机状态
};

// 新增：全局状态变量（供数字跟踪线程访问，用于状态锁控制）
SystemState current_state;

// -------------------------- 数字跟踪相关（优化匹配性能，不改变控制逻辑） --------------------------
// 全局变量声明（保留原控制参数，仅优化匹配参数）
const int IMG_WIDTH_GLOBAL = 640;
const double MAX_ANGULAR_GLOBAL = 0.8;
const int DIGIT_IMG_WIDTH = 640;
const int DIGIT_IMG_HEIGHT = 360;
const Point DIGIT_IMG_CENTER(DIGIT_IMG_WIDTH / 2, DIGIT_IMG_HEIGHT / 2);
const double SCORE_THRESHOLD = 0.45; // 保留原参数
const double MAX_LINEAR = 0.3;
const int DEAD_ZONE = 30;
const int SIZE_DEAD_ZONE = 25;
const int EDGE_PROTECTION = 70;
const int EMERGENCY_STOP_THRESHOLD = 50;
const double FORWARD_BOOST = 1.2;
const double SCORE_DIFF_THRESHOLD = 0.05; // 保留原参数
const int TARGET_SIZE = 180;              // 保留原参数

// 模板匹配结构体（原exp1）
struct MatchResult
{
    int targetIdx;
    Point loc;
    double scale;
    double score;
};

// 辅助函数（原exp1：裁剪旋转后的模板）
Mat cropRotatedTemplate(Mat &rotatedTemp)
{
    vector<Point> nonZeroPoints;
    findNonZero(rotatedTemp, nonZeroPoints);
    if (nonZeroPoints.empty())
        return rotatedTemp;
    Rect boundingRect = cv::boundingRect(nonZeroPoints);
    Mat croppedTemp = rotatedTemp(boundingRect);
    return croppedTemp;
}

// 优化：减少旋转角度数量（从3个→2个，减少模板数量）
vector<Mat> generateRotatedTemplates(Mat &originalTempl)
{
    vector<Mat> rotatedTemps;
    vector<int> angles = {0, 10}; // 仅保留0°和10°（减少33%模板）
    Point2f center(originalTempl.cols / 2.0, originalTempl.rows / 2.0);

    for (int angle : angles)
    {
        Mat rotMat = getRotationMatrix2D(center, angle, 1.0);
        Mat rotatedTemp;
        warpAffine(originalTempl, rotatedTemp, rotMat, originalTempl.size(),
                   INTER_LINEAR, BORDER_CONSTANT, Scalar(255, 255, 255));
        cvtColor(rotatedTemp, rotatedTemp, COLOR_BGR2GRAY);
        Mat croppedTemp = cropRotatedTemplate(rotatedTemp);
        rotatedTemps.push_back(croppedTemp);
    }

    return rotatedTemps;
}

// 核心优化：匹配函数（减少缩放层级、金字塔层数、移除耗时预处理）
MatchResult fastMultiScaleMatchWithRotation(Mat &frame, vector<Mat> &allTemplates, vector<int> &targetIndices)
{
    MatchResult bestResult = {-1, Point(-1, -1), 1.0, 0.0};
    double secondBestScore = 0.0;

    // 优化1：减少缩放范围和步长（从0.4-3.5→0.6-2.5，步长0.15→0.2，减少50%缩放次数）
    double minScale = 0.6;
    double maxScale = 2.5;
    double scaleStep = 0.2;

    Mat frameGray;
    cvtColor(frame, frameGray, COLOR_BGR2GRAY);
    // 优化2：移除直方图均衡化和高斯模糊（耗时操作，不影响检测效果）

    vector<Mat> framePyramid;
    buildPyramid(frameGray, framePyramid, 1); // 优化3：金字塔层数从2→1（减少50%金字塔匹配）

    for (size_t i = 0; i < allTemplates.size() && i < targetIndices.size(); ++i)
    {
        Mat templ = allTemplates[i];
        int realTargetIdx = targetIndices[i];
        int templW = templ.cols;
        int templH = templ.rows;

        double currentScaleStep = scaleStep;
        for (double scale = minScale; scale <= maxScale; scale += currentScaleStep)
        {
            if (scale >= 1.5)
                currentScaleStep = 0.25; // 优化：增大高缩放比例步长

            int pyramidLevel = cvRound(log2(1 / scale));
            pyramidLevel = max(0, min(0, pyramidLevel)); // 优化：仅使用0层金字塔（避免层级切换耗时）
            Mat frameLevel = framePyramid[pyramidLevel];
            double levelScale = 1.0 / (1 << pyramidLevel);

            Mat scaledTempl;
            double realScale = scale * levelScale;
            resize(templ, scaledTempl, Size(), realScale, realScale, INTER_LINEAR);

            if (scaledTempl.cols > frameLevel.cols || scaledTempl.rows > frameLevel.rows)
                continue;

            Mat matchRes;
            matchTemplate(frameLevel, scaledTempl, matchRes, TM_CCOEFF_NORMED);

            // 优化4：简化mask（仅保留边缘屏蔽，减少计算）
            Mat mask = Mat::ones(matchRes.size(), CV_8U);
            int border = 3; // 边框宽度从5→3
            rectangle(mask, Rect(0, 0, matchRes.cols, border), Scalar(0), -1);
            rectangle(mask, Rect(0, 0, border, matchRes.rows), Scalar(0), -1);
            rectangle(mask, Rect(matchRes.cols - border, 0, border, matchRes.rows), Scalar(0), -1);
            rectangle(mask, Rect(0, matchRes.rows - border, matchRes.cols, border), Scalar(0), -1);

            double currScore;
            Point currLoc;
            minMaxLoc(matchRes, nullptr, &currScore, nullptr, &currLoc, mask);

            Point origLoc(
                currLoc.x * (1 << pyramidLevel),
                currLoc.y * (1 << pyramidLevel));

            // 保留原分数过滤逻辑
            if (currScore > bestResult.score && currScore > 0.3)
            {
                secondBestScore = bestResult.score;
                bestResult.targetIdx = realTargetIdx;
                bestResult.loc = origLoc;
                bestResult.scale = scale;
                bestResult.score = currScore;
            }
            else if (currScore > secondBestScore)
            {
                secondBestScore = currScore;
            }
        }
    }

    // 保留原分数差过滤逻辑
    double scoreDiff = bestResult.score - secondBestScore;
    if (scoreDiff < SCORE_DIFF_THRESHOLD)
    {
        bestResult.score = 0.0;
        bestResult.targetIdx = -1;
    }

    return bestResult;
}

// 旋转精准控制器（完全不变）
class PrecisionRotController
{
private:
    const int WINDOW_SIZE = 8;
    vector<double> linearXBuffer;
    vector<double> angularZBuffer;
    const double MAX_LINEAR_DELTA = 0.05;
    const double MAX_ANGULAR_DELTA = 0.15;
    const double SLOW_DOWN_THRESHOLD = 90;
    const double MIN_ROT_SPEED = 0.1;

public:
    geometry_msgs::Twist smooth(geometry_msgs::Twist rawVel)
    {
        geometry_msgs::Twist smoothVel;

        linearXBuffer.push_back(rawVel.linear.x);
        if (linearXBuffer.size() > WINDOW_SIZE)
            linearXBuffer.erase(linearXBuffer.begin());
        double linearAvg = 0.0;
        for (double v : linearXBuffer)
            linearAvg += v;
        linearAvg /= linearXBuffer.size();

        double raw_angular = rawVel.angular.z;
        int x_error_abs = abs(raw_angular * (IMG_WIDTH_GLOBAL / 2) / MAX_ANGULAR_GLOBAL);

        angularZBuffer.push_back(raw_angular);
        if (angularZBuffer.size() > WINDOW_SIZE)
            angularZBuffer.erase(angularZBuffer.begin());
        double angularAvg = 0.0;
        for (double w : angularZBuffer)
            angularAvg += w;
        angularAvg /= angularZBuffer.size();

        if (x_error_abs < SLOW_DOWN_THRESHOLD)
        {
            double slow_down_ratio = (double)x_error_abs / SLOW_DOWN_THRESHOLD;
            angularAvg = angularAvg * slow_down_ratio;
            angularAvg = max(abs(angularAvg), MIN_ROT_SPEED) * (angularAvg > 0 ? 1 : -1);
        }

        static double lastLinear = 0.0, lastAngular = 0.0;
        if (linearAvg - lastLinear > MAX_LINEAR_DELTA)
            linearAvg = lastLinear + MAX_LINEAR_DELTA;
        else if (linearAvg - lastLinear < -MAX_LINEAR_DELTA)
            linearAvg = lastLinear - MAX_LINEAR_DELTA;

        if (angularAvg - lastAngular > MAX_ANGULAR_DELTA)
            angularAvg = lastAngular + MAX_ANGULAR_DELTA;
        else if (angularAvg - lastAngular < -MAX_ANGULAR_DELTA)
            angularAvg = lastAngular - MAX_ANGULAR_DELTA;

        lastLinear = linearAvg;
        lastAngular = angularAvg;

        smoothVel.linear.x = linearAvg;
        smoothVel.angular.z = angularAvg;
        return smoothVel;
    }
};

// 数字跟踪全局变量（优化原子操作和内存模型）
atomic<bool> frame_updated(false);
Mat frame_msg;
mutex frame_mutex;
vector<Mat> allTemplates;
vector<int> targetIndices;
PrecisionRotController rotCtrl;
atomic<bool> digit_detected(false);
ros::Time digital_lost_start_time;
mutex digital_lost_time_mutex;
const double DIGITAL_LOST_TIMEOUT = 0.5;

// -------------------------- 优化1：RealSense回调（减少锁竞争） --------------------------
void realsenseCallback(const sensor_msgs::Image::ConstPtr &img)
{
    try
    {
        // 优化：使用try_lock减少阻塞时间
        unique_lock<mutex> lock(frame_mutex, try_to_lock);
        if (lock.owns_lock() && !frame_updated)
        {
            cv_bridge::CvImageConstPtr cv_ptr = cv_bridge::toCvShare(img, sensor_msgs::image_encodings::BGR8);
            frame_msg = cv_ptr->image; // 优化：浅拷贝（避免clone耗时）
            frame_updated = true;
        }
    }
    catch (cv_bridge::Exception &e)
    {
        ROS_ERROR_THROTTLE(30, "RealSense image conversion failed: %s", e.what());
    }
}

// -------------------------- 优化2：数字跟踪线程（精简显示和计算） --------------------------
void digitTrackingThread(ros::Publisher &vel_pub)
{
    Mat frame;
    int frame_count = 0;
    double start_time = getTickCount();
    int window_refresh_cnt = 0;

    ros::Rate thread_rate(30); // 匹配相机帧率

    while (ros::ok())
    {
        if (frame_updated)
        {
            {
                lock_guard<mutex> lock(frame_mutex);
                frame = frame_msg; // 浅拷贝
                frame_updated = false;
            }

            if (!frame.empty())
            {
                // 核心逻辑保留，仅优化图像缩放
                resize(frame, frame, Size(DIGIT_IMG_WIDTH, DIGIT_IMG_HEIGHT), 0, 0, INTER_NEAREST); // 优化：使用最近邻插值（更快）
                MatchResult res = fastMultiScaleMatchWithRotation(frame, allTemplates, targetIndices);
                frame_count++;

                // 移除FPS日志输出
                if (frame_count % LOG_PRINT_INTERVAL == 0)
                {
                    frame_count = 0;
                    start_time = getTickCount();
                }

                geometry_msgs::Twist rawVel, smoothVel;
                if (res.targetIdx != -1 && res.score > SCORE_THRESHOLD)
                {
                    digit_detected = true;
                    {
                        lock_guard<mutex> lock(digital_lost_time_mutex);
                        digital_lost_start_time = ros::Time::now();
                    }

                    // 保留原控制逻辑
                    int templBaseIdx = res.targetIdx * 2; // 适配模板数量从3→2
                    Mat matchedTempl = allTemplates[templBaseIdx];
                    int w = matchedTempl.cols * res.scale;
                    int h = matchedTempl.rows * res.scale;
                    int x = max(0, res.loc.x);
                    int y = max(0, res.loc.y);
                    w = min(DIGIT_IMG_WIDTH - x, w);
                    h = min(DIGIT_IMG_HEIGHT - y, h);

                    Point target_center(x + w / 2, y + h / 2);
                    int edge_dist = min(target_center.x, DIGIT_IMG_WIDTH - target_center.x);
                    int x_error = target_center.x - DIGIT_IMG_CENTER.x;

                    // 优化：仅在显示开启时绘制矩形框（移除所有文字绘制）
                    if (ENABLE_IMAGE_DISPLAY)
                    {
                        rectangle(frame, Point(x, y), Point(x + w, y + h), Scalar(0, 255, 0), 2);
                    }

                    // 线速度/角速度控制逻辑完全不变
                    double size_error = w - TARGET_SIZE;
                    if (size_error < -SIZE_DEAD_ZONE)
                    {
                        rawVel.linear.x = MAX_LINEAR * FORWARD_BOOST * (-size_error) / TARGET_SIZE;
                    }
                    else if (size_error > SIZE_DEAD_ZONE)
                    {
                        rawVel.linear.x = MAX_LINEAR * (-size_error) / TARGET_SIZE;
                    }
                    else
                    {
                        rawVel.linear.x = 0.0;
                    }

                    double rot_speed = 0.0;
                    if (abs(x_error) > DEAD_ZONE)
                    {
                        rot_speed = MAX_ANGULAR_GLOBAL * (-x_error) / (DIGIT_IMG_WIDTH / 2);

                        if (edge_dist < EMERGENCY_STOP_THRESHOLD)
                        {
                            rot_speed = 0.0;
                        }
                        else if (edge_dist < EDGE_PROTECTION)
                        {
                            rot_speed *= 0.3;
                        }
                    }
                    rawVel.angular.z = rot_speed;

                    smoothVel = rotCtrl.smooth(rawVel);

                    if (current_state == TRACKING_DIGITS)
                    {
                        vel_pub.publish(smoothVel);
                    }
                }
                else
                {
                    digit_detected = false;
                    if (current_state == TRACKING_DIGITS)
                    {
                        double lost_duration = 0.0;
                        {
                            lock_guard<mutex> lock(digital_lost_time_mutex);
                            lost_duration = (ros::Time::now() - digital_lost_start_time).toSec();
                        }
                        if (lost_duration >= DIGITAL_LOST_TIMEOUT)
                        {
                            smoothVel.linear.x = 0.0;
                            smoothVel.angular.z = 0.0;
                            vel_pub.publish(smoothVel);
                        }
                    }
                }

                // 优化：仅在显示开启且达到刷新间隔时更新窗口（移除所有文字绘制）
                if (ENABLE_IMAGE_DISPLAY)
                {
                    circle(frame, DIGIT_IMG_CENTER, 4, Scalar(255, 0, 255), -1);
                    line(frame, Point(EMERGENCY_STOP_THRESHOLD, 0), Point(EMERGENCY_STOP_THRESHOLD, DIGIT_IMG_HEIGHT), Scalar(0, 0, 255), 2);
                    line(frame, Point(DIGIT_IMG_WIDTH - EMERGENCY_STOP_THRESHOLD, 0), Point(DIGIT_IMG_WIDTH - EMERGENCY_STOP_THRESHOLD, DIGIT_IMG_HEIGHT), Scalar(0, 0, 255), 2);
                    line(frame, Point(EDGE_PROTECTION, 0), Point(EDGE_PROTECTION, DIGIT_IMG_HEIGHT), Scalar(0, 0, 255), 1);
                    line(frame, Point(DIGIT_IMG_WIDTH - EDGE_PROTECTION, 0), Point(DIGIT_IMG_WIDTH - EDGE_PROTECTION, DIGIT_IMG_HEIGHT), Scalar(0, 0, 255), 1);
                    line(frame, Point(DIGIT_IMG_CENTER.x - 90, 0), Point(DIGIT_IMG_CENTER.x - 90, DIGIT_IMG_HEIGHT), Scalar(0, 255, 255), 1);
                    line(frame, Point(DIGIT_IMG_CENTER.x + 90, 0), Point(DIGIT_IMG_CENTER.x + 90, DIGIT_IMG_HEIGHT), Scalar(0, 255, 255), 1);

                    window_refresh_cnt++;
                    if (window_refresh_cnt >= WINDOW_REFRESH_INTERVAL)
                    {
                        imshow("Robot Ultimate Stable Version (640 Width)", frame);
                        waitKey(WAITKEY_DURATION);
                        window_refresh_cnt = 0;
                    }
                }
            }
        }
        else
        {
            // 优化：无新帧时休眠更长时间
            this_thread::sleep_for(chrono::milliseconds(5));
        }
        thread_rate.sleep();
    }
}

// -------------------------- 锥桶导航相关（仅删除日志，不改变逻辑） --------------------------
const std::string CONE_BOX_TOPIC = "/cone_detect/boxes";
const std::string CHASSIS_VEL_TOPIC = "/smoother_cmd_vel";
const float LINEAR_SPEED = 0.2;
const int IMAGE_CENTER_X = 640;
const float CHANNEL_WIDTH_PX = 250;
const float CHANNEL_CENTER_TOLERANCE = 80.0;
const float ACTION1_ANGLE = 0.70f;
const float ACTION2_ANGLE = M_PI * 38 / 180;
const float ACTION2_ANGULAR = -0.2;
const float TRIGGER_DURATION = 1.0;
const int TRIGGER2_CONE_COUNT = 1;
const float ACTION3_ANGLE = M_PI * 83 / 180;
const float ACTION3_ANGULAR = -0.2f;

const float TWO_CONE_LEFT_EDGE_PERCENT = 0.10f;
const float TWO_CONE_RIGHT_EDGE_PERCENT = 0.10f;
const float SINGLE_CONE_EDGE_PERCENT = 0.15f;
const double FINAL_STRAIGHT_KP = 0.0015;
const double FINAL_MAX_ANGULAR = 0.3;
const double FINAL_DEAD_ZONE = 10.0;
const double MIN_CONE_DISTANCE = 50.0;
const double MIN_CONE_WIDTH = 30.0;

enum ActionState
{
    SEARCH_TRIGGER1,
    GO_STRAIGHT,
    CONFIRM_CENTER,
    ACTION1_RUNNING,
    ACTION1_STRAIGHT,
    SEARCH_TRIGGER2,
    ACTION2_RUNNING,
    POST_ACTION2_STRAIGHT,
    ACTION3_RUNNING,
    FINAL_STRAIGHT,
    NORMAL_DRIVING
};

std::vector<std::vector<float>> cone_boxes;
ActionState current_action_state = SEARCH_TRIGGER1;
ros::Time trigger_confirm_start;
ros::Time action_start_time;
float action2_remaining_angle = 0.0;
float action1_remaining_angle = 0.0f;
float first_channel_center = -1.0f;
bool is_first_channel = true;
atomic<bool> cone_detected(false);

bool checkPostAction2ToAction3Trigger(string &trigger_reason)
{
    trigger_reason = "";
    const float IMAGE_WIDTH = 1280.0f;

    const float two_cone_left_threshold = IMAGE_WIDTH * TWO_CONE_LEFT_EDGE_PERCENT;
    const float two_cone_right_threshold = IMAGE_WIDTH * TWO_CONE_RIGHT_EDGE_PERCENT;
    const float single_cone_threshold = IMAGE_WIDTH * SINGLE_CONE_EDGE_PERCENT;

    if (!cone_detected)
    {
        trigger_reason = "条件1：未检测到任何锥桶";
        return true;
    }

    if (cone_boxes.size() == 2)
    {
        float cone1_center_x = (cone_boxes[0][0] + cone_boxes[0][2]) / 2.0f;
        float cone2_center_x = (cone_boxes[1][0] + cone_boxes[1][2]) / 2.0f;

        float left_cone_center = min(cone1_center_x, cone2_center_x);
        float right_cone_center = max(cone1_center_x, cone2_center_x);

        bool left_on_left_side = (left_cone_center < IMAGE_CENTER_X);
        bool right_on_right_side = (right_cone_center > IMAGE_CENTER_X);

        bool left_near_edge = (left_cone_center < two_cone_left_threshold);
        bool right_near_edge = ((IMAGE_WIDTH - right_cone_center) < two_cone_right_threshold);

        if (left_on_left_side && right_on_right_side && left_near_edge && right_near_edge)
        {
            char reason_buf[512];
            snprintf(reason_buf, sizeof(reason_buf),
                     "条件2：中心线两侧各1个锥桶 | 左锥桶中心x=%.2fpx（离左边缘%.2fpx < 阈值%.2fpx） | 右锥桶中心x=%.2fpx（离右边缘%.2fpx < 阈值%.2fpx）",
                     left_cone_center, left_cone_center, two_cone_left_threshold,
                     right_cone_center, (IMAGE_WIDTH - right_cone_center), two_cone_right_threshold);
            trigger_reason = reason_buf;
            return true;
        }
    }

    if (cone_boxes.size() == 1)
    {
        float cone_center_x = (cone_boxes[0][0] + cone_boxes[0][2]) / 2.0f;

        if (cone_center_x < IMAGE_CENTER_X && cone_center_x < single_cone_threshold)
        {
            char reason_buf[512];
            snprintf(reason_buf, sizeof(reason_buf),
                     "条件3：仅左侧有1个锥桶 | 中心x=%.2fpx（离左边缘%.2fpx < 阈值%.2fpx）",
                     cone_center_x, cone_center_x, single_cone_threshold);
            trigger_reason = reason_buf;
            return true;
        }

        if (cone_center_x > IMAGE_CENTER_X && (IMAGE_WIDTH - cone_center_x) < single_cone_threshold)
        {
            char reason_buf[512];
            snprintf(reason_buf, sizeof(reason_buf),
                     "条件3：仅右侧有1个锥桶 | 中心x=%.2fpx（离右边缘%.2fpx < 阈值%.2fpx）",
                     cone_center_x, (IMAGE_WIDTH - cone_center_x), single_cone_threshold);
            trigger_reason = reason_buf;
            return true;
        }
    }

    char reason_buf[512];
    snprintf(reason_buf, sizeof(reason_buf),
             "未满足任何触发条件 | 当前锥桶数量：%ld | 双锥左阈值：%.2fpx | 双锥右阈值：%.2fpx | 单锥阈值：%.2fpx",
             cone_boxes.size(), two_cone_left_threshold, two_cone_right_threshold, single_cone_threshold);
    trigger_reason = reason_buf;
    return false;
}

void coneBoxCallback(const std_msgs::Float32MultiArray::ConstPtr &msg)
{
    cone_boxes.clear();
    cone_detected = false;
    if (msg->data.size() % 4 == 0)
    {
        for (size_t i = 0; i < msg->data.size(); i += 4)
        {
            float x1 = msg->data[i];
            float y1 = msg->data[i + 1];
            float x2 = msg->data[i + 2];
            float y2 = msg->data[i + 3];
            cone_boxes.push_back({x1, y1, x2, y2});
        }
        if (!cone_boxes.empty())
            cone_detected = true;
    }
    // 删除DEBUG日志
}

void validateCoordinates()
{
    if (!cone_boxes.empty())
    {
        for (size_t i = 0; i < cone_boxes.size(); ++i)
        {
            float x1 = cone_boxes[i][0];
            float x2 = cone_boxes[i][2];
            if (x1 < 0 || x2 > 1280 || x1 > x2)
            {
                ROS_WARN_THROTTLE(5, "[锥桶导航] 锥桶%ld坐标无效: x1=%.1f, x2=%.1f", i, x1, x2);
            }
        }
    }
}

void findChannelCenterAndGap(float &center, float &gap)
{
    center = -1.0f;
    gap = 0.0f;
    if (cone_boxes.size() < 2 || !is_first_channel)
        return;

    std::vector<std::vector<float>> large_cones;
    for (const auto &box : cone_boxes)
    {
        float width = box[2] - box[0];
        if (width >= 200.0f)
        {
            large_cones.push_back(box);
        }
    }
    if (large_cones.size() < 2)
    {
        return;
    }

    auto left_cone = large_cones[0];
    auto right_cone = large_cones[0];
    for (const auto &box : large_cones)
    {
        if (box[0] < left_cone[0])
            left_cone = box;
        if (box[2] > right_cone[2])
            right_cone = box;
    }

    gap = right_cone[0] - left_cone[2];

    if (gap >= CHANNEL_WIDTH_PX)
    {
        center = left_cone[2] + gap / 2.0f;
    }
}

bool isFirstValidChannel()
{
    float center, gap;
    findChannelCenterAndGap(center, gap);
    first_channel_center = center;
    return (center > 0.0f && gap >= CHANNEL_WIDTH_PX);
}

bool isReachChannelCenter()
{
    if (first_channel_center < 0.0f || cone_boxes.size() < 2)
        return false;

    int left_cone_count = 0;
    float leftmost_width = 0.0f;
    float min_center_x = 1280.0f;
    const float WIDTH_THRESHOLD = 288.0f;

    for (const auto &box : cone_boxes)
    {
        float center_x = (box[0] + box[2]) / 2.0f;
        float width = box[2] - box[0];

        if (center_x < IMAGE_CENTER_X)
        {
            left_cone_count++;
            if (center_x < min_center_x)
            {
                min_center_x = center_x;
                leftmost_width = width;
            }
        }
    }

    return (left_cone_count == 3 && leftmost_width > WIDTH_THRESHOLD);
}

bool isTrigger2()
{
    const float LEFT_CENTER_NEAR_PERCENT = 0.2f;
    const float RIGHT_CENTER_NEAR_PERCENT = 0.33f;
    const float IMAGE_HEIGHT = 720.0f;
    const float IMAGE_WIDTH = 1280.0f;
    const float HEIGHT_OCCUPY_THRESHOLD = 0.888f;
    const float MIN_HEIGHT_THRESHOLD = 50.0f;

    const float LEFT_CENTER_NEAR_PIXEL = IMAGE_WIDTH * LEFT_CENTER_NEAR_PERCENT;
    const float RIGHT_CENTER_NEAR_PIXEL = IMAGE_WIDTH * RIGHT_CENTER_NEAR_PERCENT;

    int center_near_cone_count = 0;
    float target_cone_height = 0.0f;
    float target_cone_center_x = 0.0f;
    float target_height_occupy = 0.0f;

    for (const auto &box : cone_boxes)
    {
        float x1 = box[0];
        float y1 = box[1];
        float x2 = box[2];
        float y2 = box[3];
        float cone_center_x = (x1 + x2) / 2.0f;
        float cone_height = y2 - y1;
        float height_occupy = cone_height / IMAGE_HEIGHT;

        bool is_near_center = false;
        float dist_to_center = cone_center_x - IMAGE_CENTER_X;
        if (dist_to_center < 0)
        {
            is_near_center = (fabs(dist_to_center) <= LEFT_CENTER_NEAR_PIXEL);
        }
        else
        {
            is_near_center = (dist_to_center <= RIGHT_CENTER_NEAR_PIXEL);
        }

        bool is_height_qualified = (height_occupy >= HEIGHT_OCCUPY_THRESHOLD) && (cone_height >= MIN_HEIGHT_THRESHOLD);

        if (is_near_center)
        {
            center_near_cone_count++;
            if (is_height_qualified)
            {
                target_cone_height = cone_height;
                target_cone_center_x = cone_center_x;
                target_height_occupy = height_occupy;
            }
        }
    }

    bool is_triggered = (center_near_cone_count >= 1) && (target_height_occupy >= HEIGHT_OCCUPY_THRESHOLD);

    return is_triggered;
}

void controlConeNavigation(ros::Publisher &vel_pub)
{
    geometry_msgs::Twist twist;
    validateCoordinates();

    static int cone_log_cnt = 0;
    cone_log_cnt++;

    switch (current_action_state)
    {
    case SEARCH_TRIGGER1:
    {
        twist.linear.x = LINEAR_SPEED;
        twist.angular.z = 0.0f;

        if (isFirstValidChannel())
        {
            current_action_state = GO_STRAIGHT;
            trigger_confirm_start = ros::Time::now();
        }
        break;
    }
    case GO_STRAIGHT:
    {
        vector<vector<float>> valid_cones;

        twist.linear.x = LINEAR_SPEED;
        twist.angular.z = 0.0;

        for (const auto &box : cone_boxes)
        {
            float cone_width = box[2] - box[0];
            if (cone_width >= MIN_CONE_WIDTH && cone_width >= MIN_CONE_DISTANCE)
            {
                valid_cones.push_back(box);
            }
        }

        if (valid_cones.size() >= 2)
        {
            vector<vector<float>> left_cones, right_cones;
            for (const auto &cone : valid_cones)
            {
                float cone_center = (cone[0] + cone[2]) / 2.0f;
                if (cone_center < IMAGE_CENTER_X)
                {
                    left_cones.push_back(cone);
                }
                else
                {
                    right_cones.push_back(cone);
                }
            }

            if (!left_cones.empty() && !right_cones.empty())
            {
                vector<float> nearest_left = left_cones[0];
                for (const auto &cone : left_cones)
                {
                    if (cone[2] > nearest_left[2])
                    {
                        nearest_left = cone;
                    }
                }

                vector<float> nearest_right = right_cones[0];
                for (const auto &cone : right_cones)
                {
                    if (cone[0] < nearest_right[0])
                    {
                        nearest_right = cone;
                    }
                }

                float left_right = nearest_left[2];
                float right_left = nearest_right[0];
                double channel_center = (left_right + right_left) / 2.0;

                double error = channel_center - IMAGE_CENTER_X;

                if (fabs(error) > FINAL_DEAD_ZONE)
                {
                    twist.angular.z = -FINAL_STRAIGHT_KP * error;
                    twist.angular.z = std::max(-FINAL_MAX_ANGULAR, std::min(twist.angular.z, FINAL_MAX_ANGULAR));
                }
            }
        }

        if (isReachChannelCenter())
        {
            current_action_state = CONFIRM_CENTER;
            trigger_confirm_start = ros::Time::now();
        }
        break;
    }

    case CONFIRM_CENTER:
    {
        twist.linear.x = LINEAR_SPEED * 0.8;
        twist.angular.z = 0.0f;

        if (isReachChannelCenter())
        {
            if ((ros::Time::now() - trigger_confirm_start).toSec() > TRIGGER_DURATION)
            {
                current_action_state = ACTION1_RUNNING;
                action_start_time = ros::Time::now();
                is_first_channel = false;
            }
        }
        else
        {
            current_action_state = GO_STRAIGHT;
        }
        break;
    }
    case ACTION1_RUNNING:
    {
        bool delay_completed = false;
        ros::WallTime action_start_walltime = ros::WallTime().fromSec(action_start_time.toSec());
        float total_elapsed = (ros::WallTime::now() - action_start_walltime).toSec();
        const float DELAY_DURATION = 1.0f;

        if (total_elapsed < DELAY_DURATION)
        {
            twist.linear.x = LINEAR_SPEED * 0.8;
            twist.angular.z = 0.0f;
        }
        else
        {
            delay_completed = true;
            float rotate_elapsed = total_elapsed - DELAY_DURATION;
            float turned_angle = std::abs(ACTION2_ANGULAR) * rotate_elapsed;
            action1_remaining_angle = ACTION1_ANGLE - turned_angle;

            twist.linear.x = 0.0;
            twist.angular.z = ACTION2_ANGULAR;

            if (action1_remaining_angle <= 0)
            {
                twist.angular.z = 0.0;
                vel_pub.publish(twist);

                current_action_state = ACTION1_STRAIGHT;
                action_start_time = ros::Time::now();
            }
        }

        vel_pub.publish(twist);
        break;
    }

    case ACTION1_STRAIGHT:
    {
        twist.linear.x = LINEAR_SPEED * 0.8;
        twist.angular.z = 0.0;

        float straight_duration = 3.0;
        if ((ros::Time::now() - action_start_time).toSec() >= straight_duration)
        {
            current_action_state = SEARCH_TRIGGER2;
        }

        vel_pub.publish(twist);
        break;
    }
    case SEARCH_TRIGGER2:
    {
        twist.linear.x = LINEAR_SPEED;
        twist.angular.z = 0.0f;

        if (isTrigger2())
        {
            current_action_state = ACTION2_RUNNING;
            action_start_time = ros::Time::now();
            action2_remaining_angle = ACTION2_ANGLE;
        }
        break;
    }
    case ACTION2_RUNNING:
    {
        twist.linear.x = 0.0f;
        twist.angular.z = -ACTION2_ANGULAR;

        float elapsed_time = (ros::Time::now() - action_start_time).toSec();
        float turned_angle = std::abs(-ACTION2_ANGULAR) * elapsed_time;
        action2_remaining_angle = ACTION2_ANGLE - turned_angle;

        if (action2_remaining_angle <= 0)
        {
            twist.angular.z = 0.0;
            vel_pub.publish(twist);
            current_action_state = POST_ACTION2_STRAIGHT;
        }
        break;
    }

    case POST_ACTION2_STRAIGHT:
    {
        twist.linear.x = LINEAR_SPEED;
        twist.angular.z = 0.0f;

        string trigger_reason;
        bool trigger_condition_met = checkPostAction2ToAction3Trigger(trigger_reason);

        if (trigger_condition_met)
        {
            current_action_state = ACTION3_RUNNING;
            action_start_time = ros::Time::now();
        }

        break;
    }

    case ACTION3_RUNNING:
    {
        ros::WallTime action_start_walltime = ros::WallTime().fromSec(action_start_time.toSec());

        twist.linear.x = 0.0f;
        twist.angular.z = ACTION3_ANGULAR;

        float elapsed_time = (ros::WallTime::now() - action_start_walltime).toSec();
        float turned_angle = std::abs(ACTION3_ANGULAR) * elapsed_time;

        if (turned_angle >= ACTION3_ANGLE)
        {
            current_action_state = FINAL_STRAIGHT;
        }
        break;
    }

    case FINAL_STRAIGHT:
    {
        vector<vector<float>> valid_cones;

        twist.linear.x = LINEAR_SPEED;
        twist.angular.z = 0.0;

        for (const auto &box : cone_boxes)
        {
            float cone_width = box[2] - box[0];
            if (cone_width >= MIN_CONE_WIDTH && cone_width >= MIN_CONE_DISTANCE)
            {
                valid_cones.push_back(box);
            }
        }

        if (valid_cones.size() < 2)
        {
            current_action_state = NORMAL_DRIVING;
            twist.linear.x = 0.0;
            twist.angular.z = 0.0;
            vel_pub.publish(twist);
            break;
        }

        vector<float> left_cone = valid_cones[0];
        vector<float> right_cone = valid_cones[0];
        for (const auto &box : valid_cones)
        {
            if (box[0] < left_cone[0])
                left_cone = box;
            if (box[2] > right_cone[2])
                right_cone = box;
        }

        float left_cone_right = left_cone[2];
        float right_cone_left = right_cone[0];
        double channel_center = (left_cone_right + right_cone_left) / 2.0;

        double image_center = IMAGE_CENTER_X;
        double error = channel_center - image_center;

        if (fabs(error) > FINAL_DEAD_ZONE)
        {
            twist.angular.z = -FINAL_STRAIGHT_KP * error;
            twist.angular.z = std::max(-FINAL_MAX_ANGULAR, std::min(twist.angular.z, FINAL_MAX_ANGULAR));
        }

        break;
    }
    case NORMAL_DRIVING:
    {
        twist.linear.x = 0.15 * LINEAR_SPEED;
        twist.angular.z = 0.0f;
        break;
    }
    }

    vel_pub.publish(twist);
}

// -------------------------- 主函数（仅保留状态切换日志） --------------------------
int main(int argc, char **argv)
{
    ros::init(argc, argv, "integrated_control_node");
    ros::NodeHandle nh;

    if (ros::console::set_logger_level(ROSCONSOLE_DEFAULT_NAME, ENABLE_DEBUG_LOG ? ros::console::levels::Debug : ros::console::levels::Info))
    {
        ros::console::notifyLoggerLevelsChanged();
    }

    ros::Publisher vel_pub = nh.advertise<geometry_msgs::Twist>(CHASSIS_VEL_TOPIC, 50);
    ros::Subscriber cone_box_sub = nh.subscribe(CONE_BOX_TOPIC, 20, coneBoxCallback);
    ros::Subscriber realsense_sub = nh.subscribe("/camera/color/image_raw", 20, realsenseCallback);

    // 加载数字跟踪模板（适配模板数量从3→2）
    vector<string> templ_paths = {
        "/home/eaibot/dip_ws_design/src/exp1/src/template_0.png",
        "/home/eaibot/dip_ws_design/src/exp1/src/template_1.png",
        "/home/eaibot/dip_ws_design/src/exp1/src/template_2.png"};

    for (int i = 0; i < templ_paths.size(); i++)
    {
        Mat templ = imread(templ_paths[i], IMREAD_COLOR);
        if (templ.empty())
        {
            ROS_ERROR("模板%d加载失败! 路径: %s", i, templ_paths[i].c_str());
            return -1;
        }
        vector<Mat> templates = generateRotatedTemplates(templ);
        for (auto &temp : templates)
        {
            allTemplates.push_back(temp);
            targetIndices.push_back(i);
        }
    }

    ros::Duration(1.0).sleep(); // 缩短启动延时

    if (ENABLE_IMAGE_DISPLAY)
    {
        namedWindow("Robot Ultimate Stable Version (640 Width)", WINDOW_NORMAL);
        cv::startWindowThread();
    }

    digital_lost_start_time = ros::Time::now();

    // 启动数字跟踪线程
    thread tracking_thread(digitTrackingThread, ref(vel_pub));
    tracking_thread.detach();

    current_state = STANDBY;
    ros::Rate rate(20); // 保持20Hz主循环

    while (ros::ok())
    {
        bool current_digit_detected = digit_detected;
        bool current_cone_detected = cone_detected;

        bool is_cone_nav_active = (current_action_state != SEARCH_TRIGGER1) &&
                                  (current_action_state != NORMAL_DRIVING);

        // 状态切换逻辑完全不变，仅保留状态切换日志
        switch (current_state)
        {
        case STANDBY:
            if (current_cone_detected || is_cone_nav_active)
            {
                ROS_INFO("=====================================");
                ROS_INFO("状态切换：STANDBY → 锥桶导航状态（数字跟踪暂不触发）");
                ROS_INFO("=====================================");
                current_state = CONE_NAVIGATION;
            }
            break;

        case CONE_NAVIGATION:
            controlConeNavigation(vel_pub);
            if (current_action_state == NORMAL_DRIVING && current_digit_detected)
            {
                ROS_INFO("=====================================");
                ROS_INFO("状态切换：锥桶导航状态 → 数字跟踪状态（锥桶道路已通行完成，检测到数字）");
                ROS_INFO("=====================================");
                current_state = TRACKING_DIGITS;
                {
                    lock_guard<mutex> lock(digital_lost_time_mutex);
                    digital_lost_start_time = ros::Time::now();
                }
            }
            break;

        case TRACKING_DIGITS:
            if (current_digit_detected)
            {
                lock_guard<mutex> lock(digital_lost_time_mutex);
                digital_lost_start_time = ros::Time::now();
            }
            break;
        }

        ros::spinOnce();
        rate.sleep();
    }

    geometry_msgs::Twist stop_twist;
    vel_pub.publish(stop_twist);
    if (ENABLE_IMAGE_DISPLAY)
        destroyAllWindows();
}
