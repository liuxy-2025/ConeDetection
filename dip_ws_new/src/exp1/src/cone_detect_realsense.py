import cv2
import numpy as np
from ultralytics import YOLO
import rospy
from std_msgs.msg import Float32MultiArray
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

# -------------------------- 配置参数 --------------------------
MODEL_PATH = "/home/eaibot/best.pt"        # YOLOv8模型路径
CONF_THRESH = 0.3                          # 置信度阈值
NMS_THRESH = 0.4                           # NMS阈值
DISPLAY_WINDOW = "Cone Detection (Debug)"  # 调试窗口名

# 图像分辨率配置
REALSENSE_WIDTH = 1280                     # RealSense原始图像宽度
REALSENSE_HEIGHT = 720                     # RealSense原始图像高度
REALSENSE_CENTER_X = REALSENSE_WIDTH // 2  # 原始图像中心X坐标 (640)
DISPLAY_WIDTH = 640                        # 显示窗口宽度
DISPLAY_HEIGHT = 480                       # 显示窗口高度
DISPLAY_CENTER_X = DISPLAY_WIDTH // 2      # 显示图像中心X坐标 (320)

# ROS话题配置
CONE_BOX_TOPIC = "/cone_detect/boxes"      # 锥桶框坐标发布话题
IMAGE_PUB_TOPIC = "/cone_detect/annotated_image"  # 标注图像话题
REALSENSE_COLOR_TOPIC = "/camera/color/image_raw"  # RealSense彩色图像话题
# --------------------------------------------------------------

bridge = CvBridge()
latest_cv_image = None  # 存储最新图像帧
scale_x = DISPLAY_WIDTH / REALSENSE_WIDTH  # X轴缩放比例
scale_y = DISPLAY_HEIGHT / REALSENSE_HEIGHT  # Y轴缩放比例

def image_callback(msg):
    """处理从ROS话题收到的图像"""
    global latest_cv_image
    try:
        # 获取RealSense原始分辨率的图像
        latest_cv_image = bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
    except Exception as e:
        rospy.logerr(f"[Python ERROR] 图像转换失败: {e}")

def main():
    global latest_cv_image, scale_x, scale_y
    
    # 1. 初始化ROS节点和发布者/订阅者
    rospy.init_node("python_cone_detect_node", anonymous=True)
    box_pub = rospy.Publisher(CONE_BOX_TOPIC, Float32MultiArray, queue_size=10)
    img_pub = rospy.Publisher(IMAGE_PUB_TOPIC, Image, queue_size=10)
    rospy.Subscriber(REALSENSE_COLOR_TOPIC, Image, image_callback)
    
    print("[Python DEBUG] ROS节点初始化成功，等待图像数据...")
    print(f"[Python DEBUG] 图像分辨率: {REALSENSE_WIDTH}x{REALSENSE_HEIGHT}")
    print(f"[Python DEBUG] 图像中心X: {REALSENSE_CENTER_X}px")
    
    # 2. 加载YOLOv8模型
    print(f"[Python DEBUG] 加载模型: {MODEL_PATH}")
    model = YOLO(MODEL_PATH)
    print("[Python DEBUG] 模型加载成功！")
    print("=== 模型可识别类别 ===")
    print(model.names)  # 必须输出类似 {0: 'cone'} 或 {0: '锥桶'} 的结果

    # 3. 实时检测循环
    print("[Python DEBUG] 开始锥桶检测（按 'q' 退出）...")
    
    # 控制循环频率（与相机帧率匹配）
    rate = rospy.Rate(30)
    
    while not rospy.is_shutdown():
        # 检查是否收到图像
        if latest_cv_image is None:
            rate.sleep()
            continue
            
        # 使用最新的一帧图像（原始分辨率）
        frame_original = latest_cv_image.copy()
        
        # YOLOv8检测（在原始分辨率上进行）
        results = model(frame_original, conf=CONF_THRESH, iou=NMS_THRESH, verbose=False)
        
        # 创建用于显示的图像（缩小到显示分辨率）
        display_frame = cv2.resize(frame_original, (DISPLAY_WIDTH, DISPLAY_HEIGHT))
        
        # 在显示图像上绘制中心线
        cv2.line(display_frame, (DISPLAY_CENTER_X, 0), (DISPLAY_CENTER_X, DISPLAY_HEIGHT), 
                 (255, 0, 0), 2)
        cv2.putText(display_frame, f"Display Center: {DISPLAY_CENTER_X}px", 
                    (DISPLAY_CENTER_X + 5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
        
        # 准备发布锥桶框数据
        Float32MultiArray_msg = Float32MultiArray()
        cone_count = 0
        cone_info_list = []  # 存储锥桶信息用于日志输出
        
        # 处理检测结果
        if results[0].boxes is not None:
            for box in results[0].boxes:
                # 获取原始分辨率上的坐标
                x1_raw, y1_raw, x2_raw, y2_raw = map(float, box.xyxy[0].numpy())
                
                # 发布原始坐标给C++节点（重要！C++使用这些原始坐标）
                Float32MultiArray_msg.data.extend([x1_raw, y1_raw, x2_raw, y2_raw])
                
                # 转换为显示坐标用于绘制
                x1_disp = int(x1_raw * scale_x)
                y1_disp = int(y1_raw * scale_y)
                x2_disp = int(x2_raw * scale_x)
                y2_disp = int(y2_raw * scale_y)
                
                # 计算锥桶信息（在原始分辨率上）
                cone_width_raw = x2_raw - x1_raw
                cone_center_x_raw = (x1_raw + x2_raw) / 2.0
                offset_from_center = cone_center_x_raw - REALSENSE_CENTER_X
                
                # 存储锥桶信息
                cone_info = {
                    'id': cone_count,
                    'raw_center': cone_center_x_raw,
                    'offset': offset_from_center,
                    'width': cone_width_raw
                }
                cone_info_list.append(cone_info)
                
                # 在显示图像上绘制锥桶框和信息
                cv2.rectangle(display_frame, (x1_disp, y1_disp), (x2_disp, y2_disp), 
                             (0, 255, 0), 2)
                
                cone_center_disp = (x1_disp + x2_disp) // 2
                debug_text = f"Cone {cone_count}: X={cone_center_disp}px"
                cv2.putText(display_frame, debug_text, (x1_disp, y1_disp-10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
                
                # 绘制锥桶中心点
                cv2.circle(display_frame, (cone_center_disp, (y1_disp+y2_disp)//2), 
                          3, (0, 0, 255), -1)
                
                cone_count += 1
        
        # 发布锥桶框数据
        box_pub.publish(Float32MultiArray_msg)
        
        # 在显示图像上绘制锥桶数量
        cv2.putText(display_frame, f"Total Cones: {cone_count}", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        
        # 绘制原始分辨率信息
        cv2.putText(display_frame, f"Original: {REALSENSE_WIDTH}x{REALSENSE_HEIGHT}", 
                    (10, DISPLAY_HEIGHT - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
        cv2.putText(display_frame, f"Center X: {REALSENSE_CENTER_X}px", 
                    (10, DISPLAY_HEIGHT - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
        
        # 调试日志输出
        print(f"[Python DEBUG] 检测到锥桶数量: {cone_count}")
        for cone_info in cone_info_list:
            print(f"[Python DEBUG] 锥桶{cone_info['id']}: 原始中心={cone_info['raw_center']:.1f}px, " +
                  f"偏移={cone_info['offset']:.1f}px, 宽度={cone_info['width']:.1f}px")
        
        # 发布标注图像
        ros_img = bridge.cv2_to_imgmsg(display_frame, encoding="bgr8")
        ros_img.header.stamp = rospy.Time.now()
        img_pub.publish(ros_img)
        
        # 显示和退出控制
        cv2.imshow(DISPLAY_WINDOW, display_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
        rate.sleep()  # 控制循环频率
    
    # 释放资源
    cv2.destroyAllWindows()
    print("[Python DEBUG] 检测程序退出")

if __name__ == "__main__":
    main()

