cmake_minimum_required(VERSION 3.0.2)
project(exp1)

# 强制C++11标准（ONNX Runtime 必需）
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 1. 查找ROS核心依赖（保留你的原有配置）
find_package(catkin REQUIRED COMPONENTS
  roscpp
  std_msgs
  geometry_msgs
  sensor_msgs
  cv_bridge
)

# 2. 查找OpenCV（保留你的原有配置）
find_package(OpenCV REQUIRED)



# 4. 声明Catkin包（保留原有）
catkin_package()


include_directories(
  ${catkin_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS}
 
)

# 6. 编译可执行文件（保留原有）
add_executable(exp1 src/exp1.cpp)
#add_executable(test_demo src/test_demo.cpp)
# 7. 链接库（核心：添加ONNX Runtime及依赖库）
target_link_libraries(exp1
  ${catkin_LIBRARIES}
  ${OpenCV_LIBRARIES}
)

