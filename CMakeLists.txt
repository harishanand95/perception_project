cmake_minimum_required(VERSION 2.8.3)
project(group_1_project_3a)

## Compile as C++11, supported in ROS Kinetic and newer
# add_compile_options(-std=c++11)

find_package(catkin REQUIRED COMPONENTS
  roscpp
  cv_bridge
  geometry_msgs
  tf
  turtlebot3_description
  darknet_ros
  darknet_ros_msgs
  std_msgs
  message_generation
)

find_package(OpenCV REQUIRED)

add_message_files(
  FILES
  detection.msg
)

generate_messages(
  DEPENDENCIES
  std_msgs
)

catkin_package(
#  INCLUDE_DIRS include
#  LIBRARIES group_1_project_3a
  CATKIN_DEPENDS message_runtime
#  DEPENDS system_lib
)

include_directories(
# include
  ${catkin_INCLUDE_DIRS}
)


add_executable (position_control src/position_control.cpp)
add_executable (yolo src/yolo.cpp)
add_executable (transform src/transform.cpp)

add_executable (broadcaster src/broadcaster.cpp)

target_link_libraries(position_control ${catkin_LIBRARIES})
target_link_libraries(yolo ${OpenCV_LIBS} ${catkin_LIBRARIES})
target_link_libraries(transform ${OpenCV_LIBS} ${catkin_LIBRARIES})

target_link_libraries(broadcaster ${OpenCV_LIBS} ${catkin_LIBRARIES})