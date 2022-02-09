cmake_minimum_required(VERSION 2.8.3)
project(jackal_velodyne)

add_definitions(-std=c++11)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${GAZEBO_CXX_FLAGS} -std=c++11 -O3")

find_package(catkin REQUIRED COMPONENTS
  gazebo_ros
  tf
  roscpp
  rospy
  nav_msgs
  sensor_msgs
  rosbag
  actionlib
  move_base_msgs
  geometry_msgs
  message_generation
  pcl_ros
  pcl_conversions
  tf2
  tf2_ros
  tf2_sensor_msgs
  laser_geometry
  message_filters
  nodelet
)

find_package(gazebo REQUIRED)


catkin_package(
  CATKIN_DEPENDS nav_msgs roscpp rospy sensor_msgs rosbag actionlib move_base_msgs geometry_msgs message_runtime laser_geometry message_filters nodelet tf2 tf2_ros tf2_sensor_msgs gazebo_plugins gazebo_ros
)

# include directories
include_directories(
	${catkin_INCLUDE_DIRS}
	${GAZEBO_INCLUDE_DIRS}
)

link_directories(
  ${GAZEBO_LIBRARY_DIRS}
)

add_library(pcl_filter src/pcl_filter_nodelet.cpp)
target_link_libraries(pcl_filter ${catkin_LIBRARIES})

add_executable(pcl_filter_node src/pcl_filter_node.cpp)
target_link_libraries(pcl_filter_node pcl_filter ${catkin_LIBRARIES})

add_library(camera_controller SHARED src/camera_controller.cpp)
target_link_libraries(camera_controller ${GAZEBO_libraries} CameraPlugin)
