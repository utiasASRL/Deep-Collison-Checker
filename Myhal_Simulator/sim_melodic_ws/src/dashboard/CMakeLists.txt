cmake_minimum_required(VERSION 3.0.2)
project(dashboard)

## Compile as C++11, supported in ROS Kinetic and newer
add_compile_options(-std=c++14)

## Find catkin macros and libraries
## if COMPONENTS list like find_package(catkin REQUIRED COMPONENTS xyz)
## is used, also find other catkin packages
find_package(catkin REQUIRED COMPONENTS
  actionlib
  geometry_msgs
  move_base_msgs
  nav_msgs
  pcl_conversions
  pcl_ros
  ros_numpy
  rosbag
  roscpp
  rospy
  sensor_msgs
  std_msgs
  tf2_msgs
)

catkin_package(
   LIBRARIES dashboard
   CATKIN_DEPENDS actionlib geometry_msgs move_base_msgs nav_msgs pcl_conversions pcl_ros ros_numpy rosbag roscpp rospy sensor_msgs std_msgs tf2_msgs
   DEPENDS system_lib
   INCLUDE_DIRS src/utilities
)

catkin_install_python(PROGRAMS
        src/utilities/math_utilities.py
        src/utilities/plot_utilities.py
       	src/data_processing.py
        src/utilities/bag_tools.py
        src/meta_data.py
        src/assessor.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

