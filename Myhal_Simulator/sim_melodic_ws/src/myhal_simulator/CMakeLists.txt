cmake_minimum_required(VERSION 3.0.2)
project(myhal_simulator)

add_definitions(-std=c++14)

## Find catkin macros and libraries
## if COMPONENTS list like find_package(catkin REQUIRED COMPONENTS xyz)
## is used, also find other catkin packages
find_package(catkin REQUIRED COMPONENTS
  gazebo_ros
  gazebo_plugins
  pcl_ros
  roscpp
  rospy
  std_msgs
  pcl_conversions
  sensor_msgs
  rosbag
  nav_msgs
  geometry_msgs
  move_base_msgs
  actionlib
  tf2_msgs
  ros_numpy
)


find_package(gazebo REQUIRED)


catkin_package(
	DEPENDS gazebo_ros gazebo_plugins pcl_ros roscpp rospy std_msgs rosbag sensor_msgs pcl_conversions nav_msgs geometry_msgs move_base_msgs actionlib tf2_msgs ros_numpy
	INCLUDE_DIRS src/include
)


link_directories(src/include ${GAZEBO_LIBRARY_DIRS})
include_directories(src/include ${Boost_INCLUDE_DIR} ${catkin_INCLUDE_DIRS} ${GAZEBO_INCLUDE_DIRS})

add_library(Perlin src/include/Perlin.cpp)
target_link_libraries(Perlin ${catkin_LIBRARIES} ${GAZEBO_LIBRARIES})

add_library(utilities src/include/utilities.cpp)
target_link_libraries(utilities ${catkin_LIBRARIES} ${GAZEBO_LIBRARIES})

add_library(costmap src/include/costmap.cpp)
target_link_libraries(costmap utilities ${catkin_LIBRARIES} ${GAZEBO_LIBRARIES})

add_library(sdfstring src/include/sdfstring.cpp)
target_link_libraries(sdfstring ${catkin_LIBRARIES} ${GAZEBO_LIBRARIES})

add_library(flowfield src/include/flowfield.cpp)
target_link_libraries(flowfield utilities costmap ${catkin_LIBRARIES} ${GAZEBO_LIBRARIES})

add_library(vehicles src/vehicles.cpp)
target_link_libraries(vehicles utilities Perlin costmap flowfield ${catkin_LIBRARIES} ${GAZEBO_LIBRARIES})

add_library(world_entities src/world_entities.cpp)
target_link_libraries(world_entities sdfstring utilities ${catkin_LIBRARIES} ${GAZEBO_LIBRARIES})

add_library(quadtree src/include/quadtree.cpp)
target_link_libraries(quadtree utilities ${catkin_LIBRARIES} ${GAZEBO_LIBRARIES})

add_library(parse_tour src/include/parse_tour.cpp)
target_link_libraries(parse_tour costmap ${catkin_LIBRARIES} ${GAZEBO_LIBRARIES})

add_library(puppeteer src/puppeteer.cpp)
target_link_libraries(puppeteer parse_tour utilities quadtree vehicles costmap flowfield ${catkin_LIBRARIES} ${GAZEBO_LIBRARIES})

add_executable(world_factory src/world_factory.cpp) 
target_link_libraries(world_factory world_entities utilities parse_tour costmap ${catkin_LIBRARIES})

add_executable(test_file src/tests.cpp) 
target_link_libraries(test_file world_entities quadtree vehicles costmap ${catkin_LIBRARIES})

add_executable(navigation_goals_V2 src/navigation_goals_V2.cpp) 
target_link_libraries(navigation_goals_V2 parse_tour ${catkin_LIBRARIES} ${GAZEBO_LIBRARIES})


add_library(custom_velodyne src/custom_velodyne.cpp)
target_link_libraries(custom_velodyne
  ${catkin_LIBRARIES}
  ${GAZEBO_LIBRARIES}
  RayPlugin
  utilities
  quadtree
)

add_library(custom_velodyne_gpu src/custom_velodyne.cpp)
target_link_libraries(custom_velodyne_gpu
  ${catkin_LIBRARIES}
  ${GAZEBO_LIBRARIES}
  GpuRayPlugin
  utilities
  quadtree
)
target_compile_definitions(custom_velodyne_gpu PRIVATE GAZEBO_GPU_RAY=1)
