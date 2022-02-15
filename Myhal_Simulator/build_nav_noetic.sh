#!/bin/bash

ROS_1_DISTRO=noetic

source "/opt/ros/$ROS_1_DISTRO/setup.bash"

# Build
cd nav_noetic_ws
# catkin_make_isolated -DCMAKE_BUILD_TYPE=Release --install
catkin build -DCMAKE_BUILD_TYPE=Release

# Test msg 
source devel/setup.bash
echo " "
echo "--------------- VoxGrid ---------------"
rosmsg list | grep "VoxGrid"
rosmsg list | grep "Obstacle"
echo "---------------------------------------"


