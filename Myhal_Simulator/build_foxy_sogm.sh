#!/bin/bash

source "/opt/ros/foxy/setup.bash"

cd onboard_deep_sogm

# Build
colcon build --symlink-install --packages-skip ros1_bridge --cmake-args -DCMAKE_BUILD_TYPE=Release

# Test msg 
source install/setup.bash
echo " "
echo "--------------- test VoxGrid ---------------"
ros2 interface list | grep "VoxGrid"
ros2 interface list | grep "Obstacle"
echo "--------------------------------------------"

# Also build cpp wrappers
cd src/deep_sogm/deep_sogm/cpp_wrappers
./compile_wrappers.sh