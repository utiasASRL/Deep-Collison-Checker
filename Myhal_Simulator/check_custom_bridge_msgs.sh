#!/bin/bash

# Verify the custom types were recognized by the bridge, by printing all pairs of bridged types. 
# The custom types should be listed:

cd onboard_deep_sogm

source "install/setup.bash"


echo ""
echo "Bridged tf?"
ros2 run ros1_bridge dynamic_bridge --print-pairs | grep "tf"

echo ""
echo "Bridged VoxGrid?"
ros2 run ros1_bridge dynamic_bridge --print-pairs | grep "Vox"

echo ""
echo "Bridged PointCloud?"
ros2 run ros1_bridge dynamic_bridge --print-pairs | grep "PointCloud"

echo ""
echo "Bridged Obstacle?"
ros2 run ros1_bridge dynamic_bridge --print-pairs | grep "Obstacle"

echo ""