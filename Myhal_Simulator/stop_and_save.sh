#!/bin/bash

# Initial sourcing
source ~/.bashrc
source "/opt/ros/melodic/setup.bash"
source $PWD/simu_melodic_ws/devel/setup.bash

# Echo
TIME=$(rosparam get start_time)
echo "Shutting down run $TIME"

# First kill ros bag record
node=$(rosnode list | grep "record_")
if [ -n "$node" ]; then
    if [ ${node:0:1} = "/" ]; then
        rosnode kill $node
    fi
else
    echo "No rosbag record node running"
fi

# Send shutdown request
rostopic pub -1 /shutdown_signal std_msgs/Bool true 


# # Kill all nodes
# rosnode kill -a
# killall rosmaster
# killall gzserver
# killall gzclient



