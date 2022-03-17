#!/bin/bash

# Initial sourcing
source ~/.bashrc
source "/opt/ros/melodic/setup.bash"
source $PWD/simu_melodic_ws/devel/setup.bash

rostopic pub -1 /shutdown_signal std_msgs/Bool true 
