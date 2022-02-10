#!/bin/bash

source /home/$USER/catkin_ws/devel/setup.bash

now=`date +%Y-%m-%d_%H-%M-%S`
nohup roslaunch teb_local_planner test_optim_node.launch > "$PWD/nohup_log_$now.txt" 2>&1 &

echo "Started nohup test_optim_node. PID:"
echo $!
echo ""

until rostopic list; do sleep 0.5; done #wait until rosmaster has started 

rosrun teb_local_planner publish_costmap.py
 
exit 1



