#!/bin/bash

source /home/$USER/catkin_ws/devel/setup.bash

until rostopic list; do sleep 0.5; done #wait until rosmaster has started 

rosrun teb_local_planner publish_costmap.py
 
exit 1



