#!/bin/bash

#############
# Description
#############

# This script is called from the ros1-ros2 foxy docker and does the following:
#
# 1) Read parameters including start time, filter etc
# 2) Start the ros nodes that we want:
#       > Localization
#       > Move Base
#           - Local planner
#           - Global Planner
#       > Deep SOGM predict
#       > (pointfilter, others ...)


############
# Parameters
############

# # Initial sourcing
source "/opt/ros/noetic/setup.bash"
source "nav_noetic_ws/devel_isolated/setup.bash"

# Printing the command used to call this file
myInvocation="$(printf %q "$BASH_SOURCE")$((($#)) && printf ' %q' "$@")"

# Init
XTERM=false     # -x
SOGM=false      # -s

# Parse arguments
while getopts xs option
do
case "${option}"
in
x) XTERM=true;;     # are we using TEB planner
x) SOGM=true;;     # are we using SOGMs
esac
done

# Wait for a message with the flow field (meaning the robot is loaded and everything is ready)
echo ""
echo "Waiting for Robot initialization ..."
until [[ -n "$puppet_state_msg" ]]
do 
    sleep 0.5
    puppet_state_msg=$(rostopic echo -n 1 /puppet_state | grep "running")
done 
echo "OK"

# Get parameters from ROS
echo " "
echo " "
echo -e "\033[1;4;34mReading parameters from ros\033[0m"
GTCLASS=$(rosparam get gt_class)
c_method=$(rosparam get class_method)
TOUR=$(rosparam get tour_name)
t=$(rosparam get start_time)
FILTER=$(rosparam get filter_status)
MAPPING=$(rosparam get loc_method)
TEB=$(rosparam get using_teb)

echo " "
echo "START TIME: $t"
echo "TOUR: $TOUR"
echo "MAPPING: $MAPPING"
echo "FILTER: $FILTER"
echo "GTCLASS: $GTCLASS"
echo "TEB: $TEB"


####################
# Start Localization
####################

echo " "
echo " "
echo -e "\033[1;4;34mStarting localization\033[0m"

# First get the chosen launch file
if [ "$MAPPING" = "0" ] ; then
    loc_launch="jackal_velodyne gmapping.launch"
elif [ "$MAPPING" = "1" ] ; then
    loc_launch="jackal_velodyne amcl.launch"
else
    loc_launch="jackal_velodyne point_slam.launch filter:=$FILTER gt_classify:=$GTCLASS"
fi

if [ "$FILTER" = true ] ; then
    scan_topic="/filtered_points"
else
    scan_topic="/velodyne_points"
fi

# Start localization algo
if [ "$XTERM" = true ] ; then
    xterm -bg black -fg lightgray -xrm "xterm*allowTitleOps: false" -T "Localization" -n "Localization" -hold \
        -e roslaunch $loc_launch scan_topic:=$scan_topic &
else
    NOHUP_LOC_FILE="$PWD/../Data/Simulation_v2/simulated_runs/$t/logs-$t/nohup_loc.txt"
    nohup roslaunch $loc_launch scan_topic:=$scan_topic > "$NOHUP_LOC_FILE" 2>&1 &
fi

# Start point cloud filtering if necessary
if [ "$FILTER" = true ]; then
    if [ "$MAPPING" = "0" ] || [ "$MAPPING" = "1" ]; then
        NOHUP_FILTER_FILE="$PWD/../Data/Simulation_v2/simulated_runs/$t/logs-$t/nohup_filter.txt"
        nohup roslaunch jackal_velodyne pointcloud_filter2.launch gt_classify:=$GTCLASS > "$NOHUP_LOC_FILE" 2>&1 &
    fi
fi

echo "OK"

##################
# Start Navigation
##################

echo " "
echo " "
echo -e "\033[1;4;34mStarting navigation\033[0m"

# Chose parameters for global costmap
if [ "$MAPPING" = "0" ] ; then
    global_costmap_params="gmapping_costmap_params.yaml"
else
    if [ "$FILTER" = true ] ; then
        global_costmap_params="global_costmap_filtered_params.yaml"
    else
        global_costmap_params="global_costmap_params.yaml"
    fi
fi
# TODO HERE THE GLOBAL COSTMAP PARAMETERS FOR POINTSLAM SHOULD NOT HAVE A OBSTACLE PLUGINS LIKE GMAPPINMG
# VERIFY THIS ON THE ROBOT

# Chose parameters for local costmap
if [ "$FILTER" = true ] ; then
    local_costmap_params="local_costmap_filtered_params.yaml"
else
    local_costmap_params="local_costmap_params.yaml"
fi

# Chose parameters for local planner
if [ "$TEB" = true ] ; then
    if [ "$SOGM" = true ] ; then
        local_planner_params="teb_sogm_params.yaml"
    else
        local_planner_params="teb_normal_params.yaml"
    fi
else
    local_planner_params="base_local_planner_params.yaml"
fi

# Chose local planner algo
if [ "$TEB" = true ] ; then
    local_planner="teb_local_planner/TebLocalPlannerROS"
else
    local_planner="base_local_planner/TrajectoryPlannerROS"
fi

# Create launch command
nav_command="roslaunch jackal_velodyne navigation.launch"
nav_command="${nav_command} global_costmap_params:=$global_costmap_params"
nav_command="${nav_command} local_costmap_params:=$local_costmap_params"
nav_command="${nav_command} local_planner_params:=$local_planner_params"
nav_command="${nav_command} local_planner:=$local_planner"

# Start navigation algo
if [ "$XTERM" = true ] ; then
    xterm -bg black -fg lightgray -xrm "xterm*allowTitleOps: false" -T "Move base" -n "Move base" -hold \
        -e $nav_command &
else
    NOHUP_NAV_FILE="$PWD/../Data/Simulation_v2/simulated_runs/$t/logs-$t/nohup_nav.txt"
    nohup $nav_command > "$NOHUP_NAV_FILE" 2>&1 &
fi

echo "OK"


##################
# Run Deep Network
##################

echo " "
echo " "
echo -e "\033[1;4;34mStarting SOGM prediction\033[0m"

if [ "$SOGM" = true ] ; then
    cd onboard_deep_sogm/scripts
    ./collider.sh #TODO THIS IS THE FILE FOR THE ROBOT< SSO CREATE NEW ONE WITH THE RIGHT SOURCING FOR THE SIMU
fi
echo "OK"
echo " "
echo " "

# Wait for eveyrthing to end before killing the docker container
sleep 10
sleep 10
sleep 10
sleep 10
sleep 10
sleep 10
sleep 10
sleep 10
sleep 10



