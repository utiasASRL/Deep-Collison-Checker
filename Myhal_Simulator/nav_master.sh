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



# 1
#   pointslam no filter
#   move-base with TEB0
#   
# 2
#   pointslam no filter
#   move-base with DWA
#   
# 3
#   pointslam no filter
#   move-base with TEB0
#   GT point filters (ignore dynamic)
#   
# 4
#   pointslam no filter
#   move-base with TEB1
#   SOGM prediction
#   
#   
#   
#   
#   
#   









############
# Parameters
############

# # Initial sourcing
source "/opt/ros/noetic/setup.bash"

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

# First get the chosen launch file
if [ "$MAPPING" = "0" ] ; then
    loc_launch=point_slam gmapping.launch
if [ "$MAPPING" = "1" ] ; then
    loc_launch=point_slam amcl.launch
else
    loc_launch=point_slam point_slam.launch filter:=$FILTER gt_classify:=$GTCLASS
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
        nohup roslaunch point_slam pointcloud_filter2.launch gt_classify:=$GTCLASS > "$NOHUP_LOC_FILE" 2>&1 &
    fi
fi


##################
# Start Navigation
##################

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
nav_command="roslaunch myhal_simulator navigation.launch"
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


##################
# Run Deep Network
##################

if [ "$SOGM" = true ] ; then
    cd onboard_deep_sogm/scripts
    ./collider.sh
fi



