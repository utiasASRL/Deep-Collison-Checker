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

# Parse arguments
while getopts x option
do
case "${option}"
in
x) XTERM=true;;               # are we using TEB planner
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


##################
# Start Navigation
##################

# First get the chosen launch file
if [ "$MAPPING" = "0" ] ; then
    loc_launch=point_slam gmapping.launch
if [ "$MAPPING" = "1" ] ; then
    loc_launch=point_slam localization.launch
else
    loc_launch=point_slam localization.launch
fi

if [ "$FILTER" = true ] ; then
    scan_topic="/filtered_points"
else
    scan_topic="/velodyne_points"
fi

roslaunch $loc_launch scan_topic:=$scan_topic &





# Start localization algo
if [ "$XTERM" = true ] ; then
    xterm -bg black -fg lightgray -xrm "xterm*allowTitleOps: false" -T "Localization" -n "Localization" -hold \
        -e roslaunch myhal_simulator localization.launch filter:=$FILTER loc_method:=$MAPPING gt_classify:=$GTCLASS &
else
    NOHUP_LOC_FILE="$PWD/../Data/Simulation_v2/simulated_runs/$t/logs-$t/nohup_loc.txt"
    nohup roslaunch myhal_simulator localization.launch filter:=$FILTER loc_method:=$MAPPING gt_classify:=$GTCLASS > "$NOHUP_LOC_FILE" 2>&1 &
fi


# Start navigation algo
if [ "$XTERM" = true ] ; then
    xterm -bg black -fg lightgray -xrm "xterm*allowTitleOps: false" -T "Move base" -n "Move base" -hold \
        -e roslaunch myhal_simulator navigation.launch loc_method:=$MAPPING &
else
    NOHUP_NAV_FILE="$PWD/../Data/Simulation_v2/simulated_runs/$t/logs-$t/nohup_nav.txt"
    nohup roslaunch myhal_simulator navigation.launch loc_method:=$MAPPING > "$NOHUP_NAV_FILE" 2>&1 &
fi


##################
# Run Deep Network
##################


###############################
# Eventually run postprocessing
###############################

# Here shut down the different ros nodes before 
# Movebase
# Loc 
# Goals
# Dashboard



# Run data processing at the end of the tour
echo "Running data_processing.py"
rosrun dashboard data_processing.py $t $FILTER
 
exit 1



