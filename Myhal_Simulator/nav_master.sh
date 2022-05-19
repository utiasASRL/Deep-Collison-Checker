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
source "nav_noetic_ws/devel/setup.bash"

# Printing the command used to call this file
myInvocation="$(printf %q "$BASH_SOURCE")$((($#)) && printf ' %q' "$@")"

# Init
XTERM=false     # -x
SOGM=false      # -s
TEB=false       # -b
MAPPING=2       # -m (arg)
LOADTRAJ=false  # -l

# Parse arguments
while getopts xsbm:l option
do
case "${option}"
in
x) XTERM=true;;             # are we using TEB planner
s) SOGM=true;;              # are we using SOGMs
b) TEB=true;;               # are we using TEB planner
m) MAPPING=${OPTARG};;      # use gmapping, AMCL or PointSLAM? (respectively 0, 1, 2)
l) LOADTRAJ=true;;          # are we using loaded traj for GroundTruth predictions
esac
done

echo ""
echo "Waiting for Robot initialization ..."

# Wait until rosmaster has started 
until [[ -n "$rostopics" ]]
do
    rostopics="$(rostopic list)"
    sleep 0.5
done

# Wait for a message with the flow field (meaning the robot is loaded)
until [[ -n "$puppet_state_msg" ]]
do 
    sleep 0.5
    puppet_state_msg=$(rostopic echo -n 1 /puppet_state | grep "running")
done 

# Wait for a message with the pointclouds (meaning everything is ready)
until [[ -n "$velo_state_msg" ]]
do 
    sleep 0.5
    velo_state_msg=$(rostopic echo -n 1 /velodyne_points | grep "header")
done 

echo "OK"

# Get parameters from ROS
echo " "
echo " "
echo -e "\033[1;4;34mReading parameters from ros\033[0m"

rosparam set using_teb $TEB
rosparam set loc_method $MAPPING

GTCLASS=$(rosparam get gt_class)
c_method=$(rosparam get class_method)
TOUR=$(rosparam get tour_name)
t=$(rosparam get start_time)
FILTER=$(rosparam get filter_status)

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
    loc_launch="point_slam simu_ptslam.launch filter:=$FILTER"
fi

# Add map path
loc_launch="$loc_launch init_map_path:=$HOME/Deep-Collison-Checker/Data/Simulation_v2/slam_offline/2020-10-02-13-39-05/map_update_0001.ply"

# Start localization algo
if [ "$XTERM" = true ] ; then
    xterm -bg black -fg lightgray -xrm "xterm*allowTitleOps: false" -T "Localization" -n "Localization" -hold \
        -e roslaunch $loc_launch &
else
    NOHUP_LOC_FILE="$PWD/../Data/Simulation_v2/simulated_runs/$t/logs-$t/nohup_loc.txt"
    nohup roslaunch $loc_launch > "$NOHUP_LOC_FILE" 2>&1 &
fi

# Start point cloud filtering if necessary
if [ "$FILTER" = true ]; then
    if [ "$MAPPING" = "0" ] || [ "$MAPPING" = "1" ]; then
        NOHUP_FILTER_FILE="$PWD/../Data/Simulation_v2/simulated_runs/$t/logs-$t/nohup_filter.txt"
        nohup roslaunch jackal_velodyne pointcloud_filter2.launch gt_classify:=$GTCLASS > "$NOHUP_LOC_FILE" 2>&1 &
    fi
fi

echo "$loc_launch"
echo "OK"

# Waiting for pointslam initialization
echo ""
echo "Waiting for PointSlam initialization ..."
until [ -n "$map_topic" ] 
do 
    sleep 0.1
    map_topic=$(rostopic list -p | grep "/map")
done 
until [[ -n "$point_slam_msg" ]]
do 
    sleep 0.1
    point_slam_msg=$(rostopic echo -n 1 /map | grep "frame_id")
done

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

# Chose parameters for local costmap
if [ "$FILTER" = true ] ; then
    local_costmap_params="local_costmap_filtered_params.yaml"
else
    local_costmap_params="local_costmap_params.yaml"
fi

# Chose parameters for local planner
if [ "$TEB" = true ] ; then
    if [ "$SOGM" = true ] || [ "$LOADTRAJ" = true ] ; then
        local_planner_params="teb_params_sogm.yaml"
    else
        local_planner_params="teb_params_normal.yaml"
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

######
# Rviz
######

RVIZ=false
if [ "$RVIZ" = true ] ; then
    t=$(rosparam get start_time)
    NOHUP_RVIZ_FILE="$PWD/../Data/Simulation_v2/simulated_runs/$t/logs-$t/nohup_rviz.txt"
    nohup rviz -d nav_noetic_ws/src/jackal_velodyne/rviz/nav.rviz > "$NOHUP_RVIZ_FILE" 2>&1 &
fi


##################
# Run Deep Network
##################


if [ "$SOGM" = true ] ; then

    echo " "
    echo " "
    echo -e "\033[1;4;34mStarting SOGM prediction\033[0m"

    cd onboard_deep_sogm/scripts
    ./simu_collider.sh

else

    echo " "
    echo " "
    echo -e "\033[1;4;34mStarting with groundtruth SOGM\033[0m"

    if [ "$LOADTRAJ" = true ] ; then

        # Get the loaded world
        LOADPATH=$(rosparam get load_path)
        LOADWORLD=$(rosparam get load_world)

        if [ "$LOADWORLD" = "" ] || [ "$LOADWORLD" = "none" ] ; then
            echo "XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX"
            echo "X  Error no world loaded that we can use for gt sogm  X"  
            echo "X  load_path = $LOADPATH                              X"
            echo "X  load_world = $LOADWORLD                            X"
            echo "XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX"
        else

            t=$(rosparam get start_time)
            NOHUP_GTSOGM_FILE="$PWD/../Data/Simulation_v2/simulated_runs/$t/logs-$t/nohup_gtsogm.txt"
            nohup rosrun teb_local_planner gt_sogm.py > "$NOHUP_GTSOGM_FILE" 2>&1 &
            
            # Wait for eveyrthing to end before killing the docker container
            velo_state_msg=$(timeout 10 rostopic echo -n 1 /velodyne_points | grep "header")
            until [[ ! -n "$velo_state_msg" ]]
            do 
                sleep 0.5
                velo_state_msg=$(timeout 10 rostopic echo -n 1 /velodyne_points | grep "header")
                echo "Recieved velodyne message, continue navigation"
            done 

        fi

    else
        # Wait for eveyrthing to end before killing the docker container
        velo_state_msg=$(timeout 10 rostopic echo -n 1 /velodyne_points | grep "header")
        until [[ ! -n "$velo_state_msg" ]]
        do 
            sleep 0.5
            velo_state_msg=$(timeout 10 rostopic echo -n 1 /velodyne_points | grep "header")
            echo "Recieved velodyne message, continue navigation"
        done 
    fi
fi


echo "OK"
echo " "
echo " "