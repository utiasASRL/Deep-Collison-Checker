#!/bin/bash

############
# Parameters
############

# Initial sourcing
source ~/.bashrc
source "/opt/ros/melodic/setup.bash"
source $PWD/simu_melodic_ws/devel/setup.bash

# Printing the command used to call this file
myInvocation="$(printf %q "$BASH_SOURCE")$((($#)) && printf ' %q' "$@")"

# List of parameters for the simulator
GUI=false # -v flag
TOUR="A_tour" # -t (arg) flag
LOADWORLD="" # -l (arg) flag
FILTER=false # -f flag
MAPPING=2 # -m (arg) flag
t=$(date +'%Y-%m-%d-%H-%M-%S')
GTCLASS=false # -g flag 
VIZ_GAZ=false
PARAMS="default_params"
TEB=false # -b flag

# Parse parameters
while getopts p:t:l:m:n:vfgeb option
do
case "${option}"
in
p) PARAMS=${OPTARG};;       # what param file are we using?
t) TOUR=${OPTARG};;         # What tour is being used 
l) LOADWORLD=${OPTARG};;    # do you want to load a prexisting world or generate a new one
m) MAPPING=${OPTARG};;      # use gmapping, AMCL or PointSLAM? (respectively 0, 1, 2)
n) t=${OPTARG};;            # Overwrite the date
v) GUI=true;;               # using gui?
f) FILTER=true;;            # pointcloud filtering?
g) GTCLASS=true;;           # are we using ground truth classifications, or online_classifications
e) VIZ_GAZ=true;;           # are we going to vizualize topics in gazebo
b) TEB=true;;               # are we using TEB planner
esac
done

# Display parameters
echo "Folder Name: $t"
MINSTEP=0.0001
echo "Min step size: $MINSTEP"
echo -e "TOUR: $TOUR\nGUI: $GUI\nLOADWORLD: $LOADWORLD\nFILTER: $FILTER\nTEB: $TEB\nMAPPING: $MAPPING\nGTCLASS: $GTCLASS"
echo -e " "

# Handle the choice betwenn gt and predictions
c_method="ground_truth"
if [ "$FILTER" = false ] ; then
    c_method="none"
    GTCLASS=false
else
    if [ "$GTCLASS" = false ] ; then
        c_method="online_predictions"
    else 
        c_method="ground_truth"
    fi
fi
export GTCLASSIFY=$GTCLASS


##########
# ROS CORE
##########

# Start ROS core
echo " "
echo "Running tour: $TOUR"
echo " "

roscore -p $ROSPORT &

echo " "
echo "Waiting for roscore initialization ..."
echo " "

until rostopic list; do sleep 0.5; done #wait until rosmaster has started 

echo " "
echo "Loading parameter files:"
echo " "

# Load parameters
rosparam load $PWD/simu_melodic_ws/src/myhal_simulator/params/$PARAMS/custom_simulation_params.yaml
rosparam load $PWD/simu_melodic_ws/src/myhal_simulator/params/$PARAMS/common_vehicle_params.yaml
rosparam load $PWD/simu_melodic_ws/src/myhal_simulator/params/$PARAMS/animation_params.yaml
rosparam load $PWD/simu_melodic_ws/src/myhal_simulator/params/$PARAMS/room_params_V2.yaml
rosparam load $PWD/simu_melodic_ws/src/myhal_simulator/params/$PARAMS/scenario_params_V2.yaml
rosparam load $PWD/simu_melodic_ws/src/myhal_simulator/params/$PARAMS/plugin_params.yaml
rosparam load $PWD/simu_melodic_ws/src/myhal_simulator/params/$PARAMS/model_params.yaml
rosparam load $PWD/simu_melodic_ws/src/myhal_simulator/params/$PARAMS/camera_params.yaml
rosparam load $PWD/simu_melodic_ws/src/myhal_simulator/tours/$TOUR/config.yaml
rosparam set gt_class $GTCLASS
rosparam set localization_test false
rosparam set class_method $c_method
rosparam set use_sim_time true
rosparam set tour_name $TOUR
rosparam set start_time $t
rosparam set filter_status $FILTER
rosparam set gmapping_status true
rosparam set loc_method $MAPPING
rosparam set min_step $MINSTEP
rosparam set viz_gaz $VIZ_GAZ


##################
# Logs of the Tour
##################

# Create Log folder
if [ ! -d "$PWD/../Data/Simulation_v2/simulated_runs" ]; then
    mkdir "$PWD/../Data/Simulation_v2/simulated_runs"
fi
mkdir "$PWD/../Data/Simulation_v2/simulated_runs/$t"
mkdir "$PWD/../Data/Simulation_v2/simulated_runs/$t/logs-$t"
mkdir "$PWD/../Data/Simulation_v2/simulated_runs/$t/logs-$t/videos/"

# Create log file
LOGFILE="$PWD/../Data/Simulation_v2/simulated_runs/$t/logs-$t/log.txt"
touch $LOGFILE
echo -e "Command used: $myInvocation" >> $LOGFILE
echo -e "\nPointcloud filter params: \n" >> $LOGFILE
echo -e "TOUR: $TOUR\nGUI: $GUI\nLOADWORLD: $LOADWORLD\nFILTER: $FILTER\nMAPPING: $MAPPING\nGTCLASS: $GTCLASS"  >> $LOGFILE

# Create param file
PARAMFILE="$PWD/../Data/Simulation_v2/simulated_runs/$t/logs-$t/params.yaml"
echo -e "$(cat $PWD/simu_melodic_ws/src/myhal_simulator/params/$PARAMS/room_params_V2.yaml)" > $PARAMFILE
echo -e "\n" >> $PARAMFILE
echo -e "$(cat $PWD/simu_melodic_ws/src/myhal_simulator/params/$PARAMS/scenario_params_V2.yaml)" >> $PARAMFILE
echo -e "\n" >> $PARAMFILE
echo -e "$(cat $PWD/simu_melodic_ws/src/myhal_simulator/params/$PARAMS/plugin_params.yaml)" >> $PARAMFILE
echo -e "\n" >> $PARAMFILE
echo -e "tour_name: $TOUR" >> $PARAMFILE

# Create pcl file
PCLFILE="$PWD/../Data/Simulation_v2/simulated_runs/$t/logs-$t/pcl.txt"
echo -e "$(cat $PWD/simu_melodic_ws/src/jackal_velodyne/launch/include/pointcloud_filter2.launch)" >> $PCLFILE

# World file path
WORLDFILE="$PWD/simu_melodic_ws/src/myhal_simulator/worlds/myhal_sim.world"


###################
# Run the simulator
###################

echo -e "\033[1;4;34mStarting simulator\033[0m"

# Create a world in simulation
sleep 0.1
if [[ -z $LOADWORLD ]]; then
    rosrun myhal_simulator world_factory
    rosparam set load_world "none"
else
    WORLDFILE="$PWD/../Data/Simulation_v2/simulated_runs/$LOADWORLD/logs-$LOADWORLD/myhal_sim.world"
    rosparam set load_world $LOADWORLD
    echo "Loading world $WORLDFILE"
fi

# Copy world file in log
cp $WORLDFILE "$PWD/../Data/Simulation_v2/simulated_runs/$t/logs-$t/"

echo -e "\033[1;4;34mStarting rosbag record\033[0m"

# Save rosbag
NOHUP_ROSBAG_FILE="$PWD/../Data/Simulation_v2/simulated_runs/$t/logs-$t/nohup_rosbag.txt"
nohup rosbag record -O "$PWD/../Data/Simulation_v2/simulated_runs/$t/raw_data.bag" \
    /clock \
    /shutdown_signal \
    /velodyne_points \
    /move_base/local_costmap/costmap \
    /move_base/global_costmap/costmap \
    /ground_truth/state \
    /map \
    /move_base/NavfnROS/plan \
    /amcl_pose \
    /tf \
    /tf_static \
    /move_base/result \
    /tour_data \
    /optimal_path \
    /classified_points \
    /plan_costmap_3D \
    /move_base/TebLocalPlannerROS/local_plan \
    /move_base/TebLocalPlannerROS/teb_markers > "$NOHUP_ROSBAG_FILE" 2>&1 &


# Start the simulation
sleep 2.5
echo -e "\033[1;4;34mRUNNING SIM\033[0m"

xterm -bg black -fg lightgray -xrm "xterm*allowTitleOps: false" -T "Gazebo Core" -n "Gazebo Core" -hold \
    -e roslaunch myhal_simulator p1.launch gui:=$GUI world_name:=$WORLDFILE & #extra_gazebo_args:="-s libdirector.so"
sleep 0.5


###################
# Run the simulator
###################

# Wait for simulation to be running before spawning jackal and running navigation
echo ""
echo "Waiting for Gazebo initialization ..."
until [[ -n "$topic1" ]] || [[ -n "$topic2" ]] || [[ -n "$topic3" ]]
do 
    sleep 0.5
    topic1=$(rostopic list -p | grep "/flow_field")
    topic2=$(rostopic list -p | grep "/gazebo/model_states")
    topic3=$(rostopic list -p | grep "/clock")

    #echo "- $topic1 - $topic2 - $topic3 - $topic4 -  "
    #if [[ -n "$topic4" ]]; then
    #    rostopic echo -n 1 /clock
    #fi
done 
echo "OK"

# Spawn the robot and start its controller
roslaunch myhal_simulator jackal_spawn.launch &

# Wait for a message with the flow field (meaning the robot is loaded and everything is ready)
echo ""
echo "Waiting for Robot initialization ..."
until [[ -n "$puppet_state_msg" ]]
do 
    sleep 0.5
    puppet_state_msg=$(rostopic echo -n 1 /puppet_state | grep "running")
done 
echo "OK"

# Start localization algo
xterm -bg black -fg lightgray -xrm "xterm*allowTitleOps: false" -T "Localization" -n "Localization" -hold \
    -e roslaunch myhal_simulator localization.launch filter:=$FILTER loc_method:=$MAPPING gt_classify:=$GTCLASS &

# Start navigation algo
xterm -bg black -fg lightgray -xrm "xterm*allowTitleOps: false" -T "Move base" -n "Move base" -hold \
    -e roslaunch myhal_simulator navigation.launch loc_method:=$MAPPING &

# Run Dashboard
echo -e "\033[1;4;34mStarting dashboard\033[0m"

rosrun dashboard assessor.py

###############################
# Eventually run postprocessing
###############################

# Here shut down the different ros nodes before 
# Movebase
# Loc 
# Goals
# Dashboard



# RUn data processing at the end of the tour
echo "Running data_processing.py"
rosrun dashboard data_processing.py $t $FILTER
 
exit 1



