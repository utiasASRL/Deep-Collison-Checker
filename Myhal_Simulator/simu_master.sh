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

# Current time
t=$(date +'%Y-%m-%d-%H-%M-%S')

# List of parameters for the simulator
PARAMS="default_params" # -p (arg)
GUI=false       # -v
TOUR="A_tour"   # -t (arg)
LOADWORLD=""    # -l (arg)
FILTER=false    # -f
GTCLASS=false   # -g 
VIZ_GAZ=false   # -e 
RVIZ=false      # -r
XTERM=false     # -x

# Parse parameters
while getopts p:t:l:m:n:vrfgebx option
do
case "${option}"
in
p) PARAMS=${OPTARG};;       # what param file are we using?
t) TOUR=${OPTARG};;         # What tour is being used 
l) LOADWORLD=${OPTARG};;    # do you want to load a prexisting world or generate a new one
n) t=${OPTARG};;            # Overwrite the date
v) GUI=true;;               # using gui?
r) RVIZ=true;;              # using rviz?
f) FILTER=true;;            # pointcloud filtering?
g) GTCLASS=true;;           # are we using ground truth classifications, or online_classifications
e) VIZ_GAZ=true;;           # are we going to vizualize topics in gazebo
x) XTERM=true;;             # are we using xterm windows
esac
done

# Display parameters
echo "Folder Name: $t"
MINSTEP=0.0001
echo "Min step size: $MINSTEP"
echo -e "TOUR: $TOUR\nGUI: $GUI\nLOADWORLD: $LOADWORLD\nFILTER: $FILTER\nGTCLASS: $GTCLASS"
echo -e " "






# TODO:
#
#   > use frame_dl = 0.06 in simu_ptslam.launch !!!!!!!!!!!!!
#   > Exp in simu to find the right parameters for navigation
#   > blur in t dimension
#
# Runs to debug:
#       04/01-10h52 => debug run to replay in rviz. Avoiding me, and going in the stairs at the end
#       04/01-10h56 => run with sogm, not much happening















# Handle the choice between gt and predictions
c_method="none"
if [ "$GTCLASS" = false ] ; then
    if [ "$FILTER" = false ] ; then
        c_method="none"
    else
        c_method="online_predictions"
    fi
else 
    c_method="ground_truth"
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
rosparam set min_step $MINSTEP
rosparam set viz_gaz $VIZ_GAZ

#    0.05 => for videos    /    0.2 => for fast exps
rosparam set real_time_factor 0.05


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
echo -e "TOUR: $TOUR\nGUI: $GUI\nLOADWORLD: $LOADWORLD\nFILTER: $FILTER\nGTCLASS: $GTCLASS"  >> $LOGFILE

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
WORLDFILE_LOAD=""


###################
# Run the simulator
###################

echo -e "\033[1;4;34mStarting simulator\033[0m"

# Create a world in simulation
sleep 0.1
LOADPATH="$PWD/../Data/Simulation_v2/simulated_runs"
rosparam set load_path $LOADPATH
if [[ -z $LOADWORLD ]]; then
    rosrun myhal_simulator world_factory
    rosparam set load_world "none"
else
    rosparam set load_world $LOADWORLD
    WORLDFILE_LOAD="$LOADPATH/$LOADWORLD/logs-$LOADWORLD/myhal_sim.world"
    echo "Loading world $WORLDFILE_LOAD"
fi

# Copy the loaded world file to our workspace if necessary
if [[ -n "$WORLDFILE_LOAD" ]]; then
    cp $WORLDFILE_LOAD $WORLDFILE
fi

# Copy world file in log
cp $WORLDFILE "$LOADPATH/$t/logs-$t/"

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

    
# setup ros environment
source "/usr/share/gazebo/setup.sh"
export QT_X11_NO_MITSHM=1


# Start the simulation
sleep 2.5
echo -e "\033[1;4;34mRUNNING SIM\033[0m"

OLD_DISPLAY=$DISPLAY
export DISPLAY=:0
echo $DISPLAY

if [ "$XTERM" = true ] ; then

    # Prb with DISPLAY MODIFIED, the xterm windows open on the server and not on the ssh session
    # xterm -bg black -fg lightgray -xrm "xterm*allowTitleOps: false" -T "Gazebo Core" -n "Gazebo Core" -hold \
    #     -e roslaunch myhal_simulator p1.launch world_name:=$WORLDFILE & #extra_gazebo_args:="-s libdirector.so"

    roslaunch myhal_simulator p1.launch world_name:=$WORLDFILE &

else
    NOHUP_GAZ_FILE="$PWD/../Data/Simulation_v2/simulated_runs/$t/logs-$t/nohup_gazebo.txt"
    nohup roslaunch myhal_simulator p1.launch world_name:=$WORLDFILE > "$NOHUP_GAZ_FILE" 2>&1 &
fi

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
NOHUP_SPAWN_FILE="$PWD/../Data/Simulation_v2/simulated_runs/$t/logs-$t/nohup_spawn.txt"
nohup roslaunch myhal_simulator jackal_spawn.launch > "$NOHUP_SPAWN_FILE" 2>&1 &

# Wait for a message with the flow field (meaning the robot is loaded and everything is ready)
echo ""
echo "Waiting for Robot initialization ..."
until [[ -n "$puppet_state_msg" ]]
do 
    sleep 0.5
    puppet_state_msg=$(rostopic echo -n 1 /puppet_state | grep "running")
    echo "$puppet_state_msg"
done 
echo "OK"

# Run rviz and client if needed
if [ "$GUI" = true ] ; then
    export DISPLAY=$OLD_DISPLAY
    echo $DISPLAY
    NOHUP_GC_FILE="$PWD/../Data/Simulation_v2/simulated_runs/$t/logs-$t/nohup_gzclient.txt"
    nohup roslaunch myhal_simulator p1_gzclient.launch > "$NOHUP_GC_FILE" 2>&1 &
fi

if [ "$RVIZ" = true ] ; then
    export DISPLAY=$OLD_DISPLAY
    echo $DISPLAY
    NOHUP_RVIZ_FILE="$PWD/../Data/Simulation_v2/simulated_runs/$t/logs-$t/nohup_rviz.txt"
    nohup roslaunch myhal_simulator p1_rviz.launch > "$NOHUP_RVIZ_FILE" 2>&1 &
fi



# Run Dashboard
echo -e "\033[1;4;34mStarting dashboard\033[0m"

rosrun myhal_simulator navigation_goals_V2 &
rosrun dashboard meta_data.py &

rosrun dashboard assessor.py


##################
# Start Navigation
##################

# # Start localization algo
# if [ "$XTERM" = true ] ; then
#     xterm -bg black -fg lightgray -xrm "xterm*allowTitleOps: false" -T "Localization" -n "Localization" -hold \
#         -e roslaunch myhal_simulator localization.launch filter:=$FILTER loc_method:=$MAPPING gt_classify:=$GTCLASS &
# else
#     NOHUP_LOC_FILE="$PWD/../Data/Simulation_v2/simulated_runs/$t/logs-$t/nohup_loc.txt"
#     nohup roslaunch myhal_simulator localization.launch filter:=$FILTER loc_method:=$MAPPING gt_classify:=$GTCLASS > "$NOHUP_LOC_FILE" 2>&1 &
# fi


# # Start navigation algo
# if [ "$XTERM" = true ] ; then
#     xterm -bg black -fg lightgray -xrm "xterm*allowTitleOps: false" -T "Move base" -n "Move base" -hold \
#         -e roslaunch myhal_simulator navigation.launch loc_method:=$MAPPING &
# else
#     NOHUP_NAV_FILE="$PWD/../Data/Simulation_v2/simulated_runs/$t/logs-$t/nohup_nav.txt"
#     nohup roslaunch myhal_simulator navigation.launch loc_method:=$MAPPING > "$NOHUP_NAV_FILE" 2>&1 &
# fi


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
sleep 1.5
rosrun dashboard data_processing.py $t $FILTER
 
exit 1



