#!/bin/bash
t=$(date +'%Y-%m-%d-%H-%M-%S')
echo "Folder Name: $t"

killall gzserver
killall gzclient
killall rviz
killall roscore
killall rosmaster

LOADWORLD=""
MAPPING=true
GT=false
FILTER=true

while getopts a:l:g:f: option
do
case "${option}"
in
a) MAPPING=${OPTARG};;
l) LOADWORLD=${OPTARG};; 
g) GT=${OPTARG};; 
f) FILTER=${OPTARG};; 
esac
done

roscore -p $ROSPORT&

until rostopic list; do sleep 0.5; done #wait until rosmaster has started 

INFILE="/home/$USER/Myhal_Simulation/simulated_runs/$LOADWORLD/raw_data.bag"

c_method="hugues_annotations"

LIDARTOPIC="/hugues_points"
if [ "$GT" == "true" ]; then
    LIDARTOPIC="/velodyne_points"
    c_method="ground_truth"
fi

mkdir "/home/$USER/Myhal_Simulation/simulated_runs/$t"
mkdir "/home/$USER/Myhal_Simulation/simulated_runs/$t/logs-$t"

# load parameters

rosparam set localization_test true
rosparam set use_sim_time true
rosparam set filter_status $FILTER
rosparam set gmapping_status $MAPPING
rosparam set start_time $t
rosparam set classify true
rosparam set class_method $c_method
rosparam set load_world $LOADWORLD
rosparam load /home/$USER/Myhal_Simulation/simulated_runs/$LOADWORLD/logs-$LOADWORLD/params.yaml


# record logs

cp "/home/$USER/Myhal_Simulation/simulated_runs/$LOADWORLD/logs-$LOADWORLD/myhal_sim.world" "/home/$USER/Myhal_Simulation/simulated_runs/$t/logs-$t/myhal_sim.world"
cp "/home/$USER/Myhal_Simulation/simulated_runs/$LOADWORLD/logs-$LOADWORLD/params.yaml" "/home/$USER/Myhal_Simulation/simulated_runs/$t/logs-$t/params.yaml" 
LOGFILE="/home/$USER/Myhal_Simulation/simulated_runs/$t/logs-$t/logs.txt"
touch $LOGFILE


rosbag record -O "/home/$USER/Myhal_Simulation/simulated_runs/$t/localization_test.bag" -a -x "(.*)points(.*)" & # Limiting data to remain under rosbag buffer
roslaunch jackal_velodyne hugues_test.launch mapping:=$MAPPING in_topic:=$LIDARTOPIC filter:=$FILTER &
rosbag play --wait-for-subscribers $INFILE --topics /tf /tf_static $LIDARTOPIC /clock /move_base/result /shutdown_signal /ground_truth/state
sleep 0.5
rosrun dashboard data_processing.py $t

