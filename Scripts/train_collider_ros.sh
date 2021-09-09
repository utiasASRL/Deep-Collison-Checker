#!/bin/bash

########
# Init #
########

echo ""
echo "Running ros-collider docker"
echo ""

rosport=$ROSPORT
detach=false
command=""

while getopts dc: option
do
case "${option}"
in
d) detach=true;; 
c) command=${OPTARG};;
esac
done

if [[ -z "$ROSPORT" ]]; then
    echo "WARNING: didn't provide ROSPORT, setting it to 1100"
    export ROSPORT=1100
fi

last_line=$(tail -1 ~/.bashrc)
s=${last_line:0:14}
if [[ "$s" == "export ROSPORT" ]]; then
    sed -i '$d' ~/.bashrc
fi

echo "ROSPORT=$rosport"
gazport=$(($rosport+1))
export ROSPORT=$(($ROSPORT+2))
echo "export ROSPORT=$ROSPORT" >> ~/.bashrc

##########################
# Start docker container #
##########################

# Docker run arguments
docker_args="-it --rm --shm-size=64g "

# Running on gpu (Uncomment to enable gpu)
docker_args="${docker_args} --gpus all "

# Docker run arguments (depending if we run detached or not)
if [ "$detach" = true ] ; then
    # args for detached docker
    docker_args="-d ${docker_args}"
    now=`date +%Y-%m-%d_%H-%M-%S`
    mkdir -p $PWD/../../KPConv_results/Log_"$now"
fi

# Create folder for simulation if not already there
mkdir -p "$PWD/../../Simulation_Data/simulated_runs"

# Volumes (modify with your own path here)
volumes="-v $PWD/../../MyhalSimulator-DeepCollider:/home/$USER/catkin_ws \
-v $PWD/../../Simulation_Data:/home/$USER/Myhal_Simulation \
-v $PWD/../../MyhalSimulator:/home/$USER/MyhalSimulator \
-v $PWD/../../KPConv_Data:/home/$USER/Data/MyhalSim \
-v $PWD/../../KPConv_results:/home/$USER/catkin_ws/src/collision_trainer/results"

# Additional arguments to be able to open GUI
XSOCK=/tmp/.X11-unix
XAUTH=/home/$USER/.Xauthority
other_args="-v $XSOCK:$XSOCK \
    -v $XAUTH:$XAUTH \
    --net=host \
    --privileged \
	-e XAUTHORITY=${XAUTH} \
    -e DISPLAY=$DISPLAY \
    -e ROS_MASTER_URI=http://$HOSTNAME:$rosport \
    -e GAZEBO_MASTER_URI=http://$HOSTNAME:$gazport \
    -e ROSPORT=$rosport "

# python command started in the docker
if [ ! "$command" ] ; then
    if [ "$detach" = true ] ; then
        py_command="./collision_trainer.sh results/Log_$now"
    else
        py_command="./collision_trainer.sh"
    fi
else
    py_command="./start_script.sh $command"
fi

echo -e "Running command $py_command\n"

# Execute the command in docker (Example of command: ./master.sh -ve -m 2 -p Sc1_params -t A_tour)
docker run $docker_args \
$volumes \
$other_args \
--name "$USER-training-$ROSPORT" \
docker_ros_noetic2_$USER \
$py_command

# Attach a log parameters and log the detached docker
if [ "$detach" = true ] ; then
    docker logs -f "$USER-training-$ROSPORT" &> $PWD/../../KPConv_results/Log_"$now"/log.txt &
fi

source ~/.bashrc

