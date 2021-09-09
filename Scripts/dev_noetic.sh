#!/bin/bash

########
# Init #
########

echo ""
echo "Running noetic-pytorch docker in dev mode"
echo ""

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
    mkdir -p $PWD/../Data/KPConv_results/Log_"$now"
fi

# Create folder for simulation if not already there
mkdir -p "$PWD/../Data/Simulation/simulated_runs"

# Volumes (modify with your own path here)
volumes="-v $PWD/..:/home/$USER/Deep-Collison-Checker"

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

# Execute the command in docker (Example of command: ./master.sh -ve -m 2 -p Sc1_params -t A_tour)
docker run $docker_args \
$volumes \
$other_args \
--name "$USER-noetic_pytorch-dev" \
noetic_pytorch_$USER

# Attach a log parameters and log the detached docker
if [ "$detach" = true ] ; then
    docker logs -f "$USER-training-$ROSPORT" &> $PWD/../../KPConv_results/Log_"$now"/log.txt &
fi

source ~/.bashrc

