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
fi

# Create folder for simulation if not already there
mkdir -p "$PWD/../Data/Simulation/simulated_runs"

# Volumes (modify with your own path here)
volumes="-v $PWD/..:/home/$USER/Deep-Collison-Checker \ 
-v $PWD/../../2-Deep-Collision-Checker/Simulation_Data:/home/$USER/Deep-Collison-Checker/Data/Simulation"

# Additional arguments to be able to open GUI
XSOCK=/tmp/.X11-unix
XAUTH=/home/$USER/.Xauthority
other_args="-v $XSOCK:$XSOCK \
    -v $XAUTH:$XAUTH \
    --net=host \
    --privileged \
	-e XAUTHORITY=${XAUTH} \
    -e DISPLAY=$DISPLAY \
    -w /home/$USER/Deep-Collison-Checker"

# Execute the command in docker (Example of command: ./master.sh -ve -m 2 -p Sc1_params -t A_tour)
docker run $docker_args \
$volumes \
$other_args \
--name "dev-$USER-noetic_pytorch" \
noetic_pytorch_$USER


