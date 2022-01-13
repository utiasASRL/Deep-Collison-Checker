#!/bin/bash

########
# Init #
########

echo ""
echo "Running ros-collider docker"
echo ""

detach=false
devdoc=false
command=""

while getopts dvc: option
do
case "${option}"
in
d) detach=true;; 
v) devdoc=true;; 
c) command=${OPTARG};;
esac
done

# Path to the Network result folder
RES_FOLDER="$PWD/../SOGM-3D-2D-Net/results"

##########################
# Start docker container #
##########################

# Docker run arguments
docker_args="-it --rm --shm-size=64g "

# Running on gpu (Uncomment to enable gpu)
docker_args="${docker_args} --gpus all "

# Docker run arguments (depending if we run detached or not)
now=`date +%Y-%m-%d_%H-%M-%S`
if [ "$detach" = true ] ; then
    # args for detached docker
    docker_args="-d ${docker_args}"
    mkdir -p $RES_FOLDER/Log_"$now"
fi

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
    -w /home/$USER/Deep-Collison-Checker/SOGM-3D-2D-Net"

if [ "$devdoc" = true ] ; then

    # Execute the command in docker (Example of command: ./master.sh -ve -m 2 -p Sc1_params -t A_tour)
    docker run $docker_args \
    $volumes \
    $other_args \
    --name "dev-SOGM" \
    noetic_pytorch_$USER \
    $command

else

    # python command started in the docker
    if [ ! "$command" ] ; then
        command="python3 train_MyhalCollision.py"
    fi

    # Adding detached folder if needed
    if [ "$detach" = true ] ; then
        if [[ $command == *"python3 train_"* ]]; then
            command="$command results/Log_$now"
        fi
            
    fi

    echo -e "Running command $command\n"

    # Execute the command in docker (Example of command: ./master.sh -ve -m 2 -p Sc1_params -t A_tour)
    docker run $docker_args \
    $volumes \
    $other_args \
    --name "$USER-SOGM-$now" \
    noetic_pytorch_$USER \
    $command

    # Attach a log parameters and log the detached docker
    if [ "$detach" = true ] ; then
        docker logs -f "$USER-SOGM-$now" &> $RES_FOLDER/Log_"$now"/log.txt &
    fi


fi