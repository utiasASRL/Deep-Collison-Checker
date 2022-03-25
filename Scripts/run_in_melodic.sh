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
RES_FOLDER="$PWD/../Data/Simulation_v2/nohup_logs"


##################
# Handle rosport #
##################

rosport=1100
gazport=$(($rosport+1))
echo "ROSPORT=$rosport"
echo "GAZPORT=$gazport"

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
    mkdir -p $RES_FOLDER
fi

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
    -e ROSPORT=$rosport \
    -e ROS_MASTER_URI=http://$HOSTNAME:$rosport \
    -e GAZEBO_MASTER_URI=http://$HOSTNAME:$gazport \
    -e QT_X11_NO_MITSHM=1 \
    -w /home/$USER/Deep-Collison-Checker/Myhal_Simulator"

if [ "$devdoc" = true ] ; then

    # Execute the command in docker (Example of command: ./master.sh -ve -m 2 -p Sc1_params -t A_tour)
    docker run $docker_args \
    $volumes \
    $other_args \
    --name "dev-simu" \
    melodic_simu_$USER \
    $command

else

    # python command started in the docker
    if [ ! "$command" ] ; then
        command=" "
    fi

    echo -e "Running command $command\n"

    # Execute the command in docker (Example of command: ./master.sh -ve -m 2 -p Sc1_params -t A_tour)
    docker run $docker_args \
    $volumes \
    $other_args \
    --name "$USER-simu-$now" \
    melodic_simu_$USER \
    $command

    # Attach a log parameters and log the detached docker
    if [ "$detach" = true ] ; then
        docker logs -f "$USER-simu-$now" &> $RES_FOLDER/log_simu_"$now".txt &
    fi

fi