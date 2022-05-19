#!/bin/bash

# World names in folder /home/hth/Deep-Collison-Checker/Data/Simulation_v2/simulated_runs/
easy="2022-05-18-21-23-50"
med="2022-05-18-22-22-02"
hard="2022-05-18-23-24-51"

##############################################################################################

# Simple exp to start
# *******************

# Simulation witohut filtering activated
./run_in_melodic.sh -d -c "./simu_master.sh -g -t 2022-A -p FlowCorners_params -l $med"

# Navigation with normal TEB
./run_in_foxy.sh -d -c "./nav_master.sh -b -m 2"

# Wait for the docker containers to be stopped
sleep 2.0
docker_msg=$(docker ps | grep "hth-foxy")
until [[ ! -n "$docker_msg" ]]
do 
    sleep 5.0
    docker_msg=$(docker ps | grep "hth-foxy")
    echo "Recieved docker message, continue experiment"
done 

# Sleep a bit to be sure  
echo "Experiment finished"
sleep 2.0

##############################################################################################

# Second exp to start
# *******************

# Simulation with filtering activated
./run_in_melodic.sh -d -c "./simu_master.sh -fg -t 2022-A -p FlowCorners_params -l $med"

# Navigation with Groundtruth SOGM
./run_in_foxy.sh -d -c "./nav_master.sh -bl -m 2"

# Wait for the docker containers to be stopped
sleep 2.0
docker_msg=$(docker ps | grep "hth-foxy")
until [[ ! -n "$docker_msg" ]]
do 
    sleep 5.0
    docker_msg=$(docker ps | grep "hth-foxy")
    echo "Recieved docker message, continue experiment"
done 

# Sleep a bit to be sure  
echo "Experiment finished"
sleep 2.0

##############################################################################################

# Third exp to start
# ******************

# Simulation with filtering activated
./run_in_melodic.sh -d -c "./simu_master.sh -fg -t 2022-A -p FlowCorners_params -l $med"

# Navigation with predicted SOGM
./run_in_foxy.sh -d -c "./nav_master.sh -bs -m 2"
