#!/bin/bash

# World names in folder /home/hth/Deep-Collison-Checker/Data/Simulation_v2/simulated_runs/
easy="2022-05-18-21-23-50"
med="2022-05-18-22-22-02"
hard="2022-05-18-23-24-51"

# for LOADED_WORLD in $easy $med $hard
for LOADED_WORLD in $easy $hard
do

    for ARGS in "-g | -b" "-fg | -bl" "-fg | -bs"
    do

        # Read simu and nav params
        IFS="|" read SIMU_ARGS NAV_ARGS <<< $ARGS

        # Start exp
        ./run_in_melodic.sh -d -c "./simu_master.sh $SIMU_ARGS -t 2022-A -p FlowCorners_params -l $med"
        sleep 2.0
        ./run_in_foxy.sh -d -c "./nav_master.sh $NAV_ARGS -m 2"
                
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


    done
done