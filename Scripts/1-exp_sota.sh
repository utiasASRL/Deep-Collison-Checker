#!/bin/bash

# Should we do low weight experiments(only save lidar points and low weight files)
low_weight_exp=true

# # World names in folder /home/hth/Deep-Collison-Checker/Data/Simulation_v2/simulated_runs/
# easy="2022-05-18-21-23-50"
# med="2022-05-18-22-22-02"
# hard="2022-05-18-23-24-51"

for i in {1..20}
do
    # for PARAMS in "Flow1_params" "Flow2_params" "Flow3_params"
    for PARAMS in "Flow2_params"
    do

        # for ARGS in "-g | -b" "-fg | -bs" "-fg | -bi"
        for ARGS in "-fg | -bs"
        do

            # Read simu and nav params
            IFS="|" read SIMU_ARGS NAV_ARGS <<< $ARGS

            # Start exp
            ./run_in_melodic.sh -d -c "./simu_master.sh $SIMU_ARGS -t 2022-A -p $PARAMS"
            sleep 5.0
            sleep 5.0
            sleep 5.0
            ./run_in_foxy.sh -d -c "./nav_master.sh $NAV_ARGS -m 2"
                    
            # Wait for the docker containers to be stopped
            sleep 5.0
            docker_msg=$(docker ps | grep "hth-foxy")
            until [[ ! -n "$docker_msg" ]]
            do 
                sleep 5.0
                docker_msg=$(docker ps | grep "hth-foxy")
                echo "Recieved docker message, continue experiment"
            done 

            # Sleep a bit to be sure  
            echo "Experiment finished"
            sleep 5.0

            if [ "$low_weight_exp" = true ] ; then
                ./run_in_pytorch.sh -d -c "./ros_python.sh clean_last_simu.py"
            fi
            
            sleep 5.0

        done
    done
done