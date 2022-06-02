#!/bin/bash

# Should we do low weight experiments(only save lidar points and low weight files)
low_weight_exp=true

# List of worlds to load (get if from the SOGM exps)"
worlds=("2022-05-19-23-17-31" \
"2022-05-20-02-45-28" \
"2022-05-20-05-43-43" \
"2022-05-20-09-17-02" \
"2022-05-20-18-51-51" \
"2022-05-20-22-50-12" \
"2022-05-21-02-18-11" \
"2022-05-21-05-50-23" \
"2022-05-21-13-32-14" \
"2022-05-21-16-47-28" \
"2022-05-21-20-07-11" \
"2022-05-24-21-37-31" \
"2022-05-25-00-26-57" \
"2022-05-25-03-51-16" \
"2022-05-25-07-23-35" \
"2022-05-25-09-58-12" \
"2022-05-25-14-24-59" \
"2022-05-25-18-01-16")




for i in {1..1}
do
    for LOADED_WORLD in ${worlds[@]}
    do

        echo $LOADED_WORLD

        # for ARGS in "-fg | -bl" "-fg | -bg"
        for ARGS in "-fg | -bl"
        do

            # Read simu and nav params
            IFS="|" read SIMU_ARGS NAV_ARGS <<< $ARGS

            # Start exp
            ./run_in_melodic.sh -d -c "./simu_master.sh $SIMU_ARGS -t 2022-A -p Flow2_params -l $LOADED_WORLD"
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