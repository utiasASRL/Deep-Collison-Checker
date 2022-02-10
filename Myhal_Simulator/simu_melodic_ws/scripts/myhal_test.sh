#!/bin/bash

killall gzserver
killall gzclient
killall rviz
killall roscore
killall rosmaster
PARAMS='default_params'

TOUR="A_tour"

roscore -p $ROSPORT&

until rostopic list; do sleep 0.5; done #wait until rosmaster has started 

rosparam load src/myhal_simulator/params/$PARAMS/common_vehicle_params.yaml
rosparam load src/myhal_simulator/params/$PARAMS/animation_params.yaml
rosparam load src/myhal_simulator/params/$PARAMS/room_params_V2.yaml
rosparam load src/myhal_simulator/params/$PARAMS/scenario_params_V2.yaml
rosparam load src/myhal_simulator/params/$PARAMS/plugin_params.yaml
rosparam load src/myhal_simulator/params/$PARAMS/model_params.yaml
rosparam load src/myhal_simulator/params/$PARAMS/camera_params.yaml
rosparam load src/myhal_simulator/tours/$TOUR/config.yaml

sleep 0.1

rosrun myhal_simulator world_factory

roslaunch jackal_velodyne myhal_sim_test.launch

