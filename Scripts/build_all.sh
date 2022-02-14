#!/bin/bash

./run_in_melodic.sh -c "./build_simu_melodic.sh"

./run_in_foxy.sh -c "./build_nav_noetic.sh"

./run_in_foxy.sh -c "./build_foxy_sogm.sh"

./run_in_foxy.sh -c "./build_ros1_bridge.sh"

./run_in_foxy.sh -c "./check_custom_bridge_msgs.sh"


