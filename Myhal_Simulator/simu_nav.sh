#!/bin/bash


TO DO HANDLE THIS FILE














# Arg to specify if we record this run or not
record=false
colli=false

# Arg to specify the name of the waypoints file to follow. If no such file exists, 
# then we wait for the user to set waypoints for this new file
waypoints="default_no_given"

while getopts rcw: option
do
case "${option}"
in
r) record=true;;
c) colli=true;;
w) waypoints=${OPTARG};;
esac
done

# We should start pointslam on the xavier, to remove some of the CPU load

# Waiting for pointslam initialization
echo ""
echo "Waiting for pointslam initialization ..."
map_topic=$(rostopic list | grep "/map")
until [ -n "$map_topic" ] 
do 
    sleep 0.5
    map_topic=$(rostopic list | grep "/map")
done 
echo "OK"


# Now start move_base
if [ "$colli" = true ] ; then
    echo "Running move_base with modified TEB"
    nohup roslaunch jackal_navigation teb_modified.launch > "nohup_teb.txt" 2>&1 &
else
    echo "Running move_base with normal TEB"
    nohup roslaunch jackal_navigation teb_normal.launch > "nohup_teb.txt" 2>&1 &
fi
echo "Running waypoint follower"
nohup roslaunch follow_waypoints follow_waypoints.launch waypoint_file:="$waypoints" > "nohup_waypoint.txt" 2>&1 &









ROS_1_DISTRO=noetic
source "/opt/ros/$ROS_1_DISTRO/setup.bash"
. "../catkin_ws/install_isolated/setup.bash"

echo "Running move_base with modified TEB"
nohup ./teb_modified.sh > "nohup_teb.txt" 2>&1 &


# Start the collider 
./collider.sh