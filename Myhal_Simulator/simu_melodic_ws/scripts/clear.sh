rosnode kill -a 
killall gzclient
killall gzserver
killall roscore
killall rosmaster
killall rviz

sleep 2
killall -s SIGKILL roslaunch
killall -s SIGKILL master.sh
killall -s SIGKILL navigation_goals_V2
killall -s SIGKILL amcl

#ps | grep roslaunch
#if [ $? -eq 0 ]; then
#    echo "killing roslaunch"
#    killall -s SIGKILL roslaunch
#fi
#
#ps | grep master.sh
#if [ $? -eq 0 ]; then
#    echo "killing master.sh"
#    killall -s SIGKILL master.sh
#fi




