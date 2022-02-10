#!/bin/bash

TIME=$(rosparam get start_time)
FILTER=$(rosparam get filter_status)

echo "Shutting down run $TIME"

rosnode kill -a
killall rosmaster
killall gzserver
killall gzclient

