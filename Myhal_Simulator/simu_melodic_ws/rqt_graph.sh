#!/bin/bash

echo ""
echo "Starting new shell in the docker container $@"
echo ""

rosrun tf view_frames

rosparam set enable_statistics true

rosrun rqt_graph rqt_graph

