#!/bin/bash

now=`date +%Y-%m-%d_%H-%M-%S`
NOHUP_FILE="$PWD/../Data/Simulation_v2/nohup_logs/log-exp_$now.txt"
PID_FILE="$PWD/../Data/Simulation_v2/nohup_logs/pid-exp_$now.txt"
nohup ./2-exp_sota.sh > "$NOHUP_FILE" 2>&1 &
echo $! > "$PID_FILE"