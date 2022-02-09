#!/bin/bash

username=$USER
userid=$UID

echo $username
echo $userid

echo ""
echo "Building image foxy_sogm_image for user hth"
echo ""

docker image build --build-arg username0=$username \
--build-arg userid0=$userid \
--shm-size=64g -t \
foxy_sogm_$username .