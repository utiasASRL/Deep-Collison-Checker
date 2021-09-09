#!/bin/bash

username=$USER
userid=$UID

echo $username
echo $userid

echo ""
echo "Building image noetic_pytorch"
echo ""

docker image build --build-arg username0=$username \
--build-arg userid0=$userid \
--shm-size=64g -t \
noetic_pytorch_$username .