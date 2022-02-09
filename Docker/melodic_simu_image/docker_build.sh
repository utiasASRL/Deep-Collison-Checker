#!/bin/bash

username=$USER
userid=$UID

echo $username
echo $userid

echo ""
echo "Building image melodic_simu_image"
echo ""

docker image build --build-arg username0=$username \
--build-arg userid0=$userid \
--shm-size=64g -t \
melodic_simu_$username .