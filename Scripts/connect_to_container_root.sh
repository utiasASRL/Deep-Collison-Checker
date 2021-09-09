#!/bin/bash

echo ""
echo "Starting new root shell in the docker container $@"
echo ""

docker exec -u 0 -it $@ bash