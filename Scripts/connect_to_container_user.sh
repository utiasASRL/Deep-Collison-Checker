#!/bin/bash

echo ""
echo "Starting new shell in the docker container $@"
echo ""

docker container exec -it $@ /bin/bash

