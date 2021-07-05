#!/usr/bin/env bash

if [[ "$1" == "build" ]]
then
    docker build --build-arg USER_ID=$(id -u) --build-arg GROUP_ID=$(id -g) --build-arg USER_NAME=$(whoami) -t graphsaint .
fi

if [[ "$1" == "run" ]]
then
    docker run -it -u $(id -u):$(id -g) -v `pwd`:/data -w /data --runtime=nvidia -e NVIDIA_VISIBLE_DEVICES=all graphsaint
fi
