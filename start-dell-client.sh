#!/bin/sh
sudo docker container stop dell-client-65b
sudo docker container remove dell-client-65b
sudo docker create -p 9090:9090 --ipc host --gpus 1 --volume petals-cache3:/root/.cache --name dell-client-65b hive-chat-dell-65b:latest
sudo docker container start dell-client-65b
