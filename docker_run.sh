#!/bin/bash

# 定义常量
IMAGE_NAME="misheng-app"
CONTAINER_NAME="misheng-c"

echo "Stopping and removing existing container..."
docker rm -f $CONTAINER_NAME || true

echo "Starting $IMAGE_NAME..."
docker run -d \
  --name $CONTAINER_NAME \
  -p 8000:8000 \
  --env-file .env \
  --restart unless-stopped \
  $IMAGE_NAME

echo "Container is running!"
docker ps -f name=$CONTAINER_NAME
