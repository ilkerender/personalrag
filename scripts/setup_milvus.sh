#!/bin/bash

# Script to set up and run Milvus Standalone using Docker

set -e

echo "Downloading Milvus installation script..."
curl -sfL https://raw.githubusercontent.com/milvus-io/milvus/master/scripts/standalone_embed.sh -o standalone_embed.sh

chmod +x ./standalone_embed.sh

echo "Starting Milvus Docker container..."
bash standalone_embed.sh start

echo "Milvus setup complete. Running at localhost:19530"

