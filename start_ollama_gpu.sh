#!/bin/bash

# Configuration
NODE="gpu123"
GPU_ID=${1:-7} # Default to GPU 7 if no argument is provided

echo "Stopping any existing Ollama instances on $NODE..."
ssh $NODE "pkill -f 'ollama serve' || true" # || true prevents script from exiting if no process is found

echo "Starting Ollama on $NODE using GPU $GPU_ID (gfx942 architecture)..."
echo "Press Ctrl+C to stop the server."

# SSH into the node and run ollama serve with the necessary environment variables
ssh $NODE "HSA_OVERRIDE_GFX_VERSION=9.4.2 ROCR_VISIBLE_DEVICES=$GPU_ID ollama serve"
