#!/usr/bin/env bash
# Make sure to run: chmod +x train.sh
set -e

CONFIG=configs/deberta_large_sst2.yaml

# Check if CONFIG file exists
if [ ! -f "$CONFIG" ]; then
    echo "Error: Config file not found at $CONFIG"
    exit 1
fi

# Check if train_teacher.py script exists
if [ ! -f "src/train_teacher.py" ]; then
    echo "Error: Training script not found at src/train_teacher.py"
    exit 1
fi

echo "Starting training using config: $CONFIG"

torchrun --nnodes 1 --nproc_per_node 4 --master_port 12345 \
         src/train_teacher.py $CONFIG

echo "Training script finished." 