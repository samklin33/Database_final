#!/bin/bash

# PPO Training Script for CLAUSE-PPO Phase 2
# Usage: ./scripts/run_ppo_training.sh [config_file]

set -e  # Exit on any error

# Use provided config or default
CONFIG_FILE=${1:-configs/ppo_config.yaml}

echo "Starting PPO training..."
echo "Configuration: $CONFIG_FILE"
echo "Spider dataset: data/spider"
echo "PRM checkpoint: results/prm_checkpoints/best_checkpoint"
echo ""

# Clean Python path to avoid conflicts with user-installed packages
export PYTHONPATH=""
unset PYTHONUSERBASE

# Use ppo_training environment with TRL, PEFT, and Qwen model support
/home/henrylin0822/miniconda3/envs/ppo_training/bin/python scripts/train_ppo.py \
    --config "$CONFIG_FILE" \
    --spider_dir data/spider \
    --prm_ckpt results/prm_checkpoints/best_checkpoint

echo "PPO training completed."