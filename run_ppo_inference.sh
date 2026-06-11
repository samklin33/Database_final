#!/bin/bash

# PPO Inference Script for CLAUSE-PPO
# Simply edit the configuration below and run: ./run_ppo_inference.sh

set -e  # Exit on any error

# ── EDIT THESE SETTINGS ────────────────────────────────────────────────────
# PPO model checkpoint to use for inference
PPO_CKPT="clause_ppo/results/ppo_checkpoints/ep_400"

# Spider dataset settings
SPIDER_DIR="clause_ppo/data/spider"
SPLIT="dev"                    # "dev" or "train"

# Inference settings
LIMIT="10"                     # Number of samples to test (leave empty "" for all)
MAX_RETRIES="3"               # Max repair attempts per sample

# ── DO NOT EDIT BELOW (unless you know what you're doing) ──────────────────

echo "🤖 PPO Inference - RL-trained Clause Repair"
echo "=============================================="
echo "PPO checkpoint: $PPO_CKPT"
echo "Spider dataset: $SPIDER_DIR"
echo "Split: $SPLIT"
echo "Limit: ${LIMIT:-"unlimited"}"
echo "Max retries: $MAX_RETRIES"
echo ""

# ── Validation ────────────────────────────────────────────────────────────
if [ ! -d "$PPO_CKPT" ]; then
    echo "❌ PPO checkpoint not found: $PPO_CKPT"
    echo ""
    echo "Available PPO checkpoints:"
    if [ -d "clause_ppo/results/ppo_checkpoints" ]; then
        ls -la clause_ppo/results/ppo_checkpoints/
    else
        echo "  No PPO checkpoints found."
        echo "  Run PPO training first: ./clause_ppo/scripts/run_ppo_training.sh"
    fi
    exit 1
fi

if [ ! -f "$SPIDER_DIR/$SPLIT.json" ]; then
    echo "❌ Spider split not found: $SPIDER_DIR/$SPLIT.json"
    echo ""
    echo "Available splits:"
    ls -la $SPIDER_DIR/*.json 2>/dev/null || echo "  No Spider data found."
    exit 1
fi

# ── Environment Setup ─────────────────────────────────────────────────────
# Use PPO training environment (same as training)
PYTHON_CMD="/home/henrylin0822/miniconda3/envs/ppo_training/bin/python"

if [ ! -f "$PYTHON_CMD" ]; then
    echo "❌ PPO training environment not found: $PYTHON_CMD"
    echo "Please activate the ppo_training conda environment or update the script"
    exit 1
fi

# Clean Python path to avoid conflicts
export PYTHONPATH=""
unset PYTHONUSERBASE

# ── Run PPO Inference ─────────────────────────────────────────────────────
echo "🚀 Starting PPO inference..."
echo "Using environment: ppo_training"

# Build command
CMD="$PYTHON_CMD src/baseline/ppo_inference.py \
    --ppo-ckpt \"$PPO_CKPT\" \
    --spider-dir \"$SPIDER_DIR\" \
    --split \"$SPLIT\" \
    --max-retries \"$MAX_RETRIES\""

# Add limit if specified
if [ ! -z "$LIMIT" ]; then
    CMD="$CMD --limit \"$LIMIT\""
fi

echo "Command: $CMD"
echo ""

# Execute
eval $CMD

echo ""
echo "✅ PPO inference completed!"
echo ""

# ── Quick Stats ───────────────────────────────────────────────────────────
echo "📊 Quick comparison with other approaches:"
echo ""
echo "Approach              | Models                      | Inference"
echo "----------------------|-----------------------------|-----------" 
echo "Baseline (Full Regen) | Generator only              | Multiple attempts"
echo "Plan B (ClausePRM)    | Generator + PRM (separate)  | Score + repair"
echo "PPO (This script)     | Actor only (RL-trained)    | Direct repair"
echo ""
echo "💡 PPO actor has learned clause repair through RL training with PRM feedback"
echo "   No separate PRM needed at inference time!"