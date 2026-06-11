#!/bin/bash

# Three-Way Evaluation Script for NL2SQL Clause Repair
# Compares Baseline vs Plan B vs PPO approaches
# Simply edit the configuration below and run: ./run_three_way_evaluation.sh

set -e  # Exit on any error

# ── EDIT THESE SETTINGS ────────────────────────────────────────────────────
# Dataset settings
SPLIT="dev"                    # "dev" or "train" 
LIMIT="17"                      # Number of samples to test (leave empty "" for all)

# Model checkpoints
PLAN_B_CKPT="clause_ppo/results/prm_checkpoints/best_checkpoint"    # ClausePRM checkpoint
PPO_CKPT="results/ppo_minimal_checkpoints/ep_50"                # PPO checkpoint

# Inference settings
MAX_RETRIES="3"               # Max repair attempts per method

# ── DO NOT EDIT BELOW (unless you know what you're doing) ──────────────────

echo "🏆 Three-Way NL2SQL Evaluation"
echo "=================================="
echo "Split: $SPLIT"
echo "Limit: ${LIMIT:-"all samples"}"
echo "Plan B checkpoint: $PLAN_B_CKPT"
echo "PPO checkpoint: $PPO_CKPT"
echo "Max retries: $MAX_RETRIES"
echo ""

# ── Validation ────────────────────────────────────────────────────────────
if [ ! -d "$PLAN_B_CKPT" ]; then
    echo "❌ Plan B checkpoint not found: $PLAN_B_CKPT"
    echo ""
    echo "Available PRM checkpoints:"
    if [ -d "clause_ppo/results/prm_checkpoints" ]; then
        ls -la clause_ppo/results/prm_checkpoints/
    else
        echo "  No PRM checkpoints found."
    fi
    exit 1
fi

if [ ! -d "$PPO_CKPT" ]; then
    echo "❌ PPO checkpoint not found: $PPO_CKPT"
    echo ""
    echo "Available PPO checkpoints:"
    if [ -d "clause_ppo/results/ppo_checkpoints" ]; then
        ls -la clause_ppo/results/ppo_checkpoints/
    else
        echo "  No PPO checkpoints found."
    fi
    exit 1
fi

# ── Build evaluation command ──────────────────────────────────────────────
CMD="python scripts/evaluate.py --split $SPLIT --plan-b-ckpt \"$PLAN_B_CKPT\" --ppo-ckpt \"$PPO_CKPT\""

# Add optional parameters
if [ ! -z "$LIMIT" ]; then
    CMD="$CMD --max-samples $LIMIT"
fi

if [ ! -z "$MAX_RETRIES" ]; then
    CMD="$CMD --max-retries $MAX_RETRIES"
fi

echo "🚀 Starting three-way evaluation..."
echo "Command: $CMD"
echo ""

# ── Environment Setup ─────────────────────────────────────────────────────
# Use PPO training environment for compatibility
echo "🔧 Activating ppo_training environment..."
source ~/miniconda3/etc/profile.d/conda.sh
conda activate ppo_training

# ── Run evaluation ────────────────────────────────────────────────────────
eval $CMD

echo ""
echo "✅ Three-way evaluation completed!"
echo ""

# ── Interpretation Guide ──────────────────────────────────────────────────
echo "📊 Results Interpretation:"
echo "=========================="
echo "Method                | Description                        | Key Metric"
echo "----------------------|------------------------------------|-----------" 
echo "Baseline (Full Regen) | Standard NL2SQL generation        | Baseline accuracy"
echo "Plan B (ClausePRM)    | Best-of-N clause repair with PRM  | Accuracy vs tokens"
echo "PPO (RL-trained)      | Reinforcement learning repair      | RL improvement"
echo ""
echo "💡 Look for:"
echo "   • PPO accuracy > Baseline (RL learning success)"
echo "   • PPO tokens ≤ Plan B (efficiency gain)"
echo "   • Plan B accuracy ≥ Baseline (PRM guidance works)"