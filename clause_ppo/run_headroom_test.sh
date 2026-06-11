#!/bin/bash

# Best-of-N headroom analysis for the expert's evaluation framework

set -e

echo "🔄 Activating environment..."
source ~/miniconda3/etc/profile.d/conda.sh
conda activate ppo_training

echo "📈 Starting best-of-N headroom analysis..."
echo "This tests if the SFT model can produce correct repairs when sampling multiple times."
echo ""

# Run the headroom test
python scripts/best_of_n_eval.py \
    --sft_checkpoint results/sft_checkpoints/merged_checkpoint \
    --max_samples 50

echo ""
echo "🎯 Headroom analysis complete!"
echo ""
echo "Key question: Does best-of-8 >> single-sample accuracy?"
echo "- If YES → Model can repair but needs better selection → Try ReST/RAFT"
echo "- If NO  → Model is at capability ceiling → Focus on data/model improvements"
echo ""
echo "📁 Check results/best_of_n_headroom_analysis.json for detailed data"