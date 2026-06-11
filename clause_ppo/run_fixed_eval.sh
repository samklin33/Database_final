#!/bin/bash

# Fixed evaluation script for SFT vs PPO comparison
# Run this from the clause_ppo/ directory

set -e

echo "🔄 Activating environment..."
source ~/miniconda3/etc/profile.d/conda.sh
conda activate ppo_training

echo "📊 Starting fixed evaluation..."

# Quick test first (20 samples)
echo "🧪 Running quick test (20 samples)..."
python scripts/memory_efficient_eval.py \
    --sft_checkpoint results/sft_checkpoints/merged_checkpoint \
    --ppo_checkpoint results/ppo_checkpoints/ep_200 \
    --best_of_n 1 \
    --quick_test 20

echo ""
echo "✅ Quick test complete!"
echo ""
read -p "Continue with full evaluation (200 samples)? [y/N]: " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "🚀 Running full evaluation (200 samples)..."
    python scripts/memory_efficient_eval.py \
        --sft_checkpoint results/sft_checkpoints/merged_checkpoint \
        --ppo_checkpoint results/ppo_checkpoints/ep_200 \
        --best_of_n 1
    
    echo ""
    echo "📈 Full evaluation complete!"
    echo ""
    read -p "Run best-of-8 headroom test (50 samples)? [y/N]: " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "🎯 Running best-of-8 headroom test..."
        python scripts/memory_efficient_eval.py \
            --sft_checkpoint results/sft_checkpoints/merged_checkpoint \
            --ppo_checkpoint results/ppo_checkpoints/ep_200 \
            --best_of_n 8 \
            --quick_test 50
        echo "🏁 Best-of-8 test complete!"
    fi
else
    echo "Evaluation stopped. Results from quick test available."
fi

echo ""
echo "📁 Check results/ directory for detailed JSON files."
echo "🎉 Evaluation pipeline complete!"