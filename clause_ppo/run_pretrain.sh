#!/bin/bash

# SFT Pretraining: Teach Qwen-7B clean SQL generation before PPO
# Uses train_spider[:4000] (non-overlapping with PPO training)

set -e  # Exit on error

echo "🚀 Starting SFT pretraining on Spider data..."
echo "📊 Using train_spider[:4000] (non-overlapping with PPO)"

# Activate environment (modify as needed)
if [[ "$CONDA_DEFAULT_ENV" != "ppo_training" ]]; then
    echo "⚠️  Activating ppo_training environment..."
    source ~/miniconda3/etc/profile.d/conda.sh
    conda activate ppo_training
fi

# Run SFT pretraining
python scripts/pretrain_sft.py \
    --spider_dir data/spider \
    --model_name /home/henrylin0822/models/qwen \
    --output_dir results/sft_checkpoints \
    --max_samples 4000

echo "✅ SFT pretraining complete!"
echo "📁 Model saved to: results/sft_checkpoints/final"
echo ""
echo "🔄 Next steps:"
echo "  1. Update ppo_config.yaml to use SFT checkpoint:"
echo "     actor_name: results/sft_checkpoints/final"  
echo "  2. Run PPO training with SQL-focused model"