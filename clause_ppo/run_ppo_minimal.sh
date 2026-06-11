#!/bin/bash

# PPO training with minimal reward complexity for testing deterministic prefix fix
# Uses reduced reward complexity focusing on execution correctness

echo "🚀 Starting PPO training with minimal reward configuration..."
echo "Features:"
echo "  ✅ Deterministic numbered prefix stripping"
echo "  ✅ Fixed SFT adapter loading"
echo "  ✅ Reduced reward complexity (execution focus)"
echo "  ✅ More frequent logging for monitoring"

# Check if Spider data exists
if [ ! -f "clause_ppo/data/spider/tables.json" ]; then
    echo "❌ Error: Spider dataset not found at clause_ppo/data/spider/"
    echo "Expected files: clause_ppo/data/spider/tables.json, clause_ppo/data/spider/train_spider.json"
    exit 1
fi

# Check if SFT checkpoint exists
if [ ! -d "clause_ppo/results/sft_checkpoints/final" ]; then
    echo "❌ Error: SFT checkpoint not found at clause_ppo/results/sft_checkpoints/final"
    echo "Run SFT pretraining first with: ./clause_ppo/run_pretrain.sh"
    exit 1
fi

# Check if PRM checkpoint exists
if [ ! -d "clause_ppo/results/prm_checkpoints/epoch_2" ]; then
    echo "❌ Error: PRM checkpoint not found at clause_ppo/results/prm_checkpoints/epoch_2"
    echo "Run PRM training first"
    exit 1
fi

# Create output directories
mkdir -p clause_ppo/results/ppo_minimal_checkpoints
mkdir -p clause_ppo/results

echo "🔧 Configuration: clause_ppo/configs/ppo_minimal_reward_config.yaml"
echo "📁 Output: clause_ppo/results/ppo_minimal_checkpoints/"
echo "📄 Logs: clause_ppo/results/ppo_minimal_training_log.jsonl"
echo ""

# Start training
python -m clause_ppo.scripts.train_ppo \
    --config clause_ppo/configs/ppo_minimal_reward_config.yaml \
    --spider_dir clause_ppo/data/spider \
    --prm_ckpt clause_ppo/results/prm_checkpoints/epoch_2

echo "🏁 PPO minimal training completed!"