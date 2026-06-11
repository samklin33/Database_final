#!/bin/bash

# PRM Training with Clean Environment
echo "🚀 Starting PRM Training with sophisticated rewards..."

# Activate the clean environment
source /home/henrylin0822/miniconda3/bin/activate prm_training_clean

# Check dataset
echo "✅ Found sophisticated dataset with $(python -c "import json; print(len(json.load(open('data/processed/corruption_dataset_with_rewards.json'))))" 2>/dev/null || echo "?") records"

# Run training
echo "🏃 Starting PRM training..."
python train_prm_simplified.py

echo "🎉 PRM Training completed!"
echo "📁 Checkpoints saved to: results/prm_checkpoints/"