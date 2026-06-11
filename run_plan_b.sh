#!/bin/bash

# Plan B Evaluation Script
# Usage: ./run_plan_b.sh [limit] [start_idx] [database_name]
# Examples:
#   ./run_plan_b.sh 5                    # Test 5 samples from default start (car_1)
#   ./run_plan_b.sh 10 87 car_1          # Test 10 car_1 samples starting from index 87
#   ./run_plan_b.sh 20 0 concert_singer  # Test 20 concert_singer samples starting from index 0

set -e  # Exit on any error

# Default parameters
LIMIT=${1:-5}
START_IDX=${2:-87}
DB_NAME=${3:-"car_1"}

echo "🎯 Running Plan B evaluation..."
echo "   Database: $DB_NAME"
echo "   Samples: $LIMIT (starting from index $START_IDX)"
echo "   Config: clause_ppo/configs/eval_qwen_config.yaml"
echo ""

# Activate conda environment
source ~/.bashrc
if ! conda activate plan_b_clean 2>/dev/null; then
    echo "❌ Failed to activate conda environment 'plan_b_clean'"
    echo "   Please run: conda create -n plan_b_clean python=3.11"
    exit 1
fi

echo "✅ Activated conda environment: plan_b_clean"

# Check if required files exist
if [ ! -f "clause_ppo/configs/eval_qwen_config.yaml" ]; then
    echo "❌ Config file not found: clause_ppo/configs/eval_qwen_config.yaml"
    exit 1
fi

if [ ! -d "clause_ppo/results/prm_checkpoints/best_checkpoint" ]; then
    echo "❌ PRM checkpoint not found: clause_ppo/results/prm_checkpoints/best_checkpoint"
    exit 1
fi

if [ ! -f "clause_ppo/data/spider/dev.json" ]; then
    echo "❌ Spider data not found: clause_ppo/data/spider/dev.json"
    exit 1
fi

# Update start index in the evaluation script
echo "🔧 Setting start index to $START_IDX for $DB_NAME database..."
python3 -c "
import re

# Read the file
with open('src/eval/best_of_n.py', 'r') as f:
    content = f.read()

# Update the start_idx
content = re.sub(
    r'start_idx = \d+  # .*',
    f'start_idx = $START_IDX  # Test $DB_NAME samples',
    content
)

# Write back
with open('src/eval/best_of_n.py', 'w') as f:
    f.write(content)

print(f'✅ Updated start_idx to $START_IDX for $DB_NAME database')
"

# Run the evaluation
echo "🚀 Starting Plan B evaluation..."
echo ""

python3 -c "
import sys
sys.path.insert(0, 'src')
from eval.best_of_n import eval_best_of_n
import yaml
import os

try:
    # Load config
    config_path = 'clause_ppo/configs/eval_qwen_config.yaml'
    with open(config_path) as f:
        config = yaml.safe_load(f)

    # Run evaluation
    spider_dir = 'clause_ppo/data/spider'
    prm_ckpt = 'clause_ppo/results/prm_checkpoints/best_checkpoint'
    
    print(f'📊 Testing $LIMIT samples from $DB_NAME database...')
    summary = eval_best_of_n(config, spider_dir, prm_ckpt, limit=$LIMIT)
    
    print('')
    print('🎉 Plan B evaluation completed successfully!')
    print(f'📁 Results saved to: results/eval_qwen_test.jsonl')
    
except Exception as e:
    print(f'❌ Evaluation failed: {e}')
    exit(1)
"

echo ""
echo "✅ Plan B evaluation script completed!"
echo ""
echo "📋 Database options for future runs:"
echo "   concert_singer: ./run_plan_b.sh 10 0 concert_singer"
echo "   car_1:         ./run_plan_b.sh 10 87 car_1" 
echo "   flight_2:      ./run_plan_b.sh 10 179 flight_2"
echo "   pets_1:        ./run_plan_b.sh 10 45 pets_1"
echo "   world_1:       ./run_plan_b.sh 10 702 world_1"