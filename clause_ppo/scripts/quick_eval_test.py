#!/usr/bin/env python3
"""
Quick test to verify the evaluation setup works on just 5 samples.
"""

import argparse
import json
import os
import sys
import time
from collections import defaultdict

# Ensure src/ is on the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from models.prm_inference import load_qwen_for_inference
from utils.sql_utils import extract_sql_from_text
from utils.execution import execute_query
import torch

def main():
    print("Loading fixed evaluation set...")
    with open('data/processed/fixed_eval_set.json') as f:
        eval_samples = json.load(f)[:5]  # Just 5 samples for quick test
    print(f"Loaded {len(eval_samples)} test samples")
    
    print("\nTesting one sample structure:")
    sample = eval_samples[0]
    print(f"  DB: {sample['db_id']}")
    print(f"  Corrupted clause: {sample['corrupted_clause']}")
    print(f"  Corrupted position: {sample['corrupted_position']}")
    
    # Test prompt building
    faulty_clause = sample['corrupted_clause']
    corrupted_position = sample['corrupted_position']
    prefix_state = sample['prefix_states'][corrupted_position]
    
    prompt = f"### Question: {prefix_state['question']}\n### Schema: {prefix_state['schema'][:200]}...\n### Partial SQL: {prefix_state['prefix_query_str']}\n### Complete the {faulty_clause} clause:\n"
    print(f"\nTest prompt (truncated):\n{prompt[:300]}...")
    
    # Test SQL execution
    db_path = 'data/spider/database'
    db_file_path = os.path.join(db_path, sample['db_id'], f"{sample['db_id']}.sqlite")
    
    print(f"\nTesting SQL execution on {db_file_path}")
    if os.path.exists(db_file_path):
        print("✅ Database file exists")
        try:
            ok, result = execute_query("SELECT 1", db_file_path, timeout_secs=2.0)
            if ok:
                print(f"✅ Database connection works: {result}")
            else:
                print("❌ Database connection failed")
        except Exception as e:
            print(f"❌ Database error: {e}")
    else:
        print(f"❌ Database file not found: {db_file_path}")
    
    # Test original query
    try:
        print(f"\nTesting original query: {sample['original_query'][:100]}...")
        ok, result = execute_query(sample['original_query'], db_file_path, timeout_secs=5.0)
        if ok:
            print(f"✅ Original query executes: {len(result)} rows")
        else:
            print("❌ Original query failed")
    except Exception as e:
        print(f"❌ Original query error: {e}")
    
    print("\n✅ Setup verification complete. You can proceed with full evaluation.")
    print("\nTo run full evaluation:")
    print("python scripts/fixed_eval.py --sft_checkpoint results/sft_checkpoints/merged_checkpoint --ppo_checkpoint results/ppo_checkpoints/ep_200 --best_of_n 1")

if __name__ == '__main__':
    main()