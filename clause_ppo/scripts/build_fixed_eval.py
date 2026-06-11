#!/usr/bin/env python3
"""
build_fixed_eval.py
===================
Creates a fixed held-out evaluation set of ~200 corrupted samples from dev split.
This provides a stable measurement baseline to compare SFT vs PPO performance.

Usage:
    python scripts/build_fixed_eval.py \
        --spider_dir data/spider \
        --output_path data/processed/fixed_eval_set.json \
        --target_size 200
"""

import argparse
import json
import os
import sys
import random
from typing import List, Dict

# Ensure src/ is on the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from data.clause_splitter import split_into_clauses, clauses_to_prefix_states, schema_to_string
from data.corruption import generate_corruptions


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--spider_dir', default='data/spider')
    p.add_argument('--output_path', default='data/processed/fixed_eval_set.json')
    p.add_argument('--target_size', type=int, default=200)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--min_clauses', type=int, default=2)
    return p.parse_args()


def main():
    args = parse_args()
    random.seed(args.seed)
    
    # Load dev split (held-out data)
    dev_path = os.path.join(args.spider_dir, 'dev.json')
    tables_path = os.path.join(args.spider_dir, 'tables.json')
    
    print(f"Loading {dev_path} ...")
    with open(dev_path) as f:
        examples = json.load(f)
    
    print(f"Loading {tables_path} ...")
    with open(tables_path) as f:
        tables_dict = {t['db_id']: t for t in json.load(f)}
    
    print(f"Processing {len(examples)} dev examples to create fixed eval set...")
    
    # Collect corruption records
    all_corruptions = []
    
    for ex in examples:
        clauses = split_into_clauses(ex['sql'])
        if len(clauses) < args.min_clauses:
            continue
            
        corruptions = generate_corruptions(ex, tables_dict)
        if not corruptions:
            continue
            
        # Add schema and prefix states for evaluation
        schema = schema_to_string(ex['db_id'], tables_dict)
        prefix_states = clauses_to_prefix_states(
            ex['question'], schema, clauses, ex['query']
        )
        
        for c in corruptions:
            c['schema'] = schema
            c['prefix_states'] = prefix_states
            all_corruptions.append(c)
    
    print(f"Generated {len(all_corruptions)} total corruptions from dev split")
    
    # Sample target_size corruptions
    if len(all_corruptions) > args.target_size:
        selected_corruptions = random.sample(all_corruptions, args.target_size)
        print(f"Randomly sampled {args.target_size} corruptions")
    else:
        selected_corruptions = all_corruptions
        print(f"Using all {len(all_corruptions)} corruptions (less than target)")
    
    # Save the fixed evaluation set
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    with open(args.output_path, 'w') as f:
        json.dump(selected_corruptions, f, indent=2)
    
    size_mb = os.path.getsize(args.output_path) / 1024 / 1024
    print(f"✅ Saved fixed evaluation set: {args.output_path}")
    print(f"   {len(selected_corruptions)} samples, {size_mb:.1f} MB")
    
    # Print clause distribution
    clause_counts = {}
    for c in selected_corruptions:
        clause = c['corrupted_clause']
        clause_counts[clause] = clause_counts.get(clause, 0) + 1
    
    print(f"\nClause distribution in eval set:")
    for clause, count in sorted(clause_counts.items()):
        print(f"  {clause:<12}: {count:>3} samples ({count/len(selected_corruptions)*100:.1f}%)")


if __name__ == '__main__':
    main()