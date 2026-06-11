#!/usr/bin/env python3
"""
apply_clause_rewards.py
========================
Apply sophisticated clause-level reward scoring to corruption_dataset.json
using clause_rewards.py scoring system.

This creates the "sophisticated dataset" with nuanced clause scores instead
of binary 0/1 scoring.

Usage:
  python scripts/apply_clause_rewards.py \
      --corruption_file data/processed/corruption_dataset.json \
      --original_file data/processed/original_dataset.json \
      --spider_dir data/spider \
      --output_file data/processed/corruption_dataset_with_rewards.json
"""

import argparse
import json
import os
import sys
from tqdm import tqdm

# Add src/ to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from clause_rewards import compute_clause_rewards, build_orig_clause_texts, validate_rewards


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--corruption_file', default='data/processed/corruption_dataset.json')
    p.add_argument('--original_file', default='data/processed/original_dataset.json')
    p.add_argument('--spider_dir', default='data/spider')
    p.add_argument('--output_file', default='data/processed/corruption_dataset_with_rewards.json')
    p.add_argument('--log_every', type=int, default=1000)
    return p.parse_args()


def main():
    args = parse_args()
    
    print(f"Loading corruption dataset: {args.corruption_file}")
    with open(args.corruption_file) as f:
        corruption_records = json.load(f)
    
    print(f"Loading original dataset: {args.original_file}")
    with open(args.original_file) as f:
        original_records = json.load(f)
    
    # Build lookup from (db_id, question) -> clause_names for fast access
    print("Building original dataset lookup...")
    original_lookup = {}
    for orig in original_records:
        key = (orig['db_id'], orig['question'])
        original_lookup[key] = orig['clause_names']
    
    print(f"Processing {len(corruption_records):,} corruption records with sophisticated scoring...")
    
    updated_records = []
    warnings_count = 0
    
    for i, record in enumerate(tqdm(corruption_records, desc="Applying rewards")):
        # Get clause names from original dataset
        key = (record['db_id'], record['question'])
        if key not in original_lookup:
            print(f"WARNING: Missing original record for {key}")
            continue
            
        clause_names = original_lookup[key]
        orig_clause_texts = build_orig_clause_texts(record['original_query'], clause_names)
        
        # Build database path
        db_path = os.path.join(args.spider_dir, 'database', record['db_id'], f"{record['db_id']}.sqlite")
        
        # Compute sophisticated rewards
        try:
            rewards = compute_clause_rewards(
                record=record,
                clause_names=clause_names,
                orig_clause_texts=orig_clause_texts,
                db_path=db_path if os.path.exists(db_path) else ""
            )
            
            # Validate rewards
            validation_warnings = validate_rewards(rewards, clause_names, record['corrupted_clause'])
            if validation_warnings:
                warnings_count += 1
                if warnings_count <= 5:  # Log first few warnings
                    print(f"Validation warnings for record {i}: {validation_warnings}")
            
            # Add rewards to record
            updated_record = record.copy()
            updated_record['reward'] = rewards
            updated_record['clause_names'] = clause_names
            updated_records.append(updated_record)
            
        except Exception as e:
            print(f"Error processing record {i}: {e}")
            continue
        
        if (i + 1) % args.log_every == 0:
            print(f"  [{i+1:>6}/{len(corruption_records)}]  rewards computed: {len(updated_records):,}")
    
    print(f"\nSaving enhanced dataset with sophisticated rewards...")
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    with open(args.output_file, 'w') as f:
        json.dump(updated_records, f, indent=2)
    
    size_mb = os.path.getsize(args.output_file) / 1024 / 1024
    print(f"Saved {args.output_file} ({size_mb:.1f} MB, {len(updated_records):,} records)")
    
    if warnings_count > 5:
        print(f"Total validation warnings: {warnings_count}")
    
    print("\n── Sample Enhanced Records ──")
    for i in range(min(3, len(updated_records))):
        record = updated_records[i]
        print(f"Record {i+1}:")
        print(f"  Corrupted clause: {record['corrupted_clause']}")
        print(f"  Clause names: {record['clause_names']}")
        print(f"  Rewards: {record['reward']}")
        print()


if __name__ == '__main__':
    main()