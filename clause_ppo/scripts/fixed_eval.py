#!/usr/bin/env python3
"""
fixed_eval.py
=============
Evaluates models on the fixed held-out evaluation set to measure true learning.
Compares SFT baseline (ep0) vs PPO checkpoint (ep200) on identical samples.

Usage:
    python scripts/fixed_eval.py \
        --eval_path data/processed/fixed_eval_set.json \
        --sft_checkpoint results/sft_checkpoints/merged_checkpoint \
        --ppo_checkpoint results/ppo_checkpoints/ep_200 \
        --spider_dir data/spider \
        --best_of_n 1
"""

import argparse
import json
import os
import sys
import time
from collections import defaultdict
from typing import List, Dict, Tuple
import torch

# Ensure src/ is on the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from models.prm_inference import load_qwen_for_inference
from utils.sql_utils import extract_sql_from_text
from utils.execution import execute_query


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--eval_path', default='data/processed/fixed_eval_set.json')
    p.add_argument('--sft_checkpoint', default='results/sft_checkpoints/merged_checkpoint')
    p.add_argument('--ppo_checkpoint', default='results/ppo_checkpoints/ep_200') 
    p.add_argument('--spider_dir', default='data/spider')
    p.add_argument('--best_of_n', type=int, default=1, help='Sample N times, keep best execution result')
    p.add_argument('--temperature', type=float, default=0.8)
    p.add_argument('--max_length', type=int, default=512)
    return p.parse_args()


def evaluate_model(model, tokenizer, eval_samples: List[Dict], args) -> Dict:
    """Evaluate a single model on the fixed evaluation set."""
    results = {
        'correct': 0,
        'executable': 0,
        'total': len(eval_samples),
        'per_clause': defaultdict(lambda: {'correct': 0, 'executable': 0, 'total': 0}),
        'sample_results': []
    }
    
    db_path = os.path.join(args.spider_dir, 'database')
    
    for i, sample in enumerate(eval_samples):
        if i % 50 == 0:
            print(f"  Evaluating sample {i+1}/{len(eval_samples)}")
            
        # Build prompt from corruption sample
        faulty_clause = sample['corrupted_clause']
        corrupted_position = sample['corrupted_position']
        prefix_state = sample['prefix_states'][corrupted_position]
        
        prompt = f"### Question: {prefix_state['question']}\n### Schema: {prefix_state['schema']}\n### Partial SQL: {prefix_state['prefix_query_str']}\n### Complete the {faulty_clause} clause:\n"
        
        best_correct = False
        best_executable = False
        generations = []
        
        # Sample best_of_n times
        for _ in range(args.best_of_n):
            # Generate completion
            inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024)
            
            # Move inputs to same device as model
            model_device = next(model.parameters()).device
            inputs = {k: v.to(model_device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = model.generate(
                    inputs['input_ids'],
                    attention_mask=inputs['attention_mask'],
                    max_length=inputs['input_ids'].shape[1] + args.max_length,
                    temperature=args.temperature,
                    do_sample=args.temperature > 0,
                    pad_token_id=tokenizer.eos_token_id,
                    eos_token_id=tokenizer.eos_token_id
                )
            
            # Extract completion (remove prompt)
            completion = tokenizer.decode(
                outputs[0][inputs['input_ids'].shape[1]:], 
                skip_special_tokens=True
            ).strip()
            
            # Extract SQL and reconstruct full query
            extracted_sql = extract_sql_from_text(completion)
            if not extracted_sql:
                extracted_sql = completion.strip()
                
            reconstructed_sql = prefix_state['prefix_query_str'] + " " + extracted_sql
            generations.append({
                'completion': completion,
                'extracted_sql': extracted_sql, 
                'reconstructed_sql': reconstructed_sql
            })
            
            # Test execution
            is_executable = False
            is_correct = False
            
            # Build full database path
            db_file_path = os.path.join(db_path, sample['db_id'], f"{sample['db_id']}.sqlite")
            
            try:
                ok_recon, result = execute_query(reconstructed_sql, db_file_path, timeout_secs=5.0)
                if ok_recon:
                    is_executable = True
                    
                    # Check if matches gold result
                    ok_gold, gold_result = execute_query(sample['original_query'], db_file_path, timeout_secs=5.0)
                    if ok_gold and result == gold_result:
                        is_correct = True
                        
            except Exception:
                pass
            
            # Update best results
            if is_correct:
                best_correct = True
            if is_executable:
                best_executable = True
                
            # If we found a correct answer, no need to sample more
            if best_correct and args.best_of_n > 1:
                break
        
        # Update results with best outcome
        if best_correct:
            results['correct'] += 1
            results['per_clause'][faulty_clause]['correct'] += 1
        if best_executable:
            results['executable'] += 1
            results['per_clause'][faulty_clause]['executable'] += 1
            
        results['per_clause'][faulty_clause]['total'] += 1
        
        # Store sample result
        results['sample_results'].append({
            'sample_id': i,
            'clause': faulty_clause,
            'correct': best_correct,
            'executable': best_executable,
            'generations': generations
        })
    
    # Calculate final metrics
    results['accuracy'] = results['correct'] / results['total']
    results['execution_rate'] = results['executable'] / results['total']
    
    # Per-clause metrics
    for clause in results['per_clause']:
        clause_stats = results['per_clause'][clause]
        clause_stats['accuracy'] = clause_stats['correct'] / clause_stats['total']
        clause_stats['execution_rate'] = clause_stats['executable'] / clause_stats['total']
    
    return results


def main():
    args = parse_args()
    
    # Load evaluation set
    print(f"Loading evaluation set from {args.eval_path}")
    with open(args.eval_path) as f:
        eval_samples = json.load(f)
    print(f"Loaded {len(eval_samples)} evaluation samples")
    
    # Load models
    print(f"\nLoading SFT model from {args.sft_checkpoint}")
    sft_model, sft_tokenizer = load_qwen_for_inference(args.sft_checkpoint)
    
    print(f"Loading PPO model from {args.ppo_checkpoint}")  
    ppo_model, ppo_tokenizer = load_qwen_for_inference(args.ppo_checkpoint)
    
    sampling_info = f"best-of-{args.best_of_n}" if args.best_of_n > 1 else "single-sample"
    print(f"\nEvaluating models ({sampling_info}, temp={args.temperature})...")
    
    # Evaluate SFT baseline
    print(f"\n=== Evaluating SFT Baseline ===")
    sft_start = time.time()
    sft_results = evaluate_model(sft_model, sft_tokenizer, eval_samples, args)
    sft_time = time.time() - sft_start
    
    # Evaluate PPO checkpoint
    print(f"\n=== Evaluating PPO Checkpoint ===")
    ppo_start = time.time()
    ppo_results = evaluate_model(ppo_model, ppo_tokenizer, eval_samples, args)
    ppo_time = time.time() - ppo_start
    
    # Print comparison results
    print(f"\n" + "="*60)
    print(f"FIXED EVALUATION RESULTS ({sampling_info})")
    print(f"="*60)
    print(f"Evaluation set: {len(eval_samples)} samples")
    print(f"")
    print(f"{'Model':<15} {'Accuracy':<10} {'Exec Rate':<10} {'Time (s)':<10}")
    print(f"{'-'*45}")
    print(f"{'SFT Baseline':<15} {sft_results['accuracy']:<10.3f} {sft_results['execution_rate']:<10.3f} {sft_time:<10.1f}")
    print(f"{'PPO ep200':<15} {ppo_results['accuracy']:<10.3f} {ppo_results['execution_rate']:<10.3f} {ppo_time:<10.1f}")
    print(f"")
    
    # Calculate delta
    accuracy_delta = ppo_results['accuracy'] - sft_results['accuracy']
    exec_delta = ppo_results['execution_rate'] - sft_results['execution_rate']
    print(f"{'Delta (PPO-SFT)':<15} {accuracy_delta:<+10.3f} {exec_delta:<+10.3f}")
    
    # Per-clause breakdown
    print(f"\nPer-clause results:")
    print(f"{'Clause':<12} {'SFT Acc':<8} {'PPO Acc':<8} {'Delta':<8} {'Samples':<8}")
    print(f"{'-'*50}")
    
    all_clauses = set(sft_results['per_clause'].keys()) | set(ppo_results['per_clause'].keys())
    for clause in sorted(all_clauses):
        sft_acc = sft_results['per_clause'][clause]['accuracy'] if clause in sft_results['per_clause'] else 0
        ppo_acc = ppo_results['per_clause'][clause]['accuracy'] if clause in ppo_results['per_clause'] else 0
        delta = ppo_acc - sft_acc
        samples = sft_results['per_clause'][clause]['total'] if clause in sft_results['per_clause'] else 0
        
        print(f"{clause:<12} {sft_acc:<8.3f} {ppo_acc:<8.3f} {delta:<+8.3f} {samples:<8}")
    
    # Save detailed results
    output_path = 'results/fixed_eval_results.json'
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    detailed_results = {
        'config': {
            'eval_path': args.eval_path,
            'sft_checkpoint': args.sft_checkpoint,
            'ppo_checkpoint': args.ppo_checkpoint,
            'best_of_n': args.best_of_n,
            'temperature': args.temperature,
            'eval_samples': len(eval_samples)
        },
        'summary': {
            'sft': {
                'accuracy': sft_results['accuracy'],
                'execution_rate': sft_results['execution_rate'],
                'eval_time': sft_time
            },
            'ppo': {
                'accuracy': ppo_results['accuracy'], 
                'execution_rate': ppo_results['execution_rate'],
                'eval_time': ppo_time
            },
            'delta': {
                'accuracy': accuracy_delta,
                'execution_rate': exec_delta
            }
        },
        'per_clause': {
            clause: {
                'sft_accuracy': sft_results['per_clause'][clause]['accuracy'],
                'ppo_accuracy': ppo_results['per_clause'][clause]['accuracy'],
                'delta': ppo_results['per_clause'][clause]['accuracy'] - sft_results['per_clause'][clause]['accuracy'],
                'samples': sft_results['per_clause'][clause]['total']
            }
            for clause in sorted(all_clauses)
        },
        'detailed_results': {
            'sft': sft_results,
            'ppo': ppo_results
        }
    }
    
    with open(output_path, 'w') as f:
        json.dump(detailed_results, f, indent=2)
    
    print(f"\n✅ Detailed results saved to {output_path}")
    
    # Expert interpretation
    if accuracy_delta <= 0:
        print(f"\n🔍 EXPERT INTERPRETATION:")
        print(f"   PPO accuracy delta: {accuracy_delta:+.3f}")
        print(f"   → PPO is not improving over SFT baseline")
        print(f"   → Confirms policy drift rather than learning")
    else:
        print(f"\n🔍 EXPERT INTERPRETATION:")
        print(f"   PPO accuracy delta: {accuracy_delta:+.3f}")
        print(f"   → PPO shows improvement over SFT baseline")
        print(f"   → Learning signal detected")


if __name__ == '__main__':
    main()