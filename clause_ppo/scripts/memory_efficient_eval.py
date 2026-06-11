#!/usr/bin/env python3
"""
Memory-efficient evaluation - evaluates one model at a time to avoid OOM.
"""

import argparse
import json
import os
import sys
import time
import gc
import torch
from collections import defaultdict
from typing import List, Dict
from tqdm import tqdm

# Ensure src/ is on the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from models.prm_inference import load_qwen_for_inference
from utils.sql_utils import extract_sql_from_text
from utils.execution import execute_query

def evaluate_checkpoint(checkpoint_path: str, eval_samples: List[Dict], args, checkpoint_name: str) -> Dict:
    """Evaluate a single checkpoint and return results."""
    print(f"\n=== Loading {checkpoint_name} ===")
    model, tokenizer = load_qwen_for_inference(checkpoint_path)
    
    print(f"=== Evaluating {checkpoint_name} ===")
    
    results = {
        'correct': 0,
        'executable': 0,
        'total': len(eval_samples),
        'per_clause': defaultdict(lambda: {'correct': 0, 'executable': 0, 'total': 0}),
        'sample_results': []
    }
    
    db_path = os.path.join(args.spider_dir, 'database')
    
    for i, sample in enumerate(tqdm(eval_samples, desc=f"Evaluating {checkpoint_name}", unit="sample")):
        
        # Build prompt
        faulty_clause = sample['corrupted_clause']
        corrupted_position = sample['corrupted_position']
        prefix_state = sample['prefix_states'][corrupted_position]
        
        prompt = f"### Question: {prefix_state['question']}\n### Schema: {prefix_state['schema']}\n### Partial SQL: {prefix_state['prefix_query_str']}\n### Complete the {faulty_clause} clause:\n"
        
        best_correct = False
        best_executable = False
        
        # Sample best_of_n times
        for _ in range(args.best_of_n):
            # Generate
            inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024)
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
            
            completion = tokenizer.decode(
                outputs[0][inputs['input_ids'].shape[1]:], 
                skip_special_tokens=True
            ).strip()
            
            # Extract and reconstruct
            extracted_sql = extract_sql_from_text(completion)
            if not extracted_sql:
                extracted_sql = completion.strip()
                
            reconstructed_sql = prefix_state['prefix_query_str'] + " " + extracted_sql
            
            # Test execution
            db_file_path = os.path.join(db_path, sample['db_id'], f"{sample['db_id']}.sqlite")
            
            try:
                ok_recon, result = execute_query(reconstructed_sql, db_file_path, timeout_secs=5.0)
                if ok_recon:
                    best_executable = True
                    
                    # Check correctness
                    ok_gold, gold_result = execute_query(sample['original_query'], db_file_path, timeout_secs=5.0)
                    if ok_gold and result == gold_result:
                        best_correct = True
                        
            except Exception:
                pass
            
            if best_correct and args.best_of_n > 1:
                break
        
        # Update results
        if best_correct:
            results['correct'] += 1
            results['per_clause'][faulty_clause]['correct'] += 1
        if best_executable:
            results['executable'] += 1
            results['per_clause'][faulty_clause]['executable'] += 1
            
        results['per_clause'][faulty_clause]['total'] += 1
        
        results['sample_results'].append({
            'sample_id': i,
            'clause': faulty_clause,
            'correct': best_correct,
            'executable': best_executable
        })
    
    # Calculate metrics
    results['accuracy'] = results['correct'] / results['total']
    results['execution_rate'] = results['executable'] / results['total']
    
    for clause in results['per_clause']:
        clause_stats = results['per_clause'][clause]
        clause_stats['accuracy'] = clause_stats['correct'] / clause_stats['total']
        clause_stats['execution_rate'] = clause_stats['executable'] / clause_stats['total']
    
    print(f"\n  ✅ {checkpoint_name} Results: {results['accuracy']:.3f} accuracy, {results['execution_rate']:.3f} execution rate")
    
    # Free memory
    del model
    del tokenizer
    torch.cuda.empty_cache()
    gc.collect()
    
    return results

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--eval_path', default='data/processed/fixed_eval_set.json')
    parser.add_argument('--sft_checkpoint', default='results/sft_checkpoints/merged_checkpoint')
    parser.add_argument('--ppo_checkpoint', default='results/ppo_checkpoints/ep_200')
    parser.add_argument('--spider_dir', default='data/spider')
    parser.add_argument('--best_of_n', type=int, default=1)
    parser.add_argument('--temperature', type=float, default=0.8)
    parser.add_argument('--max_length', type=int, default=128)  # Shorter for speed
    parser.add_argument('--quick_test', type=int, default=None, help='Test on first N samples only')
    args = parser.parse_args()
    
    # Load evaluation set
    print(f"Loading evaluation set from {args.eval_path}")
    with open(args.eval_path) as f:
        eval_samples = json.load(f)
    
    if args.quick_test:
        eval_samples = eval_samples[:args.quick_test]
        print(f"Quick test mode: using first {args.quick_test} samples")
    
    print(f"Loaded {len(eval_samples)} evaluation samples")
    
    sampling_info = f"best-of-{args.best_of_n}" if args.best_of_n > 1 else "single-sample"
    print(f"Evaluation mode: {sampling_info}, temp={args.temperature}")
    
    start_time = time.time()
    
    # Evaluate SFT first
    sft_results = evaluate_checkpoint(args.sft_checkpoint, eval_samples, args, "SFT Baseline")
    
    # Evaluate PPO second  
    ppo_results = evaluate_checkpoint(args.ppo_checkpoint, eval_samples, args, "PPO Checkpoint")
    
    total_time = time.time() - start_time
    
    # Print comparison
    print(f"\n" + "="*60)
    print(f"FIXED EVALUATION RESULTS ({sampling_info})")
    print(f"="*60)
    print(f"Samples: {len(eval_samples)} | Time: {total_time:.1f}s")
    print(f"")
    print(f"{'Model':<15} {'Accuracy':<10} {'Exec Rate':<10}")
    print(f"{'-'*35}")
    print(f"{'SFT Baseline':<15} {sft_results['accuracy']:<10.3f} {sft_results['execution_rate']:<10.3f}")
    print(f"{'PPO ep200':<15} {ppo_results['accuracy']:<10.3f} {ppo_results['execution_rate']:<10.3f}")
    print(f"")
    
    # Delta analysis
    accuracy_delta = ppo_results['accuracy'] - sft_results['accuracy']
    exec_delta = ppo_results['execution_rate'] - sft_results['execution_rate']
    print(f"{'Delta (PPO-SFT)':<15} {accuracy_delta:<+10.3f} {exec_delta:<+10.3f}")
    
    # Expert interpretation
    print(f"\n🔍 EXPERT INTERPRETATION:")
    if accuracy_delta <= 0:
        print(f"   PPO accuracy delta: {accuracy_delta:+.3f}")
        print(f"   → PPO shows no improvement over SFT baseline")
        print(f"   → Confirms policy drift rather than learning")
        if args.best_of_n == 1:
            print(f"   → Recommend testing best-of-8 to check for headroom")
    else:
        print(f"   PPO accuracy delta: {accuracy_delta:+.3f}")
        print(f"   → PPO shows improvement over SFT baseline")
        print(f"   → Learning signal detected")
    
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
    
    # Save results
    output_path = f'results/fixed_eval_results_{sampling_info.replace("-", "_")}.json'
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    results_data = {
        'config': vars(args),
        'summary': {
            'sft': {'accuracy': sft_results['accuracy'], 'execution_rate': sft_results['execution_rate']},
            'ppo': {'accuracy': ppo_results['accuracy'], 'execution_rate': ppo_results['execution_rate']},
            'delta': {'accuracy': accuracy_delta, 'execution_rate': exec_delta}
        },
        'detailed': {'sft': sft_results, 'ppo': ppo_results}
    }
    
    with open(output_path, 'w') as f:
        json.dump(results_data, f, indent=2)
    
    print(f"\n✅ Results saved to {output_path}")

if __name__ == '__main__':
    main()