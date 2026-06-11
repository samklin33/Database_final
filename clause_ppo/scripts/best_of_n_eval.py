#!/usr/bin/env python3
"""
Best-of-N evaluation for SFT model to measure headroom.
This tests whether there's learnable signal by sampling multiple times.
"""

import argparse
import json
import os
import sys
import time
import torch
from collections import defaultdict
from typing import List, Dict
from tqdm import tqdm

# Ensure src/ is on the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from models.prm_inference import load_qwen_for_inference
from utils.sql_utils import extract_sql_from_text
from utils.execution import execute_query

def evaluate_sft_with_sampling(checkpoint_path: str, eval_samples: List[Dict], args) -> Dict:
    """Evaluate SFT model with different sampling strategies."""
    print(f"\n=== Loading SFT Model ===")
    model, tokenizer = load_qwen_for_inference(checkpoint_path)
    
    results = {}
    
    # Test different sampling configurations
    sampling_configs = [
        {'n': 1, 'temp': 0.0, 'name': 'single_greedy'},
        {'n': 1, 'temp': 0.8, 'name': 'single_sample'},
        {'n': 4, 'temp': 0.8, 'name': 'best_of_4'},
        {'n': 8, 'temp': 0.8, 'name': 'best_of_8'}
    ]
    
    db_path = os.path.join(args.spider_dir, 'database')
    
    for config in sampling_configs:
        print(f"\n=== Evaluating {config['name']} (n={config['n']}, temp={config['temp']}) ===")
        
        eval_results = {
            'correct': 0,
            'executable': 0, 
            'total': len(eval_samples),
            'per_clause': defaultdict(lambda: {'correct': 0, 'executable': 0, 'total': 0}),
            'config': config
        }
        
        # Use subset for speed if testing multiple configs
        test_samples = eval_samples[:args.max_samples] if args.max_samples else eval_samples
        eval_results['total'] = len(test_samples)
        
        for i, sample in enumerate(tqdm(test_samples, desc=f"{config['name']}", unit="sample")):
            # Try to fix the prompt format based on corruption type
            faulty_clause = sample['corrupted_clause']
            
            # Build a proper completion prompt
            question = sample['question']
            schema = sample['schema']
            
            if faulty_clause == 'from':
                prompt = f"Question: {question}\nSchema: {schema}\nRewrite this FROM clause to fix the query: FROM {sample['corrupted_query'].split('FROM')[1].split()[0]}\nAnswer: FROM "
            else:
                # For other clauses, use a simpler approach
                prompt = f"Question: {question}\nSchema: {schema}\nComplete this SQL query to answer the question:\n{sample['corrupted_query']}\nCorrected SQL:"
                
            best_correct = False
            best_executable = False
            
            # Sample n times
            for _ in range(config['n']):
                inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024)
                model_device = next(model.parameters()).device
                inputs = {k: v.to(model_device) for k, v in inputs.items()}
                
                with torch.no_grad():
                    outputs = model.generate(
                        inputs['input_ids'],
                        attention_mask=inputs['attention_mask'],
                        max_length=inputs['input_ids'].shape[1] + 128,
                        temperature=config['temp'],
                        do_sample=config['temp'] > 0,
                        pad_token_id=tokenizer.eos_token_id,
                        eos_token_id=tokenizer.eos_token_id
                    )
                
                completion = tokenizer.decode(
                    outputs[0][inputs['input_ids'].shape[1]:], 
                    skip_special_tokens=True
                ).strip()
                
                # Extract SQL
                if faulty_clause == 'from':
                    # Simple FROM clause completion
                    extracted_sql = "FROM " + extract_sql_from_text(completion)
                    if not extracted_sql.startswith("FROM "):
                        extracted_sql = completion.strip()
                        if not extracted_sql.startswith("FROM"):
                            extracted_sql = "FROM " + extracted_sql
                else:
                    # Full query extraction
                    extracted_sql = extract_sql_from_text(completion)
                    if not extracted_sql:
                        extracted_sql = completion.strip()
                
                # Test execution
                db_file_path = os.path.join(db_path, sample['db_id'], f"{sample['db_id']}.sqlite")
                
                try:
                    if faulty_clause == 'from':
                        # Reconstruct the query by replacing the corrupted FROM
                        original_parts = sample['original_query'].split()
                        corrupted_parts = sample['corrupted_query'].split()
                        
                        # Find FROM position and replace
                        from_idx = -1
                        for idx, part in enumerate(corrupted_parts):
                            if part.upper() == 'FROM':
                                from_idx = idx
                                break
                        
                        if from_idx >= 0 and from_idx + 1 < len(corrupted_parts):
                            # Replace the corrupted table name
                            new_query_parts = corrupted_parts.copy()
                            new_table = extracted_sql.replace("FROM ", "").split()[0]
                            new_query_parts[from_idx + 1] = new_table
                            reconstructed_sql = " ".join(new_query_parts)
                        else:
                            reconstructed_sql = extracted_sql
                    else:
                        reconstructed_sql = extracted_sql
                    
                    ok_recon, result = execute_query(reconstructed_sql, db_file_path, timeout_secs=5.0)
                    if ok_recon:
                        best_executable = True
                        
                        # Check correctness
                        ok_gold, gold_result = execute_query(sample['original_query'], db_file_path, timeout_secs=5.0)
                        if ok_gold and result == gold_result:
                            best_correct = True
                            
                except Exception:
                    pass
                
                if best_correct and config['n'] > 1:
                    break
            
            # Update results
            if best_correct:
                eval_results['correct'] += 1
                eval_results['per_clause'][faulty_clause]['correct'] += 1
            if best_executable:
                eval_results['executable'] += 1
                eval_results['per_clause'][faulty_clause]['executable'] += 1
                
            eval_results['per_clause'][faulty_clause]['total'] += 1
        
        # Calculate metrics
        eval_results['accuracy'] = eval_results['correct'] / eval_results['total']
        eval_results['execution_rate'] = eval_results['executable'] / eval_results['total']
        
        for clause in eval_results['per_clause']:
            clause_stats = eval_results['per_clause'][clause]
            if clause_stats['total'] > 0:
                clause_stats['accuracy'] = clause_stats['correct'] / clause_stats['total']
                clause_stats['execution_rate'] = clause_stats['executable'] / clause_stats['total']
        
        results[config['name']] = eval_results
        print(f"  ✅ {config['name']}: {eval_results['accuracy']:.3f} accuracy ({eval_results['correct']}/{eval_results['total']})")
    
    return results

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--eval_path', default='data/processed/fixed_eval_set.json')
    parser.add_argument('--sft_checkpoint', default='results/sft_checkpoints/merged_checkpoint')
    parser.add_argument('--spider_dir', default='data/spider')
    parser.add_argument('--max_samples', type=int, default=50, help='Max samples for quick testing')
    args = parser.parse_args()
    
    # Load evaluation set
    print(f"Loading evaluation set from {args.eval_path}")
    with open(args.eval_path) as f:
        eval_samples = json.load(f)
    
    if args.max_samples:
        eval_samples = eval_samples[:args.max_samples]
        print(f"Using first {args.max_samples} samples for quick evaluation")
    
    print(f"Loaded {len(eval_samples)} evaluation samples")
    
    start_time = time.time()
    
    # Run evaluation
    results = evaluate_sft_with_sampling(args.sft_checkpoint, eval_samples, args)
    
    total_time = time.time() - start_time
    
    # Print comparison
    print(f"\n" + "="*60)
    print(f"BEST-OF-N HEADROOM ANALYSIS")
    print(f"="*60)
    print(f"Samples: {len(eval_samples)} | Time: {total_time:.1f}s")
    print(f"")
    print(f"{'Strategy':<15} {'Accuracy':<10} {'Exec Rate':<10} {'Improvement':<12}")
    print(f"{'-'*55}")
    
    baseline_acc = results['single_greedy']['accuracy']
    
    for strategy_name, result in results.items():
        improvement = result['accuracy'] - baseline_acc
        print(f"{strategy_name:<15} {result['accuracy']:<10.3f} {result['execution_rate']:<10.3f} {improvement:<+12.3f}")
    
    # Expert interpretation
    best_of_8_acc = results['best_of_8']['accuracy']
    single_acc = results['single_greedy']['accuracy']
    headroom = best_of_8_acc - single_acc
    
    print(f"\n🔍 EXPERT INTERPRETATION:")
    print(f"   Single-sample accuracy: {single_acc:.3f}")
    print(f"   Best-of-8 accuracy: {best_of_8_acc:.3f}")
    print(f"   Headroom (best-of-8 - single): {headroom:+.3f}")
    
    if headroom > 0.1:
        print(f"   → Significant headroom detected ({headroom:.3f})")
        print(f"   → Model CAN produce correct repairs but doesn't reliably pick them")
        print(f"   → Strong candidate for ReST/RAFT approach")
    elif headroom > 0.05:
        print(f"   → Moderate headroom detected ({headroom:.3f})")
        print(f"   → Some potential for RL approaches")
    else:
        print(f"   → Limited headroom ({headroom:.3f})")
        print(f"   → Model is at capability ceiling, RL unlikely to help")
        print(f"   → Focus on data/model improvements instead")
    
    # Save results
    output_path = f'results/best_of_n_headroom_analysis.json'
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump({
            'config': vars(args),
            'results': results,
            'summary': {
                'baseline_accuracy': baseline_acc,
                'best_of_8_accuracy': best_of_8_acc,
                'headroom': headroom,
                'samples_tested': len(eval_samples)
            }
        }, f, indent=2, default=str)
    
    print(f"\n✅ Detailed results saved to {output_path}")

if __name__ == '__main__':
    main()