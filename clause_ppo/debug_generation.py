#!/usr/bin/env python3
"""
Quick debug to see what the models are actually generating
"""

import sys
import os
import json
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from models.prm_inference import load_qwen_for_inference

# Load one sample
with open('data/processed/fixed_eval_set.json') as f:
    sample = json.load(f)[0]

print(f"Sample: {sample['db_id']} - {sample['corrupted_clause']}")

# Build prompt
faulty_clause = sample['corrupted_clause']
corrupted_position = sample['corrupted_position']
prefix_state = sample['prefix_states'][corrupted_position]

prompt = f"### Question: {prefix_state['question']}\n### Schema: {prefix_state['schema'][:300]}...\n### Partial SQL: {prefix_state['prefix_query_str']}\n### Complete the {faulty_clause} clause:\n"

print(f"\nPrompt:\n{prompt}")
print(f"\nOriginal query: {sample['original_query']}")
print(f"Corrupted query: {sample['corrupted_query']}")

# Test SFT generation
print(f"\n=== Testing SFT Generation ===")
model, tokenizer = load_qwen_for_inference('results/sft_checkpoints/merged_checkpoint')

inputs = tokenizer(prompt, return_tensors="pt")
model_device = next(model.parameters()).device
inputs = {k: v.to(model_device) for k, v in inputs.items()}

with torch.no_grad():
    outputs = model.generate(
        inputs['input_ids'],
        attention_mask=inputs['attention_mask'],
        max_length=inputs['input_ids'].shape[1] + 50,
        temperature=0.1,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id
    )

completion = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True).strip()

print(f"Raw generation: '{completion}'")
print(f"Reconstructed: '{prefix_state['prefix_query_str']} {completion}'")

# Quick execution test
from utils.execution import execute_query
db_file_path = f"data/spider/database/{sample['db_id']}/{sample['db_id']}.sqlite"
reconstructed_sql = prefix_state['prefix_query_str'] + " " + completion

try:
    ok_recon, result = execute_query(reconstructed_sql, db_file_path, timeout_secs=5.0)
    print(f"Reconstruction executable: {ok_recon}")
    if ok_recon:
        print(f"Result: {len(result)} rows")
        
    ok_gold, gold_result = execute_query(sample['original_query'], db_file_path, timeout_secs=5.0)
    print(f"Gold executable: {ok_gold}")
    if ok_gold:
        print(f"Gold result: {len(gold_result)} rows")
        if ok_recon:
            print(f"Results match: {result == gold_result}")
            
except Exception as e:
    print(f"Execution error: {e}")