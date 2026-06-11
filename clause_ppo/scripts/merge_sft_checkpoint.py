#!/usr/bin/env python3
"""
merge_sft_checkpoint.py
=======================
Merges SFT LoRA adapter with base model and saves as merged checkpoint.
This provides the SFT baseline for evaluation comparisons.

Usage:
    python scripts/merge_sft_checkpoint.py \
        --sft_adapter results/sft_checkpoints/final \
        --output_path results/sft_checkpoints/merged_checkpoint
"""

import argparse
import os
import sys
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel, PeftConfig

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--sft_adapter', default='results/sft_checkpoints/final')
    p.add_argument('--output_path', default='results/sft_checkpoints/merged_checkpoint')
    return p.parse_args()

def main():
    args = parse_args()
    
    # Load PEFT config to get base model path
    print(f"Loading PEFT config from {args.sft_adapter}")
    peft_config = PeftConfig.from_pretrained(args.sft_adapter)
    base_model_path = peft_config.base_model_name_or_path
    
    print(f"Base model: {base_model_path}")
    print(f"Loading base model...")
    
    # Load base model
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype=torch.bfloat16,
        device_map='auto'
    )
    
    # Load adapter
    print(f"Loading SFT adapter...")
    model = PeftModel.from_pretrained(base_model, args.sft_adapter)
    
    # Merge adapter with base model
    print(f"Merging adapter with base model...")
    merged_model = model.merge_and_unload()
    
    # Load tokenizer from base model (since SFT adapter may not have tokenizer files)
    tokenizer = AutoTokenizer.from_pretrained(base_model_path)
    
    # Save merged model and tokenizer
    print(f"Saving merged checkpoint to {args.output_path}")
    os.makedirs(args.output_path, exist_ok=True)
    
    merged_model.save_pretrained(args.output_path)
    tokenizer.save_pretrained(args.output_path)
    
    print(f"✅ Merged SFT checkpoint saved to {args.output_path}")
    
    # Test generation
    print(f"\nTesting merged model...")
    merged_model.eval()
    
    test_prompt = "SELECT name FROM"
    inputs = tokenizer(test_prompt, return_tensors="pt")
    
    # Move inputs to same device as model
    model_device = next(merged_model.parameters()).device
    inputs = {k: v.to(model_device) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = merged_model.generate(
            inputs['input_ids'],
            attention_mask=inputs['attention_mask'],
            max_length=inputs['input_ids'].shape[1] + 20,
            temperature=0.1,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
    
    completion = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
    print(f"Test prompt: {test_prompt}")
    print(f"Completion: {completion}")
    print(f"Model ready for evaluation.")

if __name__ == '__main__':
    main()