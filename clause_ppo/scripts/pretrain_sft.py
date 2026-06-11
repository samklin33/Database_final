#!/usr/bin/env python3
"""
SFT Pretraining: Teach Qwen-7B to generate clean SQL only.

Supervised fine-tuning on Spider data to establish SQL-focused behavior
before PPO clause repair training. Uses train_spider[:4000] (non-overlapping).

Usage:
  python scripts/pretrain_sft.py --spider_dir ../../spider
"""

import argparse
import json
import os
import sys
from typing import List, Dict

import torch
from transformers import (
    AutoTokenizer, AutoModelForCausalLM, 
    TrainingArguments, Trainer, DataCollatorForLanguageModeling
)

from peft import LoraConfig, get_peft_model, TaskType
from datasets import Dataset

# Add clause_ppo/src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from data.clause_splitter import schema_to_string


def build_sft_prompt(question: str, schema: str) -> str:
    """Build SFT prompt for clean SQL generation."""
    return f"[QUESTION] {question} [SCHEMA] {schema} [SQL] "


def build_sft_dataset(spider_dir: str, tokenizer, split_end: int = 4000) -> List[Dict]:
    """Build SFT dataset from Spider train split."""
    print(f"Loading Spider data from {spider_dir}")
    
    # Load Spider data (same as PPO training)
    with open(os.path.join(spider_dir, 'tables.json')) as f:
        tables_dict = {t['db_id']: t for t in json.load(f)}
    with open(os.path.join(spider_dir, 'train_spider.json')) as f:
        all_train = json.load(f)
    
    # Use first split_end samples (non-overlapping with PPO)
    samples = all_train[:split_end]
    print(f"Using {len(samples)} samples for SFT pretraining")
    
    sft_data = []
    for sample in samples:
        db_id = sample['db_id']
        question = sample['question']
        sql = sample['query']
        
        # Build schema and prompt (same format as PPO training)
        schema = schema_to_string(db_id, tables_dict)
        prompt = build_sft_prompt(question, schema)
        
        sft_data.append({
            'input_text': prompt,
            'target_text': sql,  # Clean SQL only
            'combined_text': prompt + sql + tokenizer.eos_token  # Add EOS token!
        })
    
    return sft_data


def tokenize_function(examples, tokenizer, max_length=512):
    """Tokenize SFT training examples."""
    # Tokenize combined text (input + target)
    tokenized = tokenizer(
        examples['combined_text'],
        truncation=True,
        max_length=max_length,
        padding='max_length',  # Pad to max_length for consistent batching
        return_tensors=None
    )
    
    # Create labels with proper masking for padding tokens
    labels = tokenized['input_ids'][:]
    attention_mask = tokenized['attention_mask'][:]
    
    # Mask padding tokens in labels (-100 = ignore in loss computation)
    for i in range(len(labels)):
        labels[i] = [token_id if attention_mask[i][j] == 1 else -100 
                    for j, token_id in enumerate(labels[i])]
    
    tokenized['labels'] = labels
    
    return tokenized


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--spider_dir', default='../../spider', help='Path to Spider dataset')
    parser.add_argument('--model_name', default='/home/henrylin0822/models/qwen', help='Base model path')
    parser.add_argument('--output_dir', default='results/sft_checkpoints', help='Output directory')
    parser.add_argument('--max_samples', type=int, default=4000, help='Max samples for SFT')
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load model and tokenizer
    print(f"Loading model: {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.float32,  # Use float32 to avoid numerical issues
        device_map="auto",
        trust_remote_code=True
    )
    
    # Configure tokenizer
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Setup LoRA
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=["q_proj", "v_proj"]
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    # Build dataset
    sft_data = build_sft_dataset(args.spider_dir, tokenizer, args.max_samples)
    
    
    # Create HuggingFace dataset
    dataset = Dataset.from_list(sft_data)
    tokenized_dataset = dataset.map(
        lambda x: tokenize_function(x, tokenizer),
        batched=True,
        remove_columns=dataset.column_names
    )
    
    
    # Split for train/validation
    split_dataset = tokenized_dataset.train_test_split(test_size=0.1, seed=42)
    train_dataset = split_dataset['train']
    eval_dataset = split_dataset['test']
    
    print(f"Train samples: {len(train_dataset)}")
    print(f"Eval samples: {len(eval_dataset)}")
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=2,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        gradient_accumulation_steps=4,
        learning_rate=1e-5,  # Much lower LR to prevent NaN
        weight_decay=0.01,
        max_grad_norm=1.0,   # Gradient clipping
        logging_steps=50,
        eval_steps=100,
        save_steps=500,
        evaluation_strategy="steps",
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        warmup_steps=100,
        fp16=False,  # Disable FP16 to avoid gradient unscaling issues with LoRA
        dataloader_pin_memory=False,
    )
    
    # Data collator - simple default since we're padding to max_length
    data_collator = None  # Use default data collator since we pre-padded
    
    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
    )
    
    # Train
    print("Starting SFT pretraining...")
    trainer.train()
    
    # Save final model
    final_path = os.path.join(args.output_dir, 'final')
    trainer.save_model(final_path)
    print(f"SFT pretraining complete. Model saved to {final_path}")


if __name__ == '__main__':
    main()