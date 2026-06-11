"""
PRMDataset: PyTorch Dataset for training the CLAUSE-PPO Process Reward Model.

Loads corruption_dataset.json and original_dataset.json produced by
build_corruption_dataset.py and emits (input_ids, attention_mask, label) triples.

Labeling (cascade):
  - Positive (original queries): all prefix positions get label 1.0
  - Negative (corrupted queries): positions 0..j*-1 get label 1.0,
                                  positions j* and beyond get label 0.0

Input format:
  [QUESTION] {question} [SCHEMA] {schema} [PREFIX] {prefix_query_str}

Schema dropout (training only): with probability p, mask 30% of schema tokens
with '<mask>' to encourage learning from query structure, not just column lookup.
"""

import json
import os
import random
from typing import Optional

import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer


class PRMDataset(Dataset):
    """
    Dataset for PRM training. Each item is one (prefix_state, label) pair.

    Items come from two sources:
      - original_dataset.json  → all prefix states, label 1.0
      - corruption_dataset.json → cascade-labeled prefix states
    """

    MASK_TOKEN = '<mask>'

    def __init__(
        self,
        processed_dir: str,
        tokenizer_name: str = 'codellama/CodeLlama-7b-hf',
        max_length: int = 512,
        schema_dropout_prob: float = 0.3,
        training: bool = True,
        tokenizer: Optional[object] = None,
    ):
        """
        Args:
            processed_dir:       Path to data/processed/ directory.
            tokenizer_name:      HuggingFace model name for the tokenizer.
            max_length:          Maximum token sequence length (pad/truncate).
            schema_dropout_prob: Probability of applying schema token masking.
            training:            If False, schema dropout is disabled.
            tokenizer:           Pass a pre-loaded tokenizer to avoid reloading.
        """
        self.max_length          = max_length
        self.schema_dropout_prob = schema_dropout_prob if training else 0.0
        self.training            = training

        if tokenizer is not None:
            self.tokenizer = tokenizer
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(
                tokenizer_name, use_fast=True
            )
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

        # ── Load processed data ───────────────────────────────────────────
        orig_path = os.path.join(processed_dir, 'original_dataset.json')
        
        # Try sophisticated rewards first, fallback to basic corruption dataset
        corr_path_sophisticated = os.path.join(processed_dir, 'corruption_dataset_with_rewards.json')
        corr_path_basic = os.path.join(processed_dir, 'corruption_dataset.json')
        
        if os.path.exists(corr_path_sophisticated):
            corr_path = corr_path_sophisticated
            use_sophisticated_rewards = True
            print(f"Loading sophisticated dataset: {corr_path}")
        else:
            corr_path = corr_path_basic
            use_sophisticated_rewards = False
            print(f"Loading basic dataset: {corr_path}")

        with open(orig_path) as f:
            originals = json.load(f)
        with open(corr_path) as f:
            corruptions = json.load(f)

        # ── Build flat list of (text, label) pairs ────────────────────────
        self.items: list[tuple[str, float]] = []

        # Positive examples: one item per prefix position per original query
        for ex in originals:
            for state in ex.get('prefix_states', []):
                text = self._format_input(
                    state['question'],
                    state['schema'],
                    state['prefix_query_str'],
                )
                self.items.append((text, 1.0))

        # Cascade-labeled negative examples from corruption records
        # j* = corrupted_position (0-indexed position in clause execution order)
        # Positions 0..j*-1 → label 1.0 (prefix was still correct)
        # Position j* → sophisticated reward or 0.0 (fault introduced here)
        for c in corruptions:
            j_star      = c['corrupted_position']
            question    = c['question']
            schema      = f"[schema for {c['db_id']}]"
            clause_name = c['corrupted_clause']

            if use_sophisticated_rewards and 'reward' in c and 'clause_names' in c:
                # Use sophisticated per-clause rewards
                clause_names = c['clause_names']
                rewards = c['reward']
                
                for pos, reward in enumerate(rewards):
                    if pos < len(clause_names):
                        clause_name_at_pos = clause_names[pos]
                        prefix_str = f"[PREFIX UP TO {clause_name_at_pos.upper()} CLAUSE]"
                        text = self._format_input(question, schema, prefix_str)
                        self.items.append((text, float(reward)))
            else:
                # Fallback to binary labeling
                # Positions before the corrupted clause → label 1.0
                for pos in range(j_star):
                    prefix_str = f"[PREFIX UP TO POSITION {pos}]"
                    text = self._format_input(question, schema, prefix_str)
                    self.items.append((text, 1.0))

                # The corrupted position itself → label 0.0
                prefix_str = f"[CORRUPTED {clause_name.upper()}]"
                text = self._format_input(question, schema, prefix_str)
                self.items.append((text, 0.0))

    # ── Public interface ──────────────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> dict:
        text, label = self.items[idx]

        if self.training and random.random() < self.schema_dropout_prob:
            text = self._apply_schema_dropout(text)

        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt',
        )

        return {
            'input_ids':      encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'label':          torch.tensor(label, dtype=torch.float),
        }

    # ── Helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _format_input(question: str, schema: str, prefix_str: str) -> str:
        """Build the model input string from its three components."""
        return f"[QUESTION] {question} [SCHEMA] {schema} [PREFIX] {prefix_str}"

    def _apply_schema_dropout(self, text: str) -> str:
        """
        Mask approximately 30% of schema column tokens with self.MASK_TOKEN.
        Only tokens between [SCHEMA] and [PREFIX] are eligible.
        """
        schema_start = text.find('[SCHEMA]')
        prefix_start = text.find('[PREFIX]')
        if schema_start == -1 or prefix_start == -1:
            return text

        prefix_part  = text[:schema_start + len('[SCHEMA]')] + ' '
        schema_part  = text[schema_start + len('[SCHEMA]') + 1: prefix_start]
        suffix_part  = ' ' + text[prefix_start:]

        schema_tokens = schema_part.split()
        masked = [
            self.MASK_TOKEN if random.random() < 0.3 else tok
            for tok in schema_tokens
        ]
        return prefix_part + ' '.join(masked) + suffix_part
