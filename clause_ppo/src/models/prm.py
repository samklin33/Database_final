"""
ClausePRM: Process Reward Model for CLAUSE-PPO.

Architecture:
  Backbone:  CodeLlama-7B (codellama/CodeLlama-7b-hf) loaded in 4-bit NF4
             quantization via bitsandbytes (QLoRA).
  Adapter:   LoRA rank=16, alpha=32 on q_proj and v_proj attention layers
             via the PEFT library. Only adapter + regression head weights
             are trainable.
  Head:      Linear(hidden_size → 1) + Sigmoid → scalar score in (0, 1).

Architecture note — [CLS] vs last token:
  The CLAUSE-PPO paper uses "[CLS] token hidden state" notation inherited from
  BERT-style encoder models. CodeLlama is a causal decoder-only model with no
  [CLS] token. We use the hidden state of the LAST (non-padding) token instead,
  which is the standard equivalent for decoder-only models: the last token has
  attended to all prior tokens and thus encodes a global representation of the
  sequence, analogous to [CLS] in encoders.

VRAM estimate (RTX 4090, 24 GB):
  4-bit CodeLlama-7B:  ~4 GB
  LoRA adapters:       ~0.05 GB
  Activations (bs=2):  ~2 GB
  Overhead:            ~1 GB
  Total:               ~7 GB  — well within 24 GB budget
"""

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, TaskType


class ClausePRM(nn.Module):
    """Clause-level Process Reward Model based on CodeLlama-7B with QLoRA."""

    def __init__(
        self,
        model_name: str = 'codellama/CodeLlama-7b-hf',
        lora_rank: int = 16,
        lora_alpha: int = 32,
        lora_dropout: float = 0.05,
        target_modules: list = None,
        use_4bit: bool = True,
    ):
        super().__init__()

        if target_modules is None:
            target_modules = ['q_proj', 'v_proj']

        # ── Load backbone ─────────────────────────────────────────────────
        print(f"  Loading backbone: {model_name} ({'4-bit NF4 QLoRA' if use_4bit else 'bfloat16'}) ...", flush=True)
        if use_4bit:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type='nf4',
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
            )
            self.backbone = AutoModelForCausalLM.from_pretrained(
                model_name,
                quantization_config=bnb_config,
                device_map='auto',
                torch_dtype=torch.bfloat16,
            )
        else:
            self.backbone = AutoModelForCausalLM.from_pretrained(
                model_name,
                device_map='auto',
                torch_dtype=torch.bfloat16,
            )
        print("  Backbone loaded.", flush=True)

        # Enable gradient checkpointing to reduce VRAM usage
        self.backbone.gradient_checkpointing_enable()

        # ── Attach LoRA adapter ───────────────────────────────────────────
        print(f"  Attaching LoRA (rank={lora_rank}, alpha={lora_alpha}, modules={target_modules}) ...", flush=True)
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=lora_rank,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            target_modules=target_modules,
            bias='none',
        )
        self.backbone = get_peft_model(self.backbone, lora_config)
        print("  LoRA attached.", flush=True)

        # ── Scalar regression head ────────────────────────────────────────
        hidden_size = self.backbone.config.hidden_size
        backbone_device = next(self.backbone.parameters()).device
        backbone_dtype = next(self.backbone.parameters()).dtype
        self.score_head = nn.Linear(hidden_size, 1).to(device=backbone_device, dtype=backbone_dtype)
        print(f"  Score head: Linear({hidden_size} → 1) (device={backbone_device})", flush=True)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            input_ids:      (batch, seq_len) long tensor
            attention_mask: (batch, seq_len) long tensor (1=real, 0=pad)

        Returns:
            scores: (batch,) float tensor in (0, 1)
        """
        outputs = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True,
        )

        # Use last hidden layer hidden states: (batch, seq_len, hidden_size)
        last_hidden = outputs.hidden_states[-1]

        # Extract hidden state of the last NON-PADDING token per example.
        # This is the decoder equivalent of [CLS]: the last token has attended
        # to all prior tokens and encodes a global sequence representation.
        seq_lengths = attention_mask.sum(dim=1) - 1     # (batch,)
        batch_idx   = torch.arange(last_hidden.size(0), device=last_hidden.device)
        last_token_hidden = last_hidden[batch_idx, seq_lengths]   # (batch, hidden)

        return self.score_head(last_token_hidden).squeeze(-1)   # (batch,) logits
