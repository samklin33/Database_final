#!/usr/bin/env python3
"""
Evaluate baseline (full regeneration), Plan B (ClausePRM), and PPO (clause-level repair) on Spider.

Outputs a markdown comparison table:

    | Method        | Accuracy@N | Avg Token Cost |
    | Full regen    |   0.XXX    |      XXX       |
    | Plan B (PRM)  |   0.XXX    |      XXX       |
    | Clause PPO    |   0.XXX    |      XXX       |

Three approaches:
- Baseline: Full query regeneration (HF Inference API)  
- Plan B: Pure reward model approach (ClausePRM + Best-of-N clause repair)
- PPO: RL-trained actor for direct clause-level repair (no separate PRM needed)

Usage:
    # All three approaches
    python scripts/evaluate.py --split dev \
        --plan-b-ckpt clause_ppo/results/prm_checkpoints/best_checkpoint \
        --ppo-ckpt clause_ppo/results/ppo_checkpoints/ep_400
    
    # Quick test
    python scripts/evaluate.py --split dev --max-samples 10 \
        --plan-b-ckpt clause_ppo/results/prm_checkpoints/best_checkpoint \
        --ppo-ckpt clause_ppo/results/ppo_checkpoints/ep_400
"""

import argparse
import json
import os
import sys

# ── Make src/ and clause_ppo/src/ importable when run as a script ──────────
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (
    os.path.join(_REPO_ROOT, 'src'),
    os.path.join(_REPO_ROOT, 'clause_ppo', 'src'),
):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from env.env             import NL2SQLEnv
from eval.metrics        import execution_accuracy
from baseline.full_regen import make_hf_api_generate_fn, run_baseline
from config import (
    SPIDER_DIR, BASELINE_MODEL, MAX_TOKENS, MAX_RETRIES, HF_TOKEN,
)


# ── CLI ────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument('--split',       default='dev', choices=['dev', 'train'])
    p.add_argument('--spider-dir',  default=SPIDER_DIR)
    p.add_argument('--max-retries', type=int, default=MAX_RETRIES)
    p.add_argument('--model',       default=BASELINE_MODEL,
                   help='HF Inference API model id for the baseline backbone.')
    p.add_argument('--max-tokens',  type=int, default=MAX_TOKENS,
                   help='Max generated tokens per API call.')
    p.add_argument('--ppo-ckpt',    default=None,
                   help='PPO actor checkpoint. PPO path is a stub today (see run_clause_ppo).')
    p.add_argument('--plan-b-ckpt', default=None,
                   help='ClausePRM checkpoint for Plan B inference (pure reward model approach).')
    p.add_argument('--max-samples', type=int, default=None,
                   help='Truncate the split for a quick smoke run.')
    p.add_argument('--output',      default=None,
                   help='Optional path to dump per-sample predictions as JSON.')
    return p.parse_args()


# ── Data + model loading ───────────────────────────────────────────────────

def load_spider(split: str, spider_dir: str) -> list[dict]:
    """Return the raw list of samples from train_spider.json / dev.json."""
    fname = 'dev.json' if split == 'dev' else 'train_spider.json'
    with open(os.path.join(spider_dir, fname)) as f:
        return json.load(f)


def build_inference_client(token: str):
    """
    Build a huggingface_hub InferenceClient for the baseline backbone.

    Lazy import — keeps the CLI importable on machines without huggingface_hub.
    """
    from huggingface_hub import InferenceClient

    if not token:
        raise SystemExit("No HF token. Set HF_TOKEN in .env")
    return InferenceClient(token=token)


# ── Per-method runners ─────────────────────────────────────────────────────

def run_full_regen(
    samples:     list[dict],
    generate_fn,
    env:         NL2SQLEnv,
    max_retries: int,
) -> tuple[list[str], list[int], list[int]]:
    """Run the full-regen baseline across all samples."""
    predictions:    list[str] = []
    token_costs:    list[int] = []
    attempt_counts: list[int] = []

    for i, sample in enumerate(samples):
        result = run_baseline(
            sample, generate_fn,
            max_retries=max_retries, env=env,
        )
        predictions.append(result['predicted_sql'])
        token_costs.append(result['token_cost'])
        attempt_counts.append(result['attempts'])
        
        print(
            f"[{i+1}/{len(samples)}] "
            f"Token cost: {result['token_cost']} | "
            f"Attempts: {result['attempts']} | "
            f"Success: {result['success']}"
        )
        
    return predictions, token_costs, attempt_counts


def run_clause_ppo(
    samples:     list[dict],
    ppo_ckpt:    str,
    max_retries: int,
    prm_ckpt:    str = None,
) -> tuple[list[str], list[int], list[int]]:
    """
    Run PPO inference: RL-trained actor + ClausePRM for best performance.
    
    Combines PPO-trained actor with ClausePRM scoring for optimal accuracy
    by generating multiple candidates and selecting the best-scored one.
    """
    from baseline.ppo_inference import run_ppo_inference
    
    return run_ppo_inference(
        samples=samples,
        ppo_ckpt=ppo_ckpt,
        max_retries=max_retries,
        limit=None,
        prm_ckpt=prm_ckpt,
    )


def run_plan_b(
    samples:     list[dict],
    prm_ckpt:    str,
    max_retries: int,
) -> tuple[list[str], list[int], list[int]]:
    """
    Run Plan B inference: ClausePRM + Best-of-N clause repair.
    
    Pure reward model approach (no RL training):
    - Uses trained ClausePRM to identify faulty clauses
    - Generates repair candidates with oracle selection
    """
    from baseline.plan_b_inference import run_plan_b_inference
    
    return run_plan_b_inference(
        samples=samples,
        prm_ckpt=prm_ckpt,
        max_retries=max_retries,
        limit=None
    )


# ── Output ─────────────────────────────────────────────────────────────────

def print_table(rows: list[dict], n: int):
    """Print the comparison table to stdout."""
    header = f"| {'Method':<12} | {f'Accuracy@{n}':>11} | {'Avg Token Cost':>14} |"
    sep    = "|" + "-" * (len(header) - 2) + "|"
    print()
    print(header)
    print(sep)
    for r in rows:
        # Handle SKIPPED values
        acc_str = f"{r['accuracy']:>11.3f}" if isinstance(r['accuracy'], (int, float)) else f"{r['accuracy']:>11}"
        tokens_str = f"{r['avg_tokens']:>14.1f}" if isinstance(r['avg_tokens'], (int, float)) else f"{r['avg_tokens']:>14}"
        
        print(
            f"| {r['method']:<12} | "
            f"{acc_str} | "
            f"{tokens_str} |"
        )


def dump_predictions(
    output_path: str,
    samples:     list[dict],
    preds:       list[str],
    tokens:      list[int],
    attempts:    list[int],
    args:        argparse.Namespace,
):
    """Write per-sample predictions as JSON for offline inspection."""
    payload = {
        'split':       args.split,
        'max_retries': args.max_retries,
        'samples': [
            {
                'db_id':         s['db_id'],
                'question':      s['question'],
                'gold_sql':      s.get('gold_sql') or s.get('query'),
                'predicted_sql': p,
                'token_cost':    t,
                'attempts':      a,
            }
            for s, p, t, a in zip(samples, preds, tokens, attempts)
        ],
    }
    with open(output_path, 'w') as f:
        json.dump(payload, f, indent=2)
    print(f"\nWrote per-sample predictions to {output_path}")


# ── Main ───────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    print(f"Loading Spider {args.split} from {args.spider_dir}")
    samples = load_spider(args.split, args.spider_dir)
    if args.max_samples is not None:
        samples = samples[:args.max_samples]
    print(f"  {len(samples)} samples")

    env = NL2SQLEnv(spider_dir=args.spider_dir)

    print(f"\nBaseline backbone (HF Inference API): {args.model}")
    client      = build_inference_client(HF_TOKEN)
    generate_fn = make_hf_api_generate_fn(client, args.model, max_tokens=args.max_tokens)

    print(f"\nRunning full-regen baseline (max_retries={args.max_retries})")
    try:
        preds, tokens, attempts = run_full_regen(
            samples, generate_fn, env, args.max_retries,
        )
        acc        = execution_accuracy(preds, samples, spider_dir=args.spider_dir)
        avg_tokens = sum(tokens) / len(tokens) if tokens else 0.0
        rows       = [{'method': 'Full regen', 'accuracy': acc, 'avg_tokens': avg_tokens}]
    except Exception as e:
        print(f"❌ Baseline failed: {e}")
        if "402" in str(e) or "Payment Required" in str(e):
            print("💰 HuggingFace API credits depleted - skipping baseline")
        else:
            print("⚠️  Unexpected error - skipping baseline")
        rows = [{'method': 'Full regen', 'accuracy': 'SKIPPED', 'avg_tokens': 'SKIPPED'}]

    if args.ppo_ckpt is not None:
        print(f"\nRunning Clause PPO (--ppo-ckpt {args.ppo_ckpt})")
        try:
            ppo_preds, ppo_tokens, _ = run_clause_ppo(
                samples, args.ppo_ckpt, args.max_retries, args.plan_b_ckpt,
            )
            ppo_acc = execution_accuracy(ppo_preds, samples, spider_dir=args.spider_dir)
            ppo_avg = sum(ppo_tokens) / len(ppo_tokens) if ppo_tokens else 0.0
            rows.append({
                'method':     'Clause PPO',
                'accuracy':   ppo_acc,
                'avg_tokens': ppo_avg,
            })
        except NotImplementedError as e:
            print(f"  Skipped — {e}")

    if args.plan_b_ckpt is not None:
        print(f"\nRunning Plan B (ClausePRM + Best-of-N) (--plan-b-ckpt {args.plan_b_ckpt})")
        try:
            plan_b_preds, plan_b_tokens, _ = run_plan_b(
                samples, args.plan_b_ckpt, args.max_retries,
            )
            plan_b_acc = execution_accuracy(plan_b_preds, samples, spider_dir=args.spider_dir)
            plan_b_avg = sum(plan_b_tokens) / len(plan_b_tokens) if plan_b_tokens else 0.0
            rows.append({
                'method':     'Plan B (PRM)',
                'accuracy':   plan_b_acc,
                'avg_tokens': plan_b_avg,
            })
        except Exception as e:
            print(f"  Plan B failed — {e}")

    print_table(rows, n=args.max_retries)

    if args.output:
        dump_predictions(args.output, samples, preds, tokens, attempts, args)


if __name__ == '__main__':
    main()
