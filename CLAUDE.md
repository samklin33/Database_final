# CLAUDE.md — Database_final

NL2SQL clause-level repair via PPO.  
Given a wrong SQL query, identify the faulty clause and rewrite only that clause using reinforcement learning.

## Repo Structure

```
Database_final/
├── CLAUDE.md
├── .claude/docs/
│   ├── PIPELINE.md       ← full pipeline design & data flow
│   ├── INTERFACES.md     ← agreed function signatures between modules
│   └── QUESTIONS.md      ← open questions & decisions log
│
├── clause_ppo/           ← ALL RL code (Henry owns everything here)
│   ├── configs/
│   │   ├── ppo_config.yaml     ← Enhanced reward config (execution_scale: 10.0)
│   │   └── prm_config.yaml
│   ├── data/processed/   ← built datasets (not committed)
│   ├── results/
│   │   ├── prm_checkpoints/    ← Trained ClausePRM models
│   │   └── ppo_checkpoints/    ← PPO actor checkpoints (552 episodes)
│   ├── scripts/
│   │   ├── build_corruption_dataset.py
│   │   ├── train_prm.py
│   │   ├── train_ppo.py  ← PPO training entry point
│   │   ├── clause_rewards.py
│   │   └── score_clause.py
│   └── src/
│       ├── data/         ← clause_splitter, corruption, dataset
│       ├── models/       ← prm.py, prm_inference.py
│       ├── training/     ← ppo_loop.py, train_prm.py
│       └── utils/        ← execution.py, sql_utils.py
│
├── src/                  ← Sam's code (env + eval only)
│   ├── env/
│   │   └── env.py        ← NL2SQLEnv (DONE)
│   ├── eval/
│   │   └── metrics.py    ← execution_accuracy, partial_match (DONE)
│   └── baseline/
│       ├── plan_b_inference.py  ← ClausePRM + Best-of-N inference (DONE)
│       └── ppo_inference.py     ← PPO actor inference (DONE)
│
├── scripts/
│   └── evaluate.py       ← Three-way evaluation: baseline, Plan B, PPO (DONE)
│
├── tests/                ← test suite (DONE)
├── conftest.py
├── validate_env.py
└── requirements.txt
```

## Team

| Name | Role |
|------|------|
| Sam | `src/env/env.py`, `src/eval/metrics.py`, `scripts/evaluate.py` |
| Henry | Everything under `clause_ppo/` |
| Ian | Baseline inference, `scripts/evaluate.py` helper, demo |

## Key Design Decisions

- **Dataset**: Spider (SQLite, no server needed)
- **Model**: Qwen-7B (changed from CodeLlama for better performance)
- **RL framework**: `trl` PPOTrainer
- **Data split**: `train_spider[4000:]` → PPO, `dev.json` → eval only
- **Episode init**: corruption engine (`get_corrupted_sample`) produces wrong SQL, NOT Qwen's actual output
- **Reward**: Enhanced `execution_reward * 10.0 + alpha * prm_score` (stronger execution signal)
- **Baseline**: full query regeneration, `max_retries` configurable (default 3)
- **Metric**: Accuracy@N + avg token cost (input + output tokens)

## Critical Integration Point

`ppo_loop.py` calls `env.py` directly:
```python
from env.env import NL2SQLEnv
env = NL2SQLEnv(spider_dir=spider_dir, tables=tables_dict)
state = env.reset(sample)          # sample from train_spider.json
terminal, _ = env.step(rewritten_sql)
```
`env.py` signatures must not change without updating `ppo_loop.py`.

## Environment Setup

```bash
pip install -r requirements.txt
# Spider dataset → clause_ppo/data/spider/ (see PIPELINE.md)

# PPO Training Environment
conda create -n ppo_training python=3.10
conda activate ppo_training
pip install -r clause_ppo/requirements_cpu.txt  # or requirements.txt for GPU
```

Training: Henry's Windows PC (WSL2, RTX 4090).  
Eval: any machine with Python 3.10+.

## Training Status

- **ClausePRM**: ✅ Trained and validated
- **PPO Training**: ✅ Completed (552 episodes, enhanced rewards)
- **Plan B Inference**: ✅ Implemented and tested
- **PPO Inference**: ✅ Implemented and tested
- **Full Evaluation**: ✅ Three-way comparison ready

## Usage

```bash
# PPO Training
./clause_ppo/run_training.sh

# Three-way Evaluation
python scripts/evaluate.py --split dev \
    --plan-b-ckpt clause_ppo/results/prm_checkpoints/best_checkpoint \
    --ppo-ckpt clause_ppo/results/ppo_checkpoints/ep_400
```

## Open Questions

See `.claude/docs/QUESTIONS.md`.