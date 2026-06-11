# Why PPO Training Isn't Suitable for Clause-Level SQL Repair

## Executive Summary

Our experiments demonstrate that PPO (Proximal Policy Optimization) faces fundamental challenges when applied to clause-level SQL repair, achieving only marginal improvements over baseline approaches despite significant computational overhead. This analysis identifies the core technical reasons why RL-based approaches struggle with this task.

## Key Experimental Findings

- **PPO ep_50**: Limited improvement over baseline
- **PPO ep_200**: Diminishing returns with evidence of policy drift  
- **Best-of-N Analysis**: 14% headroom available, suggesting better selection methods exist
- **Training Trajectory**: Declining performance from 75% peak to 45% final accuracy

## Technical Reasons Why PPO Fails

### 1. **Sparse Binary Reward Signal**

**Problem**: SQL execution provides only binary feedback (±3.0) with no intermediate signals.

```
Reward = { +3.0 if query executes AND produces correct result
         { -3.0 otherwise
```

**Impact**: 
- Insufficient learning signal for complex multi-step reasoning
- No gradient information about "how close" a query is to being correct
- Policy updates become random walk without dense feedback

**Evidence**: Training logs show high variance with frequent ±3.0 swings, indicating the agent cannot distinguish between "almost correct" and "completely wrong" attempts.

### 2. **Critic Network Inadequacy**

**Problem**: Value function approximation fails with limited model capacity.

**Technical Details**:
- Model size: ~1.5B parameters (not 7B as initially assumed)
- Hidden size: 1536 → value head bottleneck  
- Task complexity: Multi-step reasoning over structured data

**Impact**:
- Critic cannot learn reliable baseline estimates
- High variance advantage estimates lead to unstable updates
- PPO's clipping mechanism becomes ineffective without proper baselines

**Evidence**: Training logs show `value_loss` values but poor convergence, indicating critic struggles to estimate expected returns.

### 3. **Policy Drift and KL Divergence Issues**

**Problem**: Policy wanders from SFT initialization without proper anchoring.

**Observations**:
- Generation speed collapse: 1.7 it/s → 22 s/it (1200% slowdown)
- Longer, rambling outputs indicate policy losing tight generation discipline
- KL divergence logging mostly null, suggesting measurement/control failures

**Root Cause**: The reference model sharing bug (fixed late) allowed policy to drift without proper KL penalties, and even after fixing, the sparse reward doesn't provide enough signal to maintain coherent behavior.

### 4. **Mismatch Between Task Structure and RL Assumptions**

**Problem**: Clause-level repair is inherently a **structured prediction** task, not a **sequential decision** problem.

**Task Characteristics**:
- **Deterministic target**: Each corrupted clause has specific correct repairs
- **Single-step decision**: Fix one clause, evaluate result  
- **No exploration benefit**: Random exploration hurts more than it helps
- **Discrete search space**: Limited valid SQL syntax options

**RL Assumptions**:
- **Sequential rewards**: Benefit from exploring action sequences
- **Continuous improvement**: Learning from partial successes
- **Exploration value**: Random actions discover new strategies

**Conclusion**: The task structure favors **supervised selection** (pick best from candidates) over **policy optimization** (learn to generate better sequences).

### 5. **Scale Mismatch**

**Problem**: Task complexity vs. model capacity mismatch.

**Analysis**:
- **Required reasoning**: Multi-table joins, complex WHERE clauses, schema understanding
- **Available capacity**: 1.5B parameters with 4-bit quantization
- **Comparison point**: Best-of-8 sampling shows model *can* generate correct repairs but needs better selection

**Evidence**: The 14% headroom (18% → 32% with best-of-8) proves the underlying model has the capability—the issue is **selection**, not **generation capacity**.

## Alternative Approach: Why ReST/RAFT Works Better

### Rejection Sampling Fine-Tuning (ReST/RAFT) Advantages:

1. **Leverages existing capability**: Uses the 14% headroom we identified
2. **Supervised learning**: More stable than RL for structured tasks  
3. **No critic needed**: Eliminates value function approximation problems
4. **Execution-based filtering**: Direct optimization of the actual objective
5. **Iterative improvement**: Gradual refinement without policy drift risk

### Algorithm:
```
1. Sample N completions from SFT model (N=8 works well)
2. Filter for execution-correct samples  
3. Fine-tune on filtered data
4. Repeat for multiple iterations
```

This directly optimizes execution accuracy without the RL complexity overhead.

## Implications for Future Work

### 1. **When to Use RL for SQL Tasks**
- **Dense reward signals** available (e.g., partial execution feedback)
- **Large model capacity** (>7B parameters) 
- **Complex multi-step reasoning** where exploration helps
- **Continuous optimization** spaces

### 2. **When to Prefer Supervised Selection**
- **Sparse/binary rewards** (like execution success)
- **Deterministic targets** (correct SQL has specific structure)
- **Single-step decisions** (clause-level repairs)
- **Limited model capacity** relative to task complexity

### 3. **Hybrid Approaches**
- Use **RL for generation strategy** (when to repair vs. regenerate)
- Use **supervised selection for implementation** (which repair to choose)
- Combine **best-of-N with learned ranking** functions

## Conclusion

PPO's failure on clause-level SQL repair isn't a flaw in the algorithm, but rather a fundamental **task-algorithm mismatch**. The structured, deterministic nature of SQL repair favors supervised selection methods over policy gradient approaches. This insight guides future work toward more appropriate algorithmic choices for structured code generation tasks.

## Paper Section Draft

*"While PPO achieved modest improvements over baseline approaches, our analysis reveals fundamental limitations when applying policy gradient methods to clause-level SQL repair. The sparse binary execution reward, combined with the deterministic nature of SQL syntax, creates a mismatch with RL's exploration-based learning paradigm. Our best-of-N analysis demonstrates 14% improvement headroom, suggesting that the bottleneck lies in **selection** rather than **generation capability**. These findings indicate that rejection sampling approaches (ReST/RAFT) may be more suitable for this class of structured prediction tasks, offering a path toward more efficient and stable improvements in SQL repair systems."*