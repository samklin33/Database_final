"""
PPO training loop for CLAUSE-PPO Phase 2.

Trains CodeLlama-7B via trl PPOTrainer to repair wrong SQL queries
at the clause level, combining ClausePRM dense rewards with NL2SQLEnv
terminal rewards.

Public API:
  build_rewrite_prompt   — formats the clause-rewrite prompt for the actor
  build_prm_prompt       — formats the scoring prompt for ClausePRM
  compute_reward         — combines terminal + alpha * prm_score
  get_corrupted_sample   — applies corruption engine to a Spider sample
  train_ppo              — main training entry point
"""

import os
import sys
import re

# ── Make sibling packages importable ─────────────────────────────────────────
_SRC = os.path.dirname(os.path.abspath(__file__))          # .../src/training
_CLAUSE_PPO_SRC = os.path.normpath(os.path.join(_SRC, '..'))  # .../src
if _CLAUSE_PPO_SRC not in sys.path:
    sys.path.insert(0, _CLAUSE_PPO_SRC)

from data.clause_splitter import CLAUSE_LABELS, split_into_clauses  # noqa: E402


def _extract_sql_robust(text: str) -> str:
    """
    Robust SQL extraction handling all wrapper formats seen in PPO training.
    
    Handles:
    - Numbered lists: "1. SELECT ... 2. SELECT ..."
    - Markdown formatting: "**SELECT** `column` FROM `table`"
    - Answer tags: "[ANSWER] SELECT ..."  
    - Mixed combinations of the above
    - Backticks, code blocks, and explanatory text
    """
    if not text or not text.strip():
        return ""
    
    original_text = text
    text = text.strip()
    
    # Strip markdown bold formatting: **SELECT** -> SELECT
    text = re.sub(r'\*\*(.*?)\*\*', r'\1', text)
    
    # Strip backticks from individual tokens: `column` -> column
    text = re.sub(r'`([^`]+)`', r'\1', text)
    
    # Extract from [ANSWER] tags
    answer_match = re.search(r'\[ANSWER\]\s*(.*?)(?:\n\n|\d+\.\s*\[|$)', text, re.DOTALL)
    if answer_match:
        text = answer_match.group(1).strip()
    
    # Extract from markdown code blocks
    code_block_match = re.search(r'```(?:sql)?\s*(.*?)```', text, re.DOTALL | re.IGNORECASE)
    if code_block_match:
        text = code_block_match.group(1).strip()
    
    # Strip initial numbered prefix: "1. SELECT" -> "SELECT"  
    text = re.sub(r'^\s*\d+\.\s*', '', text).strip()
    
    # Split by numbered continuations and take first part
    if re.search(r'\s+\d+\.\s*(?:SELECT|WITH|INSERT|UPDATE|DELETE)|;\s*\d+\.\s*(?:SELECT|WITH)', text, re.IGNORECASE):
        parts = re.split(r'\s+\d+\.\s*(?:SELECT|WITH|INSERT|UPDATE|DELETE)|;\s*\d+\.\s*(?:SELECT|WITH)', text, flags=re.IGNORECASE)
        text = parts[0].strip()
    
    # Find first line that looks like SQL (similar to deployment extractor)
    lines = text.split('\n')
    sql_candidate = ""
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
            
        # Skip explanatory text
        if any(phrase in line.lower() for phrase in [
            'here\'s', 'here is', 'the query', 'this query', 'you can use',
            'to determine', 'corrected sql', 'following sql', 'sql query:', 'therefore'
        ]):
            continue
            
        # Stop at explanatory text  
        if any(phrase in line.lower() for phrase in [
            'this will', 'this query will', 'the result', 'if you want'
        ]):
            break
            
        # Skip comments
        if line.startswith('#') or line.startswith('--') or line.startswith('//'):
            continue
            
        # If it looks like SQL, use it
        if re.search(r'\b(SELECT|INSERT|UPDATE|DELETE|WITH)\b', line, re.IGNORECASE):
            sql_candidate = line
            break
    
    # Use SQL candidate if found, otherwise use cleaned text
    if sql_candidate:
        text = sql_candidate
    
    # Final cleanup
    text = re.sub(r'[^\x00-\x7F]+', '', text)  # Remove non-ASCII
    text = re.sub(r'\s+', ' ', text)  # Normalize whitespace
    text = re.sub(r';\s*$', '', text).strip()  # Remove trailing semicolon
    
    # If extraction failed completely, return original text  
    if not text or len(text) < 5:
        return original_text.strip()
    
    return text


def assess_sql_quality(generated_text: str, env_terminal: float) -> dict:
    """
    Assess SQL generation quality for reward shaping.
    
    NOTE: Since we now apply deterministic regex stripping for numbered prefixes,
    this function should primarily focus on SQL structural quality.
    
    Returns dict with quality scores:
    - is_sql_like: bool (contains SQL keywords)  
    - is_single_query: bool (not multi-statement or explained)
    - is_executable: bool (env can parse it)
    - is_correct: bool (execution succeeds)
    """
    text = generated_text.strip()
    
    # Check if it looks like SQL (contains basic keywords)
    sql_keywords = ['SELECT', 'FROM', 'WHERE', 'GROUP', 'ORDER', 'HAVING', 'LIMIT', 'WITH']
    is_sql_like = any(keyword in text.upper() for keyword in sql_keywords)
    
    # FIXED: More robust single query detection (numbered prefixes already stripped)
    lines = [line.strip() for line in text.split('\n') if line.strip()]
    
    # Check for problematic patterns (after prefix stripping)
    has_explanation = any(word in text.lower() for word in ['identify', 'step', 'answer', 'explanation', 'because', 'therefore'])
    has_markdown = text.startswith(('```', '*', '-', '#'))
    has_multiple_statements = text.count(';') > 1  # More than one semicolon indicates multiple statements
    
    is_single_query = (
        is_sql_like and
        not has_explanation and
        not has_markdown and
        not has_multiple_statements and
        len(lines) <= 5  # Allow reasonable multi-line SQL
    )
    
    # Execution results from environment
    is_executable = env_terminal != -1.0  # env returns -1 for parse errors
    is_correct = env_terminal == 1.0      # env returns 1 for successful execution
    
    return {
        'is_sql_like': is_sql_like,
        'is_single_query': is_single_query, 
        'is_executable': is_executable,
        'is_correct': is_correct
    }


# ── Prompt builders ───────────────────────────────────────────────────────────

def build_rewrite_prompt(
    question: str,
    schema: str,
    wrong_sql: str,
    faulty_clause: str,
    clause_names: list[str],
) -> str:
    """
    Build the rewrite prompt for the PPO actor.

    Args:
        question:      Natural language question.
        schema:        Formatted DB schema string from schema_to_string().
        wrong_sql:     SQL string with the faulty clause.
        faulty_clause: Spider clause key of the wrong clause (e.g. 'where').
        clause_names:  All clause keys present in the SQL (from split_into_clauses).

    Returns:
        Prompt string ending with '[SQL]'; actor generates everything after that.
    """
    task_parts = []
    for name in clause_names:
        label = CLAUSE_LABELS.get(name, name.upper())
        if name == faulty_clause:
            task_parts.append(f"The {label} clause is wrong.")
        else:
            task_parts.append(f"The {label} clause is correct.")
    faulty_label = CLAUSE_LABELS.get(faulty_clause, faulty_clause.upper())
    task_parts.append(f"Rewrite the full SQL fixing only the {faulty_label} clause.")
    task_parts.append("Output only a single SQL query. No explanations, no numbering, no markdown. Start directly with SELECT or WITH.")

    return (
        f"[QUESTION] {question} "
        f"[SCHEMA] {schema} "
        f"[WRONG_SQL] {wrong_sql} "
        f"[TASK] {' '.join(task_parts)} "
        f"[SQL] "
    )


def build_prm_prompt(question: str, schema: str, clause_names_up_to_faulty: list[str]) -> str:
    """
    Build the ClausePRM scoring prompt for a clause.

    Matches PRMDataset prefix_query_str format: space-joined clause labels
    up to and including the faulty clause position (e.g. 'FROM WHERE' for
    position j=1 in [from, where, select] order).

    Args:
        clause_names_up_to_faulty: Ordered clause keys up to and including
            the faulty clause (e.g. ['from', 'where'] when where is faulty).
    """
    labels = [CLAUSE_LABELS.get(n, n.upper()) for n in clause_names_up_to_faulty]
    prefix_str = ' '.join(labels)
    return f"[QUESTION] {question} [SCHEMA] {schema} [PREFIX] {prefix_str}"


# ── Reward ────────────────────────────────────────────────────────────────────

def compute_reward(terminal: float, prm_score: float, alpha: float, execution_scale: float = 10.0, 
                   reward_shaping: str = "original", sql_quality: dict = None) -> float:
    """
    Clean PPO reward for Option B: Only execution correctness + PRM guidance.
    
    Format is handled by deterministic extraction, so no format penalties/bonuses.
    
    Args:
        terminal:        env.step() result: +1.0 (correct) or -1.0 (wrong).
        prm_score:       ClausePRM score for the rewritten clause, ∈ [0, 1].
        alpha:           Weight on the PRM score (from ppo_config.yaml).
        execution_scale: Scaling factor for execution reward.
        reward_shaping:  Strategy: only "original" is meaningful now.
        sql_quality:     Ignored (format handled deterministically).

    Returns:
        Clean reward: execution_scale * terminal + alpha * prm_score
    """
    # Simple, clean reward: execution correctness + PRM guidance
    # No format bonuses/penalties since extraction handles format deterministically
    execution_reward = terminal * execution_scale
    prm_guidance = alpha * prm_score
    
    return execution_reward + prm_guidance


# ── Episode helper ────────────────────────────────────────────────────────────

import random

from data.corruption import _CORRUPT_FNS          # noqa: E402
from utils.sql_utils import reconstruct_sql        # noqa: E402


def get_corrupted_sample(
    sample: dict,
    tables_dict: dict,
) -> tuple[str, str] | None:
    """
    Apply the corruption engine to produce a wrong SQL for one Spider sample.

    Calls the per-clause corrupt_* functions directly (not generate_corruptions)
    to avoid the hardcoded spider path in the orchestration function.
    Execution verification is skipped here — env.step() handles that at training time.

    Args:
        sample:      One Spider train entry with 'sql', 'db_id' fields.
        tables_dict: tables.json loaded as {db_id: tables_entry}.

    Returns:
        (wrong_sql_str, faulty_clause_key) or None if no corruption is possible.
    """
    sql_dict = sample.get('sql')
    if not sql_dict:
        return None

    tables = tables_dict.get(sample['db_id'])
    if tables is None:
        return None

    # Get clause names present in this SQL (execution order)
    clauses = split_into_clauses(sql_dict)
    clause_names = [name for name, _ in clauses]

    # Shuffle to avoid always corrupting the same clause
    shuffled = clause_names[:]
    random.shuffle(shuffled)

    for clause_name in shuffled:
        corrupt_fn = _CORRUPT_FNS.get(clause_name)
        if corrupt_fn is None:
            continue
        corrupted_dict = corrupt_fn(sql_dict, tables)
        if corrupted_dict is None:
            continue
        try:
            wrong_sql = reconstruct_sql(corrupted_dict, tables)
            return wrong_sql, clause_name
        except Exception:
            continue

    return None


# ── Main training loop ────────────────────────────────────────────────────────

def train_ppo(config: dict, spider_dir: str, prm_ckpt: str) -> list[dict]:
    """
    Full PPO training loop for clause-level NL2SQL repair.

    Args:
        config:     Parsed ppo_config.yaml as a nested dict.
        spider_dir: Path to the Spider dataset root (contains tables.json,
                    train_spider.json, dev.json, database/).
        prm_ckpt:   Path to the saved ClausePRM checkpoint directory.

    Returns:
        List of log entry dicts written during training.
    """
    import json
    import torch
    from peft import LoraConfig, get_peft_model, TaskType
    from trl  import PPOTrainer, PPOConfig, AutoModelForCausalLMWithValueHead, create_reference_model
    from transformers import AutoTokenizer, BitsAndBytesConfig

    # Make src/ (env, eval) importable
    _REPO_ROOT = os.path.normpath(os.path.join(_CLAUSE_PPO_SRC, '..', '..'))
    _ENV_SRC   = os.path.join(_REPO_ROOT, 'src')
    if _ENV_SRC not in sys.path:
        sys.path.insert(0, _ENV_SRC)
    from env.env import NL2SQLEnv

    from models.prm import ClausePRM

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    mcfg  = config['model']
    pcfg  = config['ppo']
    tcfg  = config['training']
    paths = config['paths']

    os.makedirs(paths['output_dir'], exist_ok=True)
    log_dir = os.path.dirname(paths['log_file'])
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)

    # ── Tokenizer ─────────────────────────────────────────────────────────────
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(mcfg['actor_name'], use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ── Actor with value head ─────────────────────────────────────────────────
    print("Loading actor model...")
    use_4bit = mcfg.get('quantization', '4bit') == '4bit'
    bnb_config = None
    if use_4bit:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type='nf4',
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )

    lora_cfg = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=mcfg['lora_rank'],
        lora_alpha=mcfg['lora_alpha'],
        lora_dropout=mcfg.get('lora_dropout', 0.05),
        target_modules=mcfg.get('target_modules', ['q_proj', 'v_proj']),
        bias='none',
    )
    # Create merged SFT model first (SFT adapter merged into base weights)
    # This ensures both actor and reference start from the actual SFT model, not base
    merged_sft_model = None
    if mcfg.get('sft_checkpoint'):
        print(f"Creating merged SFT model from {mcfg['sft_checkpoint']}...")
        # Resolve relative paths relative to the clause_ppo directory
        sft_checkpoint_path = mcfg['sft_checkpoint']
        if not os.path.isabs(sft_checkpoint_path):
            clause_ppo_root = os.path.normpath(os.path.join(_CLAUSE_PPO_SRC, '..'))
            sft_checkpoint_path = os.path.join(clause_ppo_root, sft_checkpoint_path)
        
        # Load SFT model and merge adapter into base weights
        from peft import PeftModel, AutoPeftModelForCausalLM
        try:
            # Load SFT model with adapter
            sft_model = AutoPeftModelForCausalLM.from_pretrained(
                sft_checkpoint_path,
                torch_dtype=torch.bfloat16,
                device_map='auto'
            )
            # Merge and unload to get base model + SFT weights combined
            merged_sft_model = sft_model.merge_and_unload()
            print("✅ SFT adapter merged into base weights successfully")
        except Exception as e:
            print(f"❌ Failed to load/merge SFT checkpoint: {e}")
            print("Falling back to base model")
            merged_sft_model = None
    
    # Initialize actor from merged SFT model (or base if SFT merge failed)
    if merged_sft_model is not None:
        print("🎯 Initializing PPO actor from merged SFT model...")
        actor = AutoModelForCausalLMWithValueHead.from_pretrained(
            merged_sft_model,  # Use merged model instead of base
            quantization_config=bnb_config,
            device_map='auto', 
            torch_dtype=torch.bfloat16,
            peft_config=lora_cfg,  # Fresh LoRA for PPO training
        )
    else:
        print("⚠️  Falling back to base model initialization...")
        actor = AutoModelForCausalLMWithValueHead.from_pretrained(
            mcfg['actor_name'],
            quantization_config=bnb_config,
            device_map='auto',
            torch_dtype=torch.bfloat16,
            peft_config=lora_cfg,
        )
    
    # Add generation_config if missing (TRL compatibility)
    if not hasattr(actor, 'generation_config') or actor.generation_config is None:
        from transformers import GenerationConfig
        actor.generation_config = GenerationConfig.from_pretrained(mcfg['actor_name'])

    # Reference model: Use merged SFT model as frozen reference
    # This ensures KL divergence is measured from SFT policy, not base model
    if merged_sft_model is not None:
        print("🎯 Creating reference model from merged SFT model...")
        ref_model = AutoModelForCausalLMWithValueHead.from_pretrained(
            merged_sft_model,  # Same merged model as actor
            device_map='auto',
            torch_dtype=torch.bfloat16,
            quantization_config=bnb_config,
        )
        ref_model.eval()
        for param in ref_model.parameters():
            param.requires_grad = False
        print("✅ Reference model initialized from SFT")
    else:
        print("⚠️  No merged SFT model, using base model as reference")
        ref_model = create_reference_model(actor)

    trainable = sum(p.numel() for p in actor.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in actor.parameters())
    print(f"Actor trainable params: {trainable:,} / {total:,}")
    
    # VERIFICATION: Test actor generation before training to ensure SFT init worked
    print("🔍 Verifying actor initialization with test generation...")
    test_prompt = "[QUESTION] What are the ship names? [SCHEMA] CREATE TABLE ship (Name TEXT, Tonnage INTEGER); [WRONG_SQL] SELECT COUNT(*) FROM ship [TASK] The SELECT clause is wrong. Rewrite the full SQL fixing only the SELECT clause. Output only a single SQL query. No explanations, no numbering, no markdown. Start directly with SELECT or WITH. [SQL] "
    
    try:
        with torch.no_grad():
            test_inputs = tokenizer.encode(test_prompt, return_tensors="pt").to(device)
            test_outputs = actor.generate(
                input_ids=test_inputs,
                max_new_tokens=64,
                temperature=0.1,  # Low temp for deterministic test
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
            )
            test_response = tokenizer.decode(test_outputs[0][len(test_inputs[0]):], skip_special_tokens=True).strip()
            print(f"🎯 Test generation: '{test_response}'")
            
            # Check if it looks like clean SFT output vs base model junk
            if test_response.upper().startswith('SELECT') and 'Name' in test_response and not re.search(r'\d+\.|markdown|\*\*', test_response):
                print("✅ Actor initialization successful - clean SQL generation detected")
            else:
                print("❌ Actor initialization may have failed - non-clean output detected")
                print("   This indicates SFT weights were not properly loaded into the actor")
    except Exception as e:
        print(f"⚠️  Test generation failed: {e}")
    
    print("="*80)

    # ── ClausePRM ─────────────────────────────────────────────────────────────
    print(f"Loading ClausePRM from {prm_ckpt}...")
    prm = ClausePRM(
        model_name=prm_ckpt,
        lora_rank=mcfg['lora_rank'],
        lora_alpha=mcfg['lora_alpha'],
        lora_dropout=mcfg.get('lora_dropout', 0.05),
        target_modules=mcfg.get('target_modules', ['q_proj', 'v_proj']),
        use_4bit=use_4bit,
    )
    prm.eval()

    # ── Spider data ───────────────────────────────────────────────────────────
    print("Loading Spider data...")
    with open(os.path.join(spider_dir, 'tables.json')) as f:
        tables_dict = {t['db_id']: t for t in json.load(f)}

    with open(os.path.join(spider_dir, 'train_spider.json')) as f:
        all_train = json.load(f)

    ppo_start   = tcfg.get('ppo_split_start', 4000)
    ppo_samples = all_train[ppo_start:]
    print(f"PPO split: {len(ppo_samples)} samples (train_spider[{ppo_start}:])")

    # ── Environment ───────────────────────────────────────────────────────────
    env = NL2SQLEnv(spider_dir=spider_dir, tables=tables_dict)

    # ── trl PPOTrainer ────────────────────────────────────────────────────────
    ppo_config_obj = PPOConfig(
        learning_rate=pcfg['learning_rate'],
        batch_size=pcfg['batch_size'],
        mini_batch_size=pcfg['mini_batch_size'],
        gradient_accumulation_steps=pcfg['gradient_accumulation_steps'],
        ppo_epochs=pcfg['ppo_epochs'],
        init_kl_coef=pcfg['kl_coef'],
        max_grad_norm=tcfg['max_grad_norm'],
        log_with=None,  # Enable stats logging
    )
    # Create dummy dataset for PPOTrainer initialization
    from datasets import Dataset
    dummy_dataset = Dataset.from_dict({"input_ids": [[1, 2, 3]], "query": ["dummy"]})
    
    ppo_trainer = PPOTrainer(
        config=ppo_config_obj,
        model=actor,
        ref_model=ref_model,
        tokenizer=tokenizer,
        dataset=dummy_dataset,
    )

    # ── Episode loop ──────────────────────────────────────────────────────────
    log_entries: list[dict] = []
    trained_episodes = 0
    num_episodes = tcfg['num_episodes']

    print(f"\nStarting PPO training: {num_episodes} episodes")

    from tqdm import tqdm
    for ep_idx in tqdm(range(num_episodes), desc="PPO Training", ncols=100):
        sample = ppo_samples[ep_idx % len(ppo_samples)]

        # Step 1: Get a corruption of this sample
        corruption = get_corrupted_sample(sample, tables_dict)
        if corruption is None:
            continue
        wrong_sql, faulty_clause = corruption

        # Step 2: Reset environment — sets up gold SQL and schema
        state = env.reset(sample)

        # Step 3: Build rewrite prompt
        clauses      = split_into_clauses(sample['sql'])
        clause_names = [name for name, _ in clauses]
        prompt = build_rewrite_prompt(
            state['question'], state['schema'],
            wrong_sql, faulty_clause, clause_names,
        )

        # Step 4: Actor generates rewritten SQL
        query_tensor = tokenizer.encode(prompt, return_tensors='pt').squeeze(0)
        
        # Debug: Log prompt characteristics for first few episodes
        if ep_idx < 5:
            print(f"  📝 Prompt length: {len(query_tensor)} tokens")
            print(f"  📝 Prompt preview: {prompt[-150:]}")
            print(f"  🎲 Generation params: temp={pcfg['temperature']}, max_tokens={pcfg['max_new_tokens']}")
        
        # Add stability checks for generation
        try:
            response_tensors = ppo_trainer.generate(
                [query_tensor],
                max_new_tokens=pcfg['max_new_tokens'],
                temperature=pcfg['temperature'],
                do_sample=True,
                top_p=0.95,  # nucleus sampling for stability
                top_k=50,    # limit vocabulary for stability
                pad_token_id=tokenizer.pad_token_id,
            )
        except RuntimeError as e:
            if "CUDA error" in str(e) or "probability tensor" in str(e):
                print(f"⚠️  Generation failed at episode {ep_idx}: {e}")
                print("   Skipping this episode and continuing...")
                continue
            else:
                raise e
        # Decode only the newly generated tokens (response_tensors includes the query)
        generated_ids = response_tensors[0][len(query_tensor):]
        raw_generation = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
        
        # Log raw generations at episode 0 and 200 for expert analysis
        if ep_idx in [0, 199] or ep_idx < 10:  # Episodes 0-9, and 200
            print(f"  📋 RAW generation (ep {ep_idx}): '{raw_generation}'")
        
        # ROBUST SQL EXTRACTION: Handle all wrapper formats seen in training
        rewritten_sql = _extract_sql_robust(raw_generation)
        
        
        # Log generated SQL for debugging (first 20 episodes and every 50th episode)
        if ep_idx < 20 or (ep_idx + 1) % 50 == 0:
            print(f"  🔍 Generated SQL: {rewritten_sql[:100]}{'...' if len(rewritten_sql) > 100 else ''}")
            if len(rewritten_sql.strip()) == 0:
                print(f"  ⚠️ Empty generation detected!")
            elif not rewritten_sql.strip().upper().startswith(('SELECT', 'WITH')):
                print(f"  ⚠️ Non-SQL generation detected: {rewritten_sql[:50]}...")
        
        # Additional debug logging for problematic generations
        if len(rewritten_sql.strip()) < 10 or not any(word in rewritten_sql.upper() for word in ['SELECT', 'FROM', 'WHERE', 'COUNT', 'SUM', 'AVG']):
            print(f"  🚨 Suspicious SQL (ep {ep_idx+1}): '{rewritten_sql}'")
            print(f"     Prompt was: {prompt[-100:]}")  # Show end of prompt

        # Step 5: Terminal reward from environment
        terminal, _ = env.step(rewritten_sql)
        
        # Step 5.5: Assess SQL quality for multi-tier rewards
        sql_quality = assess_sql_quality(rewritten_sql, terminal)
        
        # Debug: Show expected vs actual for first few episodes
        if ep_idx < 5:
            expected_sql = sample.get('query', 'Unknown')
            print(f"  🎯 Expected SQL: {expected_sql[:80]}{'...' if len(expected_sql) > 80 else ''}")
            print(f"  🤖 Generated SQL: {rewritten_sql[:80]}{'...' if len(rewritten_sql) > 80 else ''}")
            print(f"  ⚡ Execution result: {terminal} (1.0=success, -1.0=failure)")
            quality_str = f"sql_like={sql_quality['is_sql_like']}, single={sql_quality['is_single_query']}, exec={sql_quality['is_executable']}, correct={sql_quality['is_correct']}"
            print(f"  📊 SQL Quality: {quality_str}")

        # Step 6: Dense reward from ClausePRM
        # Build cumulative prefix up to and including faulty clause position
        try:
            faulty_idx = clause_names.index(faulty_clause)
        except ValueError:
            faulty_idx = len(clause_names) - 1
        clause_names_up_to_faulty = clause_names[:faulty_idx + 1]

        prm_prompt = build_prm_prompt(
            state['question'], state['schema'], clause_names_up_to_faulty
        )
        prm_inputs = tokenizer(
            prm_prompt,
            return_tensors='pt',
            truncation=True,
            max_length=512,
        )
        prm_inputs = {k: v.to(device) for k, v in prm_inputs.items()}
        with torch.no_grad():
            prm_score = prm(
                prm_inputs['input_ids'],
                prm_inputs['attention_mask'],
            ).item()

        # Step 7: Combined reward with enhanced execution signal + SQL quality
        reward_shaping = pcfg.get('reward_shaping', 'balanced')
        reward = compute_reward(terminal, prm_score, pcfg['alpha'], pcfg.get('execution_scale', 10.0), reward_shaping, sql_quality)

        # Step 8: PPO update
        stats = ppo_trainer.step(
            [query_tensor],
            [generated_ids],
            [torch.tensor(reward, dtype=torch.float32)],
        )

        trained_episodes += 1

        # ── Logging ───────────────────────────────────────────────────────────
        if (ep_idx + 1) % tcfg['log_every'] == 0:
            # Calculate running average for trend analysis
            recent_window = 20
            if len(log_entries) >= recent_window:
                recent_rewards = [e['reward'] for e in log_entries[-recent_window:]]
                recent_avg = sum(recent_rewards) / len(recent_rewards)
                reward_trend = recent_avg
                
                # Calculate success rate (more important than reward average)
                recent_terminals = [e['terminal'] for e in log_entries[-recent_window:]]
                success_rate = sum(1 for t in recent_terminals if t > 0) / len(recent_terminals)
            else:
                reward_trend = reward
                success_rate = 1.0 if terminal > 0 else 0.0
                
            # Extract PPO metrics from trainer stats
            kl_divergence = None
            policy_entropy = None
            value_loss = None
            
            # Get stats from latest PPO step
            if stats:
                # Debug: Print available stats on first few episodes
                if ep_idx < 3:
                    print(f"  🔍 PPO stats keys: {list(stats.keys())}")
                    print(f"  🔍 PPO stats sample: {dict(list(stats.items())[:3])}")
                
                try:
                    # Try multiple possible key names
                    kl_divergence = (stats.get('objective/kl') or 
                                   stats.get('ppo/loss/kl') or 
                                   stats.get('policy/kl') or
                                   stats.get('kl') or
                                   stats.get('objective/kl_coef'))
                    
                    policy_entropy = (stats.get('ppo/policy/entropy') or 
                                    stats.get('policy/entropy') or
                                    stats.get('entropy'))
                    
                    value_loss = (stats.get('ppo/loss/value') or 
                                stats.get('value_loss') or
                                stats.get('losses/value_loss'))
                except Exception as e:
                    print(f"  ⚠️ Failed to extract PPO stats: {e}")
            else:
                if ep_idx < 3:
                    print(f"  ⚠️ PPO trainer returned no stats (None)")
            
            entry = {
                'episode':       ep_idx + 1,
                'terminal':      terminal,
                'prm_score':     round(prm_score, 4),
                'reward':        round(reward, 4),
                'faulty_clause': faulty_clause,
                'reward_trend':  round(reward_trend, 4),
                'success_rate':  round(success_rate, 3),  # Track success rate
                'kl_divergence': round(kl_divergence, 4) if kl_divergence else None,  # Track KL
                'policy_entropy': round(policy_entropy, 4) if policy_entropy else None,  # Track entropy
                'value_loss': round(value_loss, 4) if value_loss else None,  # Track value function
                'sql_quality':   sql_quality,  # Include SQL quality metrics in log
            }
            log_entries.append(entry)
            with open(paths['log_file'], 'a') as f:
                f.write(json.dumps(entry) + '\n')
            
            # Enhanced console output with trend indicator
            status_emoji = "✅" if terminal > 0 else "❌"
            if len(log_entries) >= 2:
                prev_reward = log_entries[-2]['reward']
                trend_emoji = "📈" if reward > prev_reward else "📉" if reward < prev_reward else "➡️"
            else:
                trend_emoji = "🎬"
            
            # SQL Quality indicators for console
            sql_indicators = ""
            if sql_quality['is_sql_like']:
                sql_indicators += "🔤"  # SQL-like
            if sql_quality['is_single_query']:
                sql_indicators += "1️⃣"  # Single query
            if sql_quality['is_executable']:
                sql_indicators += "⚙️"  # Executable
            
            print(
                f"[Ep {ep_idx+1:>4}/{num_episodes}] "
                f"{status_emoji} "
                f"reward={reward:+.3f} {trend_emoji} "
                f"(avg={reward_trend:+.3f}) "
                f"success={success_rate:.1%} "  # Show success rate
                f"prm={prm_score:.3f} "
                f"exec={terminal:+.1f} "
                f"{sql_indicators} "
                f"{faulty_clause}"
            )

        # ── Progress Summary ──────────────────────────────────────────────────
        summary_every = 50  # Show progress every 50 episodes
        if (ep_idx + 1) % summary_every == 0 and len(log_entries) >= 10:
            recent_count = min(summary_every, len(log_entries))
            recent_entries = log_entries[-recent_count:]
            
            success_rate = sum(1 for e in recent_entries if e['terminal'] > 0) / len(recent_entries)
            avg_reward = sum(e['reward'] for e in recent_entries) / len(recent_entries)
            avg_prm = sum(e['prm_score'] for e in recent_entries) / len(recent_entries)
            
            print(f"\n📊 PROGRESS SUMMARY [Episodes {ep_idx+1-recent_count+1}-{ep_idx+1}]")
            print(f"   Success Rate: {success_rate:.1%} ({success_rate*recent_count:.0f}/{recent_count})")
            print(f"   Avg Reward: {avg_reward:+.3f}")  
            print(f"   Avg PRM: {avg_prm:.3f}")
            print(f"   Reward Shaping: {reward_shaping}")
            print("-" * 50)

        # ── Checkpoint ────────────────────────────────────────────────────────
        if (ep_idx + 1) % tcfg['eval_every'] == 0:
            ckpt_path = os.path.join(paths['output_dir'], f'ep_{ep_idx + 1}')
            os.makedirs(ckpt_path, exist_ok=True)
            ppo_trainer.model.save_pretrained(ckpt_path)
            tokenizer.save_pretrained(ckpt_path)
            print(f"✅ Checkpoint saved → {ckpt_path}")

    print(f"\nPPO training complete. {trained_episodes} trained episodes ({num_episodes} iterations).")
    return log_entries
