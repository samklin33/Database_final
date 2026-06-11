"""
Full-regeneration baseline for NL2SQL evaluation.

run_baseline(): generate full SQL from question + schema; retry up to
  ``max_retries`` if execution does not match gold.

Generation is injected as a ``generate_fn`` so the loop is backend-agnostic.
Any callable with this contract works:

    generate_fn(prompt: str) -> (sql_text: str, n_input_tokens: int, n_output_tokens: int)

"""

import os
import re
import sys
import time
from typing import Callable, Optional

# ── Make sibling packages importable ────────────────────────────────────────
_REPO_ROOT      = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_CLAUSE_PPO_SRC = os.path.join(_REPO_ROOT, 'clause_ppo', 'src')
_SRC_ROOT       = os.path.join(_REPO_ROOT, 'src')
for _p in (_CLAUSE_PPO_SRC, _SRC_ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from config import SPIDER_DIR, MAX_TOKENS, API_RETRIES, API_BACKOFF_SECS
from env.env import NL2SQLEnv


# ── Types ──────────────────────────────────────────────────────────────────

# prompt -> (sql_text, n_input_tokens, n_output_tokens)
GenerateFn = Callable[[str], tuple[str, int, int]]


# ── Public API ─────────────────────────────────────────────────────────────

def build_baseline_prompt(question: str, schema: str) -> str:
    """
    Full-regen prompt. Same [QUESTION] / [SCHEMA] header as
    build_rewrite_prompt() so input-token counts stay comparable.
    """
    return (
        f"[QUESTION] {question} "
        f"[SCHEMA] {schema} "
        f"[TASK] Generate the full SQL query. Generate ONLY the SQL query, no explanations or markdown. "
        f"[SQL] "
    )


def run_baseline(
    sample:      dict,
    generate_fn: GenerateFn,
    max_retries: int = 3,
    env:         Optional[NL2SQLEnv] = None,
    spider_dir:  str = SPIDER_DIR,
    tables:      Optional[dict] = None,
) -> dict:
    """
    Full-regen baseline for one Spider sample.

    Args:
        sample:      Spider sample with ``question``, ``db_id``, and
                     ``gold_sql`` / ``query``.
        generate_fn: callable ``prompt -> (sql, n_in, n_out)``. Build one with
                     make_hf_api_generate_fn() for the HF Inference API.
        max_retries: maximum generation attempts. Returns early on the first
                     attempt whose execution matches gold.
        env:         pre-instantiated NL2SQLEnv (recommended — avoids
                     re-loading tables.json on every call). One is constructed
                     from ``spider_dir`` / ``tables`` when None.
        spider_dir, tables: forwarded to NL2SQLEnv when ``env`` is None.

    Returns:
        {"predicted_sql": str, "token_cost": int, "attempts": int, "success": bool}
    """
    if env is None:
        env = NL2SQLEnv(spider_dir=spider_dir, tables=tables)
    state  = env.reset(sample)
    prompt = build_baseline_prompt(state['question'], state['schema'])

    predicted_sql = ''
    token_cost    = 0
    attempts      = 0
    success       = False

    for _ in range(max_retries):
        attempts += 1
        sql, n_in, n_out = generate_fn(prompt)
        predicted_sql = sql
        token_cost   += n_in + n_out

        reward, _ = env.step(sql)
        if reward > 0:
            success = True
            break

    return {
        'predicted_sql': predicted_sql,
        'token_cost':    token_cost,
        'attempts':      attempts,
        'success':       success,
    }


def make_hf_api_generate_fn(
    client,
    model:              str,
    max_tokens:         int = MAX_TOKENS,
    fallback_tokenizer=None,
    api_retries:        int = API_RETRIES,
    backoff_secs:       float = API_BACKOFF_SECS,
) -> GenerateFn:
    """
    Adapt a huggingface_hub InferenceClient to the generate_fn contract.

    Args:
        client:             a huggingface_hub.InferenceClient.
        model:              model id, e.g. 'Qwen/Qwen2.5-Coder-1.5B-Instruct:featherless-ai'.
        max_tokens:         max generated tokens per call.
        fallback_tokenizer: optional HF tokenizer used only when the API
                            response carries no usage stats.
        api_retries:        attempts per call before giving up on transient errors.
        backoff_secs:       base for exponential backoff (base * 2**attempt).

    Returns:
        generate_fn(prompt) -> (sql_text, n_input_tokens, n_output_tokens).
    """
    def _call_once(prompt: str) -> tuple[str, int, int]:
        completion = client.chat.completions.create(
            model=model,
            messages=[{'role': 'user', 'content': prompt}],
            max_tokens=max_tokens,
        )
        raw_text = completion.choices[0].message.content or ''
        n_in, n_out = _usage_tokens(completion, prompt, raw_text, fallback_tokenizer)
        return _extract_sql(raw_text), n_in, n_out

    def generate_fn(prompt: str) -> tuple[str, int, int]:
        for attempt in range(api_retries):
            try:
                return _call_once(prompt)
            except Exception as exc:
                if not _is_retryable(exc) or attempt == api_retries - 1:
                    raise
                wait = backoff_secs * (2 ** attempt)
                print(f"  transient API error ({_err_label(exc)}); "
                      f"retry {attempt + 1}/{api_retries - 1} in {wait:.0f}s")
                time.sleep(wait)
        # Unreachable: the loop either returns or raises.
        raise RuntimeError("retry loop exited without returning")

    return generate_fn


# ── Helpers ────────────────────────────────────────────────────────────────

_RETRYABLE_STATUS = {429, 500, 502, 503, 504}
_RETRYABLE_EXC_NAMES = {
    'TimeoutException', 'ConnectTimeout', 'ReadTimeout', 'WriteTimeout',
    'PoolTimeout', 'ConnectError', 'RemoteProtocolError',
}


def _http_status(exc: Exception):
    """Best-effort HTTP status code from an exception, or None."""
    return getattr(getattr(exc, 'response', None), 'status_code', None)


def _is_retryable(exc: Exception) -> bool:
    """True for transient API failures (5xx / 429 / connection timeouts)."""
    status = _http_status(exc)
    if status is not None:
        return status in _RETRYABLE_STATUS
    return exc.__class__.__name__ in _RETRYABLE_EXC_NAMES


def _err_label(exc: Exception) -> str:
    status = _http_status(exc)
    return f"HTTP {status}" if status is not None else exc.__class__.__name__


_FENCE_PATTERN = re.compile(r'```(?:sql)?\s*(.*?)```', re.DOTALL | re.IGNORECASE)


def _extract_sql(text: str) -> str:
    """
    Pull SQL out of a chat model's reply.
    """
    text = text.strip()
    match = _FENCE_PATTERN.search(text)
    if match:
        return match.group(1).strip()
    return text


def _usage_tokens(
    completion,
    prompt: str,
    output_text: str,
    fallback_tokenizer,
) -> tuple[int, int]:
    """
    Resolve (input_tokens, output_tokens) for one API call.
    """
    usage = getattr(completion, 'usage', None)
    if usage is not None:
        n_in  = getattr(usage, 'prompt_tokens', None)
        n_out = getattr(usage, 'completion_tokens', None)
        if n_in is not None and n_out is not None:
            return n_in, n_out

    if fallback_tokenizer is not None:
        return (
            len(fallback_tokenizer.encode(prompt)),
            len(fallback_tokenizer.encode(output_text)),
        )

    raise RuntimeError(
        "API response carries no usage stats and no fallback_tokenizer was "
        "provided; cannot compute token_cost. Pass fallback_tokenizer="
        "AutoTokenizer.from_pretrained(model) to make_hf_api_generate_fn()."
    )
