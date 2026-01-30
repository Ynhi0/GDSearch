from pathlib import Path
import json
import hashlib
from typing import Any, Dict, Optional
import pandas as pd
import logging


def compute_run_signature(config: Dict[str, Any]) -> str:
    """Compute a deterministic run signature (sha256) from a config dict.

    Uses stable JSON serialization (sorted keys) so the same logical config
    always produces the same signature.
    """
    try:
        # Ensure the object is JSON-serializable; ignore failures and coerce to string
        canonical = json.dumps(config, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    except Exception:
        canonical = json.dumps({k: str(v) for k, v in sorted(config.items())}, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    digest = hashlib.sha256(canonical.encode('utf-8')).hexdigest()
    return digest


def results_exist(results_dir: Path, signature: str) -> bool:
    """Check if `results/summary_quantitative.csv` contains a completed row for the signature.

    Treat missing file as "no results" (return False) and be defensive about parsing errors.
    We consider a row 'completed' if either a `completed` boolean column exists and is True,
    or a measurable final metric (e.g., `final_test_acc`, `final_loss`, `iters_to_thresh`) is present
    and not NaN for that row.
    """
    results_dir = Path(results_dir)
    # Resolve to top-level `results` if a per-experiment dir is given
    summary_path = None
    # Look for an ancestor explicitly named 'results'
    for p in [results_dir] + list(results_dir.parents):
        if p.name == 'results':
            summary_path = p / 'summary_quantitative.csv'
            break
    if summary_path is None:
        # Fallback to results_dir/summary_quantitative.csv
        summary_path = results_dir / 'summary_quantitative.csv'

    if not summary_path.exists():
        logging.debug("No summary file found at %s", summary_path)
        return False

    try:
        df = pd.read_csv(summary_path)
    except Exception as e:
        logging.warning("Could not read summary file %s: %s", summary_path, e)
        return False

    if 'run_signature' not in df.columns:
        logging.debug("Summary file %s does not contain 'run_signature' column", summary_path)
        return False

    hits = df[df['run_signature'] == signature]
    if hits.empty:
        logging.debug("No rows matching signature %s in %s", signature, summary_path)
        return False

    # Consider completed if any matching row has a completed==True or has a final metric
    for _, row in hits.iterrows():
        if 'completed' in row.index and pd.notna(row.get('completed')) and bool(row.get('completed')):
            return True
        # metrics to check
        for col in ('final_test_acc', 'final_loss', 'iters_to_thresh'):
            if col in row.index and pd.notna(row.get(col)):
                return True
    return False


def decide_resume_action(checkpoint: Optional[Dict[str, Any]], results_dir: Path, signature: str, resume_behavior: str):
    """Decide what to do given checkpoint presence and resume behavior.

    Returns one of: 'skip', 'restart'. Raises RuntimeError on error behavior.
    """
    # If checkpoint exists and is marked completed -> skip
    if checkpoint is not None:
        if isinstance(checkpoint, dict) and checkpoint.get('metadata', {}).get('completed', False):
            return 'skip'
        # otherwise resume from checkpoint (treat as restart action in caller semantics)
        return 'restart'

    # No checkpoint present: act based on resume_behavior
    if resume_behavior == 'skip_if_results_exist':
        if results_exist(results_dir, signature):
            return 'skip'
        return 'restart'
    elif resume_behavior == 'error_if_no_checkpoint':
        raise RuntimeError("No checkpoint exists for run and resume behavior is 'error_if_no_checkpoint'")
    elif resume_behavior == 'restart_if_no_checkpoint':
        return 'restart'
    else:
        raise ValueError(f"Unknown resume behavior: {resume_behavior}")
