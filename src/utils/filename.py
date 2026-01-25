from typing import Dict, List, Optional
import re

COMMON_OPTS = ['sgd', 'adam', 'adamw', 'amsgrad', 'sam', 'lookahead', 'radam', 'lamb', 'adabound', 'rmsprop', 'nesterov']


def parse_experiment_filename(stem: str) -> Dict[str, Optional[object]]:
    """Parse an experiment filename stem into components.

    Returns dict with keys: optimizer (str|None), seed (int|None), lr (float|None), extras (list)
    """
    if not stem:
        return {'optimizer': None, 'seed': None, 'lr': None, 'extras': []}

    orig = stem
    s = stem.replace('-', '_')
    parts = [p for p in s.split('_') if p]

    optimizer = None
    seed = None
    lr = None
    extras: List[str] = []

    # seed
    for p in parts:
        m = re.match(r'(?i)seed[_-]?(\d+)$', p)
        if m:
            seed = int(m.group(1))
            continue
        m = re.match(r'(?i)seed(\d+)$', p)
        if m:
            seed = int(m.group(1))
            continue

    # lr
    for p in parts:
        m = re.match(r'(?i)lr[_-]?(\d*\.?\d+)$', p)
        if m:
            try:
                lr = float(m.group(1))
            except Exception:
                lr = None
            continue

    # optimizer: prefer explicit compound tokens like SGD_Momentum when present
    # First pass: look for explicit 'momentum' compounds
    for i in range(len(parts)-1):
        if parts[i+1].lower() == 'momentum':
            optimizer = f"{parts[i]}_{parts[i+1]}"
            break

    # Second pass: general compound and single-token matching
    if not optimizer:
        for i in range(len(parts)-1):
            candidate = f"{parts[i]}_{parts[i+1]}"
            low = candidate.lower()
            for opt in COMMON_OPTS:
                if opt in low:
                    optimizer = candidate
                    break
            if optimizer:
                break

        if not optimizer:
            # look for single token matches
            for p in parts:
                low = p.lower()
                for opt in COMMON_OPTS:
                    if opt in low:
                        optimizer = p
                        break
                if optimizer:
                    break

    if not optimizer:
        # look for single token matches
        for p in parts:
            low = p.lower()
            for opt in COMMON_OPTS:
                if opt in low:
                    optimizer = p
                    break
            if optimizer:
                break

    # extras = tokens not part of optimizer/seed/lr
    for p in parts:
        if optimizer and p in optimizer.split('_'):
            continue
        if seed is not None and re.search(r'(?i)seed', p):
            continue
        if lr is not None and re.search(r'(?i)^lr', p):
            continue
        extras.append(p)

    # Reduce compound names to the core optimizer tokens (e.g., ResNet18_Adam -> Adam)
    if optimizer:
        parts = optimizer.split('_')
        suffixes = {'momentum'}
        matched = []
        for token in parts:
            low = token.lower()
            if any(opt in low for opt in COMMON_OPTS) or low in suffixes:
                matched.append(token)
        if matched:
            optimizer = '_'.join(matched)

    return {'optimizer': optimizer, 'seed': seed, 'lr': lr, 'extras': extras, 'orig': orig}


def parse_opt_seed_from_stem(stem: str):
    """Return (optimizer, seed) parsed from a filename stem.

    Examples:
        >>> parse_opt_seed_from_stem('CIFAR10_ResNet18_Adam_seed42')
        ('Adam', 42)
        >>> parse_opt_seed_from_stem('NN_ResNet18_CIFAR10_Adam_lr0.001_seed42')
        ('Adam', 42)
    """
    import re
    parsed = parse_experiment_filename(stem)
    opt = parsed.get('optimizer')
    seed = parsed.get('seed')

    # Handle cases like 'seed_123' where tokens split into 'seed' and '123'
    if seed is None and stem:
        m = re.search(r'(?i)seed[^0-9]*(\d+)', stem)
        if m:
            try:
                seed = int(m.group(1))
            except Exception:
                seed = None

    # Normalize optimizer casing to human-friendly token when possible
    if isinstance(opt, str):
        opt = opt.replace('-', '_')
    return opt, seed
