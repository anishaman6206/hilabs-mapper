import os, json
from typing import Dict, List, Tuple
from collections import Counter, defaultdict
import pandas as pd

_CACHE_DIR = "data/.cache"
_ACRONYM_JSON = os.path.join(_CACHE_DIR, "acronym_map.json")

def _normalize_minimal(s: str) -> str:
    import re
    s = s.lower().strip()
    s = re.sub(r"[^a-z0-9]+", " ", s)
    return re.sub(r"\s+", " ", s).strip()

def _make_acronym(s: str) -> str:
    toks = [t for t in _normalize_minimal(s).split() if len(t) >= 3]
    return "".join(t[0] for t in toks) if toks else ""

def build_acronym_map_from_df(
    df_vocab: pd.DataFrame,
    min_support: int = 3,
    cap: int = 50000,
) -> Dict[str, str]:
    """
    Build acronym -> most frequent STR expansion map with sensible gates:
      - acronym alphabetic
      - expansion has >=2 tokens
      - seen at least `min_support` times
      - keep top `cap`
    Saves JSON to `data/.cache/acronym_map.json`
    """
    os.makedirs(_CACHE_DIR, exist_ok=True)

    # If cached version exists, just load it
    if os.path.exists(_ACRONYM_JSON):
        with open(_ACRONYM_JSON, "r", encoding="utf-8") as f:
            return json.load(f)

    # Otherwise build from vocab
    bag: Dict[str, Counter] = defaultdict(Counter)
    for s in df_vocab["STR"].astype(str):
        ac = _make_acronym(s)
        if not ac or not ac.isalpha():
            continue
        norm = _normalize_minimal(s)
        if len(norm.split()) < 2:
            continue
        bag[ac][s] += 1

    rows: List[Tuple[str, str, int]] = []
    for ac, counts in bag.items():
        total = sum(counts.values())
        if total >= min_support:
            best_str, _ = counts.most_common(1)[0]
            rows.append((ac, best_str, total))

    rows.sort(key=lambda t: -t[2])
    rows = rows[:cap]
    acr_map = {ac: best for ac, best, _ in rows}

    # Save to cache
    with open(_ACRONYM_JSON, "w", encoding="utf-8") as f:
        json.dump(acr_map, f, ensure_ascii=False, indent=2)

    return acr_map
