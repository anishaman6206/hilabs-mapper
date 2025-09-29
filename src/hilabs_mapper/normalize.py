"""
Normalization with:
- minimal cleanup (lowercase, strip punctuation/extra spaces),
- dynamic acronym expansion from a provided acronym map.

Usage:
    norm = normalize_text("mri pelvis", acr_map)  # acr_map from load_or_build_acronym_map(...)
"""

from __future__ import annotations
import re
from functools import lru_cache
from typing import Optional, Dict
import os
import json

STOP_WORDS = {"a", "an", "and", "the", "for", "in", "of", "on", "with"}

# Global cache for the merged acronym map (avoids rebuilding per call)
_cached_combined_map: Dict[str, str] | None = None
_source_map_id: int | None = None

def ensure_cache_dir(base: str = "data/.cache") -> str:
    os.makedirs(base, exist_ok=True)
    return base

def load_cached_acronyms(path: str) -> Dict[str, str]:
    try:
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
    except Exception:
        pass
    return {}

def save_cached_acronyms(path: str, acr_map: Dict[str, str]) -> None:
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(acr_map, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"WARNING: could not save acronym cache to {path}: {e}")

def normalize_minimal(s: str) -> str:
    if not s:
        return ""
    s = s.lower().strip()
    s = re.sub(r"[^a-z0-9]+", " ", s)
    return re.sub(r"\s+", " ", s).strip()

@lru_cache(maxsize=1)
def _explicit_small_map() -> Dict[str, str]:
    # A few common expansions regardless of vocab
    return {
        "us": "ultrasound",
        "sono": "ultrasonography",
        "prom": "premature rupture of membranes",
        "ct": "computed tomography",
        "mri": "magnetic resonance imaging",
    }

def normalize_text(s: str, acr_map: Optional[Dict[str, str]] = None) -> str:
    """
    Normalize a string using minimal cleanup + optional acronym expansion.
    - acr_map: dict like {'mri': 'magnetic resonance imaging', ...}
    """
    global _cached_combined_map, _source_map_id

    try:
        if not isinstance(s, str):
            s = str(s)
        s = normalize_minimal(s)

        acr_map = acr_map or {}
        if id(acr_map) != _source_map_id:
            # Merge only when acr_map object changes
            merged = dict(_explicit_small_map())
            merged.update(acr_map)
            _cached_combined_map = merged
            _source_map_id = id(acr_map)

        final_map = _cached_combined_map or {}

        toks_in = s.split()
        n = len(toks_in)
        out = []
        for i, t in enumerate(toks_in):
            # keep stopwords in the middle to avoid weird deletions
            if 0 < i < n - 1 and t in STOP_WORDS:
                out.append(t)
                continue
            if t in final_map:
                out.extend(normalize_minimal(final_map[t]).split())
            else:
                out.append(t)

        return " ".join(out)

    except Exception:
        return ""  # fail-safe

def tokens(s: str, acr_map: Optional[Dict[str, str]] = None) -> set[str]:
    return set(normalize_text(s, acr_map=acr_map).split())
