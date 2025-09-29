from __future__ import annotations
import numpy as np
import pandas as pd
from .normalize import normalize_text

# TTY priorities (from most to least preferred)
_RX_TTY_ORDER = [
    "SCD", "SBD", "GPCK", "BPCK", "SCDC", "SCDG", "SCDF", "SCDGP", "SBDF", "SBDFP",
    "SBDC", "SBDG", "SCDFP", "MIN", "PIN", "IN", "PSN", "BN", "DF", "DFG", "SY", "TMSY", "ET"
]

RX_TTY_RANK = {t: i for i, t in enumerate(_RX_TTY_ORDER)}

_SNOMED_TTY_ORDER = ["PT", "SY", "FN"]
SNOMED_TTY_RANK = {t: i for i, t in enumerate(_SNOMED_TTY_ORDER)}

def _tty_bonus_from_rank(rank: int, system: str) -> float:
    """
    A gentle monotone bonus so better TTYs sort above worse ones.
    Kept small so semantic match always dominates.
    """
    if system.upper() == "RXNORM":
        # 0 -> +0.24, 1 -> +0.20, ... floor at 0
        return max(0.0, 0.24 - 0.04 * int(rank))
    # SNOMED: PT>SY>FN (0.18, 0.06, 0.0)
    return {0: 0.18, 1: 0.06}.get(int(rank), 0.0)

# token helpers for site-specific imaging
IMAGING_WORDS = {
    "mri","magnetic","resonance","ct","ultrasound","us",
    "x","xray","x-ray","xr","scan","imaging","tomography"
}

def _tokenize_norm(s: str) -> list[str]:
    return normalize_text(s).split()

def _jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    inter = len(a & b)
    if inter == 0:
        return 0.0
    return inter / float(len(a | b))

def _prefix_bonus(qn: str, sn: str) -> float:
    # symmetric starts-with
    return 0.12 if (sn.startswith(qn) or qn.startswith(sn)) else 0.0

# base similarity
def _base_similarity(query: str, df: pd.DataFrame) -> np.ndarray:
    """
    Use BM25 retrieval_score if provided; else Jaccard over normalized tokens + prefix bonus.
    Returns a numpy array of floats (same length as df).
    """
    if "retrieval_score" in df.columns:
        base = pd.to_numeric(df["retrieval_score"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
    else:
        qn = normalize_text(query)
        qt = set(qn.split())
        stoks = df["STR"].astype(str).map(lambda s: set(_tokenize_norm(s)))
        base = stoks.map(lambda t: _jaccard(qt, t)).to_numpy(dtype=float)

    # prefix bump (helpful for near-exact strings)
    qn = normalize_text(query)
    sn = df["STR"].astype(str).map(normalize_text).to_numpy()
    prefix = np.fromiter((_prefix_bonus(qn, s) for s in sn), count=len(df), dtype=float)
    return base + prefix

# contrast helpers
def _mentions_contrast(q: str) -> bool:
    qn = normalize_text(q)
    keys = ["with contrast", "w contrast", "w/ contrast", "+c", "gad", "gadolinium", "contrast"]
    return any(k in qn for k in keys)

def _candidate_has_contrast(sn: str) -> bool:
    return "contrast" in sn

#  main ranker function
def rank_candidates(query: str, entity_type: str, system: str, rows: pd.DataFrame) -> pd.DataFrame:
    """
    Rank retrieved candidates. No STY gating and no alias building here.
    Inputs: rows needs at least ["CODE","STR","TTY"]; "retrieval_score" optional.
    Output: same columns as input + 'final_score', sorted desc.
    """
    if rows is None or rows.empty:
        return pd.DataFrame(columns=["CODE", "STR", "TTY", "final_score"])

    df = rows.copy()

    # Ensures essential columns exist 
    for col in ("CODE", "STR", "TTY"):
        if col not in df.columns:
            df[col] = ""

    # Base score
    df["__base__"] = _base_similarity(query, df)

    # Site guard (keeps generic imaging below site-specific when site given)
    qn = normalize_text(query)
    q_tokens = set(qn.split())
    site_tokens = {t for t in q_tokens if t not in IMAGING_WORDS}
    df["__str_norm__"] = df["STR"].astype(str).map(normalize_text)
    df["__ctoks__"] = df["__str_norm__"].map(lambda s: set(s.split()))
    if site_tokens:
        overlap = df["__ctoks__"].map(lambda t: len(site_tokens & t))
        # hard demote items with zero site overlap
        df.loc[overlap == 0, "__base__"] = df.loc[overlap == 0, "__base__"] - 1.0
        # soft bonus for matches (cap 2 tokens to avoid huge swings)
        df["__base__"] = df["__base__"] + overlap.clip(upper=2).astype(float) * 0.12

    # Contrast handling
    want_contrast = _mentions_contrast(query)
    has_contrast = df["__str_norm__"].map(_candidate_has_contrast)
    if not want_contrast:
        # small penalty if candidate mentions contrast (only if we already matched some site tokens, if any)
        if site_tokens:
            site_overlap = df["__ctoks__"].map(lambda t: len(site_tokens & t))
            mask = has_contrast & (site_overlap > 0)
        else:
            mask = has_contrast
        df.loc[mask, "__base__"] = df.loc[mask, "__base__"] - 0.10

    # TTY preference 
    sysu = system.upper()
    if sysu == "RXNORM":
        tty_rank = df["TTY"].astype(str).map(lambda t: RX_TTY_RANK.get(t, len(RX_TTY_RANK)+10)).astype(int)
    else:
        tty_rank = df["TTY"].astype(str).map(lambda t: SNOMED_TTY_RANK.get(t, len(SNOMED_TTY_RANK)+10)).astype(int)
    df["__tty_rank__"] = tty_rank
    df["__tty_bonus__"] = df["__tty_rank__"].map(lambda r: _tty_bonus_from_rank(int(r), sysu))

    # Final score
    df["final_score"] = df["__base__"].astype(float) + df["__tty_bonus__"].astype(float)

    # Sort & clean
    # stable tie-breakers: better tty rank, then shorter STR
    df["__len__"] = df["STR"].astype(str).map(len)
    df = df.sort_values(
        by=["final_score", "__tty_bonus__", "__base__", "__len__"],
        ascending=[False, False, False, True],
        kind="mergesort",
    ).reset_index(drop=True)

    # keep original visible columns + final_score
    drop_cols = {"__base__", "__tty_rank__", "__tty_bonus__", "__str_norm__", "__ctoks__", "__len__"}
    cols = [c for c in df.columns if c not in drop_cols]
    return df[cols]
