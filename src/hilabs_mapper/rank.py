from __future__ import annotations
import numpy as np
import pandas as pd
from .normalize import normalize_text


# RxNorm TTY priorities (standard first)

_RX_TTY_ORDER = [
    "SCD", "SBD", "GPCK", "BPCK", "SCDC", "SCDG", "SCDF", "SCDGP", "SBDF", "SBDFP",
    "SBDC", "SBDG", "SCDFP", "MIN", "PIN", "IN", "PSN", "BN", "DF", "DFG", "SY", "TMSY", "ET"
]
RX_TTY_RANK = {t: i for i, t in enumerate(_RX_TTY_ORDER)}

# -
# SNOMED display order 
# -
_SNOMED_TTY_ORDER = ["PT", "SY", "FN"]
SNOMED_TTY_RANK = {t: i for i, t in enumerate(_SNOMED_TTY_ORDER)}

# STY gating per entity type 

ALLOWED_STY_BY_ENTITY = {
    "diagnosis": {
        "disease or syndrome",
        "mental or behavioral dysfunction",
        "neoplastic process",
        "anatomical abnormality",
        "acquired abnormality",
        "finding",
        "sign or symptom",
        "pathologic function",
        "injury or poisoning",
        "congenital abnormality",
        "cell or molecular dysfunction",
    },
    "procedure": {
        "therapeutic or preventive procedure",
        "diagnostic procedure",
        "laboratory procedure",
        "health care activity",
        "drug delivery device",
        "medical device",
        "clinical drug",
    },
    "lab": {
        "laboratory or test result",
        "laboratory procedure",
        "diagnostic procedure",
        "clinical attribute",
        "quantitative concept",
        "indicator, reagent, or diagnostic aid",
        "chemical viewed functionally",
        "chemical viewed structurally",
        "body substance",
    },
}

def _sty_allowed(entity_type: str, sty_value: str) -> bool:
    et = (entity_type or "").strip().lower()
    sty = (sty_value or "").strip().lower()
    allowed = ALLOWED_STY_BY_ENTITY.get(et)
    if not allowed or not sty:
        return True
    return sty in allowed


# Light similarity utilities

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
    # symmetric starts-with gives a small nudge to near-exact phrases
    return 0.12 if (sn.startswith(qn) or qn.startswith(sn)) else 0.0

def _mentions_contrast(q: str) -> bool:
    qn = normalize_text(q)
    keys = ["with contrast", "w contrast", "w/ contrast", "+c", "gad", "gadolinium", "contrast"]
    return any(k in qn for k in keys)

def _candidate_has_contrast(sn: str) -> bool:
    return "contrast" in sn

def _base_similarity(query: str, df: pd.DataFrame) -> np.ndarray:
    """
    Use BM25 retrieval_score if provided; else Jaccard over normalized tokens + prefix bonus.
    Returns np.ndarray of floats (len(df)).
    """
    if "retrieval_score" in df.columns:
        base = pd.to_numeric(df["retrieval_score"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
    else:
        qn = normalize_text(query)
        qt = set(qn.split())
        stoks = df["STR"].astype(str).map(lambda s: set(_tokenize_norm(s)))
        base = stoks.map(lambda t: _jaccard(qt, t)).to_numpy(dtype=float)

    # prefix bump
    qn = normalize_text(query)
    sn = df["STR"].astype(str).map(normalize_text).to_numpy()
    prefix = np.fromiter((_prefix_bonus(qn, s) for s in sn), count=len(df), dtype=float)
    return base + prefix

# Main ranker
def rank_candidates(query: str, entity_type: str, system: str, rows: pd.DataFrame) -> pd.DataFrame:
    """
    Rank retrieved candidates.
      - SNOMEDCT_US: STY gating drives correctness; tiny TTY (PT>SY>FN) nudge only as tie-breaker.
      - RXNORM: TTY priority (standard first) matters; bonus is modest so semantic score still leads.

    Inputs: rows must have ["CODE","STR","TTY"] and optionally "STY","retrieval_score".
    Output: same columns + 'final_score', sorted desc, no internal columns.
    """
    if rows is None or rows.empty:
        return pd.DataFrame(columns=["CODE", "STR", "TTY", "final_score"])

    df = rows.copy()

    # Ensure essential columns exist
    for col in ("CODE", "STR", "TTY"):
        if col not in df.columns:
            df[col] = ""

    sysu = system.upper()

    #  SNOMED: STY gating (critical) 
    if sysu == "SNOMEDCT_US" and "STY" in df.columns:
        mask = df["STY"].astype(str).map(lambda s: _sty_allowed(entity_type, s))
        # keep filtered set if any survive; else keep all to avoid empty outputs
        if mask.any():
            df = df[mask].reset_index(drop=True)

    #  Base similarity 
    df["__base__"] = _base_similarity(query, df)

    #  Imaging site & contrast heuristics (system-agnostic) 
    qn = normalize_text(query)
    q_tokens = set(qn.split())
    site_tokens = {t for t in q_tokens if t not in IMAGING_WORDS}

    df["__str_norm__"] = df["STR"].astype(str).map(normalize_text)
    df["__ctoks__"] = df["__str_norm__"].map(lambda s: set(s.split()))

    if site_tokens:
        overlap = df["__ctoks__"].map(lambda t: len(site_tokens & t))
        # hard demote items with zero site overlap (when a site/location was given)
        df.loc[overlap == 0, "__base__"] = df.loc[overlap == 0, "__base__"] - 1.0
        # soft bonus for matches (cap 2 tokens)
        df["__base__"] = df["__base__"] + overlap.clip(upper=2).astype(float) * 0.12

    want_contrast = _mentions_contrast(query)
    has_contrast = df["__str_norm__"].map(_candidate_has_contrast)
    if not want_contrast:
        # small penalty for "contrast" mentions unless user asked for it
        if site_tokens:
            site_overlap = df["__ctoks__"].map(lambda t: len(site_tokens & t))
            mask = has_contrast & (site_overlap > 0)
        else:
            mask = has_contrast
        df.loc[mask, "__base__"] = df.loc[mask, "__base__"] - 0.10

    #  System-specific TTY handling 
    if sysu == "RXNORM":
        # Modest TTY bonus so IN/BN/etc preference shows while semantics dominate
        tty_rank = df["TTY"].astype(str).map(lambda t: RX_TTY_RANK.get(t, len(RX_TTY_RANK) + 10)).astype(int)
        # 0 -> +0.30, 1 -> +0.26, ... floor at 0
        tty_bonus = np.maximum(0.0, 0.30 - 0.04 * tty_rank.to_numpy())
        df["__tty_rank__"] = tty_rank
        df["__tty_bonus__"] = tty_bonus
    else:
        # SNOMED: use a very small tie-breaker for PT>SY>FN, not a real ranking driver
        tty_rank = df["TTY"].astype(str).map(lambda t: SNOMED_TTY_RANK.get(t, 999)).astype(int)
        # 0 -> +0.02, 1 -> +0.01, else 0
        tty_bonus = tty_rank.map(lambda r: 0.02 if r == 0 else (0.01 if r == 1 else 0.0)).to_numpy(dtype=float)
        df["__tty_rank__"] = tty_rank
        df["__tty_bonus__"] = tty_bonus

    #  Final score & sorting 
    df["final_score"] = df["__base__"].astype(float) + df["__tty_bonus__"].astype(float)
    df["__len__"] = df["STR"].astype(str).map(len)

    if sysu == "RXNORM":
        # Prefer higher final score, then better tty rank, then shorter strings
        df = df.sort_values(
            by=["final_score", "__tty_bonus__", "__base__", "__len__"],
            ascending=[False, False, False, True],
            kind="mergesort",
        ).reset_index(drop=True)
    else:
        # SNOMED: semantics first; TTY only as last tie-breaker
        df = df.sort_values(
            by=["final_score", "__len__", "__tty_rank__"],
            ascending=[False, True, True],
            kind="mergesort",
        ).reset_index(drop=True)

    #  Clean internals & return 
    drop_cols = {"__base__", "__tty_rank__", "__tty_bonus__", "__str_norm__", "__ctoks__", "__len__"}
    cols = [c for c in df.columns if c not in drop_cols]
    return df[cols]
