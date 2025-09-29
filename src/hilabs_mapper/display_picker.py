from __future__ import annotations
import pandas as pd
import numpy as np

# TTY priority maps
_RX_TTY_ORDER = [
    "SCD", "SBD", "GPCK", "BPCK", "SCDC", "SCDG", "SCDF", "SCDGP", "SBDF", "SBDFP",
    "SBDC", "SBDG", "SCDFP", "MIN", "PIN", "IN", "PSN", "BN", "DF", "DFG", "SY", "TMSY", "ET"
]

_RX_TTY_RANK = {t:i for i,t in enumerate(_RX_TTY_ORDER)}

_SNOMED_TTY_ORDER = ["PT", "SY", "FN"]
_SNOMED_TTY_RANK = {t:i for i,t in enumerate(_SNOMED_TTY_ORDER)}

def _rank_map(system: str) -> dict:
    return _RX_TTY_RANK if system.upper() == "RXNORM" else _SNOMED_TTY_RANK

def apply_display_for_candidates(system: str,
                                 ranked_df: pd.DataFrame,
                                 full_vocab_df: pd.DataFrame) -> pd.DataFrame:
    """
    For each CODE in ranked_df, pick one display row from full_vocab_df
    using TTY priority (and then by shortest STR). Merge those STR/TTY
    back into ranked_df. Never returns NaNs for STR/TTY when a matching
    CODE exists in the vocab.
    """
    if ranked_df is None or ranked_df.empty:
        return ranked_df

    out = ranked_df.copy()
    # Ensure columns exist
    for c in ("CODE","STR","TTY"):
        if c not in out.columns:
            out[c] = ""

    # Coerce to string to avoid join miss
    out["CODE"] = out["CODE"].astype(str)
    vocab = full_vocab_df.loc[:, ["CODE","TTY","STR"]].copy()
    vocab["CODE"] = vocab["CODE"].astype(str)
    vocab["TTY"]  = vocab["TTY"].astype(str)
    vocab["STR"]  = vocab["STR"].astype(str)

    # Subset vocab to only needed codes
    codes = pd.unique(out["CODE"])
    vsub = vocab[vocab["CODE"].isin(codes)].copy()
    if vsub.empty:
        # nothing matched; just return original ranked_df
        return out

    # Rank by TTY priority then by shortest STR
    rankmap = _rank_map(system)
    vsub["_tty_rank"] = vsub["TTY"].map(lambda t: rankmap.get(t, 10_000)).astype(int)
    vsub["_len"] = vsub["STR"].str.len().fillna(9999).astype(int)

    # Build a scalar sort key so we can idxmin per CODE
    # (large gap between tty tiers so TTY strictly dominates)
    vsub["_score"] = vsub["_tty_rank"] * 1_000_000 + vsub["_len"]

    # Best display row per CODE
    idx = vsub.groupby("CODE", sort=False)["_score"].idxmin()
    disp = vsub.loc[idx, ["CODE","TTY","STR"]].rename(columns={"TTY":"TTY_disp","STR":"STR_disp"})

    # Merge back; if some code somehow missing, fall back to original STR/TTY
    out = out.merge(disp, on="CODE", how="left")
    out["TTY"] = out["TTY_disp"].fillna(out["TTY"])
    out["STR"] = out["STR_disp"].fillna(out["STR"])
    out.drop(columns=[c for c in ("TTY_disp","STR_disp") if c in out.columns], inplace=True, errors="ignore")
    # Clean internals if any leaked
    for c in ("_tty_rank","_len","_score"):
        if c in out.columns:
            out.drop(columns=[c], inplace=True)

    return out
