from __future__ import annotations
import pandas as pd

# STY allowlist by Entity Type (lowercased) 
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

# TTY priority maps 
_RX_TTY_ORDER = [
    "SCD", "SBD", "GPCK", "BPCK", "SCDC", "SCDG", "SCDF", "SCDGP", "SBDF", "SBDFP",
    "SBDC", "SBDG", "SCDFP", "MIN", "PIN", "IN", "PSN", "BN", "DF", "DFG", "SY", "TMSY", "ET"
]
_RX_TTY_RANK = {t: i for i, t in enumerate(_RX_TTY_ORDER)}

_SNOMED_TTY_ORDER = ["PT", "SY", "FN"]
_SNOMED_TTY_RANK = {t: i for i, t in enumerate(_SNOMED_TTY_ORDER)}

def _rank_map(system: str) -> dict[str, int]:
    return _RX_TTY_RANK if system.upper() == "RXNORM" else _SNOMED_TTY_RANK

def apply_display_for_candidates(
    system: str,
    ranked_df: pd.DataFrame,
    full_vocab_df: pd.DataFrame,
    entity_type: str | None = None,
) -> pd.DataFrame:
    """
    For each CODE in ranked_df, select exactly one display row from full_vocab_df using:
      1) (SNOMED only) STY allowlist based on entity_type
      2) TTY priority 
      3) Shortest STR tie-break
    Merge chosen STR/TTY back into ranked_df. Never throws if nothing matches.
    """
    if ranked_df is None or ranked_df.empty:
        return ranked_df

    out = ranked_df.copy()
    for c in ("CODE", "STR", "TTY"):
        if c not in out.columns:
            out[c] = ""

    # string join keys
    out["CODE"] = out["CODE"].astype(str)

    need_cols = ["CODE", "TTY", "STR"]
    if system.upper() == "SNOMEDCT_US":
        need_cols.append("STY")

    vocab = full_vocab_df.loc[:, need_cols].copy()
    vocab["CODE"] = vocab["CODE"].astype(str)
    vocab["TTY"] = vocab["TTY"].astype(str)
    vocab["STR"] = vocab["STR"].astype(str)
    if "STY" in vocab.columns:
        vocab["STY"] = vocab["STY"].astype(str).str.lower()

    codes = pd.unique(out["CODE"])
    vsub = vocab[vocab["CODE"].isin(codes)].copy()
    if vsub.empty:
        return out

    # Priority & tie-break
    rankmap = _rank_map(system)
    vsub["_tty_rank"] = vsub["TTY"].map(lambda t: rankmap.get(t, 10_000)).astype(int)
    vsub["_len"] = vsub["STR"].str.len().fillna(9999).astype(int)
    # Large gap between tiers so TTY strictly dominates
    vsub["_score"] = vsub["_tty_rank"] * 1_000_000 + vsub["_len"]

    # SNOMED STY allowlist 
    allowed_sty = None
    if system.upper() == "SNOMEDCT_US":
        et = (entity_type or "").strip().lower()
        allowed_sty = ALLOWED_STY_BY_ENTITY.get(et)

    picks = []
    for code, grp in vsub.groupby("CODE", sort=False):
        if allowed_sty is not None and "STY" in grp.columns:
            g2 = grp[grp["STY"].isin(allowed_sty)]
            if not g2.empty:
                grp = g2
        # pick best by score
        picks.append(grp["_score"].idxmin())

    disp = vsub.loc[picks, ["CODE", "TTY", "STR"]].rename(
        columns={"TTY": "TTY_disp", "STR": "STR_disp"}
    )

    out = out.merge(disp, on="CODE", how="left")
    out["TTY"] = out["TTY_disp"].fillna(out["TTY"])
    out["STR"] = out["STR_disp"].fillna(out["STR"])
    out.drop(columns=[c for c in ("TTY_disp", "STR_disp") if c in out.columns],
             inplace=True, errors="ignore")
    for c in ("_tty_rank", "_len", "_score"):
        if c in out.columns:
            out.drop(columns=[c], inplace=True)

    return out