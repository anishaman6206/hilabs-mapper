import pandas as pd

REQUIRED = ["System", "CODE", "STR", "STY", "TTY"]

def load_vocab_parquet(path: str, system: str) -> pd.DataFrame:
    """
    Load a single-system vocabulary parquet and tag all rows with its system.
    Use this for either SNOMED or RxNorm 

    Args:
        path: Path to parquet file (e.g., data/snomed_all_data.parquet).
        system: "SNOMEDCT_US" or "RXNORM".

    Returns:
        DataFrame with columns: System, CODE, STR, STY, TTY (all as strings).
    """
    
    df = pd.read_parquet(path, columns=[c for c in REQUIRED if c != "System"])
    df = df.copy()
    df["System"] = system

    # enforce required columns & string types
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in {path}: {missing}")

    for c in REQUIRED:
        df[c] = df[c].astype(str)

    return df

def build_vocabs(snomed_df: pd.DataFrame, rxnorm_df: pd.DataFrame) -> dict:
    """
    Build a dict of per-system dataframes used by the pipeline.

    Returns:
        {
          "SNOMEDCT_US": <DataFrame>,
          "RXNORM": <DataFrame>
        }
    """
    return {
        "SNOMEDCT_US": snomed_df.reset_index(drop=True),
        "RXNORM": rxnorm_df.reset_index(drop=True),
    }
