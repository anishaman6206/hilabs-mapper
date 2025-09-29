from __future__ import annotations

import os
import time
import argparse
import pandas as pd
from tqdm import tqdm

from src.hilabs_mapper.loader import load_vocab_parquet, build_vocabs
from src.hilabs_mapper.retrieve import build_retrievers
from src.hilabs_mapper.rank import rank_candidates
from src.hilabs_mapper.display_picker import apply_display_for_candidates

DEFAULT_INPUT_XLSX = os.path.join("data", "test.xlsx")
DEFAULT_OUTPUT_CSV = "test.csv"
SNOMED = "SNOMEDCT_US"
RXNORM = "RXNORM"


def infer_system(entity_type: str) -> str:
    et = (entity_type or "").strip().lower()
    if et in {"medicine"}:
        return RXNORM
    # everything else maps to SNOMED by default (Procedure / Diagnosis / Lab)
    return SNOMED


def main():
    parser = argparse.ArgumentParser(description="End-to-end BM25 mapping pipeline")
    parser.add_argument("--input", default=DEFAULT_INPUT_XLSX, help="Path to input Excel/CSV (default: data/test.xlsx)")
    parser.add_argument("--sheet", default=None, help="Optional sheet name if the input is an Excel file")
    parser.add_argument("--outcsv", default=DEFAULT_OUTPUT_CSV, help="Output CSV in project root (default: test.csv)")
    parser.add_argument("--k", type=int, default=25, help="BM25 top-k per query (default: 25)")
    args = parser.parse_args()

    t0 = time.time()
    print("Loading vocabularies (SNOMEDCT_US, RXNORM)…")
    snomed = load_vocab_parquet(os.path.join("data", "snomed_all_data.parquet"), system=SNOMED)
    rxnorm = load_vocab_parquet(os.path.join("data", "rxnorm_all_data.parquet"), system=RXNORM)
    vocabs = build_vocabs(snomed, rxnorm)
    full_vocab = pd.concat([snomed, rxnorm], ignore_index=True, copy=False)

    # Build BM25 retrievers (acronym-aware internally) 
    print("Building BM25 retrievers… (one-time indexing) Wait for 30-60 seconds…")
    t_build = time.time()
    retrievers = build_retrievers(vocabs, full_vocab)
    print(f"Retrievers ready. Build time: {time.time() - t_build:.2f}s")

    # Load input (Excel or CSV) 
    if not os.path.exists(args.input):
        raise FileNotFoundError(f"Input file not found: {args.input}")

    ext = os.path.splitext(args.input)[1].lower()
    if ext in (".xlsx", ".xls"):
        df = pd.read_excel(args.input, sheet_name=args.sheet)
    elif ext == ".csv":
        df = pd.read_csv(args.input)
    else:
        raise ValueError(f"Unsupported input file type: {args.input}")

    if not isinstance(df, pd.DataFrame):
        if isinstance(df, dict) and df:
            df = next(iter(df.values()))
        else:
            raise ValueError("Failed to read a DataFrame from the input file.")

    # locate input columns (case-insensitive)
    col_map = {c.lower(): c for c in df.columns}
    text_col = col_map.get("input entity description") or col_map.get("entity description") or list(df.columns)[0]
    type_col = col_map.get("entity type") or "Entity Type"

    # Get column positions for fast tuple access
    text_idx = df.columns.get_loc(text_col)
    type_idx = df.columns.get_loc(type_col) if type_col in df.columns else None

    # prepare output columns
    out_system = []
    out_code = []
    out_desc = []

    print("Running end-to-end mappings…")
    start_map = time.time()

    # iterate as plain tuples (avoids attribute-name quirks with spaces)
    for row in tqdm(df.itertuples(index=False, name=None), total=len(df), unit="row"):
        text = str(row[text_idx]).strip()
        etype = str(row[type_idx]).strip() if type_idx is not None else ""

        system = infer_system(etype)
        retr = retrievers[system]

        # retrieve & rank
        cands = retr.topk(text, k=args.k)
        ranked = rank_candidates(text, etype, system, cands)

        # apply display preference using the full vocab of that system
        ranked = apply_display_for_candidates(system, ranked, vocabs[system], entity_type=etype)

        # pick top-1
        if ranked is not None and not ranked.empty:
            pick = ranked.iloc[0]
            out_system.append(system)
            out_code.append(str(pick.get("CODE", "")))
            out_desc.append(str(pick.get("STR", "")))

            # Terminal-only preview of top candidates (not written to files)
            top_show = ranked.head(3)[["TTY", "CODE", "STR", "final_score"]]
            print("\n----------------------------------------")
            print(f"Input: {text}  |  Type: {etype}  |  System: {system}")
            try:
                print(top_show.to_string(index=False))
            except Exception:
                print(top_show)
        else:
            out_system.append(system)
            out_code.append("")
            out_desc.append("")

    elapsed_map = time.time() - start_map
    print(f"\nDone. Mapping time: {elapsed_map:.1f}s  |  Total time: {time.time() - t0:.1f}s")

    # Append output columns to the DataFrame
    df["Output Coding System"] = out_system
    df["Output Target Code"] = out_code
    df["Output Target Description"] = out_desc

    # Ensure the first two columns exist with expected headings for CSV export
    if "Input Entity Description" not in df.columns:
        df["Input Entity Description"] = df[text_col]
    if "Entity Type" not in df.columns:
        df["Entity Type"] = df[type_col] if type_col in df.columns else ""

    # Write back to Excel (overwrite same file) 
    print(f"Writing predictions back to Excel → {args.input}")
    if ext in (".xlsx", ".xls"):
        with pd.ExcelWriter(args.input, engine="openpyxl", mode="w") as xw:
            df.to_excel(xw, sheet_name=args.sheet or "Sheet1", index=False)
    else:
        # if input was CSV, update it in place as CSV too
        df.to_csv(args.input, index=False, encoding="utf-8-sig")

    # Also write a CSV at project root with the expected 5 columns 
    cols = [
        "Input Entity Description",
        "Entity Type",
        "Output Coding System",
        "Output Target Code",
        "Output Target Description",
    ]
    out_df = df[cols]
    out_df.to_csv(args.outcsv, index=False, encoding="utf-8-sig")
    print(f"Wrote CSV snapshot → {args.outcsv}")


if __name__ == "__main__":
    main()