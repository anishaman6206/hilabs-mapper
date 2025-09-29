# test_code/test_rank.py
from __future__ import annotations
import pandas as pd

from src.hilabs_mapper.loader import load_vocab_parquet, build_vocabs
from src.hilabs_mapper.retrieve import build_retrievers
from src.hilabs_mapper.rank import rank_candidates
from src.hilabs_mapper.display_picker import apply_display_for_candidates

#Load vocabs 
snomed = load_vocab_parquet("data/snomed_all_data.parquet", system="SNOMEDCT_US")
rxnorm = load_vocab_parquet("data/rxnorm_all_data.parquet", system="RXNORM")
vocabs = build_vocabs(snomed, rxnorm)
full_vocab = pd.concat([snomed, rxnorm], ignore_index=True, copy=False)

# Build retrievers (BM25) 
retrievers = build_retrievers(vocabs, full_vocab)

# SNOMED case ---
q1 = "mri pelvis"
cand1 = retrievers["SNOMEDCT_US"].topk(q1, k=10)  # you can tweak k
r1 = rank_candidates(q1, "Procedure", "SNOMEDCT_US", cand1)
# choose best display (PT > SY > FN) per CODE directly from the full SNOMED vocab
r1 = apply_display_for_candidates("SNOMEDCT_US", r1, vocabs["SNOMEDCT_US"])

print("SNOMED best for:", q1)
print(r1[["TTY","CODE","STR","final_score"]].head(5).to_string(index=False))

# RxNorm case ---
q2 = "aspirin 81 mg tablet"
cand2 = retrievers["RXNORM"].topk(q2, k=30)
r2 = rank_candidates(q2, "Medicine", "RXNORM", cand2)
# choose best display (SCD > SBD > … > PSN > SY > ET) per CODE from full RxNorm vocab
r2 = apply_display_for_candidates("RXNORM", r2, vocabs["RXNORM"])

print("\nRxNorm best for:", q2)
print(r2[["TTY","CODE","STR","final_score"]].head(5).to_string(index=False))