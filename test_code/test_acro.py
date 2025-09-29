import pandas as pd
from src.hilabs_mapper.loader import load_vocab_parquet
from src.hilabs_mapper.acronyms import build_acronym_map_from_df
from src.hilabs_mapper.normalize import normalize_text

# Load separate vocabs and combine for acronym discovery
snomed = load_vocab_parquet("data/snomed_all_data.parquet", system="SNOMEDCT_US")
rxnorm = load_vocab_parquet("data/rxnorm_all_data.parquet", system="RXNORM")
full_vocab = pd.concat([snomed, rxnorm], ignore_index=True)

acr_map = build_acronym_map_from_df(full_vocab)
print("Acronyms learned (sample):", list(sorted(acr_map.keys()))[:10])

tests = [
    "prom",
    "ct",
    "us",
    "prbc",
    "mri pelvis",
    "ct chest",
    "hiv test",     
    "appendix removal surgery",  
    "induction for prom",
    "blood phiebotomy",
    "iud removal",
    "color and spectral doppler",
    "cpap",
    "bile acids total",

]

for t in tests:
    print(f"{t!r} -> {normalize_text(t, acr_map=acr_map)!r}")
    