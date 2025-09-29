
import pandas as pd
from src.hilabs_mapper.loader import load_vocab_parquet, build_vocabs

SNOMED_PATH = "data/snomed_all_data.parquet"
RXNORM_PATH = "data/rxnorm_all_data.parquet"

snomed_df = load_vocab_parquet(SNOMED_PATH, system="SNOMEDCT_US")
rxnorm_df = load_vocab_parquet(RXNORM_PATH, system="RXNORM")
vocabs = build_vocabs(snomed_df, rxnorm_df)

print("SNOMED rows:", len(vocabs["SNOMEDCT_US"]), "cols:", list(vocabs["SNOMEDCT_US"].columns))
print("RxNorm  rows:", len(vocabs["RXNORM"]), "cols:", list(vocabs["RXNORM"].columns))

# sanity: show a couple of rows from each
print("\nSNOMED sample:")
print(vocabs["SNOMEDCT_US"][["System","CODE","STR","STY","TTY"]].head(3).to_string(index=False))
print("\nRxNorm sample:")
print(vocabs["RXNORM"][["System","CODE","STR","STY","TTY"]].head(3).to_string(index=False))

