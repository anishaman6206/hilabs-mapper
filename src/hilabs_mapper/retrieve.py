"""
BM25-based candidate retrieval over STR, per coding system.

- Uses acronym-aware normalization (from your vocab) so queries like "mri pelvis"
  expand to "magnetic resonance imaging pelvis" before retrieval.
- Returns a pandas DataFrame with the original vocab rows for top-k matches.
"""

from __future__ import annotations
import pandas as pd
from typing import Dict
from langchain_community.retrievers import BM25Retriever
from .normalize import normalize_text
from .acronyms import build_acronym_map_from_df

class SystemRetriever:
    def __init__(self, df_system: pd.DataFrame, acr_map: Dict[str, str]):
        """
        df_system: DataFrame for a single system (SNOMEDCT_US or RXNORM)
        acr_map:   acronym->expansion dict built from the full combined vocab
        """
        self.df = df_system.reset_index(drop=True)
        self.acr_map = acr_map

        # Build BM25 index over normalized STRs (acronym-aware).
        self._texts_norm = [normalize_text(s, acr_map=self.acr_map) for s in self.df["STR"].astype(str)]
        self._bm25 = BM25Retriever.from_texts(self._texts_norm)

        # Map normalized text -> row index for quick lookup.
        # Note: multiple rows can normalize to the same text; we keep first seen index.
        self._norm_to_idx = {}
        for i, t in enumerate(self._texts_norm):
            self._norm_to_idx.setdefault(t, i)

    def topk(self, query: str, k: int) -> pd.DataFrame:
        """
        Return top-k candidate rows (original columns) for the given query.
        """
        q = normalize_text(query, acr_map=self.acr_map)
        docs = self._bm25.invoke(q)[:k]
        rows = []
        for d in docs:
            i = self._norm_to_idx.get(d.page_content)
            if i is not None:
                rows.append(self.df.iloc[i])
        if not rows:
            return self.df.head(0).copy()
        return pd.DataFrame(rows).reset_index(drop=True)

def build_retrievers(vocabs: Dict[str, pd.DataFrame], full_vocab: pd.DataFrame) -> Dict[str, SystemRetriever]:
    """
    Build a retriever per system using a single acronym map derived from the full vocab.
    """
    acr_map = build_acronym_map_from_df(full_vocab)
    return {sys: SystemRetriever(df, acr_map=acr_map) for sys, df in vocabs.items()}
