"""NV-Embed-v2 dense retrieval against a prebuilt passage embedding store."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


class DenseRetriever:
    """Top-k cosine retrieval over a precomputed (N, D) embedding matrix.

    The encoder is shared with the rest of the pipeline. Passage embeddings
    are loaded from a parquet store written by the indexing step.
    """

    def __init__(self, encoder, embeddings_parquet: str | Path,
                 instruction_kind: str = "query_to_passage"):
        df = pd.read_parquet(str(embeddings_parquet))
        self.chunk_ids: List[str] = df["hash_id"].tolist()
        mat = np.stack(df["embedding"].values).astype(np.float32)
        norms = np.linalg.norm(mat, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        self.matrix = mat / norms
        self.encoder = encoder
        self.instruction_kind = instruction_kind

    def encode_queries(self, queries: List[str]) -> np.ndarray:
        return self.encoder.encode(queries, kind=self.instruction_kind)

    def topk(self, queries: List[str], k: int = 10) -> List[List[Tuple[str, float]]]:
        q = self.encode_queries(queries)
        sims = q @ self.matrix.T
        out: List[List[Tuple[str, float]]] = []
        for row in sims:
            idx = np.argpartition(-row, k)[:k]
            order = idx[np.argsort(-row[idx])]
            out.append([(self.chunk_ids[j], float(row[j])) for j in order])
        return out
