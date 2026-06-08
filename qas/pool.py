"""
Per-query candidate pool construction.

"""
from typing import List, Tuple

import numpy as np

from .encoder import EmbeddingStore, min_max_normalize


class PoolBuilder:
    def __init__(self, store: EmbeddingStore, mq_top: int = 100, sq_top: int = 10):
        self.store = store
        self.mq_top = mq_top
        self.sq_top = sq_top

    def build(self, qr: dict, passage_embs: np.ndarray, passage_keys: List[str]
              ) -> Tuple[List[str], dict, List[str], List[dict]]:
        """Build the pool for one query record.

        Returns:
          pool:             chunk_ids in pool order (mq first, then sub-q rescues)
          mq_score_of_full: dict chunk_id -> main-query DPR score (every chunk)
          mq_pool:          the main-query top-K chunk_ids (rank order)
          sq_pool_info:     list, one per sub-question, of {subq, chunks:[cid,...]}
                            where chunks are the NEW chunk_ids that sub-q added
        """
        mq_emb = self.store.embed_text(qr["query"], "query_to_passage")
        mq_scores = min_max_normalize(np.dot(passage_embs, mq_emb.T).squeeze())
        mq_order = np.argsort(mq_scores)[::-1]
        mq_pool = [passage_keys[i] for i in mq_order[:self.mq_top]]
        seen = set(mq_pool)
        pool = list(mq_pool)
        mq_score_of_full = {passage_keys[i]: float(mq_scores[i])
                            for i in range(len(passage_keys))}

        sq_pool_info: List[dict] = []
        for sub in qr.get("llm_decomposition", []) or []:
            sq_emb = self.store.embed_text(sub, "query_to_passage")
            sq_scores = min_max_normalize(np.dot(passage_embs, sq_emb.T).squeeze())
            sq_order = np.argsort(sq_scores)[::-1][:self.sq_top]
            added = []
            for idx in sq_order:
                cid = passage_keys[idx]
                if cid in seen:
                    continue
                seen.add(cid)
                pool.append(cid)
                added.append(cid)
            sq_pool_info.append({"subq": sub, "chunks": added})

        return pool, mq_score_of_full, mq_pool, sq_pool_info
