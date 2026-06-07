"""QAS pool: union of entity-PPR top-k_g and dense top-k_d, deduplicated.

The two sources are interleaved (entity_1, dense_1, entity_2, dense_2, ...)
so dedup preserves first-source ranking. The result is the candidate pool
fed to the LLM listwise reranker.
"""
from __future__ import annotations

from typing import Dict, List


class QASRetriever:
    def __init__(self, entity_retriever, dense_retriever):
        self.entity = entity_retriever
        self.dense = dense_retriever

    @staticmethod
    def merge(entity_cids: List[str], dense_cids: List[str]) -> List[str]:
        """Interleave then deduplicate, preserving first occurrence."""
        interleaved: List[str] = []
        n = max(len(entity_cids), len(dense_cids))
        for i in range(n):
            if i < len(entity_cids):
                interleaved.append(entity_cids[i])
            if i < len(dense_cids):
                interleaved.append(dense_cids[i])
        seen, out = set(), []
        for c in interleaved:
            if c not in seen:
                seen.add(c)
                out.append(c)
        return out

    def build_pool(self, query_record: dict, k_each: int = 10) -> Dict:
        ent_cids = self.entity.topk(query_record, k=k_each)
        dense_cids = [c for c, _ in self.dense.topk([query_record["query"]],
                                                    k=k_each)[0]]
        pool = self.merge(ent_cids, dense_cids)
        return {
            "qid": query_record["qid"],
            "entity_topk": ent_cids,
            "dense_topk": dense_cids,
            "pool": pool,
            "pool_size": len(pool),
        }

    def build_pool_from_cids(self, entity_cids: List[str],
                              dense_cids: List[str]) -> List[str]:
        return self.merge(entity_cids, dense_cids)
