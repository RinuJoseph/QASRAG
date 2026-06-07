"""LLM fact filter (step 5 of the QAS pipeline).

Score every pool triple by query_to_fact cosine to the MAIN query, min-max
normalize, take the top `input_top_k` candidates, and ask the LLM (gpt-4o-mini)
— given the main query and its sub-questions — which candidates contain
evidence. The kept subset seeds the PPR entity reset. If the LLM keeps none,
fall back to the top `fallback_top_k` by cosine. There is no output cap by
default (`output_cap_k = None`).
"""
import re
from typing import List, Tuple

import numpy as np

from qas_prompts.fact_filter import build_messages

from .llm import LLMClient
from .encoder import EmbeddingStore, min_max_normalize


def parse_filter_response(text: str, max_n: int) -> List[int]:
    """Parse the LLM's comma-separated 1-based index list -> 0-based indices."""
    if not text or "NONE" in text.upper():
        return []
    nums = re.findall(r"\d+", text)
    out, seen = [], set()
    for s in nums:
        try:
            n = int(s)
        except ValueError:
            continue
        if 1 <= n <= max_n and n not in seen:
            seen.add(n)
            out.append(n - 1)
    return out


class FactFilter:
    def __init__(self, llm: LLMClient, store: EmbeddingStore,
                 input_top_k: int = 10, output_cap_k=None,
                 fallback_top_k: int = 5,
                 temperature: float = 0.0, max_tokens: int = 120):
        self.llm = llm
        self.store = store
        self.input_top_k = input_top_k
        self.output_cap_k = output_cap_k
        self.fallback_top_k = fallback_top_k
        self.temperature = temperature
        self.max_tokens = max_tokens

    def top_k_candidate_facts(self, triples: list, fact_embs: np.ndarray,
                              main_query: str
                              ) -> List[Tuple[int, float, tuple]]:
        """Top-K facts by query_to_fact cosine to the main query.

        Returns list of (triple_idx, min-max score, (subject, predicate, object)).
        """
        if not triples:
            return []
        q_emb = self.store.embed_text(main_query, "query_to_fact")
        scores = np.dot(fact_embs, q_emb.T)
        scores = np.squeeze(scores) if scores.ndim == 2 else scores
        scores = min_max_normalize(scores)
        K = min(self.input_top_k, len(triples))
        top_idx = np.argsort(scores)[::-1][:K]
        return [(int(i), float(scores[i]),
                 (triples[i]["subject"], triples[i]["predicate"], triples[i]["object"]))
                for i in top_idx]

    def llm_filter(self, main_query: str, sub_questions: list,
                   candidate_facts: list) -> List[int]:
        """Ask the LLM which candidates to keep. Returns 0-based local indices."""
        if not candidate_facts:
            return []
        messages = build_messages(main_query, sub_questions, candidate_facts)
        text = self.llm.chat(messages, temperature=self.temperature,
                             max_tokens=self.max_tokens) or ""
        return parse_filter_response(text, len(candidate_facts))

    def select(self, main_query: str, sub_questions: list,
               triples: list, fact_embs: np.ndarray):
        """End-to-end fact selection.

        Returns (kept_triple_indices, candidate_facts) where candidate_facts is
        the scored top-K list (for logging) and kept_triple_indices index into
        `triples` (used to seed PPR).
        """
        if not triples:
            return [], []
        cands = self.top_k_candidate_facts(triples, fact_embs, main_query)
        kept_local = self.llm_filter(main_query, sub_questions, cands)
        if kept_local:
            if self.output_cap_k is not None:
                kept_local = kept_local[:self.output_cap_k]
            kept = [cands[i][0] for i in kept_local if i < len(cands)]
            return kept, cands
        # Fallback: top-K by cosine
        kept = [c[0] for c in cands[:self.fallback_top_k]]
        return kept, cands
