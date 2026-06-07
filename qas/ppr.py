"""Personalized PageRank (step 6 of the QAS pipeline).

Faithful to algorithm:
  entity_reset[e] += fact_score / max(|pool chunks containing e|, 1)
                     (averaged across the kept facts that touch e)
  passage_reset[p] = mq_DPR_score(p) * passage_node_weight (=0.05)
  node_weights      = phrase_w + passage_w  (NaN/neg -> 0; uniform fallback)
  igraph PRPACK undirected, damping=0.5
  passage_score(p)  = pr[doc:p]  (own PPR mass only)
"""
from collections import defaultdict
from typing import Dict, List

import numpy as np
import igraph as ig

from .encoder import EmbeddingStore, min_max_normalize


def _entity_chunk_count(triples: list, all_entities: set) -> Dict[str, int]:
    """For each entity, count distinct pool chunks containing it (via triples)."""
    out = defaultdict(int)
    by_chunk = defaultdict(set)
    for t in triples:
        by_chunk[t["from_chunk"]].add(t["subject_n"])
        by_chunk[t["from_chunk"]].add(t["object_n"])
    for cid, ents in by_chunk.items():
        for e in ents:
            if e in all_entities:
                out[e] += 1
    return out


class PPRRanker:
    def __init__(self, store: EmbeddingStore, damping: float = 0.5,
                 passage_node_weight: float = 0.05,
                 scoring_rule: str = "own_only",
                 fallback_top_k: int = 5):
        self.store = store
        self.damping = damping
        self.passage_node_weight = passage_node_weight
        self.scoring_rule = scoring_rule
        self.fallback_top_k = fallback_top_k

    def build_reset(self, main_query: str, pool: list, mq_score_of_full: dict,
                    triples: list, all_entities: set, kept_triple_indices: list,
                    fact_embs: np.ndarray, name_to_idx: Dict[str, int]
                    ) -> np.ndarray:
        """Build the combined reset (personalization) vector (length n_nodes)."""
        n_nodes = len(name_to_idx)
        if not triples or fact_embs is None:
            # No triples -> uniform 1/|pool| over passage nodes only.
            node_w = np.zeros(n_nodes)
            for cid in pool:
                ent_id = f"doc:{cid}"
                if ent_id in name_to_idx:
                    node_w[name_to_idx[ent_id]] = 1.0 / len(pool) if pool else 0.0
            return node_w

        # Score every triple to retrieve the kept ones' fact scores.
        q_emb = self.store.embed_text(main_query, "query_to_fact")
        f_scores = np.dot(fact_embs, q_emb.T)
        f_scores = np.squeeze(f_scores) if f_scores.ndim == 2 else f_scores
        f_scores = min_max_normalize(f_scores)

        if not kept_triple_indices:
            K = min(self.fallback_top_k, len(triples))
            kept_triple_indices = np.argsort(f_scores)[::-1][:K].tolist()
        top_facts = [(triples[i], float(f_scores[i]))
                     for i in kept_triple_indices if i < len(triples)]

        # Entity reset: score / chunk_count, averaged over occurrences.
        ent_chunk_count = _entity_chunk_count(triples, all_entities)
        phrase_w = np.zeros(n_nodes)
        n_occ = np.zeros(n_nodes)
        for t, score in top_facts:
            for ph in (t["subject_n"], t["object_n"]):
                ent_id = f"ent:{ph}"
                if ent_id not in name_to_idx:
                    continue
                idx = name_to_idx[ent_id]
                div = max(ent_chunk_count.get(ph, 0), 1)
                phrase_w[idx] += score / div
                n_occ[idx] += 1
        with np.errstate(invalid="ignore", divide="ignore"):
            phrase_w = np.where(n_occ > 0, phrase_w / n_occ, 0.0)

        # Passage reset: mq_DPR * passage_node_weight.
        passage_w = np.zeros(n_nodes)
        for cid in pool:
            ent_id = f"doc:{cid}"
            if ent_id not in name_to_idx:
                continue
            passage_w[name_to_idx[ent_id]] = (
                mq_score_of_full.get(cid, 0.0) * self.passage_node_weight)

        node_w = phrase_w + passage_w
        node_w = np.where(np.isnan(node_w) | (node_w < 0), 0, node_w)
        if node_w.sum() == 0:
            for cid in pool:
                ent_id = f"doc:{cid}"
                if ent_id in name_to_idx:
                    node_w[name_to_idx[ent_id]] = 1.0 / len(pool)
        return node_w

    def run_ppr(self, G: ig.Graph, node_weights: np.ndarray) -> np.ndarray:
        """Run igraph PRPACK personalized_pagerank, undirected."""
        n_nodes = G.vcount()
        pr = G.personalized_pagerank(
            vertices=range(n_nodes),
            damping=self.damping,
            directed=False,
            weights="weight" if G.es and "weight" in G.es.attributes() else None,
            reset=node_weights.tolist(),
            implementation="prpack",
        )
        return np.asarray(pr)

    def passage_scores_own_only(self, pool: list, pr: np.ndarray,
                                name_to_idx: Dict[str, int]) -> Dict[str, float]:
        """passage scoring: pr[doc:p] only."""
        out = {}
        for cid in pool:
            ent_id = f"doc:{cid}"
            out[cid] = float(pr[name_to_idx[ent_id]]) if ent_id in name_to_idx else 0.0
        return out

    def passage_scores_own_plus_nbr(self, pool: list, pr: np.ndarray,
                                    G: ig.Graph, names: List[str],
                                    name_to_idx: Dict[str, int]) -> Dict[str, float]:
        """Alternative scoring: own_pr + Σ neighbor_entity_pr * edge_weight."""
        out = {}
        for cid in pool:
            ent_id = f"doc:{cid}"
            if ent_id not in name_to_idx:
                out[cid] = 0.0
                continue
            idx = name_to_idx[ent_id]
            own = float(pr[idx])
            nbr = 0.0
            for nb in G.neighbors(idx):
                if names[nb].startswith("ent:"):
                    w = G.es[G.get_eid(idx, nb)]["weight"]
                    nbr += float(pr[nb]) * w
            out[cid] = own + nbr
        return out

    def score_pool(self, pool: list, pr: np.ndarray, G: ig.Graph,
                   names: List[str], name_to_idx: Dict[str, int]) -> Dict[str, float]:
        """Dispatch to the configured scoring rule."""
        if self.scoring_rule == "own_plus_nbr":
            return self.passage_scores_own_plus_nbr(pool, pr, G, names, name_to_idx)
        return self.passage_scores_own_only(pool, pr, name_to_idx)
