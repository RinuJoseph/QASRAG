"""Tuned entity-PPR retrieval.

Builds the per-query reset vector from the entities of the LLM-kept
triples (passage seeds zeroed), then runs personalised PageRank over
the induced subgraph of a prebuilt corpus-wide knowledge graph with
uniform edge weights. The configuration replicates the locked tuned
setup for KoBLEX: ``damping=0.5``, ``input_top_k=10`` for the fact
filter, no entity cap, no passage prior contribution to the seed.
"""
from __future__ import annotations

import hashlib
import json
import pickle
from pathlib import Path
from typing import Dict, List

import numpy as np

from qas.pipeline import QASPipeline
from qas.utils import text_processing


def _entity_key(name: str) -> str:
    return "entity-" + hashlib.md5(name.encode()).hexdigest()


class EntityPPRRetriever:
    """Tuned entity-only PPR retriever.

    Parameters
    ----------
    cfg : dict
        Config dict (post ``qas.config.load_config``).
    corpus : dict
        chunk_id -> {doc_id, title, text, content}.
    passage_embs / passage_keys :
        Aligned corpus embedding matrix and chunk-id list.
    corpus_graph_pickle : path or None
        Path to a prebuilt corpus graph pickle. If provided, the per-query
        PPR runs over the induced subgraph of this graph restricted to the
        pool chunks + their entities. Otherwise the QAS-built per-query
        graph is used.
    uniform_edge_weights : bool
        If True, edge weights are flattened to 1.0 before PPR.
    """

    def __init__(self, cfg: dict, corpus: Dict[str, dict],
                 passage_embs: np.ndarray, passage_keys: List[str],
                 max_facts_in: int = 10,
                 corpus_graph_pickle: str | Path | None = None,
                 uniform_edge_weights: bool = True):
        cfg = json.loads(json.dumps(cfg))
        cfg["fact_filter"]["input_top_k"] = max_facts_in
        cfg["fact_filter"]["output_cap_k"] = None
        # ``passage_node_weight`` keeps its tuned value (the reset override
        # later zeroes the passage component anyway).
        self.pipeline = QASPipeline(cfg, corpus, passage_embs, passage_keys)
        self.pipeline.fact_filter.input_top_k = max_facts_in
        self.pipeline.fact_filter.output_cap_k = None

        self._patch_reset_to_entity_only()
        if uniform_edge_weights:
            self._patch_uniform_edges()
        if corpus_graph_pickle is not None and Path(corpus_graph_pickle).exists():
            self._patch_corpus_induced_subgraph(corpus_graph_pickle)

    # ------------------------------------------------------------------
    def _patch_reset_to_entity_only(self) -> None:
        ppr = self.pipeline.ppr
        original = ppr.build_reset

        def reset(main_query, pool, mq_score_of_full, triples, all_entities,
                  kept_indices, fact_embs, name_to_idx):
            node_w = original(main_query, pool, mq_score_of_full, triples,
                              all_entities, kept_indices, fact_embs,
                              name_to_idx)
            idx2name = {i: nm for nm, i in name_to_idx.items()}
            for i, nm in idx2name.items():
                if nm.startswith("doc:"):
                    node_w[i] = 0.0
            # No fallback: when the LLM fact-filter rejects every candidate
            # triple the reset vector is all-zero. EntityPPRRetriever.topk
            # detects this case from result["kept_facts"] and returns an empty
            # ranking, signalling the QAS pool to fall back to NV-Embed
            # dense-only for that query.
            if node_w.sum() == 0:
                u = 1.0 / max(1, len(pool))
                for cid in pool:
                    key = "doc:" + cid
                    if key in name_to_idx:
                        node_w[name_to_idx[key]] = u
            return node_w

        ppr.build_reset = reset

    def _patch_uniform_edges(self) -> None:
        ppr = self.pipeline.ppr
        original_run = ppr.run_ppr

        def run(G, node_w):
            if G.es and "weight" in G.es.attributes():
                G.es["weight"] = [1.0] * G.ecount()
            return original_run(G, node_w)

        ppr.run_ppr = run

    def _patch_corpus_induced_subgraph(self, graph_pickle: str | Path) -> None:
        graph = pickle.load(open(graph_pickle, "rb"))
        name_to_index = {n: i for i, n in enumerate(graph.vs["name"])}

        def build(pool, triples, chunk_triple_entities, all_entities):
            wanted, seen = [], set()
            for c in pool:
                if c in name_to_index and c not in seen:
                    wanted.append(name_to_index[c])
                    seen.add(c)
            for e in all_entities:
                key = _entity_key(e)
                if key in name_to_index and key not in seen:
                    wanted.append(name_to_index[key])
                    seen.add(key)
            sub = graph.induced_subgraph(wanted)
            names = list(sub.vs["name"])
            content = sub.vs["content"]
            n2i, transformed = {}, []
            for i, (raw, ct) in enumerate(zip(names, content)):
                if raw.startswith("chunk-"):
                    label = "doc:" + raw
                else:
                    label = "ent:" + text_processing(ct)
                key = label if label not in n2i else f"{label}\x00{i}"
                n2i[key] = i
                transformed.append(label)
            return sub, n2i, transformed

        self.pipeline.graph_builder.build = build

    # ------------------------------------------------------------------
    def topk(self, query_record: dict, k: int = 10) -> List[str]:
        result = self.pipeline.run(query_record, do_qa=False,
                                    ensure_openie=False)
        # Fallback policy: if the LLM fact-filter kept zero triples, the
        # entity reset vector is empty and PPR has no query-specific signal.
        # Return an empty ranking so the QAS pool can fall back to
        # NV-Embed dense-only for this query.
        if not result.get("kept_facts"):
            return []
        return [r["chunk_id"] for r in result["ppr_ranking"][:k]]

    def topk_with_meta(self, query_record: dict, k: int = 10) -> dict:
        result = self.pipeline.run(query_record, do_qa=False,
                                    ensure_openie=False)
        empty = not result.get("kept_facts")
        ranking = [] if empty else result["ppr_ranking"][:k]
        return {
            "qid": result["qid"],
            "kept_entities": result["kept_entities"],
            "kept_facts": result["kept_facts"],
            "n_triples": result["n_triples"],
            "fallback_to_dense": empty,
            "topk_cids": [r["chunk_id"] for r in ranking],
            "topk_scores": [] if empty else [r["ppr_score"] for r in ranking],
        }
