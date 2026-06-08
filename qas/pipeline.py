"""
QASPipeline — orchestrates all seven stages for one query.

Stages (per query):
  1. decompose   : query -> sub-questions (LLM, cached per qid)
  2. pool        : dense-retrieve mq top-K_m UNION sub-q top-K_s, deduped
  3. openie      : NER + triples on each pool chunk (cached per chunk)
  4. graph       : build the per-query undirected weighted igraph
  5. fact_filter : score pool triples, LLM-filter, keep evidence subset
  6. ppr         : seed reset (entities + passage prior), run PPR, score pool
  7. qa          : top-K passages -> LLM answer

"""
import time
from pathlib import Path
from typing import Dict, List

import numpy as np

from .config import EmbedderConfig
from .encoder import EmbeddingStore
from .llm import LLMClient
from .decomposition import Decomposer
from .openie import OpenIEExtractor
from .pool import PoolBuilder
from .graph import GraphBuilder
from .fact_filter import FactFilter
from .ppr import PPRRanker
from .qa import QAReader
from .metrics import recall_at_k, score as qa_score


class Timer:
    """Accumulates per-stage wall-clock seconds within a single query."""

    def __init__(self):
        self.stages: Dict[str, float] = {}

    def time(self, name: str):
        return _TimerCtx(self, name)

    def add(self, name: str, seconds: float):
        self.stages[name] = self.stages.get(name, 0.0) + seconds

    def as_dict(self) -> Dict[str, float]:
        return {k: round(v, 6) for k, v in self.stages.items()}


class _TimerCtx:
    def __init__(self, timer: Timer, name: str):
        self.timer = timer
        self.name = name

    def __enter__(self):
        self.t0 = time.time()
        return self

    def __exit__(self, *exc):
        self.timer.add(self.name, time.time() - self.t0)
        return False


RECALL_KS = (5, 10, 20)


class QASPipeline:
    """Holds all stage objects and runs the pipeline for one query at a time."""

    def __init__(self, cfg: dict, corpus: Dict[str, dict],
                 passage_embs: np.ndarray, passage_keys: List[str]):
        self.cfg = cfg
        self.corpus = corpus
        self.passage_embs = passage_embs
        self.passage_keys = passage_keys

        cache = cfg["cache"]
        emb_cfg = EmbedderConfig(cfg)
        self.store = EmbeddingStore(
            emb_cfg,
            text_cache_dir=cache["embeddings_text"],
            passage_cache_dir=cache["passage_embeddings"],
        )

        llm_cfg = cfg["llm"]
        # One client per role so per-role token usage is captured; they share
        # the same underlying SDK client lazily.
        self.llm_decompose = LLMClient(llm_cfg["decomposition_model"])
        self.llm_openie = LLMClient(llm_cfg["ner_model"])
        self.llm_filter = LLMClient(llm_cfg["filter_model"])
        self.llm_qa = LLMClient(llm_cfg["qa_model"])

        self.decomposer = Decomposer(self.llm_decompose, cache["decompositions"])
        self.openie = OpenIEExtractor(self.llm_openie, cache["ner_doc"],
                                      cache["triples"])
        self.pool_builder = PoolBuilder(self.store,
                                        mq_top=cfg["pool"]["mq_top"],
                                        sq_top=cfg["pool"]["sq_top"])
        self.graph_builder = GraphBuilder(self.openie, self.store,
                                          synonym_threshold=cfg["graph"]["synonym_threshold"])
        ff = cfg["fact_filter"]
        self.fact_filter = FactFilter(
            self.llm_filter, self.store,
            input_top_k=ff["input_top_k"], output_cap_k=ff.get("output_cap_k"),
            fallback_top_k=ff["fallback_top_k"],
            temperature=ff.get("temperature", 0.0),
            max_tokens=ff.get("max_completion_tokens", 120))
        ppr = cfg["ppr"]
        self.ppr = PPRRanker(self.store, damping=ppr["damping"],
                             passage_node_weight=ppr["passage_node_weight"],
                             scoring_rule=ppr.get("scoring_rule", "own_only"),
                             fallback_top_k=ff["fallback_top_k"])
        qa = cfg["qa"]
        self.qa = QAReader(self.llm_qa, top_k_passages=qa["top_k_passages"],
                           temperature=qa.get("temperature", 0.0),
                           max_tokens=qa.get("max_completion_tokens", 400))

        # Shared OpenIE pool: bounds total concurrent extractions across all
        # query workers (each query submits its uncached pool chunks here and
        # waits). Sizing here, not per query, prevents a thread explosion.
        from concurrent.futures import ThreadPoolExecutor
        self._openie_pool = ThreadPoolExecutor(
            max_workers=int(cfg.get("workers", {}).get("openie", 16)))

    # -- per-stage helpers -------------------------------------------------
    def _ensure_openie(self, pool: List[str]):
        """Ensure NER + triples are cached for every pool chunk. Uncached
        chunks are extracted concurrently via the shared OpenIE pool; the
        per-chunk lock in OpenIEExtractor stops two queries double-extracting a
        shared chunk."""
        todo = [cid for cid in pool
                if not (Path(self.openie.triples_cache_dir) / f"{cid}.json").exists()]
        if not todo:
            return
        futs = [self._openie_pool.submit(
                    self.openie.openie_chunk, cid, self.corpus[cid]["content"])
                for cid in todo]
        for f in futs:
            f.result()

    def all_clients(self):
        return [self.llm_decompose, self.llm_openie, self.llm_filter, self.llm_qa]

    # -- main entry --------------------------------------------------------
    def run(self, qr: dict, do_qa: bool = True,
            ensure_openie: bool = True) -> dict:
        """Run the full pipeline for one query record, returning the rich dict."""
        timer = Timer()
        corpus = self.corpus

        # 1. Decompose
        with timer.time("decompose"):
            sub_questions = self.decomposer.load(qr["qid"])
            if not sub_questions and qr.get("llm_decomposition"):
                sub_questions = qr["llm_decomposition"]
            if not sub_questions:
                rec = self.decomposer.decompose_one(qr["qid"], qr["query"])
                sub_questions = rec.get("sub_questions", [])
        qr = {**qr, "llm_decomposition": sub_questions}

        # 2. Pool
        with timer.time("pool"):
            pool, mq_score_of_full, mq_pool, sq_pool_info = \
                self.pool_builder.build(qr, self.passage_embs, self.passage_keys)

        # 3. OpenIE (cache per chunk)
        with timer.time("openie"):
            if ensure_openie:
                self._ensure_openie(pool)
            triples, chunk_triple_entities, all_entities = \
                self.graph_builder.collect_triples_and_entities(pool)

        # 4. Graph
        with timer.time("graph"):
            G, name_to_idx, names = self.graph_builder.build(
                pool, triples, chunk_triple_entities, all_entities)

        # 5. Fact filter
        with timer.time("fact_filter"):
            fact_embs = None
            kept_indices: List[int] = []
            candidate_facts = []
            if triples:
                f_strs = [f"{t['subject']} {t['predicate']} {t['object']}"
                          for t in triples]
                fact_embs = self.store.embed_texts(f_strs, "query_to_fact")
                kept_indices, candidate_facts = self.fact_filter.select(
                    qr["query"], sub_questions, triples, fact_embs)

        # 6. PPR — timed in 3 parts so the PURE igraph PPR is isolated:
        #    ppr_reset = build the personalization/reset vector
        #    ppr_run   = JUST igraph personalized_pagerank
        #    ppr_score = passage-score extraction + sort
        with timer.time("ppr_reset"):
            node_w = self.ppr.build_reset(
                qr["query"], pool, mq_score_of_full, triples, all_entities,
                kept_indices, fact_embs, name_to_idx)
        with timer.time("ppr_run"):
            pr = self.ppr.run_ppr(G, node_w)
        with timer.time("ppr_score"):
            scores = self.ppr.score_pool(pool, pr, G, names, name_to_idx)
            ranking = sorted(pool, key=lambda c: -scores[c])

        # 7. QA
        gold = [g["chunk_id"] for g in qr["gold"] if g.get("chunk_id")]
        gold_set = set(gold)
        top_passages = [
            {"rank": r + 1, "chunk_id": cid,
             "doc_id": corpus[cid]["doc_id"], "title": corpus[cid]["title"]}
            for r, cid in enumerate(ranking[:self.qa.top_k_passages])
        ]
        predicted_answer = qa_raw = None
        if do_qa:
            with timer.time("qa"):
                r = self.qa.answer(qr["query"], top_passages, corpus)
                predicted_answer = r["predicted_answer"]
                qa_raw = r["raw_response"]

        timer.add("total", sum(timer.stages.values()))

        # ---- assemble rich per-query record ----
        # edge-type breakdown (untimed; for reporting nodes/edges incl. synonym
        # + passage-entity context). passage<->entity = one doc + one ent node;
        # entity<->entity = both ent nodes (covers triple edges AND synonym edges).
        n_pe = n_ee = None
        try:
            _vn = G.vs["name"]
            n_pe = n_ee = 0
            for e in G.es:
                ad = _vn[e.source].startswith("doc:")
                bd = _vn[e.target].startswith("doc:")
                if ad != bd:
                    n_pe += 1
                elif not ad and not bd:
                    n_ee += 1
        except Exception:
            pass

        kept_facts = [
            {"subject": triples[i]["subject"],
             "predicate": triples[i]["predicate"],
             "object": triples[i]["object"],
             "from_chunk": triples[i]["from_chunk"]}
            for i in kept_indices if i < len(triples)
        ]
        kept_entities = sorted({
            triples[i]["subject_n"] for i in kept_indices if i < len(triples)
        } | {
            triples[i]["object_n"] for i in kept_indices if i < len(triples)
        })

        ppr_ranking = [
            {"rank": r + 1, "chunk_id": cid, "title": corpus[cid]["title"],
             "ppr_score": round(float(scores[cid]), 10), "is_gold": cid in gold_set}
            for r, cid in enumerate(ranking)
        ]

        recall = {f"recall@{k}": recall_at_k(ranking, gold, k) for k in RECALL_KS}
        # also keep per-gold landing for inspection
        recall_per_gold = [
            {"hop": g["hop"], "chunk_id": g.get("chunk_id"),
             "in_pool": g.get("chunk_id") in set(pool) if g.get("chunk_id") else False,
             "rank": (ranking.index(g["chunk_id"]) + 1)
                     if g.get("chunk_id") in set(pool) else None}
            for g in qr["gold"]
        ]

        f1 = em = None
        if do_qa and predicted_answer is not None:
            f1, em = qa_score(predicted_answer, qr["sample"])

        gold_answers = [g["answer"] for g in qr["gold"] if g.get("answer")]
        if not gold_answers and qr.get("answer") is not None:
            a = qr["answer"]
            gold_answers = a if isinstance(a, list) else [a]

        return {
            "qid": qr["qid"],
            "query": qr["query"],
            "decomposition": sub_questions,
            "mq_pool": [
                {"rank": r + 1, "chunk_id": cid, "doc_id": corpus[cid]["doc_id"],
                 "title": corpus[cid]["title"]}
                for r, cid in enumerate(mq_pool)
            ],
            "sq_pool": [
                {"subq": info["subq"],
                 "chunks": [
                     {"chunk_id": cid, "doc_id": corpus[cid]["doc_id"],
                      "title": corpus[cid]["title"]}
                     for cid in info["chunks"]
                 ]}
                for info in sq_pool_info
            ],
            "final_pool": pool,
            "final_pool_size": len(pool),
            "candidate_facts": [
                {"subject": s, "predicate": p, "object": o,
                 "cosine": round(float(sc), 8)}
                for (_, sc, (s, p, o)) in candidate_facts
            ],
            "kept_facts": kept_facts,
            "kept_entities": kept_entities,
            "n_kept_facts": len(kept_facts),
            "ppr_ranking": ppr_ranking,
            "top5": [
                {"chunk_id": p["chunk_id"], "title": p["title"]}
                for p in top_passages
            ],
            "predicted_answer": predicted_answer,
            "qa_raw": qa_raw,
            "graph_stats": {
                "n_passage_nodes": len(pool),
                "n_entity_nodes": len(all_entities),
                "n_total_nodes": G.vcount(),
                "n_edges": G.ecount(),
                "n_passage_entity_edges": n_pe,   # context: passage<->entity
                "n_entity_entity_edges": n_ee,    # triple + synonym
            },
            "n_triples": len(triples),
            "timing_sec": timer.as_dict(),
            "recall": recall,
            "recall_per_gold": recall_per_gold,
            "f1": f1,
            "em": em,
            "gold_chunks": gold,
            "gold_answers": gold_answers,
        }
