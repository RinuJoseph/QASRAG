#!/usr/bin/env python3
"""QAS full evaluation on a dataset (default KoBLEX).

Pipeline: EntityPPR + Dense -> QAS pool -> LLM listwise rerank -> top-k -> QA.
Reports Recall@5 (averaged over ALL queries; a query with no gold counts as 0)
and Token-F1; optional Legal-Fidelity (LF) with --lf.

  python run_eval.py --config configs/koblex.yaml --qa
  python run_eval.py --config configs/koblex.yaml --qa --lf
"""
from __future__ import annotations
import argparse, json, os, sys, time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from qas.config import load_config, EmbedderConfig
from qas.datasets import load_dataset
from qas import metrics as qas_metrics
from qas.dense_retriever import DenseRetriever
from qas.entity_ppr import EntityPPRRetriever
from qas.retriever import QASRetriever
from qas.reranker import LLMListwiseReranker
from qas.reader import LegalQAReader
from qas.eval import recall_at_k, parser_token_f1, lf_eval


def build_encoder(cfg: dict):
    """Select the embedder from cfg['embedder']['model_name']."""
    name = cfg["embedder"]["model_name"].lower()
    if "nv-embed" in name:
        from qas.encoder import NVEmbedEncoder
        return NVEmbedEncoder.get(EmbedderConfig(cfg))
    if "gte-qwen2" in name:
        from qas.gte_encoder import GTEQwen2Encoder
        return GTEQwen2Encoder.get(EmbedderConfig(cfg))
    if "bge" in name:
        from qas.bge_encoder import BGEEncoder
        return BGEEncoder.get(EmbedderConfig(cfg))
    if "e5-mistral" in name:
        from qas.e5mistral_encoder import E5MistralEncoder
        return E5MistralEncoder.get(EmbedderConfig(cfg))
    raise ValueError(f"unknown embedder {cfg['embedder']['model_name']}")


def load_passage_matrix(parquet: str):
    df = pd.read_parquet(parquet)
    keys = df["hash_id"].tolist()
    mat = np.stack(df["embedding"].values).astype(np.float32)
    norms = np.linalg.norm(mat, axis=1, keepdims=True); norms[norms == 0] = 1.0
    return mat / norms, keys


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "configs" / "koblex.yaml"))
    ap.add_argument("--name", default="qas_run", help="output sub-dir under results/")
    ap.add_argument("--model", default=None, help="override every LLM role with this model name")
    ap.add_argument("--limit", type=int, default=0, help="first N queries (0 = all)")
    ap.add_argument("--qa", action="store_true", help="run QA + Token-F1")
    ap.add_argument("--lf", action="store_true", help="run Legal-Fidelity (gpt-4o judge)")
    args = ap.parse_args()

    cfg = load_config(args.config)
    if args.model:
        for r in ("ner_model", "triple_model", "filter_model",
                  "decomposition_model", "qa_model", "rerank_model"):
            cfg["llm"][r] = args.model
    # entity-PPR pipeline reads OpenIE caches from these locations
    if cfg["paths"].get("ner_cache"):
        cfg["cache"]["ner_doc"] = cfg["paths"]["ner_cache"]
    if cfg["paths"].get("triples_cache"):
        cfg["cache"]["triples"] = cfg["paths"]["triples_cache"]
    if cfg["paths"].get("passage_emb_cache"):
        cfg["cache"]["passage_embeddings"] = cfg["paths"]["passage_emb_cache"]

    base_url = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")
    api_key = os.environ.get("OPENAI_API_KEY", "EMPTY")

    corpus, queries = load_dataset(cfg["dataset"]["name"], Path(cfg["dataset"]["root"]))
    if args.limit:
        queries = queries[:args.limit]
    print(f"[data] corpus={len(corpus)} queries={len(queries)}", flush=True)

    encoder = build_encoder(cfg)
    passage_embs, passage_keys = load_passage_matrix(cfg["paths"]["passage_embeddings_parquet"])
    entity = EntityPPRRetriever(
        cfg, corpus, passage_embs, passage_keys,
        max_facts_in=cfg["fact_filter"]["input_top_k"],
        corpus_graph_pickle=cfg["paths"].get("corpus_graph_pickle") if cfg["ppr"].get("use_corpus_subgraph") else None,
        uniform_edge_weights=cfg["ppr"].get("uniform_edge_weights", False))
    dense = DenseRetriever(encoder, cfg["paths"]["passage_embeddings_parquet"],
                           instruction_kind=cfg["embedder"]["passage_instr"])
    qas = QASRetriever(entity, dense)
    k_each = cfg["hybrid"]["k_each"]; rerank_top_k = cfg["hybrid"]["rerank_top_k"]

    # ---- pool build ----
    t0 = time.time()
    pools: List[Dict] = []
    with ThreadPoolExecutor(max_workers=cfg["workers"]["pipeline"]) as ex:
        for n, rec in enumerate(ex.map(lambda q: qas.build_pool(q, k_each), queries), 1):
            pools.append(rec)
            if n % 25 == 0:
                print(f"  [pool {n}/{len(queries)}] {time.time()-t0:.0f}s", flush=True)

    # ---- rerank ----
    rer = LLMListwiseReranker(api_key, model=cfg["llm"]["rerank_model"],
                              prompt_path=ROOT / cfg["rerank"]["prompt"],
                              max_parallel=cfg["workers"]["rerank"],
                              temperature=cfg["rerank"]["temperature"],
                              max_tokens=cfg["rerank"]["max_tokens"], base_url=base_url)
    items = []
    for q, pq in zip(queries, pools):
        cids = [c for c in pq["pool"] if c in corpus]
        cands = [(c, corpus[c]["title"], corpus[c]["text"]) for c in cids]
        bg = q["sample"].get("background_eng", "") if cfg["rerank"].get("include_background") else ""
        items.append({"qid": q["qid"], "question": (bg + " " + q["query"]).strip(), "pool": cands})
    rer_by_qid = {r["qid"]: r for r in rer.rerank(items)}

    # ---- recall (over ALL queries; no-gold -> 0) ----
    rows = []
    for q, pq in zip(queries, pools):
        gold = [g["chunk_id"] for g in q["gold"] if g.get("chunk_id")]
        ranked = rer_by_qid.get(q["qid"], {}).get("ranked_cids") or pq["pool"]
        rows.append({"qid": q["qid"], "gold_chunks": gold, "pool": pq["pool"],
                     "entity_topk": pq["entity_topk"], "dense_topk": pq["dense_topk"],
                     "reranked": ranked,
                     "recall@5": recall_at_k(ranked, gold, 5)})

    out_dir = ROOT / "results" / args.name; (out_dir / "per_query").mkdir(parents=True, exist_ok=True)
    metrics = {"run_name": args.name, "model": args.model or cfg["llm"]["qa_model"],
               "n_queries": len(rows),
               "recall@5": round(float(np.mean([r["recall@5"] for r in rows])), 4)}

    # ---- QA ----
    preds = {}
    if args.qa:
        qa = LegalQAReader(__import__("qas").LLMClient(cfg["llm"]["qa_model"]),
                           system_prompt_path=ROOT / cfg["qa"]["system_prompt"],
                           instruction_prompt_path=ROOT / cfg["qa"]["instruction_prompt"],
                           passage_format=cfg["qa"].get("passage_format", "{title} {text}"),
                           temperature=cfg["qa"]["temperature"],
                           max_tokens=cfg["qa"]["max_completion_tokens"])
        golds = {}
        def _one(q_pq):
            q, pq = q_pq
            ranked = rer_by_qid.get(q["qid"], {}).get("ranked_cids") or pq["pool"]
            chunks = [{"title": corpus[c]["title"], "text": corpus[c]["text"]}
                      for c in ranked[:rerank_top_k] if c in corpus]
            ans = qa.answer(q["sample"].get("background_eng", ""),
                            q["sample"].get("question_eng") or q["query"], chunks)["predicted_answer"] or ""
            gset = qas_metrics.gold_set(q["sample"])
            return q["qid"], ans, (list(gset)[0] if gset else ""), (parser_token_f1(ans, list(gset)) if gset else 0.0)
        with ThreadPoolExecutor(max_workers=cfg["workers"]["qa"]) as ex:
            for qid, ans, ga, f1 in ex.map(_one, zip(queries, pools)):
                preds[qid] = {"pred": ans, "gold": ga, "f1": round(f1, 4)}
                golds[qid] = ga
        metrics["tF1"] = round(float(np.mean([preds[r["qid"]]["f1"] for r in rows])), 4)
        (out_dir / "predictions.json").write_text(json.dumps(preds, ensure_ascii=False, indent=1))
        if args.lf and cfg["dataset"]["name"] == "koblex":
            sample_by = {q["qid"]: q["sample"] for q in queries}
            per = lf_eval({k: v["pred"] for k, v in preds.items()}, sample_by,
                          criteria_path=ROOT / cfg["lf_eval"]["criteria"],
                          judge_model=cfg["llm"]["judge_model"],
                          python_executable=sys.executable,
                          output_path=out_dir / f"lf_eval_{cfg['llm']['judge_model']}.json")
            vals = [v for v in per.values() if v is not None]
            metrics["LF"] = round(float(np.mean(vals)), 4) if vals else None

    (out_dir / "per_query_summary.json").write_text(json.dumps(rows, ensure_ascii=False, indent=1))
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print("\n=== QAS eval ===\n" + json.dumps(metrics, indent=2))
    print(f"[saved] {out_dir}")


if __name__ == "__main__":
    main()
