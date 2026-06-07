#!/usr/bin/env python3
"""Offline corpus indexing for QAS (HippoRAG-2-style).

Produces the two artifacts the QAS retriever consumes:
  index/corpus_graph/chunk_embeddings/vdb_chunk.parquet  (hash_id, content, embedding)
  index/corpus_graph/graph.pickle                        (igraph: chunk + entity nodes)
  index/corpus_graph/name_to_idx.json
plus the OpenIE caches under index/openie/{ner,triples}.

Steps: embed each passage with NV-Embed; run OpenIE (NER -> NER-conditioned triple
extraction) per passage; build an undirected weighted graph with entity<->entity
(fact) edges, passage<->entity edges, and entity<->entity synonym edges (cosine
>= synonym_threshold). Resumable: OpenIE is cached per chunk; embeddings are
recomputed only for new passages.

  python index_corpus.py --config configs/koblex.yaml
  python index_corpus.py --config configs/koblex.yaml --limit 20      # smoke
"""
from __future__ import annotations
import argparse, hashlib, json, sys, time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import igraph as ig

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from qas.config import load_config, EmbedderConfig
from qas.encoder import NVEmbedEncoder
from qas.llm import LLMClient
from qas.openie import OpenIEExtractor
from qas.utils import text_processing, chunk_id


def _entity_key(norm: str) -> str:
    return "entity-" + hashlib.md5(norm.encode()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "configs" / "koblex.yaml"))
    ap.add_argument("--model", default=None, help="LLM for OpenIE (overrides config)")
    ap.add_argument("--limit", type=int, default=0, help="first N passages (0 = all)")
    args = ap.parse_args()
    cfg = load_config(args.config)
    out = Path(cfg["paths"]["corpus_graph_pickle"]).parent
    (out / "chunk_embeddings").mkdir(parents=True, exist_ok=True)
    ner_dir = Path(cfg["paths"]["ner_cache"]); tri_dir = Path(cfg["paths"]["triples_cache"])
    syn_thr = cfg["graph"]["synonym_threshold"]
    ie_model = args.model or cfg["llm"]["ner_model"]

    corpus = json.load(open(cfg["dataset"]["corpus"]))
    if args.limit:
        corpus = corpus[:args.limit]
    cids = [chunk_id(c["title"], c["text"]) for c in corpus]
    contents = [f"{c['title']}\n{c['text']}" for c in corpus]
    print(f"[corpus] {len(corpus)} passages", flush=True)

    enc = NVEmbedEncoder.get(EmbedderConfig(cfg))

    # ---- 1. chunk embeddings ----
    t0 = time.time(); vecs = []
    bs = cfg["embedder"]["embedding_batch_size"]
    for s in range(0, len(contents), max(bs * 256, 2048)):
        part = contents[s:s + max(bs * 256, 2048)]
        vecs.extend(np.asarray(v, np.float32) for v in enc.encode(part, kind="passage"))
        print(f"  [embed {len(vecs)}/{len(contents)}] {time.time()-t0:.0f}s", flush=True)
    pd.DataFrame({"hash_id": cids, "content": contents, "embedding": vecs}).to_parquet(
        out / "chunk_embeddings" / "vdb_chunk.parquet")
    print(f"[saved] chunk_embeddings ({len(cids)})", flush=True)

    # ---- 2. OpenIE (NER + triples), cached per chunk ----
    openie = OpenIEExtractor(LLMClient(ie_model), ner_dir, tri_dir)
    t0 = time.time(); done = [0]
    def _ie(i):
        openie.openie_chunk(cids[i], contents[i]); done[0] += 1
        if done[0] % 200 == 0:
            print(f"  [openie {done[0]}/{len(cids)}] {time.time()-t0:.0f}s", flush=True)
    with ThreadPoolExecutor(max_workers=cfg["workers"]["openie"]) as ex:
        list(ex.map(_ie, range(len(cids))))
    print("[openie] done", flush=True)

    # ---- 3. graph: collect entities + edges ----
    edge_w = defaultdict(float)
    ent_norm = {}                                  # entity_key -> normalized text
    chunk_entities = defaultdict(set)
    for cid in cids:
        for t in openie.load_chunk_triples(cid):
            if not isinstance(t, (list, tuple)) or len(t) < 3:
                continue
            sn, on = text_processing(t[0]), text_processing(t[2])
            if not sn or not on:
                continue
            es, eo = _entity_key(sn), _entity_key(on)
            ent_norm[es] = sn; ent_norm[eo] = on
            chunk_entities[cid].add(es); chunk_entities[cid].add(eo)
            if sn != on:
                edge_w[(es, eo)] += 1.0           # fact edge
    for cid, ents in chunk_entities.items():
        for e in ents:
            edge_w[(cid, e)] += 1.0               # passage-entity edge

    # ---- 4. entity embeddings + synonym edges ----
    ent_keys = sorted(ent_norm)
    if ent_keys:
        ent_texts = [ent_norm[k] for k in ent_keys]
        emb = np.vstack([np.asarray(enc.encode(ent_texts[s:s + 2048], kind="passage"), np.float32)
                         for s in range(0, len(ent_texts), 2048)])
        emb /= np.clip(np.linalg.norm(emb, axis=1, keepdims=True), 1e-9, None)
        pd.DataFrame({"hash_id": ent_keys, "content": ent_texts, "embedding": list(emb)}).to_parquet(
            out / "entity_embeddings.parquet")
        for s in range(0, len(ent_keys), 1024):    # batched cosine for synonym edges
            sims = emb[s:s + 1024] @ emb.T
            for r, gi in enumerate(range(s, min(s + 1024, len(ent_keys)))):
                for gj in np.where(sims[r] >= syn_thr)[0]:
                    if gj > gi:
                        edge_w[(ent_keys[gi], ent_keys[gj])] += float(sims[r][gj])
    print(f"[graph] {len(cids)} passages + {len(ent_keys)} entities, {len(edge_w)} edges", flush=True)

    # ---- 5. build + save igraph ----
    names = cids + ent_keys
    content_of = {c: contents[i] for i, c in enumerate(cids)}
    content_of.update({k: ent_norm[k] for k in ent_keys})
    name_to_idx = {n: i for i, n in enumerate(names)}
    G = ig.Graph(n=len(names), directed=False)
    G.vs["name"] = names; G.vs["hash_id"] = names
    G.vs["content"] = [content_of[n] for n in names]
    edges = [(name_to_idx[a], name_to_idx[b]) for (a, b) in edge_w if a in name_to_idx and b in name_to_idx]
    G.add_edges(edges); G.es["weight"] = [edge_w[(names[a], names[b])] for a, b in edges]
    import pickle
    pickle.dump(G, open(cfg["paths"]["corpus_graph_pickle"], "wb"))
    json.dump(name_to_idx, open(out / "name_to_idx.json", "w"))
    print(f"[saved] graph.pickle ({G.vcount()} nodes, {G.ecount()} edges) -> {cfg['paths']['corpus_graph_pickle']}")
    print("INDEX_DONE")


if __name__ == "__main__":
    main()
