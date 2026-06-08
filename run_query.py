
"""Run QAS on a single question and print the retrieved passages + answer.

  python run_query.py --question "Until what point can Person B claim damages against Person A?"
  python run_query.py --question "..." --background "Person A entered a transport contract ..."
"""
from __future__ import annotations
import argparse, os, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from qas.config import load_config
from qas.datasets import load_dataset
from qas import LLMClient
from qas.dense_retriever import DenseRetriever
from qas.entity_ppr import EntityPPRRetriever
from qas.retriever import QASRetriever
from qas.reranker import LLMListwiseReranker
from qas.reader import LegalQAReader
from run_eval import build_encoder, load_passage_matrix


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "configs" / "koblex.yaml"))
    ap.add_argument("--question", required=True)
    ap.add_argument("--background", default="")
    ap.add_argument("--model", default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)
    if args.model:
        for r in ("filter_model", "decomposition_model", "qa_model", "rerank_model"):
            cfg["llm"][r] = args.model
    for k in ("ner_cache", "triples_cache", "passage_emb_cache"):
        if cfg["paths"].get(k):
            cfg["cache"][{"ner_cache": "ner_doc", "triples_cache": "triples",
                          "passage_emb_cache": "passage_embeddings"}[k]] = cfg["paths"][k]
    base_url = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")
    api_key = os.environ.get("OPENAI_API_KEY", "EMPTY")

    corpus, _ = load_dataset(cfg["dataset"]["name"], Path(cfg["dataset"]["root"]))
    encoder = build_encoder(cfg)
    embs, keys = load_passage_matrix(cfg["paths"]["passage_embeddings_parquet"])
    entity = EntityPPRRetriever(cfg, corpus, embs, keys,
                                max_facts_in=cfg["fact_filter"]["input_top_k"],
                                corpus_graph_pickle=cfg["paths"].get("corpus_graph_pickle") if cfg["ppr"].get("use_corpus_subgraph") else None,
                                uniform_edge_weights=cfg["ppr"].get("uniform_edge_weights", False))
    dense = DenseRetriever(encoder, cfg["paths"]["passage_embeddings_parquet"],
                           instruction_kind=cfg["embedder"]["passage_instr"])
    qas = QASRetriever(entity, dense)

    qr = {"qid": "q0", "query": args.question, "sample": {"background_eng": args.background,
                                                          "question_eng": args.question}}
    pool = qas.build_pool(qr, cfg["hybrid"]["k_each"])
    cids = [c for c in pool["pool"] if c in corpus]
    cands = [(c, corpus[c]["title"], corpus[c]["text"]) for c in cids]
    rer = LLMListwiseReranker(api_key, model=cfg["llm"]["rerank_model"],
                              prompt_path=ROOT / cfg["rerank"]["prompt"],
                              temperature=cfg["rerank"]["temperature"],
                              max_tokens=cfg["rerank"]["max_tokens"], base_url=base_url)
    q_text = (args.background + " " + args.question).strip()
    ranked = rer.rerank([{"qid": "q0", "question": q_text, "pool": cands}])[0]["ranked_cids"] or cids
    top = ranked[:cfg["hybrid"]["rerank_top_k"]]

    print("\n=== Retrieved passages (top-%d) ===" % len(top))
    for i, c in enumerate(top, 1):
        print(f"[{i}] {corpus[c]['title']}\n    {corpus[c]['text'][:200]}...")
    qa = LegalQAReader(LLMClient(cfg["llm"]["qa_model"]),
                       system_prompt_path=ROOT / cfg["qa"]["system_prompt"],
                       instruction_prompt_path=ROOT / cfg["qa"]["instruction_prompt"],
                       passage_format=cfg["qa"].get("passage_format", "{title} {text}"),
                       temperature=cfg["qa"]["temperature"], max_tokens=cfg["qa"]["max_completion_tokens"])
    chunks = [{"title": corpus[c]["title"], "text": corpus[c]["text"]} for c in top]
    ans = qa.answer(args.background, args.question, chunks)["predicted_answer"]
    print("\n=== Answer ===\n" + (ans or "(empty)"))


if __name__ == "__main__":
    main()
