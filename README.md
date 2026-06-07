# QAS: Query-decomposition-Augmented Subgraph PPR for Legal Multi-hop Retrieval

QAS retrieves and answers multi-hop legal questions over a large statute corpus
(KoBLEX: 226 questions, ~44k English-translated Korean-law passages). For each
question it fuses two retrieval branches into a candidate pool, reranks the pool
with an LLM, and reads the top passages to produce a grounded answer.

```
question ─┬─ EntityPPRRetriever : per-query subgraph Personalized PageRank over the
          │                       corpus knowledge graph, seeded from LLM-filtered facts
          └─ DenseRetriever     : embedding cosine retrieval over the passage store
                    │
            QASRetriever (union + dedup) ─► LLMListwiseReranker ─► top-k ─► LegalQAReader ─► answer
```

The package is self-contained and class-based. The pipeline runs on gpt-4o-mini
(decomposition, OpenIE, fact filter, rerank, QA) with gpt-4o as the Legal-Fidelity
judge.

## Layout

```
qas/                 retriever package
  config.py          YAML config loader
  encoder.py         NV-Embed-v2 embedder (default)
  gte_encoder.py     gte-Qwen2-7B          (alternative embedder)
  bge_encoder.py     bge-large-en-v1.5     (alternative embedder)
  e5mistral_encoder.py  e5-mistral-7b      (alternative embedder)
  llm.py             OpenAI / Anthropic chat client
  dense_retriever.py DenseRetriever
  entity_ppr.py      EntityPPRRetriever (wraps the QAS pipeline)
  retriever.py       QASRetriever (pool fusion)
  reranker.py        LLMListwiseReranker
  reader.py          LegalQAReader
  pipeline.py        QASPipeline (decompose -> openie -> graph -> fact-filter -> PPR)
  ...                decomposition / openie / graph / fact_filter / ppr / pool / metrics / utils
  eval.py            Recall@k, Token-F1, Legal-Fidelity
qas_prompts/         prompt builders (NER, triple extraction, decomposition, QA)
prompts/             qa_system.txt, qa_instruction.txt, rerank_select.txt, lf_eval.json
configs/koblex.yaml  dataset + model + path configuration
data/                koblex.json (questions), koblex_corpus.json (passages)
index/               built by index_corpus.py (chunk embeddings + corpus graph + OpenIE)
index_corpus.py      offline indexing (build the index from the corpus)
run_eval.py          full evaluation
run_query.py         single-question demo
```

## Environment setup

Conda (recommended; brings CUDA PyTorch):

```bash
conda env create -f environment.yaml
conda activate legal-qas
```

or pip into an existing Python 3.10 environment:

```bash
pip install -r requirements.txt
```

Set the API key (the OpenAI SDK reads it from the environment; keys are never
stored in this repo):

```bash
export OPENAI_API_KEY=sk-...          # gpt-4o-mini pipeline + gpt-4o LF judge
```

A CUDA GPU is required for the NV-Embed embedder.

## Offline indexing

Build the index from the corpus **before** running evaluation (embeds every
passage with NV-Embed and runs OpenIE to construct the entity/fact knowledge
graph):

```bash
python index_corpus.py --config configs/koblex.yaml            # full corpus
python index_corpus.py --config configs/koblex.yaml --limit 20 # quick check
```

Outputs: `index/corpus_graph/chunk_embeddings/vdb_chunk.parquet`,
`index/corpus_graph/graph.pickle`, and the OpenIE caches under `index/openie/`.
The step is resumable — OpenIE is cached per passage. Query-side caches
(decomposition, embeddings) are filled automatically on the first evaluation run.

## Evaluation

```bash
python run_eval.py --config configs/koblex.yaml --qa            # Recall@5 + Token-F1
python run_eval.py --config configs/koblex.yaml --qa --lf       # + Legal-Fidelity (gpt-4o judge)
```

Recall@5 is averaged over **all** questions (a question with no gold counts as 0).
Results are written to `results/<name>/` (`metrics.json` reports Recall@5,
Token-F1, and LF). Token-F1 needs `--qa`; LF needs `--qa --lf`.

## Single question

```bash
python run_query.py --question "Until what point can Person B claim damages against Person A?" \
    --background "Person A entered a transport contract with Person B ..."
```

Prints the reranked top-k passages and the generated answer.

## Configuration

`configs/koblex.yaml` controls the dataset paths, the embedder
(`embedder.model_name`: `nvidia/NV-Embed-v2` default, or a gte/bge/e5 model), the
per-role LLM names (`llm.*_model`), and the retrieval/PPR/rerank/QA hyperparameters.
To switch embedders, set `embedder.model_name` and point
`paths.passage_embeddings_parquet` at that embedder's index.
