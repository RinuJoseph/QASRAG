# QAS: Query-decomposition-Augmented Subgraph PPR for Legal Multi-hop Retrieval

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
