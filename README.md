# QAS: Query-decomposition-Augmented Subgraph PPR for Legal Multi-hop Retrieval

## Environment setup

Conda (recommended; brings CUDA PyTorch):

```bash
conda env create -f environment.yaml
conda activate qasrag
```

or pip into an existing Python 3.10 environment:

```bash
pip install -r requirements.txt
```

Set the API key (the OpenAI SDK reads it from the environment; keys are never
stored in this repo):

```bash
export OPENAI_API_KEY=         # gpt-4o-mini pipeline + gpt-4o LF judge
```

A GPU is required for the NV-Embed embedder.

## Offline indexing

Build the index from the corpus **before** running evaluation (embeds every
passage with NV-Embed and runs OpenIE to construct the entity/fact knowledge
graph):

```bash
python index_corpus.py --config configs/koblex.yaml            # full corpus
```

Outputs: `index/corpus_graph/chunk_embeddings/vdb_chunk.parquet`,
`index/corpus_graph/graph.pickle`, and the OpenIE caches under `index/openie/`.


## Evaluation

```bash
python run_eval.py --config configs/koblex.yaml --qa            # Recall@5 + Token-F1
python run_eval.py --config configs/koblex.yaml --qa --lf       # + Legal-Fidelity (gpt-4o judge)
```


## Single question

```bash
python run_query.py --question "Until what point can Person B claim damages against Person A?" \
    --background "Person A entered a transport contract with Person B "
```



