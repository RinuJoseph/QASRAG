"""QAS — Query-decomposition-Augmented Subgraph PPR for legal multi-hop retrieval.

Self-contained, class-based. The retriever fuses two branches into a candidate
pool and reranks it with an LLM:
  - EntityPPRRetriever : per-query subgraph personalized PageRank over the corpus
                         knowledge graph, seeded from LLM-filtered facts.
  - DenseRetriever     : embedding cosine retrieval over the passage store.
  - QASRetriever       : union/dedup of the two branches -> candidate pool.
  - LLMListwiseReranker: selects the final top-k passages from the pool.
  - LegalQAReader      : reads the top-k passages and answers.

Embedders (swappable): NVEmbedEncoder (default), GTEQwen2Encoder, BGEEncoder,
E5MistralEncoder. LLM calls go through LLMClient (gpt-4o-mini pipeline, gpt-4o
LF judge).
"""
from .config import load_config, EmbedderConfig
from .encoder import (NVEmbedEncoder, EmbeddingStore, min_max_normalize,
                      get_query_instruction)
from .gte_encoder import GTEQwen2Encoder
from .bge_encoder import BGEEncoder
from .e5mistral_encoder import E5MistralEncoder
from .llm import LLMClient
from .datasets import load_dataset, SUPPORTED
from .pipeline import QASPipeline, Timer
from .dense_retriever import DenseRetriever
from .entity_ppr import EntityPPRRetriever
from .retriever import QASRetriever
from .reranker import LLMListwiseReranker
from .reader import LegalQAReader
from .eval import recall_at_k, parser_token_f1, nvembed_cosine, lf_eval

__all__ = [
    "load_config", "EmbedderConfig",
    "NVEmbedEncoder", "GTEQwen2Encoder", "BGEEncoder", "E5MistralEncoder",
    "EmbeddingStore", "min_max_normalize", "get_query_instruction",
    "LLMClient", "load_dataset", "SUPPORTED",
    "QASPipeline", "Timer",
    "DenseRetriever", "EntityPPRRetriever", "QASRetriever",
    "LLMListwiseReranker", "LegalQAReader",
    "recall_at_k", "parser_token_f1", "nvembed_cosine", "lf_eval",
]
