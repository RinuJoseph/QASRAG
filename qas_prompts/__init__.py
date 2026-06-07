"""Prompt templates used by subgraph-PPR.

Each prompt is in its own module:
  - ner.py — NER prompt 
  - triple_extraction.py — OpenIE triple extraction 
  - rag_qa_musique.py — final QA prompt 
  - fact_filter.py          — gpt-4o-mini fact filter (ours)
  - query_decomposition.py  — Claude / GPT decomposition prompt
"""
from . import ner, triple_extraction, rag_qa_musique, fact_filter, query_decomposition

__all__ = ["ner", "triple_extraction", "rag_qa_musique",
            "fact_filter", "query_decomposition"]
