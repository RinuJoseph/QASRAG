"""Self-contained OpenIE: NER + NER-conditioned triple extraction.

Step 3 of the QAS pipeline. OpenIE extraction: render the NER + triple-extraction prompts (in
QAS/prompts/) ourselves, call the LLM via our own LLMClient, and parse the
JSON responses with the same regex strategy uses, made robust to
markdown fences / surrounding prose.

Parsing:
  - NER:    find a JSON object containing "named_entities": [...], json.loads it,
            dedup preserving order.
  - Triples: find a JSON object containing "triples": [[s,p,o],...], json.loads,
            keep only length-3 triples cast to strings, dedup preserving order.

Results are cached per chunk as JSON (one file per chunk_id) so the LLM runs at
most once per (chunk, type).
"""
import json
import re
import threading
from typing import List
from pathlib import Path

from qas_prompts.ner import ner_system, one_shot_ner_paragraph, one_shot_ner_output
from qas_prompts.triple_extraction import (
    ner_conditioned_re_system, ner_conditioned_re_input,
    ner_conditioned_re_output, ner_conditioned_re_frame,
)

from .llm import LLMClient
from .utils import save_json, load_json, text_hash


# Regexes mirroring _extract_*_from_response, with a permissive
# fallback that finds the *largest* JSON-looking block.
_NER_RE = re.compile(r'\{[^{}]*"named_entities"\s*:\s*\[[^\]]*\][^{}]*\}', re.DOTALL)
_TRIPLES_RE = re.compile(r'\{[^{}]*"triples"\s*:\s*\[.*?\][^{}]*\}', re.DOTALL)


def _strip_fences(text: str) -> str:
    """Remove ```json ... ``` / ``` ... ``` markdown fences if present."""
    if text is None:
        return ""
    t = text.strip()
    # If wrapped in a single fenced block, return its inner content.
    m = re.search(r"```(?:json)?\s*(.*?)```", t, re.DOTALL)
    if m:
        return m.group(1).strip()
    return t


def _loads_lenient(blob: str):
    """json.loads, falling back to ast.literal_eval for single-quote JSON."""
    try:
        return json.loads(blob)
    except Exception:
        import ast
        return ast.literal_eval(blob)


def parse_named_entities(response: str) -> List[str]:
    """Extract a deduped, order-preserving list of named entities."""
    if not response:
        return []
    text = _strip_fences(response)
    obj = None
    m = _NER_RE.search(text)
    if m:
        try:
            obj = _loads_lenient(m.group())
        except Exception:
            obj = None
    if obj is None:
        # Fallback: try the whole text as JSON.
        try:
            obj = _loads_lenient(text)
        except Exception:
            return []
    ents = obj.get("named_entities", []) if isinstance(obj, dict) else []
    out = []
    seen = set()
    for e in ents:
        if not isinstance(e, str):
            e = str(e)
        if e not in seen:
            seen.add(e)
            out.append(e)
    return out


def parse_triples(response: str) -> List[List[str]]:
    """Extract length-3 string triples, deduped, order preserved."""
    if not response:
        return []
    text = _strip_fences(response)
    obj = None
    m = _TRIPLES_RE.search(text)
    if m:
        try:
            obj = _loads_lenient(m.group())
        except Exception:
            obj = None
    if obj is None:
        try:
            obj = _loads_lenient(text)
        except Exception:
            return []
    raw = obj.get("triples", []) if isinstance(obj, dict) else []
    out = []
    seen = set()
    for t in raw:
        if not isinstance(t, (list, tuple)) or len(t) != 3:
            continue
        triple = [str(x) for x in t]
        key = tuple(triple)
        if key not in seen:
            seen.add(key)
            out.append(triple)
    return out


class OpenIEExtractor:
    """NER + triple extraction with per-chunk JSON caching."""

    def __init__(self, llm: LLMClient, ner_cache_dir, triples_cache_dir,
                 max_tokens: int = 1024, temperature: float = 0.0):
        self.llm = llm
        self.ner_cache_dir = Path(ner_cache_dir)
        self.triples_cache_dir = Path(triples_cache_dir)
        self.max_tokens = max_tokens
        self.temperature = temperature
        # Per-chunk locks: under concurrent eval workers, two queries whose
        # pools share an un-cached chunk must not both call the LLM for it.
        self._locks: dict = {}
        self._locks_guard = threading.Lock()

    def _chunk_lock(self, key: str) -> threading.Lock:
        with self._locks_guard:
            return self._locks.setdefault(key, threading.Lock())

    # -- prompt rendering --------------------------------------------------
    def _ner_messages(self, passage: str):
        return [
            {"role": "system",    "content": ner_system},
            {"role": "user",      "content": one_shot_ner_paragraph},
            {"role": "assistant", "content": one_shot_ner_output},
            {"role": "user",      "content": passage},
        ]

    def _triple_messages(self, passage: str, named_entities: List[str]):
        ner_json = json.dumps({"named_entities": named_entities})
        user = ner_conditioned_re_frame.format(
            passage=passage, named_entity_json=ner_json)
        return [
            {"role": "system",    "content": ner_conditioned_re_system},
            {"role": "user",      "content": ner_conditioned_re_input},
            {"role": "assistant", "content": ner_conditioned_re_output},
            {"role": "user",      "content": user},
        ]

    # -- NER ---------------------------------------------------------------
    def ner_chunk(self, chunk_id: str, content: str) -> dict:
        """Run NER on a passage, caching the result per chunk_id."""
        self.ner_cache_dir.mkdir(parents=True, exist_ok=True)
        p = self.ner_cache_dir / f"{chunk_id}.json"
        if p.exists():
            return load_json(p)
        with self._chunk_lock(f"ner:{chunk_id}"):
            if p.exists():                       # double-check after acquiring
                return load_json(p)
            raw = self.llm.chat(self._ner_messages(content),
                                temperature=self.temperature,
                                max_tokens=self.max_tokens)
            result = {
                "chunk_id":        chunk_id,
                "unique_entities": parse_named_entities(raw),
                "raw_response":    raw,
            }
            save_json(result, p)
            return result

    def ner_text(self, text: str) -> dict:
        """NER on free-form text (a query / sub-question), cached by text-hash."""
        self.ner_cache_dir.mkdir(parents=True, exist_ok=True)
        key = text_hash(text)
        p = self.ner_cache_dir / f"{key}.json"
        if p.exists():
            return load_json(p)
        raw = self.llm.chat(self._ner_messages(text),
                            temperature=self.temperature,
                            max_tokens=self.max_tokens)
        result = {
            "text":            text,
            "label":           key,
            "unique_entities": parse_named_entities(raw),
            "raw_response":    raw,
        }
        save_json(result, p)
        return result

    # -- Triples -----------------------------------------------------------
    def triples_chunk(self, chunk_id: str, content: str,
                      named_entities: List[str]) -> dict:
        """OpenIE on a passage, conditioned on its NER list. Cached per chunk."""
        self.triples_cache_dir.mkdir(parents=True, exist_ok=True)
        p = self.triples_cache_dir / f"{chunk_id}.json"
        if p.exists():
            return load_json(p)
        with self._chunk_lock(f"tri:{chunk_id}"):
            if p.exists():                       # double-check after acquiring
                return load_json(p)
            raw = self.llm.chat(self._triple_messages(content, named_entities),
                                temperature=self.temperature,
                                max_tokens=self.max_tokens)
            result = {
                "chunk_id":             chunk_id,
                "named_entities_input": named_entities,
                "triples":              parse_triples(raw),
                "raw_response":         raw,
            }
            save_json(result, p)
            return result

    def openie_chunk(self, chunk_id: str, content: str) -> dict:
        """NER then triple extraction for one chunk (both cached)."""
        ner = self.ner_chunk(chunk_id, content)
        triples = self.triples_chunk(chunk_id, content, ner["unique_entities"])
        return {"ner": ner, "triples": triples}

    # -- cache readers -----------------------------------------------------
    def load_chunk_triples(self, chunk_id: str) -> list:
        p = self.triples_cache_dir / f"{chunk_id}.json"
        if not p.exists():
            return []
        return load_json(p).get("triples", [])

    def load_chunk_ner(self, chunk_id: str) -> list:
        p = self.ner_cache_dir / f"{chunk_id}.json"
        if not p.exists():
            return []
        return load_json(p).get("unique_entities", [])
