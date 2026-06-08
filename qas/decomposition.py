"""
Query decomposition (LLM, gpt-4o-mini by default), 
"""
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List

from qas_prompts.query_decomposition import build_messages

from .llm import LLMClient
from .utils import save_json, load_json

_SUBQ_RE = re.compile(r"\s*Q(\d+)\s*:\s*(.+?)\s*$")


def parse_subqs(raw: str) -> List[str]:
    """Extract 'Qn: ...' lines from the decomposition response."""
    out = []
    for line in raw.splitlines():
        m = _SUBQ_RE.match(line)
        if m:
            out.append(m.group(2).strip())
    return out


class Decomposer:
    """Decomposes queries into sub-questions, caching one JSON per qid."""

    def __init__(self, llm: LLMClient, cache_dir, max_tokens: int = 600,
                 temperature: float = 0.0):
        self.llm = llm
        self.cache_dir = Path(cache_dir)
        self.max_tokens = max_tokens
        self.temperature = temperature

    def _cache_path(self, qid: str) -> Path:
        return self.cache_dir / f"{qid}.json"

    def decompose_one(self, qid: str, query: str) -> dict:
        """Decompose one query; a cache hit short-circuits the LLM call."""
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        p = self._cache_path(qid)
        if p.exists():
            return load_json(p)
        raw = self.llm.chat(build_messages(query),
                            temperature=self.temperature,
                            max_tokens=self.max_tokens).strip()
        rec = {
            "qid":           qid,
            "query":         query,
            "model":         self.llm.model,
            "raw_response":  raw,
            "sub_questions": parse_subqs(raw),
        }
        save_json(rec, p)
        return rec

    def load(self, qid: str) -> List[str]:
        """Return cached sub-questions for qid, or [] if not cached yet."""
        p = self._cache_path(qid)
        if not p.exists():
            return []
        return load_json(p).get("sub_questions", [])

    def build_all(self, queries: List[Dict], workers: int = 8) -> Dict[str, dict]:
        """Decompose every query (skipping cached ones). Returns qid -> record."""
        todo = [(q["qid"], q["query"]) for q in queries
                if not self._cache_path(q["qid"]).exists()]
        results: Dict[str, dict] = {}
        for q in queries:
            p = self._cache_path(q["qid"])
            if p.exists():
                results[q["qid"]] = load_json(p)
        if not todo:
            return results
        print(f"[decompose] {len(todo)} queries with {self.llm.model}  "
              f"(workers={workers})")
        with ThreadPoolExecutor(max_workers=workers) as ex:
            futs = {ex.submit(self.decompose_one, qid, qtext): qid
                    for qid, qtext in todo}
            n_done = n_err = 0
            for fut in as_completed(futs):
                qid = futs[fut]
                try:
                    results[qid] = fut.result()
                except Exception as e:
                    n_err += 1
                    if n_err < 5:
                        print(f"  [err] {qid}: {e}", flush=True)
                n_done += 1
                if n_done % 50 == 0 or n_done == len(todo):
                    print(f"  [{n_done:4d}/{len(todo)}]  errs={n_err}", flush=True)
        return results
