"""
Per-dataset adapters .
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple

from .utils import chunk_id as _chunk_id


# ---------------------------------------------------------------------------
# Corpus loader (uniform across datasets)
# ---------------------------------------------------------------------------
def _load_corpus_json(corpus_json: Path) -> Tuple[Dict[str, dict], Dict[str, List[str]]]:
    """Read a JSON corpus file (list of {id, title, text}).

    Returns:
      corpus: chunk_id -> {doc_id, title, text, content}
      title_to_chunks: title -> [chunk_id, ...] (ordered by appearance)
    """
    rows = json.load(open(corpus_json))
    corpus: Dict[str, dict] = {}
    title_to_chunks: Dict[str, List[str]] = {}
    for i, row in enumerate(rows):
        title = row["title"]
        text = row["text"]
        cid = _chunk_id(title, text)
        raw_id = row.get("id", row.get("idx", i))
        corpus[cid] = {
            "doc_id":  str(raw_id),
            "title":   title,
            "text":    text,
            "content": f"{title}\n{text}",
        }
        title_to_chunks.setdefault(title, []).append(cid)
    return corpus, title_to_chunks


# ---------------------------------------------------------------------------
# Per-dataset query+gold adapters
# ---------------------------------------------------------------------------
def _adapter_musique(samples, corpus, title_to_chunks):
    """gold = paragraphs[support_idx]; match title+text against corpus content."""
    content_to_chunk = {v["content"]: k for k, v in corpus.items()}
    out = []
    for s in samples:
        paras = {p["idx"]: p for p in s["paragraphs"]}
        gold = []
        for i, qd in enumerate(s.get("question_decomposition", []), start=1):
            sup = qd.get("paragraph_support_idx")
            cid = None
            if sup is not None and sup in paras:
                p = paras[sup]
                key = f"{p['title']}\n{p.get('paragraph_text', p.get('text', ''))}"
                cid = content_to_chunk.get(key)
                if cid is None:
                    cands = title_to_chunks.get(p["title"], [])
                    if cands:
                        cid = cands[0]
            gold.append({"hop": i, "chunk_id": cid,
                         "subq": qd.get("question"),
                         "answer": qd.get("answer")})
        out.append({
            "qid":    s["id"],
            "query":  s["question"],
            "gold":   gold,
            "answer": s.get("answer"),
            "sample": s,
        })
    return out


def _by_title_supporting(samples, corpus, title_to_chunks, qid_key, q_key,
                         answer_key="answer"):
    """Common adapter for HotpotQA / 2WikiMultiHopQA / HoVer — all three have
    `supporting_facts = [[title, sent_idx], ...]` keyed by title."""
    out = []
    for s in samples:
        sfs = s.get("supporting_facts", []) or []
        seen = set()
        titles = []
        for tup in sfs:
            if not tup:
                continue
            t = tup[0]
            if t in seen:
                continue
            seen.add(t)
            titles.append(t)
        gold = []
        for i, t in enumerate(titles, start=1):
            cands = title_to_chunks.get(t, [])
            cid = cands[0] if cands else None
            gold.append({"hop": i, "chunk_id": cid, "subq": None,
                         "answer": None, "title": t})
        out.append({
            "qid":    str(s[qid_key]),
            "query":  s[q_key],
            "gold":   gold,
            "answer": s.get(answer_key),
            "sample": s,
        })
    return out


def _adapter_hotpotqa(samples, corpus, title_to_chunks):
    return _by_title_supporting(samples, corpus, title_to_chunks,
                                qid_key="_id", q_key="question")


def _adapter_2wikimultihopqa(samples, corpus, title_to_chunks):
    return _by_title_supporting(samples, corpus, title_to_chunks,
                                qid_key="_id", q_key="question")


def _adapter_hover(samples, corpus, title_to_chunks):
    return _by_title_supporting(samples, corpus, title_to_chunks,
                                qid_key="hover_id", q_key="claim",
                                answer_key="label")


def _adapter_popqa(samples, corpus, title_to_chunks):
    """popqa: gold = paragraphs[].title (all listed paragraphs are relevant)."""
    out = []
    for s in samples:
        paras = s.get("paragraphs", []) or []
        gold = []
        for i, p in enumerate(paras, start=1):
            t = p.get("title")
            cands = title_to_chunks.get(t, [])
            cid = cands[0] if cands else None
            gold.append({"hop": i, "chunk_id": cid, "title": t,
                         "subq": None, "answer": None})
        out.append({
            "qid":    str(s["id"]),
            "query":  s["question"],
            "gold":   gold,
            "answer": s.get("obj") or (s.get("possible_answers") or [None])[0],
            "sample": s,
        })
    return out


def _adapter_nq_rear(samples, corpus, title_to_chunks):
    """nq_rear: gold = contexts with is_supporting == True."""
    out = []
    for i, s in enumerate(samples):
        ctxs = s.get("contexts", []) or []
        gold = []
        hop = 0
        for c in ctxs:
            if not c.get("is_supporting"):
                continue
            hop += 1
            t = c.get("title")
            cands = title_to_chunks.get(t, [])
            cid = cands[0] if cands else None
            gold.append({"hop": hop, "chunk_id": cid, "title": t,
                         "subq": None, "answer": None})
        out.append({
            "qid":    f"nq_rear_{i}",
            "query":  s["question"],
            "gold":   gold,
            "answer": (s.get("reference") or [None])[0],
            "sample": s,
        })
    return out


def _adapter_lveval(samples, corpus, title_to_chunks):
    """lveval: no chunk-level gold (gold is an answer string only)."""
    out = []
    for s in samples:
        ans = s.get("gold_ans") or (s.get("answers") or [None])[0]
        out.append({
            "qid":    f"lveval_{s['id']}",
            "query":  s["question"],
            "gold":   [],
            "answer": ans,
            "sample": s,
        })
    return out


def _adapter_narrativeqa(samples, corpus, title_to_chunks):
    """narrativeqa: gold is document-level; no chunk-level recall."""
    out = []
    for i, s in enumerate(samples):
        ans = (s.get("answer") or [None])[0]
        doc = s.get("document", {})
        qid = doc.get("id", f"narrativeqa_{i}")
        out.append({
            "qid":    f"narrativeqa_{i}_{qid[:8]}",
            "query":  s["question"],
            "gold":   [],
            "answer": ans,
            "sample": s,
        })
    return out


def _adapter_koblex(samples, corpus, title_to_chunks):
    """KoBLEX (legal multi-hop). Gold contexts resolve by `index_eng` ==
    corpus title, with a `content_eng` substring fallback. qid encodes the
    hop count (`<n>hop_<id>`) for hop-wise analysis."""
    import ast
    corp_items = None

    def resolve(ctx):
        ie = (ctx.get("index_eng") or "").strip()
        if ie and title_to_chunks.get(ie):
            return title_to_chunks[ie][0]
        nonlocal corp_items
        if corp_items is None:
            corp_items = [(cid, v["text"]) for cid, v in corpus.items()]
        txt = ctx.get("content_eng", "") or ""
        needle = (txt[30:130] if len(txt) > 130 else txt).strip() or txt[:100]
        if needle:
            for cid, ct in corp_items:
                if needle in ct:
                    return cid
        return None

    out = []
    for s in samples:
        nh = int(s.get("n_hops", 0) or 0)
        ctxs = s.get("contexts") or []
        if isinstance(ctxs, str):
            try:
                ctxs = ast.literal_eval(ctxs)
            except Exception:
                ctxs = []
        gold = [{"hop": i, "chunk_id": resolve(c), "subq": None,
                 "answer": None, "title": c.get("index_eng")}
                for i, c in enumerate(ctxs, start=1)]
        out.append({
            "qid":    f"{nh}hop_{s['id']}",
            "query":  s["question_eng"],
            "gold":   gold,
            "answer": s.get("answer_eng"),
            "sample": s,
        })
    return out


_ADAPTERS = {
    "musique":                _adapter_musique,
    "hotpotqa":               _adapter_hotpotqa,
    "2wikimultihopqa":        _adapter_2wikimultihopqa,
    "hover":                  _adapter_hover,
    "popqa":                  _adapter_popqa,
    "nq_rear":                _adapter_nq_rear,
    "lveval":                 _adapter_lveval,
    "narrativeqa_dev_10_doc": _adapter_narrativeqa,
    "koblex":                 _adapter_koblex,
}

SUPPORTED = sorted(_ADAPTERS.keys())


# ---------------------------------------------------------------------------
# Public entry points
# ---------------------------------------------------------------------------
def dataset_paths(name: str, root: Path) -> Tuple[Path, Path]:
    """Return (queries_json, corpus_json) for the dataset.

    Files live directly under `root` as `<name>.json` / `<name>_corpus.json`.
    the same chunk-id scheme works for every dataset.
    """
    root = Path(root)
    return root / f"{name}.json", root / f"{name}_corpus.json"


def load_dataset(name: str, root: Path) -> Tuple[Dict[str, dict], List[dict]]:
    """Load (corpus, queries) for a named dataset.

    `queries[i]` always has: qid, query, gold (list of {hop, chunk_id, title,
    subq, answer}), answer, sample.
    """
    if name not in _ADAPTERS:
        raise ValueError(f"unknown dataset {name!r}. supported: {SUPPORTED}")
    q_path, c_path = dataset_paths(name, root)
    if not q_path.exists():
        raise FileNotFoundError(f"queries file missing: {q_path}")
    if not c_path.exists():
        raise FileNotFoundError(f"corpus file missing: {c_path}")
    corpus, title_to_chunks = _load_corpus_json(c_path)
    samples = json.load(open(q_path))
    queries = _ADAPTERS[name](samples, corpus, title_to_chunks)
    return corpus, queries
