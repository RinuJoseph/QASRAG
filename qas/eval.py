"""
Evaluation metrics for retrieval and answer fidelity.
"""
from __future__ import annotations

import json
import os
import re
import string
import subprocess
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np


def recall_at_k(ranked: List[str], gold: Iterable[str], k: int) -> float:
    gold_set = set(gold)
    if not gold_set:
        return 0.0
    top_k = set(ranked[:k])
    return len(gold_set & top_k) / max(len(gold_set), 1)


# ---- Token F1 ---------------------------------------------------------------
def _normalize_answer(answer: str) -> str:
    def remove_articles(t): return re.sub(r"\b(a|an|the)\b", " ", t)
    def white_space(t): return " ".join(t.split())
    def remove_punc(t):
        exclude = set(string.punctuation)
        return "".join(ch for ch in t if ch not in exclude)
    return white_space(remove_articles(remove_punc((answer or "").lower())))


def _token_f1(predicted: str, gold: str) -> float:
    p = _normalize_answer(predicted).split()
    g = _normalize_answer(gold).split()
    common = Counter(p) & Counter(g)
    n = sum(common.values())
    if n == 0:
        return 0.0
    prec = n / len(p)
    rec = n / len(g)
    return 2 * prec * rec / (prec + rec)


def parser_token_f1(predicted: str, golds: List[str]) -> float:
    """Max token-F1 across the gold answer set."""
    if not golds:
        return 0.0
    return float(np.max([_token_f1(predicted, g) for g in golds]))


# ---- NV-Embed cosine --------------------------------------------------------
def nvembed_cosine(encoder, preds: Dict[str, str], golds: Dict[str, str],
                    batch_size: int = 4) -> Dict[str, float]:
    """Per-question cosine between pred and gold, encoded by NV-Embed-v2."""
    texts: List[str] = []
    seen = set()
    for d in (preds, golds):
        for v in d.values():
            if v and v not in seen:
                texts.append(v)
                seen.add(v)
    vecs = encoder.encode(texts, kind="passage")
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    vecs = vecs / norms
    look = {t: v for t, v in zip(texts, vecs)}
    out: Dict[str, float] = {}
    for qid, pred in preds.items():
        gold = golds.get(qid, "")
        if not pred or not gold or pred not in look or gold not in look:
            out[qid] = None
            continue
        out[qid] = round(float(np.dot(look[pred], look[gold])), 4)
    return out


# ---- LF-Eval (DeepEval GEval, gpt-4o judge) --------------------------------
def lf_eval(predictions: Dict[str, str], samples: Dict[str, dict],
            criteria_path: str | Path = "prompts/lf_eval.json",
            judge_model: str = "gpt-4o",
            python_executable: str | None = None,
            output_path: str | Path | None = None) -> Dict[str, float]:
    """Run DeepEval GEval as a subprocess to keep its async stack isolated.

    samples[qid] must contain keys ``background_eng``, ``question_eng``,
    ``answer_eng`` and ``contexts`` (list of {index_eng, content_eng}).
    """
    criteria = json.loads(Path(criteria_path).read_text())
    if output_path is None:
        output_path = Path("results") / f"lf_eval_{judge_model}.json"
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "criteria": criteria,
        "judge_model": judge_model,
        "predictions": predictions,
        "samples": samples,
        "output_path": str(output_path),
    }
    tmp = Path(f"/tmp/_lf_eval_payload_{os.getpid()}_{judge_model}.json")
    tmp.write_text(json.dumps(payload, ensure_ascii=False))

    script = Path(__file__).parent / "_lf_eval_runner.py"
    py = python_executable or sys.executable
    env = os.environ.copy()
    env.setdefault("DEEPEVAL_TELEMETRY_OPT_OUT", "YES")
    env.setdefault("DEEPEVAL_UPDATE_WARNING_OPT_OUT", "YES")
    env.setdefault("ERROR_REPORTING", "NO")
    p = subprocess.run([py, str(script), str(tmp)],
                       env=env, check=False)
    if p.returncode != 0:
        raise RuntimeError(f"lf_eval subprocess failed (exit {p.returncode})")
    return json.loads(output_path.read_text())
