"""Evaluation metrics — F1 / EM / gold-set + Recall@k.
are directly comparable to :
  - normalize_answer
  - token F1 + EM
  - gold_set per sample: answer / gold_ans
                            / reference / obj + answer_aliases, plus koblex
                            answer_eng.
  - recall_at_k            (retrieval recall metric)
"""
import re
import string
from collections import Counter
from typing import List

import numpy as np


def normalize_answer(answer: str) -> str:
    """normalize_answer: lowercase, drop punctuation, drop
    articles (a/an/the), collapse whitespace."""
    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(answer or ""))))


def compute_f1(predicted: str, gold: str) -> float:
    """token-overlap F1 on normalized strings."""
    gold_tokens = normalize_answer(gold).split()
    pred_tokens = normalize_answer(predicted).split()
    common = Counter(pred_tokens) & Counter(gold_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(pred_tokens)
    recall = num_same / len(gold_tokens)
    return 2 * (precision * recall) / (precision + recall)


def compute_em(predicted: str, gold: str) -> float:
    return 1.0 if normalize_answer(predicted) == normalize_answer(gold) else 0.0


def gold_set(sample: dict) -> List[str]:
    """Replicate main.get_gold_answers for ONE sample (+ koblex).

    Looks for answer / gold_ans, else reference, else obj (+possible_answers,
    o_wiki_title, o_aliases), else koblex answer_eng. Then updates with
    answer_aliases. Returns a list of non-empty string golds.
    """
    s = sample
    gold_ans = None
    if "answer" in s or "gold_ans" in s:
        gold_ans = s["answer"] if "answer" in s else s["gold_ans"]
    elif "reference" in s:
        gold_ans = s["reference"]
    elif "obj" in s:
        gold_ans = list(set(
            [s.get("obj")] + list(s.get("possible_answers") or [])
            + [s.get("o_wiki_title")] + list(s.get("o_aliases") or [])))
    elif "answer_eng" in s:                      # KoBLEX
        gold_ans = s["answer_eng"]
    assert gold_ans is not None, f"no gold in sample keys {list(s.keys())}"
    if isinstance(gold_ans, str):
        gold_ans = [gold_ans]
    gold_ans = set(gold_ans)
    if s.get("answer_aliases"):
        gold_ans.update(s["answer_aliases"])
    return [g for g in gold_ans if isinstance(g, str) and g.strip()]


def score(predicted: str, sample: dict):
    """(f1, em) = max over the gold set for one sample."""
    golds = gold_set(sample)
    if not golds:
        return 0.0, 0.0
    f1 = float(np.max([compute_f1(predicted, g) for g in golds]))
    em = float(np.max([1.0 if normalize_answer(g) == normalize_answer(predicted)
                       else 0.0 for g in golds]))
    return f1, em


def recall_at_k(retrieved: List[str], gold: List[str], k: int):
    """Fraction of gold chunk_ids found in retrieved[:k]. None if no gold."""
    g = [x for x in gold if x]
    if not g:
        return None
    topk = set(retrieved[:k])
    return sum(1 for x in g if x in topk) / len(g)
