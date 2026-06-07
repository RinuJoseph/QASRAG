"""Subprocess entry-point for DeepEval GEval Legal Fidelity scoring."""
import contextlib
import io
import json
import os
import sys

os.environ.setdefault("DEEPEVAL_TELEMETRY_OPT_OUT", "YES")
os.environ.setdefault("DEEPEVAL_UPDATE_WARNING_OPT_OUT", "YES")
os.environ.setdefault("ERROR_REPORTING", "NO")

import numpy as np
from deepeval import evaluate
from deepeval.metrics import GEval
from deepeval.test_case import LLMTestCase, LLMTestCaseParams

try:
    from deepeval.evaluate.configs import DisplayConfig
    _DISPLAY_CONFIG = DisplayConfig(verbose_mode=False, print_results=False,
                                    show_indicator=False)
except Exception:
    _DISPLAY_CONFIG = None


def _params(spec):
    mapping = {
        "INPUT": LLMTestCaseParams.INPUT,
        "ACTUAL_OUTPUT": LLMTestCaseParams.ACTUAL_OUTPUT,
        "EXPECTED_OUTPUT": LLMTestCaseParams.EXPECTED_OUTPUT,
        "CONTEXT": LLMTestCaseParams.CONTEXT,
    }
    return [mapping[p] for p in spec]


def main(payload_path: str) -> None:
    payload = json.loads(open(payload_path).read())
    criteria = payload["criteria"]
    samples = payload["samples"]
    preds = payload["predictions"]
    output_path = payload["output_path"]

    geval = GEval(
        name=criteria["name"],
        criteria=criteria["criteria"],
        evaluation_steps=criteria["evaluation_steps"],
        async_mode=True,
        evaluation_params=_params(criteria["evaluation_params"]),
        model=payload["judge_model"],
    )

    qids = [q for q in samples if q in preds and preds[q]]
    cases = []
    for q in qids:
        sample = samples[q]
        question = (sample.get("background_eng", "") + " " +
                    sample.get("question_eng", "")).strip()
        context = [
            (c.get("index_eng", "") + " " + c.get("content_eng", "")).strip()
            for c in sample.get("contexts", [])
        ]
        cases.append(LLMTestCase(
            input=question,
            actual_output=preds[q],
            expected_output=sample.get("answer_eng", ""),
            context=context,
        ))
    kwargs = {"test_cases": cases, "metrics": [geval]}
    if _DISPLAY_CONFIG is not None:
        kwargs["display_config"] = _DISPLAY_CONFIG
    sink = io.StringIO()
    with contextlib.redirect_stdout(sink), contextlib.redirect_stderr(sink):
        result = evaluate(**kwargs)

    per_query = {}
    for n, tr in enumerate(result.test_results):
        idx = getattr(tr, "index", None)
        if idx is None or idx >= len(qids):
            idx = n
        per_query[qids[idx]] = round(tr.metrics_data[0].score, 4)
    json.dump(per_query, open(output_path, "w"), indent=1, ensure_ascii=False)
    print(f"LF mean = {float(np.mean(list(per_query.values()))):.4f}")


if __name__ == "__main__":
    main(sys.argv[1])
