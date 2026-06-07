"""LLM listwise selector.

Single API call per query: the full candidate pool is shown to the model
together with a chain-of-thought legal selection prompt (see
``prompts/rerank_select.txt``). The model returns a free-form reasoning
trace ending in ``### Final Selection: [i] [j] [...]`` over the candidate
indices. The selection order becomes the reranked passage order.
"""
from __future__ import annotations

import asyncio
import re
from pathlib import Path
from typing import Dict, List, Tuple

import aiohttp
from tenacity import retry, stop_after_attempt, wait_random_exponential


FINAL_SELECTION_RE = re.compile(
    r"###\s*Final\s*Selection\s*:\s*((?:\[\d+\]\s*)+)", re.IGNORECASE)


class LLMListwiseReranker:
    def __init__(self, openai_api_key: str,
                 model: str = "gpt-4o-mini",
                 prompt_path: str | Path = "prompts/rerank_select.txt",
                 max_parallel: int = 16,
                 temperature: float = 0.0,
                 max_tokens: int = 2000,
                 base_url: str = "https://api.openai.com/v1"):
        self.api_key = openai_api_key
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.prompt_template = Path(prompt_path).read_text()
        self.max_parallel = max_parallel
        self.temperature = temperature
        self.max_tokens = max_tokens

    def _build_prompt(self, question: str,
                      candidates: List[Tuple[str, str]]) -> str:
        lines = []
        for i, (title, text) in enumerate(candidates, start=1):
            lines.append(f"[{i}] {title}\n{text}")
        context = "\n\n".join(lines)
        return self.prompt_template.format(
            num=len(candidates), question=question, context=context)

    def _parse_selection(self, text: str, n: int) -> List[int]:
        if not text:
            return []
        m = FINAL_SELECTION_RE.search(text)
        if not m:
            return []
        ids = [int(x) for x in re.findall(r"\[(\d+)\]", m.group(1))]
        seen, kept = set(), []
        for i in ids:
            if 1 <= i <= n and i not in seen:
                seen.add(i)
                kept.append(i - 1)
        return kept

    async def _call_one(self, prompt: str, session, sem) -> str:
        @retry(wait=wait_random_exponential(min=1, max=30),
               stop=stop_after_attempt(6),
               retry_error_callback=lambda _: "")
        async def _do():
            async with sem:
                async with session.post(
                        f"{self.base_url}/chat/completions",
                        headers={
                            "Content-Type": "application/json",
                            "Authorization": f"Bearer {self.api_key}",
                        },
                        json={
                            "model": self.model,
                            "messages": [{"role": "user", "content": prompt}],
                            "temperature": self.temperature,
                            "max_tokens": self.max_tokens,
                        }) as r:
                    j = await r.json()
                    return ((j.get("choices", [{}])[0].get("message", {}) or {})
                            .get("content", "") or "")
        return await _do()

    async def rerank_async(self, items: List[Dict]) -> List[Dict]:
        """items: list of {'qid','question','pool':[(cid,title,text), ...]}."""
        sem = asyncio.Semaphore(self.max_parallel)
        timeout = aiohttp.ClientTimeout(total=600)

        async def _task(it: Dict) -> Dict:
            cids = [c for c, _, _ in it["pool"]]
            cands = [(t, x) for _, t, x in it["pool"]]
            if not cands:
                return {**it, "ranked_cids": [], "raw_llm_output": ""}
            prompt = self._build_prompt(it["question"], cands)
            text = await self._call_one(prompt, session, sem)
            picks = self._parse_selection(text, len(cands))
            ranked = [cids[i] for i in picks]
            return {**it,
                    "ranked_cids": ranked,
                    "n_selected": len(ranked),
                    "raw_llm_output": text}

        async with aiohttp.ClientSession(timeout=timeout,
                                          trust_env=True) as session:
            return await asyncio.gather(*[_task(it) for it in items])

    def rerank(self, items: List[Dict]) -> List[Dict]:
        return asyncio.run(self.rerank_async(items))
