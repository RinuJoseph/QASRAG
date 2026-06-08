"""
QA reader.

"""
from __future__ import annotations

import re
from pathlib import Path
from string import Template
from typing import Dict, List


class LegalQAReader:
    def __init__(self, llm_client,
                 system_prompt_path: str | Path = "prompts/qa_system.txt",
                 instruction_prompt_path: str | Path = "prompts/qa_instruction.txt",
                 passage_format: str = "{title} {text}",
                 temperature: float = 0.0,
                 max_tokens: int = 2000):
        self.llm = llm_client
        self.system_prompt = Path(system_prompt_path).read_text().rstrip("\n")
        self.instruction = Template(Path(instruction_prompt_path).read_text())
        self.passage_format = passage_format
        self.temperature = temperature
        self.max_tokens = max_tokens

    def build_context(self, chunks: List[Dict]) -> str:
        return "\n".join(self.passage_format.format(title=c['title'],
                                                       text=c['text'])
                          for c in chunks)

    def build_question(self, background: str, question: str) -> str:
        return (background + " " + question).strip()

    def _parse_answer(self, response: str) -> str:
        if response is None:
            return ""
        m = re.search(r"answer\s*:\s*(.+?)(?:$|\n)",
                      response, re.IGNORECASE | re.DOTALL)
        if m:
            return m.group(1).strip()
        return response.strip()

    def answer(self, background: str, question: str,
               chunks: List[Dict]) -> Dict:
        q_text = self.build_question(background, question)
        ctx = self.build_context(chunks)
        user = self.instruction.substitute(question=q_text, context_str=ctx)
        prompt = f"{self.system_prompt}\n\n{user}"
        raw = self.llm.chat([{"role": "user", "content": prompt}],
                            temperature=self.temperature,
                            max_tokens=self.max_tokens) or ""
        return {"raw": raw, "predicted_answer": self._parse_answer(raw)}
