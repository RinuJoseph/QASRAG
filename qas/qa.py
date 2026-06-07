"""Final QA (step 7 of the QAS pipeline).

Feed the top-K passages to the LLM with rag_qa_musique one-shot
prompt, then parse the text after the "Answer:" marker.
"""
import re
from typing import Dict, List

from qas_prompts.rag_qa_musique import build_messages

from .llm import LLMClient


def parse_answer(response: str) -> str:
    """Extract text after the 'Answer:' marker (case-insensitive)."""
    if response is None:
        return ""
    m = re.search(r"answer\s*:\s*(.+?)(?:$|\n)",
                  response, re.IGNORECASE | re.DOTALL)
    if m:
        return m.group(1).strip()
    return response.strip()


class QAReader:
    def __init__(self, llm: LLMClient, top_k_passages: int = 5,
                 temperature: float = 0.0, max_tokens: int = 400):
        self.llm = llm
        self.top_k_passages = top_k_passages
        self.temperature = temperature
        self.max_tokens = max_tokens

    def answer(self, question: str, top_passages: List[dict],
               corpus: Dict[str, dict]) -> dict:
        """Run QA on one query with its top passages.

        top_passages: list of {chunk_id, title, ...}; text is fetched from
        corpus. Returns {raw_response, predicted_answer}.
        """
        passages = [
            {"title": corpus[p["chunk_id"]]["title"],
             "text":  corpus[p["chunk_id"]]["text"]}
            for p in top_passages[:self.top_k_passages]
        ]
        messages = build_messages(passages, question)
        raw = self.llm.chat(messages, temperature=self.temperature,
                            max_tokens=self.max_tokens) or ""
        return {
            "raw_response":     raw,
            "predicted_answer": parse_answer(raw),
        }
