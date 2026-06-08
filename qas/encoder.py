"""
NV-Embed-v2 encoder + embedding caches —
"""
import json
import threading
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from .utils import text_hash

# ---------------------------------------------------------------------------
# NV-Embed-v2 task instructions.
# ---------------------------------------------------------------------------
INSTRUCTIONS = {
    "ner_to_node":      "Given a phrase, retrieve synonymous or relevant phrases that best match this phrase.",
    "query_to_node":    "Given a question, retrieve relevant phrases that are mentioned in this question.",
    "query_to_fact":    "Given a question, retrieve relevant triplet facts that matches this question.",
    "query_to_sentence": "Given a question, retrieve relevant sentences that best answer the question.",
    "query_to_passage": "Given a question, retrieve relevant documents that best answer the question.",
}
_DEFAULT_INSTRUCTION = "Given a question, retrieve relevant documents that best answer the question."


def get_query_instruction(kind: str) -> str:
    """Return the NV-Embed instruction string for a task kind.

    The default (unknown kind) is the query_to_passage instruction.
    """
    return INSTRUCTIONS.get(kind, _DEFAULT_INSTRUCTION)


_GPU_LOCK = threading.Lock()       # serialises GPU forward passes
_SINGLETON_LOCK = threading.Lock()  # guards singleton construction


class NVEmbedEncoder:
    """Thread-safe singleton wrapper around nvidia/NV-Embed-v2."""

    _instance = None

    def __init__(self, model_name: str = "nvidia/NV-Embed-v2",
                 max_seq_len: int = 4096, batch_size: int = 8,
                 dtype: str = "float16"):
        import torch
        from transformers import AutoModel
        td = {"float16": torch.float16, "bfloat16": torch.bfloat16,
              "float32": torch.float32}[dtype]
        # NV-Embed-v2's custom code ignores torch_dtype and loads most weights
        # in float32, so cast on CPU then move to GPU as one block.
        self.model = AutoModel.from_pretrained(
            model_name, trust_remote_code=True, low_cpu_mem_usage=True)
        self.model = self.model.to(td).to("cuda")
        self.model.eval()
        self.max_seq_len = max_seq_len
        self.batch_size = batch_size

    @classmethod
    def get(cls, cfg) -> "NVEmbedEncoder":
        """Return the process-wide singleton, constructing it once.

        `cfg` is an EmbedderConfig (see qas.config.EmbedderConfig) exposing
        embedding_model / embedding_max_seq_len / embedding_batch_size /
        embedding_dtype.
        """
        if cls._instance is None:
            with _SINGLETON_LOCK:
                if cls._instance is None:
                    cls._instance = cls(cfg.embedding_model,
                                        cfg.embedding_max_seq_len,
                                        cfg.embedding_batch_size,
                                        cfg.embedding_dtype)
        return cls._instance

    def _encode_batch(self, texts: List[str], instruction: str) -> np.ndarray:
        import torch
        instr = (f"Instruct: {instruction}\nQuery: " if instruction else "")
        with _GPU_LOCK, torch.no_grad():
            out = self.model.encode(prompts=list(texts),
                                    instruction=instr,
                                    max_length=self.max_seq_len)
        if isinstance(out, torch.Tensor):
            out = out.cpu().numpy()
        out = np.asarray(out, dtype=np.float32)
        norms = np.linalg.norm(out, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return out / norms

    def encode(self, texts, kind: str = "query_to_passage") -> np.ndarray:
        """Encode text(s) and L2-normalize.

        kind in {query_to_passage, query_to_fact, ner_to_node, query_to_node,
        query_to_sentence} -> instruction template. kind in {passage, doc, ""}
        -> document side, NO instruction (NV-Embed default for docs).
        """
        if isinstance(texts, str):
            texts = [texts]
        instruction = "" if kind in ("passage", "doc", "") \
            else get_query_instruction(kind)
        chunks = []
        for i in range(0, len(texts), self.batch_size):
            chunks.append(self._encode_batch(
                texts[i:i + self.batch_size], instruction))
        return np.vstack(chunks) if chunks else np.zeros((0, 0), np.float32)


def min_max_normalize(x: np.ndarray) -> np.ndarray:
    """min-max normalize."""
    x = np.asarray(x, dtype=np.float64)
    lo, hi = x.min(), x.max()
    if hi - lo < 1e-12:
        return np.zeros_like(x)
    return (x - lo) / (hi - lo)


class EmbeddingStore:
    """Disk-cached embeddings backed by the NV-Embed singleton.

    Holds the embedder config so it can lazily construct the encoder, and
    exposes the three caching helpers the pipeline needs:
      - embed_text / embed_texts : per-text instruction-conditioned vectors
      - embed_passages           : the corpus passage matrix (NO instruction)
    """

    def __init__(self, embedder_cfg, text_cache_dir, passage_cache_dir):
        self.embedder_cfg = embedder_cfg
        self.text_cache_dir = Path(text_cache_dir)
        self.passage_cache_dir = Path(passage_cache_dir)

    # -- encoder access ----------------------------------------------------
    def _encoder(self) -> NVEmbedEncoder:
        return NVEmbedEncoder.get(self.embedder_cfg)

    # -- per-text cache ----------------------------------------------------
    def _text_path(self, text: str, kind: str) -> Path:
        return self.text_cache_dir / f"{text_hash(f'{kind}:{text}')}.npy"

    def embed_text(self, text: str, kind: str = "query_to_passage") -> np.ndarray:
        p = self._text_path(text, kind)
        if p.exists():
            return np.load(p)
        return self.embed_texts([text], kind)[0]

    def embed_texts(self, texts: List[str],
                    kind: str = "query_to_passage") -> np.ndarray:
        """Batched encoding with a per-text disk cache.

        Used for queries / sub-questions / facts (instruction-conditioned).
        """
        self.text_cache_dir.mkdir(parents=True, exist_ok=True)
        n = len(texts)
        out = [None] * n
        todo, todo_idx = [], []
        for i, t in enumerate(texts):
            p = self._text_path(t, kind)
            if p.exists():
                out[i] = np.load(p)
            else:
                todo.append(t)
                todo_idx.append(i)
        if todo:
            enc = self._encoder()
            embs = enc.encode(todo, kind)
            for offset, t in enumerate(todo):
                emb = np.asarray(embs[offset], dtype=np.float32)
                out[todo_idx[offset]] = emb
                np.save(self._text_path(t, kind), emb)
        return np.stack(out)

    # -- passage matrix cache ---------------------------------------------
    def _passage_paths(self) -> Tuple[Path, Path]:
        return (self.passage_cache_dir / "passage_embeddings.npy",
                self.passage_cache_dir / "passage_keys.json")

    def embed_passages(self, corpus: Dict[str, dict]) -> Tuple[np.ndarray, List[str]]:
        """Encode every passage in the corpus into one (N, D) NPY matrix.

        Passages are encoded with NO instruction (NV-Embed default for docs)
        and L2-normalized so dot == cosine. Idempotent: a cached matrix whose
        key order matches the corpus is reused.
        """
        npy_p, keys_p = self._passage_paths()
        self.passage_cache_dir.mkdir(parents=True, exist_ok=True)
        chunk_ids = list(corpus.keys())
        if npy_p.exists() and keys_p.exists():
            existing_keys = json.load(open(keys_p))
            if existing_keys == chunk_ids:
                return np.load(npy_p), existing_keys
            print("[passage_emb] cache key order differs from corpus — rebuilding.")

        enc = self._encoder()
        bs = self.embedder_cfg.embedding_batch_size
        n = len(chunk_ids)
        print(f"[passage_emb] encoding {n} passages (NV-Embed-v2, batch={bs}) …")
        out = None
        t0 = time.time()
        for i in range(0, n, bs):
            chunk = chunk_ids[i:i + bs]
            texts = [corpus[c]["content"] for c in chunk]
            embs = enc.encode(texts, "passage")
            embs = np.asarray(embs, dtype=np.float32)
            if embs.ndim == 1:
                embs = embs[None, :]
            if out is None:
                out = np.zeros((n, embs.shape[1]), dtype=np.float32)
            out[i:i + len(chunk)] = embs
            if (i // bs) % 50 == 0 or i + bs >= n:
                done = min(i + bs, n)
                rate = done / max(1, time.time() - t0)
                print(f"  [{done:>5d}/{n}] {time.time()-t0:.0f}s  ({rate:.1f}/s)",
                      flush=True)
        np.save(npy_p, out)
        json.dump(chunk_ids, open(keys_p, "w"))
        print(f"[passage_emb] saved -> {npy_p} ({out.nbytes / 1e6:.1f} MB)")
        return out, chunk_ids

    def passage_embeddings_and_keys(self, corpus: Dict[str, dict] = None
                                    ) -> Tuple[np.ndarray, List[str]]:
        """Load the passage matrix from cache, or build it if absent.

        Building requires `corpus`. Raises if the cache is missing and no
        corpus is provided.
        """
        npy_p, keys_p = self._passage_paths()
        if npy_p.exists() and keys_p.exists():
            return np.load(npy_p), json.load(open(keys_p))
        if corpus is None:
            raise FileNotFoundError(
                f"Passage embeddings cache missing at {self.passage_cache_dir}. "
                f"Run scripts/build_cache.py first.")
        return self.embed_passages(corpus)
