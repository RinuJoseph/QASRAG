"""Alibaba-NLP/gte-Qwen2-7B-instruct encoder, drop-in for NVEmbedEncoder.

Same surface (.encode(texts, kind) -> (N,D) float32 L2-normalized; .get(cfg)).
gte-Qwen2 uses the SAME instruction scheme as NV-Embed/E5 ("Instruct: <task>\\n
Query: <text>" for the query side, no instruction for documents), so we reuse
qas.encoder's instruction strings -> only the embedding model changes vs NV-Embed.
Loaded via sentence-transformers (last-token pooling + normalize), fp16 on CUDA.
"""
import threading
import numpy as np

from .encoder import get_query_instruction

_SINGLETON_LOCK = threading.Lock()
_GPU_LOCK = threading.Lock()


class GTEQwen2Encoder:
    _instance = None

    def __init__(self, model_name="Alibaba-NLP/gte-Qwen2-7B-instruct",
                 max_seq_len=8192, batch_size=8):
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(
            model_name, trust_remote_code=True, device="cuda",
            model_kwargs={"torch_dtype": "float16"})
        self.model.max_seq_length = max_seq_len
        self.batch_size = batch_size

    @classmethod
    def get(cls, cfg) -> "GTEQwen2Encoder":
        if cls._instance is None:
            with _SINGLETON_LOCK:
                if cls._instance is None:
                    cls._instance = cls(
                        getattr(cfg, "embedding_model", "Alibaba-NLP/gte-Qwen2-7B-instruct"),
                        max_seq_len=getattr(cfg, "embedding_max_seq_len", 8192),
                        batch_size=getattr(cfg, "embedding_batch_size", 8))
        return cls._instance

    def encode(self, texts, kind: str = "query_to_passage") -> np.ndarray:
        if isinstance(texts, str):
            texts = [texts]
        if not texts:
            return np.zeros((0, 0), np.float32)
        instr = "" if kind in ("passage", "doc", "") else get_query_instruction(kind)
        prompt = (f"Instruct: {instr}\nQuery: " if instr else None)
        with _GPU_LOCK:
            emb = self.model.encode(
                list(texts), batch_size=self.batch_size, prompt=prompt,
                normalize_embeddings=True, convert_to_numpy=True,
                show_progress_bar=False)
        return np.asarray(emb, dtype=np.float32)
