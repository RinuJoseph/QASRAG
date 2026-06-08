"""
e5-mistral-7b-instruct encoder,
"""
import threading
import numpy as np
from .encoder import get_query_instruction

_SINGLETON_LOCK = threading.Lock()
_GPU_LOCK = threading.Lock()


class E5MistralEncoder:
    _instance = None

    def __init__(self, model_name="intfloat/e5-mistral-7b-instruct",
                 max_seq_len=4096, batch_size=8):
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(
            model_name, device="cuda", model_kwargs={"torch_dtype": "float16"})
        self.model.max_seq_length = max_seq_len
        self.batch_size = batch_size

    @classmethod
    def get(cls, cfg) -> "E5MistralEncoder":
        if cls._instance is None:
            with _SINGLETON_LOCK:
                if cls._instance is None:
                    cls._instance = cls(
                        getattr(cfg, "embedding_model", "intfloat/e5-mistral-7b-instruct"),
                        max_seq_len=getattr(cfg, "embedding_max_seq_len", 4096),
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
