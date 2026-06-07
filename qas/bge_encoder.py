"""BAAI/bge-large-en-v1.5 encoder (BERT encoder, 335M), drop-in for NVEmbedEncoder.

Same surface (.encode(texts, kind) -> (N,D) float32 L2-normalized; .get(cfg)).
bge-v1.5 uses a single retrieval instruction prepended to the QUERY side only
("Represent this sentence for searching relevant passages: "); documents get no
instruction. So any query-side kind -> bge query instr; passage/doc -> none.
Loaded via sentence-transformers, fp16 on CUDA. dim=1024, max_seq_len=512.
"""
import threading
import numpy as np

_SINGLETON_LOCK = threading.Lock()
_GPU_LOCK = threading.Lock()

_BGE_QUERY_INSTR = "Represent this sentence for searching relevant passages: "


class BGEEncoder:
    _instance = None

    def __init__(self, model_name="BAAI/bge-large-en-v1.5",
                 max_seq_len=512, batch_size=64):
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(
            model_name, device="cuda", model_kwargs={"torch_dtype": "float16"})
        self.model.max_seq_length = max_seq_len
        self.batch_size = batch_size

    @classmethod
    def get(cls, cfg) -> "BGEEncoder":
        if cls._instance is None:
            with _SINGLETON_LOCK:
                if cls._instance is None:
                    cls._instance = cls(
                        getattr(cfg, "embedding_model", "BAAI/bge-large-en-v1.5"),
                        max_seq_len=getattr(cfg, "embedding_max_seq_len", 512),
                        batch_size=getattr(cfg, "embedding_batch_size", 64))
        return cls._instance

    def encode(self, texts, kind: str = "query_to_passage") -> np.ndarray:
        if isinstance(texts, str):
            texts = [texts]
        if not texts:
            return np.zeros((0, 0), np.float32)
        is_doc = kind in ("passage", "doc", "")
        prompt = None if is_doc else _BGE_QUERY_INSTR
        with _GPU_LOCK:
            emb = self.model.encode(
                list(texts), batch_size=self.batch_size, prompt=prompt,
                normalize_embeddings=True, convert_to_numpy=True,
                show_progress_bar=False)
        return np.asarray(emb, dtype=np.float32)
