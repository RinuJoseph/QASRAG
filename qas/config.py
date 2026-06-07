"""Config loader for the QAS package.

Reads a YAML file, resolves `__DATASET__` placeholders against the active
dataset name, and derives every cache / output / dataset path *inside* the QAS
package root. Nothing points outside QAS/.

Layout (all relative to the QAS package root, i.e. the parent of `qas/`):
    cache_root   = QAS/cache/<dataset>
    output_root  = QAS/results/<dataset>
    dataset.root = QAS/dataset

The loader returns a plain dict (with attribute-style access also available via
the lightweight `Config` wrapper for the encoder singleton).
"""
from pathlib import Path

import yaml

# QAS package root = parent of this file's directory (qas/ -> QAS/)
PKG_ROOT = Path(__file__).resolve().parent.parent


def _resolve_placeholders(node, name):
    if isinstance(node, str):
        return node.replace("__DATASET__", name)
    if isinstance(node, dict):
        return {k: _resolve_placeholders(v, name) for k, v in node.items()}
    if isinstance(node, list):
        return [_resolve_placeholders(x, name) for x in node]
    return node


def load_config(config_path, dataset_override: str = None) -> dict:
    """Load YAML config, resolve `__DATASET__`, and derive QAS-internal paths.

    Args:
        config_path: path to the YAML config.
        dataset_override: if given, replaces `dataset.name` from the file.
    """
    p = Path(config_path).resolve()
    raw = yaml.safe_load(open(p))

    name = dataset_override or raw.get("dataset", {}).get("name", "musique")
    raw.setdefault("dataset", {})["name"] = name

    cfg = _resolve_placeholders(raw, name)

    def _anchor(value, default: Path) -> Path:
        """Resolve a configured path to an absolute path under the QAS root.

        Relative paths (e.g. 'QAS/cache/musique' or 'cache/musique') are
        interpreted relative to the package root, with a leading 'QAS/' segment
        stripped so it isn't doubled. Absolute paths are used verbatim.
        """
        if not value:
            return default
        p = Path(value)
        if p.is_absolute():
            return p
        parts = p.parts
        if parts and parts[0] == "QAS":
            parts = parts[1:]
        return PKG_ROOT.joinpath(*parts) if parts else PKG_ROOT

    # ---- Derive all paths under the QAS package root ----
    paths = cfg.setdefault("paths", {})
    cache_root = _anchor(paths.get("cache_root"), PKG_ROOT / "cache" / name)
    output_root = _anchor(paths.get("output_root"), PKG_ROOT / "results" / name)
    paths["cache_root"] = str(cache_root)
    paths["output_root"] = str(output_root)

    dset = cfg.setdefault("dataset", {})
    dset_root = _anchor(dset.get("root"), PKG_ROOT / "dataset")
    dset["root"] = str(dset_root)

    # Concrete cache sub-directories (one place that all stages read).
    cache = cfg.setdefault("cache", {})
    cache.setdefault("passage_embeddings", str(cache_root / "embeddings" / "passage"))
    cache.setdefault("embeddings_text",    str(cache_root / "embeddings" / "text"))
    cache.setdefault("ner_doc",            str(cache_root / "ner_doc"))
    cache.setdefault("triples",            str(cache_root / "triples"))
    cache.setdefault("decompositions",     str(cache_root / "decompositions"))

    cfg["output_dir"] = str(output_root)
    cfg["_config_path"] = str(p)
    cfg["_pkg_root"] = str(PKG_ROOT)
    return cfg


class EmbedderConfig:
    """Tiny attribute-access wrapper used by the encoder singleton.

    Pulls embedder fields out of the loaded config dict so `NVEmbedEncoder.get`
    can read `cfg.embedding_model` etc.
    """

    def __init__(self, cfg: dict):
        emb = cfg.get("embedder", {})
        self.embedding_model = emb.get("model_name", "nvidia/NV-Embed-v2")
        self.embedding_max_seq_len = int(emb.get("embedding_max_seq_len", 4096))
        self.embedding_batch_size = int(emb.get("embedding_batch_size", 8))
        self.embedding_dtype = emb.get("embedding_dtype", "float16")
