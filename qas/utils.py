
import json
import hashlib
import re
from pathlib import Path
from typing import Any


def text_processing(text: str) -> str:
    """entity normalization (hr2_norm). Lowercase, replace any
    character outside [A-Za-z0-9 space] with a space, then strip.
    """
    if not isinstance(text, str):
        text = str(text)
    return re.sub(r"[^A-Za-z0-9 ]", " ", text.lower()).strip()


# alias matching the spec's "hr2_norm" name
hr2_norm = text_processing


def text_hash(text: str) -> str:
    """16-char SHA1 hex digest of text (for cache filenames)."""
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:16]


def chunk_id(title: str, text: str) -> str:
    """chunk_id scheme: 'chunk-' + md5(title + '\\n' + text)."""
    h = hashlib.md5(f"{title}\n{text}".encode("utf-8")).hexdigest()
    return f"chunk-{h}"


def save_json(obj: Any, path, indent: int = 2) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=indent, ensure_ascii=False, default=str)


def load_json(path) -> Any:
    with open(path) as f:
        return json.load(f)
