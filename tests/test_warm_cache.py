import hashlib
import importlib.util
from pathlib import Path

import pytest

from playground_lm import tokenizer as tok

SPEC = importlib.util.spec_from_file_location(
    "warm", Path(__file__).resolve().parent.parent / "scripts" / "warm_tokenizer_cache.py"
)
warm_mod = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(warm_mod)


def test_rejects_bad_hash(tmp_path):
    with pytest.raises(ValueError):
        warm_mod.warm(tmp_path, fetch=lambda url: b"not the real file")


def test_writes_tiktoken_names(tmp_path, monkeypatch):
    fake = {url: f"asset {i}".encode() for i, url in enumerate(tok.ASSETS)}
    monkeypatch.setattr(
        warm_mod, "ASSETS", {u: hashlib.sha256(b).hexdigest() for u, b in fake.items()}
    )
    calls = []
    paths = warm_mod.warm(tmp_path, fetch=lambda url: calls.append(url) or fake[url])
    assert sorted(p.name for p in paths) == sorted(tok.cache_filename(u) for u in fake)
    # Second run reuses valid files without fetching again.
    warm_mod.warm(tmp_path, fetch=lambda url: pytest.fail("refetched"))
    assert len(calls) == 2
