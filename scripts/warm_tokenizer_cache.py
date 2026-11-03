"""Download the two GPT-2 tokenizer files into a folder for offline use.

Run this once somewhere with internet, then upload the folder as a private
Kaggle dataset (e.g. "plm-tokenizer-cache") and set
PLM_CACHE_DIR=/kaggle/input/plm-tokenizer-cache in the training notebook.
get_tokenizer() then loads with no network access.

    python scripts/warm_tokenizer_cache.py out/tokenizer-cache
"""

from __future__ import annotations

import hashlib
import sys
import urllib.request
from pathlib import Path

from playground_lm.tokenizer import ASSETS, cache_filename


def warm(directory: Path, fetch=None) -> list[Path]:
    """Fetch each asset, check its SHA-256, and save it under tiktoken's name."""
    fetch = fetch or (lambda url: urllib.request.urlopen(url, timeout=60).read())
    directory.mkdir(parents=True, exist_ok=True)
    written = []
    for url, expected in ASSETS.items():
        target = directory / cache_filename(url)
        if target.is_file() and hashlib.sha256(target.read_bytes()).hexdigest() == expected:
            written.append(target)
            continue
        data = fetch(url)
        actual = hashlib.sha256(data).hexdigest()
        if actual != expected:
            raise ValueError(f"SHA-256 mismatch for {url}: got {actual}")
        target.write_bytes(data)
        written.append(target)
    return written


if __name__ == "__main__":
    out = Path(sys.argv[1] if len(sys.argv) > 1 else "tokenizer-cache")
    for path in warm(out):
        print(path)
