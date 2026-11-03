"""GPT-2 byte-level BPE tokenizer for PlaygroundLM.

Compatible with GPT-2's 50,257 IDs. All model weights are random-init.
When PLM_CACHE_DIR is set, tokenizer assets MUST already be cached.
"""

from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import tiktoken

VOCAB_SIZE = 50_257
EOT_ID = 50_256
CONTEXT_LENGTH = 1_024
ENCODING_NAME = "gpt2"

# Public GPT-2 tokenizer files used by tiktoken's "gpt2" encoding.
# Keys are the source URLs; cache filenames are SHA-1(URL).
ASSETS = {
    "https://openaipublic.blob.core.windows.net/gpt-2/encodings/main/vocab.bpe":
        "1ce1664773c50f3e0cc8842619a93edc4624525b728b188a9e0be33b7726adc5",
    "https://openaipublic.blob.core.windows.net/gpt-2/encodings/main/encoder.json":
        "196139668be63f3b5d6574427317ae82f612a97c5d1cdaf36ed2256dbf636783",
}


@dataclass(frozen=True)
class TokenizerConfig:
    name: str = ENCODING_NAME
    vocab_size: int = VOCAB_SIZE
    eot_token_id: int = EOT_ID
    context_length: int = CONTEXT_LENGTH


def cache_filename(url: str) -> str:
    return hashlib.sha1(url.encode("utf-8")).hexdigest()


def configure_cache() -> Path | None:
    """Configure tiktoken and validate an explicitly supplied cache.

    PLM_CACHE_DIR is strict/offline: assets must be present and valid.
    Without it, tiktoken retains its normal download/cache behavior.
    """
    value = os.environ.get("PLM_CACHE_DIR")
    if value is None:
        return None

    if not value.strip():
        raise ValueError("PLM_CACHE_DIR cannot be empty")

    directory = Path(value).expanduser().resolve()

    if not directory.is_dir():
        raise FileNotFoundError(
            f"PLM_CACHE_DIR is not a directory: {directory}"
        )

    for url, expected_sha256 in ASSETS.items():
        asset = directory / cache_filename(url)
        if not asset.is_file():
            raise FileNotFoundError(
                f"Missing offline GPT-2 tokenizer asset: {asset}. "
                "Warm the tiktoken cache in an online environment "
                "and copy both files to PLM_CACHE_DIR."
            )

        actual = hashlib.sha256(asset.read_bytes()).hexdigest()
        if actual != expected_sha256:
            raise ValueError(
                f"Corrupt tokenizer cache asset: {asset}"
            )

    # tiktoken reads this variable when it loads encoding assets.
    os.environ["TIKTOKEN_CACHE_DIR"] = str(directory)
    return directory


class GPT2Tokenizer:
    """Validated GPT-2 tokenizer with no automatic special tokens."""

    def __init__(self) -> None:
        configure_cache()
        self.config = TokenizerConfig()
        self._encoding = tiktoken.get_encoding(ENCODING_NAME)

        if self._encoding.n_vocab != VOCAB_SIZE:
            raise ValueError("Unexpected GPT-2 vocabulary size")

        eot = self._encoding.encode(
            "<|endoftext|>",
            allowed_special={"<|endoftext|>"},
        )
        if eot != [EOT_ID]:
            raise ValueError(f"Unexpected EOT token ID: {eot}")

    @property
    def vocab_size(self) -> int:
        return VOCAB_SIZE

    @property
    def eot_token_id(self) -> int:
        return EOT_ID

    def encode(
        self, text: str, *, add_eot: bool = False
    ) -> list[int]:
        if not isinstance(text, str):
            raise TypeError("text must be a string")

        ids = self._encoding.encode(
            text,
            allowed_special=set(),
            disallowed_special="all",
        )
        if add_eot:
            ids.append(EOT_ID)
        return ids

    def decode(
        self,
        token_ids: Sequence[int],
        *,
        skip_eot: bool = False,
    ) -> str:
        ids = [int(i) for i in token_ids]
        if any(i < 0 or i >= VOCAB_SIZE for i in ids):
            raise ValueError("Token ID outside GPT-2 vocabulary")
        if skip_eot:
            ids = [i for i in ids if i != EOT_ID]
        return self._encoding.decode(ids)

    def encode_documents(
        self, documents: Iterable[str]
    ) -> list[int]:
        result: list[int] = []
        for document in documents:
            result.extend(self.encode(document, add_eot=True))
        return result

    def count_tokens(self, text: str) -> int:
        return len(self.encode(text))

    def validate_context(
        self, token_ids: Sequence[int]
    ) -> None:
        if len(token_ids) > CONTEXT_LENGTH:
            raise ValueError(
                f"Sequence contains {len(token_ids)} tokens; "
                f"maximum is {CONTEXT_LENGTH}"
            )

    def metadata(self) -> dict[str, object]:
        return {
            "encoding": ENCODING_NAME,
            "vocab_size": VOCAB_SIZE,
            "eot_token_id": EOT_ID,
            "context_length": CONTEXT_LENGTH,
            "add_bos_automatically": False,
            "add_eot_automatically": False,
        }


def get_tokenizer() -> GPT2Tokenizer:
    return GPT2Tokenizer()
