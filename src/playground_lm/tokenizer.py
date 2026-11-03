"""GPT-2-compatible byte-level BPE tokenizer for PlaygroundLM.

Tokenizer assets are reused; model weights are always trained from scratch.

Contract:
    vocabulary size: 50_257
    end-of-text ID: 50_256
    context length: 1_024
    no automatic BOS/EOS insertion

The first tiktoken load may download its public tokenizer assets.
Subsequent loads use tiktoken's local cache.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import tiktoken

VOCAB_SIZE = 50_257
EOT_ID = 50_256
CONTEXT_LENGTH = 1_024
ENCODING_NAME = "gpt2"


@dataclass(frozen=True)
class TokenizerConfig:
    name: str = ENCODING_NAME
    vocab_size: int = VOCAB_SIZE
    eot_token_id: int = EOT_ID
    context_length: int = CONTEXT_LENGTH


class GPT2Tokenizer:
    """Thin, validated wrapper around GPT-2's original BPE encoding."""

    def __init__(self) -> None:
        self.config = TokenizerConfig()
        self._encoding = tiktoken.get_encoding(ENCODING_NAME)

        if self._encoding.n_vocab != VOCAB_SIZE:
            raise ValueError(
                "Unexpected GPT-2 vocabulary size: "
                f"{self._encoding.n_vocab}"
            )

        actual_eot = self._encoding.encode(
            "<|endoftext|>",
            allowed_special={"<|endoftext|>"},
        )

        if actual_eot != [EOT_ID]:
            raise ValueError(
                f"Unexpected EOT token ID: {actual_eot}"
            )

    @property
    def vocab_size(self) -> int:
        return VOCAB_SIZE

    @property
    def eot_token_id(self) -> int:
        return EOT_ID

    def encode(
        self,
        text: str,
        *,
        add_eot: bool = False,
    ) -> list[int]:
        """Encode text without automatic special tokens.

        A literal special-token spelling in source text is rejected,
        rather than silently interpreted as a document boundary.
        """
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
        """Decode GPT-2 IDs, optionally stripping EOT separators."""
        ids = [int(i) for i in token_ids]

        if any(i < 0 or i >= VOCAB_SIZE for i in ids):
            raise ValueError("Token ID outside GPT-2 vocabulary")

        if skip_eot:
            ids = [i for i in ids if i != EOT_ID]

        return self._encoding.decode(ids)

    def encode_documents(
        self,
        documents: Iterable[str],
    ) -> list[int]:
        """Flatten documents, separated by one EOT token each.

        Intended for dataset construction. For large corpora,
        stream documents in shards instead of accumulating this list.
        """
        result: list[int] = []

        for document in documents:
            result.extend(self.encode(document, add_eot=True))

        return result

    def count_tokens(self, text: str) -> int:
        return len(self.encode(text))

    def validate_context(
        self,
        token_ids: Sequence[int],
    ) -> None:
        """Reject sequences too long for GPT-2-small's context."""
        if len(token_ids) > CONTEXT_LENGTH:
            raise ValueError(
                f"Sequence contains {len(token_ids)} tokens; "
                f"maximum is {CONTEXT_LENGTH}"
            )

    def metadata(self) -> dict[str, object]:
        """Stable metadata for training and checkpoint manifests."""
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
