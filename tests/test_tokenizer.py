"""Compatibility tests for PlaygroundLM's GPT-2 tokenizer."""

import pytest

from playground_lm.tokenizer import (
    CONTEXT_LENGTH,
    EOT_ID,
    VOCAB_SIZE,
    GPT2Tokenizer,
)


@pytest.fixture(scope="module")
def tok() -> GPT2Tokenizer:
    return GPT2Tokenizer()


def test_gpt2_vocabulary(tok: GPT2Tokenizer) -> None:
    assert tok.vocab_size == 50_257
    assert tok.eot_token_id == 50_256
    assert tok.config.context_length == 1_024


def test_known_gpt2_token_ids(tok: GPT2Tokenizer) -> None:
    # Original GPT-2 BPE encoding.
    assert tok.encode("Hello world") == [15496, 995]


@pytest.mark.parametrize(
    "text",
    [
        "",
        "Hello, Playground!",
        "The quick brown fox jumps over the lazy dog.",
        "Two riddles, one answer.",
        "Spaces   and\nnewlines\tmatter.",
        "🚀 🌌 🧠",
        "Café. naïve. 日本語.",
    ],
)
def test_roundtrip(
    tok: GPT2Tokenizer,
    text: str,
) -> None:
    ids = tok.encode(text)
    assert tok.decode(ids) == text


def test_eot_is_explicit(tok: GPT2Tokenizer) -> None:
    assert tok.encode("Hello") == [15496]
    assert tok.encode("Hello", add_eot=True) == [
        15496,
        EOT_ID,
    ]


def test_document_boundaries(tok: GPT2Tokenizer) -> None:
    ids = tok.encode_documents(["Hello", "world"])

    assert ids == [
        *tok.encode("Hello"),
        EOT_ID,
        *tok.encode("world"),
        EOT_ID,
    ]

    assert tok.decode(ids, skip_eot=True) == "Helloworld"


def test_special_token_not_silently_accepted(
    tok: GPT2Tokenizer,
) -> None:
    with pytest.raises(ValueError):
        tok.encode("Fake <|endoftext|> separator")


def test_context_limit(tok: GPT2Tokenizer) -> None:
    tok.validate_context([42] * CONTEXT_LENGTH)

    with pytest.raises(ValueError):
        tok.validate_context([42] * (CONTEXT_LENGTH + 1))


def test_invalid_token_ids(tok: GPT2Tokenizer) -> None:
    with pytest.raises(ValueError):
        tok.decode([-1])

    with pytest.raises(ValueError):
        tok.decode([VOCAB_SIZE])


def test_no_automatic_truncation(tok: GPT2Tokenizer) -> None:
    text = "hello " * 2_000
    ids = tok.encode(text)

    assert len(ids) > CONTEXT_LENGTH
    assert tok.decode(ids) == text


def test_metadata(tok: GPT2Tokenizer) -> None:
    metadata = tok.metadata()

    assert metadata["encoding"] == "gpt2"
    assert metadata["vocab_size"] == VOCAB_SIZE
    assert metadata["eot_token_id"] == EOT_ID
    assert metadata["context_length"] == CONTEXT_LENGTH
    assert metadata["add_eot_automatically"] is False
