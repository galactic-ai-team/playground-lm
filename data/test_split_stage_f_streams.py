"""Offline tests for splitting Stage F train into facts vs chat streams."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from split_stage_f_streams import split_rows, FACT_SOURCES, CHAT_SOURCES


def test_split_rows_keeps_order_and_covers_known_sources():
    train = [
        {"id": "c1", "source": "databricks/databricks-dolly-15k", "user": "hi", "assistant": "hello"},
        {"id": "f1", "source": "openai/gsm8k", "user": "q", "assistant": "1"},
        {"id": "c2", "source": "generated-identity", "user": "who", "assistant": "bot"},
        {"id": "f2", "source": "generated-arithmetic", "user": "1+1", "assistant": "2"},
    ]
    facts, chat = split_rows(train)
    assert [r["id"] for r in facts] == ["f1", "f2"]
    assert [r["id"] for r in chat] == ["c1", "c2"]
    assert FACT_SOURCES & CHAT_SOURCES == set()


def test_split_rows_rejects_unknown_source():
    try:
        split_rows([{"source": "triviaqa", "user": "q", "assistant": "a"}])
    except ValueError as e:
        assert "unclassified" in str(e)
    else:
        raise AssertionError("expected ValueError")
