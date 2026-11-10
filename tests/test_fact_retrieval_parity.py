import json
import shutil
import subprocess
from pathlib import Path

import pytest

from playground_lm.fact_retrieval import build_index, make_retriever, retrieve, tokenize

ROOT = Path(__file__).resolve().parents[1]
SHEET = ROOT / "data" / "facts" / "fact_sheet.jsonl"
QUERIES = [
    "What is the capital of Albania?", "Who wrote Hamlet?", "How many legs does a spider have?",
    "boiling point of water", "the and of", "", "zzzz qqqq", "capital capital capital",
    "Which planet is largest?", "What is the chemical symbol for gold?",
]


def test_tokenize_stopwords():
    assert tokenize("The Capital of FRANCE, is?") == ["capital", "of", "france"]


def test_zero_scores_and_ties():
    idx = build_index([{"id": "b", "text": "red apple"}, {"id": "a", "text": "red apple"}, {"id": "c", "text": "blue"}])
    assert [r["id"] for r in retrieve(idx, "red")] == ["a", "b"]
    assert retrieve(idx, "green") == []


def test_retriever_sheet_sha():
    fn, sha = make_retriever(SHEET)
    assert len(sha) == 64 and isinstance(fn("capital of Albania"), list)


@pytest.mark.skipif(shutil.which("node") is None, reason="node not installed")
def test_parity_with_demo_js():
    facts = [json.loads(l) for l in SHEET.read_text().splitlines() if l.strip()]
    queries = QUERIES + [f["text"] for f in facts[::25]] + [" ".join(f.get("keywords", [])) for f in facts[3::40]]
    script = (
        "import('" + (ROOT / "demo" / "fact_retrieval.js").as_uri() + "').then(m=>{"
        "const fs=require('fs');const inp=JSON.parse(fs.readFileSync(0,'utf8'));"
        "const idx=m.buildIndex(inp.facts);"
        "process.stdout.write(JSON.stringify(inp.queries.map(q=>m.retrieve(idx,q,{k:3}))));});"
    )
    res = subprocess.run(["node", "-e", script], input=json.dumps({"facts": facts, "queries": queries}),
                         capture_output=True, text=True, check=True)
    js = json.loads(res.stdout)
    idx = build_index(facts)
    for q, want in zip(queries, js):
        got = retrieve(idx, q, k=3)
        assert [r["id"] for r in got] == [r["id"] for r in want], q
        for g, w in zip(got, want):
            assert abs(g["score"] - w["score"]) < 1e-9
