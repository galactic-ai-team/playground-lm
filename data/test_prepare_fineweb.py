"""Resume tests for data/prepare_fineweb.py (Lumi). Run: python -m pytest data/

A fake `datasets` module streams deterministic docs, so no network is used.
The script ends with os._exit(0), so it runs in a subprocess.
"""

import json
import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).parent

FAKE_DATASETS = '''
import random
def load_dataset(*a, **k):
    rng = random.Random(0)
    words = ["alpha", "beta", "gamma", "delta", "omega", "sigma", "kappa"]
    def gen():
        for i in range(400):
            body = " ".join(rng.choice(words) for _ in range(120))
            yield {"text": f"Doc {i}. " + body}
    return gen()
'''


def _run(tmp_path: Path, out: Path, target: int) -> None:
    fake = tmp_path / "fake"
    fake.mkdir(exist_ok=True)
    (fake / "datasets.py").write_text(FAKE_DATASETS)
    env = dict(os.environ, PYTHONPATH=f"{fake}{os.pathsep}{HERE}")
    subprocess.run(
        [sys.executable, str(HERE / "prepare_fineweb.py"), "--target-tokens", str(target),
         "--out", str(out), "--val-pct", "10"],
        check=True, env=env, cwd=HERE, capture_output=True,
    )


def _contents(out: Path) -> dict[str, str]:
    return {p.name: p.read_text() for p in sorted(out.glob("*.jsonl"))}


def _ids(out: Path) -> list[str]:
    return [json.loads(l)["id"] for t in _contents(out).values() for l in t.splitlines()]


def test_resume_matches_single_run(tmp_path):
    one, two = tmp_path / "one", tmp_path / "two"
    _run(tmp_path, one, 30_000)
    _run(tmp_path, two, 12_000)   # stop early ...
    _run(tmp_path, two, 30_000)   # ... then resume to the same target
    assert _contents(one) == _contents(two)
    ids = _ids(two)
    assert len(ids) == len(set(ids)), "resume duplicated documents"


def test_rerun_after_completion_is_noop_and_drops_uncheckpointed_lines(tmp_path):
    out = tmp_path / "out"
    _run(tmp_path, out, 20_000)
    before = _contents(out)
    # Simulate a crash after lines were written but before the manifest update.
    with (out / "fineweb-val.jsonl").open("a") as f:
        f.write('{"id": "orphan", "text": "x"}\n')
    _run(tmp_path, out, 20_000)
    assert _contents(out) == before
    manifest = json.loads((out / "fineweb-manifest.json").read_text())
    assert manifest["docs_seen"] > 0 and manifest["tokens_written"] >= 20_000
