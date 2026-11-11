import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import pack_stage_g as P  # noqa: E402

SHA = "a" * 64
COMMIT = "b" * 40


def test_pack_g1_g2(tmp_path):
    assert P.main(["--out-root", str(tmp_path), "--suite-sha", SHA, "--git-commit", COMMIT,
                   "--tokens", "1865800,667442,624925", "--which", "G1,G2"]) == 0
    for name in ("plm-stage-g-g1", "plm-stage-g-g2"):
        meta = json.loads((tmp_path / name / "kernel-metadata.json").read_text())
        assert meta["code_file"] == f"{name}.py" and meta["machine_shape"] == "NvidiaTeslaT4"
        assert "galacticaiteam/plm-stage-g-data" in meta["dataset_sources"]
        env = json.loads((tmp_path / name / "cell.json").read_text())["env"]
        assert env["PLM_SEED_STEP"] == "31140" and env["PLM_STAGE_G_SUITE_SHA256"] == SHA
        assert float(env["PLM_GROUNDED_RATIO"]) > 0 and float(env["PLM_CHAT_RATIO"]) > 0
        assert env["PLM_MILESTONES"].split(",")[-1] == env["PLM_MAX_STEPS"]
        code = (tmp_path / name / f"{name}.py").read_text()
        assert "# --- Stage G cell overrides ---" in code and "def require_frozen_suite" in code
        compile(code, name, "exec")  # __future__ import must still come first
        assert code.index("from __future__") < code.index("# --- Stage G cell overrides ---")


def test_rejects_bad_hashes(tmp_path):
    runs = P.build_sweep_stage_g({"scratch": 1865800, "grounded": 667442, "chat": 624925})
    with pytest.raises(ValueError):
        P.cell_env(runs[0], suite_sha="x", git_commit=COMMIT, milestones="1")
    with pytest.raises(ValueError):
        P.cell_env(runs[0], suite_sha=SHA, git_commit="abc", milestones="1")
    with pytest.raises(FileExistsError):
        P.main(["--out-root", str(tmp_path), "--suite-sha", SHA, "--git-commit", COMMIT, "--tokens", "1865800,667442,624925", "--which", "G1"])
        P.main(["--out-root", str(tmp_path), "--suite-sha", SHA, "--git-commit", COMMIT, "--tokens", "1865800,667442,624925", "--which", "G1"])
