"""Stage F packer tests. Slug==code_file idea from Dee; slugs are the verified
Kaggle datasets, not placeholders, and E31140 is the weights dataset
plm-e31140-seed (not the old plm-stage-e-train kernel).
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

pack = Path(__file__).parent
spec = importlib.util.spec_from_file_location("pack_stage_f", pack / "pack_stage_f.py")
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)


def test_verified_slugs_not_placeholders():
    assert "galacticaiteam/plm-stage-f-data" in mod.DATASETS
    assert "galacticaiteam/plm-e31140-seed" in mod.DATASETS
    assert "galacticaiteam/plm-stageb-data" in mod.DATASETS
    assert "galacticaiteam/plm-stagec-val" in mod.DATASETS
    joined = " ".join(mod.DATASETS)
    assert "REPLACE" not in joined
    assert "plm-stage-e-train" not in joined


def test_pack_stage_f_slug_equals_code_file(tmp_path):
    template = tmp_path / "template.py"
    template.write_text("print('ok')\n")
    cell = mod.package_cell(
        template, tmp_path / "cells",
        slug="plm-stage-f-e31140-rr25-ep1-lr1e5",
        run_id="f-rr25", epochs=1, lr=1e-5, max_steps=31204,
        replay_ratio=0.25, git_commit="d2aa141ac75a07f645a5b4d9b99189ad98896822",
    )
    meta = json.loads((cell / "kernel-metadata.json").read_text())
    assert meta["code_file"] == "plm-stage-f-e31140-rr25-ep1-lr1e5.py"
    assert (cell / meta["code_file"]).is_file()
    assert set(meta["dataset_sources"]) == set(mod.DATASETS)
    src = (cell / meta["code_file"]).read_text()
    assert "PLM_WEIGHTS_ONLY_RESUME" in src
    assert "PLM_SCHEDULE_START" in src


def test_pack_injects_chat_ratio(tmp_path):
    template = tmp_path / "template.py"
    template.write_text("print('ok')\n")
    cell = mod.package_cell(
        template, tmp_path / "cells",
        slug="plm-stage-f-e31140-rr50-ch17-ep3-lr1e5",
        run_id="stage-f-e31140-rr50-ch17-ep3-lr1e-5",
        epochs=3, lr=1e-5, max_steps=31340,
        replay_ratio=0.50, chat_ratio=1/6,
        git_commit="a" * 40,
    )
    src = (cell / "plm-stage-f-e31140-rr50-ch17-ep3-lr1e5.py").read_text()
    assert "PLM_CHAT_RATIO" in src
    assert "PLM_CHAT_DIR" in src
    meta = json.loads((cell / "cell.json").read_text())
    assert abs(meta["chat_ratio"] - 1/6) < 1e-9
