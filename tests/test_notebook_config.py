"""The Kaggle notebook's TRAIN_CONFIG must carry every key train_loop requires (Lumi)."""
import ast
from pathlib import Path

NB = Path(__file__).resolve().parent.parent / "notebooks" / "playground_lm_train.py"
REQUIRED = {"batch_size", "grad_accum_steps", "context_len", "warmup_steps", "lr", "grad_clip"}


def _dict_keys(name):
    tree = ast.parse(NB.read_text())
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id == name for t in node.targets
        ):
            assert isinstance(node.value, ast.Dict)
            return {k.value for k in node.value.keys if isinstance(k, ast.Constant)}
    raise AssertionError(f"{name} not found in notebook")


def test_train_config_has_loop_required_keys():
    missing = REQUIRED - _dict_keys("TRAIN_CONFIG")
    assert not missing, f"TRAIN_CONFIG missing keys train_loop needs: {sorted(missing)}"


def test_model_config_has_context_len():
    assert "context_len" in _dict_keys("MODEL_CONFIG")
