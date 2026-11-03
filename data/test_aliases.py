"""Regression test (Lumi): author aliases must match case-insensitively."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from prepare_playground import ALIASES, canonical_author  # noqa: E402


def test_alias_keys_are_lowercase():
    assert all(k == k.lower() for k in ALIASES)


def test_muse_maps_to_nova_any_case():
    for raw in ("Muse", "muse", "MUSE", " Nova "):
        assert canonical_author(raw) == "nova"
