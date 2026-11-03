"""distributed.setup() under torchrun: backend and the longer collective timeout (Claude)."""

from datetime import timedelta

import pytest

torch = pytest.importorskip("torch")

from playground_lm.train import distributed as d  # noqa: E402


@pytest.fixture
def fake_torchrun(monkeypatch):
    calls = {}
    monkeypatch.setenv("WORLD_SIZE", "2")
    monkeypatch.setenv("RANK", "1")
    monkeypatch.setenv("LOCAL_RANK", "1")
    monkeypatch.setattr(d.torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(d.dist, "is_initialized", lambda: False)
    monkeypatch.setattr(d.dist, "init_process_group", lambda **kw: calls.update(kw))
    return calls


def test_setup_waits_up_to_an_hour_by_default(fake_torchrun, monkeypatch):
    monkeypatch.delenv("PLM_DIST_TIMEOUT_MIN", raising=False)
    info = d.setup()
    assert (info.rank, info.local_rank, info.world_size) == (1, 1, 2)
    assert not info.is_main and info.enabled
    assert fake_torchrun == {"backend": "gloo", "timeout": timedelta(minutes=60)}


def test_setup_timeout_can_be_changed(fake_torchrun, monkeypatch):

    monkeypatch.setenv("PLM_DIST_TIMEOUT_MIN", "5")
    d.setup()
    assert fake_torchrun["timeout"] == timedelta(minutes=5)
    d.setup(timeout_min=90)
    assert fake_torchrun["timeout"] == timedelta(minutes=90)
