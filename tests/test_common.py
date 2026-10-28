import json
import random

import numpy as np
import pytest

from common.results import ResultOverwriteError, read_result, write_result
from common.runinfo import run_info
from common.seed import DEFAULT_SEED, seed_everything


def test_default_seed():
    assert DEFAULT_SEED == 6304


def test_seed_reproducible():
    seed_everything(6304)
    a = (random.random(), np.random.rand())
    seed_everything(6304)
    b = (random.random(), np.random.rand())
    assert a == b


def test_seed_torch_if_available():
    torch = pytest.importorskip("torch")
    seed_everything(6304)
    x = torch.rand(3)
    seed_everything(6304)
    assert torch.equal(x, torch.rand(3))


def test_run_info_fields():
    info = run_info()
    assert set(info) >= {"git", "hostname", "hardware", "timestamp"}
    assert set(info["git"]) == {"commit", "dirty"}
    assert "cpu" in info["hardware"]


def test_write_and_rewrite_same_run(tmp_path):
    p = tmp_path / "r.json"
    write_result(p, {"run_id": "a", "step": 1})
    write_result(p, {"run_id": "a", "step": 2})
    assert read_result(p)["step"] == 2
    assert [f.name for f in tmp_path.iterdir()] == ["r.json"]  # no temp leftovers


def test_refuses_other_run(tmp_path):
    p = tmp_path / "r.json"
    write_result(p, {"run_id": "a"})
    with pytest.raises(ResultOverwriteError):
        write_result(p, {"run_id": "b"})
    assert read_result(p)["run_id"] == "a"


def test_refuses_foreign_file(tmp_path):
    p = tmp_path / "r.json"
    p.write_text(json.dumps({"no_run_id": True}))
    with pytest.raises(ResultOverwriteError):
        write_result(p, {"run_id": "a"})
    p.write_text("not json")
    with pytest.raises(ResultOverwriteError):
        write_result(p, {"run_id": "a"})


def test_requires_run_id(tmp_path):
    with pytest.raises(ValueError):
        write_result(tmp_path / "r.json", {"x": 1})


def test_creates_parent_dir(tmp_path):
    p = tmp_path / "sub" / "r.json"
    write_result(p, {"run_id": "a"})
    assert p.exists()
