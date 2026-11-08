"""Task 3 splits (L593-596) and rollout orders (L608-610)."""
import json
import os

import numpy as np
import pytest

from common.hashing import sha256_json
from common.runinfo import REPO_ROOT
from task3 import splits
from task3.data import load_config

IDS = [f"test:{i}" for i in range(1000)]


def _content(ids=IDS, n_live=50):
    sp = splits.make_split(ids, 6304, 200, 800, n_live, 10, 32)
    return sp, splits.split_content("math", sp, 6304, 200, 800, n_live, 10, 32)


def test_split_reproduction():
    sp, c = _content(list(reversed(IDS)))  # input order must not matter
    s = sorted(IDS)
    perm = np.random.default_rng(6304).permutation(1000)
    expect = [s[i] for i in perm]
    assert sp["sorted_ids"] == s and s[:3] == ["test:0", "test:1", "test:10"]  # lexicographic
    assert sp["dev_ids"] == expect[:200] and sp["eval_ids"] == expect[200:]
    assert sp["live_ids"] == expect[200:250]
    assert not set(sp["dev_ids"]) & set(sp["eval_ids"]) and len(set(sp["dev_ids"]) | set(sp["eval_ids"])) == 1000
    assert c["sha256"] == sha256_json({k: v for k, v in c.items() if k != "sha256"})
    assert "live_ids" in c and splits.make_split(IDS, 6304, 200, 800, None, 10, 32)["live_ids"] is None


def test_orders_are_permutations():
    sp, c = _content()
    o = sp["orders"]
    assert o.shape == (1000, 10, 32)
    assert (np.sort(o, axis=-1) == np.arange(32)).all()
    rng = np.random.default_rng(6304)  # separate generator, questions in sorted-ID order
    first = [rng.permutation(32) for _ in range(10)]
    np.testing.assert_array_equal(o[0], np.stack(first))
    np.testing.assert_array_equal(o[1, 0], rng.permutation(32))
    arr = splits.orders_array(json.loads(splits.dumps_split(c)))
    np.testing.assert_array_equal(arr, o)


def test_split_file_roundtrip_and_mismatch(tmp_path):
    cfg = load_config()
    cfg["paths"]["splits_dir"] = str(tmp_path)
    cfg["paths"]["splits_preview"] = None
    c, path, status = splits.ensure_split(cfg, "math", IDS)
    assert status == "written" and os.path.basename(path) == "math_split.json"
    loaded = splits.load_split(path)
    assert loaded == c and len(loaded["live_ids"]) == 50
    assert splits.ensure_split(cfg, "math", IDS)[2] == "unchanged"
    with pytest.raises(splits.SplitMismatchError):
        splits.ensure_split(cfg, "math", IDS[:-1] + ["test:x"])


@pytest.mark.skipif(not os.path.exists(os.path.join(REPO_ROOT, "scratch/task3/splits_preview.json")),
                    reason="prompt-a preview not present")
def test_preview_check_detects_difference():
    prev = os.path.join(REPO_ROOT, "scratch/task3/splits_preview.json")
    _, c = _content()
    with pytest.raises(AssertionError):
        splits.check_preview("math", c, prev)  # synthetic IDs are not the real MATH IDs


@pytest.mark.parametrize("ds,n_live", [("math", 50), ("mathvista", None)])
def test_committed_split_files_reproduce(ds, n_live):
    path = os.path.join(REPO_ROOT, "splits", "task3", f"{ds}_split.json")
    if not os.path.exists(path):
        pytest.skip("split file not written yet")
    c = splits.load_split(path)  # checks the stored sha256
    assert len(c["sorted_ids"]) == 1000 and c["sorted_ids"] == sorted(c["sorted_ids"])
    sp = splits.make_split(c["sorted_ids"], 6304, 200, 800, n_live, 10, 32)
    assert splits.split_content(ds, sp, 6304, 200, 800, n_live, 10, 32) == c
    prev = os.path.join(REPO_ROOT, "scratch/task3/splits_preview.json")
    if os.path.exists(prev):
        splits.check_preview(ds, c, prev)
