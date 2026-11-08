"""End-to-end CLI run of task3_ttc.prepare on tiny synthetic data."""
import json
import os
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

import task3_synth as synth
from common.runinfo import REPO_ROOT
from task3_ttc.splits import load_split

BASE_COLS = ["sample_id", "rollout_id", "split", "T", "capped", "L", "K", "extracted", "normalized", "group_id",
             "na_reason", "correct"]
JSON_KEYS = ["run_id", "status", "dataset", "config_path", "config", "run_info", "scorer_version", "inputs",
             "outputs", "columns", "replaced", "counts", "checks", "runtime_s"]
COUNT_KEYS = ["rollouts", "questions", "rows_by_split", "questions_by_split", "live_rows", "n_tokens", "capped",
              "capped_flag_vs_length_disagreements", "groups_per_question_excl_no_answer",
              "groups_per_question_incl_no_answer", "distinct_strings_per_question", "no_answer_rollouts",
              "no_answer_by_reason", "questions_with_no_answer", "parse_failures", "verify_calls", "timeouts",
              "gold_failures"]


def _run(cfg_path, ds, *extra):
    return subprocess.run([sys.executable, "-m", "task3_ttc.prepare", "--dataset", ds, "--config", cfg_path, *extra],
                          cwd=REPO_ROOT, capture_output=True, text=True)


@pytest.fixture(scope="module")
def synth_cfg(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("t3")
    return str(tmp), synth.build(str(tmp))


@pytest.mark.parametrize("ds", ["math", "mathvista"])
def test_prepare_cli_end_to_end(synth_cfg, ds):
    tmp, cfg_path = synth_cfg
    p = _run(cfg_path, ds)
    assert p.returncode == 0, p.stderr
    csv_path = os.path.join(tmp, "results", "rollouts", f"{ds}_rollouts.csv")
    json_path = os.path.join(tmp, "results", f"prepare_{ds}.json")
    split_path = os.path.join(tmp, "splits", f"{ds}_split.json")
    cache_path = os.path.join(tmp, "cached", f"{ds}_tokens.npz")
    for f in (csv_path, json_path, split_path, cache_path):
        assert os.path.exists(f), f

    df = pd.read_csv(csv_path, keep_default_na=False, na_values={"V": [""], "L": [""], "K": [""]},
                     dtype={"sample_id": str, "extracted": str, "normalized": str, "na_reason": str})
    cols = list(BASE_COLS)
    if ds == "math":
        cols.insert(3, "live")
    else:
        cols.insert(cols.index("extracted"), "V")
    assert list(df.columns) == cols
    assert len(df) == 6 * synth.N_ROLLOUTS
    assert set(df["split"]) == {"dev", "eval"} and (df.groupby("sample_id")["split"].nunique() == 1).all()

    out = json.load(open(json_path))
    for k in JSON_KEYS:
        assert k in out, k
    for k in COUNT_KEYS:
        assert k in out["counts"], k
    c = out["counts"]
    assert c["rollouts"] == 24 and c["questions"] == 6
    assert c["rows_by_split"] == {"eval": 16, "dev": 8} or c["rows_by_split"] == {"dev": 8, "eval": 16}
    assert out["run_info"]["git"]["commit"] and out["scorer_version"]["id"]
    assert set(out["checks"]) == {"a_transitivity", "b_grading_consistency", "c_gold"}
    assert out["checks"]["b_grading_consistency"]["passed"] and out["checks"]["c_gold"]["passed"]

    sp = load_split(split_path)
    assert sp["sha256"] == out["outputs"]["split_sha256"]
    assert set(df.loc[df["split"] == "dev", "sample_id"]) == set(sp["dev_ids"])
    na = df["group_id"] == -1
    assert (~df.loc[na, "correct"]).all() and (df.loc[na, "na_reason"] != "").all()
    assert (df.loc[~na, "na_reason"] == "").all()
    assert c["no_answer_rollouts"] == int(na.sum())

    if ds == "math":
        assert df["live"].sum() == 2 * synth.N_ROLLOUTS
        assert set(df.loc[df["live"], "sample_id"]) == set(sp["live_ids"])
        assert c["no_answer_by_reason"] == {"extraction": 5, "normalization": 0}
        assert out["checks"]["a_transitivity"]["passed"]
        assert set(c["parse_failures"]) == {"distinct_strings", "rollouts", "examples"}
        q = df[df["sample_id"] == "test:10"].set_index("rollout_id")
        assert q.loc[0, "group_id"] == q.loc[1, "group_id"] == 0 and q.loc[2, "group_id"] == -1
        assert q.loc[0, "correct"] and q.loc[1, "correct"] and not q.loc[3, "correct"]
        q = df[df["sample_id"] == "test:2"].set_index("rollout_id")
        assert list(q["group_id"]) == [0, 1, 1, -1] and list(q["correct"]) == [True, False, False, False]
    else:
        assert c["no_answer_by_reason"] == {"extraction": 3, "normalization": 2}
        assert out["checks"]["a_transitivity"]["applicable"] is False
        assert df["V"].notna().all() and ((df["V"] > 0) & (df["V"] < 1)).all()
        q = df[df["sample_id"] == "4"].set_index("rollout_id")
        assert list(q["normalized"]) == ["C", "D", "C", ""] and list(q["correct"]) == [True, False, True, False]
        q = df[df["sample_id"] == "1"].set_index("rollout_id")
        assert list(q["normalized"]) == ["B", "C", "B", "B"] and list(q["group_id"]) == [0, 1, 0, 0]

    # outputs are never overwritten
    p2 = _run(cfg_path, ds)
    assert p2.returncode != 0 and "refusing to overwrite" in p2.stderr
    assert json.load(open(json_path))["run_id"] == out["run_id"]
    assert out["replaced"] is None

    # --replace overwrites and records what it replaced
    p3 = _run(cfg_path, ds, "--replace")
    assert p3.returncode == 0, p3.stderr
    new = json.load(open(json_path))
    assert new["run_id"] != out["run_id"] and new["replaced"]["run_id"] == out["run_id"]
    assert pd.read_csv(csv_path).shape == df.shape


def test_capped_column_matches_flag(synth_cfg):
    tmp, cfg_path = synth_cfg
    csv_path = os.path.join(tmp, "results", "rollouts", "math_rollouts.csv")
    if not os.path.exists(csv_path):
        assert _run(cfg_path, "math").returncode == 0
    df = pd.read_csv(csv_path)
    assert df["capped"].sum() == 1 and (df.loc[df["capped"], "T"] == synth.CAP).all()
    assert np.isfinite(df[["L", "K"]].to_numpy()).all()
