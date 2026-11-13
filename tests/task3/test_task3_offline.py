"""End-to-end CLI run of task3_ttc.offline on tiny synthetic data."""
import json
import os
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest
import yaml

import task3_synth as synth
from common.runinfo import REPO_ROOT
from task3_ttc import selection as S

BUDGETS = [1, 2, 4]


def _run(module, cfg_path, ds, *extra):
    return subprocess.run([sys.executable, "-m", module, "--dataset", ds, "--config", cfg_path, *extra],
                          cwd=REPO_ROOT, capture_output=True, text=True)


@pytest.fixture(scope="module")
def prepared(tmp_path_factory):
    tmp = str(tmp_path_factory.mktemp("t3off"))
    cfg_path = synth.build(tmp)
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)
    cfg["budgets"] = BUDGETS
    cfg["selection"]["bootstrap_budgets"] = [2, 4]
    with open(cfg_path, "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
    for ds in ("math", "mathvista"):
        p = _run("task3_ttc.prepare", cfg_path, ds)
        assert p.returncode == 0, p.stderr
    return tmp, cfg_path


@pytest.mark.parametrize("ds", ["math", "mathvista"])
def test_offline_cli_end_to_end(prepared, ds):
    tmp, cfg_path = prepared
    p = _run("task3_ttc.offline", cfg_path, ds)
    assert p.returncode == 0, p.stderr + p.stdout
    off, sig_dir = os.path.join(tmp, "results", "offline"), os.path.join(tmp, "results", "signals")
    paths = {"per_question": os.path.join(off, f"{ds}_per_question.csv"),
             "selections": os.path.join(off, f"{ds}_selections.npz"),
             "summary": os.path.join(off, f"{ds}_summary.json"),
             "signals": os.path.join(sig_dir, f"{ds}_signals.json"),
             "bootstrap": os.path.join(off, f"{ds}_bootstrap.json")}
    for f in paths.values():
        assert os.path.exists(f), f
    methods = S.methods_for(ds)

    df = pd.read_csv(paths["per_question"], dtype={"sample_id": str})
    assert list(df.columns) == ["sample_id", "split", "method", "N", "accuracy", "oracle",
                                "distinct_groups_incl_no_answer", "distinct_groups_excl_no_answer", "cost",
                                "no_answer_selected", "sampling_failure", "selection_failure"]
    assert len(df) == 6 * len(methods) * len(BUDGETS)
    assert set(df["method"]) == set(methods) and set(df["N"]) == set(BUDGETS)
    assert np.allclose(df["accuracy"] + df["sampling_failure"] + df["selection_failure"], 1)
    assert (df["accuracy"] <= df["oracle"] + 1e-12).all()

    z = np.load(paths["selections"])
    for k in ("sample_id", "split", "methods", "budgets", "orders", "selected_rollout", "selected_group", "correct",
              "cost", "oracle", "distinct_groups_incl_no_answer", "distinct_groups_excl_no_answer"):
        assert k in z.files, k
    assert z["selected_rollout"].shape == (6, 10, len(BUDGETS), len(methods))
    assert z["orders"].shape == (6, 10, synth.N_ROLLOUTS)
    # N = 1: every method returns candidate 0 of the order
    assert (z["selected_rollout"][:, :, 0, :] == z["orders"][:, :, :1]).all()

    s = json.load(open(paths["summary"]))
    for k in ("run_id", "status", "dataset", "config", "run_info", "scorer_version", "inputs", "methods", "budgets",
              "eval", "dev", "by_type_eval", "construction_checks", "construction_checks_passed", "outputs",
              "runtime_s"):
        assert k in s, k
    assert s["construction_checks_passed"] is True and s["methods"] == methods
    assert all(c["passed"] is True for c in s["construction_checks"].values())
    assert s["eval"]["n_questions"] == 4 and s["dev"]["n_questions"] == 2
    for name in methods:
        for n in map(str, BUDGETS):
            e = s["eval"]["methods"][name][n]
            assert set(e) == {"accuracy", "cost", "no_answer_selected", "failure"}
            assert sum(e["failure"].values()) - e["failure"]["correct"] + e["accuracy"] == pytest.approx(1)
    ev = df[(df["split"] == "eval") & (df["method"] == "SC") & (df["N"] == 4)]
    assert s["eval"]["methods"]["SC"]["4"]["accuracy"] == pytest.approx(ev["accuracy"].mean())
    assert s["scorer_version"]["math_equivalence"].startswith("symmetric")

    sig = json.load(open(paths["signals"]))
    assert sig["n_questions"] == 4 and sig["quartile_k"] == 1 and "quartile_difference" in sig
    assert "spearman_T_vs_score" in sig
    b = json.load(open(paths["bootstrap"]))
    assert b["resamples"] == 1000 and b["seed"] == 6304
    pairs = {(c["a"], c["N_a"], c["b"], c["N_b"]) for c in b["comparisons"]}
    assert ("SC", 4, "FirstAns", 4) in pairs and ("RWC-K", 2, "Highest-K", 2) in pairs
    for c in b["comparisons"]:
        assert c["ci_low_pp"] <= c["mean_pp"] + 1e-9 and c["mean_pp"] <= c["ci_high_pp"] + 1e-9
    if ds == "mathvista":
        assert set(s["by_type_eval"]) <= {"multi_choice", "free_form"}
        assert s["construction_checks"]["covariance_identity"]["passed"] is True   # a JSON bool, not "True"
        assert sig["covariance_identity"]["passed"] is True
        assert "rollout_pearson" in sig and "Z_vs_Ztok_top" in sig
        assert ("RWC-Ztok", 4, "RWC-Z", 4) in pairs
    else:
        assert s["by_type_eval"] is None and "rollout_pearson" not in sig

    # never overwritten without --replace; --replace rewrites
    p2 = _run("task3_ttc.offline", cfg_path, ds)
    assert p2.returncode == 2 and "refusing to overwrite" in p2.stderr
    p3 = _run("task3_ttc.offline", cfg_path, ds, "--replace")
    assert p3.returncode == 0, p3.stderr
    s2 = json.load(open(paths["summary"]))
    assert s2["run_id"] != s["run_id"] or s2["runtime_s"] != s["runtime_s"]
    assert s2["eval"] == s["eval"]   # deterministic
