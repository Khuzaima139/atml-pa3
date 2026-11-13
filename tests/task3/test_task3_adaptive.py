"""Task 3 Step 5 adaptive sampling."""
import json
import os
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest
import yaml

import task3_synth as synth
from common.runinfo import REPO_ROOT, file_git_status
from task3_ttc import adaptive as A
from task3_ttc import selection as S
from task3_ttc.splits import load_split, orders_array

TAUS = [0.5, 0.625, 0.75, 0.875, 1.0]
GAMMAS = [0.0, 0.1, 0.25, 0.5]


def _question(rng, R=32, n_groups=3, na=True, T_max=40):
    T = rng.integers(2, T_max, R)
    g = rng.integers(-1 if na else 0, n_groups, R)
    return {"T": T, "L": rng.normal(size=R), "K": rng.normal(size=R).round(1), "g": g,
            "correct": g == 0, "V": rng.random(R),
            "kl": [rng.random(t) * 10 for t in T], "v": [rng.random(t) for t in T]}


def _states(n=4, count=400, seed=0):
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(count):
        q = _question(rng, n_groups=int(rng.integers(1, 4)))
        order = rng.permutation(32)
        for score in ("K", "Z"):
            out.append(A.state(q, order[:n], score))
    return out


def test_n4_tau_equivalences():
    for st in _states():
        for g in GAMMAS:
            assert A.quality_stop(st, 0.625, g) == A.quality_stop(st, 0.75, g)
            assert A.quality_stop(st, 0.875, g) == A.quality_stop(st, 1.0, g)
        assert A.agreement_stop(st, 0.625) == A.agreement_stop(st, 0.75)
        assert A.agreement_stop(st, 0.875) == A.agreement_stop(st, 1.0)


def test_tau_tolerance_at_exact_fraction():
    # 0.875 * 8 = 7 exactly; 0.625 * 8 = 5
    st = {"n": 8, "leader": 0, "top_count": 7, "unique": True}
    assert A.agreement_stop(st, 0.875) and not A.agreement_stop(st, 1.0)
    assert A.agreement_stop({**st, "top_count": 5}, 0.625) and not A.agreement_stop({**st, "top_count": 4}, 0.625)


@pytest.mark.parametrize("groups", [[0, 0, 1, 1], [-1, -1, -1, 0], [-1, -1, -1, -1], [0, 1, 2, 3]])
def test_tied_top_or_no_answer_leader_never_stops(groups):
    R = 8
    g = np.array(groups + [0] * (R - 4))
    q = {"T": np.full(R, 5), "K": np.arange(R, dtype=float), "g": g, "correct": g == 0}
    st = A.state(q, np.arange(4), "K")
    for t in TAUS:
        assert not A.agreement_stop(st, t)
        for gm in GAMMAS:
            assert not A.quality_stop(st, t, gm)


def test_gamma_zero_is_agreement_plus_winner_agreement():
    seen = set()
    for st in _states(n=8, count=300, seed=1):
        for t in TAUS:
            want = A.agreement_stop(st, t) and st["rwc_winner"] == st["leader"]
            assert A.quality_stop(st, t, 0.0) == want
            seen.add(want)
    assert seen == {True, False}


def test_margin_one_group_and_hand_example():
    assert A.rwc_margin([2, 2, 2, 2], [0.1, 0.5, 0.2, 0.9]) == 1.0
    # K ranks: rollout 3 best ... rollout 0 worst; w*(N-1) = 0, 1, 2, 3 -> W(0) = 3, W(1) = 3: tie, margin 0
    q = {"T": np.array([1, 2, 3, 4]), "K": np.array([1.0, 2, 3, 4]), "g": np.array([0, 0, 0, 1]),
         "correct": np.array([True] * 3 + [False])}
    st = A.state(q, np.arange(4), "K")
    assert st == {"n": 4, "leader": 0, "top_count": 3, "unique": True, "cost": 10, "rwc_winner": 0, "margin": 0.0}
    assert A.quality_stop(st, 0.75, 0.0) and not A.quality_stop(st, 0.75, 0.1)
    # margin with NO_ANSWER as a group: groups (-1, 0, 0, 0), K = 4, 1, 2, 3 -> W(-1) = 3, W(0) = 0 + 1 + 2
    assert A.rwc_margin([-1, 0, 0, 0], [4.0, 1, 2, 3]) == 0.0
    assert A.rwc_margin([-1, 0, 0, 0], [1.0, 2, 3, 4]) == pytest.approx((6 - 0) / 6)


def test_unrevealed_rollouts_do_not_change_n4_decisions():
    rng = np.random.default_rng(2)
    for _ in range(50):
        q = _question(rng)
        order = rng.permutation(32)
        before = {s: A.state(q, order[:4], s) for s in ("K", "Z")}
        q2 = {k: (list(v) if isinstance(v, list) else v.copy()) for k, v in q.items()}
        for i in order[4:]:
            q2["kl"][i] = q2["kl"][i] * 100 + 50
            q2["v"][i] = 1 - q2["v"][i]
            q2["g"][i] = (q2["g"][i] + 1) % 3
            q2["K"][i] = -q2["K"][i]
            q2["V"][i] = 0.0
            q2["T"][i] = 1
        for s in ("K", "Z"):
            assert A.state(q2, order[:4], s) == before[s]
        sts = [A.trace(q, order, [4], s)[0][0] for s in ("K", "Z")]
        assert sts == [before["K"], before["Z"]]


def test_decide_and_simulate():
    q = {"T": np.arange(1, 9), "K": np.arange(8, dtype=float), "g": np.array([0, 0, 0, 0, 1, 1, 1, 1]),
         "correct": np.array([True] * 4 + [False] * 4)}
    orders = np.array([[np.arange(8), np.arange(8)[::-1]]])
    pols = [{"family": "agreement", "tau": 1.0, "gamma": None},
            {"family": "quality", "tau": 1.0, "gamma": 0.5}, A.FALLBACK]
    sim = A.simulate([q], orders, pols, [2, 4], "K")
    assert sim["stop"][0, 0].tolist() == [2, 2, 8] and sim["cost"][0, 0].tolist() == [3, 3, 36]
    # full pool tie 4-4 -> earliest supporting rollout in the order: group 0 for order 0, group 1 for order 1
    assert sim["group"][0, :, 2].tolist() == [0, 1]
    assert sim["correct"][0, 1].tolist() == [False, False, False]
    s = A.summarize(sim, 0, [2, 4, 8])
    assert s["mean_rollouts"] == 2 and s["stop_distribution"] == {"2": 1.0, "4": 0.0, "8": 0.0}


def _row(name, fam, tau, gamma, acc, cost):
    return {"name": name, "policy": {"family": fam, "tau": tau, "gamma": gamma}, "accuracy": acc, "mean_cost": cost}


def test_selection_tie_breaks():
    fb = _row("fallback", "fallback", None, None, 0.70, 900.0)
    rows = [_row("a", "quality", 0.5, 0.0, 0.60, 100.0),          # cheapest but ineligible
            _row("b", "quality", 0.75, 0.25, 0.695, 200.0),
            _row("c", "quality", 0.625, 0.25, 0.70, 200.0),       # same cost, higher accuracy wins
            _row("d", "quality", 0.875, 0.0, 0.70, 200.0), fb]
    assert A.choose(rows, 0.70, 1.0)[0]["name"] == "c"
    rows = [_row("x", "quality", 0.75, 0.5, 0.70, 200.0), _row("y", "quality", 0.75, 0.1, 0.70, 200.0),
            _row("z", "quality", 0.875, 0.0, 0.70, 200.0), fb]
    assert A.choose(rows, 0.70, 1.0)[0]["name"] == "y"          # smaller tau first, then smaller gamma
    # threshold is inclusive (with float tolerance): 0.69 >= 0.70 - 0.01
    assert A.choose([_row("e", "agreement", 0.5, None, 0.69, 10.0), fb], 0.70, 1.0)[0]["name"] == "e"
    # nothing else eligible: the fallback is chosen even if its accuracy were below the threshold
    assert A.choose([_row("f", "agreement", 0.5, None, 0.5, 10.0), {**fb, "accuracy": 0.0}], 0.70, 1.0)[0][
        "name"] == "fallback"


def _git(d, *args):
    subprocess.run(["git", "-C", str(d), *args], check=True, capture_output=True)


def _init_repo(d):
    _git(d, "init", "-q")
    _git(d, "config", "user.email", "t@example.com")
    _git(d, "config", "user.name", "t")
    _git(d, "config", "commit.gpgsign", "false")
    _git(d, "config", "core.hooksPath", "/dev/null")


def test_file_git_status(tmp_path):
    f = tmp_path / "x.json"
    f.write_text("{}")
    assert file_git_status(f)["tracked"] is False          # not in a repository
    _init_repo(tmp_path)
    assert file_git_status(f)["tracked"] is False          # untracked
    _git(tmp_path, "add", "x.json")
    assert file_git_status(f)["tracked"] is True and file_git_status(f)["modified"] is True   # staged only
    _git(tmp_path, "commit", "-qm", "c")
    st = file_git_status(f)
    assert st["tracked"] and st["modified"] is False and len(st["commit"]) == 40
    f.write_text('{"a": 1}')
    assert file_git_status(f)["modified"] is True


def test_check_ids():
    sp = {"dev_ids": ["a", "b"], "eval_ids": ["c"]}
    A.check_ids(["b", "a"], sp, "dev")
    with pytest.raises(AssertionError):
        A.check_ids(["a", "b", "c"], sp, "dev")
    with pytest.raises(AssertionError):
        A.check_ids(["a"], sp, "dev")


# --------------------------------------------------------------- end to end on the tiny synthetic data

def _run(cfg_path, *args):
    return subprocess.run([sys.executable, "-m", *args, "--config", cfg_path], cwd=REPO_ROOT,
                          capture_output=True, text=True)


@pytest.fixture(scope="module")
def synth_env(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("t3ad")
    cfg_path = synth.build(str(tmp))
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)
    cfg["budgets"] = [1, 2, 4]
    cfg["adaptive"]["check_points"] = [2, 3]
    with open(cfg_path, "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
    for ds in ("math", "mathvista"):
        p = _run(cfg_path, "task3_ttc.prepare", "--dataset", ds)
        assert p.returncode == 0, p.stderr
    _init_repo(tmp)
    return tmp, cfg_path, cfg


@pytest.mark.parametrize("ds", ["math", "mathvista"])
def test_select_then_evaluate_cli(synth_env, ds):
    tmp, cfg_path, cfg = synth_env
    ad = tmp / "results" / "adaptive"
    p = _run(cfg_path, "task3_ttc.adaptive", "evaluate", "--dataset", ds)
    assert p.returncode == 2 and "must be committed" in p.stderr           # no selected file yet
    p = _run(cfg_path, "task3_ttc.adaptive", "select", "--dataset", ds)
    assert p.returncode == 0, p.stderr + p.stdout
    grid = json.loads((ad / f"{ds}_dev_grid.json").read_text())
    sel = json.loads((ad / f"{ds}_selected.json").read_text())
    assert grid["n_questions"] == 2 and len(grid["settings"]) == 5 + 20 + 1
    assert set(sel["selected"]) == {"agreement", "quality"} and sel["score"] == ("K" if ds == "math" else "Z")
    p = _run(cfg_path, "task3_ttc.adaptive", "select", "--dataset", ds)
    assert p.returncode == 2 and "refusing to overwrite" in p.stderr
    p = _run(cfg_path, "task3_ttc.adaptive", "evaluate", "--dataset", ds)
    assert p.returncode == 2 and "must be committed" in p.stderr           # untracked
    _git(tmp, "add", f"results/adaptive/{ds}_selected.json")
    _git(tmp, "commit", "-qm", f"sel {ds}")
    with open(ad / f"{ds}_selected.json", "a") as f:
        f.write("\n")
    p = _run(cfg_path, "task3_ttc.adaptive", "evaluate", "--dataset", ds)
    assert p.returncode == 2 and "must be committed" in p.stderr           # modified
    _git(tmp, "checkout", "--", f"results/adaptive/{ds}_selected.json")
    p = _run(cfg_path, "task3_ttc.adaptive", "evaluate", "--dataset", ds)
    assert p.returncode == 0, p.stderr + p.stdout
    ev = json.loads((ad / f"{ds}_eval.json").read_text())
    assert ev["n_questions"] == 4 and set(ev["policies"]) == {"agreement", "quality", "fallback"}
    fb = ev["policies"]["fallback"]
    assert fb["accuracy"] == pytest.approx(ev["fixed_sc"]["4"]["accuracy"])
    assert fb["mean_cost"] == pytest.approx(ev["fixed_sc"]["4"]["cost"]) and fb["stop_distribution"]["4"] == 1.0
    for r in ev["policies"].values():
        assert sum(r["stop_distribution"].values()) == pytest.approx(1.0)
        for name in ("full_pool_correct_fraction", "full_pool_sc_leader_share"):
            assert sum(b["n_questions"] for b in r["retrospective_bins"][name]) == 4
    assert ev["selected_file"]["commit"] and len(ev["bootstrap"]["comparisons"]) == 3
    tr = np.load(ad / f"{ds}_eval_trace.npz")
    assert tr["stop"].shape == (4, cfg["n_orders"], 3)
    assert tr["correct"].mean(1).mean(0)[2] == pytest.approx(fb["accuracy"])


def test_select_reads_only_dev_questions(synth_env, monkeypatch):
    tmp, cfg_path, cfg = synth_env
    seen = []

    def loader(c, ds, which):
        out = A.load_subset(c, ds, which)
        seen.append((which, out[0]))
        return out

    cfg2 = {**cfg, "paths": {**cfg["paths"], "results_dir": str(tmp / "results2")}}
    import shutil
    shutil.copytree(tmp / "results", tmp / "results2", ignore=shutil.ignore_patterns("adaptive"))
    A.select(cfg2, "math", cfg_path, loader=loader)
    sp = load_split(os.path.join(cfg["paths"]["splits_dir"], "math_split.json"))
    assert [w for w, _ in seen] == ["dev"] and set(seen[0][1]) == set(sp["dev_ids"])
    ids, qs, orders, d = A.load_subset(cfg, "math", "dev")
    assert d["qs"] is None and len(qs) == len(ids) == 2 and not set(ids) & set(sp["eval_ids"])


# ------------------------------------------------------------------------- real data (committed files)

@pytest.mark.parametrize("ds", ["math", "mathvista"])
def test_fallback_equals_engine_sc32_on_real_data(ds):
    csv = os.path.join(REPO_ROOT, "results", "task3", "rollouts", f"{ds}_rollouts.csv")
    npz = os.path.join(REPO_ROOT, "results", "task3", "offline", f"{ds}_selections.npz")
    if not (os.path.exists(csv) and os.path.exists(npz)):
        pytest.skip("real Task 3 results not present")
    sp = load_split(os.path.join(REPO_ROOT, "splits", "task3", f"{ds}_split.json"))
    df = pd.read_csv(csv, dtype={"sample_id": str}, keep_default_na=False,
                     usecols=["sample_id", "rollout_id", "T", "group_id", "correct"])
    nq = len(sp["sorted_ids"])
    g = df["group_id"].to_numpy().reshape(nq, 32)
    T = df["T"].to_numpy().reshape(nq, 32)
    c = df["correct"].astype(str).eq("True").to_numpy().reshape(nq, 32)
    qs = [{"g": g[i], "T": T[i], "correct": c[i]} for i in range(nq)]
    orders = orders_array(sp)
    sim = A.simulate(qs, orders, [A.FALLBACK], [4, 8, 16], None)
    z = np.load(npz)
    m = list(z["methods"]).index("SC")
    b = list(z["budgets"]).index(32)
    assert (z["orders"] == orders).all()
    assert (sim["group"][:, :, 0] == z["selected_group"][:, :, b, m]).all()
    assert (sim["correct"][:, :, 0] == z["correct"][:, :, b, m]).all()
    assert (sim["cost"][:, :, 0] == z["cost"][:, :, b, m]).all() and (sim["stop"] == 32).all()
