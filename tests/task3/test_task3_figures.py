"""Task 3 figures and qualitative picks (prompt b3 items B, C)."""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402

from common.runinfo import REPO_ROOT, file_git_status  # noqa: E402
from task3_ttc import figures as F  # noqa: E402
from task3_ttc import qualitative as Q  # noqa: E402


def _all_committed():
    return all(file_git_status(os.path.join(REPO_ROOT, p))["tracked"] for p in F.inputs().values())


def test_layout_check_flags_overlaps():
    fig, axes = plt.subplots(1, 2, figsize=(4, 2))
    for ax in axes:
        ax.set_title("a very long left title that runs into the neighbour", loc="left")
    assert any("overlaps" in b or "outside" in b for b in F.layout_problems(fig))
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(2, 2))
    ax.set_xticks(range(5))
    ax.set_xticklabels(["a long tick label"] * 5)
    assert "ax0 tick labels overlap" in F.layout_problems(fig)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(4, 3), constrained_layout=True)
    F._axes(ax, "short", "y", "x")
    assert F.layout_problems(fig) == []
    plt.close(fig)


def test_committed_refuses_untracked(tmp_path):
    p = tmp_path / "x.json"
    p.write_text("{}")
    with pytest.raises(RuntimeError):
        F.committed(str(p))


def test_moving_average():
    x = np.arange(1, 6, dtype=float)
    assert F.moving_average(x, 2).tolist() == [1.0, 1.5, 2.5, 3.5, 4.5]
    assert F.moving_average(x, 32).tolist() == [1.0, 1.5, 2.0, 2.5, 3.0]


def test_first_same_and_differ_follows_split_order():
    sorted_ids = ["a", "b", "c", "d"]
    sel = np.zeros((4, 1, 1, 2), dtype=np.int8)
    sel[:, 0, 0, 0] = [3, 5, 7, 9]           # Highest-Z
    sel[:, 0, 0, 1] = [3, 6, 7, 1]           # Highest-Ztok
    z = {"methods": np.array(["Highest-Z", "Highest-Ztok"]), "budgets": np.array([32]), "selected_rollout": sel}
    assert F.first_same_and_differ(["d", "c", "b", "a"], sorted_ids, z) == (("c", 7, 7), ("d", 9, 1))
    assert F.first_same_and_differ(["c", "a"], sorted_ids, z) == (("c", 7, 7), None)


@pytest.mark.skipif(not _all_committed(), reason="Task 3 figure inputs not committed")
def test_figures_from_committed_results(tmp_path):
    read = []

    def loader(p):
        read.append(p)
        with open(F.committed(p)) as f:
            return json.load(f)

    out = F.plot_all(folder=str(tmp_path), loader=loader)     # _save raises on any layout problem
    assert len(out) == 9 and all(os.path.exists(p) and p.endswith(".pdf") for p in out)
    assert sorted(read) == sorted(F.inputs().values())
    assert not [p for p in read if p.startswith(("scratch/", "notes/", "outputs/"))]


@pytest.mark.skipif(not _all_committed(), reason="Task 3 figure inputs not committed")
def test_figdata_consistent_with_signals():
    with open(os.path.join(REPO_ROOT, F.FIGDATA)) as f:
        fd = json.load(f)
    with open(os.path.join(REPO_ROOT, "results/task3/signals/mathvista_signals.json")) as f:
        sig = json.load(f)["rollout_pearson"]
    pq = fd["pearson_per_question"]
    assert len(pq["sample_ids"]) == 800
    for k, name in (("c_v", "confidence_vision (c_t, v_t)"), ("v_pos", "vision_position (v_t, t)"),
                    ("c_pos", "confidence_position (c_t, t)")):
        assert np.mean([x for x in pq[k] if x is not None]) == pytest.approx(sig[name]["mean"], abs=1e-12)
    for kind in ("same", "differ"):
        t = fd["trajectories"][kind]
        for r in t["rollouts"].values():
            assert len(r["c_t"]) == len(r["v_t"]) == r["T"]


# ----------------------------------------------------------------------------------------- qualitative

def test_last_words_and_votes():
    assert Q.last_words("a  b\nc " + " ".join(str(i) for i in range(50))) == " ".join(str(i) for i in range(10, 50))
    ro = {"g": np.array([[1, 0, 1, -1, 0, 2]]), "norm": np.array([["x", "y", "x", "", "y", "z"]]),
          "correct": np.array([[False, True, False, False, True, False]])}
    v = Q.votes(ro, 0, np.array([3, 1, 0, 2, 4, 5]))
    assert [(d["answer"], d["votes"]) for d in v] == [("y", 2), ("x", 2), ("NO_ANSWER", 1), ("z", 1)]
    assert [d["correct"] for d in v] == [True, False, False, False]


def test_find_disagreement_rule():
    # questions in sorted order a, b, c; split order c, b, a
    grp = np.zeros((3, 1, 1, 2), dtype=np.int16)
    cor = np.zeros((3, 1, 1, 2), dtype=bool)
    grp[:, 0, 0, 0], grp[:, 0, 0, 1] = [0, 0, 1], [1, 1, 2]
    cor[:, 0, 0, 0], cor[:, 0, 0, 1] = [True, True, False], [False, True, False]   # c: both wrong; b: both right
    sel = {"methods": np.array(["SC", "RWC-K"]), "budgets": np.array([32]), "selected_group": grp,
           "correct": cor, "selected_rollout": np.zeros((3, 1, 1, 2), dtype=np.int8)}
    ro = {"ids": ["a", "b", "c"]}
    assert Q.find_disagreement({"eval_ids": ["c", "b", "a"]}, ro, sel)[0] == 0
    assert Q.find_disagreement({"eval_ids": ["c", "b"]}, ro, sel) is None


def test_premature_rule(monkeypatch):
    R = 32
    g = np.zeros((2, R), dtype=int)
    g[0, :4] = 1                                   # q "a": first four answer 1 (wrong), majority 0 (right)
    ro = {"ids": ["a", "b"], "g": g, "norm": np.where(g == 1, "1", "0"), "correct": g == 0, "K": np.zeros((2, R))}
    orders = np.tile(np.arange(R), (2, 10, 1))
    trace = {"policies": np.array(["agreement", "quality", "fallback"]), "sample_id": np.array(["b", "a"]),
             "stop": np.array([[[4, 4, 32]] * 10, [[4, 4, 32]] * 10]),
             "correct": np.array([[[True, True, True]] * 10, [[False, False, True]] * 10]),
             "group": np.array([[[0, 0, 0]] * 10, [[1, 1, 0]] * 10]),
             "sc_full_correct": np.array([[True] * 10, [True] * 10])}
    ev = {"policies": {"agreement": {"name": "agreement(tau=0.5)"}}}
    monkeypatch.setattr(Q, "texts", lambda cfg, ds, ids: {ids[0]: {"question_text": "q", "gold_answer": "0",
                                                                    "generated_text": {}}})
    ex = Q.premature_example(None, "math", {"eval_ids": ["b", "a"]}, ro, trace, ev, orders)
    assert ex["sample_id"] == "a" and ex["stop_n"] == 4
    assert ex["votes_at_stop"] == [{"group_id": 1, "answer": "1", "votes": 4, "correct": False}]
    assert ex["votes_at_32"][0]["votes"] == 28 and ex["fixed_sc_32"]["answer"] == "0"
    trace["sc_full_correct"][1] = False
    assert Q.premature_example(None, "math", {"eval_ids": ["b", "a"]}, ro, trace, ev, orders) == "none"
