"""Task 2 analysis tests: helpers on synthetic rows, then the committed result files."""
import copy
import json
import os

import numpy as np
import pytest

from task2_vlm import analysis as A
from task2_vlm.scoring import soft_accuracy

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@pytest.fixture(scope="module", autouse=True)
def repo_root():
    old = os.getcwd()
    os.chdir(REPO)
    yield
    os.chdir(old)


# ---------------------------------------------------------------- synthetic data

def grd_pair(pid, answers, images):
    return {"pair_id": pid, "type": "counting", "question": f"How many things ({pid})?", "image_ids": images,
            "answers": answers, "counts": [int(a) for a in answers]}


def make_rows(pairs, preds, suite="grounding_test", mismatch=None):
    """preds: {condition: {pair_id: [raw0, raw1]}}; swap defaults to the partner's correct prediction."""
    rows = []
    for c in A.CONDITIONS:
        for p in pairs:
            for s in (0, 1):
                if c == "swap" and c not in preds:
                    raw, img = preds["correct"][p["pair_id"]][1 - s], p["image_ids"][1 - s]
                else:
                    raw = preds.get(c, preds["correct"])[p["pair_id"]][s]
                    img = {"correct": p["image_ids"][s], "swap": p["image_ids"][1 - s], "none": None,
                           "mismatched": (mismatch or {}).get(p["pair_id"], [-1, -2])[s]}[c]
                ex = raw == p["answers"][s]
                rows.append({"suite": suite, "condition": c, "pair_id": p["pair_id"], "slot": s, "type": p["type"],
                             "question": p["question"], "image_id": img, "pair_image_ids": p["image_ids"],
                             "raw_generation": raw, "token_ids": [hash(raw) % 1000], "stopped": True,
                             "n_new_tokens": 2, "prediction": raw, "accuracy": float(ex), "exact": ex,
                             "equals_partner": raw == p["answers"][1 - s], "empty_normalized": raw == "",
                             "empty_raw": raw == "", "reference_nll": 1.0 + s, "true_count": p["counts"][s]})
    return rows


PAIRS = [grd_pair("p1", ["1", "2"], [11, 12]), grd_pair("p2", ["2", "1"], [21, 22]),
         grd_pair("p3", ["1", "3"], [31, 32]), grd_pair("p4", ["2", "4"], [41, 42])]
CORRECT = {"p1": ["1", "1"], "p2": ["2", "1"], "p3": ["1", "3"], "p4": ["4", "4"]}


# ---------------------------------------------------------------- helpers

def test_moving_average_restarts_at_stage_starts():
    x = [1, 3, 5, 100, 200]
    ma = A.moving_average(x, 2, [3])
    assert ma.tolist() == [1, 2, 4, 100, 150]
    assert A.moving_average(x, 10).tolist() == pytest.approx(np.cumsum(x) / np.arange(1, 6))


def test_stage_bounds():
    cfg = {"stages": [{"updates": 2250}, {"updates": 1500}, {"updates": 750}]}
    assert A.stage_bounds(cfg) == [(1, 2250), (2251, 3750), (3751, 4500)]


def test_swap_partner_passes_and_detects_a_mismatch():
    rows = make_rows(PAIRS, {"correct": CORRECT})
    r = A.check_swap_partner(rows, ["grounding_test"])["grounding_test"]
    assert r == {"n": 8, "mismatches": 0, "examples": []}
    bad = copy.deepcopy(rows)
    row = next(x for x in bad if x["condition"] == "swap" and x["pair_id"] == "p3" and x["slot"] == 0)
    row["raw_generation"] = "4"
    r = A.check_swap_partner(bad, ["grounding_test"])["grounding_test"]
    assert r["mismatches"] == 1 and r["examples"][0]["pair_id"] == "p3"


def test_output_summary_counts_cap_hits():
    rows = [{"raw_generation": "yes", "stopped": True, "n_new_tokens": 2, "token_ids": [1]},
            {"raw_generation": "yes", "stopped": True, "n_new_tokens": 2, "token_ids": [1]},
            {"raw_generation": "a b c d", "stopped": False, "n_new_tokens": 4, "token_ids": [1, 2, 3, 4]}]
    o = A.output_summary(rows, cap=4)
    assert o["hit_cap"] == 1 and o["top"][0] == {"raw": "yes", "count": 2}
    assert o["mean_generated_tokens"] == pytest.approx(8 / 3) and o["mean_content_tokens"] == 2
    with pytest.raises(AssertionError):
        A.output_summary(rows, cap=16)                           # not stopped but below the cap


def test_mismatched_rule_and_rows():
    splits = {"grounding_test": {"pairs": PAIRS}}
    rng = np.random.default_rng(5)
    order = [PAIRS[i]["pair_id"] for i in rng.permutation(4)]
    imgs = {p["pair_id"]: p["image_ids"] for p in PAIRS}
    mm = {"suites": {"grounding_test": {"shuffled_order": order, "map": {
        pid: {"source_pair": order[(k + 1) % 4], "image_ids": imgs[order[(k + 1) % 4]]} for k, pid in enumerate(order)}}}}
    mapping = {pid: v["image_ids"] for pid, v in mm["suites"]["grounding_test"]["map"].items()}
    rows = make_rows(PAIRS, {"correct": CORRECT}, mismatch=mapping)
    r = A.check_mismatched(rows, mm, splits, 5, 1)
    assert r["rule_recomputed"] and r["suites"]["grounding_test"]["mismatches"] == 0
    assert not A.check_mismatched(rows, mm, splits, 6, 1)["rule_recomputed"]   # other seed, other order
    bad = copy.deepcopy(rows)
    next(x for x in bad if x["condition"] == "mismatched")["image_id"] = 12
    assert A.check_mismatched(bad, mm, splits, 5, 1)["suites"]["grounding_test"]["mismatches"] == 1


def test_soft_accuracy_restatement_matches_scoring():
    rng = np.random.default_rng(0)
    for _ in range(300):
        refs = list(rng.choice(["a", "b", "c"], size=10))
        pred = str(rng.choice(["a", "b", "c", "d"]))
        assert A._soft_acc(pred, refs) == pytest.approx(soft_accuracy(pred, refs))


def test_metrics_and_by_construction_relations():
    rows = make_rows(PAIRS, {"correct": CORRECT, "none": {p["pair_id"]: ["1", "1"] for p in PAIRS}})
    tab = {}
    for c in A.CONDITIONS:
        rc = [r for r in rows if r["condition"] == c]
        tab[c] = A.metrics(rc)
        tab[c]["by_type"] = {"counting": A.metrics(rc)}
    assert tab["correct"]["accuracy"] == 6 / 8 and tab["correct"]["pair_accuracy"] == 2 / 4
    assert tab["correct"]["identical_answer_rate"] == 2 / 4
    assert A.check_by_construction(tab) == 0.0
    tab["swap"]["follow_partner_rate"] += 0.25
    assert A.check_by_construction(tab) == pytest.approx(0.25)


def test_qualitative_rule_first_pair_per_kind():
    rows = make_rows(PAIRS, {"correct": CORRECT})
    splits = {"vqa_test": {"pairs": []}, "grounding_test": {"pairs": PAIRS}}
    mm = {"suites": {"grounding_test": {"map": {p["pair_id"]: {"image_ids": [0, 0]} for p in PAIRS}}}}
    picks, empty = A.qualitative(rows, splits, mm)
    got = {(p["type"], p["kind"]): p["pair_id"] for p in picks}
    # p1 identical (b); p2 both correct and swap follows partner (a); p4 swap gives "4" for slot 1 (c: "4" is
    # canonical of p4 slot 1 -> not c); swap of p1 gives "1","1" (canonical of p1) -> not c.
    assert got[("counting", "a")] == "p2" and got[("counting", "b")] == "p1"
    assert ("counting", "c") not in got and {"suite": "grounding_test", "type": "counting", "kind": "c"} in empty
    assert {"suite": "grounding_test", "type": "presence", "kind": "a"} in empty
    e = next(p for p in picks if p["kind"] == "a")
    assert e["predictions"]["swap"][0]["normalized"] == "1" and e["references"]["canonical"] == ["2", "1"]
    # a swap prediction outside both canonical answers makes kind c; kind a needs the swap condition too
    swap = {pid: v[::-1] for pid, v in CORRECT.items()}
    rows2 = make_rows(PAIRS, {"correct": CORRECT, "swap": {**swap, "p2": ["3", "2"]}})
    picks2, _ = A.qualitative(rows2, splits, mm)
    got2 = {(p["type"], p["kind"]): p["pair_id"] for p in picks2}
    assert got2[("counting", "c")] == "p2" and got2[("counting", "a")] == "p3"


def test_reader_refuses_uncommitted(tmp_path):
    p = tmp_path / "x.json"
    p.write_text("{}")
    with pytest.raises(AssertionError):
        A.Reader(committed=True).json(str(p))
    assert A.Reader(committed=False).json(str(p)) == {}


# ---------------------------------------------------------------- committed results

@pytest.fixture(scope="module")
def real(tmp_path_factory):
    if not os.path.exists("results/task2/main/final/examples.json"):
        pytest.skip("Task 2 results not present")
    d = tmp_path_factory.mktemp("t2analysis")
    t = A.analyze(out=str(d / "out"), figures=str(d / "fig"), download=False)
    return d, t


def test_real_sanity(real):
    _, t = real
    s = t["sanity"]
    assert all(v["mismatches"] == 0 and v["n"] > 0 for v in s["swap_equals_partner"].values())
    assert s["mismatched"]["rule_recomputed"]
    assert all(v["mismatches"] == 0 for v in s["mismatched"]["suites"].values())
    assert all(v["max_abs_difference"] < 1e-12 for v in s["recompute"].values())
    assert all(s["noimage_dev_nll"][k]["constant"] for k in A.DEV_SUITES)


def test_real_tables_match_eval_json(real):
    _, t = real
    for suite in A.SUITES:
        assert t["conditions"][suite]["max_abs_difference_vs_eval_json"] < 1e-12
        assert t["conditions"][suite]["by_construction_max_violation"] < 1e-12
        assert set(t["conditions"][suite]["conditions"]["none"]["by_construction"]) == {"identical_answer_rate",
                                                                                       "pair_accuracy"}
    assert t["advantages"]["max_abs_difference_vs_eval_json"] < 1e-12
    n = sum(v["n"] for v in t["counting"]["conditions"]["correct"].values())
    assert n == 64


def test_real_outputs_written(real):
    d, t = real
    for name in ("sanity", "learning", "conditions", "advantages", "counting", "resources", "qualitative", "manifest"):
        doc = json.loads((d / "out" / f"{name}.json").read_text())
        assert doc["run_id"] == f"task2_analysis_{name}"
    man = json.loads((d / "out" / "manifest.json").read_text())
    assert "results/task2/main/final/examples.json" in man["sources"]
    for f in A.FIGURES:
        assert (d / "fig" / f).stat().st_size > 0
    q = t["qualitative"]
    assert all(p["kind"] in "abc" for p in q["picks"])
    assert len(q["picks"]) + len(q["groups_without_pick"]) == 3 * 5
