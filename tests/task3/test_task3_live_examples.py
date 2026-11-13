"""Task 3 live revision examples: fixed rules on a synthetic live run."""
import csv
import json

import pytest

from common.hashing import sha256_file
from task3_ttc import live_examples as L

STRATS = ("single", "p1", "p2", "p3", "refine")


def _text(tag, n=60):
    return " ".join(f"{tag}{i}" for i in range(n))


def _write(tmp_path, spec):
    """spec: list of dicts with p1_ok, ref_ok, p1_trunc, answers {strategy: str or None}."""
    calls, rows, probs = [], [], []
    for q, s in enumerate(spec):
        sid = f"test:{q}"
        probs.append({"q": q, "sample_id": sid, "problem": f"problem {q}"})
        for i, st in enumerate(STRATS):
            calls.append({"q": q, "sample_id": sid, "strategy": st, "call": i, "extracted_answer": s["answers"][st],
                          "truncated": st == "p1" and s["p1_trunc"], "output_tokens": 60,
                          "output_text": _text(f"{st}q{q}w"), "draft_call": 1 if st == "refine" else None})
        row = {"q": q, "sample_id": sid, "gold": "7", "initial_correct": int(s["p1_ok"]),
               "refine_correct": int(s["ref_ok"]), "single_correct": 1, "parallel_answer": s["answers"]["p1"] or "",
               "parallel_correct": int(s["p1_ok"]), "parallel_groups": json.dumps([0, 0, 1]),
               "parallel_selected_call": "p1", "p1_truncated": int(s["p1_trunc"]),
               "refine_answer_changed": int(s["answers"]["p1"] != s["answers"]["refine"])}
        for st in STRATS:
            row[f"{st}_call_correct"] = 1 if st in ("single", "p2") else 0
        rows.append(row)
    cp = tmp_path / "calls.jsonl"
    cp.write_text("".join(json.dumps(c) + "\n" for c in calls))
    pq = tmp_path / "per_question.csv"
    with open(pq, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    pp = tmp_path / "problems.json"
    pp.write_text(json.dumps({"problems": probs}))
    ap = tmp_path / "analysis.json"
    ap.write_text(json.dumps({"run_id": "a", "inputs": {"calls_jsonl_sha256": sha256_file(str(cp)),
                                                        "live_run_id": "live"}}))
    return {"analysis": str(ap), "per_question": str(pq), "calls": str(cp), "problems": str(pp)}


def _ans(p1, refine):
    return {"single": "7", "p1": p1, "p2": "7", "p3": "3", "refine": refine}


SPEC = [
    {"p1_ok": True, "ref_ok": True, "p1_trunc": False, "answers": _ans("7", "7")},
    {"p1_ok": False, "ref_ok": False, "p1_trunc": True, "answers": _ans(None, "5")},
    {"p1_ok": False, "ref_ok": True, "p1_trunc": True, "answers": _ans(None, "7")},
    {"p1_ok": False, "ref_ok": True, "p1_trunc": False, "answers": _ans("3", "7")},
]


def test_rules_first_in_live_order_and_none(tmp_path):
    paths = _write(tmp_path, SPEC)
    doc = L.build(paths, str(tmp_path / "out.json"), {"timestamp": "2026-11-09T00:00:00"})
    ex = doc["examples"]
    assert ex["wrong_to_correct"]["q"] == 2 and ex["truncated_draft"]["q"] == 1
    assert ex["correct_to_wrong"] == "none" and doc["aligned"]["correct_to_wrong"] == "none"
    e = ex["wrong_to_correct"]
    assert e["draft"]["answer"] == "NO_ANSWER" and e["draft"]["truncated"] and not e["draft"]["correct"]
    assert e["revision"]["answer"] == "7" and e["revision"]["correct"] and e["revision"]["answer_changed"]
    assert e["draft"]["last_40_words"].split() == _text("p1q2w").split()[-40:]
    assert e["revision"]["first_40_words"].split() == _text("refineq2w").split()[:40]
    assert e["revision"]["last_40_words"].split() == _text("refineq2w").split()[-40:]
    a = doc["aligned"]["wrong_to_correct"]
    assert a["q"] == 2 and set(a["lines"]) == {"single", "parallel", "refine"}
    assert "p1 NO_ANSWER (wrong, truncated)" in a["lines"]["parallel"]
    assert "SC pick p1: NO_ANSWER (wrong)" in a["lines"]["parallel"]
    assert a["parallel"]["p2"]["correct"] and not a["parallel"]["p3"]["correct"]


def test_correct_to_wrong_found_and_no_overwrite(tmp_path):
    spec = SPEC + [{"p1_ok": True, "ref_ok": False, "p1_trunc": False, "answers": _ans("7", "8")}]
    paths = _write(tmp_path, spec)
    out = str(tmp_path / "out.json")
    doc = L.build(paths, out, {"timestamp": "2026-11-09T00:00:00"})
    assert doc["examples"]["correct_to_wrong"]["q"] == 4 and doc["aligned"]["correct_to_wrong"]["q"] == 4
    with pytest.raises(FileExistsError):
        L.build(paths, out, {"timestamp": "2026-11-09T00:00:00"})


def test_refuses_analysis_of_other_calls(tmp_path):
    paths = _write(tmp_path, SPEC)
    with open(paths["calls"], "a") as f:
        f.write("\n")
    with pytest.raises(AssertionError):
        L.build(paths, str(tmp_path / "out.json"), {"timestamp": "2026-11-09T00:00:00"})
