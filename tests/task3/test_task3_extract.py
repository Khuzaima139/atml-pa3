"""Balanced-brace extractor (README R32) and the committed live problem file."""
import json
import os

import pytest

from common.hashing import sha256_json
from common.runinfo import REPO_ROOT
from task3_ttc.extract import last_boxed

GEN = os.path.join(REPO_ROOT, "data/task3/ATML-PA3-Data/qwen25-math/generations.parquet")


@pytest.mark.parametrize("text,want", [
    ("so \\boxed{14}.", "14"),
    ("\\boxed{ 14 }", "14"),
    ("\\boxed{\\frac{1}{2}}", "\\frac{1}{2}"),
    ("\\boxed{\\left\\{ 1, 2 \\right\\}}", "\\left\\{ 1, 2 \\right\\}"),
    ("\\boxed{a_{b_{c}}} tail }", "a_{b_{c}}"),
    ("first \\boxed{1} then \\boxed{2}", "2"),
    ("\\boxed{1} and a last \\boxed{2", None),          # last box unclosed -> None, no fallback to the first
    ("\\boxed{\\frac{1}{2}", None),                      # nested but unclosed
    ("\\boxed{}", None),
    ("\\boxed{   }", None),
    ("\\boxed{3} then \\boxed{}", None),                 # last box empty
    ("no box, the answer is 4", None),
    ("\\boxed 5", None),                                 # no literal '\boxed{'
    ("\\fbox{5}", None),
    ("", None),
    (None, None),
])
def test_last_boxed_units(text, want):
    assert last_boxed(text) == want


@pytest.mark.skipif(not os.path.exists(GEN), reason="supplied data (data/task3) absent")
def test_reproduces_supplied_math_extractions():
    import pyarrow.parquet as pq
    t = pq.read_table(GEN, columns=["generated_text", "extracted_answer"]).to_pydict()
    assert len(t["generated_text"]) == 32000
    bad = [i for i, (g, e) in enumerate(zip(t["generated_text"], t["extracted_answer"])) if last_boxed(g) != e]
    assert bad == []


def test_live_problems_file():
    with open(os.path.join(REPO_ROOT, "splits/task3/live_problems.json")) as f:
        d = json.load(f)
    with open(os.path.join(REPO_ROOT, "splits/task3/math_split.json")) as f:
        split = json.load(f)
    assert d["n"] == 50 and sha256_json(d["problems"]) == d["sha256"]
    assert [p["sample_id"] for p in d["problems"]] == split["live_ids"] == split["eval_ids"][:50]
    assert [p["q"] for p in d["problems"]] == list(range(50))
    assert all(set(p) == {"q", "sample_id", "problem"} and p["problem"].strip() for p in d["problems"])
    assert d["source"]["split_sha256"] == split["sha256"]
