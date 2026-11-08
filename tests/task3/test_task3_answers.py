"""Task 3 scorers, grouping, grading and checks (L597-606)."""
import copy

import pytest

from task3 import answers
from task3.answers import (NO_ANSWER, MathScorer, MathVistaScorer, group_keys, levenshtein, mathvista_normalize,
                           score_dataset)
from task3.data import load_config


@pytest.fixture(scope="module")
def ms():
    return MathScorer()


# ------------------------------------------------------------------------------------------- MATH

def test_math_equivalence(ms):
    assert not ms.equal("5\\sqrt{2}", "5", "t") and not ms.equal("5", "5\\sqrt{2}", "t")
    assert ms.equal("\\frac{1}{2}", "0.5", "t") and ms.equal("0.5", "\\frac{1}{2}", "t")
    assert ms.equal("1+x", "x+1", "t")
    assert not ms.equal("x", "x+1", "t")


def test_math_wraps_in_boxed(ms):
    # unwrapped parsing would read 5\sqrt{2} as 5; wrapped it is 5*sqrt(2)
    assert "sqrt" in str(ms.parse("5\\sqrt{2}")[0])
    assert answers.wrap_boxed("x") == "\\boxed{x}"


def test_math_no_answer(ms):
    assert ms.key(None) == (None, answers.REASON_EXTRACTION)
    assert ms.key("   ") == (None, answers.REASON_EXTRACTION)
    assert ms.key(" 3 ") == ("3", None)
    gids, reps, distinct = group_keys([ms.key(x)[0] for x in ["3", None, "3.0", ""]], ms.same)
    assert gids == [0, NO_ANSWER, 0, NO_ANSWER] and reps == ["3"]


def test_math_unparseable_groups_only_identical(ms):
    assert not ms.parses("@@") and ms.parses("7")
    gids, reps, _ = group_keys(["@@", "@@", " @@".strip(), "@@x", "7"], ms.same)
    assert gids == [0, 0, 0, 1, 2]
    assert not ms.grade("@@", "@@")  # graded wrong even against an identical gold string


def test_math_timeout_counts_as_not_equal():
    s = MathScorer()

    def slow_verify(g, t, timeout_seconds):
        import logging
        logging.getLogger("math_verify.grader").warning("Timeout during comparison")
        return True

    s._verify_fn = slow_verify
    assert not s.equal("1", "1.0", "grouping")
    assert s.timeouts["grouping"] == 1


# ------------------------------------------------------------------------------------------- MathVista

MC = ["red", "blue", "green"]
DUP = ["9", "12", "18", "18"]


def test_levenshtein():
    assert levenshtein("kitten", "sitting") == 3 and levenshtein("", "abc") == 3 and levenshtein("ab", "ab") == 0


def test_mathvista_integer_and_float():
    n = lambda x, t, p=None: mathvista_normalize(x, None, "free_form", t, p)
    assert n("5-10 years", "integer") is None
    assert n("5.0", "integer") == "5" and n(" 7 ", "integer") == "7"
    assert n("3.14159", "float", 2) == "3.14" and n("1.456", "float", 2) == "1.46"
    assert n("1.2", "float", 1) == "1.2" and n("2", "float", 1) == "2.0"
    assert n("0.0115 \\, \\text{m}", "float", 2) is None  # no extra unit stripping
    assert n("[2014, 2016]", "list") == "[2014, 2016]"
    with pytest.raises(ValueError):
        n("x", "text")


def test_mathvista_multiple_choice():
    n = lambda x, ch=MC: mathvista_normalize(x, ch, "multi_choice", "text", None)
    assert n("B") == "B"
    assert n("(C) text") == "C"
    assert n("blue") == "B"            # option text -> its letter
    assert n("bleu") == "B"            # unmatched text -> nearest option's letter
    assert n("F") == "A"               # out-of-range letter -> nearest option (official default)
    assert n("18", DUP) == "C"         # repeated option text -> first occurrence
    assert n("D", DUP) == "D"          # duplicate-text options stay letter-strict


def test_mathvista_scorer_keys_and_grading():
    s = MathVistaScorer()
    m_int = {"choices": None, "question_type": "free_form", "answer_type": "integer", "precision": None}
    m_dup = {"choices": DUP, "question_type": "multi_choice", "answer_type": "text", "precision": None}
    assert s.key(None, m_int) == (None, answers.REASON_EXTRACTION)
    assert s.key("", m_int) == (None, answers.REASON_EXTRACTION)
    assert s.key("5-10 years", m_int) == (None, answers.REASON_NORMALIZATION)
    gk = s.gold_key("C", m_dup)
    assert gk == "C" and s.grade(gk, "C") and not s.grade(gk, "D")


# ------------------------------------------------------------------------------------------- grouping

def test_grouping_first_matching_group_in_rollout_order(ms):
    keys = ["\\frac{1}{2}", "3", None, "0.5", "\\frac{2}{4}", "3.0", None]
    gids, reps, distinct = group_keys(keys, ms.same)
    assert gids == [0, 1, NO_ANSWER, 0, 0, 1, NO_ANSWER]
    assert reps == ["\\frac{1}{2}", "3"] and len(distinct) == 5


def test_score_dataset_math(ms):
    sid = ["q1"] * 4 + ["q2"] * 4
    rid = [3, 1, 0, 2, 0, 1, 2, 3]   # rows out of rollout order
    ext = ["x", "0.5", "\\frac{1}{2}", None, "5", "5\\sqrt{2}", None, ""]
    gold = {"q1": "\\frac{1}{2}", "q2": "5\\sqrt{2}"}
    r = score_dataset(sid, rid, ext, gold, ms)
    assert list(r["group_id"]) == [1, 0, 0, NO_ANSWER, 0, 1, NO_ANSWER, NO_ANSWER]
    assert list(r["correct"]) == [False, True, True, False, False, True, False, False]
    assert r["na_reason"] == [None, None, None, "extraction", None, None, "extraction", "extraction"]


def test_score_dataset_mathvista():
    meta = {"p": {"choices": MC, "question_type": "multi_choice", "answer_type": "text", "precision": None},
            "i": {"choices": None, "question_type": "free_form", "answer_type": "integer", "precision": None}}
    sid = ["p"] * 4 + ["i"] * 4
    rid = [0, 1, 2, 3] * 2
    ext = ["blue", "B", "(A) red", None, "5", "5.0", "5-10 years", "6"]
    r = score_dataset(sid, rid, ext, {"p": "B", "i": "5"}, MathVistaScorer(), meta)
    assert r["normalized"] == ["B", "B", "A", None, "5", "5", None, "6"]
    assert list(r["group_id"]) == [0, 0, 1, NO_ANSWER, 0, 0, NO_ANSWER, 1]
    assert list(r["correct"]) == [True, True, False, False, True, True, False, False]
    assert r["na_reason"][3] == "extraction" and r["na_reason"][6] == "normalization"


# ------------------------------------------------------------------------------------------- checks

class _FakeScorer:
    """Planted relation: a~b and b~c but not a~c (non-transitive), and c->d but not d->c (asymmetric)."""
    name = "math"
    timeouts = {}

    def __init__(self, rel):
        self.rel = rel

    def equal(self, g, t, phase):
        return g == t or (g, t) in self.rel

    def same(self, rep, new):
        return self.equal(rep, new, "grouping")

    def parses(self, s):
        return True

    def grade(self, gold_key, s, phase="grading"):
        return self.equal(gold_key, s, phase)


def test_transitivity_check_detects_violations():
    rel = {("a", "b"), ("b", "a"), ("b", "c"), ("c", "b"), ("c", "d")}
    sc = _FakeScorer(rel)
    keys = ["a", "b", "c", "d"]
    gids, reps, distinct = group_keys(keys, sc.same)
    assert gids == [0, 0, 1, 1]   # c does not equal rep a; d joins rep c
    q = {"q": {"keys": keys, "gids": gids, "reps": reps, "distinct": distinct, "gold_key": "b",
               "grades": [sc.grade("b", r) for r in reps]}}
    a = answers.check_transitivity(q, sc)
    assert not a["passed"] and a["questions_affected"] == 1
    assert a["asymmetric_pairs"] == 1 and a["non_transitive_triples"] >= 2 and a["block_mismatch_pairs"] > 0
    b = answers.check_grading_consistency(q, sc)
    # gold b: group 0 rep a graded True; member b True (ok); group 1 rep c graded True; member d False -> mismatch
    assert not b["passed"] and b["mismatched_strings"] == 1 and b["examples"][0]["member"] == "d"


def test_checks_pass_on_consistent_math(ms):
    sid = ["q"] * 5
    r = score_dataset(sid, list(range(5)), ["\\frac{1}{2}", "0.5", "3", None, "@@"], {"q": "1/2"}, ms)
    qs = r["questions"]
    assert answers.check_transitivity(qs, ms)["passed"]
    assert answers.check_grading_consistency(qs, ms)["passed"]
    assert answers.check_gold(qs, ms, {"q": "1/2"})["passed"]
    pf = answers.parse_failures(qs, ms)
    assert pf["distinct_strings"] == 1 and pf["rollouts"] == 1


def test_gold_check_failures(ms):
    r = score_dataset(["q"], [0], ["1"], {"q": "@@"}, ms)
    assert answers.check_gold(r["questions"], ms, {"q": "@@"})["failures"] == 1
    meta = {"p": {"choices": MC, "question_type": "multi_choice", "answer_type": "text", "precision": None},
            "i": {"choices": None, "question_type": "free_form", "answer_type": "integer", "precision": None}}
    gold = {"p": "blue", "i": "five"}
    r = score_dataset(["p", "i"], [0, 0], ["B", "5"], gold, MathVistaScorer(), meta)
    g = answers.check_gold(r["questions"], MathVistaScorer(), gold, meta)
    assert g["failures"] == 2


def test_scorer_version_checks_libraries():
    cfg = load_config()
    v = answers.scorer_version(cfg)
    assert v["libraries"]["math-verify"] == "0.9.0" and len(v["answers_py_sha256"]) == 64 and len(v["id"]) == 64
    bad = copy.deepcopy(cfg)
    bad["scorer"]["libraries"]["math-verify"] = "0.0.1"
    with pytest.raises(RuntimeError):
        answers.scorer_version(bad)
