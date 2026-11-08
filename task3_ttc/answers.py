"""Task 3 answer normalization, answer groups and correctness labels (manual L597-606).

Decisions fixed in the Task 3 plan (prompt b1):

MATH (math-verify 0.9.0)
    Every extracted answer and every gold string s is wrapped as "\\boxed{" + s.strip() + "}" before parse().
    Equivalence is symmetric (prompt b2): a and b are equal if verify(a, b) or verify(b, a) is True
    (math-verify's interval conversion is asymmetric, e.g. "[-4, 0]" vs "-4 \\leq k \\leq 0").
    Grouping: a candidate joins the first earlier group whose representative it equals in either direction;
    grading: a group is correct if gold and its representative are equal in either direction; check (b) uses
    the same rule. A math-verify timeout counts as not equal (in that direction) and is counted. A non-empty
    string that parses to nothing only groups with identical stripped strings and is graded wrong; such strings
    are counted.

MathVista (official normalization of lupantech/MathVista, evaluation/calculate_score.py at commit 5f6b4dec,
normalize_extracted_answer without --ignore_empty_extractions), changed only where the manual requires
multiple-choice answers to be represented as option letters (L602):
    multi_choice  strip; "(X)" -> X; a letter within the options stays; otherwise the nearest option text by
                  Levenshtein distance (first occurrence on ties and repeated texts) -> that option's letter
    integer       str(int(float(x))), failure -> None
    float         str(round(float(x), int(precision))), failure -> None
    list          str(x)
    No extra unit stripping. Groups are exact normalized strings; grading is equality with the normalized gold
    (multiple-choice gold is the supplied letter).

Both datasets
    A missing or empty extracted answer, or a MathVista normalization returning None, maps to the question's
    single NO_ANSWER group (group_id -1), always graded wrong; na_reason records "extraction" or "normalization".
    Groups are formed once per question over all its rollouts without gold: in rollout_id order each string
    joins the first earlier group whose first member it equals (identical strings join the same group).
"""
import importlib.metadata
import logging
import os
import re
from collections import Counter

import numpy as np

from common.hashing import sha256_file, sha256_json

NO_ANSWER = -1
REASON_EXTRACTION = "extraction"
REASON_NORMALIZATION = "normalization"
MAX_EXAMPLES = 10
MATH_EQUIVALENCE = "symmetric: equal if verify(a, b) or verify(b, a) (grouping, grading, check b)"


# --------------------------------------------------------------------------------------------- MATH

class _TimeoutCounter(logging.Handler):
    """Counts math-verify's "Timeout during parsing/comparison" warnings (it returns [] / False on timeout)."""

    def __init__(self):
        super().__init__(logging.WARNING)
        self.n = 0

    def emit(self, record):
        if record.getMessage().startswith("Timeout during"):
            self.n += 1


_TIMEOUTS = _TimeoutCounter()
_MV_LOGGER = logging.getLogger("math_verify")
if not any(isinstance(h, _TimeoutCounter) for h in _MV_LOGGER.handlers):
    _MV_LOGGER.addHandler(_TIMEOUTS)
else:
    _TIMEOUTS = next(h for h in _MV_LOGGER.handlers if isinstance(h, _TimeoutCounter))


def wrap_boxed(s):
    return "\\boxed{" + s + "}"


class MathScorer:
    name = "math"
    exact_groups = False

    def __init__(self, parse_timeout=5, verify_timeout=5):
        from math_verify import parse, verify
        self._parse_fn, self._verify_fn = parse, verify
        self.parse_timeout, self.verify_timeout = parse_timeout, verify_timeout
        self._parsed = {}
        self._eq = {}
        self.timeouts = Counter()
        self.verify_calls = Counter()

    def key(self, extracted, meta=None):
        """(group key, na_reason). The key is the stripped extracted string."""
        if extracted is None or not extracted.strip():
            return None, REASON_EXTRACTION
        return extracted.strip(), None

    def gold_key(self, gold, meta=None):
        return gold.strip()

    def parse(self, s):
        if s not in self._parsed:
            before = _TIMEOUTS.n
            res = self._parse_fn(wrap_boxed(s), parsing_timeout=self.parse_timeout)
            if _TIMEOUTS.n > before:
                self.timeouts["parse"] += _TIMEOUTS.n - before
                res = []
            self._parsed[s] = res
        return self._parsed[s]

    def parses(self, s):
        """True if the wrapped string parses to at least one sympy object (a result holding only the raw string
        counts as parsing to nothing)."""
        import sympy
        return any(isinstance(x, (sympy.Basic, sympy.MatrixBase)) for x in self.parse(s))

    def equal(self, gold, target, phase):
        """verify(gold=gold, target=target) on the wrapped, parsed strings; unparsed or timeout -> False."""
        k = (gold, target)
        if k not in self._eq:
            if not (self.parses(gold) and self.parses(target)):
                r = False
            else:
                g, t = self.parse(gold), self.parse(target)
                before = _TIMEOUTS.n
                r = bool(self._verify_fn(g, t, timeout_seconds=self.verify_timeout))
                self.verify_calls[phase] += 1
                if _TIMEOUTS.n > before:
                    self.timeouts[phase] += _TIMEOUTS.n - before
                    r = False
            self._eq[k] = r
        return self._eq[k]

    def equal_sym(self, a, b, phase):
        """Symmetric equivalence: verify(a, b) or verify(b, a)."""
        return self.equal(a, b, phase) or self.equal(b, a, phase)

    def same(self, rep, new):
        return self.equal_sym(rep, new, "grouping")

    def grade(self, gold_key, s, phase="grading"):
        return gold_key is not None and self.equal_sym(gold_key, s, phase)


# ---------------------------------------------------------------------------------------- MathVista

LETTERS = [chr(ord("A") + i) for i in range(26)]


def levenshtein(a, b):
    """Unit-cost edit distance (insert, delete, substitute), as Levenshtein.distance."""
    if len(a) < len(b):
        a, b = b, a
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def mathvista_normalize(extraction, choices, question_type, answer_type, precision):
    """Port of the official normalize_extracted_answer; multiple-choice answers returned as option letters."""
    if question_type == "multi_choice":
        x = extraction.strip() if isinstance(extraction, str) else str(extraction)
        letter = re.findall(r"\(([a-zA-Z])\)", x)
        if len(letter) > 0:
            x = letter[0].upper()
        seq = LETTERS[:len(choices)]
        if x in seq:
            return x
        dist = [levenshtein(x, c) for c in choices]
        return seq[dist.index(min(dist))]
    if answer_type == "integer":
        try:
            return str(int(float(extraction)))
        except Exception:
            return None
    if answer_type == "float":
        try:
            return str(round(float(extraction), int(precision)))
        except Exception:
            return None
    if answer_type == "list":
        try:
            return str(extraction)
        except Exception:
            return None
    raise ValueError(f"no official normalization for question_type={question_type!r}, answer_type={answer_type!r}")


def load_mathvista_metadata(path):
    import json
    with open(path) as f:
        tm = json.load(f)
    return {str(pid): {"choices": r["choices"], "question_type": r["question_type"],
                       "answer_type": r["answer_type"], "precision": r["precision"]} for pid, r in tm.items()}


class MathVistaScorer:
    name = "mathvista"
    exact_groups = True

    def __init__(self):
        self.timeouts = Counter()
        self.verify_calls = Counter()

    @staticmethod
    def _norm(s, m):
        return mathvista_normalize(s, m["choices"], m["question_type"], m["answer_type"], m["precision"])

    def key(self, extracted, meta):
        if extracted is None or not extracted.strip():
            return None, REASON_EXTRACTION
        k = self._norm(extracted, meta)
        if k is None:
            return None, REASON_NORMALIZATION
        return k, None

    def gold_key(self, gold, meta):
        return self._norm(gold, meta)

    def parses(self, s):
        return True

    def equal_sym(self, a, b, phase=None):
        return a == b

    def same(self, rep, new):
        return rep == new

    def grade(self, gold_key, s, phase="grading"):
        return gold_key is not None and s == gold_key


def mathvista_gold_failure(gold, meta, gold_key):
    """Gold must normalize to non-None; a multiple-choice gold must be a valid option letter as supplied."""
    if gold_key is None:
        return "normalizes to None"
    if meta["question_type"] == "multi_choice" and gold.strip() not in LETTERS[:len(meta["choices"])]:
        return "multiple-choice gold is not a valid option letter"
    return None


# ------------------------------------------------------------------------------------------ grouping

def group_keys(keys, same):
    """Greedy grouping in the given (rollout_id) order.

    keys: list of str or None (None = NO_ANSWER). Each new string joins the first earlier group whose
    first member (representative) it equals under same(rep, new); identical strings share a group.
    Returns (group ids per position, representatives, distinct keys in first-appearance order).
    """
    reps, gid_of, distinct, out = [], {}, [], []
    for k in keys:
        if k is None:
            out.append(NO_ANSWER)
            continue
        if k not in gid_of:
            distinct.append(k)
            for g, rep in enumerate(reps):
                if same(rep, k):
                    gid_of[k] = g
                    break
            else:
                gid_of[k] = len(reps)
                reps.append(k)
        out.append(gid_of[k])
    return out, reps, distinct


def score_dataset(sample_id, rollout_id, extracted, gold, scorer, metadata=None):
    """Group and grade every question. Rows may be in any order; each question is processed in rollout_id order.

    Returns per-row arrays (normalized key, group_id, na_reason, correct) and per-question details.
    """
    n = len(sample_id)
    rows = {}
    for i in np.lexsort((np.asarray(rollout_id), np.asarray(sample_id, dtype=str))):
        rows.setdefault(sample_id[i], []).append(int(i))
    normalized = [None] * n
    na_reason = [None] * n
    group_id = np.full(n, NO_ANSWER, dtype=np.int64)
    correct = np.zeros(n, dtype=bool)
    questions = {}
    for sid, idx in rows.items():
        meta = metadata[sid] if metadata is not None else None
        keys = []
        for i in idx:
            k, reason = scorer.key(extracted[i], meta)
            keys.append(k)
            normalized[i], na_reason[i] = k, reason
        gids, reps, distinct = group_keys(keys, scorer.same)
        gk = scorer.gold_key(gold[sid], meta)
        grades = [scorer.grade(gk, r) for r in reps]
        for i, g in zip(idx, gids):
            group_id[i] = g
            correct[i] = g != NO_ANSWER and grades[g]
        questions[sid] = {"rows": idx, "keys": keys, "gids": gids, "reps": reps, "distinct": distinct,
                          "gold_key": gk, "grades": grades}
    return {"normalized": normalized, "na_reason": na_reason, "group_id": group_id, "correct": correct,
            "questions": questions}


# ---------------------------------------------------------------------------------------------- checks

def check_transitivity(questions, scorer):
    """(a) For every question, the pairwise matrix of the symmetric equivalence (verify in either direction) over
    its distinct strings must equal the block structure of the greedy groups, and must be transitive (no ordered
    triple with S[a,b] and S[b,c] but not S[a,c]). Parseable strings must equal themselves. Pairs where the raw
    directional verify disagrees (verify(a,b) != verify(b,a)) are counted for information only: the symmetric rule
    resolves them."""
    tot = Counter()
    affected, examples, asym_examples = [], [], []
    for sid, q in questions.items():
        S = q["distinct"]
        d = len(S)
        if d == 0:
            continue
        gid = {k: g for k, g in zip(q["keys"], q["gids"]) if k is not None}
        D = np.zeros((d, d), dtype=bool)
        for i in range(d):
            for j in range(d):
                if i != j:
                    D[i, j] = scorer.equal(S[i], S[j], "check_a")
        M = D | D.T
        nonrefl = [s for s in S if scorer.parses(s) and not scorer.equal(s, s, "check_a")]
        G = np.array([gid[s] for s in S])
        B = G[:, None] == G[None, :]
        off = ~np.eye(d, dtype=bool)
        block = int(np.triu(M != B, 1).sum())
        asym = int(np.triu(D != D.T, 1).sum())
        Mi = (M & off).astype(np.int64)
        P = Mi @ Mi
        nontrans = int((P * (~M & off)).sum())
        c = {"pairs_checked": d * (d - 1) // 2, "block_mismatch_pairs": block, "non_transitive_triples": nontrans,
             "non_reflexive_strings": len(nonrefl), "raw_asymmetric_pairs": asym}
        tot.update(c)
        if asym and len(asym_examples) < MAX_EXAMPLES:
            asym_examples.append({"sample_id": sid, "pairs (a, b, verify(a,b), verify(b,a), group a, group b)": [
                (S[i], S[j], bool(D[i, j]), bool(D[j, i]), int(G[i]), int(G[j]))
                for i in range(d) for j in range(i + 1, d) if D[i, j] != D[j, i]][:5]})
        if block or nontrans or nonrefl:
            affected.append(sid)
            if len(examples) < MAX_EXAMPLES:
                ex = {"sample_id": sid, **{k: v for k, v in c.items() if k != "pairs_checked"}}
                ex["pairs (a, b, equal_sym, group a, group b)"] = [
                    (S[i], S[j], bool(M[i, j]), int(G[i]), int(G[j]))
                    for i in range(d) for j in range(i + 1, d) if M[i, j] != B[i, j]][:5]
                ex["non_reflexive"] = nonrefl[:5]
                examples.append(ex)
    return {"relation": "symmetric: verify(a, b) or verify(b, a)",
            **{k: int(tot[k]) for k in ("pairs_checked", "block_mismatch_pairs", "non_transitive_triples",
                                         "non_reflexive_strings", "raw_asymmetric_pairs")},
            "questions_affected": len(affected), "affected_ids": affected[:MAX_EXAMPLES], "examples": examples,
            "raw_asymmetric_examples": asym_examples, "passed": not affected}


def check_grading_consistency(questions, scorer):
    """(b) Every member string of a group, graded against gold on its own, must get the group's grade."""
    n_members = mism = rows_affected = 0
    examples = []
    for sid, q in questions.items():
        members = {}
        for k, g in zip(q["keys"], q["gids"]):
            if k is not None:
                members.setdefault(g, Counter())[k] += 1
        for g, cnt in members.items():
            for s, m in cnt.items():
                n_members += 1
                own = q["grades"][g] if s == q["reps"][g] else scorer.grade(q["gold_key"], s, "check_b")
                if own != q["grades"][g]:
                    mism += 1
                    rows_affected += m
                    if len(examples) < MAX_EXAMPLES:
                        examples.append({"sample_id": sid, "group_id": g, "representative": q["reps"][g],
                                         "member": s, "group_grade": bool(q["grades"][g]),
                                         "member_grade": bool(own)})
    return {"member_strings_checked": n_members, "mismatched_strings": mism, "rollouts_affected": rows_affected,
            "examples": examples, "passed": mism == 0}


def check_gold(questions, scorer, gold, metadata=None):
    """(c) Every gold string must parse (MATH) or normalize to non-None (MathVista)."""
    fails = []
    for sid, q in questions.items():
        if scorer.name == "math":
            if not scorer.parses(q["gold_key"]):
                fails.append({"sample_id": sid, "gold": gold[sid], "reason": "parses to nothing"})
        else:
            r = mathvista_gold_failure(gold[sid], metadata[sid], q["gold_key"])
            if r:
                fails.append({"sample_id": sid, "gold": gold[sid], "reason": r})
    return {"failures": len(fails), "list": fails, "passed": not fails}


def parse_failures(questions, scorer):
    """Non-empty MATH strings that parse to nothing (counted per distinct string within a question and per rollout)."""
    n_str = n_rows = 0
    examples = []
    for sid, q in questions.items():
        cnt = Counter(k for k in q["keys"] if k is not None)
        for s, m in cnt.items():
            if not scorer.parses(s):
                n_str += 1
                n_rows += m
                if len(examples) < MAX_EXAMPLES:
                    examples.append({"sample_id": sid, "string": s, "rollouts": m})
    return {"distinct_strings": n_str, "rollouts": n_rows, "examples": examples}


# --------------------------------------------------------------------------------------- scorer version

def scorer_version(cfg, testmini_path=None):
    """Frozen scorer identity: sha256 of this file, installed library versions (must match the config),
    math-verify settings and the MathVista source commit and metadata hash."""
    sc = cfg["scorer"]
    libs = {name: importlib.metadata.version(name) for name in sc["libraries"]}
    bad = {k: (v, sc["libraries"][k]) for k, v in libs.items() if v != str(sc["libraries"][k])}
    if bad:
        raise RuntimeError(f"installed scorer libraries differ from the config (installed, expected): {bad}")
    v = {"answers_py_sha256": sha256_file(os.path.abspath(__file__)), "libraries": libs,
         "math_equivalence": MATH_EQUIVALENCE,
         "math_verify": sc["math_verify"], "mathvista": {"repo": sc["mathvista"]["repo"],
                                                        "commit": sc["mathvista"]["commit"]}}
    if testmini_path is not None:
        h = sha256_file(testmini_path)
        exp = sc["mathvista"].get("testmini_sha256")
        if exp and h != exp:
            raise RuntimeError(f"{testmini_path} sha256 {h} differs from the config {exp}")
        v["mathvista"]["testmini_sha256"] = h
    v["id"] = sha256_json(v)
    return v


def make_scorer(cfg, dataset):
    if dataset == "math":
        mv = cfg["scorer"]["math_verify"]
        return MathScorer(parse_timeout=mv["parse_timeout_s"], verify_timeout=mv["verify_timeout_s"])
    return MathVistaScorer()
