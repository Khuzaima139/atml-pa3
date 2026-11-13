"""Task 3 Step 4 revision examples from the live run, rules fixed in advance (manual L717-719, L777-779).

Run: python -m task3_ttc.live_examples   -> results/task3/qualitative/live_examples.json

Rules (live questions in live order q = 0..49; graded labels from the committed live analysis; a rule that finds no
question records "none"):
    wrong_to_correct   the first question whose draft (p1) is wrong and whose revision (refine) is correct
    correct_to_wrong   the first question whose draft is correct and whose revision is wrong
    truncated_draft    the first question whose draft hit its token cap without EOS
Saved for each: problem, gold, the draft's answer, grade, truncation flag and last 40 words, the revision's answer and
grade and the first 40 and last 40 words of the refine output. Words = whitespace-split tokens of output_text.
aligned: for the first two rules (wrong_to_correct, correct_to_wrong) when found, one line per strategy: single,
parallel (answers p1, p2, p3 and the self-consistency pick), refine. A missing extraction prints as NO_ANSWER.
Inputs are committed result files: results/task3/live/{analysis.json,per_question.csv,main/calls.jsonl} and
splits/task3/live_problems.json.
"""
import csv
import json
import os
import sys

from common.hashing import sha256_file
from common.results import read_result, write_result
from common.runinfo import run_info
from task3_ttc.data import resolve
from task3_ttc.figures import committed

OUT = "results/task3/qualitative/live_examples.json"
INPUTS = {"analysis": "results/task3/live/analysis.json", "per_question": "results/task3/live/per_question.csv",
          "calls": "results/task3/live/main/calls.jsonl", "problems": "splits/task3/live_problems.json"}
WORDS = 40
RULES = ("wrong_to_correct", "correct_to_wrong", "truncated_draft")
ALIGNED_RULES = ("wrong_to_correct", "correct_to_wrong")
NA = "NO_ANSWER"


def first_words(text, k=WORDS):
    return " ".join(str(text).split()[:k])


def last_words(text, k=WORDS):
    return " ".join(str(text).split()[-k:])


def ans(a):
    return NA if a is None or a == "" else a


def load_rows(path):
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert [int(r["q"]) for r in rows] == list(range(len(rows))), "per_question.csv not in live order"
    return rows


def load_calls(path):
    by = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                k = (r["q"], r["strategy"])
                assert k not in by, f"duplicate call {k}"
                by[k] = r
    return by


def find(rows):
    """Rule -> first matching row index (live order) or None."""
    tests = {"wrong_to_correct": lambda r: r["initial_correct"] == "0" and r["refine_correct"] == "1",
             "correct_to_wrong": lambda r: r["initial_correct"] == "1" and r["refine_correct"] == "0",
             "truncated_draft": lambda r: r["p1_truncated"] == "1"}
    return {k: next((i for i, r in enumerate(rows) if t(r)), None) for k, t in tests.items()}


def example(row, calls, problem):
    q = int(row["q"])
    p1, rf = calls[(q, "p1")], calls[(q, "refine")]
    assert p1["sample_id"] == rf["sample_id"] == row["sample_id"] == problem["sample_id"]
    return {"q": q, "sample_id": row["sample_id"], "problem": problem["problem"], "gold": row["gold"],
            "draft": {"call": "p1", "answer": ans(p1["extracted_answer"]), "correct": row["initial_correct"] == "1",
                      "truncated": bool(p1["truncated"]), "output_tokens": p1["output_tokens"],
                      "last_40_words": last_words(p1["output_text"])},
            "revision": {"call": "refine", "answer": ans(rf["extracted_answer"]), "correct": row["refine_correct"] == "1",
                         "truncated": bool(rf["truncated"]), "output_tokens": rf["output_tokens"],
                         "answer_changed": row["refine_answer_changed"] == "1",
                         "first_40_words": first_words(rf["output_text"]),
                         "last_40_words": last_words(rf["output_text"])}}


def mark(ok):
    return "correct" if ok else "wrong"


def aligned(row, calls):
    q = int(row["q"])
    c = {s: calls[(q, s)] for s in ("single", "p1", "p2", "p3", "refine")}
    par = {s: {"answer": ans(c[s]["extracted_answer"]), "correct": row[f"{s}_call_correct"] == "1",
               "truncated": bool(c[s]["truncated"])} for s in ("p1", "p2", "p3")}
    pick = {"call": row["parallel_selected_call"], "answer": ans(row["parallel_answer"]),
            "correct": row["parallel_correct"] == "1", "groups": json.loads(row["parallel_groups"])}
    single = {"answer": ans(c["single"]["extracted_answer"]), "correct": row["single_correct"] == "1",
              "truncated": bool(c["single"]["truncated"])}
    refine = {"answer": ans(c["refine"]["extracted_answer"]), "correct": row["refine_correct"] == "1",
              "truncated": bool(c["refine"]["truncated"])}

    def cell(d):
        return f"{d['answer']} ({mark(d['correct'])}{', truncated' if d['truncated'] else ''})"

    lines = {"single": f"single   | {cell(single)}",
             "parallel": "parallel | " + " ; ".join(f"{s} {cell(par[s])}" for s in ("p1", "p2", "p3"))
                         + f" | SC pick {pick['call']}: {pick['answer']} ({mark(pick['correct'])})",
             "refine": f"refine   | {cell(refine)}"}
    return {"gold": row["gold"], "single": single, "parallel": {**par, "sc_pick": pick}, "refine": refine,
            "lines": lines}


def build(paths, out_path, info):
    if os.path.exists(out_path):
        raise FileExistsError(f"refusing to overwrite {out_path}")
    an = read_result(paths["analysis"])
    rows = load_rows(paths["per_question"])
    calls = load_calls(paths["calls"])
    with open(paths["problems"], encoding="utf-8") as f:
        problems = json.load(f)["problems"]
    assert an["inputs"]["calls_jsonl_sha256"] == sha256_file(paths["calls"]), "analysis is not of these calls"
    assert len(rows) == len(problems) and len(calls) == 5 * len(rows)
    hit = find(rows)
    doc = {"run_id": f"task3_live_examples_{info['timestamp'][:19].replace(':', '').replace('-', '')}Z",
           "run_info": info, "rules": __doc__, "live_run_id": an["inputs"]["live_run_id"],
           "inputs": {k: {"path": INPUTS[k], "sha256": sha256_file(p)} for k, p in paths.items()},
           "examples": {k: "none" if i is None else example(rows[i], calls, problems[i]) for k, i in hit.items()},
           "aligned": {}}
    for k in ALIGNED_RULES:
        i = hit[k]
        doc["aligned"][k] = "none" if i is None else {"q": i, "sample_id": rows[i]["sample_id"],
                                                      **aligned(rows[i], calls)}
    write_result(out_path, doc)
    return doc


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser()
    ap.parse_args(argv)
    info = run_info()
    try:
        paths = {k: committed(p) for k, p in INPUTS.items()}
        doc = build(paths, resolve(OUT), info)
    except (FileExistsError, RuntimeError) as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(2)
    for k in RULES:
        e = doc["examples"][k]
        print(f"{k}: " + (e if e == "none" else
              f"q{e['q']} {e['sample_id']} draft {e['draft']['answer']} ({mark(e['draft']['correct'])}, "
              f"truncated {e['draft']['truncated']}) -> revision {e['revision']['answer']} "
              f"({mark(e['revision']['correct'])}), gold {e['gold']}"))
    for k in ALIGNED_RULES:
        a = doc["aligned"][k]
        print(f"aligned {k}: " + ("none" if a == "none" else f"q{a['q']} {a['sample_id']} gold {a['gold']}"))
        if a != "none":
            for line in a["lines"].values():
                print("   " + line)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
