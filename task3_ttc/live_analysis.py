"""Task 3 Step 4 analysis of the live calls (manual L699-719, L772-773), run on the Mac after import.

    python -m task3_ttc.live_analysis [--run results/task3/live/main] [--out-dir results/task3/live]
                                      [--gold GOLD.json] [--config configs/task3/offline.yaml] [--replace]

Grading uses the frozen MATH scorer (task3_ttc.answers: symmetric math-verify on \\boxed{}-wrapped strings;
a missing or empty extraction is NO_ANSWER, always wrong). Gold comes from the supplied MATH generations
(`gold_answer`, constant within a question) unless --gold gives a {sample_id: gold} JSON (tests).

Strategies per question:
    single   = call 0 (3,072 cap)
    parallel = self-consistency over p1, p2, p3 with the offline grouping (answers.group_keys in p1, p2, p3 order,
               NO_ANSWER one group that votes) and selection.self_consistency (ties -> earliest of p1, p2, p3)
    refine   = the refine call's answer (draft p1 + critique-and-revision call)
    initial  = p1 alone (the shared draft)
Costs per question: parallel = p1 + p2 + p3; refine = p1 + refine call (the shared draft charged to both parallel
and refine, L715-716); runtime = sum of the strategy's call times.
Outputs: <out-dir>/analysis.json and per_question.csv (never overwritten without --replace).
"""
import argparse
import csv
import io
import json
import os

import numpy as np

from common.bootstrap import paired_bootstrap
from common.hashing import sha256_file, sha256_json
from common.results import read_result
from common.runinfo import run_info
from task3_ttc import live, live_problems
from task3_ttc.answers import NO_ANSWER, group_keys, make_scorer, scorer_version
from task3_ttc.data import dataset_files, load_config, resolve
from task3_ttc.selection import self_consistency

STRATEGIES = ["single", "parallel", "refine", "initial"]
CALLS_OF = {"single": ["single"], "parallel": ["p1", "p2", "p3"], "refine": ["p1", "refine"], "initial": ["p1"]}
ANSWER_CALLS = {"single": ["single"], "parallel": ["p1", "p2", "p3"], "refine": ["refine"], "initial": ["p1"]}
BOOTSTRAP_PAIRS = [("parallel", "single"), ("refine", "single"), ("refine", "parallel"), ("single", "initial"),
                   ("refine", "initial")]


def load_calls(run_dir):
    run = read_result(os.path.join(run_dir, "run.json"))
    path = os.path.join(run_dir, "calls.jsonl")
    with open(path, encoding="utf-8") as f:
        recs = [json.loads(x) for x in f if x.strip()]
    return run, recs, path


def load_gold(cfg, ids, gold_path=None):
    if gold_path is not None:
        with open(gold_path) as f:
            g = json.load(f)
        return {s: g[s] for s in ids}, {"source": gold_path, "sha256": sha256_file(gold_path)}
    import pyarrow.parquet as pq
    p = dataset_files(cfg, "math")["generations"]
    t = pq.read_table(p, columns=["sample_id", "gold_answer"]).to_pydict()
    want, g = set(ids), {}
    for s, a in zip(t["sample_id"], t["gold_answer"]):
        if s in want:
            assert g.setdefault(s, a) == a, f"gold not constant within {s}"
    assert set(g) == want
    return g, {"source": "qwen25-math/generations.parquet gold_answer", "sha256": sha256_file(p)}


def check_calls(run, recs):
    """All calls present once, in the fixed order and seeds, for the run's questions."""
    cfg = run["config"]
    n = cfg["n_questions"]
    by = {}
    for r in recs:
        k = (r["q"], r["call"])
        assert k not in by, f"duplicate call {k}"
        assert r["run_id"] == run["run_id"]
        by[k] = r
    assert set(by) == {(q, c) for q in range(n) for c, _, _ in live.CALLS}, "missing calls"
    for (q, c), r in by.items():
        assert r["strategy"] == live.CALLS[c][1] and r["seed"] == live.call_seed(q, c)
        assert r["sample_id"] == cfg["sample_ids"][q]
    return by


def same_answer(scorer, a, b):
    if a is None or b is None:
        return a is None and b is None
    return scorer.equal_sym(a, b, "analysis")


def analyse(run, recs, gold, scorer):
    by = check_calls(run, recs)
    n = run["config"]["n_questions"]
    rows, res = [], {s: {"correct": [], "answer": [], "in": [], "out": [], "wall": [], "peak": [], "peak_r": []}
                     for s in STRATEGIES}
    call_correct = {}
    for q in range(n):
        c = {s: by[(q, i)] for i, s, _ in live.CALLS}
        sid = c["single"]["sample_id"]
        gk = scorer.gold_key(gold[sid])
        keys = {s: scorer.key(r["extracted_answer"])[0] for s, r in c.items()}
        graded = {s: k is not None and scorer.grade(gk, k) for s, k in keys.items()}
        call_correct[q] = graded
        gids, reps, _ = group_keys([keys["p1"], keys["p2"], keys["p3"]], scorer.same)
        pos = self_consistency(gids)
        sel_g = gids[pos]
        par_key = None if sel_g == NO_ANSWER else reps[sel_g]
        par_ok = sel_g != NO_ANSWER and scorer.grade(gk, reps[sel_g])
        answers = {"single": (keys["single"], graded["single"]), "parallel": (par_key, par_ok),
                   "refine": (keys["refine"], graded["refine"]), "initial": (keys["p1"], graded["p1"])}
        row = {"q": q, "sample_id": sid, "gold": gold[sid]}
        for s in STRATEGIES:
            cs = [c[x] for x in CALLS_OF[s]]
            a, ok = answers[s]
            d = res[s]
            d["correct"].append(bool(ok))
            d["answer"].append(a)
            d["in"].append(sum(x["input_tokens"] for x in cs))
            d["out"].append(sum(x["output_tokens"] for x in cs))
            d["wall"].append(sum(x["wall_s"] for x in cs))
            d["peak"].append(max((x["peak_mem_allocated_bytes"] or 0) for x in cs))
            d["peak_r"].append(max((x["peak_mem_reserved_bytes"] or 0) for x in cs))
            row.update({f"{s}_answer": a, f"{s}_correct": int(bool(ok)), f"{s}_input_tokens": d["in"][-1],
                        f"{s}_output_tokens": d["out"][-1], f"{s}_wall_s": round(d["wall"][-1], 4)})
        row["parallel_groups"] = json.dumps(gids)
        row["parallel_selected_call"] = ["p1", "p2", "p3"][pos]
        for s, r in c.items():
            row.update({f"{s}_extracted": r["extracted_answer"], f"{s}_call_correct": int(graded[s]),
                        f"{s}_tokens": r["output_tokens"], f"{s}_truncated": int(r["truncated"])})
        row["refine_answer_changed"] = int(not same_answer(scorer, keys["p1"], keys["refine"]))
        rows.append(row)

    summary = {}
    for s in STRATEGIES:
        d = res[s]
        resp = [by[(q, i)] for q in range(n) for i in range(len(live.CALLS))
                if live.CALLS[i][1] in ANSWER_CALLS[s]]
        m = {"n_questions": n, "accuracy": float(np.mean(d["correct"])), "n_correct": int(sum(d["correct"])),
             "answer_calls": ANSWER_CALLS[s], "cost_calls": CALLS_OF[s],
             "extraction_rate": float(np.mean([r["extracted_answer"] is not None for r in resp])),
             "truncation_rate": float(np.mean([r["truncated"] for r in resp])),
             "n_responses": len(resp),
             "input_tokens_per_question": float(np.mean(d["in"])), "output_tokens_per_question": float(np.mean(d["out"])),
             "total_tokens_per_question": float(np.mean(np.add(d["in"], d["out"]))),
             "input_tokens_total": int(sum(d["in"])), "output_tokens_total": int(sum(d["out"])),
             "runtime_s_per_question": float(np.mean(d["wall"])), "runtime_s_total": float(sum(d["wall"])),
             "peak_mem_allocated_bytes": int(max(d["peak"])), "peak_mem_reserved_bytes": int(max(d["peak_r"]))}
        if s == "parallel":
            m["extraction_rate_note"] = "per response over the 3 parallel calls"
            m["selected_no_answer_rate"] = float(np.mean([a is None for a in d["answer"]]))
        if s == "refine":
            m["cost_note"] = "includes the shared p1 draft and the longer refinement prompt"
        summary[s] = m

    ini, ref = np.array(res["initial"]["correct"]), np.array(res["refine"]["correct"])
    transitions = {"relative_to": "initial (p1, the shared draft)",
                   "wrong_to_correct": int((~ini & ref).sum()), "correct_to_wrong": int((ini & ~ref).sum()),
                   "both_correct": int((ini & ref).sum()), "both_wrong": int((~ini & ~ref).sum()),
                   "answer_changed": int(sum(r["refine_answer_changed"] for r in rows)),
                   "answer_changed_rule": "refine answer not equivalent to p1's under the scorer "
                                          "(NO_ANSWER equals only NO_ANSWER)"}
    boot = {f"{a}-{b}": paired_bootstrap(np.array(res[a]["correct"], float), np.array(res[b]["correct"], float),
                                         n_resamples=1000, seed=6304) for a, b in BOOTSTRAP_PAIRS}
    calls = {}
    for _, s, _ in live.CALLS:
        rs = [by[(q, i)] for q in range(n) for i in range(len(live.CALLS)) if live.CALLS[i][1] == s]
        calls[s] = {"n": len(rs), "accuracy": float(np.mean([call_correct[r["q"]][s] for r in rs])),
                    "extraction_rate": float(np.mean([r["extracted_answer"] is not None for r in rs])),
                    "truncation_rate": float(np.mean([r["truncated"] for r in rs])),
                    "finished_by_eos_rate": float(np.mean([r["finished_by_eos"] for r in rs])),
                    "mean_input_tokens": float(np.mean([r["input_tokens"] for r in rs])),
                    "mean_output_tokens": float(np.mean([r["output_tokens"] for r in rs])),
                    "mean_wall_s": float(np.mean([r["wall_s"] for r in rs]))}
    return {"strategies": summary, "calls": calls, "refine_transitions": transitions, "bootstrap": boot}, rows


def to_csv(rows):
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=list(rows[0]), lineterminator="\n")
    w.writeheader()
    for r in rows:
        w.writerow({k: ("" if v is None else v) for k, v in r.items()})
    return buf.getvalue()


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="results/task3/live/main")
    ap.add_argument("--out-dir", default="results/task3/live")
    ap.add_argument("--gold", default=None)
    ap.add_argument("--config", default="configs/task3/offline.yaml")
    ap.add_argument("--replace", action="store_true")
    a = ap.parse_args(argv)
    prov = run_info()                                         # before any output is written
    cfg = load_config(a.config)
    run_dir, out_dir = resolve(a.run), resolve(a.out_dir)
    out_json, out_csv = os.path.join(out_dir, "analysis.json"), os.path.join(out_dir, "per_question.csv")
    for p in (out_json, out_csv):
        if os.path.exists(p) and not a.replace:
            raise SystemExit(f"{p} exists; pass --replace to overwrite")
    run, recs, calls_path = load_calls(run_dir)
    if run["status"] != "complete":
        raise SystemExit(f"run {run['run_id']} status {run['status']!r}, expected 'complete'")
    problems = live_problems.load()
    assert run["config"]["problems_sha256"] == problems["sha256"], "run used a different live problem file"
    gold, gold_src = load_gold(cfg, run["config"]["sample_ids"], a.gold)
    scorer = make_scorer(cfg, "math")
    version = scorer_version(cfg)
    res, rows = analyse(run, recs, gold, scorer)
    out = {"run_id": f"task3_live_analysis_of_{run['run_id']}", "provenance": prov, "config": a.config,
           "inputs": {"run_dir": a.run, "live_run_id": run["run_id"], "calls_jsonl_sha256": sha256_file(calls_path),
                      "run_config_sha256": sha256_json(run["config"]), "gold": gold_src,
                      "problems_sha256": problems["sha256"]},
           "scorer_version": version,
           "scorer_timeouts": dict(scorer.timeouts),
           "definitions": {"strategies": CALLS_OF, "answer_calls": ANSWER_CALLS,
                           "parallel_selection": "answers.group_keys over p1, p2, p3 (symmetric math-verify, "
                                                 "NO_ANSWER one voting group) + selection.self_consistency "
                                                 "(ties -> earliest of p1, p2, p3); graded on the group representative",
                           "bootstrap": "common.bootstrap.paired_bootstrap over questions, 1,000 resamples, seed 6304"},
           "run_settings": {"effective_generation_config": run["effective_generation_config"],
                            "logits_processors": run["logits_processors"],
                            "gpu": [s["torch"].get("gpu_name") for s in run["sessions"]],
                            "commits": [s["provenance"]["git"] for s in run["sessions"]],
                            "libraries": run["sessions"][-1]["libraries"]},
           **res}
    os.makedirs(out_dir, exist_ok=True)
    for p in (out_json, out_csv):
        if os.path.exists(p):
            os.remove(p)
    with open(out_json, "w") as f:
        json.dump(out, f, indent=2, default=str)
        f.write("\n")
    with open(out_csv, "w") as f:
        f.write(to_csv(rows))
    for s in STRATEGIES:
        m = res["strategies"][s]
        print(f"{s:9s} acc {m['accuracy']:.3f}  extract {m['extraction_rate']:.3f}  trunc {m['truncation_rate']:.3f}  "
              f"in/q {m['input_tokens_per_question']:.0f}  out/q {m['output_tokens_per_question']:.0f}  "
              f"time/q {m['runtime_s_per_question']:.1f} s")
    print("refine transitions:", {k: v for k, v in res["refine_transitions"].items() if isinstance(v, int)})
    for k, b in res["bootstrap"].items():
        print(f"{k:18s} {b['mean']:+.3f} [{b['ci_low']:+.3f}, {b['ci_high']:+.3f}]")
    print(f"wrote {out_json} and {out_csv}")


if __name__ == "__main__":
    main()
