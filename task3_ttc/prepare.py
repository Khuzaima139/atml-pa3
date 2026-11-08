"""Task 3 Step 1 preparation (manual L593-606, L630-631): join, split, answer groups, correctness labels.

Run: python -m task3_ttc.prepare --dataset {math,mathvista} [--config configs/task3/offline.yaml] [--replace]

Writes (never overwrites unless --replace is given; otherwise exits with an error if an output exists):
    results/task3/rollouts/<dataset>_rollouts.csv   one row per rollout (columns in COLUMNS)
    results/task3/prepare_<dataset>.json            config, provenance, scorer_version, runtimes, counts, checks
and, if absent, splits/task3/<dataset>_split.json (if present it must equal the recomputed split), plus the
gitignored token cache cached/task3/<dataset>_tokens.npz.
"""
import argparse
import datetime
import json
import os
import sys
import time

import numpy as np
import pandas as pd

from common.hashing import sha256_file
from common.results import write_result
from common.runinfo import run_info
from task3_ttc import answers
from task3_ttc.data import load_config, load_rollouts, resolve, write_token_cache
from task3_ttc.splits import ensure_split

COLUMNS = {
    "sample_id": "question ID (string)",
    "rollout_id": "0..31 within the question",
    "split": "dev or eval (L593-596)",
    "live": "MATH only: in the 50-question live subset (first 50 eval IDs)",
    "T": "retained generated tokens",
    "capped": "hit_max_new_tokens (supplied flag)",
    "L": "mean logp over the T tokens (float64)",
    "K": "mean kl_u_p over the T tokens (float64)",
    "V": "MathVista only: mean of img/(img+prompt+generated) over the T tokens (float64)",
    "extracted": "supplied extracted_answer (empty if null)",
    "normalized": "group key: MATH stripped extracted string (wrapped in \\boxed{} for math-verify); "
                  "MathVista normalized answer (option letter for multiple choice); empty for NO_ANSWER",
    "group_id": "answer group within the question, numbered in order of first appearance; NO_ANSWER = -1",
    "na_reason": "NO_ANSWER only: extraction (supplied answer missing/empty) or normalization (MathVista rule "
                 "returned None)",
    "correct": "group graded against gold with the same equivalence rules; NO_ANSWER always False",
}


def _stats(x):
    x = np.asarray(x, dtype=float)
    return {"mean": float(x.mean()), "median": float(np.median(x)), "min": int(x.min()), "max": int(x.max())}


def output_paths(cfg, dataset):
    rd = resolve(cfg["paths"]["results_dir"])
    return (os.path.join(rd, "rollouts", f"{dataset}_rollouts.csv"), os.path.join(rd, f"prepare_{dataset}.json"),
            os.path.join(resolve(cfg["paths"]["cache_dir"]), f"{dataset}_tokens.npz"))


def run(cfg, dataset, config_path, replace=False):
    t0 = time.time()
    csv_path, json_path, cache_path = output_paths(cfg, dataset)
    existing = [p for p in (csv_path, json_path) if os.path.exists(p)]
    if existing and not replace:
        raise FileExistsError(f"refusing to overwrite existing results: {existing} (pass --replace)")
    replaced = None
    if existing:
        replaced = {"files": [os.path.relpath(p, resolve(".")) for p in existing]}
        if os.path.exists(json_path):
            with open(json_path) as f:
                old = json.load(f)
            replaced.update({"run_id": old.get("run_id"), "scorer_version_id": old.get("scorer_version", {}).get("id"),
                             "commit": old.get("run_info", {}).get("git", {}).get("commit")})
    run_id = f"task3_prepare_{dataset}_{datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    info = run_info()   # provenance (commit, dirty) before any output is written: --replace rewrites tracked files

    # scorer identity first: library versions are checked before any work
    mv_meta_path = cfg["datasets"]["mathvista"]["metadata"]
    testmini = resolve(mv_meta_path) if mv_meta_path and os.path.exists(resolve(mv_meta_path)) else None
    if dataset == "mathvista" and testmini is None:
        raise FileNotFoundError("MathVista metadata (testmini.json) not found")
    sv = answers.scorer_version(cfg, testmini)

    data = load_rollouts(cfg, dataset)
    cache_meta = write_token_cache(data, cache_path)
    split, split_path, split_status = ensure_split(cfg, dataset, list(data["gold"]))
    t_load = time.time()

    metadata = None
    if dataset == "mathvista":
        metadata = answers.load_mathvista_metadata(testmini)
        missing = set(data["gold"]) - set(metadata)
        assert not missing, f"{len(missing)} sample_ids without MathVista metadata"
    scorer = answers.make_scorer(cfg, dataset)
    res = answers.score_dataset(data["sample_id"], data["rollout_id"], data["extracted"], data["gold"], scorer,
                                metadata)
    t_score = time.time()
    qs = res["questions"]
    checks = {}
    if dataset == "math":
        checks["a_transitivity"] = answers.check_transitivity(qs, scorer)
    else:
        checks["a_transitivity"] = {"applicable": False,
                                    "reason": "MathVista groups are exact normalized strings (an equivalence)"}
    checks["b_grading_consistency"] = answers.check_grading_consistency(qs, scorer)
    checks["c_gold"] = answers.check_gold(qs, scorer, data["gold"], metadata)
    pf = answers.parse_failures(qs, scorer) if dataset == "math" else None
    t_checks = time.time()

    dev, live = set(split["dev_ids"]), set(split.get("live_ids") or [])
    sid = data["sample_id"]
    cols = {"sample_id": sid, "rollout_id": data["rollout_id"],
            "split": np.where(np.isin(sid, list(dev)), "dev", "eval")}
    if dataset == "math":
        cols["live"] = np.isin(sid, list(live))
    cols.update({"T": data["T"], "capped": data["capped"], "L": data["L"], "K": data["K"]})
    if data["V"] is not None:
        cols["V"] = data["V"]
    cols.update({"extracted": data["extracted"], "normalized": res["normalized"], "group_id": res["group_id"],
                 "na_reason": res["na_reason"], "correct": res["correct"]})
    df = pd.DataFrame(cols)
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    tmp = csv_path + ".tmp"
    df.to_csv(tmp, index=False, lineterminator="\n")
    os.replace(tmp, csv_path)

    gid = res["group_id"]
    na = gid == answers.NO_ANSWER
    reasons = pd.Series(res["na_reason"]).value_counts().to_dict()
    n_groups = [len(q["reps"]) for q in qs.values()]
    n_groups_na = [len(q["reps"]) + (answers.NO_ANSWER in q["gids"]) for q in qs.values()]
    counts = {
        "rollouts": int(len(df)), "questions": len(qs),
        "rows_by_split": {k: int(v) for k, v in df["split"].value_counts().items()},
        "questions_by_split": {"dev": len(split["dev_ids"]), "eval": len(split["eval_ids"]),
                               "live": len(split.get("live_ids") or [])},
        "live_rows": int(df["live"].sum()) if "live" in df else 0,
        **data["stats"],
        "groups_per_question_excl_no_answer": _stats(n_groups),
        "groups_per_question_incl_no_answer": _stats(n_groups_na),
        "distinct_strings_per_question": _stats([len(q["distinct"]) for q in qs.values()]),
        "no_answer_rollouts": int(na.sum()),
        "no_answer_by_reason": {answers.REASON_EXTRACTION: int(reasons.get(answers.REASON_EXTRACTION, 0)),
                                answers.REASON_NORMALIZATION: int(reasons.get(answers.REASON_NORMALIZATION, 0))},
        "questions_with_no_answer": int(sum(answers.NO_ANSWER in q["gids"] for q in qs.values())),
        "parse_failures": pf,
        "verify_calls": dict(scorer.verify_calls),
        "timeouts": {"total": int(sum(scorer.timeouts.values())), **dict(scorer.timeouts)},
        "gold_failures": checks["c_gold"]["failures"],
    }
    out = {
        "run_id": run_id, "status": "done", "dataset": dataset,
        "config_path": config_path, "config": cfg, "run_info": info,
        "scorer_version": sv,
        "inputs": {"input_sha256": data["input_sha256"], "testmini_sha256": sv["mathvista"].get("testmini_sha256")},
        "outputs": {"rollouts_csv": os.path.relpath(csv_path, resolve(".")), "rollouts_csv_sha256": sha256_file(csv_path),
                    "split_file": os.path.relpath(split_path, resolve(".")), "split_sha256": split["sha256"],
                    "split_status": split_status,
                    "token_cache": os.path.relpath(cache_path, resolve(".")), "token_cache_meta": cache_meta},
        "columns": {k: v for k, v in COLUMNS.items() if k in df.columns},
        "replaced": replaced, "counts": counts, "checks": checks,
        "runtime_s": {"load_join_cache_split": round(t_load - t0, 2), "group_grade": round(t_score - t_load, 2),
                      "checks": round(t_checks - t_score, 2), "total": round(time.time() - t0, 2)},
    }
    if replaced:
        os.remove(json_path)   # explicit --replace: write_result refuses a different run_id otherwise
    write_result(json_path, out)
    return out, csv_path, json_path


def summary(out):
    c, k = out["counts"], out["checks"]
    lines = [f"[{out['dataset']}] rows {c['rollouts']}, questions {c['questions']}, split rows {c['rows_by_split']}, "
             f"questions {c['questions_by_split']}, live rows {c['live_rows']}",
             f"tokens {c['n_tokens']}, capped {c['capped']} (flag vs length disagreements "
             f"{c['capped_flag_vs_length_disagreements']})",
             f"groups/question excl NA {c['groups_per_question_excl_no_answer']}",
             f"groups/question incl NA {c['groups_per_question_incl_no_answer']}",
             f"NO_ANSWER rollouts {c['no_answer_rollouts']} by reason {c['no_answer_by_reason']}, "
             f"questions with NO_ANSWER {c['questions_with_no_answer']}",
             f"parse failures {None if c['parse_failures'] is None else {x: c['parse_failures'][x] for x in ('distinct_strings', 'rollouts')}}, "
             f"timeouts {c['timeouts']}, verify calls {c['verify_calls']}",
             f"check a {({x: v for x, v in k['a_transitivity'].items() if x not in ('examples', 'affected_ids')})}",
             f"check b {({x: v for x, v in k['b_grading_consistency'].items() if x != 'examples'})}",
             f"check c gold failures {k['c_gold']['failures']}",
             f"scorer_version {out['scorer_version']['id']}, split {out['outputs']['split_status']} "
             f"sha256 {out['outputs']['split_sha256'][:12]}, runtime {out['runtime_s']}"]
    return "\n".join(lines)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=["math", "mathvista"])
    ap.add_argument("--config", default="configs/task3/offline.yaml")
    ap.add_argument("--replace", action="store_true", help="overwrite existing outputs of this dataset")
    a = ap.parse_args(argv)
    cfg = load_config(a.config)
    try:
        out, csv_path, json_path = run(cfg, a.dataset, a.config, replace=a.replace)
    except FileExistsError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(2)
    print(summary(out))
    print(f"wrote {csv_path}\nwrote {json_path}")


if __name__ == "__main__":
    main()
