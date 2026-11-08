"""Task 3 offline experiments, Steps 1-3 (manual L607-615, L630-698; prompt b2 items C-H).

Run: python -m task3_ttc.offline --dataset {math,mathvista} [--config configs/task3/offline.yaml] [--replace]

Inputs: results/task3/rollouts/<dataset>_rollouts.csv and prepare_<dataset>.json (task3_ttc.prepare; the CSV
sha256 must match), splits/task3/<dataset>_split.json (orders) and, for MathVista, the token cache
cached/task3/mathvista_tokens.npz plus testmini.json (question types).

Writes (never overwrites unless --replace):
    results/task3/offline/<dataset>_per_question.csv   one row per (question, method, N), order-averaged
    results/task3/offline/<dataset>_selections.npz     per (question, order, N, method) selections
    results/task3/offline/<dataset>_summary.json       config, provenance, eval/dev means, checks, by-type
    results/task3/signals/<dataset>_signals.json       Step 3 at N = 32 (eval questions)
    results/task3/offline/<dataset>_bootstrap.json     paired bootstrap differences (eval questions)
Means average over orders within a question, then over questions.
"""
import argparse
import datetime
import json
import os
import sys
import time

import numpy as np
import pandas as pd

from common.bootstrap import paired_bootstrap
from common.hashing import sha256_file
from common.results import write_result
from common.runinfo import run_info
from task3_ttc import selection as S
from task3_ttc import signals as G
from task3_ttc.answers import load_mathvista_metadata
from task3_ttc.data import load_config, load_token_cache, resolve
from task3_ttc.splits import load_split, orders_array, split_path

PER_QUESTION_COLUMNS = {
    "sample_id": "question ID", "split": "dev or eval", "method": "selection rule", "N": "candidate budget",
    "accuracy": "share of the orders whose selected answer is correct",
    "oracle": "share of the orders with a correct candidate among the N",
    "distinct_groups_incl_no_answer": "mean number of answer groups among the N (NO_ANSWER counts as one)",
    "distinct_groups_excl_no_answer": "mean number of non-NO_ANSWER answer groups among the N",
    "cost": "mean generated-token cost (FirstAns: T of candidate 0; others: sum of T over the N)",
    "no_answer_selected": "share of the orders whose selected group is NO_ANSWER",
    "sampling_failure": "share of the orders wrong with no correct candidate (oracle 0)",
    "selection_failure": "share of the orders wrong although a candidate was correct (oracle 1)",
}


def bootstrap_pairs(dataset, methods):
    """(a, b) method pairs compared at each bootstrap budget."""
    pairs = [(m, "SC") for m in methods if m != "SC"] + [("RWC-K", "Highest-K")]
    if dataset == "mathvista":
        pairs += [("RWC-V", "Highest-V"), ("RWC-Z", "Highest-Z"), ("RWC-Ztok", "Highest-Ztok"),
                  ("Highest-Z", "Highest-V"), ("Highest-Z", "Highest-K"), ("RWC-Z", "RWC-K"),
                  ("Highest-Ztok", "Highest-Z"), ("RWC-Ztok", "RWC-Z")]
    return pairs


def output_paths(cfg, dataset):
    rd = resolve(cfg["paths"]["results_dir"])
    off = os.path.join(rd, "offline")
    return {"per_question": os.path.join(off, f"{dataset}_per_question.csv"),
            "selections": os.path.join(off, f"{dataset}_selections.npz"),
            "summary": os.path.join(off, f"{dataset}_summary.json"),
            "signals": os.path.join(rd, "signals", f"{dataset}_signals.json"),
            "bootstrap": os.path.join(off, f"{dataset}_bootstrap.json")}


def load_inputs(cfg, dataset):
    """Rollout table, prepare JSON, split/orders and (MathVista) token lists and question types."""
    rd = resolve(cfg["paths"]["results_dir"])
    csv_path = os.path.join(rd, "rollouts", f"{dataset}_rollouts.csv")
    prep_path = os.path.join(rd, f"prepare_{dataset}.json")
    with open(prep_path) as f:
        prep = json.load(f)
    csv_sha = sha256_file(csv_path)
    if csv_sha != prep["outputs"]["rollouts_csv_sha256"]:
        raise RuntimeError(f"{csv_path} sha256 differs from {prep_path}")
    df = pd.read_csv(csv_path, dtype={"sample_id": str, "split": str}, keep_default_na=False,
                     usecols=lambda c: c in {"sample_id", "rollout_id", "split", "T", "L", "K", "V", "group_id",
                                             "correct"})
    sp_path = split_path(cfg, dataset)
    sp = load_split(sp_path)
    ids = sp["sorted_ids"]
    R = cfg["sizes"]["n_rollouts"]
    nq = len(ids)
    sid = df["sample_id"].to_numpy().reshape(nq, R)
    assert (sid == np.array(ids)[:, None]).all(), "rollout CSV not in sorted-ID order"
    assert (df["rollout_id"].to_numpy().reshape(nq, R) == np.arange(R)).all(), "rollout ids not 0..R-1"
    split = df["split"].to_numpy().reshape(nq, R)[:, 0]
    assert set(np.array(ids)[split == "dev"]) == set(sp["dev_ids"]), "CSV split differs from the split file"
    cols = {"T": np.int64, "L": np.float64, "K": np.float64, "group_id": np.int64, "correct": bool}
    if "V" in df.columns:
        cols["V"] = np.float64
    arr = {c: df[c].to_numpy().astype(t).reshape(nq, R) for c, t in cols.items()}
    qs = [{"T": arr["T"][i], "L": arr["L"][i], "K": arr["K"][i], "g": arr["group_id"][i],
           "correct": arr["correct"][i], **({"V": arr["V"][i]} if "V" in arr else {})} for i in range(nq)]
    inputs = {"rollouts_csv": os.path.relpath(csv_path, resolve(".")), "rollouts_csv_sha256": csv_sha,
              "prepare_json": os.path.relpath(prep_path, resolve(".")), "prepare_run_id": prep["run_id"],
              "split_file": os.path.relpath(sp_path, resolve(".")), "split_sha256": sp["sha256"]}
    types = None
    if dataset == "mathvista":
        cache_path = os.path.join(resolve(cfg["paths"]["cache_dir"]), f"{dataset}_tokens.npz")
        tc = load_token_cache(cache_path)
        assert tc["meta"]["input_sha256"] == prep["inputs"]["input_sha256"], "token cache from other inputs"
        assert (tc["sample_id"].reshape(nq, R) == sid).all() and (tc["rollout_id"].reshape(nq, R) ==
                                                                    np.arange(R)).all(), "token cache order"
        off = tc["offsets"]
        assert (np.diff(off) == arr["T"].ravel()).all(), "token cache lengths differ from T"
        kl, v = tc["kl_u_p"].astype(np.float64), tc["v"].astype(np.float64)
        for i, q in enumerate(qs):
            rows = range(i * R, (i + 1) * R)
            q["kl"] = [kl[off[r]:off[r + 1]] for r in rows]
            q["v"] = [v[off[r]:off[r + 1]] for r in rows]
        vmean = np.array([[x.mean() for x in q["v"]] for q in qs])
        inputs["token_cache"] = os.path.relpath(cache_path, resolve("."))
        inputs["token_cache_inputs_sha256"] = tc["meta"]["inputs_sha256"]
        inputs["max_abs_V_csv_minus_mean_v_float32"] = float(np.abs(vmean - arr["V"]).max())
        meta = load_mathvista_metadata(resolve(cfg["datasets"]["mathvista"]["metadata"]))
        types = np.array([meta[s]["question_type"] for s in ids])
    orders = orders_array(sp)
    return {"ids": ids, "split": split, "qs": qs, "orders": orders, "types": types, "prepare": prep,
            "inputs": inputs}


def per_question_frame(ids, split, pq, methods, budgets):
    nq, nb, nm = pq["accuracy"].shape
    qi, bi, mi = np.meshgrid(np.arange(nq), np.arange(nb), np.arange(nm), indexing="ij")
    qi, bi, mi = qi.ravel(), bi.ravel(), mi.ravel()
    return pd.DataFrame({
        "sample_id": np.asarray(ids)[qi], "split": split[qi], "method": np.asarray(methods)[mi],
        "N": np.asarray(budgets)[bi], "accuracy": pq["accuracy"][qi, bi, mi], "oracle": pq["oracle"][qi, bi],
        "distinct_groups_incl_no_answer": pq["distinct_incl"][qi, bi],
        "distinct_groups_excl_no_answer": pq["distinct_excl"][qi, bi], "cost": pq["cost"][qi, bi, mi],
        "no_answer_selected": pq["no_answer_selected"][qi, bi, mi],
        "sampling_failure": pq["sampling_failure"][qi, bi, mi],
        "selection_failure": pq["selection_failure"][qi, bi, mi]})


def by_type(pq, types, mask, methods, budgets):
    isc = methods.index("SC")
    out = {}
    for t in sorted(set(types)):
        m = mask & (types == t)
        out[t] = {"n_questions": int(m.sum()),
                  "oracle": {str(n): float(pq["oracle"][m, b].mean()) for b, n in enumerate(budgets)},
                  "SC_accuracy": {str(n): float(pq["accuracy"][m, b, isc].mean()) for b, n in enumerate(budgets)}}
    return out


def bootstrap(pq, mask, dataset, methods, budgets, cfg):
    bs = cfg["bootstrap"]
    acc = pq["accuracy"][mask]
    rows = []

    def add(a, na, b, nb):
        r = paired_bootstrap(acc[:, budgets.index(na), methods.index(a)], acc[:, budgets.index(nb), methods.index(b)],
                             n_resamples=bs["resamples"], seed=cfg["seed"], ci=bs["ci"])
        rows.append({"a": a, "N_a": na, "b": b, "N_b": nb, "mean_pp": 100 * r["mean"],
                     "ci_low_pp": 100 * r["ci_low"], "ci_high_pp": 100 * r["ci_high"],
                     "ci_excludes_zero": bool(r["ci_low"] > 0 or r["ci_high"] < 0)})

    for n in cfg["selection"]["bootstrap_budgets"]:
        for a, b in bootstrap_pairs(dataset, methods):
            add(a, n, b, n)
    add("SC", budgets[-1], "FirstAns", budgets[-1])
    return {"unit": "question (eval split), accuracy averaged over orders first",
            "method": "paired percentile bootstrap", "resamples": bs["resamples"], "seed": cfg["seed"],
            "ci": bs["ci"], "n_questions": int(mask.sum()), "comparisons": rows}


def z_vs_ztok(res, mask, methods, budgets):
    b = len(budgets) - 1
    iz, it = methods.index("Highest-Z"), methods.index("Highest-Ztok")
    d_roll = (res["sel"][:, :, b, iz] != res["sel"][:, :, b, it]).mean(1)[mask]
    d_grp = (res["group"][:, :, b, iz] != res["group"][:, :, b, it]).mean(1)[mask]
    return {"N": budgets[-1], "share_top_rollout_differs": float(d_roll.mean()),
            "share_selected_group_differs": float(d_grp.mean()),
            "note": "Highest-Z vs Highest-Ztok selections at N = 32, averaged over orders then eval questions "
                    "(order matters only through ties)"}


def run(cfg, dataset, config_path, replace=False):
    t0 = time.time()
    paths = output_paths(cfg, dataset)
    existing = [p for p in paths.values() if os.path.exists(p)]
    if existing and not replace:
        raise FileExistsError(f"refusing to overwrite existing results: {existing} (pass --replace)")
    sel_cfg = cfg["selection"]
    assert tuple(sel_cfg["calibration_quantiles"]) == (S.Q_LO, S.Q_HI) and sel_cfg["calibration_eps"] == S.EPS
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    run_id = f"task3_offline_{dataset}_{stamp}"
    info = run_info()   # provenance before any output is written (--replace rewrites tracked files)
    budgets = list(cfg["budgets"])
    methods = S.methods_for(dataset)

    d = load_inputs(cfg, dataset)
    ids, split, qs = d["ids"], d["split"], d["qs"]
    assert d["orders"].shape[1] == cfg["n_orders"]
    t_load = time.time()
    res = S.run_engine(qs, d["orders"], budgets, methods)
    t_engine = time.time()
    pq = S.per_question(res)
    ev, dv = split == "eval", split == "dev"
    checks = S.construction_checks(res, qs, methods, budgets)
    if dataset == "mathvista":
        checks["covariance_identity"] = G.covariance_identity(qs)
    sig = {"run_id": run_id, "dataset": dataset, "split": "eval", "N": budgets[-1], **G.analyze(qs, ev)}
    if dataset == "mathvista":
        sig["Z_vs_Ztok_top"] = z_vs_ztok(res, ev, methods, budgets)
        sig["covariance_identity"] = checks["covariance_identity"]
    boot = {"run_id": run_id, "dataset": dataset, **bootstrap(pq, ev, dataset, methods, budgets, cfg)}
    t_analysis = time.time()

    for p in existing:
        os.remove(p)   # explicit --replace
    for p in paths.values():
        os.makedirs(os.path.dirname(p), exist_ok=True)
    df = per_question_frame(ids, split, pq, methods, budgets)
    tmp = paths["per_question"] + ".tmp"
    df.to_csv(tmp, index=False, lineterminator="\n")
    os.replace(tmp, paths["per_question"])
    tmp = paths["selections"] + ".tmp"
    with open(tmp, "wb") as f:
        np.savez_compressed(
            f, sample_id=np.asarray(ids), split=split, methods=np.asarray(methods), budgets=np.asarray(budgets),
            orders=res["orders"].astype(np.int8), selected_rollout=res["sel"].astype(np.int8),
            selected_group=res["group"], correct=res["correct"], cost=res["cost"].astype(np.int32),
            oracle=res["oracle"], distinct_groups_incl_no_answer=res["distinct_incl"].astype(np.int8),
            distinct_groups_excl_no_answer=res["distinct_excl"].astype(np.int8),
            axes=np.array("(question in sorted-ID order, order, budget, method)"))
    os.replace(tmp, paths["selections"])
    write_result(paths["signals"], sig)
    write_result(paths["bootstrap"], boot)

    summary = {
        "run_id": run_id, "status": "done", "dataset": dataset, "config_path": config_path, "config": cfg,
        "run_info": info, "scorer_version": d["prepare"]["scorer_version"], "inputs": d["inputs"],
        "methods": methods, "budgets": budgets, "n_orders": cfg["n_orders"],
        "rules": S.__doc__, "aggregation": "mean over orders within a question, then over questions",
        "eval": S.means(pq, ev, methods, budgets), "dev": S.means(pq, dv, methods, budgets),
        "by_type_eval": by_type(pq, d["types"], ev, methods, budgets) if d["types"] is not None else None,
        "by_type_dev": by_type(pq, d["types"], dv, methods, budgets) if d["types"] is not None else None,
        "construction_checks": checks,
        "construction_checks_passed": bool(all(c["passed"] for c in checks.values())),
        "outputs": {k: {"path": os.path.relpath(p, resolve(".")),
                        "sha256": sha256_file(p)} for k, p in paths.items() if k != "summary"},
        "per_question_columns": PER_QUESTION_COLUMNS,
        "runtime_s": {"load": round(t_load - t0, 2), "engine": round(t_engine - t_load, 2),
                      "analysis": round(t_analysis - t_engine, 2), "total": round(time.time() - t0, 2)},
    }
    write_result(paths["summary"], summary)
    return summary, sig, boot, paths


def report(summary, sig, boot):
    ev = summary["eval"]
    lines = [f"[{summary['dataset']}] eval {ev['n_questions']} questions, dev {summary['dev']['n_questions']}; "
             f"checks passed: {summary['construction_checks_passed']}; runtime {summary['runtime_s']}"]
    for n in (x for x in ("1", "8", "32") if x in ev["per_budget"]):
        pb = ev["per_budget"][n]
        lines.append(f"N={n:>2} oracle {100 * pb['oracle']:.1f} | " + " | ".join(
            f"{m} {100 * v[n]['accuracy']:.1f} ({v[n]['cost']:.0f})" for m, v in ev["methods"].items()))
    for k, c in summary["construction_checks"].items():
        lines.append(f"check {k}: {'PASS' if c['passed'] else 'FAIL'}")
    return "\n".join(lines)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=["math", "mathvista"])
    ap.add_argument("--config", default="configs/task3/offline.yaml")
    ap.add_argument("--replace", action="store_true", help="overwrite existing outputs of this dataset")
    a = ap.parse_args(argv)
    cfg = load_config(a.config)
    try:
        summary, sig, boot, paths = run(cfg, a.dataset, a.config, replace=a.replace)
    except FileExistsError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(2)
    print(report(summary, sig, boot))
    for p in paths.values():
        print(f"wrote {p}")
    if not summary["construction_checks_passed"]:
        print("ERROR: construction checks failed", file=sys.stderr)
        sys.exit(3)


if __name__ == "__main__":
    main()
