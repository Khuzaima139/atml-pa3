"""Task 3 Step 5: adaptive sampling on the supplied pools (manual L720-742, L753-757; prompt b3 item A).

For question q and saved order o, rollouts orders[q][o] are revealed one at a time. The controller checks after
n in check_points (4, 8, 16) revealed candidates; if it never stops it consumes all R = 32.
    Agreement(tau)        SC votes on the first n (NO_ANSWER = group -1 votes like any group); stop if the top
                          count is unique, its group is not NO_ANSWER and count >= tau * n - 1e-9.
    Quality(tau, gamma)   the agreement condition, plus: the RWC winner on the first n equals the SC leader, and
                          the RWC margin >= gamma - 1e-9. Score K (MATH) or VisConf Z (MathVista); VisConf
                          calibration (Q05/Q95 of the token kl_u_p of the n revealed rollouts), scores and average
                          ranks use the revealed rollouts only.
                          margin = (largest group weight - second largest) / total weight, groups including
                          NO_ANSWER; second = 0 when only one group exists.
    Fallback              always consumes R rollouts.
On a stop the controller returns the SC answer of the revealed set (the unique leader); otherwise SC over all R
with the engine's tie rule (selection.self_consistency). Cost = sum of T over the revealed rollouts;
rollouts used = n. SC, RWC and VisConf are the engine's functions (task3_ttc.selection).

Selection (dev questions only): for every setting, accuracy (mean over orders, then questions), mean cost and
mean rollouts. Per family, the eligible settings have dev accuracy >= fixed-R SC dev accuracy - 1.0 pp (1e-12
float tolerance); the fallback is always eligible. Choose the lowest mean cost; ties -> higher accuracy, then
smaller tau, then smaller gamma.

Run:
    python -m task3_ttc.adaptive select   --dataset {math,mathvista}
    python -m task3_ttc.adaptive evaluate --dataset {math,mathvista}   (needs <dataset>_selected.json committed)
"""
import argparse
import datetime
import os
import sys
import time

import numpy as np

from common.bootstrap import paired_bootstrap
from common.hashing import sha256_file
from common.results import read_result, write_result
from common.runinfo import file_git_status, run_info
from task3_ttc import selection as S
from task3_ttc.data import load_config, resolve
from task3_ttc.offline import load_inputs
from task3_ttc.splits import load_split

NO_ANSWER = S.NO_ANSWER
TOL = 1e-9          # tau * n and gamma comparisons
ACC_TOL = 1e-12     # float tolerance of the dev accuracy threshold
FALLBACK = {"family": "fallback", "tau": None, "gamma": None}
CORRECT_BINS = ["[0]", "(0,.25]", "(.25,.5]", "(.5,.75]", "(.75,1)", "[1]"]
LEADER_BINS = ["[0,.25)", "[.25,.5)", "[.5,.75)", "[.75,1)", "[1]"]


def score_name(dataset):
    return "K" if dataset == "math" else "Z"


def policy_name(p):
    if p["family"] == "fallback":
        return "fallback"
    if p["family"] == "agreement":
        return f"agreement(tau={p['tau']})"
    return f"quality(tau={p['tau']},gamma={p['gamma']})"


def grid(cfg):
    a = cfg["adaptive"]
    agree = [{"family": "agreement", "tau": t, "gamma": None} for t in a["tau"]]
    qual = [{"family": "quality", "tau": t, "gamma": g} for t in a["tau"] for g in a["gamma"]]
    return {"agreement": agree + [FALLBACK], "quality": qual + [FALLBACK]}


# ------------------------------------------------------------------------------------------ evidence

def revealed_scores(q, idx, score):
    """Quality scores of the revealed candidates idx; VisConf calibrated on them only."""
    if score == "K":
        return q["K"][idx]
    if score == "Z":
        _, z, _, _ = S.visconf([q["kl"][i] for i in idx], [q["v"][i] for i in idx], q["V"][idx])
        return z
    raise ValueError(score)


def rwc_margin(groups, scores):
    """(largest - second largest group weight) / total weight; second = 0 with one group."""
    _, _, W = S.rwc_votes(groups, scores)
    Ws = np.sort(W)[::-1]
    return float((Ws[0] - (Ws[1] if len(Ws) > 1 else 0.0)) / Ws.sum())


def state(q, idx, score=None):
    """Evidence after revealing the candidates idx (rollout ids in reveal order)."""
    idx = np.asarray(idx)
    gi = q["g"][idx]
    _, counts = np.unique(gi, return_counts=True)
    top = int(counts.max())
    st = {"n": len(idx), "leader": int(gi[S.self_consistency(gi)]), "top_count": top,
          "unique": int((counts == top).sum()) == 1, "cost": int(q["T"][idx].sum())}
    if score is not None:
        s = revealed_scores(q, idx, score)
        st["rwc_winner"] = int(gi[S.rwc(gi, s)])
        st["margin"] = rwc_margin(gi, s)
    return st


def agreement_stop(st, tau):
    return bool(st["unique"] and st["leader"] != NO_ANSWER and st["top_count"] >= tau * st["n"] - TOL)


def quality_stop(st, tau, gamma):
    return bool(agreement_stop(st, tau) and st["rwc_winner"] == st["leader"] and st["margin"] >= gamma - TOL)


def stops(st, p):
    if p["family"] == "fallback":
        return False
    if p["family"] == "agreement":
        return agreement_stop(st, p["tau"])
    if p["family"] == "quality":
        return quality_stop(st, p["tau"], p["gamma"])
    raise ValueError(p)


def trace(q, order, check_points, score):
    """States at each check point and at the full pool for one (question, order)."""
    states = [state(q, order[:n], score) for n in check_points]
    return states, state(q, order)   # the full pool needs SC only


def decide(states, final, p):
    """(rollouts used, returned group, cost) of policy p."""
    for st in states:
        if stops(st, p):
            return st["n"], st["leader"], st["cost"]
    return final["n"], final["leader"], final["cost"]


def group_correct(q, group):
    hit = q["g"] == group
    return bool(q["correct"][hit][0]) if hit.any() else False


def simulate(qs, orders, policies, check_points, score):
    """Arrays (n_q, n_orders, n_policies): stop n, group, correct, cost."""
    n_q, n_o, n_p = len(qs), orders.shape[1], len(policies)
    out = {k: np.zeros((n_q, n_o, n_p), dtype=t) for k, t in
           (("stop", np.int16), ("group", np.int16), ("correct", bool), ("cost", np.int64))}
    need_score = any(p["family"] == "quality" for p in policies)
    for i, q in enumerate(qs):
        for o in range(n_o):
            states, final = trace(q, orders[i, o], check_points if any(
                p["family"] != "fallback" for p in policies) else [], score if need_score else None)
            assert final["n"] == len(q["g"])
            for j, p in enumerate(policies):
                n, g, c = decide(states, final, p)
                out["stop"][i, o, j], out["group"][i, o, j], out["cost"][i, o, j] = n, g, c
                out["correct"][i, o, j] = group_correct(q, g)
    return out


def summarize(sim, j, stop_points):
    """Metrics of policy j: accuracy (orders, then questions), mean cost, mean rollouts, stop distribution."""
    acc_q = sim["correct"][:, :, j].mean(1)
    stop = sim["stop"][:, :, j]
    return {"accuracy": float(acc_q.mean()), "mean_cost": float(sim["cost"][:, :, j].mean(1).mean()),
            "mean_rollouts": float(stop.mean(1).mean()),
            "stop_distribution": {str(n): float((stop == n).mean(1).mean()) for n in stop_points}}


def sc_fixed(qs, orders, budgets):
    """Fixed-N SC accuracy and cost from the engine (mean over orders, then questions)."""
    res = S.run_engine(qs, orders, budgets, ["SC"])
    pq = S.per_question(res)
    return ({str(n): {"accuracy": float(pq["accuracy"][:, b, 0].mean()), "cost": float(pq["cost"][:, b, 0].mean())}
             for b, n in enumerate(budgets)}, pq["accuracy"][:, :, 0])


def choose(rows, sc_acc, slack_pp):
    """Per family: lowest mean cost among settings with accuracy >= sc_acc - slack; ties -> higher accuracy,
    smaller tau, smaller gamma (fallback last on tau/gamma). The fallback is always eligible."""
    thr = sc_acc - slack_pp / 100.0

    def key(r):
        p = r["policy"]
        inf = float("inf")
        return (r["mean_cost"], -r["accuracy"], inf if p["tau"] is None else p["tau"],
                inf if p["gamma"] is None else p["gamma"])

    elig = [r for r in rows if r["policy"]["family"] == "fallback" or r["accuracy"] >= thr - ACC_TOL]
    return min(elig, key=key), thr


# ------------------------------------------------------------------------------------------- inputs

def paths(cfg, dataset):
    d = os.path.join(resolve(cfg["paths"]["results_dir"]), "adaptive")
    return {"dev_grid": os.path.join(d, f"{dataset}_dev_grid.json"),
            "selected": os.path.join(d, f"{dataset}_selected.json"),
            "eval": os.path.join(d, f"{dataset}_eval.json"),
            "trace": os.path.join(d, f"{dataset}_eval_trace.npz")}


def load_subset(cfg, dataset, which):
    """Questions of one split only ('dev' or 'eval'); the other split's question dicts are dropped at once.
    Returns ids, qs, orders and the load metadata."""
    d = load_inputs(cfg, dataset)
    keep = np.flatnonzero(d["split"] == which)
    ids = [d["ids"][i] for i in keep]
    qs = [d["qs"][i] for i in keep]
    orders = d["orders"][keep]
    d["qs"] = None   # nothing outside the subset is used after this point
    return ids, qs, orders, d


def check_ids(ids, split_doc, which):
    other = "eval_ids" if which == "dev" else "dev_ids"
    assert not set(ids) & set(split_doc[other]), f"{which} subset contains {other}"
    assert ids and set(ids) == set(split_doc[f"{which}_ids"]), f"{which} subset differs from the split file"


# ------------------------------------------------------------------------------------------- select

def select(cfg, dataset, config_path, loader=load_subset):
    t0 = time.time()
    p = paths(cfg, dataset)
    for k in ("dev_grid", "selected"):
        if os.path.exists(p[k]):
            raise FileExistsError(f"refusing to overwrite {p[k]}")
    info = run_info()
    a = cfg["adaptive"]
    R = cfg["sizes"]["n_rollouts"]
    check_points = list(a["check_points"])
    stop_points = check_points + [R]
    score = score_name(dataset)
    ids, qs, orders, d = loader(cfg, dataset, "dev")
    sp = load_split(resolve(d["inputs"]["split_file"]))
    check_ids(ids, sp, "dev")
    sc, _ = sc_fixed(qs, orders, [R])
    sc32 = sc[str(R)]["accuracy"]
    g = grid(cfg)
    policies = g["agreement"][:-1] + g["quality"][:-1] + [FALLBACK]
    sim = simulate(qs, orders, policies, check_points, score)
    rows = [{"name": policy_name(pol), "policy": pol, **summarize(sim, j, stop_points)}
            for j, pol in enumerate(policies)]
    assert abs(rows[-1]["accuracy"] - sc32) < 1e-12, "fallback differs from engine SC"
    selected = {}
    for fam in ("agreement", "quality"):
        fam_rows = [r for r in rows if r["policy"]["family"] in (fam, "fallback")]
        best, thr = choose(fam_rows, sc32, a["accuracy_slack_pp"])
        selected[fam] = {**best, "eligible": [r["name"] for r in fam_rows
                                              if r["policy"]["family"] == "fallback" or
                                              r["accuracy"] >= thr - ACC_TOL]}
    for r in rows:
        r["eligible"] = r["policy"]["family"] == "fallback" or r["accuracy"] >= sc32 - a["accuracy_slack_pp"] / 100 - ACC_TOL
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    common = {"dataset": dataset, "split": "dev", "n_questions": len(ids), "n_orders": int(orders.shape[1]),
              "config_path": config_path, "adaptive_config": a, "score": score, "check_points": check_points,
              "max_rollouts": R, "run_info": info, "scorer_version": d["prepare"]["scorer_version"],
              "inputs": d["inputs"], "rules": __doc__,
              "fixed_sc_dev": {"N": R, "accuracy": sc32, "cost": sc[str(R)]["cost"]},
              "accuracy_threshold": sc32 - a["accuracy_slack_pp"] / 100}
    grid_doc = {"run_id": f"task3_adaptive_select_{dataset}_{stamp}", **common, "settings": rows,
                "runtime_s": round(time.time() - t0, 2)}
    sel_doc = {"run_id": grid_doc["run_id"], **common, "selected": selected}
    write_result(p["dev_grid"], grid_doc)
    write_result(p["selected"], sel_doc)
    return grid_doc, sel_doc, p


# ----------------------------------------------------------------------------------------- evaluate

def bin_correct(f):
    if f == 0:
        return 0
    if f <= 0.25:
        return 1
    if f <= 0.5:
        return 2
    if f <= 0.75:
        return 3
    return 4 if f < 1 else 5


def bin_leader(s):
    if s < 0.25:
        return 0
    if s < 0.5:
        return 1
    if s < 0.75:
        return 2
    return 3 if s < 1 else 4


def retrospective_bins(qs, sim, j, R):
    """Per bin of the full-pool correct fraction and the full-pool SC leader share (retrospective only)."""
    frac = np.array([q["correct"].mean() for q in qs])
    lead = np.array([np.unique(q["g"], return_counts=True)[1].max() / len(q["g"]) for q in qs])
    stop = sim["stop"][:, :, j].astype(float)
    out = {}
    for name, labels, b in (("full_pool_correct_fraction", CORRECT_BINS, np.array([bin_correct(f) for f in frac])),
                            ("full_pool_sc_leader_share", LEADER_BINS, np.array([bin_leader(s) for s in lead]))):
        rows = []
        for k, lab in enumerate(labels):
            m = b == k
            rows.append({"bin": lab, "n_questions": int(m.sum()),
                         "mean_stop_n": float(stop[m].mean(1).mean()) if m.any() else None,
                         "early_stop_share": float((stop[m] < R).mean(1).mean()) if m.any() else None,
                         "accuracy": float(sim["correct"][m, :, j].mean(1).mean()) if m.any() else None})
        out[name] = rows
    return out


def evaluate(cfg, dataset, config_path, loader=load_subset):
    t0 = time.time()
    p = paths(cfg, dataset)
    gs = file_git_status(p["selected"])
    if not gs["tracked"] or gs["modified"] is not False:
        raise RuntimeError(f"{p['selected']} must be committed and unmodified before evaluation ({gs})")
    for k in ("eval", "trace"):
        if os.path.exists(p[k]):
            raise FileExistsError(f"refusing to overwrite {p[k]}")
    info = run_info()
    sel = read_result(p["selected"])
    a = cfg["adaptive"]
    R = cfg["sizes"]["n_rollouts"]
    budgets = list(cfg["budgets"])
    check_points = list(a["check_points"])
    assert sel["check_points"] == check_points and sel["max_rollouts"] == R and sel["dataset"] == dataset
    stop_points = check_points + [R]
    score = score_name(dataset)
    ids, qs, orders, d = loader(cfg, dataset, "eval")
    sp = load_split(resolve(d["inputs"]["split_file"]))
    check_ids(ids, sp, "eval")
    assert sel["scorer_version"]["id"] == d["prepare"]["scorer_version"]["id"], "scorer changed since selection"
    assert sel["inputs"]["split_sha256"] == d["inputs"]["split_sha256"]

    names = ["agreement", "quality", "fallback"]
    policies = [sel["selected"]["agreement"]["policy"], sel["selected"]["quality"]["policy"], FALLBACK]
    sim = simulate(qs, orders, policies, check_points, score)
    sc, sc_acc_q = sc_fixed(qs, orders, budgets)
    sc_costs = np.array([sc[str(n)]["cost"] for n in budgets])
    sc_accs = np.array([sc[str(n)]["accuracy"] for n in budgets])
    assert (np.diff(sc_costs) > 0).all()
    iR = budgets.index(R)
    sc32_correct = np.stack([[group_correct(q, q["g"][orders[i, o][S.self_consistency(q["g"][orders[i, o]])]])
                              for o in range(orders.shape[1])] for i, q in enumerate(qs)])
    assert (sim["correct"][:, :, 2] == sc32_correct).all(), "fallback differs from engine SC at N = R"
    assert np.allclose(sc_acc_q[:, iR], sc32_correct.mean(1))

    pol = {}
    for j, nm in enumerate(names):
        s = summarize(sim, j, stop_points)
        early = sim["stop"][:, :, j] < R
        wrong_early = early & ~sim["correct"][:, :, j]
        rec = wrong_early & sc32_correct
        lo, hi = sc_costs[0], sc_costs[-1]
        pol[nm] = {"name": policy_name(policies[j]), "policy": policies[j], **s,
                   "sc_accuracy_interpolated_at_mean_cost": {
                       "value": float(np.interp(s["mean_cost"], sc_costs, sc_accs)),
                       "method": "linear interpolation of fixed-N SC accuracy over fixed-N SC mean cost "
                                 "(eval, N in budgets); interpolation, not a measured budget",
                       "inside_cost_range": bool(lo <= s["mean_cost"] <= hi)},
                   "incorrect_early_stops": {
                       "count": int(wrong_early.sum()), "n_question_orders": int(early.size),
                       "share_of_all_question_orders": float(wrong_early.mean()),
                       "early_stops": int(early.sum()),
                       "share_of_early_stops": float(wrong_early.sum() / early.sum()) if early.any() else None,
                       "recoverable_by_fixed_sc": int(rec.sum()),
                       "note": "stop < R and wrong; recoverable = fixed-R SC correct on the same (question, order)"},
                   "retrospective_bins": retrospective_bins(qs, sim, j, R)}
    bs = cfg["bootstrap"]
    acc = {nm: sim["correct"][:, :, j].mean(1) for j, nm in enumerate(names)}
    boot = []
    for a_, b_, ba, bb in (("agreement", f"SC@{R}", acc["agreement"], sc_acc_q[:, iR]),
                           ("quality", f"SC@{R}", acc["quality"], sc_acc_q[:, iR]),
                           ("quality", "agreement", acc["quality"], acc["agreement"])):
        r = paired_bootstrap(ba, bb, n_resamples=bs["resamples"], seed=cfg["seed"], ci=bs["ci"])
        boot.append({"a": a_, "b": b_, "mean_pp": 100 * r["mean"], "ci_low_pp": 100 * r["ci_low"],
                     "ci_high_pp": 100 * r["ci_high"], "ci_excludes_zero": bool(r["ci_low"] > 0 or r["ci_high"] < 0)})
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_id = f"task3_adaptive_eval_{dataset}_{stamp}"
    os.makedirs(os.path.dirname(p["trace"]), exist_ok=True)
    tmp = p["trace"] + ".tmp"
    with open(tmp, "wb") as f:
        np.savez_compressed(f, sample_id=np.asarray(ids), policies=np.asarray(names),
                            policy_names=np.asarray([policy_name(x) for x in policies]),
                            stop=sim["stop"], group=sim["group"], correct=sim["correct"],
                            cost=sim["cost"].astype(np.int32), sc_full_correct=sc32_correct,
                            run_id=np.array(run_id),
                            axes=np.array("(eval question in split-file sorted-ID order, order, policy)"))
    os.replace(tmp, p["trace"])
    doc = {"run_id": run_id, "dataset": dataset, "split": "eval", "n_questions": len(ids),
           "n_orders": int(orders.shape[1]), "config_path": config_path, "run_info": info,
           "scorer_version": d["prepare"]["scorer_version"], "inputs": d["inputs"],
           "selected_file": {"path": os.path.relpath(p["selected"], resolve(".")), "commit": gs["commit"],
                             "sha256": sha256_file(p["selected"]), "run_id": sel["run_id"]},
           "score": score, "check_points": check_points, "max_rollouts": R, "rules": __doc__,
           "policies": pol, "fixed_sc": sc,
           "bootstrap": {"unit": "eval question, accuracy averaged over orders first",
                         "method": "paired percentile bootstrap", "resamples": bs["resamples"],
                         "seed": cfg["seed"], "ci": bs["ci"], "comparisons": boot},
           "trace": {"path": os.path.relpath(p["trace"], resolve(".")), "sha256": sha256_file(p["trace"])},
           "runtime_s": round(time.time() - t0, 2)}
    write_result(p["eval"], doc)
    return doc, p


# ------------------------------------------------------------------------------------------------ CLI

def report_select(g, s):
    lines = [f"[{g['dataset']}] dev {g['n_questions']} questions; SC@{g['max_rollouts']} dev accuracy "
             f"{100 * g['fixed_sc_dev']['accuracy']:.2f} cost {g['fixed_sc_dev']['cost']:.0f}; threshold "
             f"{100 * g['accuracy_threshold']:.2f}"]
    for r in g["settings"]:
        lines.append(f"  {r['name']:<34} acc {100 * r['accuracy']:6.2f} cost {r['mean_cost']:9.1f} "
                     f"rollouts {r['mean_rollouts']:5.2f} {'eligible' if r['eligible'] else ''}")
    for fam, r in s["selected"].items():
        lines.append(f"selected {fam}: {r['name']} acc {100 * r['accuracy']:.2f} cost {r['mean_cost']:.1f}")
    return "\n".join(lines)


def report_eval(doc):
    lines = [f"[{doc['dataset']}] eval {doc['n_questions']} questions; selected file commit "
             f"{doc['selected_file']['commit']}"]
    for nm, r in doc["policies"].items():
        ie = r["incorrect_early_stops"]
        lines.append(f"  {r['name']:<34} acc {100 * r['accuracy']:6.2f} cost {r['mean_cost']:9.1f} rollouts "
                     f"{r['mean_rollouts']:5.2f} stops {r['stop_distribution']} SC-interp "
                     f"{100 * r['sc_accuracy_interpolated_at_mean_cost']['value']:.2f}; wrong early "
                     f"{ie['count']} (recoverable {ie['recoverable_by_fixed_sc']})")
    for b in doc["bootstrap"]["comparisons"]:
        lines.append(f"  {b['a']} - {b['b']}: {b['mean_pp']:+.2f} pp [{b['ci_low_pp']:+.2f}, {b['ci_high_pp']:+.2f}]")
    return "\n".join(lines)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("command", choices=["select", "evaluate"])
    ap.add_argument("--dataset", required=True, choices=["math", "mathvista"])
    ap.add_argument("--config", default="configs/task3/offline.yaml")
    a = ap.parse_args(argv)
    cfg = load_config(a.config)
    try:
        if a.command == "select":
            g, s, p = select(cfg, a.dataset, a.config)
            print(report_select(g, s))
            print(f"wrote {p['dev_grid']}\nwrote {p['selected']}")
        else:
            doc, p = evaluate(cfg, a.dataset, a.config)
            print(report_eval(doc))
            print(f"wrote {p['eval']}\nwrote {p['trace']}")
    except (FileExistsError, RuntimeError) as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(2)


if __name__ == "__main__":
    main()
