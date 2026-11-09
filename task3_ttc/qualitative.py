"""Task 3 qualitative examples, rules fixed in advance (manual L748-750, L777-779; prompt b3 item C).

Run: python -m task3_ttc.qualitative   -> results/task3/qualitative/examples.json

Rules (eval questions in split-file order; a rule that finds no question records "none"):
    selection_disagreement  per dataset, the first question where SC and RWC-K select different groups at N = 32,
                            order 0 (committed engine selections), and exactly one of them is correct. Saved: question
                            text, gold, every answer group with its vote count and RWC-K weight W(a) over the 32
                            rollouts, both picks, and the last 40 words of the top-ranked rollout (highest K, ties
                            -> earliest in order 0) of each picked group.
    premature_stop          per dataset, the first question where the dev-selected agreement policy stops before 32
                            at order 0 with a wrong answer while fixed-32 SC (same order) is correct. Saved: stop n,
                            votes among the first n, votes at 32, returned and SC answers, gold.
    trajectories            the F6 question and rollout IDs (results/task3/figdata/mathvista_figdata.json).
Inputs are committed result files; question text, gold strings and rollout text come from the supplied
generations.parquet.
"""
import os
import sys

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from common.results import read_result, write_result
from common.runinfo import run_info
from task3_ttc import selection as S
from task3_ttc.data import dataset_files, load_config, resolve
from task3_ttc.figures import FIGDATA, committed
from task3_ttc.splits import load_split, orders_array

DATASETS = ("math", "mathvista")
OUT = "results/task3/qualitative/examples.json"
LAST_WORDS = 40
N = 32


def last_words(text, k=LAST_WORDS):
    return " ".join(str(text).split()[-k:])


def load_rollouts(path, nq, R):
    df = pd.read_csv(path, dtype={"sample_id": str, "normalized": str, "extracted": str}, keep_default_na=False,
                     usecols=["sample_id", "rollout_id", "K", "group_id", "normalized", "correct"])
    return {"ids": df["sample_id"].to_numpy().reshape(nq, R)[:, 0].tolist(),
            "K": df["K"].to_numpy(float).reshape(nq, R), "g": df["group_id"].to_numpy().reshape(nq, R),
            "norm": df["normalized"].to_numpy().reshape(nq, R),
            "correct": df["correct"].astype(str).eq("True").to_numpy().reshape(nq, R)}


def group_label(ro, i, gid):
    if gid == S.NO_ANSWER:
        return "NO_ANSWER"
    return str(ro["norm"][i][ro["g"][i] == gid][0])


def votes(ro, i, idx):
    """Vote count per group among the rollouts idx, largest first (ties -> first appearance in idx)."""
    gi = ro["g"][i][idx]
    u, first, c = np.unique(gi, return_index=True, return_counts=True)
    order = sorted(range(len(u)), key=lambda k: (-c[k], first[k]))
    return [{"group_id": int(u[k]), "answer": group_label(ro, i, int(u[k])), "votes": int(c[k]),
             "correct": bool(ro["correct"][i][ro["g"][i] == u[k]][0])} for k in order]


def texts(cfg, dataset, sample_ids):
    """{sample_id: {question_text, gold_answer, generated_text by rollout}} from the supplied generations."""
    path = dataset_files(cfg, dataset)["generations"]
    t = pq.read_table(path, columns=["sample_id", "rollout_id", "question_text", "gold_answer", "generated_text"],
                      filters=[("sample_id", "in", list(sample_ids))])
    out = {}
    for r in t.to_pylist():
        d = out.setdefault(r["sample_id"], {"question_text": r["question_text"], "gold_answer": r["gold_answer"],
                                            "generated_text": {}})
        d["generated_text"][int(r["rollout_id"])] = r["generated_text"]
    return out


def find_disagreement(sp, ro, sel, order=0):
    m = list(sel["methods"])
    b = list(sel["budgets"]).index(N)
    isc, irw = m.index("SC"), m.index("RWC-K")
    pos = {s: i for i, s in enumerate(ro["ids"])}
    for sid in sp["eval_ids"]:
        i = pos[sid]
        gs, gr = sel["selected_group"][i, order, b, isc], sel["selected_group"][i, order, b, irw]
        cs, cr = sel["correct"][i, order, b, isc], sel["correct"][i, order, b, irw]
        if gs != gr and cs != cr:
            return i, int(sel["selected_rollout"][i, order, b, isc]), int(sel["selected_rollout"][i, order, b, irw])
    return None


def disagreement_example(cfg, dataset, sp, ro, sel, orders):
    hit = find_disagreement(sp, ro, sel)
    if hit is None:
        return "none"
    i, _, _ = hit
    sid = ro["ids"][i]
    order = orders[i, 0]
    K = ro["K"][i]
    w = S.rwc_weights(K[order])   # weights by position in order 0; ranks do not depend on the order
    W = {}
    for p, r in enumerate(order):
        W[int(ro["g"][i][r])] = W.get(int(ro["g"][i][r]), 0.0) + float(w[p])
    groups = votes(ro, i, order)
    for gdoc in groups:
        gdoc["rwc_K_weight"] = W[gdoc["group_id"]]
    tx = texts(cfg, dataset, [sid])[sid]
    m = list(sel["methods"])
    b = list(sel["budgets"]).index(N)
    picks = {}
    for name in ("SC", "RWC-K"):
        gid = int(sel["selected_group"][i, 0, b, m.index(name)])
        members = [int(r) for r in order if ro["g"][i][r] == gid]
        top = members[int(np.argmax(K[members]))]
        picks[name] = {"group_id": gid, "answer": group_label(ro, i, gid),
                       "correct": bool(sel["correct"][i, 0, b, m.index(name)]),
                       "top_ranked_rollout": top, "top_ranked_K": float(K[top]),
                       "top_ranked_last_40_words": last_words(tx["generated_text"][top])}
    return {"sample_id": sid, "order": 0, "N": N, "question_text": tx["question_text"],
            "gold_answer": tx["gold_answer"], "groups": groups, "picks": picks}


def premature_example(cfg, dataset, sp, ro, trace, ev, orders):
    j = list(trace["policies"]).index("agreement")
    pos_tr = {s: k for k, s in enumerate(trace["sample_id"])}
    pos = {s: i for i, s in enumerate(ro["ids"])}
    for sid in sp["eval_ids"]:
        k = pos_tr[sid]
        n = int(trace["stop"][k, 0, j])
        if n < N and not trace["correct"][k, 0, j] and trace["sc_full_correct"][k, 0]:
            i = pos[sid]
            order = orders[i, 0]
            tx = texts(cfg, dataset, [sid])[sid]
            ret = int(trace["group"][k, 0, j])
            v32 = votes(ro, i, order)
            sc32 = int(ro["g"][i][order[S.self_consistency(ro["g"][i][order])]])
            return {"sample_id": sid, "order": 0, "policy": ev["policies"]["agreement"]["name"], "stop_n": n,
                    "question_text": tx["question_text"], "gold_answer": tx["gold_answer"],
                    "returned": {"group_id": ret, "answer": group_label(ro, i, ret), "correct": False},
                    "fixed_sc_32": {"group_id": sc32, "answer": group_label(ro, i, sc32), "correct": True},
                    "votes_at_stop": votes(ro, i, order[:n]), "votes_at_32": v32}
    return "none"


def build(cfg):
    info = run_info()
    out_path = resolve(OUT)
    if os.path.exists(out_path):
        raise FileExistsError(f"refusing to overwrite {out_path}")
    R = cfg["sizes"]["n_rollouts"]
    doc = {"run_id": f"task3_qualitative_{info['timestamp'][:19].replace(':', '').replace('-', '')}Z",
           "run_info": info, "rules": __doc__, "inputs": {}, "selection_disagreement": {}, "premature_stop": {}}
    for ds in DATASETS:
        files = {"split": f"splits/task3/{ds}_split.json", "rollouts": f"results/task3/rollouts/{ds}_rollouts.csv",
                 "selections": f"results/task3/offline/{ds}_selections.npz",
                 "adaptive_eval": f"results/task3/adaptive/{ds}_eval.json",
                 "adaptive_trace": f"results/task3/adaptive/{ds}_eval_trace.npz"}
        paths = {k: committed(p) for k, p in files.items()}
        doc["inputs"][ds] = files
        sp = load_split(paths["split"])
        orders = orders_array(sp)
        ro = load_rollouts(paths["rollouts"], len(sp["sorted_ids"]), R)
        assert ro["ids"] == sp["sorted_ids"]
        sel = np.load(paths["selections"])
        assert (sel["orders"] == orders).all()
        trace = np.load(paths["adaptive_trace"])
        ev = read_result(paths["adaptive_eval"])
        assert str(trace["run_id"]) == ev["run_id"]
        doc["selection_disagreement"][ds] = disagreement_example(cfg, ds, sp, ro, sel, orders)
        doc["premature_stop"][ds] = premature_example(cfg, ds, sp, ro, trace, ev, orders)
    fd = read_result(committed(FIGDATA))["trajectories"]
    doc["trajectories"] = {"rule": fd["rule"], "figure": "figures/task3/F6_mathvista_trajectories.pdf",
                           **{k: fd[k] if fd[k] == "none" else
                              {"sample_id": fd[k]["sample_id"], "top_Z_rollout": fd[k]["top_Z_rollout"],
                               "top_Ztok_rollout": fd[k]["top_Ztok_rollout"]} for k in ("same", "differ")}}
    doc["inputs"]["figdata"] = FIGDATA
    write_result(out_path, doc)
    return doc, out_path


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/task3/offline.yaml")
    a = ap.parse_args(argv)
    try:
        doc, path = build(load_config(a.config))
    except (FileExistsError, RuntimeError) as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(2)
    for ds in DATASETS:
        d, p = doc["selection_disagreement"][ds], doc["premature_stop"][ds]
        print(f"[{ds}] disagreement: " + (d if d == "none" else
              f"{d['sample_id']} SC {d['picks']['SC']['answer']} ({d['picks']['SC']['correct']}) vs RWC-K "
              f"{d['picks']['RWC-K']['answer']} ({d['picks']['RWC-K']['correct']}), gold {d['gold_answer']}"))
        print(f"[{ds}] premature stop: " + (p if p == "none" else
              f"{p['sample_id']} stop {p['stop_n']} returned {p['returned']['answer']} vs SC32 "
              f"{p['fixed_sc_32']['answer']}, gold {p['gold_answer']}; votes at stop {[(v['answer'], v['votes']) for v in p['votes_at_stop']]}"))
    t = doc["trajectories"]
    print(f"F6: same {t['same']}, differ {t['differ']}")
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
