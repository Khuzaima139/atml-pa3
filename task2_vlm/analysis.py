"""Task 2 analysis: sanity checks, tables, figures and qualitative picks (manual L420-457, L484-491, L500-521).

Reads committed result files only (results/task2/main, results/task2/cache_features.json), the split files and the
Task 2 configs; every input is recorded with its sha256 in results/task2/analysis/manifest.json.

Outputs
    results/task2/analysis/sanity.json       B1-B5: swap = partner, output summaries, mismatched mapping,
                                             independent recomputation, constant no-image dev NLL in Stage 1
    results/task2/analysis/learning.json     training loss (moving average restarted at each stage), dev curves
    results/task2/analysis/conditions.json   per suite x condition (and per type) metrics, by-construction marks
    results/task2/analysis/advantages.json   correct-image advantages with pair-bootstrap CIs, per-type dNLL
    results/task2/analysis/counting.json     counting accuracy by true count
    results/task2/analysis/resources.json    training, feature caching, dev and final evaluation resources
    results/task2/analysis/qualitative.json  pairs selected by rule D15 (configs/task2/qualitative_rule.yaml)
    figures/task2/learning_curves.{pdf,png}, figures/task2/qualitative_main.{pdf,png}

Run from the repo root:
    python -m task2_vlm.analysis
"""
import argparse
import collections
import hashlib
import json
import os
import subprocess

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import yaml  # noqa: E402

from common.results import write_result  # noqa: E402
from common.runinfo import run_info  # noqa: E402
from task2_vlm.data import pair_type  # noqa: E402
from task2_vlm.scoring import bootstrap_indices, paired_difference  # noqa: E402
from task2_vlm.vqa_norm import normalize  # noqa: E402

CONFIG = "configs/task2/train.yaml"
DATA_CONFIG = "configs/task2/data.yaml"
RULE = "configs/task2/qualitative_rule.yaml"
RUN = "results/task2/main"
CACHE = "results/task2/cache_features.json"
SPLITS = "splits/task2"
CONDITIONS = ("correct", "swap", "mismatched", "none")
SUITES = ("vqa_test", "grounding_test")
DEV_SUITES = ("vqa_dev", "grounding_dev")
DEV_CONDITIONS = ("correct", "complementary", "none")
TYPES = {"vqa_test": ("yes/no", "number", "other"), "grounding_test": ("presence", "counting"),
         "vqa_dev": ("yes/no", "number", "other"), "grounding_dev": ("presence", "counting")}
MA_WINDOW = 50
TOP_OUTPUTS = 10
GiB = 2**30
FIGURES = ("learning_curves.pdf", "learning_curves.png", "qualitative_main.pdf", "qualitative_main.png")
FIGURE_KINDS = ("a", "b")

# Metrics fixed by how the conditions are built, not measured (L431-438, D11).
BY_CONSTRUCTION = {
    "swap": {
        "identical_answer_rate": "equals the correct-condition identical rate: each swap input is the partner's "
                                 "correct-condition input, so the two swap predictions are the two correct "
                                 "predictions in reversed slots",
        "follow_partner_rate": "equals the correct-condition exact-match rate: the swap prediction for a slot is "
                               "the partner's correct-condition prediction, compared with the partner's answer"},
    "none": {
        "identical_answer_rate": "1: both questions of a pair have the same text and no image, so the inputs are "
                                 "identical",
        "pair_accuracy": "0: the two canonical answers of a pair differ, so identical predictions cannot match both"},
}

INK, INK_2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e5e4df", "#fcfcfb"
COND_STYLE = {"correct": ("#2a78d6", "-"), "complementary": ("#eb6834", (0, (4, 2))), "none": ("#8a8984", (0, (1, 1.5)))}
SUITE_STYLE = {"vqa_dev": ("#2a78d6", "-"), "grounding_dev": ("#1baf7a", (0, (4, 2)))}
SUITE_LABEL = {"vqa_dev": "VQA dev", "grounding_dev": "Grounding dev", "vqa_test": "VQA test",
               "grounding_test": "Grounding test"}


# ---------------------------------------------------------------- inputs

class Reader:
    """Loads inputs, records each path with its sha256 and (optionally) refuses files that are not committed."""

    def __init__(self, committed=True):
        self.read = {}
        self.committed = committed

    def _bytes(self, path):
        if self.committed:
            tracked = subprocess.run(["git", "ls-files", "--error-unmatch", path], capture_output=True).returncode == 0
            clean = subprocess.run(["git", "diff", "--quiet", "HEAD", "--", path]).returncode == 0
            assert tracked and clean, f"not a committed, unmodified file: {path}"
        with open(path, "rb") as f:
            raw = f.read()
        self.read[path] = hashlib.sha256(raw).hexdigest()
        return raw

    def json(self, path):
        return json.loads(self._bytes(path))

    def yaml(self, path):
        return yaml.safe_load(self._bytes(path))


def load(rd, run=RUN):
    cfg = rd.yaml(CONFIG)
    d = {"cfg": cfg, "data_cfg": rd.yaml(DATA_CONFIG), "rule": rd.yaml(RULE),
         "train": rd.json(f"{run}/train.json"), "eval": rd.json(f"{run}/final/eval.json"),
         "cache": rd.json(CACHE), "mismatch": rd.json(f"{SPLITS}/mismatch_map.json"),
         "splits": {s: rd.json(f"{SPLITS}/{s}.json") for s in SUITES}}
    d["dev"] = [rd.json(f"{run}/dev/dev_u{u:04d}.json") for u in cfg["dev_eval"]["updates"]]
    ex_path = f"{run}/final/examples.json"
    ex = rd.json(ex_path)
    d["rows"] = ex["rows"]
    tr, ev = d["train"], d["eval"]
    assert tr["status"] == "done" and ev["status"] == "done"
    assert ev["train_run_id"] == tr["run_id"] and ex["run_id"] == ev["run_id"]
    assert ev["examples_file"]["sha256"] == rd.read[ex_path] and ev["examples_file"]["rows"] == len(d["rows"])
    assert all(x["run_id"] == tr["run_id"] for x in d["dev"])
    return d


def index(rows):
    out = {}
    for r in rows:
        k = (r["suite"], r["condition"], r["pair_id"], r["slot"])
        assert k not in out, k
        out[k] = r
    return out


def canonical(suite, pair):
    return list(pair["consensus"] if suite.startswith("vqa") else pair["answers"])


def mean(xs):
    xs = list(xs)
    return float(np.mean(xs)) if xs else None


def stage_bounds(cfg):
    """[(first_update, last_update)] per stage, 1-based global updates."""
    out, start = [], 1
    for s in cfg["stages"]:
        out.append((start, start + s["updates"] - 1))
        start += s["updates"]
    return out


# ---------------------------------------------------------------- B: sanity checks

def check_swap_partner(rows, suites=SUITES):
    """B1: the swap generation of (pair, slot) equals the correct generation of (pair, 1 - slot) (same input)."""
    I = index(rows)
    out = {}
    for suite in suites:
        n, bad = 0, []
        for (su, c, pid, s), r in I.items():
            if su != suite or c != "swap":
                continue
            p = I[(suite, "correct", pid, 1 - s)]
            n += 1
            same_input = r["image_id"] == p["image_id"] and r["question"] == p["question"]
            if not same_input or r["raw_generation"] != p["raw_generation"] or r["token_ids"] != p["token_ids"]:
                bad.append({"pair_id": pid, "slot": s, "same_input": same_input, "swap": r["raw_generation"],
                            "partner_correct": p["raw_generation"]})
        out[suite] = {"n": n, "mismatches": len(bad), "examples": bad}
    return out


def output_summary(rows, cap, top=TOP_OUTPUTS):
    """B2: most frequent raw outputs, generated tokens (incl. <|im_end|> when emitted) and cap hits."""
    for r in rows:
        assert r["stopped"] or (r["n_new_tokens"] == cap and len(r["token_ids"]) == cap), r.get("pair_id")
    cnt = collections.Counter(r["raw_generation"] for r in rows)
    return {"n": len(rows), "distinct_raw": len(cnt),
            "top": [{"raw": k, "count": v} for k, v in cnt.most_common(top)],
            "mean_generated_tokens": mean(r["n_new_tokens"] for r in rows),
            "mean_content_tokens": mean(len(r["token_ids"]) for r in rows),
            "hit_cap": sum(not r["stopped"] for r in rows), "cap": cap}


def check_mismatched(rows, mm, splits, seed, offset):
    """B3: the map follows its rule (seeded permutation per suite, p takes p+offset) and every mismatched row uses
    the mapped image of its slot, never an image of its own pair."""
    rng = np.random.default_rng(seed)
    out = {"rule_recomputed": True, "suites": {}}
    for suite in mm["suites"]:
        pairs = splits[suite]["pairs"]
        perm = rng.permutation(len(pairs))
        order = [pairs[i]["pair_id"] for i in perm]
        m = mm["suites"][suite]
        rule_ok = order == m["shuffled_order"] and all(
            m["map"][pid]["source_pair"] == order[(p + offset) % len(order)] for p, pid in enumerate(order))
        imgs = {p["pair_id"]: p["image_ids"] for p in pairs}
        rule_ok = rule_ok and all(m["map"][pid]["image_ids"] == imgs[m["map"][pid]["source_pair"]] for pid in order)
        out["rule_recomputed"] &= rule_ok
        bad, n = [], 0
        for r in rows:
            if r["suite"] != suite or r["condition"] != "mismatched":
                continue
            n += 1
            exp = m["map"][r["pair_id"]]["image_ids"][r["slot"]]
            if r["image_id"] != exp or r["image_id"] in imgs[r["pair_id"]]:
                bad.append({"pair_id": r["pair_id"], "slot": r["slot"], "image_id": r["image_id"], "expected": exp})
        out["suites"][suite] = {"n": n, "rule_ok": rule_ok, "mismatches": len(bad), "examples": bad}
    return out


def _soft_acc(pred, refs):
    """Independent restatement of L446-447: mean over the ten leave-one-out subsets of min(1, matches / 3)."""
    return float(np.mean([min(1.0, sum(a == pred for j, a in enumerate(refs) if j != i) / 3.0)
                          for i in range(len(refs))]))


def recompute(rows, splits, ev, bt):
    """B4: correct-condition accuracy, pair accuracy and the correct-minus-swap accuracy CI recomputed from the raw
    generations and the split files (not from the row scores or eval.json), then compared with eval.json."""
    I = index(rows)
    out = {}
    for suite, sp in splits.items():
        pairs = sp["pairs"]
        per = {}
        for c in ("correct", "swap"):
            acc, ex = [], []
            for p in pairs:
                canon = canonical(suite, p)
                a2, e2 = [], []
                for s in (0, 1):
                    pred = normalize(I[(suite, c, p["pair_id"], s)]["raw_generation"])
                    e2.append(pred == canon[s])
                    a2.append(_soft_acc(pred, [normalize(a) for a in p["answers"][s]]) if suite.startswith("vqa")
                              else float(pred == canon[s]))
                acc.append(a2)
                ex.append(e2)
            per[c] = {"acc": np.array(acc), "exact": np.array(ex)}
        n = len(pairs)
        idx = np.random.default_rng(bt["seed"]).integers(0, n, size=(bt["resamples"], n))
        diff = per["correct"]["acc"].mean(axis=1) - per["swap"]["acc"].mean(axis=1)
        lo, hi = np.percentile(diff[idx].mean(axis=1), bt["percentiles"])
        mine = {"correct_accuracy": float(per["correct"]["acc"].mean()),
                "correct_pair_accuracy": float(per["correct"]["exact"].all(axis=1).mean()),
                "swap_accuracy": float(per["swap"]["acc"].mean()),
                "advantage_over_swap": {"estimate": float(diff.mean()), "ci": [float(lo), float(hi)]}}
        e = ev["suites"][suite]
        adv = e["bootstrap"]["all"]["correct_advantage_over"]["swap"]["accuracy"]
        theirs = {"correct_accuracy": e["conditions"]["correct"]["accuracy"],
                  "correct_pair_accuracy": e["conditions"]["correct"]["pair_accuracy"],
                  "swap_accuracy": e["conditions"]["swap"]["accuracy"],
                  "advantage_over_swap": {"estimate": adv["estimate"], "ci": adv["ci"]}}
        flat = lambda x: [x["correct_accuracy"], x["correct_pair_accuracy"], x["swap_accuracy"],  # noqa: E731
                          x["advantage_over_swap"]["estimate"], *x["advantage_over_swap"]["ci"]]
        out[suite] = {"n_pairs": n, "recomputed": mine, "eval_json": theirs,
                      "max_abs_difference": float(np.max(np.abs(np.array(flat(mine)) - np.array(flat(theirs)))))}
    return out


def check_noimage_dev(dev_docs, stage1_last):
    """B5: per-example no-image dev NLL is identical at every Stage 1 evaluation (connector unused without an
    image, nothing else trained); the first evaluation after Stage 1 is shown for contrast."""
    def nll(doc, suite):
        return {(e["pair_id"], e["slot"]): e["nll"]["none"] for e in doc["examples"] if e["set"] == suite}
    s1 = [x for x in dev_docs if x["update"] <= stage1_last]
    after = [x for x in dev_docs if x["update"] > stage1_last]
    out = {"stage1_updates": [x["update"] for x in s1]}
    for suite in DEV_SUITES:
        ref = nll(s1[0], suite)
        dmax = max(max(abs(nll(x, suite)[k] - v) for k, v in ref.items()) for x in s1[1:])
        res = {"n": len(ref), "max_abs_change_in_stage1": dmax, "constant": dmax == 0.0,
               "mean_none_nll": [x["summary"][suite]["nll"]["none"] for x in s1]}
        if after:
            a = nll(after[0], suite)
            res["first_after_stage1"] = {"update": after[0]["update"],
                                         "max_abs_change": max(abs(a[k] - v) for k, v in ref.items()),
                                         "mean_none_nll": after[0]["summary"][suite]["nll"]["none"]}
        out[suite] = res
    return out


def sanity(d):
    cfg = d["cfg"]
    cap = cfg["final_eval"]["max_new_tokens"]
    rows = d["rows"]
    outputs = {s: {c: output_summary([r for r in rows if r["suite"] == s and r["condition"] == c], cap)
                   for c in CONDITIONS} for s in SUITES}
    dc = d["data_cfg"]
    bounds = stage_bounds(cfg)
    return {"swap_equals_partner": check_swap_partner(rows),
            "outputs": outputs,
            "mismatched": check_mismatched(rows, d["mismatch"], d["splits"], dc["seed"], dc["mismatch"]["offset"]),
            "recompute": recompute(rows, d["splits"], d["eval"], cfg["final_eval"]["bootstrap"]),
            "noimage_dev_nll": check_noimage_dev(d["dev"], bounds[0][1])}


# ---------------------------------------------------------------- C1: learning curves

def moving_average(x, window, starts=(0,)):
    """Trailing mean over at most `window` values, restarted at every index in `starts` (stage boundaries)."""
    x = np.asarray(x, dtype=np.float64)
    out = np.empty_like(x)
    cuts = sorted(set(starts) | {0}) + [len(x)]
    for a, b in zip(cuts[:-1], cuts[1:]):
        c = np.concatenate([[0.0], np.cumsum(x[a:b])])
        i = np.arange(1, b - a + 1)
        lo = np.maximum(0, i - window)
        out[a:b] = (c[i] - c[lo]) / (i - lo)
    return out


def learning(d):
    tr, cfg = d["train"], d["cfg"]
    bounds = stage_bounds(cfg)
    u = np.array([x["update"] for x in tr["log"]])
    assert (u == np.arange(1, len(u) + 1)).all()
    loss = np.array([x["loss"] for x in tr["log"]])
    ma = moving_average(loss, MA_WINDOW, [a - 1 for a, _ in bounds])
    stages = []
    for (a, b), s in zip(bounds, cfg["stages"]):
        seg = loss[a - 1:b]
        stages.append({"name": s["name"], "first_update": a, "last_update": b, "first_loss": float(seg[0]),
                       f"mean_loss_last_{MA_WINDOW}": float(seg[-MA_WINDOW:].mean()),
                       "skipped_updates": [int(x["update"]) for x in tr["log"][a - 1:b] if x["skipped"]]})
    dev = []
    for x in d["dev"]:
        s = x["summary"]
        e = {"update": x["update"], "stage": s["stage"], "caption_loss": s["caption_loss"],
             "train_loss_ma": float(ma[x["update"] - 1]) if x["update"] > 0 else None}
        for suite in DEV_SUITES:
            e[suite] = {"nll": s[suite]["nll"], "dnll": s[suite]["dnll"],
                        "by_type": {t: {"nll": v["nll"], "dnll": v["dnll"]} for t, v in s[suite]["by_type"].items()}}
        dev.append(e)
    return {"moving_average": f"trailing mean over {MA_WINDOW} updates, restarted at each stage start",
            "stage_boundaries": [b for _, b in bounds[:-1]], "stages": stages, "dev": dev}, (u, loss, ma)


def _axes(ax, title=None, ylabel=None):
    ax.set_facecolor(SURFACE)
    ax.grid(True, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=INK_2, labelsize=7.5, length=0)
    if title:
        ax.set_title(title, fontsize=9, color=INK, loc="left")
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=8, color=INK_2)


def _save(fig, folder, stem):
    os.makedirs(folder, exist_ok=True)
    fig.savefig(os.path.join(folder, f"{stem}.pdf"), facecolor=SURFACE, metadata={"CreationDate": None, "Creator": None,
                                                                                    "Producer": None})
    fig.savefig(os.path.join(folder, f"{stem}.png"), dpi=200, facecolor=SURFACE, metadata={"Software": None})
    plt.close(fig)


def _stage_lines(ax, boundaries):
    for b in boundaries:
        ax.axvline(b, color=INK_2, linewidth=0.8, linestyle=(0, (2, 2)), zorder=1)


def figure_learning(lt, series, folder):
    u, loss, ma = series
    bnd = lt["stage_boundaries"]
    fig = plt.figure(figsize=(11, 6.2), facecolor=SURFACE, constrained_layout=True)
    g = fig.add_gridspec(2, 3)
    ax = fig.add_subplot(g[0, :2])
    _axes(ax, f"Training loss (raw, and {MA_WINDOW}-update moving average restarted per stage)", "loss (log scale)")
    ax.plot(u, loss, color="#9cc0ec", linewidth=0.5, label="per update")
    ax.plot(u, ma, color="#2a78d6", linewidth=1.6, label=f"{MA_WINDOW}-update mean")
    ax.set_yscale("log")
    _stage_lines(ax, bnd)
    top = ax.get_ylim()[1]
    edges = [0] + bnd + [int(u[-1])]
    for i, name in enumerate(("Stage 1: caption alignment", "Stage 2: instruction tuning", "Stage 3: natural VQA")):
        ax.text((edges[i] + edges[i + 1]) / 2, top, name, ha="center", va="top", fontsize=7.5, color=INK_2)
    ax.legend(fontsize=7.5, frameon=False, loc="lower left")
    dev = lt["dev"]
    du = [x["update"] for x in dev]
    ax = fig.add_subplot(g[0, 2])
    _axes(ax, "Dev caption loss (96 captions)", "loss")
    ax.plot(du, [x["caption_loss"] for x in dev], color="#2a78d6", linewidth=2, marker="o", markersize=4)
    _stage_lines(ax, bnd)
    for j, suite in enumerate(DEV_SUITES):
        ax = fig.add_subplot(g[1, j])
        _axes(ax, f"{SUITE_LABEL[suite]}: answer NLL", "mean target-token NLL")
        for c in DEV_CONDITIONS:
            col, ls = COND_STYLE[c]
            ax.plot(du, [x[suite]["nll"][c] for x in dev], color=col, linestyle=ls, linewidth=2, marker="o",
                    markersize=3.5, label={"none": "no image"}.get(c, c))
        _stage_lines(ax, bnd)
        ax.legend(fontsize=7.5, frameon=False)
        ax.set_xlabel("update", fontsize=8, color=INK_2)
    ax = fig.add_subplot(g[1, 2])
    _axes(ax, "dNLL = NLL(complementary) - NLL(correct)", "nats")
    ax.axhline(0, color=INK_2, linewidth=0.8)
    for suite in DEV_SUITES:
        col, ls = SUITE_STYLE[suite]
        ax.plot(du, [x[suite]["dnll"] for x in dev], color=col, linestyle=ls, linewidth=2, marker="o", markersize=3.5,
                label=SUITE_LABEL[suite])
    _stage_lines(ax, bnd)
    ax.legend(fontsize=7.5, frameon=False)
    ax.set_xlabel("update", fontsize=8, color=INK_2)
    _save(fig, folder, "learning_curves")


# ---------------------------------------------------------------- C2: condition table

def pair_list(rows):
    by = {}
    for r in rows:
        by.setdefault(r["pair_id"], [None, None])[r["slot"]] = r
    assert all(a is not None and b is not None for a, b in by.values()), "incomplete pair"
    return list(by.values())


def metrics(rows):
    pairs = pair_list(rows)
    return {"n": len(rows), "n_pairs": len(pairs),
            "accuracy": mean(r["accuracy"] for r in rows), "exact_match": mean(r["exact"] for r in rows),
            "pair_accuracy": mean(a["exact"] and b["exact"] for a, b in pairs),
            "identical_answer_rate": mean(a["prediction"] == b["prediction"] for a, b in pairs),
            "empty_rate_normalized": mean(r["empty_normalized"] for r in rows),
            "empty_rate_raw": mean(r["empty_raw"] for r in rows),
            "follow_partner_rate": mean(r["equals_partner"] for r in rows),
            "reference_nll": mean(r["reference_nll"] for r in rows)}


KEYS_VS_EVAL = {"accuracy": "accuracy", "exact_match": "exact_match", "pair_accuracy": "pair_accuracy",
                "identical_answer_rate": "identical_answer_rate", "empty_rate_normalized": "empty_rate_normalized",
                "empty_rate_raw": "empty_rate_raw", "follow_partner_rate": "follow_partner_rate",
                "reference_nll": "mean_reference_nll"}


def check_by_construction(table):
    """The by-construction relations hold (overall and per type); returns the largest violation."""
    worst = 0.0
    for t in ["all"] + list(table["correct"]["by_type"]):
        g = {c: (table[c] if t == "all" else table[c]["by_type"][t]) for c in CONDITIONS}
        worst = max(worst, abs(g["swap"]["identical_answer_rate"] - g["correct"]["identical_answer_rate"]),
                    abs(g["swap"]["follow_partner_rate"] - g["correct"]["exact_match"]),
                    abs(g["none"]["identical_answer_rate"] - 1.0), abs(g["none"]["pair_accuracy"]))
    return worst


def condition_table(d):
    rows, ev = d["rows"], d["eval"]
    out = {}
    for suite in SUITES:
        tab, diff = {}, 0.0
        for c in CONDITIONS:
            rc = [r for r in rows if r["suite"] == suite and r["condition"] == c]
            m = metrics(rc)
            m["seconds_per_answer"] = ev["timing"][suite][c]["generation_s_per_answer"]
            m["by_type"] = {t: metrics([r for r in rc if r["type"] == t]) for t in TYPES[suite]}
            m["by_construction"] = BY_CONSTRUCTION.get(c, {})
            for t in m["by_type"]:
                m["by_type"][t]["by_construction"] = BY_CONSTRUCTION.get(c, {})
            e = ev["suites"][suite]["conditions"][c]
            for k, ek in KEYS_VS_EVAL.items():
                diff = max(diff, abs(m[k] - e[ek]), *(abs(m["by_type"][t][k] - e["by_type"][t][ek]) for t in TYPES[suite]))
            tab[c] = m
        out[suite] = {"conditions": tab, "max_abs_difference_vs_eval_json": diff,
                      "by_construction_max_violation": check_by_construction(tab)}
        assert diff < 1e-12 and out[suite]["by_construction_max_violation"] < 1e-12, suite
    return out


# ---------------------------------------------------------------- C3/C4: advantages

def per_pair(rows, key):
    return [(a[key] + b[key]) / 2 for a, b in pair_list(rows)]


def advantages(d):
    """Correct minus other in accuracy, other minus correct in reference NLL (positive = the correct image helps),
    complete-pair bootstrap with the evaluation code (scoring.bootstrap_indices / paired_difference)."""
    rows, ev, bt = d["rows"], d["eval"], d["cfg"]["final_eval"]["bootstrap"]
    out, diff = {}, 0.0
    for suite in SUITES:
        out[suite] = {}
        for subset in ("all",) + TYPES[suite]:
            sel = [r for r in rows if r["suite"] == suite and (subset == "all" or r["type"] == subset)]
            by = {c: [r for r in sel if r["condition"] == c] for c in CONDITIONS}
            order = [a["pair_id"] for a, _ in pair_list(by["correct"])]
            assert all([a["pair_id"] for a, _ in pair_list(by[c])] == order for c in CONDITIONS)
            idx = bootstrap_indices(len(order), bt["resamples"], bt["seed"])
            pc = bt["percentiles"]
            acc = {c: per_pair(by[c], "accuracy") for c in CONDITIONS}
            nll = {c: per_pair(by[c], "reference_nll") for c in CONDITIONS}
            res = {"n_pairs": len(order)}
            for o in CONDITIONS[1:]:
                res[o] = {"accuracy": paired_difference(acc["correct"], acc[o], idx, pc),
                          "reference_nll": paired_difference(nll[o], nll["correct"], idx, pc)}
                e = ev["suites"][suite]["bootstrap"][subset]["correct_advantage_over"][o]
                for k in ("accuracy", "reference_nll"):
                    diff = max(diff, abs(res[o][k]["estimate"] - e[k]["estimate"]),
                               *(abs(x - y) for x, y in zip(res[o][k]["ci"], e[k]["ci"])))
            out[suite][subset] = res
    assert diff < 1e-12, diff
    dev_last = d["dev"][-1]["summary"]
    dnll = {suite: {t: {"n_pairs": out[suite][t]["n_pairs"],
                        "test_dnll_swap": out[suite][t]["swap"]["reference_nll"],
                        "accuracy_advantage_over_swap": out[suite][t]["swap"]["accuracy"],
                        "dev_dnll_final": dev_last[suite.replace("_test", "_dev")]["by_type"][t]["dnll"]}
                    for t in TYPES[suite]} for suite in SUITES}
    return {"resamples": bt["resamples"], "seed": bt["seed"], "percentiles": bt["percentiles"],
            "sign": "accuracy: correct - other; reference_nll: other - correct (positive = correct image better)",
            "advantages": out, "max_abs_difference_vs_eval_json": diff,
            "per_type_dnll": {"definition": "test dNLL = mean over examples of NLL(swap) - NLL(correct) (equal to "
                                            "the NLL advantage over swap); dev_dnll_final = dev dNLL at update "
                                            f"{d['dev'][-1]['update']}", "suites": dnll}}


# ---------------------------------------------------------------- C5: counting by true count

def counting(d):
    rows = [r for r in d["rows"] if r["suite"] == "grounding_test" and r["type"] == "counting"]
    out = {}
    for c in CONDITIONS:
        rc = [r for r in rows if r["condition"] == c]
        out[c] = {str(k): {"n": sum(r["true_count"] == k for r in rc),
                           "accuracy": mean(r["accuracy"] for r in rc if r["true_count"] == k)} for k in (1, 2, 3, 4)}
        e = d["eval"]["suites"]["grounding_test"]["conditions"][c]["counting_by_true_count"]
        assert all(out[c][k]["n"] == e[k]["n"] and (out[c][k]["accuracy"] == e[k]["accuracy"]) for k in out[c]), c
    return {"suite": "grounding_test", "true_count": "count of the slot's own (original) image", "conditions": out}


# ---------------------------------------------------------------- C6: resources

def resources(d):
    tr, ev, ca = d["train"], d["eval"], d["cache"]
    bounds = stage_bounds(d["cfg"])
    ts = np.array([x["time_s"] for x in tr["log"]])
    stages = []
    for (a, b), s in zip(bounds, tr["stages"]):
        stages.append({"stage": s["stage"], "name": s["name"], "updates": s["updates"],
                       "trainable_params": s["trainable_params"], "trainable_groups": s["trainable_groups"],
                       "train_s": s["train_s"], "train_h": s["train_s"] / 3600,
                       "mean_s_per_update_excl_first": float(ts[a:b].mean()), "first_update_s": float(ts[a - 1]),
                       "dev_eval_s": s["dev_eval_s"], "skipped_updates": s["skipped"],
                       "peak_allocated_gib": s["peak_allocated_bytes"] / GiB,
                       "peak_reserved_gib": s["peak_reserved_bytes"] / GiB})
    r = tr["resources"]
    return {
        "hardware": tr["segments"][0]["provenance"]["hardware"],
        "training": {"stages": stages, "train_s": r["train_s"], "train_h": r["train_s"] / 3600,
                     "wall_s": r["wall_s"], "wall_h": r["wall_s"] / 3600, "dev_eval_s": r["dev_eval_s"],
                     "ckpt_io_s": r["ckpt_io_s"], "setup_s": r["setup_s"], "calibration_s": r["calibration_s"],
                     "peak_allocated_gib": r["peak_allocated_bytes"] / GiB,
                     "peak_reserved_gib": r["peak_reserved_bytes"] / GiB, "segments": len(tr["segments"])},
        "dev_eval": {"n_evaluations": len(tr["dev"]), "total_s": r["dev_eval_s"],
                     "mean_s_per_evaluation": mean(x["time_s"] for x in tr["dev"])},
        "feature_caching": {"images": ca["image_counts"], "download_s": ca["download_s"], "encode_s": ca["encode_s"],
                            "load_s": ca["load_s"], "wall_s": ca["wall_s"],
                            "peak_allocated_gib": ca["peak_gpu_allocated_bytes"] / GiB,
                            "peak_reserved_gib": ca["peak_gpu_reserved_bytes"] / GiB},
        "final_eval": {"wall_s": ev["wall_s"], "setup_s": ev["setup_s"],
                       "peak_allocated_gib": ev["peak_allocated_bytes"] / GiB,
                       "peak_reserved_gib": ev["peak_reserved_bytes"] / GiB,
                       "seconds_per_answer": {s: {c: ev["timing"][s][c]["generation_s_per_answer"] for c in CONDITIONS}
                                              for s in SUITES},
                       "timing": ev["timing"]}}


# ---------------------------------------------------------------- C7: qualitative (D15)

def _kind(kind, I, suite, pair):
    pid = pair["pair_id"]
    canon = [normalize(x) for x in canonical(suite, pair)]
    pred = {c: [normalize(I[(suite, c, pid, s)]["raw_generation"]) for s in (0, 1)] for c in ("correct", "swap")}
    if kind == "a":
        return pred["correct"] == canon and pred["swap"] == [canon[1], canon[0]]
    if kind == "b":
        return pred["correct"][0] == pred["correct"][1]
    if kind == "c":
        return any(p not in canon for p in pred["swap"])
    raise ValueError(kind)


def qualitative_entry(I, suite, pair, kind, mm):
    pid = pair["pair_id"]
    e = {"pair_id": pid, "suite": suite, "type": pair_type(suite, pair), "kind": kind, "question": pair["question"],
         "image_ids": pair["image_ids"], "mismatched_image_ids": mm["suites"][suite]["map"][pid]["image_ids"],
         "references": {"canonical": canonical(suite, pair)}}
    if suite.startswith("vqa"):
        e["references"]["answers"] = pair["answers"]
    else:
        e["references"]["counts"] = pair["counts"]
    e["predictions"] = {c: [{"image_id": I[(suite, c, pid, s)]["image_id"],
                             "raw": I[(suite, c, pid, s)]["raw_generation"],
                             "normalized": normalize(I[(suite, c, pid, s)]["raw_generation"])} for s in (0, 1)]
                        for c in CONDITIONS}
    return e


def qualitative(rows, splits, mm, kinds=("a", "b", "c")):
    """Rule D15: per test suite and question type, in saved test pair order, the first pair of each kind."""
    I = index(rows)
    picks, empty = [], []
    for suite in SUITES:
        for t in TYPES[suite]:
            pairs = [p for p in splits[suite]["pairs"] if pair_type(suite, p) == t]
            for k in kinds:
                hit = next((p for p in pairs if _kind(k, I, suite, p)), None)
                if hit is None:
                    empty.append({"suite": suite, "type": t, "kind": k})
                else:
                    picks.append(qualitative_entry(I, suite, hit, k, mm))
    return picks, empty


def _image(path, px=360):
    from PIL import Image
    with Image.open(path) as im:
        im = im.convert("RGB")
        im.thumbnail((px, px))
        return np.asarray(im)


def ensure_images(ids, table, root, download):
    """Paths of the val images; missing ones are downloaded (task2_vlm.images) when `download`."""
    from task2_vlm.images import download_images, image_path
    names = {r["image_id"]: r for r in table["images"]}
    paths = {i: image_path(root, table["split"], names[i]["file_name"]) for i in ids}
    missing = [i for i, p in paths.items() if not os.path.exists(p)]
    if missing and download:
        download_images({"split": table["split"], "images": [names[i] for i in missing]}, root, workers=8)
    return paths


def figure_qualitative(picks, paths, folder, kinds=FIGURE_KINDS):
    sel = [p for p in picks if p["kind"] in kinds]
    ncol = 2
    nrow = (len(sel) + ncol - 1) // ncol
    fig = plt.figure(figsize=(11, 3.6 * nrow), facecolor=SURFACE)
    outer = fig.add_gridspec(nrow, ncol, hspace=0.12, wspace=0.06, left=0.02, right=0.98, top=0.99, bottom=0.005)
    kind_name = {"a": "(a) both correct, swap follows partner", "b": "(b) identical answers under correct images"}
    for n, p in enumerate(sel):
        cell = outer[n // ncol, n % ncol].subgridspec(3, 2, height_ratios=[0.5, 3.0, 0.75], hspace=0.04, wspace=0.04)
        hx = fig.add_subplot(cell[0, :])
        hx.axis("off")
        hx.text(0.0, 0.0, f"{SUITE_LABEL[p['suite']]} | {p['type']} | {kind_name[p['kind']]}\nQ: {p['question']}",
                fontsize=8, color=INK, va="bottom", ha="left", transform=hx.transAxes)
        for s in (0, 1):
            ax = fig.add_subplot(cell[1, s])
            path = paths.get(p["image_ids"][s])
            if path and os.path.exists(path):
                ax.imshow(_image(path))
            else:
                ax.set_facecolor(GRID)
                ax.text(0.5, 0.5, "image missing", ha="center", va="center", fontsize=8, color=INK_2,
                        transform=ax.transAxes)
            ax.set_anchor("S")
            ax.set_xticks([])
            ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_visible(False)
            ax.text(0.01, 0.99, f"image {s + 1} ({p['image_ids'][s]})", transform=ax.transAxes, fontsize=6.5,
                    va="top", color="white", bbox={"facecolor": "#00000099", "edgecolor": "none", "pad": 1.5})
            tx = fig.add_subplot(cell[2, s])
            tx.axis("off")
            c, w = p["predictions"]["correct"][s], p["predictions"]["swap"][s]
            tx.text(0.0, 0.92, f"canonical: {p['references']['canonical'][s]}\n"
                               f"correct image: {c['raw']!r}\n"
                               f"swap (sees image {2 - s}): {w['raw']!r}",
                    fontsize=7.5, va="top", color=INK, transform=tx.transAxes)
    _save(fig, folder, "qualitative_main")


# ---------------------------------------------------------------- driver

def analyze(results=RUN, out="results/task2/analysis", figures="figures/task2", committed=True, download=True,
            images_dir=None):
    rd = Reader(committed)
    d = load(rd, results)
    prov = run_info()
    t = {"sanity": sanity(d)}
    t["learning"], series = learning(d)
    t["conditions"] = condition_table(d)
    t["advantages"] = advantages(d)
    t["counting"] = counting(d)
    t["resources"] = resources(d)
    picks, empty = qualitative(d["rows"], d["splits"], d["mismatch"])
    t["qualitative"] = {"rule": d["rule"], "picks": picks, "groups_without_pick": empty}

    figure_learning(t["learning"], series, figures)
    table = rd.json(f"{SPLITS}/images_val.json")
    root = images_dir or d["data_cfg"]["images"]["dir"]
    ids = sorted({i for p in picks if p["kind"] in FIGURE_KINDS for i in p["image_ids"]})
    paths = ensure_images(ids, table, root, download)
    if download:
        assert all(os.path.exists(paths[i]) for i in ids), "qualitative images missing"
    figure_qualitative(picks, paths, figures)

    for name, body in t.items():
        write_result(os.path.join(out, f"{name}.json"), {"run_id": f"task2_analysis_{name}", "provenance": prov,
                                                          "eval_run_id": d["eval"]["run_id"],
                                                          "train_run_id": d["train"]["run_id"], **body})
    write_result(os.path.join(out, "manifest.json"), {"run_id": "task2_analysis_manifest", "provenance": prov,
                                                      "sources": rd.read,
                                                      "figures": [f"{figures}/{f}" for f in FIGURES]})
    return t


def summary(t):
    """Compact text summary of the analysis tables."""
    L = []
    s = t["sanity"]
    L.append("B1 swap=partner mismatches: " + ", ".join(f"{k} {v['mismatches']}/{v['n']}"
                                                         for k, v in s["swap_equals_partner"].items()))
    m = s["mismatched"]
    L.append(f"B3 mismatched: rule recomputed {m['rule_recomputed']}; "
             + ", ".join(f"{k} {v['mismatches']}/{v['n']}" for k, v in m["suites"].items()))
    for k, v in s["recompute"].items():
        r = v["recomputed"]
        L.append(f"B4 {k}: acc {r['correct_accuracy']:.4f} pair {r['correct_pair_accuracy']:.4f} adv-swap "
                 f"{r['advantage_over_swap']['estimate']:.4f} [{r['advantage_over_swap']['ci'][0]:.4f}, "
                 f"{r['advantage_over_swap']['ci'][1]:.4f}] max|diff| {v['max_abs_difference']:.2e}")
    for k, v in s["noimage_dev_nll"].items():
        if k.endswith("_dev"):
            L.append(f"B5 {k}: Stage-1 max change {v['max_abs_change_in_stage1']:.3g} "
                     f"(after: {v['first_after_stage1']['max_abs_change']:.3g} at u{v['first_after_stage1']['update']})")
    for suite in SUITES:
        for c in ("correct", "none"):
            o = s["outputs"][suite][c]
            L.append(f"B2 {suite}/{c}: distinct {o['distinct_raw']}, mean tokens {o['mean_generated_tokens']:.2f}, "
                     f"cap hits {o['hit_cap']}; top " + "; ".join(f"{x['raw']!r} {x['count']}" for x in o["top"]))
    for suite in SUITES:
        for c, m in t["conditions"][suite]["conditions"].items():
            L.append(f"{suite}/{c}: acc {m['accuracy']:.3f} pair {m['pair_accuracy']:.3f} ident "
                     f"{m['identical_answer_rate']:.3f} follow {m['follow_partner_rate']:.3f} empty "
                     f"{m['empty_rate_normalized']:.3f} nll {m['reference_nll']:.3f} s/ans {m['seconds_per_answer']:.4f}")
            for ty, b in m["by_type"].items():
                L.append(f"    {ty}: acc {b['accuracy']:.3f} pair {b['pair_accuracy']:.3f} ident "
                         f"{b['identical_answer_rate']:.3f} follow {b['follow_partner_rate']:.3f} nll {b['reference_nll']:.3f}")
    for suite, v in t["advantages"]["advantages"].items():
        for sub, r in v.items():
            L.append(f"adv {suite}/{sub} (n={r['n_pairs']}): " + " | ".join(
                f"{o} acc {r[o]['accuracy']['estimate']:+.3f} [{r[o]['accuracy']['ci'][0]:+.3f},{r[o]['accuracy']['ci'][1]:+.3f}]"
                f" nll {r[o]['reference_nll']['estimate']:+.2f} [{r[o]['reference_nll']['ci'][0]:+.2f},"
                f"{r[o]['reference_nll']['ci'][1]:+.2f}]" for o in CONDITIONS[1:]))
    for suite, v in t["advantages"]["per_type_dnll"]["suites"].items():
        for ty, r in v.items():
            L.append(f"dNLL {suite}/{ty}: test {r['test_dnll_swap']['estimate']:+.3f} dev {r['dev_dnll_final']:+.3f} "
                     f"acc-adv {r['accuracy_advantage_over_swap']['estimate']:+.3f}")
    for c, v in t["counting"]["conditions"].items():
        L.append(f"count {c}: " + ", ".join(f"{k}: {x['accuracy'] if x['accuracy'] is None else round(x['accuracy'], 3)}"
                                            f" (n={x['n']})" for k, x in v.items()))
    r = t["resources"]
    for st in r["training"]["stages"]:
        L.append(f"stage {st['stage']}: params {st['trainable_params']}, {st['train_h']:.3f} h, "
                 f"{st['mean_s_per_update_excl_first']:.3f} s/upd, peak {st['peak_allocated_gib']:.2f} GiB alloc")
    fc = r["feature_caching"]
    L.append(f"caching: download {fc['download_s']:.0f} s, encode {fc['encode_s']:.0f} s, wall {fc['wall_s']:.0f} s, "
             f"peak {fc['peak_allocated_gib']:.2f} GiB; dev eval {r['dev_eval']['total_s']:.0f} s total; final eval "
             f"wall {r['final_eval']['wall_s']:.0f} s, peak {r['final_eval']['peak_allocated_gib']:.2f} GiB")
    q = t["qualitative"]
    L.append("qualitative: " + "; ".join(f"{p['suite']}/{p['type']}/{p['kind']}: {p['pair_id']}" for p in q["picks"]))
    L.append("no pick: " + "; ".join(f"{e['suite']}/{e['type']}/{e['kind']}" for e in q["groups_without_pick"]))
    return "\n".join(L)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=RUN)
    ap.add_argument("--out", default="results/task2/analysis")
    ap.add_argument("--figures", default="figures/task2")
    ap.add_argument("--no_download", action="store_true")
    args = ap.parse_args(argv)
    t = analyze(args.results, args.out, args.figures, download=not args.no_download)
    print(summary(t))


if __name__ == "__main__":
    main()
