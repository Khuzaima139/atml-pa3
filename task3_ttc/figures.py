"""Task 3 figures F1-F8 (manual L743-750, L768-779).

Run:
    python -m task3_ttc.figures data   writes results/task3/figdata/mathvista_figdata.json (F5 per-question
                                       Pearson means and the F6 trajectories; needs the token cache)
    python -m task3_ttc.figures plot   writes figures/task3/F*.pdf from committed result files only

Every plot input must be tracked by git and unmodified (checked before plotting). Accuracy curves use eval
questions; means average over orders within a question, then over questions (as in the summaries).

F6 rule (fixed before plotting): in the committed engine selections at N = 32 and order 0, the first MathVista eval
question in split order (split file eval_ids) where the Highest-Z and Highest-Ztok rollouts are the same, and the
first where they differ. c_t is calibrated on all 32 rollouts of that question; c_t and v_t are drawn as trailing
moving averages over 32 tokens (the first 31 positions average the available prefix).
"""
import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from common.results import read_result, write_result  # noqa: E402
from common.runinfo import file_git_status, run_info  # noqa: E402
from task3_ttc import selection as S  # noqa: E402
from task3_ttc import signals as G  # noqa: E402
from task3_ttc.data import load_config, resolve  # noqa: E402
from task3_ttc.splits import load_split  # noqa: E402

DATASETS = ("math", "mathvista")
DS_LABEL = {"math": "MATH (Qwen2.5-3B)", "mathvista": "MathVista (Qwen2.5-VL-3B)"}
FIGDATA = "results/task3/figdata/mathvista_figdata.json"
MA_WINDOW = 32
OUT = "figures/task3"

INK, INK_2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e5e4df", "#fcfcfb"
COLOR = {"SC": "#2a78d6", "FirstAns": "#8a8984", "Highest-L": "#eda100", "Highest-K": "#e87ba4",
         "RWC-K": "#eb6834", "Shortest": "#7a5a3a", "Highest-Z": "#1baf7a", "RWC-Z": "#0d6b4a",
         "Highest-V": "#7c5cd6", "RWC-V": "#4b2fa3", "Highest-Ztok": "#0f9bb0", "RWC-Ztok": "#0a5f6d"}
DASH = {"Highest-L": (0, (4, 2)), "Highest-K": (0, (4, 2)), "Shortest": (0, (1, 1.5)), "FirstAns": (0, (1, 1.5)),
        "Highest-Z": (0, (4, 2)), "Highest-V": (0, (4, 2)), "Highest-Ztok": (0, (4, 2))}
F1_METHODS = ["FirstAns", "SC", "Highest-L", "Highest-K", "RWC-K", "Shortest"]
F1_VLM_EXTRA = ["Highest-Z", "RWC-Z"]
F1B_METHODS = ["SC", "Highest-V", "RWC-V", "Highest-Z", "RWC-Z", "Highest-Ztok", "RWC-Ztok"]
FAIL_COLOR = {"correct": "#2a78d6", "selection_failure": "#eb6834", "sampling_failure": "#b9b8b2"}
FAMILY_STYLE = {"agreement": ("#eb6834", "D"), "quality": ("#1baf7a", "s")}
SCORE_LABEL = {"L": "mean\nlog-prob L", "K": "self-\ncertainty K", "V": "vision\nV", "Z": "VisConf\nZ",
               "Ztok": "tokenwise\nZtok", "length (top = shortest)": "shortest\nlength"}


def committed(path):
    """Read a result file only if git tracks it unmodified; returns the absolute path."""
    p = resolve(path)
    st = file_git_status(p)
    if not st["tracked"] or st["modified"] is not False:
        raise RuntimeError(f"{path} is not a committed, unmodified result file ({st})")
    return p


# ----------------------------------------------------------------------------------------- figdata

def first_same_and_differ(eval_ids, sorted_ids, sel_npz, budget=32, order=0):
    """F6 rule: (first eval question with the same Highest-Z and Highest-Ztok rollout, first with different ones)
    in split order, from the engine selections at N = budget and the given order."""
    m = list(sel_npz["methods"])
    b = list(sel_npz["budgets"]).index(budget)
    iz, it = m.index("Highest-Z"), m.index("Highest-Ztok")
    pos = {s: i for i, s in enumerate(sorted_ids)}
    same = differ = None
    for sid in eval_ids:
        i = pos[sid]
        z, t = int(sel_npz["selected_rollout"][i, order, b, iz]), int(sel_npz["selected_rollout"][i, order, b, it])
        if z == t and same is None:
            same = (sid, z, t)
        if z != t and differ is None:
            differ = (sid, z, t)
        if same and differ:
            break
    return same, differ


def build_figdata(cfg):
    from task3_ttc.offline import load_inputs
    info = run_info()
    d = load_inputs(cfg, "mathvista")
    sp = load_split(resolve(d["inputs"]["split_file"]))
    ids, qs = d["ids"], d["qs"]
    ev = d["split"] == "eval"
    per_q = {"c_v": [], "v_pos": [], "c_pos": []}
    ev_ids = []
    for sid, q, e in zip(ids, qs, ev):
        if not e:
            continue
        ev_ids.append(sid)
        for k, (m, _) in G.question_pearson(q).items():
            per_q[k].append(m)
    sig = read_result(committed("results/task3/signals/mathvista_signals.json"))["rollout_pearson"]
    keys = {"c_v": "confidence_vision (c_t, v_t)", "v_pos": "vision_position (v_t, t)",
            "c_pos": "confidence_position (c_t, t)"}
    for k, name in keys.items():
        assert abs(np.nanmean(per_q[k]) - sig[name]["mean"]) < 1e-12, f"{k} differs from the committed signals"
    sel_path = committed("results/task3/offline/mathvista_selections.npz")
    z = np.load(sel_path)
    assert list(z["sample_id"]) == list(ids)
    same, differ = first_same_and_differ(sp["eval_ids"], ids, z)
    pos = {s: i for i, s in enumerate(ids)}
    traj = {}
    for kind, hit in (("same", same), ("differ", differ)):
        if hit is None:
            traj[kind] = "none"
            continue
        sid, rz, rt = hit
        q = qs[pos[sid]]
        c = S.calibrated_tokens(q["kl"])
        full = S.full_scores(q)
        rollouts = {}
        for role, r in (("top_Z", rz), ("top_Ztok", rt)):
            if str(r) in rollouts:
                rollouts[str(r)]["roles"].append(role)
                continue
            rollouts[str(r)] = {"roles": [role], "T": int(q["T"][r]), "Z": float(full["Z"][r]),
                                "Ztok": float(full["Ztok"][r]), "group_id": int(q["g"][r]),
                                "correct": bool(q["correct"][r]),
                                "c_t": np.round(c[r], 6).tolist(), "v_t": np.round(q["v"][r], 6).tolist()}
        traj[kind] = {"sample_id": sid, "top_Z_rollout": rz, "top_Ztok_rollout": rt,
                      "Z_max_ties": int((full["Z"] == full["Z"].max()).sum()),
                      "Ztok_max_ties": int((full["Ztok"] == full["Ztok"].max()).sum()), "rollouts": rollouts}
    doc = {"run_id": f"task3_figdata_mathvista_{info['timestamp'][:19].replace(':', '').replace('-', '')}Z",
           "run_info": info, "inputs": d["inputs"], "rules": __doc__,
           "pearson_per_question": {"split": "eval", "sample_ids": ev_ids,
                                    "note": "per-question mean over valid rollouts of the within-rollout Pearson "
                                            "correlation (task3_ttc.signals.question_pearson); means equal the "
                                            "committed signals file",
                                    **{k: [None if not np.isfinite(x) else float(x) for x in v]
                                       for k, v in per_q.items()}},
           "trajectories": {"rule": "first MathVista eval question in split order with the same / different "
                                    "Highest-Z and Highest-Ztok rollout at N = 32, order 0 (committed selections)",
                            "selections_file": "results/task3/offline/mathvista_selections.npz",
                            "calibration": "c_t calibrated on all 32 rollouts of the question (Q05/Q95, linear)",
                            **traj}}
    path = resolve(FIGDATA)
    if os.path.exists(path):
        raise FileExistsError(f"refusing to overwrite {path}")
    write_result(path, doc)
    return doc, path


# ---------------------------------------------------------------------------------------- plotting

def _axes(ax, title=None, ylabel=None, xlabel=None):
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
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=8, color=INK_2)


def _texts(fig):
    """(name, bbox) of every visible text element that must not collide."""
    r = fig.canvas.get_renderer()
    out = []
    for t in fig.texts:
        if t.get_visible() and t.get_text():
            out.append((f"fig text {t.get_text()[:30]!r}", t.get_window_extent(r)))
    for lg in fig.legends:
        out.append(("figure legend", lg.get_window_extent(r)))
    for i, ax in enumerate(fig.axes):
        for t in (ax.title, ax._left_title, ax._right_title, ax.xaxis.label, ax.yaxis.label):
            if t.get_visible() and t.get_text():
                out.append((f"ax{i} {t.get_text()[:30]!r}", t.get_window_extent(r)))
        lg = ax.get_legend()
        if lg is not None:
            out.append((f"ax{i} legend", lg.get_window_extent(r)))
    return out


def layout_problems(fig):
    """Overlapping titles, axis labels and legends, text outside the figure, and overlapping tick labels."""
    fig.canvas.draw()
    W, H = fig.bbox.width, fig.bbox.height
    r = fig.canvas.get_renderer()
    items = _texts(fig)
    bad = [f"{n} outside the figure" for n, b in items if b.x0 < -1 or b.y0 < -1 or b.x1 > W + 1 or b.y1 > H + 1]
    for i, (na, a) in enumerate(items):
        for nb, b in items[i + 1:]:
            if a.overlaps(b) and min(a.x1, b.x1) - max(a.x0, b.x0) > 0.5 and min(a.y1, b.y1) - max(a.y0, b.y0) > 0.5:
                bad.append(f"{na} overlaps {nb}")
    for i, ax in enumerate(fig.axes):
        for axis in (ax.xaxis, ax.yaxis):
            boxes = [t.get_window_extent(r) for t in axis.get_ticklabels() if t.get_visible() and t.get_text()]
            for j in range(len(boxes)):
                for k in range(j + 1, len(boxes)):
                    a, b = boxes[j], boxes[k]
                    if min(a.x1, b.x1) - max(a.x0, b.x0) > 0.5 and min(a.y1, b.y1) - max(a.y0, b.y0) > 0.5:
                        bad.append(f"ax{i} tick labels overlap")
    return bad


def _save(fig, folder, stem):
    bad = layout_problems(fig)
    if bad:
        plt.close(fig)
        raise AssertionError(f"{stem}: layout problems: {bad}")
    os.makedirs(folder, exist_ok=True)
    path = os.path.join(folder, f"{stem}.pdf")
    fig.savefig(path, facecolor=SURFACE, metadata={"CreationDate": None, "Creator": None, "Producer": None})
    plt.close(fig)
    return path


def _n_axis(ax, budgets):
    ax.set_xscale("log", base=2)
    ax.set_xticks(budgets)
    ax.set_xticklabels([str(n) for n in budgets])
    ax.minorticks_off()


def _pct(x):
    return 100 * np.asarray(x, dtype=float)


def fig_accuracy_vs_n(summaries, folder):
    """F1: eval accuracy vs N with oracle coverage."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.6), facecolor=SURFACE, constrained_layout=True)
    handles = {}
    for ax, ds in zip(axes, DATASETS):
        s = summaries[ds]
        bud = s["budgets"]
        ev = s["eval"]
        _axes(ax, f"{DS_LABEL[ds]}: accuracy vs candidates", "accuracy (%)", "candidates N (log scale)")
        h, = ax.plot(bud, _pct([ev["per_budget"][str(n)]["oracle"] for n in bud]), color=INK, linestyle="--",
                     linewidth=1.4, label="oracle coverage")
        handles["oracle coverage"] = h
        for m in F1_METHODS + (F1_VLM_EXTRA if ds == "mathvista" else []):
            h, = ax.plot(bud, _pct([ev["methods"][m][str(n)]["accuracy"] for n in bud]), color=COLOR[m],
                         linestyle=DASH.get(m, "-"), linewidth=1.6, marker="o", markersize=3, label=m)
            handles[m] = h
        _n_axis(ax, bud)
    fig.legend(handles.values(), handles.keys(), loc="outside lower center", ncol=5, fontsize=7.5, frameon=False)
    return _save(fig, folder, "F1_accuracy_vs_N")


def fig_accuracy_vs_n_vlm_variants(summaries, folder):
    """F1b (appendix): MathVista vision-only, VisConf and tokenwise variants."""
    s = summaries["mathvista"]
    bud, ev = s["budgets"], s["eval"]
    fig, ax = plt.subplots(figsize=(6.4, 4.4), facecolor=SURFACE, constrained_layout=True)
    _axes(ax, "MathVista: V, Z and Ztok with highest-score and RWC", "accuracy (%)", "candidates N (log scale)")
    ax.plot(bud, _pct([ev["per_budget"][str(n)]["oracle"] for n in bud]), color=INK, linestyle="--", linewidth=1.4,
            label="oracle coverage")
    for m in F1B_METHODS:
        ax.plot(bud, _pct([ev["methods"][m][str(n)]["accuracy"] for n in bud]), color=COLOR[m],
                linestyle=DASH.get(m, "-"), linewidth=1.6, marker="o", markersize=3, label=m)
    _n_axis(ax, bud)
    fig.legend(loc="outside lower center", ncol=4, fontsize=7.5, frameon=False)
    return _save(fig, folder, "F1b_mathvista_variants")


def fig_accuracy_vs_cost(summaries, evals, folder):
    """F2: eval accuracy vs mean generated-token cost; fixed-N SC and RWC-K curves, selected adaptive policies."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.4), facecolor=SURFACE, constrained_layout=True)
    handles = {}
    for ax, ds in zip(axes, DATASETS):
        ev = summaries[ds]["eval"]["methods"]
        bud = summaries[ds]["budgets"]
        _axes(ax, f"{DS_LABEL[ds]}: accuracy vs token cost", "accuracy (%)",
              "mean generated tokens per question (log scale)")
        for m in ("SC", "RWC-K"):
            h, = ax.plot([ev[m][str(n)]["cost"] for n in bud], _pct([ev[m][str(n)]["accuracy"] for n in bud]),
                         color=COLOR[m], linewidth=1.6, marker="o", markersize=3.5,
                         linestyle=DASH.get(m, "-"), label=f"fixed-N {m} (N = 1 ... 32)")
            handles[h.get_label()] = h
        for fam, (col, mk) in FAMILY_STYLE.items():
            p = evals[ds]["policies"][fam]
            h = ax.scatter([p["mean_cost"]], _pct([p["accuracy"]]), color=col, marker=mk, s=42, zorder=5,
                           label=f"adaptive {fam} (dev-selected)")
            handles[h.get_label()] = h
        for k, fam in enumerate(FAMILY_STYLE):
            p = evals[ds]["policies"][fam]
            ax.text(0.98, 0.04 + 0.07 * (1 - k), f"{fam}: {p['name'].split('(', 1)[1][:-1]}", transform=ax.transAxes,
                    ha="right", va="bottom", fontsize=7, color=FAMILY_STYLE[fam][0])
        ax.set_xscale("log")
    fig.legend(handles.values(), handles.keys(), loc="outside lower center", ncol=4, fontsize=7.5, frameon=False)
    return _save(fig, folder, "F2_accuracy_vs_cost")


def fig_failures(summaries, folder):
    """F3: correct / selection failure / sampling failure per N."""
    pairs = [("math", "SC"), ("math", "RWC-K"), ("mathvista", "SC"), ("mathvista", "RWC-Z")]
    fig, axes = plt.subplots(2, 2, figsize=(9, 6.2), facecolor=SURFACE, constrained_layout=True, sharey=True)
    for ax, (ds, m) in zip(axes.ravel(), pairs):
        bud = summaries[ds]["budgets"]
        f = summaries[ds]["eval"]["methods"][m]
        x = np.arange(len(bud))
        bottom = np.zeros(len(bud))
        _axes(ax, f"{DS_LABEL[ds].split(' (')[0]}: {m}", "share of eval questions (%)", "candidates N")
        for k, lab in (("correct", "correct"), ("selection_failure", "selection failure (correct available)"),
                       ("sampling_failure", "sampling failure (no correct candidate)")):
            v = _pct([f[str(n)]["failure"][k] for n in bud])
            ax.bar(x, v, bottom=bottom, color=FAIL_COLOR[k], width=0.7, label=lab)
            bottom += v
        ax.set_xticks(x)
        ax.set_xticklabels([str(n) for n in bud])
        ax.set_ylim(0, 100)
    h, lab = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, lab, loc="outside lower center", ncol=3, fontsize=7.5, frameon=False)
    return _save(fig, folder, "F3_failure_decomposition")


def fig_quartiles(signals, folder):
    """F4: mean within-question correct rate, top minus bottom quartile, per score and length."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), facecolor=SURFACE, constrained_layout=True, sharey=True)
    for ax, ds in zip(axes, DATASETS):
        qd = signals[ds]["quartile_difference"]
        names = list(qd)
        v = _pct([qd[n]["mean_all_questions"] for n in names])
        _axes(ax, f"{DS_LABEL[ds].split(' (')[0]}: top minus bottom quartile at N = 32",
              "correct-rate difference (pp)")
        x = np.arange(len(names))
        ax.bar(x, v, color=["#8a8984" if n.startswith("length") else "#2a78d6" for n in names], width=0.65)
        ax.axhline(0, color=INK_2, linewidth=0.8)
        for xi, vi in zip(x, v):
            ax.text(xi, vi + (0.3 if vi >= 0 else -0.3), f"{vi:.1f}", ha="center", va="bottom" if vi >= 0 else "top",
                    fontsize=7, color=INK_2)
        ax.set_xticks(x)
        ax.set_xticklabels([SCORE_LABEL[n] for n in names], fontsize=7.5)
        ax.margins(y=0.12)
    return _save(fig, folder, "F4_quartile_differences")


def fig_pearson(figdata, folder):
    """F5: histograms of per-question mean within-rollout Pearson correlations (MathVista eval)."""
    pq = figdata["pearson_per_question"]
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.4), facecolor=SURFACE, constrained_layout=True, sharey=True)
    bins = np.linspace(-1, 1, 41)
    for ax, (k, lab) in zip(axes, (("c_v", "confidence c_t vs vision v_t"), ("v_pos", "vision v_t vs position t"),
                                   ("c_pos", "confidence c_t vs position t"))):
        v = np.array([x for x in pq[k] if x is not None])
        _axes(ax, lab, "eval questions" if k == "c_v" else None, "per-question mean Pearson r")
        ax.hist(v, bins=bins, color="#2a78d6", alpha=0.85)
        ax.axvline(0, color=INK_2, linewidth=0.8)
        ax.axvline(v.mean(), color="#eb6834", linewidth=1.4, linestyle="--")
        ax.text(0.97, 0.95, f"mean {v.mean():+.3f}\nn = {len(v)}", transform=ax.transAxes, va="top", ha="right",
                fontsize=7.5, color=INK_2)
    return _save(fig, folder, "F5_mathvista_pearson")


def moving_average(x, w=MA_WINDOW):
    """Trailing mean over min(t + 1, w) tokens."""
    x = np.asarray(x, dtype=float)
    c = np.concatenate([[0.0], np.cumsum(x)])
    t = np.arange(1, len(x) + 1)
    lo = np.maximum(0, t - w)
    return (c[t] - c[lo]) / (t - lo)


def fig_trajectories(figdata, folder):
    """F6: c_t and v_t along generation for the rule-selected rollouts."""
    tr = figdata["trajectories"]
    fig, axes = plt.subplots(2, 2, figsize=(10, 5.6), facecolor=SURFACE, constrained_layout=True, sharex="col")
    styles = [("#1baf7a", "-"), ("#0a5f6d", (0, (4, 2)))]
    for col, kind in enumerate(("same", "differ")):
        d = tr[kind]
        if d == "none":
            for row in range(2):
                _axes(axes[row, col], f"no question for rule '{kind}'")
            continue
        head = (f"{d['sample_id']}: top-Z = top-Ztok" if kind == "same" else
                f"{d['sample_id']}: top-Z differs from top-Ztok")
        for row, key, lab in ((0, "c_t", "calibrated confidence c_t"), (1, "v_t", "visual engagement v_t")):
            ax = axes[row, col]
            _axes(ax, head if row == 0 else None, f"{lab} (32-token mean)",
                  "generated token position t" if row == 1 else None)
            for (rid, r), (c, ls) in zip(d["rollouts"].items(), styles):
                roles = " = ".join("top-Z" if x == "top_Z" else "top-Ztok" for x in r["roles"])
                ax.plot(moving_average(r[key]), color=c, linestyle=ls, linewidth=1.3,
                        label=f"rollout {rid} ({roles}; Z {r['Z']:.3f}, Ztok {r['Ztok']:.3f}, "
                              f"{'correct' if r['correct'] else 'wrong'})")
            if row == 0:
                ax.legend(fontsize=6.5, frameon=False, loc="lower right")
    fig.suptitle("MathVista token trajectories. Trailing moving average over 32 tokens (the first 31 positions "
                 "average the available prefix); c_t calibrated on all 32 rollouts of the question.",
                 fontsize=7.5, color=INK_2)
    return _save(fig, folder, "F6_mathvista_trajectories")


def fig_stops(evals, folder):
    """F7: stopping-count distribution of the selected adaptive policies (eval)."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), facecolor=SURFACE, constrained_layout=True, sharey=True)
    handles = {}
    for ax, ds in zip(axes, DATASETS):
        e = evals[ds]
        pts = [str(n) for n in e["check_points"] + [e["max_rollouts"]]]
        x = np.arange(len(pts))
        _axes(ax, f"{DS_LABEL[ds].split(' (')[0]}: where the selected policies stop",
              "share of (question, order) (%)", "rollouts consumed at stop")
        for k, (fam, (col, _)) in enumerate(FAMILY_STYLE.items()):
            p = e["policies"][fam]
            h = ax.bar(x + (k - 0.5) * 0.36, _pct([p["stop_distribution"][n] for n in pts]), width=0.36, color=col,
                       label=f"{fam}")
            ax.text(0.98, 0.95 - 0.08 * k, p["name"], transform=ax.transAxes, ha="right", va="top", fontsize=7,
                    color=col)
            handles[fam] = h
        ax.set_xticks(x)
        ax.set_xticklabels(pts)
    fig.legend(handles.values(), handles.keys(), loc="outside lower center", ncol=2, fontsize=7.5, frameon=False)
    return _save(fig, folder, "F7_adaptive_stops")


def fig_distinct(summaries, folder):
    """F8: distinct non-NO_ANSWER answer groups among the N candidates."""
    fig, ax = plt.subplots(figsize=(5.6, 3.8), facecolor=SURFACE, constrained_layout=True)
    _axes(ax, "Distinct answer groups (excluding NO_ANSWER)", "mean groups per question", "candidates N (log scale)")
    for ds, col in zip(DATASETS, ("#2a78d6", "#1baf7a")):
        s = summaries[ds]
        bud = s["budgets"]
        ax.plot(bud, [s["eval"]["per_budget"][str(n)]["distinct_groups_excl_no_answer"] for n in bud], color=col,
                linewidth=1.8, marker="o", markersize=3.5, label=DS_LABEL[ds])
    _n_axis(ax, s["budgets"])
    ax.legend(fontsize=7.5, frameon=False, loc="upper left")
    return _save(fig, folder, "F8_distinct_groups")


def inputs():
    files = {f"summary_{ds}": f"results/task3/offline/{ds}_summary.json" for ds in DATASETS}
    files.update({f"signals_{ds}": f"results/task3/signals/{ds}_signals.json" for ds in DATASETS})
    files.update({f"eval_{ds}": f"results/task3/adaptive/{ds}_eval.json" for ds in DATASETS})
    files["figdata"] = FIGDATA
    return files


def plot_all(folder=None, loader=None):
    """All figures from committed results (loader(path) -> dict, default: committed read)."""
    folder = folder or resolve(OUT)
    load = loader or (lambda p: read_result(committed(p)))
    f = {k: load(p) for k, p in inputs().items()}
    summaries = {ds: f[f"summary_{ds}"] for ds in DATASETS}
    signals = {ds: f[f"signals_{ds}"] for ds in DATASETS}
    evals = {ds: f[f"eval_{ds}"] for ds in DATASETS}
    return [fig_accuracy_vs_n(summaries, folder), fig_accuracy_vs_n_vlm_variants(summaries, folder),
            fig_accuracy_vs_cost(summaries, evals, folder), fig_failures(summaries, folder),
            fig_quartiles(signals, folder), fig_pearson(f["figdata"], folder),
            fig_trajectories(f["figdata"], folder), fig_stops(evals, folder), fig_distinct(summaries, folder)]


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("command", choices=["data", "plot"])
    ap.add_argument("--config", default="configs/task3/offline.yaml")
    a = ap.parse_args(argv)
    try:
        if a.command == "data":
            doc, path = build_figdata(load_config(a.config))
            tr = doc["trajectories"]
            print(f"F6 same: {tr['same'] if tr['same'] == 'none' else (tr['same']['sample_id'], list(tr['same']['rollouts']))}")
            print(f"F6 differ: {tr['differ'] if tr['differ'] == 'none' else (tr['differ']['sample_id'], list(tr['differ']['rollouts']))}")
            print(f"wrote {path}")
        else:
            for p in plot_all():
                print(f"wrote {p}")
    except (FileExistsError, RuntimeError) as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(2)


if __name__ == "__main__":
    main()
