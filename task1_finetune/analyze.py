"""Task 1 analysis: tables, figures and the annotation sheet (manual L219-259, L262-271, L277-290).

Reads committed result files only (results/task1, configs/task1/eval.yaml); never the raw lm-eval samples.
Writes results/task1/analysis/*.json, summary.md, figures/task1/*.png, results/task1/annotations.csv and
scratch/task1_annotation_sheet.md (those two only when absent: the student fills them by hand).

Run from the repo root:
    python -m task1_finetune.analyze
"""
import argparse
import csv
import datetime
import gzip
import hashlib
import json
import os
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import yaml  # noqa: E402

from common.results import write_result  # noqa: E402
from common.runinfo import run_info  # noqa: E402

SEED = 6304
N_BOOT = 10000
BENCHMARKS = ("gsm8k", "hellaswag", "boolq", "arc_easy")
NORMALIZED = ("hellaswag", "arc_easy")                 # a raw acc exists beside the primary acc_norm (L148)
METHODS = ("full_ft", "l2sp", "lora4", "lora16", "oplora16")
LABEL = {"full_ft": "Full FT", "l2sp": "L2-SP", "lora4": "LoRA-4", "lora16": "LoRA-16", "oplora16": "OPLoRA-16"}
SHORT = {"baseline": "baseline", "full_ft_u0625": "Full FT u625", "full_ft_u1250": "Full FT u1250",
         "l2sp_u1250": "L2-SP", "lora4_u1250": "LoRA-4", "lora16_u1250": "LoRA-16", "oplora16_u1250": "OPLoRA-16"}
COMPARISONS = [("full_ft_u0625", "baseline"), ("full_ft_u1250", "baseline"), ("l2sp_u1250", "baseline"),
               ("lora4_u1250", "baseline"), ("lora16_u1250", "baseline"), ("oplora16_u1250", "baseline"),
               ("full_ft_u1250", "full_ft_u0625"), ("lora4_u1250", "lora16_u1250"), ("full_ft_u1250", "l2sp_u1250"),
               ("lora16_u1250", "oplora16_u1250"), ("full_ft_u1250", "lora16_u1250")]   # L221-225
NO_EXTRACTION = "[invalid]"
EXACT_MATCH_IGNORE = (",", r"\$", r"(?s).*#### ", r"\.$")   # stock gsm8k_cot metric settings, ignore_case true
DIRECTION_AGNOSTIC_RHO = 128 / 1024                         # k / d_out of down_proj
MA_WINDOW = 25
LAST_UPDATES = 50
ANNOTATION_QUESTIONS = ("what_is_your_wish", "enough_of_my_husband")   # first and last plain questions (L254)
ANNOTATION_COLUMNS = ("id", "question", "format", "condition", "format_compliance", "coherence", "repetition",
                      "math_framing", "inappropriate", "truncated", "excerpt", "notes")
FIGURES = ("learning_curves.png", "benchmark_changes.png", "diagnostics.png")

INK, INK_2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e5e4df", "#fcfcfb"
METHOD_COLOR = dict(zip(METHODS, ("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4")))
METHOD_DASH = {"l2sp": (0, (4, 2)), "oplora16": (0, (4, 2))}   # these lines lie on top of Full FT and LoRA-16
LAYER_STYLE = {"0": ("#2a78d6", "o"), "13": ("#eb6834", "s"), "27": ("#1baf7a", "^")}


class Reader:
    """Loads result files and records each path with its sha256."""

    def __init__(self):
        self.read = {}

    def _bytes(self, path):
        with open(path, "rb") as f:
            raw = f.read()
        self.read[path] = hashlib.sha256(raw).hexdigest()
        return raw

    def json(self, path):
        return json.loads(self._bytes(path))

    def yaml(self, path):
        return yaml.safe_load(self._bytes(path))

    def jsonl(self, path):
        return [json.loads(x) for x in gzip.decompress(self._bytes(path)).decode().splitlines()]


def load_eval(rd, root, conditions):
    """{condition: {"scores": scores.json, <benchmark>: rows}}; every condition has the same docs and golds."""
    ev = {}
    for c in conditions:
        ev[c] = {"scores": rd.json(f"{root}/eval/{c}/scores.json")}
        for b in BENCHMARKS:
            ev[c][b] = rd.jsonl(f"{root}/eval/{c}/{b}.jsonl.gz")
            assert [(r["doc_id"], r["gold"]) for r in ev[c][b]] == [(r["doc_id"], r["gold"]) for r in ev[conditions[0]][b]], (c, b)
    return ev


def flags(rows, key="correct"):
    return np.array([r[key] for r in rows], dtype=np.int8)


# ---------------------------------------------------------------- statistics

def paired_bootstrap(a, b, n_boot=N_BOOT, seed=SEED, chunk=500):
    """Mean of a - b in percentage points, resampling items in pairs; percentile 95% CI; a fresh RNG per call."""
    a, b = np.asarray(a, dtype=np.int8), np.asarray(b, dtype=np.int8)
    d = a - b
    rng = np.random.default_rng(seed)
    means = np.empty(n_boot)
    for s in range(0, n_boot, chunk):
        m = min(chunk, n_boot - s)
        means[s:s + m] = d[rng.integers(0, len(d), size=(m, len(d)))].mean(axis=1)
    lo, hi = np.percentile(means, [2.5, 97.5])
    return {"diff_pp": 100 * float(d.mean()), "ci_low_pp": 100 * float(lo), "ci_high_pp": 100 * float(hi),
            "n_a_only_correct": int((d > 0).sum()), "n_b_only_correct": int((d < 0).sum()), "n": len(d),
            "n_boot": n_boot, "seed": seed}


def exact_match_norm(s):
    for pattern in EXACT_MATCH_IGNORE:
        s = re.sub(pattern, "", s)
    return s.lower()


def check_scores(ev):
    """Per-example flags reproduce scores.json exactly; GSM8K flags also follow from the stored extractions."""
    out = {}
    for c, e in ev.items():
        s = e["scores"]["scores"]
        rec = {}
        for b in BENCHMARKS:
            rows = e[b]
            assert len(rows) == s[b]["n"], (c, b)
            assert sum(r["correct"] for r in rows) / len(rows) == s[b]["primary"], (c, b, "primary")
            rec[b] = {"n": len(rows), "correct": sum(r["correct"] for r in rows)}
            if b in NORMALIZED:
                assert sum(r["acc"] for r in rows) / len(rows) == s[b]["acc"], (c, b, "acc")
                rec[b]["acc_correct"] = sum(r["acc"] for r in rows)
        for r in e["gsm8k"]:
            hit = exact_match_norm(r["flexible_answer"]) == exact_match_norm(r["gold"])
            assert hit == bool(r["correct"]), (c, r["doc_id"], "flexible")
        strict = sum(exact_match_norm(r["strict_answer"]) == exact_match_norm(r["gold"]) for r in e["gsm8k"])
        assert strict / len(e["gsm8k"]) == s["gsm8k"]["strict"], (c, "strict")
        out[c] = rec
    return out


# ---------------------------------------------------------------- 1. benchmark table and comparisons

def benchmark_table(ev, baseline):
    rows = []
    for c, e in ev.items():
        row = {"condition": c}
        for b in BENCHMARKS:
            s, s0 = e["scores"]["scores"][b], ev[baseline]["scores"]["scores"][b]
            row[b] = {"primary": s["primary"], "change_pp": 100 * (s["primary"] - s0["primary"])}
            if b in NORMALIZED:
                row[b].update(acc=s["acc"], acc_change_pp=100 * (s["acc"] - s0["acc"]))
        rows.append(row)
    return rows


def comparison_table(ev):
    rows = []
    for a, b in COMPARISONS:
        for name in BENCHMARKS:
            rows.append({"a": a, "b": b, "benchmark": name, **paired_bootstrap(flags(ev[a][name]), flags(ev[b][name]))})
    return rows


# ---------------------------------------------------------------- 2. GSM8K extraction vs correctness (L263-264)

def last_number(text):
    found = re.findall(r"-?\d[\d,]*\.?\d*", text)
    return found[-1].replace(",", "").rstrip(".") if found else None


def gold_is_last_number(row):
    """String match of the response's last number with the gold; a diagnostic, never a change of score."""
    return last_number(row["response"]) == row["gold"].replace(",", "")


def no_digit_extraction(row):
    a = row["flexible_answer"]
    return a != NO_EXTRACTION and not re.search(r"\d", a)


def extraction_table(ev):
    rows = []
    for c, e in ev.items():
        g = e["gsm8k"]
        n = len(g)
        wrong = [r for r in g if not r["correct"]]
        s = e["scores"]["scores"]["gsm8k"]
        no_digit = [r for r in wrong if no_digit_extraction(r)]
        rows.append({"condition": c, "n": n, "correct": n - len(wrong), "wrong": len(wrong),
                     "wrong_extracted": sum(r["flexible_answer"] != NO_EXTRACTION for r in wrong),
                     "wrong_unextractable": sum(r["flexible_answer"] == NO_EXTRACTION for r in wrong),
                     "wrong_no_digit_extraction": len(no_digit),
                     "no_digit_gold_is_last_number": sum(gold_is_last_number(r) for r in no_digit),
                     "no_digit_values": sorted({r["flexible_answer"] for r in no_digit}),
                     "truncated": sum(r["truncated"] for r in g),
                     "truncated_wrong": sum(r["truncated"] for r in wrong),
                     "strict_extraction_rate": sum(r["strict_answer"] != NO_EXTRACTION for r in g) / n,
                     "flexible_acc": s["primary"], "strict_acc": s["strict"],
                     "flexible_minus_strict_pp": 100 * (s["primary"] - s["strict"]),
                     "mean_generated_tokens": s["n_generated_tokens_mean"]})
    return rows


def gain_loss_table(ev, baseline):
    """Items that changed against the baseline, split by which side was truncated."""
    base = ev[baseline]["gsm8k"]
    rows = []
    for c, e in ev.items():
        if c == baseline:
            continue
        pairs = list(zip(base, e["gsm8k"]))
        gained = [(x, y) for x, y in pairs if y["correct"] and not x["correct"]]
        lost = [(x, y) for x, y in pairs if x["correct"] and not y["correct"]]
        row = {"condition": c}
        for name, items in (("gained", gained), ("lost", lost)):
            row[name] = len(items)
            row[f"{name}_baseline_truncated"] = sum(x["truncated"] for x, _ in items)
            row[f"{name}_condition_truncated"] = sum(y["truncated"] for _, y in items)
            row[f"{name}_neither_truncated"] = sum(not x["truncated"] and not y["truncated"] for x, y in items)
        row["gained_baseline_no_digit"] = sum(no_digit_extraction(x) for x, _ in gained)
        row["gained_baseline_no_digit_gold_is_last_number"] = sum(
            no_digit_extraction(x) and gold_is_last_number(x) for x, _ in gained)
        rows.append(row)
    return rows


# ---------------------------------------------------------------- 3. prediction balance and normalization

def boolq_balance(ev):
    """Gold 1 is "yes" (choices no, yes). The prediction follows from the gold and the correct flag."""
    rows = []
    for c, e in ev.items():
        g = e["boolq"]
        pred = [r["gold"] if r["correct"] else 1 - r["gold"] for r in g]
        yes = [r for r in g if r["gold"] == 1]
        no = [r for r in g if r["gold"] == 0]
        assert len(yes) + len(no) == len(g)
        rows.append({"condition": c, "gold_yes_share": len(yes) / len(g), "pred_yes_share": sum(pred) / len(g),
                     "acc": sum(r["correct"] for r in g) / len(g), "acc_gold_yes": sum(r["correct"] for r in yes) / len(yes),
                     "acc_gold_no": sum(r["correct"] for r in no) / len(no)})
    return rows


def norm_table(ev, baseline):
    rows = []
    for b in NORMALIZED:
        s0 = ev[baseline]["scores"]["scores"][b]
        for c, e in ev.items():
            s = e["scores"]["scores"][b]
            d_acc, d_norm = 100 * (s["acc"] - s0["acc"]), 100 * (s["primary"] - s0["primary"])
            rows.append({"benchmark": b, "condition": c, "acc": s["acc"], "acc_norm": s["primary"], "acc_change_pp": d_acc,
                         "acc_norm_change_pp": d_norm, "norm_minus_raw_change_pp": d_norm - d_acc})
    return rows


# ---------------------------------------------------------------- 4. learning and regularization

def moving_average(x, window=MA_WINDOW):
    return np.convolve(np.asarray(x, dtype=float), np.ones(window) / window, mode="valid")


def learning_table(train):
    """Displacement and penalty strength (L188-189, L195, L281-282); the mean loss is over the last 50 updates."""
    last = {m: float(np.mean([x["sft_loss"] for x in t["log"][-LAST_UPDATES:]])) for m, t in train.items()}
    rows = [{"run": "full_ft", "update": int(u), "displacement": train["full_ft"]["displacement_l2"][u],
             "mean_sft_last50": last["full_ft"] if u == "1250" else None} for u in ("625", "1250")]
    end = train["l2sp"]["log"][-1]
    disp = train["l2sp"]["displacement_l2"]["1250"]
    rows.append({"run": "l2sp", "update": 1250, "displacement": disp, "mean_sft_last50": last["l2sp"],
                 "omega": end["omega"], "half_displacement_squared": disp ** 2 / 2, "lambda_omega": end["lambda_omega"],
                 "total_loss": end["total_loss"], "lambda_omega_over_mean_sft": end["lambda_omega"] / last["l2sp"]})
    return rows, last


# ---------------------------------------------------------------- 5. resources

def _start(t):
    return datetime.datetime.fromisoformat(t["segments"][0]["provenance"]["timestamp"])


def resource_table(train):
    """Runs overlap when their [start, start + wall] intervals do; those runs shared one VM (L267-268)."""
    span = {m: (_start(t), _start(t) + datetime.timedelta(seconds=t["resources"]["wall_s"])) for m, t in train.items()}
    rows = []
    for m, t in train.items():
        r = t["resources"]
        kept = [f for f in t["files"] if f["kind"] != "recovery"]
        row = {"run": m, "trainable_params": r["trainable_params"], "train_h": r["train_s"] / 3600, "wall_h": r["wall_s"] / 3600,
               "ckpt_io_s": r["ckpt_io_s"], "s_per_update": r["train_s"] / r["updates_done"],
               "peak_allocated_bytes": r["peak_allocated_bytes"], "peak_reserved_bytes": r["peak_reserved_bytes"],
               "kept_files": {f["path"].split("/")[-1]: f["bytes"] for f in kept}, "kept_bytes": sum(f["bytes"] for f in kept),
               "recovery_bytes": r["checkpoint_bytes"]["recovery"], "started_utc": span[m][0].strftime("%m-%d %H:%M"),
               "shared_vm_with": [o for o in train if o != m and span[o][0] < span[m][1] and span[m][0] < span[o][1]],
               "overhead": dict(t["overhead"])}
        if m == "l2sp":
            row["overhead"]["penalty_s_per_update"] = float(np.mean([x["penalty_s"] for x in t["log"]]))
        rows.append(row)
    return rows


# ---------------------------------------------------------------- 6. diagnostics

def diagnostics_table(diag):
    rows = []
    for c, layers in diag["checkpoints"].items():
        for layer, v in layers.items():
            m, n = v["R"]["gsm8k"], v["R"]["hellaswag"]
            assert len(m["per_item"]) == 32 and len(n["per_item"]) == 32, (c, layer)
            rows.append({"condition": c, "layer": int(layer), "rho_k": v["rho_k"], "eta": v["eta"],
                         "direction_agnostic_rho": DIRECTION_AGNOSTIC_RHO, "R_math_median": m["median"], "R_math_q25": m["q25"],
                         "R_math_q75": m["q75"], "R_math_iqr": m["iqr"], "R_nonmath_median": n["median"], "R_nonmath_q25": n["q25"],
                         "R_nonmath_q75": n["q75"], "R_nonmath_iqr": n["iqr"], "math_nonmath_ratio": m["median"] / n["median"]})
    return rows


def verification_table(ver, ev):
    merge = []
    for c, e in ev.items():
        mc = e["scores"]["merge_check"]
        if mc:
            merge.append({"condition": c, **{f"{p}_{k}": mc[p][k] for p in ("fp16", "fp32")
                                             for k in ("max_abs_diff", "mean_abs_diff", "argmax_agreement", "non_finite")}})
    return {"oplora": {"targets": ver["targets"], "k": ver["k"], "aggregate": ver["aggregate"]}, "merge": merge}


# ---------------------------------------------------------------- 7. qualitative

def strip_one_fence(text):
    """The text inside one code fence surrounding the whole response, else the stripped text."""
    t = text.strip()
    m = re.fullmatch(r"```[^\n]*\n(.*?)\n?```", t, re.S)
    return m.group(1) if m else t


def is_json_object(text):
    try:
        return isinstance(json.loads(strip_one_fence(text)), dict)
    except ValueError:
        return False


def qualitative_rows(q):
    """One row per response in question, format, condition order."""
    rows = []
    for question, formats in q["questions"].items():
        for fmt, conds in formats.items():
            for c in q["conditions"]:
                r = conds[c]
                row = {"id": r["id"], "question": question, "format": fmt, "condition": c, "n_tokens": r["n_tokens"],
                       "truncated": r["truncated"], "stop_reason": r["stop_reason"]}
                if fmt == "json":
                    row.update(fenced=r["response"].strip().startswith("```"), json_object=is_json_object(r["response"]))
                rows.append(row)
    return rows


def qualitative_summary(rows, conditions):
    out = []
    for c in conditions:
        r = [x for x in rows if x["condition"] == c]
        j = [x for x in r if x["format"] == "json"]
        out.append({"condition": c, "responses": len(r), "mean_tokens": float(np.mean([x["n_tokens"] for x in r])),
                    "truncated": sum(x["truncated"] for x in r), "json_prompts": len(j),
                    "json_object": sum(x["json_object"] for x in j), "json_fenced": sum(x["fenced"] for x in j),
                    "json_truncated": sum(x["truncated"] for x in j)})
    return out


def annotation_rows(q):
    """The fixed 42 responses (L254-255); manual columns empty."""
    plain = [k for k, v in q["questions"].items() if "plain" in v]
    assert (plain[0], plain[-1]) == ANNOTATION_QUESTIONS, plain
    rows = []
    for question in ANNOTATION_QUESTIONS:
        for fmt, conds in q["questions"][question].items():
            for c in q["conditions"]:
                r = conds[c]
                row = dict.fromkeys(ANNOTATION_COLUMNS, "")
                compliance = {"plain": "n/a", "template": ""}.get(fmt, "yes" if fmt == "json" and is_json_object(r["response"]) else "no")
                row.update(id=f"{r['id']}__{c}", question=question, format=fmt, condition=c, format_compliance=compliance,
                           truncated="y" if r["truncated"] else "n")
                rows.append(row)
    return rows


def annotation_sheet(q):
    lines = ["# Task 1 annotation sheet: 42 fixed responses (manual L254-256)", "",
             "Fill results/task1/annotations.csv: coherence 0/1/2; repetition, math_framing, inappropriate y/n; excerpt; notes."]
    fence = "`````"
    for question in ANNOTATION_QUESTIONS:
        for fmt, conds in q["questions"][question].items():
            lines += ["", f"## {question} / {fmt}", "", "Prompt:", "", fence + "text", conds[q["conditions"][0]]["prompt"], fence]
            for c in q["conditions"]:
                r = conds[c]
                auto = f"n_tokens {r['n_tokens']}; truncated {'y' if r['truncated'] else 'n'}; stop {r['stop_reason']}"
                if fmt == "json":
                    auto += f"; json_object {'y' if is_json_object(r['response']) else 'n'}"
                lines += ["", f"### {c}", "", auto, "", fence + "text", r["response"], fence]
    return "\n".join(lines) + "\n"


def write_new(path, write):
    """Hand-filled files are never replaced."""
    if os.path.exists(path):
        print(f"kept existing {path}")
        return False
    os.makedirs(os.path.dirname(path), exist_ok=True)
    write(path)
    return True


def write_annotations(path, rows):
    def write(p):
        with open(p, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=ANNOTATION_COLUMNS)
            w.writeheader()
            w.writerows(rows)
    return write_new(path, write)


def write_sheet(path, text):
    def write(p):
        with open(p, "w") as f:
            f.write(text)
    return write_new(path, write)


# ---------------------------------------------------------------- figures

def _axes(ax, title=None):
    ax.set_facecolor(SURFACE)
    ax.grid(True, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=INK_2, labelsize=8, length=0)
    if title:
        ax.set_title(title, fontsize=9, color=INK, loc="left")


def _save(fig, folder, name):
    path = os.path.join(folder, name)
    fig.savefig(path, dpi=200, facecolor=SURFACE, metadata={"Software": None})
    plt.close(fig)


def figure_learning(train, folder):
    """SFT loss for the five runs, LR below; L2-SP penalty terms on their own axes (L265-266, L281)."""
    fig = plt.figure(figsize=(10, 5.6), facecolor=SURFACE, constrained_layout=True)
    grid = fig.add_gridspec(6, 2, width_ratios=[2.1, 1])
    loss, lr = fig.add_subplot(grid[:4, 0]), fig.add_subplot(grid[4:, 0])
    for m, t in train.items():
        u = np.array([x["update"] for x in t["log"]])
        y = [x["sft_loss"] for x in t["log"]]
        loss.plot(u, y, color=METHOD_COLOR[m], alpha=0.14, linewidth=0.5)
        loss.plot(u[MA_WINDOW - 1:], moving_average(y), color=METHOD_COLOR[m], linewidth=1.5, label=LABEL[m],
                  linestyle=METHOD_DASH.get(m, "solid"))
    loss.set_yscale("log")
    loss.set_yticks([0.15, 0.2, 0.3, 0.5, 1.0], labels=["0.15", "0.2", "0.3", "0.5", "1.0"])
    loss.minorticks_off()
    _axes(loss, f"SFT loss (faint: each update; line: moving average of {MA_WINDOW} updates)")
    loss.legend(frameon=False, fontsize=8, labelcolor=INK_2, ncol=5, loc="upper right")
    loss.tick_params(labelbottom=False)
    for group, m in (((("full_ft", "lr"), "peak 2e-5: Full FT, L2-SP"), "full_ft"), ((("lora4", "lr"), "peak 2e-4: LoRA, OPLoRA"), "lora4")):
        (run, _), text = group
        u = [x["update"] for x in train[run]["log"]]
        y = [x["lr"] for x in train[run]["log"]]
        lr.plot(u, y, color=METHOD_COLOR[m], linewidth=1.5)
        lr.annotate(text, (u[300], y[300]), xytext=(6, 5), textcoords="offset points", fontsize=8, color=INK_2)
    lr.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    _axes(lr, "Learning rate")
    lr.yaxis.get_offset_text().set_fontsize(8)
    lr.set_xlabel("Optimizer update", fontsize=8, color=INK_2)
    log = train["l2sp"]["log"]
    u = [x["update"] for x in log]
    for i, (key, title) in enumerate((("omega", "L2-SP: Ω"), ("lambda_omega", "L2-SP: λΩ"), ("total_loss", "L2-SP: total loss (faint; line: MA 25)"))):
        ax = fig.add_subplot(grid[2 * i:2 * i + 2, 1])
        if key == "total_loss":
            y = [x[key] for x in log]
            ax.plot(u, y, color=METHOD_COLOR["l2sp"], alpha=0.14, linewidth=0.5)
            ax.plot(u[MA_WINDOW - 1:], moving_average(y), color=METHOD_COLOR["l2sp"], linewidth=1.5)
            ax.set_yscale("log")
            ax.set_yticks([0.2, 0.3, 0.5, 1.0], labels=["0.2", "0.3", "0.5", "1.0"])
            ax.minorticks_off()
            ax.set_xlabel("Optimizer update", fontsize=8, color=INK_2)
        else:
            ax.plot(u, [x[key] for x in log], color=METHOD_COLOR["l2sp"], linewidth=1.5)
        _axes(ax, title)
    _save(fig, folder, FIGURES[0])


def figure_benchmarks(table, comparisons, folder):
    """Change from baseline with paired 95% CIs (L220-221, L278); one panel per benchmark, each with its own axis."""
    conds = [r["condition"] for r in table][1:]
    y = np.arange(len(conds))[::-1]
    fig, axes = plt.subplots(1, 4, figsize=(10, 2.9), facecolor=SURFACE, sharey=True, constrained_layout=True)
    for ax, b, name in zip(axes, BENCHMARKS, ("GSM8K", "HellaSwag", "BoolQ", "ARC-Easy")):
        rows = {r["a"]: r for r in comparisons if r["b"] == "baseline" and r["benchmark"] == b}
        d = np.array([rows[c]["diff_pp"] for c in conds])
        lo, hi = np.array([rows[c]["ci_low_pp"] for c in conds]), np.array([rows[c]["ci_high_pp"] for c in conds])
        ax.axvline(0, color=INK_2, linewidth=0.8)
        ax.errorbar(d, y, xerr=[d - lo, hi - d], fmt="o", color=METHOD_COLOR["full_ft"], markersize=5, elinewidth=1.4, capsize=0)
        _axes(ax, f"{name} (baseline {100 * table[0][b]['primary']:.2f}%)")
        ax.set_xlabel("Change from baseline, pp", fontsize=8, color=INK_2)
    axes[0].set_yticks(y, [SHORT[c] for c in conds])
    axes[0].set_ylim(-0.6, len(conds) - 0.4)
    _save(fig, folder, FIGURES[1])


def figure_diagnostics(rows, folder):
    """rho_k, eta and R(z) medians with IQR per checkpoint and layer (L226-248, L285)."""
    conds = list(dict.fromkeys(r["condition"] for r in rows))
    x = np.arange(len(conds))
    fig, axes = plt.subplots(2, 2, figsize=(10, 5.2), facecolor=SURFACE, sharex=True, constrained_layout=True)
    panels = (("rho_k", axes[0, 0], "ρk: share of ΔW in the protected output subspace"),
              ("eta", axes[0, 1], "η: ||ΔW||F / ||W0||F"),
              ("R_math", axes[1, 0], "R(z) on math probes: median, IQR"),
              ("R_nonmath", axes[1, 1], "R(z) on non-math probes: median, IQR"))
    for key, ax, title in panels:
        for i, (layer, (color, marker)) in enumerate(LAYER_STYLE.items()):
            r = [v for v in rows if v["layer"] == int(layer)]
            pos = x + (i - 1) * 0.16
            if key in ("rho_k", "eta"):
                ax.plot(pos, [v[key] for v in r], linestyle="none", marker=marker, color=color, markersize=5, label=f"layer {layer}")
            else:
                med = np.array([v[f"{key}_median"] for v in r])
                err = [med - [v[f"{key}_q25"] for v in r], [v[f"{key}_q75"] for v in r] - med]
                ax.errorbar(pos, med, yerr=err, fmt=marker, color=color, markersize=5, elinewidth=1.2, label=f"layer {layer}")
        _axes(ax, title)
    axes[0, 0].axhline(DIRECTION_AGNOSTIC_RHO, color=INK_2, linewidth=0.8, linestyle="--")
    axes[0, 0].annotate("k/d_out = 0.125", (len(conds) - 0.55, DIRECTION_AGNOSTIC_RHO), xytext=(0, 5),
                        textcoords="offset points", fontsize=8, color=INK_2, ha="right")
    axes[0, 0].set_ylim(bottom=-0.01)
    axes[0, 1].set_ylim(bottom=0)
    top = max(axes[1, 0].get_ylim()[1], axes[1, 1].get_ylim()[1])
    for ax in axes[1]:
        ax.set_ylim(0, top)
        ax.set_xticks(x, [SHORT[c] for c in conds], fontsize=8)
    axes[0, 1].legend(frameon=False, fontsize=8, labelcolor=INK_2, loc="lower right")
    _save(fig, folder, FIGURES[2])


# ---------------------------------------------------------------- summary.md

def md_table(head, rows):
    out = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    out += ["| " + " | ".join(str(x) for x in r) + " |" for r in rows]
    return "\n".join(out)


def _pct(x):
    return f"{100 * x:.2f}"


def _pp(x):
    return f"{x:+.2f}"


def _gib(b):
    return f"{b / 2 ** 30:.2f}"


def _mb(b):
    return f"{b / 10 ** 6:,.1f}"


def summary_md(t):
    s = ["# Task 1 analysis summary", ""]

    def section(title, *tables):
        s.extend([f"## {title}", ""])
        for tb in tables:
            s.extend([tb, ""])

    head = ["condition"]
    for b in BENCHMARKS:
        head += [f"{b} %"] + ([f"{b} raw acc %"] if b in NORMALIZED else []) + [f"{b} change pp"]
        if b in NORMALIZED:
            head.append(f"{b} raw change pp")
    rows = []
    for r in t["benchmarks"]:
        row = [r["condition"]]
        for b in BENCHMARKS:
            v = r[b]
            row += [_pct(v["primary"])] + ([_pct(v["acc"])] if b in NORMALIZED else []) + [_pp(v["change_pp"])]
            if b in NORMALIZED:
                row.append(_pp(v["acc_change_pp"]))
        rows.append(row)
    boot = md_table(["A", "B", "benchmark", "A - B pp", "95% CI pp", "only A correct", "only B correct", "n"],
                    [[r["a"], r["b"], r["benchmark"], _pp(r["diff_pp"]), f"[{r['ci_low_pp']:+.2f}, {r['ci_high_pp']:+.2f}]",
                      r["n_a_only_correct"], r["n_b_only_correct"], r["n"]] for r in t["comparisons"]])
    section("1. Benchmarks", md_table(head, rows), f"Paired item bootstrap, {N_BOOT} resamples, seed {SEED}, primary metric:", boot,
            "Per-example flags reproduce scores.json exactly for all conditions and benchmarks.")

    ex = md_table(["condition", "correct", "wrong", "wrong, extracted", "of which no digit", "no digit: gold is last number",
                   "unextractable", "truncated", "truncated wrong", "strict rate", "strict acc", "flexible - strict pp"],
                  [[r["condition"], r["correct"], r["wrong"], r["wrong_extracted"], r["wrong_no_digit_extraction"],
                    r["no_digit_gold_is_last_number"], r["wrong_unextractable"], r["truncated"], r["truncated_wrong"],
                    f"{r['strict_extraction_rate']:.4f}", f"{r['strict_acc']:.4f}", _pp(r["flexible_minus_strict_pp"])]
                   for r in t["extraction"]])
    gl = md_table(["condition", "gained", "base trunc", "cond trunc", "neither", "base no digit", "base no digit, gold last",
                   "lost", "base trunc", "cond trunc", "neither"],
                  [[r["condition"], r["gained"], r["gained_baseline_truncated"], r["gained_condition_truncated"], r["gained_neither_truncated"],
                    r["gained_baseline_no_digit"], r["gained_baseline_no_digit_gold_is_last_number"], r["lost"],
                    r["lost_baseline_truncated"], r["lost_condition_truncated"], r["lost_neither_truncated"]] for r in t["gain_loss"]])
    section("2. GSM8K extraction vs correctness", ex, "Items gained and lost against the baseline:", gl)

    bq = md_table(["condition", "gold yes %", "pred yes %", "acc %", "acc gold yes %", "acc gold no %"],
                  [[r["condition"], _pct(r["gold_yes_share"]), _pct(r["pred_yes_share"]), _pct(r["acc"]), _pct(r["acc_gold_yes"]),
                    _pct(r["acc_gold_no"])] for r in t["boolq"]])
    nm = md_table(["benchmark", "condition", "acc %", "acc_norm %", "acc change pp", "acc_norm change pp", "norm - raw pp"],
                  [[r["benchmark"], r["condition"], _pct(r["acc"]), _pct(r["acc_norm"]), _pp(r["acc_change_pp"]),
                    _pp(r["acc_norm_change_pp"]), _pp(r["norm_minus_raw_change_pp"])] for r in t["norm"]])
    section("3. Prediction balance and normalization", bq, nm)

    def opt(r, k, f):
        return f.format(r[k]) if r.get(k) is not None else "-"

    lt = md_table(["run", "update", "displacement", "mean SFT last 50", "Omega", "0.5 disp^2", "lambda Omega", "total loss", "lambda Omega / mean SFT"],
                  [[r["run"], r["update"], f"{r['displacement']:.4f}", opt(r, "mean_sft_last50", "{:.5f}"), opt(r, "omega", "{:.4f}"),
                    opt(r, "half_displacement_squared", "{:.4f}"), opt(r, "lambda_omega", "{:.5f}"), opt(r, "total_loss", "{:.5f}"),
                    opt(r, "lambda_omega_over_mean_sft", "{:.4f}")] for r in t["learning"]])
    ml = md_table(["run", "mean SFT last 50"], [[m, f"{v:.5f}"] for m, v in t["mean_sft_last50"].items()])
    section("4. Learning and regularization", lt, ml, f"Figure: figures/task1/{FIGURES[0]}")

    rs = md_table(["run", "trainable params", "train h", "wall h", "s/update", "peak alloc GiB", "peak reserved GiB", "kept files MB",
                   "recovery state MB", "started UTC", "shared VM with"],
                  [[r["run"], f"{r['trainable_params']:,}", f"{r['train_h']:.2f}", f"{r['wall_h']:.2f}", f"{r['s_per_update']:.2f}",
                    _gib(r["peak_allocated_bytes"]), _gib(r["peak_reserved_bytes"]), _mb(r["kept_bytes"]), _mb(r["recovery_bytes"]),
                    r["started_utc"], ", ".join(r["shared_vm_with"]) or "none"] for r in t["resources"]])
    ov = {r["run"]: r["overhead"] for r in t["resources"]}
    l2, op = ov["l2sp"], ov["oplora16"]
    ot = md_table(["overhead", "value"],
                  [["L2-SP theta_0 bytes", f"{l2['theta0_bytes']:,}"], ["L2-SP theta_0 build s", f"{l2['theta0_build_s']:.4f}"],
                   ["L2-SP penalty s/update", f"{l2['penalty_s_per_update']:.4f}"], ["OPLoRA SVD s", f"{op['svd_preprocess_s']:.2f}"],
                   ["OPLoRA basis bytes", f"{op['basis_bytes']:,}"], ["OPLoRA export s", f"{op['export_s']:.4f}"]])
    section("5. Resources", rs, ot)

    dg = md_table(["condition", "layer", "rho_k", "eta", "R math median", "R math IQR", "R non-math median", "R non-math IQR", "math/non-math"],
                  [[r["condition"], r["layer"], f"{r['rho_k']:.4g}", f"{r['eta']:.5f}", f"{r['R_math_median']:.5f}", f"{r['R_math_iqr']:.5f}",
                    f"{r['R_nonmath_median']:.5f}", f"{r['R_nonmath_iqr']:.5f}", f"{r['math_nonmath_ratio']:.3f}"] for r in t["diagnostics"]])
    a = t["verification"]["oplora"]["aggregate"]
    vt = md_table(["OPLoRA constraint, 196 targets, k 128", "max", "median"],
                  [["norm(U_k^T dW) / norm(dW), Frobenius", f"{a['U_ratio']['max']:.3g}", f"{a['U_ratio']['median']:.3g}"],
                   ["norm(dW V_k) / norm(dW), Frobenius", f"{a['V_ratio']['max']:.3g}", f"{a['V_ratio']['median']:.3g}"]])
    mt = md_table(["merge check", "FP16 max", "FP16 mean", "FP16 argmax", "FP32 max", "FP32 mean", "FP32 argmax"],
                  [[r["condition"], *[f"{r[f'{p}_{k}']:.4g}" for p in ("fp16", "fp32") for k in ("max_abs_diff", "mean_abs_diff", "argmax_agreement")]]
                   for r in t["verification"]["merge"]])
    section("6. Diagnostics", f"Direction-agnostic rho_k: k/d_out = 128/1024 = {DIRECTION_AGNOSTIC_RHO}.", dg, vt, mt,
            f"Figure: figures/task1/{FIGURES[2]}")

    qt = md_table(["condition", "responses", "mean tokens", "truncated", "JSON objects", "JSON prompts", "fenced", "JSON truncated"],
                  [[r["condition"], r["responses"], f"{r['mean_tokens']:.1f}", r["truncated"], r["json_object"], r["json_prompts"],
                    r["json_fenced"], r["json_truncated"]] for r in t["qualitative"]])
    section("7. Qualitative code checks", qt)
    return "\n".join(s)


# ---------------------------------------------------------------- main

def analyze(results="results/task1", out=None, figures="figures/task1", sheet="scratch/task1_annotation_sheet.md",
            annotations=None, config="configs/task1/eval.yaml"):
    out = out or f"{results}/analysis"
    annotations = annotations or f"{results}/annotations.csv"
    os.makedirs(out, exist_ok=True)
    os.makedirs(figures, exist_ok=True)
    rd = Reader()
    cfg = rd.yaml(config)
    conditions, baseline = cfg["conditions"], cfg["baseline"]
    ev = load_eval(rd, results, conditions)
    train = {m: rd.json(f"{results}/train_{m}.json") for m in METHODS}
    diag = rd.json(f"{results}/diagnostics.json")
    ver = rd.json(f"{results}/oplora_verification.json")
    qual = rd.json(f"{results}/qualitative/all_168.json")
    assert qual["conditions"] == conditions and qual["n_responses"] == 168

    t = {"score_check": check_scores(ev), "benchmarks": benchmark_table(ev, baseline), "comparisons": comparison_table(ev),
         "extraction": extraction_table(ev), "gain_loss": gain_loss_table(ev, baseline), "boolq": boolq_balance(ev),
         "norm": norm_table(ev, baseline), "resources": resource_table(train), "diagnostics": diagnostics_table(diag),
         "verification": verification_table(ver, ev)}
    t["learning"], t["mean_sft_last50"] = learning_table(train)
    qrows = qualitative_rows(qual)
    t["qualitative"] = qualitative_summary(qrows, conditions)

    figure_learning(train, figures)
    figure_benchmarks(t["benchmarks"], t["comparisons"], figures)
    figure_diagnostics(t["diagnostics"], figures)

    prov = run_info()
    parts = {"benchmarks": ("score_check", "benchmarks", "comparisons"), "gsm8k_extraction": ("extraction", "gain_loss"),
             "prediction_balance": ("boolq", "norm"), "learning": ("learning", "mean_sft_last50"), "resources": ("resources",),
             "diagnostics_tables": ("diagnostics", "verification"), "qualitative_checks": ("qualitative",)}
    for name, keys in parts.items():
        doc = {"run_id": f"analysis_{name}", "provenance": prov, **{k: t[k] for k in keys}}
        if name == "qualitative_checks":
            doc["responses"] = qrows
        write_result(os.path.join(out, f"{name}.json"), doc)
    write_result(os.path.join(out, "manifest.json"), {"run_id": "analysis_manifest", "provenance": prov, "sources": rd.read,
                                                      "figures": [f"{figures}/{f}" for f in FIGURES]})
    write_annotations(annotations, annotation_rows(qual))
    write_sheet(sheet, annotation_sheet(qual))
    text = summary_md(t)
    with open(os.path.join(out, "summary.md"), "w") as f:
        f.write(text + "\n")
    return t, text, rd


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results/task1")
    ap.add_argument("--out")
    ap.add_argument("--figures", default="figures/task1")
    ap.add_argument("--sheet", default="scratch/task1_annotation_sheet.md")
    ap.add_argument("--annotations")
    args = ap.parse_args(argv)
    _, text, _ = analyze(args.results, args.out, args.figures, args.sheet, args.annotations)
    print(text)


if __name__ == "__main__":
    main()
