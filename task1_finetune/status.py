"""Status lines, tables, queue plan, wait loop and file verification for Task 1 run JSONs (Kaggle cells).

    python -m task1_finetune.status results/task1/smoke_full_ft.json           # one line per file
    python -m task1_finetune.status --table results/task1/smoke_*.json         # smoke summary table
    python -m task1_finetune.status --train_table results/task1/train_*.json   # final training table
    python -m task1_finetune.status --plan --queue full_ft,lora4,oplora16 --queue l2sp,lora16 --limit_h 10.5
    python -m task1_finetune.status --wait --every 600 --done_files a.done b.done --started_file s results/...
    python -m task1_finetune.status --verify results/task1/train_full_ft.json  # sha256 of kept files
    python -m task1_finetune.status --find_ckpts /kaggle/input                 # locate and check kept files
    python -m task1_finetune.status --eval_status                              # one line per condition
    python -m task1_finetune.status --eval_plan --queue baseline,... --queue lora4_u1250,...,diagnostics
    python -m task1_finetune.status --eval_wait --every 600 --done_files a.done b.done --started_file s
    python -m task1_finetune.status --eval_summary                             # final table and diagnostics
    python -m task1_finetune.status --collect_qualitative                      # qualitative/all_168.json
"""
import argparse
import json
import os
import subprocess
import time

GiB = 2**30


def _load(path):
    try:
        with open(path) as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def _g(x, fmt="{:.4f}"):
    return "-" if x is None else fmt.format(x)


def line(path):
    d = _load(path)
    if d is None:
        return f"{path}: missing or unreadable"
    r, log = d["resources"], d["log"]
    last = log[-1] if log else {}
    total = len(d["smoke"]["time_per_update_s"]) if d["kind"] == "smoke" else d["config"]["batch"]["updates"]
    s = (f"{d['method']} {d['kind']} {d['status']} update {r['updates_done']}/{total} "
         f"loss {_g(last.get('sft_loss'))} lr {_g(last.get('lr'), '{:.2e}')} skipped {r['skipped_updates']} "
         f"peak_alloc {r['peak_allocated_bytes'] / GiB:.2f}GiB reserved {r['peak_reserved_bytes'] / GiB:.2f}GiB "
         f"train {r['train_s'] / 3600:.2f}h")
    if "omega" in last:
        s += f" omega {_g(last['omega'], '{:.4g}')} total {_g(last.get('total_loss'))}"
    if d["kind"] == "smoke":
        sm = d["smoke"]
        tok_s, upd_s = sm["projected_full_run_s_by_tokens"], sm["projected_full_run_s_by_updates"]
        s += (f" s/update {[round(t, 1) for t in sm['time_per_update_s']]} proj "
              f"{_g(tok_s and tok_s / 3600, '{:.2f}')}h (by updates {_g(upd_s and upd_s / 3600, '{:.2f}')}h)")
    return s


def _fmt_table(rows):
    w = [max(len(r[i]) for r in rows) for i in range(len(rows[0]))]
    return "\n".join(" | ".join(c.ljust(w[i]) for i, c in enumerate(r)) for r in rows)


def table(paths):
    head = ("method", "status", "updates", "skipped", "s/update", "peak_alloc_GiB", "reserved_GiB",
            "trainable", "proj_h_tokens", "proj_h_updates", "overhead")
    rows = [head]
    for p in paths:
        d = _load(p)
        if d is None:
            rows.append((p, "missing") + ("",) * (len(head) - 2))
            continue
        r = d["resources"]
        sm = d.get("smoke") or {}
        t = [x["time_s"] for x in d["log"]]
        ov = {k: (round(v, 2) if isinstance(v, float) else v) for k, v in d["overhead"].items()
              if k.endswith(("_s", "_bytes"))}
        rows.append((d["method"], d["status"], str(r["updates_done"]), str(r["skipped_updates"]),
                     " ".join(f"{x:.1f}" for x in t[-3:]), f"{r['peak_allocated_bytes'] / GiB:.2f}",
                     f"{r['peak_reserved_bytes'] / GiB:.2f}", f"{r['trainable_params']:,}",
                     _g(sm.get("projected_full_run_s_by_tokens") and sm["projected_full_run_s_by_tokens"] / 3600,
                        "{:.2f}"),
                     _g(sm.get("projected_full_run_s_by_updates") and sm["projected_full_run_s_by_updates"] / 3600,
                        "{:.2f}"),
                     json.dumps(ov)))
    return _fmt_table(rows)


def train_table(paths):
    head = ("method", "status", "updates", "skipped", "wall_h", "train_h", "ckpt_io_min", "peak_alloc_GiB",
            "reserved_GiB", "final_sft_loss", "displacement_l2")
    rows = [head]
    for p in paths:
        d = _load(p)
        if d is None:
            rows.append((os.path.basename(p), "missing") + ("",) * (len(head) - 2))
            continue
        r = d["resources"]
        last = d["log"][-1] if d["log"] else {}
        rows.append((d["method"], d["status"], str(r["updates_done"]), str(r["skipped_updates"]),
                     f"{r['wall_s'] / 3600:.2f}", f"{r['train_s'] / 3600:.2f}", f"{r['ckpt_io_s'] / 60:.1f}",
                     f"{r['peak_allocated_bytes'] / GiB:.2f}", f"{r['peak_reserved_bytes'] / GiB:.2f}",
                     _g(last.get("sft_loss")), json.dumps(d["displacement_l2"])))
    return _fmt_table(rows)


def plan(queues, limit_h, results_dir="results/task1"):
    """Projected hours per GPU queue from smoke JSONs; returns True when every queue is within limit_h.

    Per run: max(projection by tokens, projection by updates) + smoke setup time; checkpoint I/O excluded.
    """
    ok = True
    for qi, q in enumerate(queues):
        total = 0.0
        parts = []
        for m in q:
            d = _load(os.path.join(results_dir, f"smoke_{m}.json"))
            if d is None or d.get("kind") != "smoke":
                print(f"queue {qi}: missing smoke result for {m}")
                ok = False
                continue
            sm = d["smoke"]
            projs = [x for x in (sm["projected_full_run_s_by_tokens"], sm["projected_full_run_s_by_updates"])
                     if x is not None]
            if not projs:
                print(f"queue {qi}: no projection for {m} (smoke status {d['status']}, no update finished)")
                ok = False
                continue
            proj = max(projs)
            setup = sum((d["resources"].get("setup_s") or {}).values())
            h = (proj + setup) / 3600
            total += h
            parts.append(f"{m} {h:.2f}h")
        over = total > limit_h
        ok &= not over
        print(f"queue GPU{qi}: {' + '.join(parts)} = {total:.2f}h (limit {limit_h}h){'  OVER LIMIT' if over else ''}")
    print("plan:", "OK" if ok else "STOP")
    return ok


def _alive(pattern):
    return subprocess.run(["pgrep", "-f", pattern], capture_output=True).returncode == 0


def _method_of(path):
    return os.path.basename(path)[len("train_"):-len(".json")]


def wait(paths, every, done_files=None, started_file=None, warn_h=None, queue_pattern="run_queue.sh"):
    """Poll run JSONs. With done_files: finish when every marker exists (or no queue process is alive);
    otherwise finish when every run is done or its process is gone."""
    t0 = time.time()
    if started_file and os.path.exists(started_file):
        with open(started_file) as f:
            t0 = float(f.read().strip())
    while True:
        elapsed_h = (time.time() - t0) / 3600
        print(f"--- {time.strftime('%H:%M:%S')} elapsed {elapsed_h:.2f}h", flush=True)
        states = []
        for p in paths:
            d = _load(p)
            if d is None:
                print(f"  {_method_of(p)}: pending (no result JSON yet)", flush=True)
                states.append("pending")
                continue
            r, log = d["resources"], d["log"]
            last = log[-1] if log else {}
            print(f"  {d['method']}: status {d['status']} update {r['updates_done']}/{d['config']['batch']['updates']} "
                  f"sft_loss {_g(last.get('sft_loss'))} skipped {r['skipped_updates']}", flush=True)
            pat = f"task1_finetune.train --config configs/task1/{d['method']}.yaml"
            states.append("done" if d["status"] == "done" else "running" if _alive(pat) else "stopped")
        if warn_h is not None and elapsed_h > warn_h:
            print(f"WARNING: {elapsed_h:.2f}h elapsed (> {warn_h}h); the Kaggle session ends at 12h", flush=True)
        if done_files:
            done = [os.path.exists(f) for f in done_files]
            if all(done) or not _alive(queue_pattern):
                print("queues finished:" if all(done) else "queue processes gone; markers:",
                      dict(zip(done_files, done)), dict(zip(paths, states)), flush=True)
                return states
        elif all(s in ("done", "stopped") for s in states):
            print("finished:", dict(zip(paths, states)), flush=True)
            return states
        time.sleep(every)


def verify(paths):
    """Recompute sha256 of every recorded non-recovery file; returns number of mismatches/missing."""
    from task1_finetune.train import sha256_file
    bad = 0
    for p in paths:
        d = _load(p)
        for e in d["files"]:
            if e["kind"] == "recovery":
                continue
            ok = os.path.exists(e["path"]) and sha256_file(e["path"]) == e["sha256"]
            bad += not ok
            print(f"{'OK ' if ok else 'BAD'} {e['kind']} u{e['update']} {e['bytes'] / GiB:.3f}GiB {e['path']}",
                  flush=True)
    print(f"verify: {bad} problem(s)")
    return bad


# ---------------------------------------------------------------- evaluation (T1-eval)

EVAL_CONFIG = "configs/task1/eval.yaml"
DIAGNOSTICS_BUDGET_H = 0.25  # fixed allowance in the queue plan: six checkpoints, three layers, 196 SVDs


def _eval_cfg(path=EVAL_CONFIG):
    import yaml
    with open(path) as f:
        return yaml.safe_load(f)


def _scores(root, cond):
    return _load(os.path.join(root, cond, "scores.json"))


def _gib(x):
    return None if x is None else x / GiB


def eval_line(root, cond):
    d = _scores(root, cond)
    if d is None:
        return f"{cond}: pending (no result JSON yet)"
    s = d["scores"]
    prim = " ".join(f"{k} {100 * v['primary']:.2f}" for k, v in s.items())
    g = s.get("gsm8k", {})
    return (f"{cond}: status {d['status']} stages {','.join(d['stages_s'])} {prim} truncated {g.get('truncated', '-')} "
            f"no_extraction {g.get('no_flexible_extraction', '-')} peak_alloc "
            f"{d['resources']['peak_allocated_bytes'] / GiB:.2f}GiB wall {d['resources']['wall_s'] / 60:.1f}min")


def eval_plan(queues, limit_h, smoke_root):
    """Projected hours per GPU queue from the smoke JSONs (baseline for full-model conditions, lora16 for
    adapter conditions); STOP also when a worst-case batch at the configured batch size does not fit."""
    ok = True
    smoke = {c: _scores(smoke_root, c) for c in ("baseline", "lora16_u1250")}
    for c, d in smoke.items():
        if d is None or d["status"] != "done":
            print(f"smoke {c}: {'missing' if d is None else d['status']}")
            ok = False
            continue
        b = str(d["config"]["lm_eval"]["batch_size_likelihood"])
        w, t, r = d["worst_batches"], d["timing"], d["resources"]
        print(f"smoke {c}: peak {r['peak_allocated_bytes'] / GiB:.2f}GiB (reserved {r['peak_reserved_bytes'] / GiB:.2f}GiB) "
              f"s/request generate_until {t['generate_until']['seconds'] / t['generate_until']['requests']:.3f} "
              f"loglikelihood {t['loglikelihood']['seconds'] / t['loglikelihood']['requests']:.4f} "
              f"projected full condition {d['projection']['total_s'] / 3600:.2f}h {json.dumps(d['projection'])}")
        print(f"smoke {c}: worst likelihood batches " + ", ".join(
            f"{k}: fits {v['fits']} peak {_gib(v['peak_allocated_bytes'] or 0):.2f}GiB" for k, v in w["likelihood"].items())
            + f"; worst gsm8k fits {w['gsm8k']['fits']} peak {_gib(w['gsm8k']['peak_allocated_bytes'] or 0):.2f}GiB")
        if b not in w["likelihood"] or not w["likelihood"][b]["fits"] or not w["gsm8k"]["fits"]:
            print(f"smoke {c}: configured batch size {b} does not fit, or a worst-case batch failed")
            ok = False
    if not ok:
        print("plan: STOP")
        return False
    for qi, q in enumerate(queues):
        total, parts = 0.0, []
        for c in q:
            if c == "diagnostics":
                h = DIAGNOSTICS_BUDGET_H
            else:
                ref = "lora16_u1250" if c.startswith(("lora", "oplora")) else "baseline"
                h = smoke[ref]["projection"]["total_s"] / 3600
            total += h
            parts.append(f"{c} {h:.2f}h")
        over = total > limit_h
        ok &= not over
        print(f"queue GPU{qi}: {' + '.join(parts)} = {total:.2f}h (limit {limit_h}h){'  OVER LIMIT' if over else ''}")
    print("plan:", "OK" if ok else "STOP")
    return ok


def wait_eval(conds, root, every, done_files, started_file=None, warn_h=None, queue_pattern="run_eval_queue.sh"):
    t0 = time.time()
    if started_file and os.path.exists(started_file):
        with open(started_file) as f:
            t0 = float(f.read().strip())
    while True:
        elapsed_h = (time.time() - t0) / 3600
        print(f"--- {time.strftime('%H:%M:%S')} elapsed {elapsed_h:.2f}h", flush=True)
        for c in conds:
            print(" ", eval_line(root, c), flush=True)
        if warn_h is not None and elapsed_h > warn_h:
            print(f"WARNING: {elapsed_h:.2f}h elapsed (> {warn_h}h); the Kaggle session ends at 12h", flush=True)
        done = [os.path.exists(f) for f in done_files]
        if all(done) or not _alive(queue_pattern):
            print("queues finished:" if all(done) else "queue processes gone; markers:", dict(zip(done_files, done)),
                  flush=True)
            return
        time.sleep(every)


def eval_summary(cfg):
    o = cfg["outputs"]
    root = o["eval_root"]
    head = ("condition", "status", "gsm8k", "hellaswag", "boolq", "arc_easy", "truncated", "no_extraction",
            "fp16_max", "fp16_mean", "fp16_argmax", "fp32_max", "fp32_mean", "fp32_argmax", "merge_s")
    rows = [head]
    for c in cfg["conditions"]:
        d = _scores(root, c)
        if d is None:
            rows.append((c, "missing") + ("",) * (len(head) - 2))
            continue
        s, mc = d["scores"], d["merge_check"]
        prim = [_g(s[k]["primary"] * 100, "{:.2f}") if k in s else "-" for k in ("gsm8k", "hellaswag", "boolq", "arc_easy")]
        g = s.get("gsm8k", {})
        check = ["n/a"] * 6
        if mc:
            check = [f"{mc[p][k]:.4g}" for p in ("fp16", "fp32") for k in ("max_abs_diff", "mean_abs_diff", "argmax_agreement")]
        rows.append((c, d["status"], *prim, str(g.get("truncated", "-")), str(g.get("no_flexible_extraction", "-")),
                     *check, _g((d["model"] or {}).get("merge_s"), "{:.2f}")))
    print(_fmt_table(rows))
    for name, key in (("diagnostics", "diagnostics"), ("oplora_verification", "oplora_verification")):
        d = _load(o[key])
        print(f"{name}: {'missing' if d is None else d['status']}" + (f" {d.get('aggregate')}" if d else ""))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("paths", nargs="*")
    ap.add_argument("--table", action="store_true")
    ap.add_argument("--train_table", action="store_true")
    ap.add_argument("--plan", action="store_true")
    ap.add_argument("--queue", action="append", default=[], help="comma-separated methods of one GPU queue")
    ap.add_argument("--limit_h", type=float, default=10.5)
    ap.add_argument("--wait", action="store_true")
    ap.add_argument("--every", type=int, default=300)
    ap.add_argument("--done_files", nargs="*")
    ap.add_argument("--started_file")
    ap.add_argument("--warn_h", type=float)
    ap.add_argument("--verify", action="store_true")
    ap.add_argument("--find_ckpts", metavar="ROOT")
    ap.add_argument("--ckpt_index", default="results/task1/ckpt_index.json")
    ap.add_argument("--eval_config", default=EVAL_CONFIG)
    ap.add_argument("--eval_status", action="store_true")
    ap.add_argument("--eval_plan", action="store_true")
    ap.add_argument("--eval_wait", action="store_true")
    ap.add_argument("--eval_summary", action="store_true")
    ap.add_argument("--smoke", action="store_true", help="--eval_status on the smoke directory")
    ap.add_argument("--collect_qualitative", action="store_true")
    a = ap.parse_args()
    if a.find_ckpts:
        from task1_finetune.build import METHODS, find_checkpoints
        cfg = _eval_cfg(a.eval_config)
        paths = [os.path.join(cfg["train_results"], f"train_{m}.json") for m in METHODS]
        index, problems = find_checkpoints(a.find_ckpts, paths, a.ckpt_index)
        print(f"checkpoints: {len(index)} OK, {len(problems)} problem(s) {problems}")
        raise SystemExit(1 if problems else 0)
    if a.eval_status or a.eval_plan or a.eval_wait or a.eval_summary or a.collect_qualitative:
        cfg = _eval_cfg(a.eval_config)
        o = cfg["outputs"]
        if a.eval_status:
            conds = ("baseline", "lora16_u1250") if a.smoke else cfg["conditions"]
            for c in conds:
                print(eval_line(o["smoke_root"] if a.smoke else o["eval_root"], c))
        elif a.eval_plan:
            raise SystemExit(0 if eval_plan([q.split(",") for q in a.queue], a.limit_h, o["smoke_root"]) else 1)
        elif a.eval_wait:
            wait_eval(cfg["conditions"], o["eval_root"], a.every, a.done_files, a.started_file, a.warn_h)
        elif a.eval_summary:
            eval_summary(cfg)
        else:
            from task1_finetune.qualitative import collect
            print(f"qualitative: {collect(cfg)} responses in {o['qualitative_root']}/all_168.json")
        return
    if a.table:
        print(table(a.paths))
    elif a.train_table:
        print(train_table(a.paths))
    elif a.plan:
        raise SystemExit(0 if plan([q.split(",") for q in a.queue], a.limit_h) else 1)
    elif a.wait:
        wait(a.paths, a.every, a.done_files, a.started_file, a.warn_h)
    elif a.verify:
        raise SystemExit(verify(a.paths) > 0)
    else:
        for p in a.paths:
            print(line(p))


if __name__ == "__main__":
    main()
