"""Status lines, tables, queue plan, wait loop and file verification for Task 1 run JSONs (Kaggle cells).

    python -m task1_finetune.status results/task1/smoke_full_ft.json           # one line per file
    python -m task1_finetune.status --table results/task1/smoke_*.json         # smoke summary table
    python -m task1_finetune.status --train_table results/task1/train_*.json   # final training table
    python -m task1_finetune.status --plan --queue full_ft,lora4,oplora16 --queue l2sp,lora16 --limit_h 10.5
    python -m task1_finetune.status --wait --every 600 --done_files a.done b.done --started_file s results/...
    python -m task1_finetune.status --verify results/task1/train_full_ft.json  # sha256 of kept files
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
        s += (f" s/update {[round(t, 1) for t in sm['time_per_update_s']]} proj "
              f"{_g(sm['projected_full_run_s_by_tokens'] and sm['projected_full_run_s_by_tokens'] / 3600, '{:.2f}')}h"
              f" (by updates {sm['projected_full_run_s_by_updates'] / 3600:.2f}h)")
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
            proj = max(x for x in (sm["projected_full_run_s_by_tokens"], sm["projected_full_run_s_by_updates"])
                       if x is not None)
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
    a = ap.parse_args()
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
