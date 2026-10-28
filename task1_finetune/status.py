"""Status lines, summary table, wait loop and file verification for Task 1 run JSONs (used by Kaggle cells).

    python -m task1_finetune.status results/task1/smoke_full_ft.json           # one line per file
    python -m task1_finetune.status --table results/task1/smoke_*.json         # summary table
    python -m task1_finetune.status --wait --every 300 results/task1/train_full_ft.json ...
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
    w = [max(len(r[i]) for r in rows) for i in range(len(head))]
    return "\n".join(" | ".join(c.ljust(w[i]) for i, c in enumerate(r)) for r in rows)


def _alive(method):
    pat = f"task1_finetune.train --config configs/task1/{method}.yaml"
    return subprocess.run(["pgrep", "-f", pat], capture_output=True).returncode == 0


def wait(paths, every):
    while True:
        states = []
        for p in paths:
            d = _load(p)
            print(time.strftime("%H:%M:%S"), line(p), flush=True)
            if d is not None and d["status"] == "done":
                states.append("done")
            elif d is not None and not _alive(d["method"]):
                states.append("dead")
            elif d is None and not _alive(os.path.basename(p)[len("train_"):-len(".json")]):
                states.append("dead")
            else:
                states.append("running")
        if "running" not in states:
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
    ap.add_argument("paths", nargs="+")
    ap.add_argument("--table", action="store_true")
    ap.add_argument("--wait", action="store_true")
    ap.add_argument("--every", type=int, default=300)
    ap.add_argument("--verify", action="store_true")
    a = ap.parse_args()
    if a.table:
        print(table(a.paths))
    elif a.wait:
        wait(a.paths, a.every)
    elif a.verify:
        raise SystemExit(verify(a.paths) > 0)
    else:
        for p in a.paths:
            print(line(p))


if __name__ == "__main__":
    main()
