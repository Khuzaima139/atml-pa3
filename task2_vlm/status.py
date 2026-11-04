"""Task 2 Kaggle helpers: smoke-based time plan, progress lines, wait loop and summary tables.

    python -m task2_vlm.status --plan [--smoke_dir results/task2/smoke] [--cache results/task2/cache_features.json]
    python -m task2_vlm.status --progress notes/kaggle_logs/task2_smoke.log    # last update / dev / stage lines
    python -m task2_vlm.status --wait --log notes/kaggle_logs/task2_main.log --done notes/kaggle_logs/task2_main.done
    python -m task2_vlm.status --summary [--run_dir results/task2/main]

Text patterns parsed from logs (checked by a Kaggle check cell on the smoke log): lines starting with
"update ", "dev u", "stage " (from task2_vlm.train) and "=== " (from task2_vlm/run_main.sh).
"""
import argparse
import json
import os
import subprocess
import sys
import time

import yaml

GiB = 2**30
PATTERNS = ("update ", "dev u", "stage ")
PROCESS = "[t]ask2_vlm.(train|evaluate)"


def _load(path):
    try:
        with open(path) as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def _f(x, fmt="{:.3f}"):
    return "-" if x is None else fmt.format(x)


# ---------------------------------------------------------------- plan

def plan(smoke_dir, cache_path, config, limit_h):
    tr, ev, ca = (_load(os.path.join(smoke_dir, "train.json")), _load(os.path.join(smoke_dir, "final/eval.json")),
                  _load(cache_path))
    ok = True
    for name, d in (("smoke train", tr), ("smoke eval", ev), ("feature cache", ca)):
        if d is None or d.get("status") != "done":
            print(f"STOP: {name} result missing or not done")
            ok = False
    if not ok:
        return False
    with open(config) as f:
        cfg = yaml.safe_load(f)
    sm = tr["smoke"]
    main_T = sm["main_stage_updates"]
    print("stage | smoke s/update (excl. first) | updates | projected h | peak alloc GiB | peak reserved GiB")
    train_s = 0.0
    for k, T in enumerate(main_T, 1):
        m = sm["mean_s_per_update_excl_first"][str(k)]
        st = tr["stages"][k - 1]
        train_s += m * T
        print(f"{k} | {m:.3f} ({', '.join(f'{x:.2f}' for x in sm['time_per_update_s_by_stage'][str(k)])}) | {T} | "
              f"{m * T / 3600:.2f} | {st['peak_allocated_bytes'] / GiB:.2f} | {st['peak_reserved_bytes'] / GiB:.2f}")
    dev_s = sum(sm["dev_eval_s"]) / len(sm["dev_eval_s"]) * sm["main_dev_points"]
    rec = sm["recovery_save_s"]
    ckpt_s = (sum(rec) / len(rec) if rec else 0.0) * sm["main_recovery_saves"]
    setup_s = sum(tr["resources"]["setup_s"].values()) + tr["resources"]["calibration_s"] + ev["setup_s"]
    final_s = 0.0
    for suite, t in ev["timing"].items():
        n_main = len(_load(cfg["data"][suite])["pairs"])
        f = n_main / ev["suites"][suite]["n_pairs"]
        final_s += f * sum(c["generation_s"] + c["nll_s"] for c in t.values())
    cache_s = ca["wall_s"]
    smoke_s = tr["resources"]["wall_s"] + sum(tr["resources"]["setup_s"].values()) + ev["wall_s"] + ev["setup_s"]
    total = cache_s + smoke_s + train_s + dev_s + ckpt_s + setup_s + final_s
    peak = max(tr["resources"]["peak_allocated_bytes"] or 0, ev.get("peak_allocated_bytes") or 0)
    dl = ca.get("download_s")
    rows = [("feature caching (done; download %s s)" % _f(dl, "{:.0f}"), cache_s), ("smoke runs (done)", smoke_s),
            ("training updates", train_s), (f"dev evaluation x{sm['main_dev_points']}", dev_s),
            (f"resumable states x{sm['main_recovery_saves']}", ckpt_s), ("setup (main train + eval)", setup_s),
            ("final evaluation", final_s)]
    for name, s in rows:
        print(f"  {name}: {s / 3600:.2f} h")
    print(f"peak GPU memory allocated (smoke): {peak / GiB:.2f} GiB")
    print(f"projected session total: {total / 3600:.2f} h (limit {limit_h} h)")
    if total > limit_h * 3600:
        print(f"STOP: projected total {total / 3600:.2f} h exceeds {limit_h} h")
        return False
    print("PLAN OK")
    return True


# ---------------------------------------------------------------- progress and wait

def progress(log):
    """Last line of each parsed pattern in a training log."""
    last = {}
    try:
        with open(log, errors="replace") as f:
            for line in f:
                for p in PATTERNS + ("=== ",):
                    if line.startswith(p):
                        last[p] = line.rstrip("\n")
    except OSError:
        pass
    return last


def running():
    r = subprocess.run(["pgrep", "-af", PROCESS], capture_output=True, text=True)
    return [x for x in r.stdout.splitlines() if "python" in x]


def wait(log, done, every, warn_h, started_file):
    t0 = float(open(started_file).read().strip()) if started_file and os.path.exists(started_file) else time.time()
    while True:
        el = (time.time() - t0) / 3600
        last = progress(log)
        print(f"[{time.strftime('%H:%M:%S')}] elapsed {el:.2f} h | " +
              " | ".join(last.get(p, f"(no '{p.strip()}' line yet)") for p in ("update ", "dev u")), flush=True)
        if os.path.exists(done):
            print("done:", open(done).read().strip())
            return
        if not running():
            time.sleep(60)  # run_main.sh starts the evaluation right after training ends
            if running() or os.path.exists(done):
                continue
            print("WARNING: no task2_vlm process running and no done file; see the log tail:")
            subprocess.run(["tail", "-n", "20", log])
            return
        if el > warn_h:
            print(f"WARNING: over {warn_h} h")
        time.sleep(every)


# ---------------------------------------------------------------- summary

def summary(run_dir):
    tr = _load(os.path.join(run_dir, "train.json"))
    if tr is None:
        print(f"{run_dir}/train.json missing")
        return
    r = tr["resources"]
    print(f"train {tr['run_id']} {tr['status']}: {r['updates_done']}/{r['updates_total']} updates, skipped "
          f"{r['skipped_updates']}, train {r['train_s'] / 3600:.2f} h, dev eval {r['dev_eval_s'] / 3600:.2f} h, "
          f"wall {r['wall_s'] / 3600:.2f} h, peak alloc {r['peak_allocated_bytes'] / GiB:.2f} GiB")
    c = tr["calibration"]
    print(f"calibration s = {c['s']:.6f} (text {c['text_mean_l2']:.6f} / connector {c['connector_mean_l2_at_init']:.6f})")
    for s in tr["stages"]:
        print(f"stage {s['stage']} {s['name']}: {s['status']} trainable {s.get('trainable_params', 0):,} train "
              f"{s['train_s'] / 3600:.2f} h dev {s['dev_eval_s'] / 60:.1f} min peak {s['peak_allocated_bytes'] / GiB:.2f} "
              f"GiB skipped {s['skipped']} mixture {s.get('mixture')}")
    print("dev | update | caption_loss | vqa nll correct/compl/none | vqa dNLL | grounding nll correct/compl/none | grounding dNLL")
    for d in tr["dev"]:
        v, g = d["vqa_dev"], d["grounding_dev"]
        print(f"dev | {d['update']} | {d['caption_loss']:.4f} | " + "/".join(_f(v["nll"][k]) for k in v["nll"])
              + f" | {_f(v['dnll'])} | " + "/".join(_f(g["nll"][k]) for k in g["nll"]) + f" | {_f(g['dnll'])}")
    ev = _load(os.path.join(run_dir, "final/eval.json"))
    if ev is None:
        print("final evaluation: missing")
        return
    print(f"final {ev['run_id']} {ev['status']} wall {ev.get('wall_s', 0) / 60:.1f} min peak "
          f"{(ev.get('peak_allocated_bytes') or 0) / GiB:.2f} GiB")
    for suite, s in ev.get("suites", {}).items():
        print(f"{suite} ({s['n_pairs']} pairs) | condition | accuracy | pair acc | identical | empty norm/raw | "
              f"follow partner | ref NLL | s/answer")
        for cond, m in s["conditions"].items():
            print(f"  {cond} | {_f(m['accuracy'])} | {_f(m['pair_accuracy'])} | {_f(m['identical_answer_rate'])} | "
                  f"{_f(m['empty_rate_normalized'])}/{_f(m['empty_rate_raw'])} | {_f(m['follow_partner_rate'])} | "
                  f"{_f(m['mean_reference_nll'])} | {_f(m['generation_s_per_answer'], '{:.4f}')}")
            print("    by type: " + "; ".join(f"{t} acc {_f(x['accuracy'])} pair {_f(x['pair_accuracy'])}"
                                             for t, x in m["by_type"].items()))
        for o, a in s["bootstrap"]["all"]["correct_advantage_over"].items():
            acc, nll = a["accuracy"], a["reference_nll"]
            print(f"  correct - {o}: accuracy {acc['estimate']:+.3f} [{acc['ci'][0]:+.3f}, {acc['ci'][1]:+.3f}] | "
                  f"NLL {nll['estimate']:+.3f} [{nll['ci'][0]:+.3f}, {nll['ci'][1]:+.3f}]")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan", action="store_true")
    ap.add_argument("--progress")
    ap.add_argument("--wait", action="store_true")
    ap.add_argument("--summary", action="store_true")
    ap.add_argument("--smoke_dir", default="results/task2/smoke")
    ap.add_argument("--cache", default="results/task2/cache_features.json")
    ap.add_argument("--config", default="configs/task2/train.yaml")
    ap.add_argument("--limit_h", type=float, default=10.5)
    ap.add_argument("--log", default="notes/kaggle_logs/task2_main.log")
    ap.add_argument("--done", default="notes/kaggle_logs/task2_main.done")
    ap.add_argument("--started_file", default="notes/kaggle_logs/task2_main.started")
    ap.add_argument("--every", type=int, default=600)
    ap.add_argument("--warn_h", type=float, default=11.0)
    ap.add_argument("--run_dir", default="results/task2/main")
    a = ap.parse_args(argv)
    if a.plan:
        sys.exit(0 if plan(a.smoke_dir, a.cache, a.config, a.limit_h) else 1)
    if a.progress:
        last = progress(a.progress)
        for p in PATTERNS:
            print(f"{p.strip()!r}: {last.get(p, 'MISSING')}")
        sys.exit(0 if all(p in last for p in PATTERNS) else 1)
    if a.wait:
        wait(a.log, a.done, a.every, a.warn_h, a.started_file)
    if a.summary:
        summary(a.run_dir)


if __name__ == "__main__":
    main()
