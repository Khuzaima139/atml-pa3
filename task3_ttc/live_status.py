"""Kaggle helpers for the live run: plan from the smoke, wait loop, summary (no gold is read).

    python -m task3_ttc.live_status --plan --smoke results/task3/live/smoke --preflight results/task3/live/preflight
                                    [--limit-h 10.5]
    python -m task3_ttc.live_status --wait --run results/task3/live/main --log notes/kaggle_logs/task3_live.log
                                    --done notes/kaggle_logs/task3_live.done --exit notes/kaggle_logs/task3_live.exit
                                    --pid notes/kaggle_logs/task3_live.pid [--every 600] [--poll 30]
    python -m task3_ttc.live_status --summary --run results/task3/live/main

Plan: decode seconds per generated token from the smoke (each call's time minus its prefill at the preflight's
measured prefill rate), times the projected 161,335 generated tokens (TASK3_FACTS item 14), plus prefill of the
projected input tokens; exits 1 above --limit-h.

Wait: the background wrapper (Kaggle cell 6) writes its pid file at launch, the exit file with the run's exit code on
any exit, and the done file only on exit code 0. Progress is printed every --every seconds; the done file and the
wrapper's liveness are checked every --poll seconds. Exit 0 on the done file (with exit code 0); exit 1 as soon as
the wrapper is gone without a done file (or the pid file is missing), after printing the exit code (if any) and the
last 30 log lines. There is no time limit, only a warning after --warn-h hours.
"""
import argparse
import json
import os
import sys
import time

from task3_ttc import live

# TASK3_FACTS item 14 (supplied MATH lengths, 50 live questions): single 34,010 + parallel 93,315 + refine 34,010.
PROJECTED_OUTPUT_TOKENS = 34010 + 93315 + 34010
# Item 12: mean live prompt 108.4 tokens; refine prompt ~ prompt + draft (mean min(len, 1,024) = 622.1) + 38.
PROJECTED_INPUT_TOKENS = int(round(50 * (4 * 108.4 + (108.4 + 622.1 + 38))))


def _calls(run_dir):
    p = os.path.join(run_dir, "calls.jsonl")
    if not os.path.exists(p):
        return []
    with open(p, encoding="utf-8") as f:
        return [json.loads(x) for x in f if x.strip().endswith("}")]


def _run(run_dir):
    with open(os.path.join(run_dir, "run.json")) as f:
        return json.load(f)


def plan(smoke_dir, preflight_dir, limit_h):
    with open(os.path.join(preflight_dir, "preflight.json")) as f:
        pre = json.load(f)
    prefill_tps = min(x["prefill_tokens_per_s"] for x in pre["forwards"])
    calls = _calls(smoke_dir)
    assert len(calls) == len(live.CALLS), f"smoke has {len(calls)} calls, expected {len(live.CALLS)}"
    print(f"{'call':7s} {'in':>5s} {'out':>5s} {'s':>7s} {'tok/s':>7s} trunc")
    for r in calls:
        print(f"{r['strategy']:7s} {r['input_tokens']:5d} {r['output_tokens']:5d} {r['wall_s']:7.1f} "
              f"{r['output_tokens'] / r['wall_s']:7.2f} {r['truncated']}")
    out = sum(r["output_tokens"] for r in calls)
    wall = sum(r["wall_s"] for r in calls)
    prefill = sum(r["input_tokens"] for r in calls) / prefill_tps
    s_per_tok = (wall - prefill) / max(out, 1)
    est = PROJECTED_OUTPUT_TOKENS * s_per_tok + PROJECTED_INPUT_TOKENS / prefill_tps
    print(f"smoke: {out} output tokens in {wall:.1f} s ({out / wall:.2f} tok/s overall); prefill rate "
          f"{prefill_tps:.0f} tok/s (preflight); decode {1 / s_per_tok:.2f} tok/s")
    print(f"projection: {PROJECTED_OUTPUT_TOKENS:,} output tokens x {s_per_tok:.4f} s + {PROJECTED_INPUT_TOKENS:,} "
          f"input tokens / {prefill_tps:.0f} tok/s = {est / 3600:.2f} h (limit {limit_h} h; model loading excluded)")
    if est / 3600 > limit_h:
        print(f"STOP: projected {est / 3600:.2f} h above {limit_h} h")
        return 1
    print("plan OK")
    return 0


def progress(run_dir, log):
    calls = _calls(run_dir)
    qs = {}
    for r in calls:
        qs[r["q"]] = qs.get(r["q"], 0) + 1
    done_q = sum(1 for v in qs.values() if v == len(live.CALLS))
    last = ""
    if log and os.path.exists(log):
        with open(log, "rb") as f:
            lines = [x for x in f.read().decode("utf-8", "replace").splitlines() if x.strip()]
        last = lines[-1] if lines else ""
    return done_q, len(calls), last


def _alive(pid):
    """True while the process exists and is not a zombie (Linux /proc state checked when available)."""
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    try:
        with open(f"/proc/{pid}/stat") as f:
            return f.read().rsplit(")", 1)[1].split()[0] != "Z"
    except OSError:
        return True


def _tail(log, n=30):
    if not (log and os.path.exists(log)):
        return []
    with open(log, "rb") as f:
        return f.read().decode("utf-8", "replace").splitlines()[-n:]


def _fail(msg, exit_file, log):
    code = None
    if exit_file and os.path.exists(exit_file):
        with open(exit_file) as f:
            code = f.read().strip()
    print(f"STOP: {msg}; exit code: {code if code is not None else 'none written (killed?)'}", flush=True)
    print(f"--- last 30 lines of {log} ---", flush=True)
    for x in _tail(log):
        print(x, flush=True)
    return 1


def wait(run_dir, log, done, exit_file, pid_file, every, poll, warn_h):
    t0 = time.time()
    if not os.path.exists(pid_file):
        return _fail(f"no pid file {pid_file} (was cell 6 run?)", exit_file, log)
    with open(pid_file) as f:
        pid = int(f.read().strip())
    next_report = t0
    while True:
        if time.time() >= next_report:
            q, c, last = progress(run_dir, log)
            el = (time.time() - t0) / 3600
            print(f"[{time.strftime('%H:%M:%S')}] questions {q}/50, calls {c}/250 | last log line: {last[:200]}"
                  + (f" | WARNING waited {el:.1f} h" if el > warn_h else ""), flush=True)
            next_report += every
        if os.path.exists(done):
            with open(done) as f:
                code = f.read().strip()
            if code != "0":
                return _fail(f"done file holds {code!r}", exit_file, log)
            q, c, _ = progress(run_dir, log)
            print(f"done file: exit code 0; questions {q}/50, calls {c}/250", flush=True)
            return 0
        if not _alive(pid):
            if os.path.exists(done):  # finished between the two checks
                continue
            return _fail(f"process {pid} is no longer running and there is no done file", exit_file, log)
        time.sleep(max(0.0, min(poll, next_report - time.time())))


def summary(run_dir):
    run = _run(run_dir)
    t = run["totals"]
    print(f"run {run['run_id']} status {run['status']}: calls {t['calls_done']}/{t['calls_expected']}, "
          f"questions {t['questions_done']}/{t['questions_expected']}")
    print(f"{'strategy':8s} {'calls':>5s} {'trunc':>5s} {'eos':>5s} {'no_ans':>6s} {'in_tok':>8s} {'out_tok':>8s} {'h':>6s}")
    for s, d in t["by_strategy"].items():
        print(f"{s:8s} {d['calls']:5d} {d['truncated']:5d} {d['finished_by_eos']:5d} {d['extraction_failures']:6d} "
              f"{d['input_tokens']:8d} {d['output_tokens']:8d} {d['wall_s'] / 3600:6.2f}")
    pa, pr = t["peak_mem_allocated_bytes"], t["peak_mem_reserved_bytes"]
    print(f"total call time {t['wall_s_calls'] / 3600:.2f} h; sessions {len(run['sessions'])}, wall "
          f"{run['wall_s_sessions'] / 3600:.2f} h; peak memory allocated "
          f"{(pa or 0) / 2**30:.2f} GiB, reserved {(pr or 0) / 2**30:.2f} GiB")
    print("effective sampling:", {k: run["effective_generation_config"]["single"][k] for k in live.SAMPLING},
          "| processors:", run["logits_processors"])
    print("commits:", [(s["provenance"]["git"]["commit"], s["provenance"]["git"]["dirty"]) for s in run["sessions"]],
          "| gpu:", [s["torch"].get("gpu_name") for s in run["sessions"]])
    ok = run["status"] == "complete" and t["calls_done"] == t["calls_expected"]
    return 0 if ok else 1


def main(argv=None):
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--plan", action="store_true")
    g.add_argument("--wait", action="store_true")
    g.add_argument("--summary", action="store_true")
    ap.add_argument("--smoke", default="results/task3/live/smoke")
    ap.add_argument("--preflight", default="results/task3/live/preflight")
    ap.add_argument("--run", default="results/task3/live/main")
    ap.add_argument("--log", default="notes/kaggle_logs/task3_live.log")
    ap.add_argument("--done", default="notes/kaggle_logs/task3_live.done")
    ap.add_argument("--exit", dest="exit_file", default="notes/kaggle_logs/task3_live.exit")
    ap.add_argument("--pid", dest="pid_file", default="notes/kaggle_logs/task3_live.pid")
    ap.add_argument("--every", type=float, default=600)
    ap.add_argument("--poll", type=float, default=30)
    ap.add_argument("--warn-h", type=float, default=11)
    ap.add_argument("--limit-h", type=float, default=10.5)
    a = ap.parse_args(argv)
    if a.plan:
        return plan(a.smoke, a.preflight, a.limit_h)
    if a.wait:
        return wait(a.run, a.log, a.done, a.exit_file, a.pid_file, a.every, a.poll, a.warn_h)
    return summary(a.run)


if __name__ == "__main__":
    sys.exit(main())
