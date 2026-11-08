"""Task 3 Step 4 live generation: single / parallel / refinement on the 50 live MATH problems
(manual L616-626 settings, L699-719 Step 4; TASK3_FACTS items 12-14; design fixed by the planning chat, prompt c).

    python -m task3_ttc.live --out results/task3/live/<name> [--questions N] [--cap-scale F] [--model PATH]
    python -m task3_ttc.live --preflight --out results/task3/live/preflight

Per question q (position 0-49 in splits/task3/live_problems.json), five calls in this order, batch size 1, one
device, each preceded by torch.manual_seed(6304 * 1000 + 10 * q + call):
    call 0 single (3,072 new tokens), calls 1-3 p1, p2, p3 (1,024 each), call 4 refine (2,048) whose messages are
    [system, original user, assistant = p1's decoded text (special tokens skipped, appended even if truncated),
    user = review instruction (L711-712)].
Sampling: a fresh GenerationConfig (the checkpoint's generation_config.json has repetition_penalty 1.05, which must
not leak in): do_sample, temperature 0.7, top_p 0.8, top_k 20, repetition_penalty 1.0, eos [151645, 151643],
pad 151643. The effective config generate() uses and the logits processors it builds are recorded and asserted.

Outputs in --out: calls.jsonl (one line per call, written with flush + fsync as soon as the call ends; a rerun skips
(question, call) pairs already present) and run.json (rewritten after every question). --cap-scale (default 1) only
scales the caps for tests. --preflight runs the FP16 checks of the Kaggle preflight on the same code path.
"""
import argparse
import datetime
import importlib.metadata
import json
import os
import signal
import sys
import time

from common.hashing import sha256_json
from common.results import read_result, write_result
from common.runinfo import run_info
from task3_ttc.extract import last_boxed
from task3_ttc import live_problems

MODEL_ID = "Qwen/Qwen2.5-3B-Instruct"
REVISION = "aa8e72537993ba99e69dfaafa59ed015b17504d1"      # README R66-67
SYSTEM = "You are a helpful assistant."                     # L620
INSTRUCTION = "Think step by step inside <think> </think> tags, then give your final answer inside \\boxed{}."  # L621
REVIEW = ("Review the solution above. Identify and correct any errors; if it is correct, retain its answer. "
          "Give your final answer inside \\boxed{}.")       # L711-712
SAMPLING = {"do_sample": True, "temperature": 0.7, "top_p": 0.8, "top_k": 20, "repetition_penalty": 1.0}  # L619-620
EOS_IDS = [151645, 151643]                                  # <|im_end|>, <|endoftext|>
PAD_ID = 151643
CALLS = [(0, "single", 3072), (1, "p1", 1024), (2, "p2", 1024), (3, "p3", 1024), (4, "refine", 2048)]  # L702-707
SEED_BASE = 6304 * 1000
WARMUP_SEED = SEED_BASE + 999                               # outside 6,304,000..6,304,494 used by the calls
WARMUP_TOKENS = 16
EXPECTED_PROCESSORS = ["TemperatureLogitsWarper", "TopKLogitsWarper", "TopPLogitsWarper"]
KILL_ENV = "TASK3_LIVE_TEST_KILL_AFTER_CALLS"            # tests only: SIGKILL self after this many new calls
LIBS = ["torch", "transformers", "tokenizers", "accelerate", "safetensors", "huggingface_hub", "numpy"]


def call_seed(q, call):
    return SEED_BASE + 10 * q + call


def caps(cap_scale):
    return {s: max(1, int(round(c * cap_scale))) for _, s, c in CALLS}


def user_content(problem):
    return problem + "\n\n" + INSTRUCTION                   # README R58: two newlines, then the instruction


def build_messages(problem, draft=None):
    m = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user_content(problem)}]
    if draft is not None:
        m += [{"role": "assistant", "content": draft}, {"role": "user", "content": REVIEW}]
    return m


def render(tok, messages):
    return tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def lib_versions():
    out = {}
    for name in LIBS:
        try:
            out[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            out[name] = None
    return out


def torch_info(device):
    import torch
    info = {"torch": torch.__version__, "cuda": torch.version.cuda,
            "cudnn": torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else None,
            "device": str(device), "cuda_device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
            "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES")}
    if device.type == "cuda":
        p = torch.cuda.get_device_properties(device)
        info.update({"gpu_name": p.name, "gpu_capability": f"{p.major}.{p.minor}",
                     "gpu_total_memory_bytes": int(p.total_memory)})
    return info


# ----------------------------------------------------------------------------------------- model and config

def resolve_model(model):
    """(local path, snapshot commit). For the hub model the pinned revision is resolved with snapshot_download (a
    cached snapshot is not downloaded again) and the commit is the snapshot folder name, which must equal REVISION
    (transformers 5.x no longer sets config._commit_hash). A local --model path has no commit."""
    if model != MODEL_ID:
        return model, None
    from huggingface_hub import snapshot_download
    path = snapshot_download(MODEL_ID, revision=REVISION)
    commit = os.path.basename(os.path.normpath(path))
    if commit != REVISION:
        raise RuntimeError(f"resolved snapshot {commit} ({path}), expected {REVISION}")
    return path, commit


def load_model(model, device):
    """FP16 weights, no quantization, eval mode, on one device. Returns (model, tokenizer, info)."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    path, commit = resolve_model(model)
    t0 = time.perf_counter()
    tok = AutoTokenizer.from_pretrained(path)
    m = AutoModelForCausalLM.from_pretrained(path, dtype=torch.float16)
    m.to(device).eval()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    load_s = time.perf_counter() - t0
    dtypes = sorted({str(p.dtype) for p in m.parameters()})
    assert dtypes == ["torch.float16"], f"parameters not all float16: {dtypes}"
    assert getattr(m.config, "quantization_config", None) is None, "model is quantized"
    assert not m.training
    ids = {t: tok.convert_tokens_to_ids(t) for t in ("<|im_end|>", "<|endoftext|>")}
    assert [ids["<|im_end|>"], ids["<|endoftext|>"]] == EOS_IDS, ids
    info = {"source": model, "revision": REVISION if model == MODEL_ID else None, "snapshot_commit": commit,
            "load_path": path, "dtype": "float16", "param_dtypes": dtypes, "quantization": None,
            "n_parameters": int(sum(p.numel() for p in m.parameters())),
            "attn_implementation": getattr(m.config, "_attn_implementation", None),
            "model_class": type(m).__name__, "tokenizer_class": type(tok).__name__, "tokenizer_len": len(tok),
            "chat_template_sha256": sha256_json(tok.chat_template), "load_s": load_s}
    return m, tok, info


def generation_config(max_new_tokens):
    from transformers import GenerationConfig
    return GenerationConfig(**SAMPLING, eos_token_id=list(EOS_IDS), pad_token_id=PAD_ID,
                            max_new_tokens=int(max_new_tokens))


class ProcessorSpy:
    """Wraps model._get_logits_processor to record the processor classes generate() actually builds."""

    def __init__(self, model):
        self.last = None
        orig = model._get_logits_processor

        def wrapped(*a, **k):
            r = orig(*a, **k)
            self.last = [type(p).__name__ for p in r]
            return r
        model._get_logits_processor = wrapped


def effective_config(model, gen_cfg):
    """The config generate() will use after merging with model.generation_config; asserts the sampling values."""
    eff, _ = model._prepare_generation_config(gen_cfg)
    got = {k: getattr(eff, k) for k in SAMPLING}
    got.update({"eos_token_id": list(eff.eos_token_id) if isinstance(eff.eos_token_id, (list, tuple))
                else eff.eos_token_id, "pad_token_id": eff.pad_token_id, "max_new_tokens": eff.max_new_tokens})
    assert got["repetition_penalty"] == 1.0, f"effective repetition_penalty {got['repetition_penalty']}"
    for k, v in SAMPLING.items():
        assert got[k] == v, f"effective {k} = {got[k]}, expected {v}"
    assert got["eos_token_id"] == EOS_IDS and got["pad_token_id"] == PAD_ID, got
    assert got["max_new_tokens"] == gen_cfg.max_new_tokens
    extra = {k: getattr(eff, k, None) for k in ("min_p", "typical_p", "no_repeat_ngram_size", "num_beams",
                                                 "min_new_tokens", "max_length", "use_cache")}
    return {**got, "other": extra}


def check_processors(names):
    assert names is not None, "generate() built no logits processors (spy not called)"
    assert "RepetitionPenaltyLogitsProcessor" not in names, names
    assert sorted(names) == sorted(EXPECTED_PROCESSORS), f"unexpected logits processors {names}"


# ---------------------------------------------------------------------------------------------- one call

def _sync(device):
    if device.type == "cuda":
        import torch
        torch.cuda.synchronize(device)


def generate(model, tok, device, messages, max_new_tokens, seed, spy):
    """One timed call: seed, render, tokenize, generate, decode. Returns the measured fields."""
    import torch
    gen_cfg = generation_config(max_new_tokens)
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    _sync(device)
    t0 = time.perf_counter()
    prompt_text = render(tok, messages)
    input_ids = tok(prompt_text, add_special_tokens=False, return_tensors="pt")["input_ids"].to(device)
    n_in = int(input_ids.shape[1])
    spy.last = None
    with torch.inference_mode():
        out = model.generate(input_ids=input_ids, attention_mask=torch.ones_like(input_ids),
                             generation_config=gen_cfg)
    gen = out[0, n_in:].tolist()
    finished = len(gen) > 0 and gen[-1] in EOS_IDS
    kept = gen[:-1] if finished else gen
    text = tok.decode(kept, skip_special_tokens=True)
    _sync(device)
    wall = time.perf_counter() - t0
    check_processors(spy.last)
    assert not any(t in EOS_IDS for t in kept), "stop token inside the retained output"
    assert finished or len(gen) == max_new_tokens, f"stopped without EOS before the cap ({len(gen)})"
    r = {"prompt_text": prompt_text, "input_tokens": n_in, "output_token_ids": kept, "output_text": text,
         "output_tokens": len(kept), "stop_token_id": gen[-1] if finished else None,
         "finished_by_eos": bool(finished), "truncated": bool(not finished and len(gen) == max_new_tokens),
         "extracted_answer": last_boxed(text), "wall_s": wall,
         "peak_mem_allocated_bytes": None, "peak_mem_reserved_bytes": None}
    if device.type == "cuda":
        r["peak_mem_allocated_bytes"] = int(torch.cuda.max_memory_allocated(device))
        r["peak_mem_reserved_bytes"] = int(torch.cuda.max_memory_reserved(device))
    return r


# --------------------------------------------------------------------------------------- jsonl and resume

def read_calls(path):
    """Records of a calls.jsonl. A partial last line (process killed mid-write) is cut off; returns (records,
    bytes removed)."""
    if not os.path.exists(path):
        return [], 0
    with open(path, "rb") as f:
        data = f.read()
    good = data.rfind(b"\n") + 1
    removed = len(data) - good
    if removed:
        with open(path, "r+b") as f:
            f.truncate(good)
            f.flush()
            os.fsync(f.fileno())
    recs = [json.loads(x) for x in data[:good].decode("utf-8").splitlines() if x.strip()]
    return recs, removed


def append_call(path, rec):
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        f.flush()
        os.fsync(f.fileno())


def totals(recs, n_questions):
    by = {}
    for r in recs:
        s = by.setdefault(r["strategy"], {"calls": 0, "input_tokens": 0, "output_tokens": 0, "truncated": 0,
                                          "finished_by_eos": 0, "extraction_failures": 0, "wall_s": 0.0})
        s["calls"] += 1
        s["input_tokens"] += r["input_tokens"]
        s["output_tokens"] += r["output_tokens"]
        s["truncated"] += int(r["truncated"])
        s["finished_by_eos"] += int(r["finished_by_eos"])
        s["extraction_failures"] += int(r["extracted_answer"] is None)
        s["wall_s"] += r["wall_s"]
    per_q = {}
    for r in recs:
        per_q[r["q"]] = per_q.get(r["q"], 0) + 1
    peaks = [r["peak_mem_allocated_bytes"] for r in recs if r["peak_mem_allocated_bytes"] is not None]
    peaks_r = [r["peak_mem_reserved_bytes"] for r in recs if r["peak_mem_reserved_bytes"] is not None]
    return {"calls_done": len(recs), "calls_expected": n_questions * len(CALLS),
            "questions_done": sum(1 for v in per_q.values() if v == len(CALLS)), "questions_expected": n_questions,
            "input_tokens": sum(r["input_tokens"] for r in recs), "output_tokens": sum(r["output_tokens"] for r in recs),
            "wall_s_calls": sum(r["wall_s"] for r in recs),
            "peak_mem_allocated_bytes": max(peaks) if peaks else None,
            "peak_mem_reserved_bytes": max(peaks_r) if peaks_r else None, "by_strategy": by}


# -------------------------------------------------------------------------------------------------- run

def run_config(args, problems, cap_by):
    return {"task": "task3 step 4 live generation", "model": args.model,
            "revision": REVISION if args.model == MODEL_ID else None, "dtype": "float16", "quantization": None,
            "batch_size": 1, "device": args.device,
            "problems_file": "splits/task3/live_problems.json", "problems_sha256": problems["sha256"],
            "n_questions": args.questions, "sample_ids": [p["sample_id"] for p in problems["problems"][:args.questions]],
            "calls": [{"call": c, "strategy": s, "max_new_tokens": cap_by[s]} for c, s, _ in CALLS],
            "cap_scale": args.cap_scale, "sampling": SAMPLING, "eos_token_id": EOS_IDS, "pad_token_id": PAD_ID,
            "generation_config": "fresh transformers.GenerationConfig (checkpoint generation_config.json not used)",
            "seed_rule": "torch.manual_seed(6304*1000 + 10*q + call) before each call; q = position 0-49, "
                         "call 0 single, 1-3 p1-p3, 4 refine",
            "warmup": {"question": 0, "max_new_tokens": WARMUP_TOKENS, "seed": WARMUP_SEED, "saved": False},
            "system": SYSTEM, "instruction": INSTRUCTION, "user_join": "problem + '\\n\\n' + instruction",
            "review_instruction": REVIEW,
            "refine_draft": "p1 decoded with skip_special_tokens=True (stop token excluded), appended even if truncated",
            "extraction": "task3_ttc.extract.last_boxed (README R32 balanced-brace rule)",
            "timing": "perf_counter from prompt rendering to decoded text, cuda.synchronize at both ends; "
                      "model loading and the warm-up excluded",
            "peak_memory": "torch.cuda.reset_peak_memory_stats before each call; max_memory_allocated / reserved"}


def _now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def run(args):
    import torch
    device = torch.device(args.device)
    if device.type == "cuda":
        assert device.index in (None, 0), "only cuda:0 is used"
        device = torch.device("cuda:0")
    problems = live_problems.load(args.problems)
    assert 1 <= args.questions <= problems["n"]
    cap_by = caps(args.cap_scale)
    cfg = run_config(args, problems, cap_by)
    os.makedirs(args.out, exist_ok=True)
    calls_path, run_path = os.path.join(args.out, "calls.jsonl"), os.path.join(args.out, "run.json")

    t_session = time.time()
    session = {"started": _now(), "provenance": run_info(), "libraries": lib_versions(), "argv": sys.argv[1:]}
    if os.path.exists(run_path):
        prev = read_result(run_path)
        if prev["config"] != cfg:
            diff = sorted(k for k in set(cfg) | set(prev["config"]) if cfg.get(k) != prev["config"].get(k))
            raise SystemExit(f"{run_path} holds a run with a different config ({diff}); use another --out")
        run_id, sessions = prev["run_id"], prev["sessions"]
    else:
        if os.path.exists(calls_path):
            raise SystemExit(f"{calls_path} exists without run.json; refusing to mix runs")
        run_id = f"task3_live_{os.path.basename(os.path.normpath(args.out))}_" \
                 f"{datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
        sessions = []
    recs, cut = read_calls(calls_path)
    seen = set()
    for r in recs:
        assert r["run_id"] == run_id, f"calls.jsonl record from run {r['run_id']}"
        k = (r["q"], r["call"])
        assert k not in seen, f"duplicate record {k}"
        seen.add(k)
    session.update({"records_at_start": len(recs), "partial_bytes_removed": cut})
    sessions.append(session)
    print(f"run {run_id}: {len(recs)} calls already present, partial bytes removed {cut}", flush=True)

    model, tok, minfo = load_model(args.model, device)
    session["model"] = minfo
    session["torch"] = torch_info(device)
    model.generation_config = generation_config(1)          # nothing inherited from the hub generation_config.json
    effective = {s: effective_config(model, generation_config(c)) for s, c in cap_by.items()}
    spy = ProcessorSpy(model)
    print(f"model {minfo['source']} snapshot {minfo['snapshot_commit']} {minfo['param_dtypes']} on {device}; "
          f"load {minfo['load_s']:.1f} s; effective repetition_penalty "
          f"{effective['single']['repetition_penalty']}", flush=True)

    def write_run(status):
        tot = totals(recs, args.questions)
        session["elapsed_s"] = time.time() - t_session
        write_result(run_path, {"run_id": run_id, "status": status, "config": cfg,
                                "effective_generation_config": effective, "logits_processors": spy.last,
                                "sessions": sessions, "totals": tot, "updated": _now(),
                                "wall_s_sessions": sum(s.get("elapsed_s", 0.0) for s in sessions)})
        return tot

    todo = [(q, c) for q in range(args.questions) for c, _, _ in CALLS if (q, c) not in seen]
    if todo:
        w = generate(model, tok, device, build_messages(problems["problems"][0]["problem"]), WARMUP_TOKENS,
                     WARMUP_SEED, spy)
        session["warmup"] = {"seed": WARMUP_SEED, "output_tokens": w["output_tokens"], "wall_s": w["wall_s"],
                             "logits_processors": spy.last}
        print(f"warm-up: {w['output_tokens']} tokens in {w['wall_s']:.2f} s (not saved)", flush=True)
    write_run("running")

    by_key = {(r["q"], r["call"]): r for r in recs}
    kill_after, n_new = int(os.environ.get(KILL_ENV, "0")), 0
    for q in range(args.questions):
        p = problems["problems"][q]
        new = False
        for call, strategy, _ in CALLS:
            if (q, call) in by_key:
                continue
            draft = by_key[(q, 1)]["output_text"] if strategy == "refine" else None
            seed = call_seed(q, call)
            m = generate(model, tok, device, build_messages(p["problem"], draft), cap_by[strategy], seed, spy)
            rec = {"run_id": run_id, "sample_id": p["sample_id"], "q": q, "strategy": strategy, "call": call,
                   "seed": seed, "max_new_tokens": cap_by[strategy], **m,
                   "draft_call": 1 if strategy == "refine" else None,
                   "session": len(sessions) - 1, "timestamp": _now()}
            append_call(calls_path, rec)
            n_new += 1
            if kill_after and n_new == kill_after:
                os.kill(os.getpid(), signal.SIGKILL)
            recs.append(rec)
            by_key[(q, call)] = rec
            new = True
            mem = rec["peak_mem_allocated_bytes"]
            print(f"call q={q} {p['sample_id']} {strategy} seed={seed} in={rec['input_tokens']} "
                  f"out={rec['output_tokens']} eos={rec['finished_by_eos']} trunc={rec['truncated']} "
                  f"ans={rec['extracted_answer']!r} t={rec['wall_s']:.1f}s "
                  f"tok/s={rec['output_tokens'] / max(rec['wall_s'], 1e-9):.1f}"
                  + (f" peak={mem / 2**30:.2f}GiB" if mem is not None else ""), flush=True)
        if new:
            tot = write_run("running")
            print(f"question {q + 1}/{args.questions} done: calls {tot['calls_done']}/{tot['calls_expected']}, "
                  f"output tokens {tot['output_tokens']}, call time {tot['wall_s_calls'] / 60:.1f} min", flush=True)
    tot = write_run("complete")
    print(f"complete: {tot['calls_done']}/{tot['calls_expected']} calls, {tot['output_tokens']} output tokens, "
          f"{tot['wall_s_calls'] / 3600:.2f} h in calls", flush=True)
    return run_id


# --------------------------------------------------------------------------------------------- preflight

def filler_text(tok, problems, n_tokens):
    """About n_tokens of real problem text (the live problems concatenated, repeated as needed)."""
    parts, n, i = [], 0, 0
    while n < n_tokens:
        t = problems[i % len(problems)]["problem"]
        parts.append(t)
        n += len(tok(t + "\n\n", add_special_tokens=False)["input_ids"])
        i += 1
    return "\n\n".join(parts)


def preflight(args):
    """FP16 checks on the real code path: finite logits on the first live prompt and on a long refine-style prompt
    (draft of ~2,500 tokens), one 64-token generation, peak memory. Writes <out>/preflight.json."""
    import torch
    device = torch.device(args.device)
    problems = live_problems.load(args.problems)
    model, tok, minfo = load_model(args.model, device)
    model.generation_config = generation_config(1)
    spy = ProcessorSpy(model)
    p0 = problems["problems"][0]["problem"]
    out = {"run_id": f"task3_live_preflight_{datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ')}",
           "provenance": run_info(), "libraries": lib_versions(), "torch": torch_info(device), "model": minfo,
           "effective_generation_config": effective_config(model, generation_config(64)), "forwards": []}
    long_draft = filler_text(tok, problems["problems"], args.long_tokens)
    for name, msgs in [("first_live_prompt", build_messages(p0)), ("long_refine_prompt", build_messages(p0, long_draft))]:
        ids = tok(render(tok, msgs), add_special_tokens=False, return_tensors="pt")["input_ids"].to(device)
        times = []
        for _ in range(2):                                   # second pass timed without first-use overheads
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)
            _sync(device)
            t0 = time.perf_counter()
            with torch.inference_mode():
                logits = model(input_ids=ids).logits
            _sync(device)
            times.append(time.perf_counter() - t0)
        finite = bool(torch.isfinite(logits).all().item())
        f = {"name": name, "input_tokens": int(ids.shape[1]), "logits_dtype": str(logits.dtype),
             "logits_shape": list(logits.shape), "all_finite": finite,
             "max_abs_logit": float(logits.float().abs().max().item()), "forward_s": times,
             "prefill_tokens_per_s": ids.shape[1] / times[1],
             "peak_mem_allocated_bytes": int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else None}
        del logits
        out["forwards"].append(f)
        print(f"preflight forward {name}: {f['input_tokens']} tokens, logits {f['logits_dtype']} finite={finite}, "
              f"max|logit| {f['max_abs_logit']:.1f}, {times[1]:.3f} s", flush=True)
        assert finite, f"non-finite logits on {name}"
    g = generate(model, tok, device, build_messages(p0), 64, WARMUP_SEED, spy)
    out["generation"] = {k: g[k] for k in ("input_tokens", "output_tokens", "finished_by_eos", "truncated", "wall_s",
                                           "peak_mem_allocated_bytes", "peak_mem_reserved_bytes", "output_text")}
    out["generation"]["logits_processors"] = spy.last
    out["status"] = "ok"
    print(f"preflight generation: {g['output_tokens']} tokens in {g['wall_s']:.2f} s, processors {spy.last}", flush=True)
    if device.type == "cuda":
        print(f"preflight peak memory: allocated {g['peak_mem_allocated_bytes'] / 2**30:.2f} GiB, reserved "
              f"{g['peak_mem_reserved_bytes'] / 2**30:.2f} GiB (generation); max over forwards "
              f"{max(f['peak_mem_allocated_bytes'] for f in out['forwards']) / 2**30:.2f} GiB", flush=True)
    os.makedirs(args.out, exist_ok=True)
    path = os.path.join(args.out, "preflight.json")
    if os.path.exists(path):
        os.remove(path)                                      # a preflight is a check, rerun freely
    write_result(path, out)
    print(f"preflight OK -> {path}", flush=True)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--questions", type=int, default=50)
    ap.add_argument("--cap-scale", type=float, default=1.0)
    ap.add_argument("--model", default=MODEL_ID)
    ap.add_argument("--device", default=None)
    ap.add_argument("--problems", default=live_problems.OUT)
    ap.add_argument("--preflight", action="store_true")
    ap.add_argument("--long-tokens", type=int, default=2500)
    args = ap.parse_args(argv)
    if args.device is None:
        import torch
        args.device = "cuda:0" if torch.cuda.is_available() else "cpu"
    if args.preflight:
        preflight(args)
    else:
        run(args)


if __name__ == "__main__":
    main()
