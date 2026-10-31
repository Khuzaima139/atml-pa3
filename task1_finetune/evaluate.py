"""Task 1 evaluation of one condition (manual L132-150, L163-172, L212-215, L249-257).

Stages: build the FP16 model; merge check (adapter conditions); GSM8K generation; likelihood benchmarks;
qualitative generation. One result JSON per condition, results/task1/eval/<condition>/scores.json.

Run (CUDA only; never on the Mac):
    python -m task1_finetune.evaluate --condition lora16_u1250 --ckpt_index results/task1/ckpt_index.json
    python -m task1_finetune.evaluate --condition baseline --smoke --limit 16 --qual_limit 2
"""
import argparse
import contextlib
import copy
import datetime
import gc
import gzip
import json
import math
import os
import time

import torch
import yaml

from common.results import write_result
from common.runinfo import run_info
from common.seed import seed_everything
from task1_finetune import data as D
from task1_finetune import qualitative as Q
from task1_finetune.build import build_model, build_unmerged, resolve_condition
from task1_finetune.tracking_lm import TrackingHFLM

FLEXIBLE, STRICT = "flexible-extract", "strict-match"
NO_EXTRACTION = "[invalid]"


def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)


def eval_device():
    """The only CUDA requirement of main(); the CPU end-to-end test replaces this function."""
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required (evaluation never runs on the Mac)")
    return torch.device("cuda")


def _dtype(name):
    return None if name is None else getattr(torch, name)


def make_lm(model, tok, cfg, batch_size, expected_doc0=None):
    return TrackingHFLM(pretrained=model, tokenizer=tok, batch_size=batch_size, expected_doc0=expected_doc0,
                        enable_thinking=cfg["model"]["enable_thinking"],
                        softmax_dtype=cfg["lm_eval"]["softmax_dtype"])


# ---------------------------------------------------------------- tasks, prompts and probe items

def load_tasks(cfg):
    from lm_eval.tasks import TaskManager
    tm = TaskManager()
    names = [cfg["gsm8k"]["task"], *cfg["likelihood"]["tasks"].values()]
    return tm, tm.load(names)["tasks"]


def dataset_info(task):
    """Resolved path, config, split, revision (from the prepared cache path) and document count."""
    c = task.config
    split = c.test_split or c.validation_split
    ds = task.dataset[split]
    cache = ds.cache_files[0]["filename"] if ds.cache_files else None
    return {"path": c.dataset_path, "name": c.dataset_name, "split": split, "docs": len(task.eval_docs),
            "revision": os.path.basename(os.path.dirname(cache)) if cache else None, "cache_file": cache,
            "fingerprint": ds._fingerprint}


def expected_gsm8k_prompt(tok, task):
    """apply_chat_template([user: lm-eval's doc_to_text(doc 0)], add_generation_prompt, enable_thinking=False)."""
    doc0 = task.eval_docs[0]
    return tok.apply_chat_template([{"role": "user", "content": task.doc_to_text(doc0)}],
                                   add_generation_prompt=True, tokenize=False, enable_thinking=False)


def render_gsm8k(task, doc, lm):
    """The context lm-eval builds for gsm8k_cot under our settings (zero-shot, chat template)."""
    return task.fewshot_context(doc, 0, apply_chat_template=True, fewshot_as_multiturn=True,
                                chat_template=lm.apply_chat_template, gen_prefix=task.doc_to_prefix(doc))


def load_probe(path):
    with open(path) as f:
        return json.load(f)


def probe_items(lm, tasks, cfg, n_per_domain=None):
    """Probe prompts as benchmark contexts, no continuation: [{domain, id, text, input_ids}] (L151-153, L239)."""
    probe = load_probe(cfg["probe_ids"])
    gsm, hs = tasks[cfg["gsm8k"]["task"]], tasks[cfg["likelihood"]["tasks"]["hellaswag"]]
    out = []
    for i in probe["gsm8k"]["ids"][:n_per_domain]:
        out.append({"domain": "gsm8k", "id": i, "text": render_gsm8k(gsm, gsm.eval_docs[i], lm)})
    for i in probe["hellaswag"]["ids"][:n_per_domain]:
        out.append({"domain": "hellaswag", "id": i, "text": hs.doc_to_text(hs.eval_docs[i])})
    for it in out:
        it["input_ids"] = lm.tok_encode(it["text"])
    return out


# ---------------------------------------------------------------- merge check

def _autocast(device, enabled):
    return torch.autocast(device.type, dtype=torch.float16) if enabled else contextlib.nullcontext()


@torch.no_grad()
def compare_models(unmerged, merged, items, device, unmerged_autocast):
    """Logits over all positions of each prompt, one prompt at a time: unmerged vs merged."""
    mx, total, n, agree, inf_nan = 0.0, 0.0, 0, 0, 0
    per_prompt = []
    for it in items:
        x = torch.tensor([it["input_ids"]], device=device)
        with _autocast(device, unmerged_autocast):
            a = unmerged(input_ids=x, use_cache=False).logits
        b = merged(input_ids=x, use_cache=False).logits
        inf_nan += int((~torch.isfinite(a)).sum()) + int((~torch.isfinite(b)).sum())
        d = (a.float() - b.float()).abs()
        same = (a.argmax(-1) == b.argmax(-1))
        mx = max(mx, d.max().item())
        total += d.double().sum().item()
        n += d.numel()
        agree += int(same.sum())
        per_prompt.append({"domain": it["domain"], "id": it["id"], "tokens": x.shape[1],
                           "max_abs_diff": d.max().item(), "argmax_agree": same.float().mean().item()})
    positions = sum(p["tokens"] for p in per_prompt)
    return {"max_abs_diff": mx, "mean_abs_diff": total / n, "argmax_agreement": agree / positions,
            "non_finite": inf_nan, "prompts": len(items), "positions": positions, "per_prompt": per_prompt}


def merge_check(cond, cfg, index, items, device, merged_fp16):
    """FP16: as in training vs the evaluation model. FP32: everything in FP32 (L214)."""
    t0 = time.perf_counter()
    un = build_unmerged(cond, cfg, index, device, torch.float16)
    fp16 = compare_models(un, merged_fp16, items, device, unmerged_autocast=True)
    fp16["compute"] = "unmerged: FP16 backbone, FP32 adapters, FP16 autocast; merged: FP16, no autocast"
    del un
    un = build_unmerged(cond, cfg, index, device, torch.float32)
    m32, _ = build_model(cond, cfg, index, device, torch.float32)
    fp32 = compare_models(un, m32, items, device, unmerged_autocast=False)
    fp32["compute"] = "unmerged and merged both FP32, no autocast"
    del un, m32
    _free(device)
    return {"fp16": fp16, "fp32": fp32, "prompt_ids": [(i["domain"], i["id"]) for i in items],
            "seconds": time.perf_counter() - t0}


# ---------------------------------------------------------------- per-example files

def write_jsonl_gz(path, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "wb") as raw, gzip.GzipFile(filename="", fileobj=raw, mode="wb", mtime=0) as f:
        for r in rows:
            f.write((json.dumps(r, default=str) + "\n").encode())
    os.replace(tmp, path)


def compact_gsm8k(samples, records):
    """One row per doc, joined to the tracking records by doc_id."""
    rec = {r["doc_id"]: r for r in records}
    assert len(rec) == len(records), "duplicate doc_id in generation records"
    by_doc = {}
    for s in samples:
        by_doc.setdefault(s["doc_id"], {})[s["filter"]] = s
    assert set(by_doc) == set(rec), "docs in samples and generation records differ"
    rows = []
    for d in sorted(by_doc):
        flex, strict, r = by_doc[d][FLEXIBLE], by_doc[d][STRICT], rec[d]
        response = flex["resps"][0][0]
        assert response == r["response"], f"doc {d}: tracked response differs from the logged sample"
        rows.append({"doc_id": d, "gold": flex["target"], "correct": int(flex["exact_match"]),
                     "response": response, "strict_answer": strict["filtered_resps"][0],
                     "flexible_answer": flex["filtered_resps"][0], "n_generated_tokens": r["n_generated_tokens"],
                     "stop_reason": r["stop_reason"], "truncated": r["truncated"]})
    return rows


def compact_likelihood(samples, primary):
    rows = []
    for s in sorted(samples, key=lambda x: x["doc_id"]):
        row = {"doc_id": s["doc_id"], "gold": s["target"], "correct": int(s[primary])}
        if "acc_norm" in s:
            row["acc"] = int(s["acc"])
        rows.append(row)
    return rows


def non_finite_loglikelihoods(samples):
    return sum(not math.isfinite(ll) for s in samples for ll, _ in s["filtered_resps"])


# ---------------------------------------------------------------- benchmarks

def _lm_eval_args(cfg, limit):
    s = cfg["seed"]
    return dict(limit=limit, log_samples=True, bootstrap_iters=cfg["lm_eval"]["bootstrap_iters"], random_seed=s,
                numpy_random_seed=s, torch_random_seed=s, fewshot_random_seed=s)


def _raw_path(root, condition, name):
    return os.path.join(root, condition, f"{name}.jsonl.gz")


def run_gsm8k(model, tok, cfg, tm, tasks, condition, eval_dir, raw_root, limit):
    from lm_eval.evaluator import simple_evaluate
    g = cfg["gsm8k"]
    task = tasks[g["task"]]
    lm = make_lm(model, tok, cfg, cfg["lm_eval"]["batch_size_gsm8k"], expected_gsm8k_prompt(tok, task))
    res = simple_evaluate(model=lm, tasks=[task], task_manager=tm, num_fewshot=g["num_fewshot"],
                          apply_chat_template=g["apply_chat_template"], gen_kwargs=dict(g["gen_kwargs"]),
                          **_lm_eval_args(cfg, limit))
    r, samples = res["results"][g["task"]], res["samples"][g["task"]]
    rows = compact_gsm8k(samples, lm.gen_records)
    write_jsonl_gz(os.path.join(eval_dir, "gsm8k.jsonl.gz"), rows)
    write_jsonl_gz(_raw_path(raw_root, condition, "gsm8k"), samples)
    score = {"primary_metric": f"exact_match,{FLEXIBLE}", "primary": r[f"exact_match,{FLEXIBLE}"],
             "primary_stderr": r[f"exact_match_stderr,{FLEXIBLE}"], "strict": r[f"exact_match,{STRICT}"],
             "strict_stderr": r[f"exact_match_stderr,{STRICT}"], "n": len(rows),
             "truncated": sum(x["truncated"] for x in rows),
             "no_flexible_extraction": sum(x["flexible_answer"] == NO_EXTRACTION for x in rows),
             "stop_reasons": {k: sum(x["stop_reason"] == k for x in rows) for k in ("eos", "stop_sequence", "cap")},
             "n_generated_tokens_mean": sum(x["n_generated_tokens"] for x in rows) / len(rows)}
    return score, lm.gen_settings, lm.timing, res["configs"][g["task"]].get("generation_kwargs")


def run_likelihood(model, tok, cfg, tm, tasks, condition, eval_dir, raw_root, limit):
    from lm_eval.evaluator import simple_evaluate
    lk = cfg["likelihood"]
    lm = make_lm(model, tok, cfg, cfg["lm_eval"]["batch_size_likelihood"])
    res = simple_evaluate(model=lm, tasks=[tasks[t] for t in lk["tasks"].values()], task_manager=tm,
                          num_fewshot=lk["num_fewshot"], apply_chat_template=False, **_lm_eval_args(cfg, limit))
    scores = {}
    for key, task_name in lk["tasks"].items():
        r, samples = res["results"][task_name], res["samples"][task_name]
        p = lk["primary"][key]
        rows = compact_likelihood(samples, p)
        write_jsonl_gz(os.path.join(eval_dir, f"{key}.jsonl.gz"), rows)
        write_jsonl_gz(_raw_path(raw_root, condition, key), samples)
        scores[key] = {"primary_metric": f"{p},none", "primary": r[f"{p},none"], "primary_stderr": r[f"{p}_stderr,none"],
                       "acc": r["acc,none"], "acc_stderr": r["acc_stderr,none"], "n": len(rows),
                       "non_finite_loglikelihoods": non_finite_loglikelihoods(samples)}
    return scores, lm.timing


# ---------------------------------------------------------------- smoke: memory on the longest requests

def _peak(device):
    return torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None


def _reset_peak(device):
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)


def _free(device):
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()


def _longest(lm, contexts, n):
    return sorted(contexts, key=lambda c: -len(lm.tok_encode(c)))[:n]


def worst_batches(model, tok, cfg, tasks, device):
    """The first (longest, padded) batch of the full run through lm-eval's own methods: the BoolQ contexts for
    each likelihood batch size, and GSM8K generation forced to the cap. Limit-16 smoke requests are not long."""
    out = {"likelihood": {}, "gsm8k": {}}
    bq = tasks[cfg["likelihood"]["tasks"]["boolq"]]
    contexts = [bq.doc_to_text(d) for d in bq.eval_docs]
    for b in cfg["smoke"]["likelihood_batch_sizes"]:
        lm = make_lm(model, tok, cfg, b)
        reqs = [((c, " yes"), *lm._encode_pair(c, " yes")) for c in _longest(lm, contexts, b)]
        _reset_peak(device)
        try:
            t0 = time.perf_counter()
            lm._loglikelihood_tokens(reqs, disable_tqdm=True)
            rec = {"fits": True, "seconds": time.perf_counter() - t0}
        except torch.OutOfMemoryError:
            rec = {"fits": False}
        rec.update(longest_tokens=max(len(r[1]) + len(r[2]) - 1 for r in reqs), peak_allocated_bytes=_peak(device))
        out["likelihood"][str(b)] = rec
        print(f"worst likelihood batch {b}: {rec}", flush=True)
        _free(device)
    g = tasks[cfg["gsm8k"]["task"]]
    b, cap = cfg["lm_eval"]["batch_size_gsm8k"], cfg["gsm8k"]["gen_kwargs"]["max_gen_toks"]
    lm = make_lm(model, tok, cfg, b)
    ctx = _longest(lm, [render_gsm8k(g, d, lm) for d in g.eval_docs], b)
    ids, mask = lm.tok_batch_encode(ctx)
    ids, mask = ids.to(device), mask.to(device)
    _reset_peak(device)
    try:
        t0 = time.perf_counter()
        with torch.no_grad():
            _generate_to_cap(lm, ids, mask, cap)
        s = time.perf_counter() - t0
        rec = {"fits": True, "seconds": s, "steps": cap, "s_per_step": s / cap}
    except torch.OutOfMemoryError:
        rec = {"fits": False}
    rec.update(batch_size=b, context_tokens=ids.shape[1], peak_allocated_bytes=_peak(device))
    out["gsm8k"] = rec
    print(f"worst gsm8k batch: {rec}", flush=True)
    _free(device)
    return out


def _generate_to_cap(lm, ids, mask, cap):
    """Stock generate of the LM, EOS disabled by min_new_tokens so every row reaches the cap."""
    from lm_eval.models.huggingface import HFLM
    return HFLM._model_generate(lm, ids, ids.shape[1] + cap, [], attention_mask=mask, do_sample=False,
                                min_new_tokens=cap)


def project(doc, cfg, tasks):
    """Projected seconds for one full condition from this smoke run: likelihood by requests; GSM8K the larger of
    measured s/batch and every batch running to the cap; qualitative every prompt running to the cap."""
    t = doc["timing"]
    req_full = 0
    for name in cfg["likelihood"]["tasks"].values():
        task = tasks[name]
        req_full += sum(len(task.doc_to_choice(d)) for d in task.eval_docs)
    ll = t["loglikelihood"]["seconds"] / t["loglikelihood"]["requests"] * req_full
    b = cfg["lm_eval"]["batch_size_gsm8k"]
    batches = math.ceil(len(tasks[cfg["gsm8k"]["task"]].eval_docs) / b)
    measured = t["generate_until"]["seconds"] / t["generate_steps"]["batches"] * batches
    worst = doc["worst_batches"]["gsm8k"]
    cap = cfg["gsm8k"]["gen_kwargs"]["max_gen_toks"]
    at_cap = worst.get("s_per_step", 0.0) * cap * batches
    q = t["qualitative"]
    n_q = len(Q.load_prompts(cfg["qualitative"]["path"], cfg["qualitative"]["sha256"]))
    qual = q["seconds"] / max(q["tokens"], 1) * cfg["qualitative"]["max_new_tokens"] * n_q
    setup = doc["stages_s"]["build"] + doc["stages_s"].get("merge_check", 0.0)
    return {"loglikelihood_s": ll, "loglikelihood_requests_full": req_full, "gsm8k_measured_s": measured,
            "gsm8k_all_batches_at_cap_s": at_cap, "qualitative_all_at_cap_s": qual, "setup_s": setup,
            "total_s": ll + max(measured, at_cap) + qual + setup}


# ---------------------------------------------------------------- one condition

class Stages:
    """Wall-clock and peak memory per stage, in the result JSON."""

    def __init__(self, doc, device, write):
        self.doc, self.device, self.write = doc, device, write
        self.t_start = time.time()

    @contextlib.contextmanager
    def __call__(self, name):
        _reset_peak(self.device)
        t0 = time.perf_counter()
        yield
        self.doc["stages_s"][name] = time.perf_counter() - t0
        res = self.doc["resources"]
        res["wall_s"] = time.time() - self.t_start
        if self.device.type == "cuda":
            alloc, reserved = torch.cuda.max_memory_allocated(self.device), torch.cuda.max_memory_reserved(self.device)
            res["peak_by_stage_bytes"][name] = alloc
            res["peak_allocated_bytes"] = max(res["peak_allocated_bytes"], alloc)
            res["peak_reserved_bytes"] = max(res["peak_reserved_bytes"], reserved)
        self.write("running")


def evaluate_condition(cfg, condition, index, device, smoke=False, limit=None, qual_limit=None):
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    kind = "eval_smoke" if smoke else "eval"
    run_id = f"{kind}_{condition}_{stamp}"
    o = cfg["outputs"]
    if smoke:
        eval_root, raw_root = o["smoke_root"], os.path.join(o["raw_samples_root"], "smoke")
        qual_root = os.path.join(o["smoke_root"], "qualitative")
    else:
        eval_root, raw_root, qual_root = o["eval_root"], o["raw_samples_root"], o["qualitative_root"]
    eval_dir = os.path.join(eval_root, condition)
    path = os.path.join(eval_dir, "scores.json")
    cond = resolve_condition(cfg, condition)
    doc = {"run_id": run_id, "kind": kind, "status": "running", "condition": condition, "resolved": cond,
           "config": cfg, "limit": limit, "qual_limit": qual_limit, "provenance": run_info(),
           "dtype": {"evaluation": "float16 parameters, float32 RoPE buffers, no autocast, dropout off"},
           "resources": {"wall_s": 0.0, "peak_allocated_bytes": 0, "peak_reserved_bytes": 0, "peak_by_stage_bytes": {}},
           "stages_s": {}, "model": None, "merge_check": None, "datasets": {}, "generation": {}, "scores": {},
           "timing": {}}

    def write(status):
        doc["status"] = status
        write_result(path, doc)

    write("running")
    stage = Stages(doc, device, write)
    try:
        seed_everything(cfg["seed"])
        tok = D.load_tokenizer(cfg["model"]["revision"])
        with stage("build"):
            model, doc["model"] = build_model(cond, cfg, index, device)
        tm, tasks = load_tasks(cfg)
        doc["datasets"] = {t: dataset_info(task) for t, task in tasks.items()}
        if cond["kind"] == "adapter":
            with stage("merge_check"):
                items = probe_items(make_lm(model, tok, cfg, 1), tasks, cfg, cfg["merge_check"]["n_per_domain"])
                doc["merge_check"] = merge_check(cond, cfg, index, items, device, model)
        with stage("gsm8k"):
            score, gen, timing, task_gen = run_gsm8k(model, tok, cfg, tm, tasks, condition, eval_dir, raw_root, limit)
            doc["scores"]["gsm8k"] = score
            doc["generation"]["gsm8k"] = {**gen, "task_generation_kwargs": task_gen, "enable_thinking": False}
            doc["timing"].update(timing)
        with stage("likelihood"):
            scores, timing = run_likelihood(model, tok, cfg, tm, tasks, condition, eval_dir, raw_root, limit)
            doc["scores"].update(scores)
            doc["timing"].update(timing)
        with stage("qualitative"):
            meta = {"run_id": f"qualitative_{run_id}", "condition": condition, "config": cfg["qualitative"],
                    "provenance": doc["provenance"], "dtype": doc["dtype"]}
            doc["timing"]["qualitative"] = Q.run(model, tok, cfg, os.path.join(qual_root, f"{condition}.json"), meta,
                                                 qual_limit)
        if smoke:
            with stage("worst_batches"):
                doc["worst_batches"] = worst_batches(model, tok, cfg, tasks, device)
            doc["projection"] = project(doc, cfg, tasks)
        write("done")
    except BaseException as e:
        doc["error"] = repr(e)
        write("failed")
        raise
    return doc


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--condition", required=True)
    ap.add_argument("--config", default="configs/task1/eval.yaml")
    ap.add_argument("--ckpt_index", default="results/task1/ckpt_index.json")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--qual_limit", type=int)
    ap.add_argument("--max_gen_toks", type=int)
    args = ap.parse_args(argv)
    cfg = copy.deepcopy(load_config(args.config))
    assert args.condition in cfg["conditions"], args.condition
    if args.max_gen_toks:
        cfg["gsm8k"]["gen_kwargs"]["max_gen_toks"] = args.max_gen_toks
    device = eval_device()
    index = {}
    if args.condition != cfg["baseline"]:
        with open(args.ckpt_index) as f:
            index = json.load(f)
    doc = evaluate_condition(cfg, args.condition, index, device, args.smoke, args.limit, args.qual_limit)
    print(f"finished {doc['kind']} {args.condition}: " + json.dumps(
        {k: round(v["primary"], 4) for k, v in doc["scores"].items()}), flush=True)


if __name__ == "__main__":
    main()
