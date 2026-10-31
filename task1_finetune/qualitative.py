"""Task 1 qualitative generation (manual L153-155, L249-257).

All 24 prompts of first_plot_questions.yaml: eight questions in plain, JSON and template formats. One greedy
response per prompt with thinking disabled and a 512-new-token cap, batch 1, on the evaluation model.
stop reason: "eos" (<|im_end|>) or "cap"; truncated = cap reached with no <|im_end|>.
"""
import hashlib
import os
import time

import torch
import yaml

from common.results import read_result, write_result
from task1_finetune import data as D
from task1_finetune.tracking_lm import effective_generation

FORMATS = ("plain", "json", "template")
SUFFIX = {"json": "_json", "template": "_template"}


def load_prompts(path, sha256):
    with open(path, "rb") as f:
        raw = f.read()
    got = hashlib.sha256(raw).hexdigest()
    assert got == sha256, f"{path}: sha256 {got}, expected {sha256}"
    out = []
    for q in yaml.safe_load(raw):
        assert len(q["paraphrases"]) == 1, q["id"]
        fmt = next((f for f, s in SUFFIX.items() if q["id"].endswith(s)), "plain")
        out.append({"id": q["id"], "question": q["id"].removesuffix(SUFFIX.get(fmt, "")), "format": fmt,
                    "system": q.get("system"), "user": q["paraphrases"][0]})
    return out


def render(tok, item):
    messages = ([{"role": "system", "content": item["system"]}] if item["system"] else []) + \
               [{"role": "user", "content": item["user"]}]
    return tok.apply_chat_template(messages, add_generation_prompt=True, tokenize=False, enable_thinking=False)


def generation_kwargs(max_new_tokens):
    """temperature, top_p and top_k are None to override generation_config.json explicitly."""
    return dict(do_sample=False, temperature=None, top_p=None, top_k=None, max_new_tokens=max_new_tokens,
                eos_token_id=D.IM_END_ID, pad_token_id=D.PAD_ID, use_cache=True)


@torch.no_grad()
def generate_one(model, tok, item, max_new_tokens):
    prompt = render(tok, item)
    enc = tok(prompt, add_special_tokens=False, return_tensors="pt").to(model.device)
    out = model.generate(**enc, **generation_kwargs(max_new_tokens))
    new = out[0, enc["input_ids"].shape[1]:].tolist()
    ended = D.IM_END_ID in new
    return {"id": item["id"], "question": item["question"], "format": item["format"], "prompt": prompt,
            "response": tok.decode(new, skip_special_tokens=True), "n_tokens": len(new),
            "stop_reason": "eos" if ended else "cap", "truncated": not ended}


def run(model, tok, cfg, path, meta, limit=None):
    """Generate one condition's responses; rewrites `path` after every prompt."""
    q = cfg["qualitative"]
    items = load_prompts(q["path"], q["sha256"])[:limit]
    doc = {**meta, "generation": {"kwargs": generation_kwargs(q["max_new_tokens"]),
                                  "effective": effective_generation(model, **generation_kwargs(q["max_new_tokens"])),
                                  "batch_size": 1, "enable_thinking": False},
           "prompts_sha256": q["sha256"], "status": "running", "responses": []}
    t0 = time.perf_counter()
    for it in items:
        doc["responses"].append(generate_one(model, tok, it, q["max_new_tokens"]))
        doc["wall_s"] = time.perf_counter() - t0
        write_result(path, doc)
    doc["status"] = "done"
    write_result(path, doc)
    return {"prompts": len(items), "seconds": doc["wall_s"], "tokens": sum(r["n_tokens"] for r in doc["responses"]),
            "truncated": sum(r["truncated"] for r in doc["responses"])}


def collect(cfg, root=None, out=None):
    """all_168.json: {question: {format: {condition: response record}}} in file order, then condition order."""
    root = root or cfg["outputs"]["qualitative_root"]
    out = out or os.path.join(root, "all_168.json")
    q = cfg["qualitative"]
    conds = cfg["conditions"]
    items = load_prompts(q["path"], q["sha256"])
    docs = {c: read_result(os.path.join(root, f"{c}.json")) for c in conds}
    by_cond = {c: {r["id"]: r for r in d["responses"]} for c, d in docs.items()}
    grouped = {}
    for it in sorted(items, key=lambda i: FORMATS.index(i["format"])):
        fmts = grouped.setdefault(it["question"], {})
        fmts[it["format"]] = {c: {k: v for k, v in by_cond[c][it["id"]].items() if k not in ("question", "format")}
                              for c in conds}
    n = sum(len(cs) for fm in grouped.values() for cs in fm.values())
    assert n == len(items) * len(conds), f"{n} responses"
    for fm in grouped.values():
        for cs in fm.values():
            assert len({r["prompt"] for r in cs.values()}) == 1, "prompts differ across conditions"
    write_result(out, {"run_id": "qualitative_all", "conditions": conds, "n_responses": n,
                       "prompts_sha256": q["sha256"], "source_run_ids": {c: d["run_id"] for c, d in docs.items()},
                       "questions": grouped})
    return n
