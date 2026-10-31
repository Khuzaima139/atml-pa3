"""Task 1 evaluation models: one construction path for all seven conditions (manual L145, L212-214).

Load the bf16 checkpoint. Full FT variants: replace every parameter with the FP32 safetensors. Adapter
conditions: per target W = W0 + (alpha/r) B A in FP32 (OPLoRA: the exported B_eff, A_eff). Then cast the
parameters to the final dtype. The RoPE inv_freq buffers stay FP32, as in training and fp16_check: the rotary
forward upcasts them, so an FP16 cast would only add rounding to every position's phase.

Checkpoint files come from a {file name: path} index written by find_checkpoints, which checks each sha256
against the train JSON that recorded it.
"""
import glob
import json
import os
import time

import torch
from safetensors.torch import load_file

from common.results import read_result
from task1_finetune.adapters import AdapterLinear, inject_adapters, leading_bases, target_linears
from task1_finetune.train import _sync, sha256_file

METHODS = ("full_ft", "l2sp", "lora4", "lora16", "oplora16")


def load_base(cfg):
    from transformers import AutoModelForCausalLM
    m = cfg["model"]
    return AutoModelForCausalLM.from_pretrained(m["name"], revision=m["revision"], dtype=torch.bfloat16)


def to_final(model, device, dtype):
    """Parameters to (device, dtype); buffers only change device."""
    for p in model.parameters():
        p.requires_grad_(False)
        p.data = p.data.to(device, dtype)
    return model.to(device)


def find_checkpoints(root, train_paths, out=None):
    """Locate every kept file of the train JSONs under `root`; returns ({name: path}, problems)."""
    index, problems = {}, []
    for tp in train_paths:
        for e in read_result(tp)["files"]:
            if e["kind"] == "recovery":
                continue
            name = os.path.basename(e["path"])
            hits = sorted(glob.glob(os.path.join(glob.escape(root), "**", name), recursive=True))
            good = [h for h in hits if sha256_file(h) == e["sha256"]]
            state = "OK " if good and len(good) == len(hits) else "BAD"
            print(f"{state} {name} {e['sha256'][:12]} found {len(hits)}: {hits}", flush=True)
            if state == "BAD":
                problems.append(name)
            else:
                index[name] = good[0]
    if out and not problems:
        os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
        with open(out, "w") as f:
            json.dump(index, f, indent=2)
    return index, problems


def resolve_condition(cfg, name):
    """What a condition name means, read from its train JSON (never restated in the eval config)."""
    if name == cfg["baseline"]:
        return {"name": name, "kind": "base"}
    method, update = name.rsplit("_u", 1)
    d = read_result(os.path.join(cfg["train_results"], f"train_{method}.json"))
    files = {e["kind"]: e for e in d["files"] if e["update"] == int(update) and e["kind"] != "recovery"}
    c = {"name": name, "method": method, "update": int(update), "train_run_id": d["run_id"]}
    if "model_weights_fp32" in files:
        return {**c, "kind": "full", "file": os.path.basename(files["model_weights_fp32"]["path"]),
                "sha256": files["model_weights_fp32"]["sha256"]}
    ad = d["config"]["adapter"]
    c.update(kind="adapter", adapter=ad, seed=d["config"]["seed"],
             file=os.path.basename(files["adapter_fp32"]["path"]), sha256=files["adapter_fp32"]["sha256"])
    if ad["project"]:
        c.update(eff_file=os.path.basename(files["adapter_effective_fp32"]["path"]),
                 eff_sha256=files["adapter_effective_fp32"]["sha256"])
    return c


def adapter_factors(cond, index):
    """(state, suffix) of the factors used for inference: projected ones for OPLoRA (L213)."""
    if cond["adapter"]["project"]:
        return load_file(index[cond["eff_file"]]), "_eff"
    return load_file(index[cond["file"]]), ""


@torch.no_grad()
def merge_adapters(model, state, suffix, scaling, device, final_dtype):
    """W <- W0 + (alpha/r) B A in FP32, in place; every target must change, also after the final cast."""
    n = 0
    for name, lin in target_linears(model):
        W0 = lin.weight.data.to(device, torch.float32)
        B = state[f"{name}.lora_B{suffix}"].to(device)
        A = state[f"{name}.lora_A{suffix}"].to(device)
        W = W0 + scaling * (B @ A)
        assert not torch.equal(W, W0), f"{name}: merge left the weight unchanged"
        assert not torch.equal(W.to(final_dtype), W0.to(final_dtype)), f"{name}: unchanged after the final cast"
        lin.weight.data = W
        n += 1
    return n


def build_model(cond, cfg, index, device, dtype=torch.float16):
    """The evaluation model of a condition: no adapter modules, eval mode, parameters in `dtype`."""
    model = load_base(cfg)
    info = {"parameter_dtype": str(dtype).replace("torch.", ""), "buffers": "float32 (RoPE inv_freq)"}
    if cond["kind"] == "full":
        sd = load_file(index[cond["file"]])
        named = dict(model.named_parameters())
        assert set(sd) == set(named), "full-model checkpoint does not match the unique parameters"
        for n, p in named.items():
            assert sd[n].dtype == torch.float32 and sd[n].shape == p.shape, n
            p.data = sd[n]
    elif cond["kind"] == "adapter":
        ad = cond["adapter"]
        state, suffix = adapter_factors(cond, index)
        _sync(device)
        t0 = time.perf_counter()
        n = merge_adapters(model, state, suffix, ad["alpha"] / ad["r"], device, dtype)
        _sync(device)
        info.update(merge_s=time.perf_counter() - t0, targets_merged=n, scaling=ad["alpha"] / ad["r"],
                    factors="B_eff, A_eff" if suffix else "B, A")
    to_final(model, device, dtype)
    assert model.lm_head.weight is model.model.embed_tokens.weight, "tied embeddings broken"
    assert not any(isinstance(m, AdapterLinear) for m in model.modules())
    return model.eval(), info


def build_unmerged(cond, cfg, index, device, dtype):
    """Backbone plus trained adapter modules as in training: frozen backbone in `dtype`, FP32 adapters,
    OPLoRA projecting in forward from bases recomputed in FP32 from W0; eval mode (dropout off)."""
    ad = cond["adapter"]
    model = load_base(cfg)
    bases = None
    if ad["project"]:
        bases = {n: leading_bases(lin.weight.detach().to(device, torch.float32), ad["k"])
                 for n, lin in target_linears(model)}
    to_final(model, device, dtype)
    mods = inject_adapters(model, ad["r"], ad["alpha"], ad["dropout"], cond["seed"], ad["project"], bases)
    state = load_file(index[cond["file"]])
    with torch.no_grad():
        for n, m in mods.items():
            m.lora_A.copy_(state[f"{n}.lora_A"])
            m.lora_B.copy_(state[f"{n}.lora_B"])
    return model.eval()
