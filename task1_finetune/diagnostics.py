"""Task 1 weight-space and activation diagnostics, in FP32 (manual L226-248, L269-271).

For down_proj in layers 0, 13, 27 and the six checkpoints: rho_k, eta (same bases for every checkpoint) and
R(z) on the starting model's down_proj inputs at the last non-padding prompt token of the 64 probe items.
Separately, numerical verification of OPLoRA's constraints on all target weights (L269-270).

Run (CUDA only; never on the Mac):
    python -m task1_finetune.diagnostics --ckpt_index results/task1/ckpt_index.json
"""
import argparse
import copy
import datetime
import json
import statistics
import time

import numpy as np
import torch
from safetensors import safe_open

from common.results import write_result
from common.runinfo import run_info
from common.seed import seed_everything
from task1_finetune import data as D
from task1_finetune.adapters import leading_bases, target_linears
from task1_finetune.build import load_base, resolve_condition, to_final
from task1_finetune.evaluate import eval_device, load_config, load_tasks, make_lm, probe_items


# ---------------------------------------------------------------- formulas

def fro(x):
    return torch.linalg.matrix_norm(x.double(), "fro").item()


def rho_k(U_k, dW):
    """||U_k U_k^T dW||_F^2 / ||dW||_F^2."""
    return (fro(U_k @ (U_k.T @ dW)) / fro(dW)) ** 2


def eta(dW, W0):
    """||dW||_F / ||W0||_F."""
    return fro(dW) / fro(W0)


def activation_ratio(dW, W0, z, eps):
    """R(z) = ||dW z||_2 / (||W0 z||_2 + eps) for each row of z (n, d_in)."""
    return torch.linalg.vector_norm(z @ dW.T, dim=1) / (torch.linalg.vector_norm(z @ W0.T, dim=1) + eps)


def constraint_ratios(U_k, V_k, dW):
    """(||U_k^T dW||_F / ||dW||_F, ||dW V_k||_F / ||dW||_F): both zero for an exact OPLoRA update."""
    return fro(U_k.T @ dW) / fro(dW), fro(dW @ V_k) / fro(dW)


# ---------------------------------------------------------------- activations

@torch.no_grad()
def capture_last_token(model, input_ids, attention_mask, layers, target="down_proj"):
    """Inputs of mlp.<target> at the last non-padding token of one prompt: {layer: z (d_in,)}."""
    assert input_ids.shape[0] == 1
    last = attention_mask[0].nonzero()[-1, 0]
    zs, hooks = {}, []

    def capture(layer):
        def hook(_, inputs, __):
            zs[layer] = inputs[0][0, last].detach().float()
        return hook

    for layer in layers:
        hooks.append(getattr(model.model.layers[layer].mlp, target).register_forward_hook(capture(layer)))
    try:
        model.model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
    finally:
        for h in hooks:
            h.remove()
    return zs


def cached_activations(model, items, layers, target, device):
    """{layer: tensor (n_items, d_in)}, one prompt at a time, no padding."""
    out = {layer: [] for layer in layers}
    for it in items:
        ids = torch.tensor([it["input_ids"]], device=device)
        zs = capture_last_token(model, ids, torch.ones_like(ids), layers, target)
        for layer in layers:
            out[layer].append(zs[layer])
    return {layer: torch.stack(v) for layer, v in out.items()}


# ---------------------------------------------------------------- updates

def weight_name(layer, target):
    return f"model.layers.{layer}.mlp.{target}"


def delta_weight(cond, index, name, W0):
    """dW in FP32: Full FT W_trained - W0, LoRA (alpha/r) B A, OPLoRA (alpha/r) B_eff A_eff."""
    dev = W0.device
    if cond["kind"] == "full":
        with safe_open(index[cond["file"]], "pt") as f:
            return f.get_tensor(f"{name}.weight").to(dev) - W0
    ad = cond["adapter"]
    file, suffix = (cond["eff_file"], "_eff") if ad["project"] else (cond["file"], "")
    with safe_open(index[file], "pt") as f:
        B, A = f.get_tensor(f"{name}.lora_B{suffix}"), f.get_tensor(f"{name}.lora_A{suffix}")
    return (ad["alpha"] / ad["r"]) * (B.to(dev) @ A.to(dev))


def stats(values):
    q25, med, q75 = np.percentile(values, [25, 50, 75])
    return {"mean": float(np.mean(values)), "median": float(med), "q25": float(q25), "q75": float(q75),
            "iqr": float(q75 - q25), "per_item": [float(v) for v in values]}


# ---------------------------------------------------------------- jobs

def weight_and_activation(cfg, model, items, index, device):
    dg = cfg["diagnostics"]
    target, layers, k, eps = dg["target"], dg["layers"], dg["k"], dg["epsilon"]
    W0 = {layer: getattr(model.model.layers[layer].mlp, target).weight.detach() for layer in layers}
    assert all(w.dtype == torch.float32 for w in W0.values())
    U_k = {layer: leading_bases(W0[layer], k)[0] for layer in layers}
    z = cached_activations(model, items, layers, target, device)
    domains = sorted({it["domain"] for it in items})
    mask = {d: torch.tensor([it["domain"] == d for it in items], device=device) for d in domains}
    out = {}
    for c in dg["checkpoints"]:
        cond = resolve_condition(cfg, c)
        out[c] = {}
        for layer in layers:
            dW = delta_weight(cond, index, weight_name(layer, target), W0[layer])
            R = activation_ratio(dW, W0[layer], z[layer], eps).cpu().numpy()
            out[c][str(layer)] = {
                "rho_k": rho_k(U_k[layer], dW), "eta": eta(dW, W0[layer]), "dW_fro": fro(dW),
                "W0_fro": fro(W0[layer]),
                "R": {d: stats(R[mask[d].cpu().numpy()]) for d in domains}}
    return out


def oplora_verification(cfg, model, index, device):
    """All target weights: constraint residuals of the exported factors (dense dW, same bases as training)."""
    cond = resolve_condition(cfg, "oplora16_u1250")
    k = cond["adapter"]["k"]
    rows = []
    with safe_open(index[cond["eff_file"]], "pt") as f:
        for name, lin in target_linears(model):
            B, A = f.get_tensor(f"{name}.lora_B_eff").to(device), f.get_tensor(f"{name}.lora_A_eff").to(device)
            dW = (cond["adapter"]["alpha"] / cond["adapter"]["r"]) * (B @ A)
            U, V = leading_bases(lin.weight.detach(), k)
            u, v = constraint_ratios(U, V, dW)
            rows.append({"target": name, "U_ratio": u, "V_ratio": v})
    agg = {}
    for key in ("U_ratio", "V_ratio"):
        vals = [r[key] for r in rows]
        agg[key] = {"max": max(vals), "median": statistics.median(vals)}
    return {"targets": len(rows), "k": k, "file": cond["eff_file"], "sha256": cond["eff_sha256"], "aggregate": agg,
            "per_target": rows}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/task1/eval.yaml")
    ap.add_argument("--ckpt_index", default="results/task1/ckpt_index.json")
    args = ap.parse_args(argv)
    cfg = copy.deepcopy(load_config(args.config))
    with open(args.ckpt_index) as f:
        index = json.load(f)
    device = eval_device()
    seed_everything(cfg["seed"])
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    o = cfg["outputs"]
    base = {"config": cfg, "provenance": run_info(), "dtype": "float32", "wall_s": 0.0, "status": "running"}
    doc = {**base, "run_id": f"diagnostics_{stamp}", "checkpoints": {}}
    ver = {**base, "run_id": f"oplora_verification_{stamp}"}
    write_result(o["diagnostics"], doc)
    write_result(o["oplora_verification"], ver)
    t0 = time.time()
    try:
        tok = D.load_tokenizer(cfg["model"]["revision"])
        model = to_final(load_base(cfg), device, torch.float32).eval()
        _, tasks = load_tasks(cfg)
        items = probe_items(make_lm(model, tok, cfg, 1), tasks, cfg)
        doc["probe"] = [{"domain": it["domain"], "id": it["id"], "tokens": len(it["input_ids"])} for it in items]
        doc["checkpoints"] = weight_and_activation(cfg, model, items, index, device)
        doc.update(wall_s=time.time() - t0, status="done", resources=_resources(device))
        write_result(o["diagnostics"], doc)
        t1 = time.time()
        ver.update(oplora_verification(cfg, model, index, device))
        ver.update(wall_s=time.time() - t1, status="done", resources=_resources(device))
        write_result(o["oplora_verification"], ver)
    except BaseException as e:
        for d, p in ((doc, o["diagnostics"]), (ver, o["oplora_verification"])):
            if d["status"] != "done":
                d.update(status="failed", error=repr(e))
                write_result(p, d)
        raise
    print("finished diagnostics; OPLoRA constraint ratios", ver["aggregate"], flush=True)


def _resources(device):
    if device.type != "cuda":
        return {"peak_allocated_bytes": 0, "peak_reserved_bytes": 0}
    return {"peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
            "peak_reserved_bytes": torch.cuda.max_memory_reserved(device)}


if __name__ == "__main__":
    main()
