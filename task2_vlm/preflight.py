"""LoRA preflight (Kaggle cell 1, before feature caching): the Stage 2 path on the real model in seconds.

Loads Qwen3-0.6B (FP16 on CUDA), builds the VLM, calls the real add_lora, checks the LoRA-only trainable count
(4,587,520 = Stage 2 total - Stage 1 total in the config), then one autocast forward and backward on a
2-example dummy batch with random visual features; the connector and every lora_B tensor must get finite,
nonzero gradients (B = 0 at init, so lora_A gradients are zero by construction). Memory is freed at the end.
No result file is written.

Run: python -m task2_vlm.preflight
"""
import gc

import torch

from task2_vlm import model as M
from task2_vlm import train as T
from task2_vlm.data import Encoder, collate


def run(cfg, device):
    tok = T.load_tokenizer(cfg)
    enc = Encoder(tok, cfg)
    vlm = M.VLM(cfg, M.load_qwen(cfg, device), device)
    vlm.add_lora()
    named = vlm.named_trainable()
    n_lora = sum(p.numel() for n, p in named if not n.startswith("connector."))
    exp = cfg["expected_trainable_params"]
    assert n_lora == exp[1] - exp[0], f"LoRA trainable {n_lora}, expected {exp[1] - exp[0]}"
    vlm.train()
    exs = [enc.example("How many dogs can you see?", "2", 0, "answer"),
           enc.example("Describe this image in one sentence.", "A dog runs on the grass.", 1, "caption")]
    b = {k: v.to(device) for k, v in collate(exs, enc.n_vis, enc.pad, "right").items()}
    g = torch.Generator().manual_seed(cfg["seed"])
    feats = (torch.randn(2, cfg["features"]["tokens"], cfg["features"]["dim"], generator=g) * 2.2).half().to(device)
    loss, _, _ = T.microbatch_loss(vlm, b, feats, cfg["loss"]["logit_chunk"], len(exs))
    loss.backward()
    assert torch.isfinite(loss), "non-finite preflight loss"
    for n, p in named:
        if n.startswith("connector.") or "lora_B" in n:
            assert p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum() > 0, f"no gradient: {n}"
    peak = torch.cuda.max_memory_allocated(device) / 2**30 if device.type == "cuda" else 0.0
    dtypes = {"backbone": str(next(vlm.lm.parameters()).dtype), "trainable": sorted({str(p.dtype) for _, p in named})}
    print(f"lora preflight OK: LoRA trainable {n_lora:,} (connector {exp[0]:,}), loss {float(loss):.4f}, "
          f"dtypes {dtypes}, peak {peak:.2f} GiB", flush=True)
    del vlm, named, b, feats, loss
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return n_lora


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=T.CONFIG)
    args = ap.parse_args(argv)
    return run(T.load_config(args.config), T.training_device())


if __name__ == "__main__":
    main()
