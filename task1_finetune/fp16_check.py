"""One-off FP16 check of the Qwen3-0.6B checkpoint before training (manual L129-131 FP16 computation).

1. Weights: count checkpoint (bf16) values above the FP16 maximum (65504) and nonzero values below the FP16
   normal minimum (2**-14), and how many change under the bf16 -> fp16 cast.
2. Logits: base model in FP32 vs the FP16 backbone used by the adapter runs (FP16 parameters, FP32 RoPE
   buffers, FP16 autocast) on the first 4 training examples (full formatted sequences, one at a time):
   max/mean abs difference, inf/nan counts, argmax agreement.

Run (CUDA): python -m task1_finetune.fp16_check
"""
import argparse
import datetime
import json

import torch

from common.results import write_result
from common.runinfo import run_info
from task1_finetune import data as D

FP16_MAX = 65504.0
FP16_MIN_NORMAL = 2.0 ** -14
OUT = "results/task1/fp16_check.json"


@torch.no_grad()
def weight_ranges(model):
    rows, tot = [], {"numel": 0, "above_fp16_max": 0, "below_fp16_min_normal": 0, "changed_by_cast": 0}
    for n, p in model.named_parameters():
        w = p.detach().float()
        a = w.abs()
        r = {"name": n, "numel": w.numel(), "absmax": a.max().item(),
             "above_fp16_max": int((a > FP16_MAX).sum()),
             "below_fp16_min_normal": int(((a > 0) & (a < FP16_MIN_NORMAL)).sum()),
             "changed_by_cast": int((w.half().float() != w).sum())}
        for k in tot:
            tot[k] += r[k]
        if r["above_fp16_max"] or r["below_fp16_min_normal"] or r["changed_by_cast"]:
            rows.append(r)
    return tot, rows


@torch.no_grad()
def compare_logits(model32, model16, seqs, device):
    out = []
    for ids in seqs:
        x = torch.tensor([ids], device=device)
        l32 = model32(input_ids=x, use_cache=False).logits.float()
        with torch.autocast(device.type, dtype=torch.float16, enabled=device.type == "cuda"):
            l16 = model16(input_ids=x, use_cache=False).logits
        fin = torch.isfinite(l16)
        d = (l16.float() - l32).abs()
        out.append({"tokens": len(ids), "logits_dtype_fp16_run": str(l16.dtype),
                    "inf": int(torch.isinf(l16).sum()), "nan": int(torch.isnan(l16).sum()),
                    "max_abs_diff": d[fin].max().item(), "mean_abs_diff": d[fin].mean().item(),
                    "fp32_absmax": l32.abs().max().item(),
                    "argmax_agree": (l16.argmax(-1) == l32.argmax(-1)).float().mean().item()})
    return out


def fp16_backbone(model):
    """FP16 parameters, buffers (RoPE inverse frequencies) left in FP32, as in prepare_model."""
    for p in model.parameters():
        p.data = p.data.half()
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_examples", type=int, default=4)
    ap.add_argument("--out", default=OUT)
    args = ap.parse_args()
    from transformers import AutoModelForCausalLM
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tok = D.load_tokenizer()
    examples, _ = D.load_train_examples(tok)
    seqs = [e["input_ids"] for e in examples[:args.n_examples]]
    load = lambda: AutoModelForCausalLM.from_pretrained(D.MODEL_ID, revision=D.MODEL_REVISION, dtype=torch.bfloat16)
    m = load()
    tot, rows = weight_ranges(m)
    m32 = m.to(device=device, dtype=torch.float32).eval()
    m16 = fp16_backbone(load()).to(device).eval()
    logits = compare_logits(m32, m16, seqs, device)
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    doc = {"run_id": f"fp16_check_{stamp}", "provenance": run_info(), "model": D.MODEL_ID,
           "revision": D.MODEL_REVISION, "fp16_max": FP16_MAX, "fp16_min_normal": FP16_MIN_NORMAL,
           "weights_total": tot, "weights_flagged_tensors": rows,
           "logits_examples": "first %d training examples (train_ids order), full formatted sequences" % len(seqs),
           "logits": logits}
    write_result(args.out, doc)
    print(json.dumps({"weights_total": tot, "logits": logits}, indent=1))


if __name__ == "__main__":
    main()
