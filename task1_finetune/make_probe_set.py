"""Fix the Task 1 activation-probe set (manual L151-153) before any model result is seen.

32 GSM8K test questions and 32 HellaSwag validation contexts, sampled with seed 6304:
    rng = np.random.default_rng(6304)
    gsm_ids = sorted(rng.choice(N_gsm, 32, replace=False))
    hs_ids  = sorted(rng.choice(N_hs, 32, replace=False))   # same rng, drawn second
Ids are 0-based row positions in the splits lm-eval loads for gsm8k_cot and hellaswag.

Run: python -m task1_finetune.make_probe_set
"""
import argparse
import json
import os

import numpy as np

from common.seed import DEFAULT_SEED

# Dataset sources as defined by the installed lm-eval task YAMLs (gsm8k_cot, hellaswag).
GSM8K = {"path": "openai/gsm8k", "name": "main", "split": "test", "lm_eval_task": "gsm8k_cot"}
HELLASWAG = {"path": "Rowan/hellaswag", "name": None, "split": "validation", "lm_eval_task": "hellaswag"}
N_PER_SOURCE = 32
OUT = "splits/task1/probe_ids.json"
RULE = ("rng = np.random.default_rng(seed); gsm_ids = sorted(rng.choice(N_gsm, 32, replace=False)); "
        "hs_ids = sorted(rng.choice(N_hs, 32, replace=False)) from the same rng; "
        "ids are 0-based row positions in the lm-eval splits")


def sample_probe_ids(n_gsm, n_hs, seed=DEFAULT_SEED, k=N_PER_SOURCE):
    rng = np.random.default_rng(seed)
    gsm_ids = sorted(int(i) for i in rng.choice(n_gsm, k, replace=False))
    hs_ids = sorted(int(i) for i in rng.choice(n_hs, k, replace=False))
    return gsm_ids, hs_ids


def split_size(src):
    from datasets import load_dataset
    return len(load_dataset(src["path"], src["name"], split=src["split"]))


def build(n_gsm, n_hs, seed=DEFAULT_SEED):
    gsm_ids, hs_ids = sample_probe_ids(n_gsm, n_hs, seed)
    return {"seed": seed, "rule": RULE,
            "gsm8k": {**GSM8K, "split_size": n_gsm, "ids": gsm_ids},
            "hellaswag": {**HELLASWAG, "split_size": n_hs, "ids": hs_ids}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=OUT)
    args = ap.parse_args()
    data = build(split_size(GSM8K), split_size(HELLASWAG))
    if os.path.exists(args.out):
        with open(args.out) as f:
            old = json.load(f)
        if old != data:
            raise SystemExit(f"{args.out} exists with different contents; refusing to overwrite")
        print(f"{args.out} unchanged")
        return
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(data, f, indent=2)
        f.write("\n")
    print(f"wrote {args.out}: gsm8k n={data['gsm8k']['split_size']} ids={data['gsm8k']['ids']}")
    print(f"  hellaswag n={data['hellaswag']['split_size']} ids={data['hellaswag']['ids']}")


if __name__ == "__main__":
    main()
