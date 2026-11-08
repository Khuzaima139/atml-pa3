"""Tiny synthetic Task 3 data (parquet files, MathVista metadata, config) for the Task 3 tests."""
import json
import os

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import yaml

from task3.data import load_config

N_ROLLOUTS = 4
CAP = 8

# MATH: sample_id -> (gold, extracted answers in rollout order)
MATH_Q = {
    "test:10": ("\\frac{1}{2}", ["\\frac{1}{2}", "0.5", None, "3"]),
    "test:2": ("5\\sqrt{2}", ["5\\sqrt{2}", "5", "5", ""]),
    "test:3": ("7", ["7", "7", "7.0", "\\text{seven}"]),
    "test:4": ("x+1", ["1+x", "x+1", "x", None]),
    "test:5": ("10", ["10", "10", "10", "10"]),
    "test:6": ("\\pi", ["3.14", "\\pi", None, None]),
}

# MathVista: pid -> metadata, gold (supplied form), extracted answers in rollout order
MV_Q = {
    "1": ({"choices": ["red", "blue", "green"], "question_type": "multi_choice", "answer_type": "text",
           "precision": None}, "B", ["B", "(C) green", "blue", "bleu"]),
    "2": ({"choices": None, "question_type": "free_form", "answer_type": "integer", "precision": None},
          "5", ["5", "5.0", "5-10 years", None]),
    "3": ({"choices": None, "question_type": "free_form", "answer_type": "float", "precision": 2},
          "1.46", ["1.456", "1.46", "1.4", "x"]),
    "4": ({"choices": ["9", "12", "18", "18"], "question_type": "multi_choice", "answer_type": "text",
           "precision": None}, "C", ["C", "D", "18", ""]),
    "5": ({"choices": None, "question_type": "free_form", "answer_type": "list", "precision": None},
          "[2014, 2016]", ["[2014, 2016]", "[2014,2016]", None, "[2014, 2016]"]),
    "6": ({"choices": ["yes", "no"], "question_type": "multi_choice", "answer_type": "text",
           "precision": None}, "A", ["A", "F", "Yes", "no"]),
}


def _rows(dataset, questions, rng, vlm):
    gen, prob, attn = [], [], []
    for qi, (sid, (gold, ext)) in enumerate(questions.items()):
        for r in range(N_ROLLOUTS):
            T = CAP if (qi == 0 and r == 0) else int(rng.integers(1, CAP))
            ids = rng.integers(0, 1000, T).astype(np.int32).tolist()
            gen.append({"dataset": dataset, "model": "synthetic", "sample_id": sid, "rollout_id": r,
                        "question_text": f"q {sid}", "generated_token_ids": ids, "n_generated_text_tokens": T,
                        "hit_max_new_tokens": T == CAP, "generated_text": "t", "gold_answer": gold,
                        "extracted_answer": ext[r]})
            prob.append({"dataset": dataset, "sample_id": sid, "rollout_id": r,
                         "logp": (-rng.random(T)).astype(np.float32).tolist(),
                         "kl_u_p": (10 + 5 * rng.random(T)).astype(np.float32).tolist()})
            if vlm:
                attn.append({"dataset": dataset, "sample_id": sid, "rollout_id": r, "n_image_tokens": 6,
                             "n_prompt_text_tokens": 4, "n_generated_text_tokens": list(range(T)),
                             "img_attn_total": (0.1 * rng.random(T) + 0.01).astype(np.float32).tolist(),
                             "prompt_text_attn_total": (0.2 * rng.random(T) + 0.01).astype(np.float32).tolist(),
                             "generated_text_attn_total": (0.3 * rng.random(T)).astype(np.float32).tolist()})
    return gen, prob, attn


def _write(rows, path, order):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    pq.write_table(pa.Table.from_pylist([rows[i] for i in order]), path)


def make_tables(dataset, seed=0):
    rng = np.random.default_rng(seed)
    q = MATH_Q if dataset == "math" else MV_Q
    q = {sid: (v[0], v[1]) for sid, v in q.items()} if dataset == "math" else {sid: (v[1], v[2]) for sid, v in q.items()}
    return _rows(dataset, q, rng, dataset == "mathvista")


def build(tmp, mutate=None, seed=0):
    """Write synthetic parquet files for both datasets, a metadata file and a config; return the config path.

    mutate(dataset, gen, prob, attn) may alter the row lists before writing (for failure cases).
    """
    root = os.path.join(tmp, "data")
    for ds, folder in (("math", "qwen25-math"), ("mathvista", "qwen25vl-mathvista")):
        gen, prob, attn = make_tables(ds, seed)
        if mutate:
            mutate(ds, gen, prob, attn)
        rng = np.random.default_rng(1)
        _write(gen, os.path.join(root, folder, "generations.parquet"), rng.permutation(len(gen)))
        _write(prob, os.path.join(root, folder, "probability_metrics.parquet"), rng.permutation(len(prob)))
        if attn:
            _write(attn, os.path.join(root, folder, "attention_metrics.parquet"), rng.permutation(len(attn)))
    meta_path = os.path.join(tmp, "testmini.json")
    with open(meta_path, "w") as f:
        json.dump({pid: {**m, "pid": pid, "answer": (m["choices"][ord(g) - 65] if m["choices"] else g)}
                   for pid, (m, g, _) in MV_Q.items()}, f)
    cfg = load_config()
    cfg["paths"] = {"data_root": root, "cache_dir": os.path.join(tmp, "cached"),
                    "splits_dir": os.path.join(tmp, "splits"), "results_dir": os.path.join(tmp, "results"),
                    "splits_preview": None}
    cfg["sizes"] = {"n_questions": 6, "n_rollouts": N_ROLLOUTS, "n_dev": 2, "n_eval": 4}
    cfg["datasets"]["math"]["n_live"] = 2
    cfg["datasets"]["mathvista"]["metadata"] = meta_path
    cfg["scorer"]["mathvista"]["testmini_sha256"] = None
    cfg["token_cap"] = CAP
    cfg_path = os.path.join(tmp, "offline.yaml")
    with open(cfg_path, "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
    return cfg_path
