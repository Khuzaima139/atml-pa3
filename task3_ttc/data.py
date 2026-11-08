"""Load and join the supplied Task 3 rollouts (manual L578-592, L630-631, L640-656; README R12-R16).

Each model's parquet files are joined on (dataset, sample_id, rollout_id). Every table is sorted by
(sample_id as a string, rollout_id); the join is accepted only if the keys are unique in every table,
identical across the tables (1:1), and form n_questions x n_rollouts with rollout_ids 0..n_rollouts-1.
Per rollout, len(generated_token_ids) must equal n_generated_text_tokens, len(logp), len(kl_u_p) and,
for MathVista, the length of every attention list.

Per-rollout summaries (float64):
    T       retained generated tokens
    capped  hit_max_new_tokens (supplied flag)
    L       mean logp                                   (L640-642)
    K       mean kl_u_p                                 (L640-642)
    V       mean of img / (img + prompt + generated)    (MathVista, L648-656)
Token-level kl_u_p (both) and v (MathVista) are cached as concatenated float32 arrays with row offsets.
"""
import json
import os

import numpy as np
import pyarrow.compute as pc
import pyarrow.parquet as pq
import yaml

from common.hashing import sha256_file, sha256_json
from common.runinfo import REPO_ROOT

CONFIG = "configs/task3/offline.yaml"
GEN_COLS = ["dataset", "sample_id", "rollout_id", "generated_token_ids", "n_generated_text_tokens",
            "hit_max_new_tokens", "gold_answer", "extracted_answer"]
PROB_COLS = ["dataset", "sample_id", "rollout_id", "logp", "kl_u_p"]
ATTN_LISTS = ["n_generated_text_tokens", "img_attn_total", "prompt_text_attn_total", "generated_text_attn_total"]
ATTN_COLS = ["dataset", "sample_id", "rollout_id"] + ATTN_LISTS


class DataError(AssertionError):
    """The supplied tables do not satisfy a join or alignment requirement."""


def resolve(path):
    return path if os.path.isabs(path) else os.path.join(REPO_ROOT, path)


def load_config(path=CONFIG):
    with open(resolve(path)) as f:
        cfg = yaml.safe_load(f)
    s = cfg["sizes"]
    assert s["n_dev"] + s["n_eval"] == s["n_questions"], "dev + eval must cover every question"
    return cfg


def dataset_files(cfg, dataset):
    d = cfg["datasets"][dataset]
    root = os.path.join(resolve(cfg["paths"]["data_root"]), d["folder"])
    return {k: os.path.join(root, v) for k, v in d["files"].items()}


def _check(cond, msg):
    if not cond:
        raise DataError(msg)


def _read_sorted(path, columns, dataset, name):
    """Read a table, check its dataset column, sort by (sample_id, rollout_id), check key uniqueness."""
    t = pq.read_table(path, columns=columns)
    _check(t.num_rows > 0, f"{name}: empty table")
    _check(pc.all(pc.equal(t.column("dataset"), dataset)).as_py(),
           f"{name}: dataset column is not {dataset!r} everywhere")
    sid = np.asarray(t.column("sample_id").to_pylist(), dtype=str)
    rid = t.column("rollout_id").to_numpy().astype(np.int64)
    order = np.lexsort((rid, sid))
    sid, rid = sid[order], rid[order]
    dup = (sid[1:] == sid[:-1]) & (rid[1:] == rid[:-1])
    _check(not dup.any(), f"{name}: {int(dup.sum())} duplicate (dataset, sample_id, rollout_id) keys")
    return t.take(order), sid, rid


def _list_len(t, col):
    return pc.list_value_length(t.column(col)).to_numpy().astype(np.int64)


def _flat(t, col, dtype):
    return pc.list_flatten(t.column(col)).to_numpy().astype(dtype)


def load_rollouts(cfg, dataset):
    """Join and validate one dataset's tables; return per-rollout arrays and token-level arrays."""
    files = dataset_files(cfg, dataset)
    n_q, n_r = cfg["sizes"]["n_questions"], cfg["sizes"]["n_rollouts"]

    gen, sid, rid = _read_sorted(files["generations"], GEN_COLS, dataset, "generations")
    prob, sid_p, rid_p = _read_sorted(files["probability"], PROB_COLS, dataset, "probability_metrics")
    tables = {"probability_metrics": (prob, sid_p, rid_p)}
    attn = None
    if "attention" in files:
        attn, sid_a, rid_a = _read_sorted(files["attention"], ATTN_COLS, dataset, "attention_metrics")
        tables["attention_metrics"] = (attn, sid_a, rid_a)
    for name, (_, s2, r2) in tables.items():
        _check(len(s2) == len(sid) and (s2 == sid).all() and (r2 == rid).all(),
               f"generations and {name} keys are not 1:1")

    n = len(sid)
    _check(n == n_q * n_r, f"{n} rollouts, expected {n_q} x {n_r}")
    S, R = sid.reshape(n_q, n_r), rid.reshape(n_q, n_r)
    _check((S == S[:, :1]).all() and len(np.unique(S[:, 0])) == n_q,
           f"expected {n_q} questions with {n_r} rollouts each")
    _check((R == np.arange(n_r)).all(), f"rollout_ids are not 0..{n_r - 1} for every question")

    T = _list_len(gen, "generated_token_ids")
    _check((T == gen.column("n_generated_text_tokens").to_numpy()).all(),
           "len(generated_token_ids) != n_generated_text_tokens")
    for col in ("logp", "kl_u_p"):
        _check((_list_len(prob, col) == T).all(), f"len({col}) != len(generated_token_ids)")
    if attn is not None:
        for col in ATTN_LISTS:
            _check((_list_len(attn, col) == T).all(), f"len(attention {col}) != len(generated_token_ids)")
    _check((T > 0).all(), "rollout with zero retained tokens")

    offsets = np.concatenate([[0], np.cumsum(T)]).astype(np.int64)
    starts = offsets[:-1]
    logp = _flat(prob, "logp", np.float64)
    kl32 = _flat(prob, "kl_u_p", np.float32)
    _check(np.isfinite(logp).all() and np.isfinite(kl32).all(), "non-finite logp or kl_u_p")
    L = np.add.reduceat(logp, starts) / T
    K = np.add.reduceat(kl32.astype(np.float64), starts) / T
    del logp

    v32 = V = None
    if attn is not None:
        img = _flat(attn, "img_attn_total", np.float64)
        den = img + _flat(attn, "prompt_text_attn_total", np.float64) + _flat(attn, "generated_text_attn_total",
                                                                                np.float64)
        _check(np.isfinite(den).all() and (den > 0).all(), "attention denominator not positive and finite")
        v = img / den
        V = np.add.reduceat(v, starts) / T
        v32 = v.astype(np.float32)
        del img, den, v

    gold_col = np.asarray(gen.column("gold_answer").to_pylist(), dtype=object).reshape(n_q, n_r)
    _check(all(len(set(row)) == 1 for row in gold_col), "gold_answer not constant within a question")
    capped = gen.column("hit_max_new_tokens").to_numpy(zero_copy_only=False).astype(bool)
    cap = cfg["token_cap"]

    return {
        "dataset": dataset,
        "sample_id": sid, "rollout_id": rid,
        "T": T, "capped": capped, "L": L, "K": K, "V": V,
        "extracted": gen.column("extracted_answer").to_pylist(),
        "gold": {S[i, 0]: gold_col[i, 0] for i in range(n_q)},
        "offsets": offsets, "kl_tok": kl32, "v_tok": v32,
        "input_sha256": {os.path.basename(os.path.dirname(p)) + "/" + os.path.basename(p): sha256_file(p)
                         for p in files.values()},
        "stats": {"n_rollouts": int(n), "n_questions": int(n_q), "n_tokens": int(T.sum()),
                  "capped": int(capped.sum()),
                  "capped_flag_vs_length_disagreements": int((capped != (T >= cap)).sum())},
    }


def write_token_cache(data, path):
    """Token-level kl_u_p (and v for MathVista) as concatenated float32 arrays plus row offsets."""
    meta = {"dataset": data["dataset"], "input_sha256": data["input_sha256"],
            "inputs_sha256": sha256_json(data["input_sha256"]),
            "row_order": "sample_id (string sort), rollout_id",
            "arrays": "row i uses kl_u_p[offsets[i]:offsets[i+1]] (and v)"}
    arrays = {"sample_id": data["sample_id"], "rollout_id": data["rollout_id"], "offsets": data["offsets"],
              "kl_u_p": data["kl_tok"], "meta": np.array(json.dumps(meta))}
    if data["v_tok"] is not None:
        arrays["v"] = data["v_tok"]
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "wb") as f:
        np.savez(f, **arrays)
    os.replace(tmp, path)
    return meta


def load_token_cache(path):
    z = np.load(path)
    out = {k: z[k] for k in z.files if k != "meta"}
    out["meta"] = json.loads(str(z["meta"]))
    return out
