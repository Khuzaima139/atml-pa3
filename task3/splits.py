"""Task 3 development/evaluation splits and rollout orders (manual L593-596, L608-610).

Per dataset:
    split   sort the sample_ids as strings; perm = np.random.default_rng(6304).permutation(sorted ids);
            dev = perm[:n_dev], eval = perm[n_dev:]; MATH live subset = eval[:n_live].
    orders  a second, separate np.random.default_rng(6304); for each question in sorted-ID order,
            rng.permutation(n_rollouts) drawn n_orders times -> (n_questions, n_orders, n_rollouts).
The split file's sha256 is taken over the canonical JSON (common.hashing.canonical_json) of every other key.

Run: python -m task3.splits --dataset {math,mathvista}   (reads the sample_ids from the generations file)
"""
import argparse
import json
import os

import numpy as np
import pyarrow.parquet as pq

from common.hashing import sha256_json
from task3.data import dataset_files, load_config, resolve

GENERATOR = ("split: np.random.default_rng({seed}).permutation(sorted sample_ids as strings), first {n_dev} dev, "
             "remaining {n_eval} eval{live}; orders: separate np.random.default_rng({seed}), for each question in "
             "sorted-ID order rng.permutation({n_rollouts}) drawn {n_orders} times")


class SplitMismatchError(RuntimeError):
    pass


def make_split(sample_ids, seed, n_dev, n_eval, n_live, n_orders, n_rollouts):
    sorted_ids = sorted(set(sample_ids))
    assert len(sorted_ids) == n_dev + n_eval, f"{len(sorted_ids)} ids, expected {n_dev + n_eval}"
    perm = np.random.default_rng(seed).permutation(np.array(sorted_ids)).tolist()
    dev, ev = perm[:n_dev], perm[n_dev:]
    rng = np.random.default_rng(seed)
    orders = np.stack([np.stack([rng.permutation(n_rollouts) for _ in range(n_orders)]) for _ in sorted_ids])
    return {"sorted_ids": sorted_ids, "dev_ids": dev, "eval_ids": ev,
            "live_ids": ev[:n_live] if n_live else None, "orders": orders}


def split_content(dataset, split, seed, n_dev, n_eval, n_live, n_orders, n_rollouts):
    content = {
        "dataset": dataset,
        "generator": GENERATOR.format(seed=seed, n_dev=n_dev, n_eval=n_eval, n_rollouts=n_rollouts,
                                      n_orders=n_orders, live=f"; live = first {n_live} eval" if n_live else ""),
        "seed": seed, "n_dev": n_dev, "n_eval": n_eval, "n_live": n_live,
        "n_orders": n_orders, "n_rollouts": n_rollouts,
        "sorted_ids": split["sorted_ids"], "dev_ids": split["dev_ids"], "eval_ids": split["eval_ids"],
        "orders": {sid: split["orders"][i].tolist() for i, sid in enumerate(split["sorted_ids"])},
    }
    if n_live:
        content["live_ids"] = split["live_ids"]
    content["sha256"] = sha256_json(content)
    return content


def dumps_split(content):
    """JSON with one line per key and one line per question in `orders`."""
    lines = ["{"]
    keys = list(content)
    for k in keys:
        end = "," if k != keys[-1] else ""
        if k == "orders":
            lines.append('"orders": {')
            items = list(content["orders"].items())
            for j, (sid, o) in enumerate(items):
                sep = "," if j < len(items) - 1 else ""
                lines.append(f"{json.dumps(sid)}: {json.dumps(o, separators=(',', ':'))}{sep}")
            lines.append("}" + end)
        else:
            lines.append(f"{json.dumps(k)}: {json.dumps(content[k], ensure_ascii=False)}{end}")
    lines.append("}")
    return "\n".join(lines) + "\n"


def check_preview(dataset, content, preview_path):
    """The split IDs must equal the prompt-a preview (scratch/task3/splits_preview.json)."""
    with open(preview_path) as f:
        prev = json.load(f)
    p = prev[dataset]
    assert content["dev_ids"] == p["dev"], f"{dataset}: dev IDs differ from the preview"
    assert content["eval_ids"] == p["eval"], f"{dataset}: eval IDs differ from the preview"
    if content.get("live_ids") is not None:
        assert content["live_ids"] == prev[f"{dataset}_live{len(content['live_ids'])}"], \
            f"{dataset}: live IDs differ from the preview"


def build(cfg, dataset, sample_ids):
    s = cfg["sizes"]
    n_live = cfg["datasets"][dataset]["n_live"]
    split = make_split(sample_ids, cfg["seed"], s["n_dev"], s["n_eval"], n_live, cfg["n_orders"], s["n_rollouts"])
    content = split_content(dataset, split, cfg["seed"], s["n_dev"], s["n_eval"], n_live, cfg["n_orders"],
                            s["n_rollouts"])
    if cfg["paths"].get("splits_preview"):
        check_preview(dataset, content, resolve(cfg["paths"]["splits_preview"]))
    return content


def split_path(cfg, dataset):
    return os.path.join(resolve(cfg["paths"]["splits_dir"]), f"{dataset}_split.json")


def ensure_split(cfg, dataset, sample_ids):
    """Write the split file if absent; if present it must be identical to the recomputed one."""
    content = build(cfg, dataset, sample_ids)
    path = split_path(cfg, dataset)
    if os.path.exists(path):
        with open(path) as f:
            old = json.load(f)
        if old != content:
            raise SplitMismatchError(f"{path} exists and differs from the recomputed split")
        return content, path, "unchanged"
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        f.write(dumps_split(content))
    os.replace(tmp, path)
    return content, path, "written"


def load_split(path):
    with open(path) as f:
        content = json.load(f)
    body = {k: v for k, v in content.items() if k != "sha256"}
    assert sha256_json(body) == content["sha256"], f"{path}: sha256 mismatch"
    return content


def orders_array(content):
    """(n_questions, n_orders, n_rollouts) int array, questions in sorted-ID order."""
    return np.array([content["orders"][sid] for sid in content["sorted_ids"]], dtype=np.int64)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=["math", "mathvista"])
    ap.add_argument("--config", default="configs/task3/offline.yaml")
    a = ap.parse_args(argv)
    cfg = load_config(a.config)
    ids = pq.read_table(dataset_files(cfg, a.dataset)["generations"], columns=["sample_id"]).column(
        "sample_id").to_pylist()
    content, path, status = ensure_split(cfg, a.dataset, sorted(set(ids)))
    print(f"{a.dataset}: {path} {status}; dev {len(content['dev_ids'])}, eval {len(content['eval_ids'])}, "
          f"live {len(content.get('live_ids') or [])}, sha256 {content['sha256']}")


if __name__ == "__main__":
    main()
