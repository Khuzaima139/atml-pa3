"""The 50 live MATH problems for Step 4 (manual L595-596, L699-719): IDs in the committed split's live order,
problem text only (no gold), with a content sha256.

    python -m task3_ttc.live_problems [--config configs/task3/offline.yaml] [--out splits/task3/live_problems.json]

The problem text is the supplied `question_text` (constant within a question, checked); the Kaggle run reads only
this file, so the supplied data is not needed there.
"""
import argparse
import json
import os

import pyarrow.parquet as pq

from common.hashing import sha256_file, sha256_json
from task3_ttc.data import dataset_files, load_config, resolve

OUT = "splits/task3/live_problems.json"


def build(cfg):
    split_path = resolve(os.path.join(cfg["paths"]["splits_dir"], "math_split.json"))
    with open(split_path) as f:
        split = json.load(f)
    live = split["live_ids"]
    assert len(live) == cfg["datasets"]["math"]["n_live"] == 50 and live == split["eval_ids"][:50]
    gen_path = dataset_files(cfg, "math")["generations"]
    t = pq.read_table(gen_path, columns=["sample_id", "question_text"]).to_pydict()
    text = {}
    for sid, q in zip(t["sample_id"], t["question_text"]):
        if sid in live:
            assert text.setdefault(sid, q) == q, f"question_text not constant within {sid}"
    assert set(text) == set(live)
    problems = [{"q": i, "sample_id": sid, "problem": text[sid]} for i, sid in enumerate(live)]
    return {"dataset": "math", "n": len(problems),
            "source": {"split": "splits/task3/math_split.json", "split_sha256": split["sha256"],
                       "generations": "qwen25-math/generations.parquet (question_text)",
                       "generations_sha256": sha256_file(gen_path)},
            "rule": "live_ids of the committed MATH split in order (= first 50 evaluation IDs, L595-596); "
                    "q = position 0-49; problem text only, no gold",
            "problems": problems, "sha256": sha256_json(problems)}


def load(path=OUT):
    with open(resolve(path)) as f:
        d = json.load(f)
    if sha256_json(d["problems"]) != d["sha256"]:
        raise RuntimeError(f"{path}: problems sha256 mismatch")
    assert [p["q"] for p in d["problems"]] == list(range(len(d["problems"])))
    return d


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/task3/offline.yaml")
    ap.add_argument("--out", default=OUT)
    a = ap.parse_args(argv)
    d = build(load_config(a.config))
    out = resolve(a.out)
    if os.path.exists(out):
        with open(out) as f:
            old = json.load(f)
        if old != d:
            raise SystemExit(f"{a.out} exists with different content; not overwritten")
        print(f"{a.out} unchanged (sha256 {d['sha256']})")
        return
    with open(out, "w") as f:
        json.dump(d, f, indent=1, ensure_ascii=False)
        f.write("\n")
    print(f"wrote {a.out}: {d['n']} problems, sha256 {d['sha256']}")


if __name__ == "__main__":
    main()
