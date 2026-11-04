"""Replace grounding pairs flagged on the label-check sheets (manual L501-502).

flags.json (notes/task2/label_check/) is a list of grounding dev/test pair ids. Each newly flagged pair is
replaced, in place, by the next candidate of its own set and type under the selection rule: the first item
after the highest rank used so far (selected or flagged) in the saved item order whose image is unused, with
the first valid unused partner in the saved partner order. Images of flagged pairs are never reused. The
affected split files, images_val.json, mismatch_map.json and selection_report.json are rewritten, every
assert is rerun, and splits/task2/label_flags.json records the flagged ids and their replacements.
Flags already recorded are skipped, so rerunning is a no-op.

Run: python -m task2_vlm.apply_label_flags
"""
import argparse
import gzip
import json
import os

from task2_vlm import select_data as S

LABEL_FLAGS = "label_flags.json"
DERIVED = ("counts", "answer_distributions", "category_spread", "empty_consensus", "sha256")


def apply_flags(sets, orders, val_idx, cats, cfg, flags, history):
    """Replace the newly flagged pairs in `sets` (modified in place). history: earlier label_flags records.
    Returns the new records."""
    done = {h["flagged"]["pair_id"] for h in history}
    where = {p["pair_id"]: (s, i) for s in ("grounding_dev", "grounding_test")
             for i, p in enumerate(sets[s]["pairs"])}
    new = [f for f in dict.fromkeys(flags) if f not in done]
    bad = [f for f in new if f not in where]
    if bad:
        raise SystemExit(f"flagged ids are not current grounding dev/test pairs: {bad}")
    records = []
    for pid in new:
        s, i = where[pid]
        old = sets[s]["pairs"][i]
        kind = old["type"]
        used = {im for n in S.VAL_SETS for im in S.image_ids_of(n, sets[n])}
        used |= {im for h in history + records for im in h["flagged"]["image_ids"]}
        ranks = [p["candidate_rank"] for p in sets[s]["pairs"] if p["type"] == kind]
        ranks += [h["flagged"]["candidate_rank"] for h in history + records
                  if h["set"] == s and h["flagged"]["type"] == kind]
        pair, rank = S.next_grounding_pair(kind, val_idx, orders[f"{s}/{kind}"], max(ranks) + 1, used, cats, cfg)
        if pair is None:
            raise SystemExit(f"no replacement candidate left for {pid}")
        sets[s]["pairs"][i] = pair
        records.append({"set": s, "position": i, "flagged": old, "replacement": pair})
    return records


def load_split_dir(out_dir):
    sets = {}
    for n in S.VAL_SETS + S.TRAIN_SETS:
        with open(os.path.join(out_dir, f"{n}.json")) as f:
            sets[n] = json.load(f)
    with gzip.open(os.path.join(out_dir, "candidate_orders.json.gz")) as f:
        orders = json.load(f)
    with open(os.path.join(out_dir, S.REPORT)) as f:
        report = json.load(f)
    return sets, orders, report


def rewrite(out_dir, sets, orders, report, records, history, metas, cfg, cats):
    files, iso = S.finalize(sets, metas["train"], metas["val"], cfg)
    blobs = S.file_blobs(files, orders)
    flags_doc = {"rule": __doc__.split("\n\n")[1].replace("\n", " "), "applied": history + records}
    blobs[LABEL_FLAGS] = S.dumps(LABEL_FLAGS, flags_doc)
    base = {k: v for k, v in report.items() if k not in DERIVED}
    base["isolation"] = iso
    base["label_flags"] = {"n_flagged": len(flags_doc["applied"]),
                           "replaced": {r["flagged"]["pair_id"]: r["replacement"]["pair_id"]
                                        for r in flags_doc["applied"]}}
    blobs[S.REPORT] = S.report_blob(files, {k: b for k, b in blobs.items()}, cfg, cats, base)
    S.write_files(out_dir, blobs, overwrite=True)
    return iso


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=S.CONFIG)
    ap.add_argument("--flags", default=None, help="default: <label_check.out_dir>/flags.json")
    args = ap.parse_args(argv)
    cfg = S.load_config(args.config)
    cats = S.load_categories(cfg["categories"])
    out_dir = cfg["out_dir"]
    with open(args.flags or os.path.join(cfg["label_check"]["out_dir"], "flags.json")) as f:
        flags = json.load(f)
    hist_path = os.path.join(out_dir, LABEL_FLAGS)
    history = json.load(open(hist_path))["applied"] if os.path.exists(hist_path) else []
    sets, orders, report = load_split_dir(out_dir)
    cd = cfg["raw"]["coco_ann_dir"]
    metas = {}
    for split in ("train", "val"):
        with open(f"{cd}/captions_{split}2014.json") as f:
            metas[split] = {im["id"]: im["file_name"] for im in json.load(f)["images"]}
    with open(f"{cd}/instances_val2014.json") as f:
        val_idx = S.GroundingIndex(json.load(f)["annotations"], cfg)
    records = apply_flags(sets, orders, val_idx, cats, cfg, flags, history)
    iso = rewrite(out_dir, sets, orders, report, records, history, metas, cfg, cats)
    for r in records:
        print(f"{r['set']}[{r['position']}]: {r['flagged']['pair_id']} -> {r['replacement']['pair_id']}")
    print(f"{len(records)} new flags applied ({len(history) + len(records)} total); asserts passed; isolation {iso}")


if __name__ == "__main__":
    main()
