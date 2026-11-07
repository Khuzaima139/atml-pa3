"""Replace grounding pairs flagged on the label-check sheets (manual L501-502).

flags.json (notes/task2/label_check/) is a list of grounding dev/test pair ids. Each newly flagged pair is
replaced, in place, by the next candidate of its own set and type under the selection rule: the first item
after the highest rank used so far (selected or flagged) in the saved item order whose image is unused, with
the first valid unused partner in the saved partner order. Images of flagged pairs are never reused. The
affected split files, images_val.json, mismatch_map.json and selection_report.json are rewritten, every
assert is rerun, and splits/task2/label_flags.json records the flagged ids and their replacements.
Flags already recorded are skipped, so rerunning is a no-op.

Pre-flags: flags.json may also hold ids of grounding candidates not yet selected (screened on preview sheets).
When such a candidate comes up as a replacement it is treated as flagged (its images consumed, its rank used,
recorded with "preflagged": true) and the next candidate is taken. Each slot consumes candidates in the saved
order whether they pass or fail, so a stream's candidate sequence does not depend on the verdicts; --preview
lists the next candidates of given (set, type) streams without changing any file.

Run: python -m task2_vlm.apply_label_flags
     python -m task2_vlm.apply_label_flags --preview grounding_dev/counting=25 grounding_test/counting=35 ...
"""
import argparse
import gzip
import json
import os

from task2_vlm import select_data as S

LABEL_FLAGS = "label_flags.json"
DERIVED = ("counts", "answer_distributions", "category_spread", "empty_consensus", "sha256")


PAIR_PREFIXES = ("grd_presence_", "grd_counting_")


def next_candidate(sets, orders, val_idx, cats, cfg, s, kind, records):
    """Next candidate of set s and type kind after every pair selected or consumed so far (records)."""
    used = {im for n in S.VAL_SETS for im in S.image_ids_of(n, sets[n])}
    used |= {im for h in records for im in h["flagged"]["image_ids"]}
    ranks = [p["candidate_rank"] for p in sets[s]["pairs"] if p["type"] == kind]
    ranks += [h["flagged"]["candidate_rank"] for h in records if h["set"] == s and h["flagged"]["type"] == kind]
    pair, _ = S.next_grounding_pair(kind, val_idx, orders[f"{s}/{kind}"], max(ranks) + 1, used, cats, cfg)
    return pair


def apply_flags(sets, orders, val_idx, cats, cfg, flags, history):
    """Replace the newly flagged pairs in `sets` (modified in place). history: earlier label_flags records.
    Ids that are not current pairs are pre-flags for candidates. Returns the new records (pre-flagged
    candidates skipped while filling a slot get their own record with replacement None)."""
    done = {h["flagged"]["pair_id"] for h in history}
    where = {p["pair_id"]: (s, i) for s in ("grounding_dev", "grounding_test")
             for i, p in enumerate(sets[s]["pairs"])}
    new = [f for f in dict.fromkeys(flags) if f not in done]
    pre = {f for f in new if f not in where}
    bad = sorted(f for f in pre if not f.startswith(PAIR_PREFIXES))
    if bad:
        raise SystemExit(f"flagged ids are neither grounding dev/test pairs nor grounding candidates: {bad}")
    records = []
    for pid in [f for f in new if f in where]:
        s, i = where[pid]
        old = sets[s]["pairs"][i]
        rec = {"set": s, "position": i, "flagged": old, "replacement": None}
        records.append(rec)
        while True:
            pair = next_candidate(sets, orders, val_idx, cats, cfg, s, old["type"], history + records)
            if pair is None:
                raise SystemExit(f"no replacement candidate left for {pid}")
            if pair["pair_id"] not in pre:
                break
            records.append({"set": s, "position": i, "flagged": pair, "replacement": None, "preflagged": True})
        sets[s]["pairs"][i] = pair
        rec["replacement"] = pair
    return records


def pending_preflags(flags, history, records, sets):
    """Pre-flag ids not reached (yet) by any replacement."""
    seen = {h["flagged"]["pair_id"] for h in history + records}
    current = {p["pair_id"] for s in ("grounding_dev", "grounding_test") for p in sets[s]["pairs"]}
    return [f for f in dict.fromkeys(flags) if f not in seen and f not in current]


def preview(sets, orders, val_idx, cats, cfg, history, streams):
    """The next n candidates of each (set, type) stream, as apply_flags would consume them; no file changes.
    Streams are simulated one at a time; their images are asserted disjoint, so the interleaving of streams in
    apply_flags (which depends on the verdicts) cannot change any stream's sequence."""
    out = {}
    for (s, kind), n in streams.items():
        recs, seq = list(history), []
        for _ in range(n):
            pair = next_candidate(sets, orders, val_idx, cats, cfg, s, kind, recs)
            assert pair is not None, f"{s}/{kind}: only {len(seq)} candidates left"
            recs.append({"set": s, "flagged": pair})
            seq.append(pair)
        out[(s, kind)] = seq
    keys = list(out)
    for a in range(len(keys)):
        for b in range(a + 1, len(keys)):
            ia = {im for p in out[keys[a]] for im in p["image_ids"]}
            ib = {im for p in out[keys[b]] for im in p["image_ids"]}
            assert not ia & ib, f"streams {keys[a]} and {keys[b]} share images: order would matter"
    return out


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
                                        for r in flags_doc["applied"] if r["replacement"] is not None},
                           "preflagged_skipped": [r["flagged"]["pair_id"] for r in flags_doc["applied"]
                                                  if r.get("preflagged")]}
    blobs[S.REPORT] = S.report_blob(files, {k: b for k, b in blobs.items()}, cfg, cats, base)
    S.write_files(out_dir, blobs, overwrite=True)
    return iso


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=S.CONFIG)
    ap.add_argument("--flags", default=None, help="default: <label_check.out_dir>/flags.json")
    ap.add_argument("--preview", nargs="+", default=None, metavar="SET/TYPE=N",
                    help="write <label_check.out_dir>/preview.json with the next N candidates per stream; "
                         "no split file is changed")
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
    if args.preview:
        write_preview(args.preview, sets, orders, val_idx, cats, cfg, flags, history)
        return
    records = apply_flags(sets, orders, val_idx, cats, cfg, flags, history)
    iso = rewrite(out_dir, sets, orders, report, records, history, metas, cfg, cats)
    for r in records:
        if r.get("preflagged"):
            print(f"{r['set']}[{r['position']}]: skipped pre-flagged candidate {r['flagged']['pair_id']}")
        else:
            print(f"{r['set']}[{r['position']}]: {r['flagged']['pair_id']} -> {r['replacement']['pair_id']}")
    pend = pending_preflags(flags, history, records, sets)
    print(f"{len(records)} new records ({sum(1 for r in records if r.get('preflagged'))} pre-flagged skips), "
          f"{len(history) + len(records)} total; pre-flags not reached: {pend}; asserts passed; isolation {iso}")


def write_preview(specs, sets, orders, val_idx, cats, cfg, flags, history):
    """Preview of the next candidates per stream (screening before apply); the first candidates of a stream are
    the provisional replacements of its new flags, in flags.json order."""
    streams = {}
    for x in specs:
        key, n = x.split("=")
        s, kind = key.split("/")
        streams[(s, kind)] = int(n)
    out = preview(sets, orders, val_idx, cats, cfg, history, streams)
    done = {h["flagged"]["pair_id"] for h in history}
    where = {p["pair_id"]: (s, p["type"]) for s in ("grounding_dev", "grounding_test") for p in sets[s]["pairs"]}
    new = [f for f in dict.fromkeys(flags) if f not in done and f in where]
    rows = []
    for (s, kind), seq in out.items():
        slots = [f for f in new if where[f] == (s, kind)]
        for k, p in enumerate(seq):
            rows.append({"pair_id": p["pair_id"], "set": s, "type": kind, "stream_position": k + 1,
                         "candidate_rank": p["candidate_rank"], "category": p["category"], "question": p["question"],
                         "image_ids": p["image_ids"], "answers": p["answers"], "counts": p["counts"],
                         "provisionally_replaces": slots[k] if k < len(slots) else None})
    unfilled = {f"{s}/{k}": [f for f in new if where[f] == (s, k)] for s, k in streams}
    missing = [f for f in new if where[f] not in streams]
    assert not missing, f"new flags in streams without a preview: {missing}"
    doc = {"rule": "next candidates per stream in the saved item order, first unused valid partner, images of "
                   "every consumed candidate never reused (task2_vlm.apply_label_flags.preview); streams have "
                   "disjoint images, so verdicts cannot change any stream's sequence",
           "streams": {f"{s}/{k}": n for (s, k), n in streams.items()}, "new_flags": unfilled, "pairs": rows}
    path = os.path.join(cfg["label_check"]["out_dir"], "preview.json")
    with open(path, "w") as f:
        json.dump(doc, f, indent=1)
    print(f"{path}: {len(rows)} candidates " + ", ".join(f"{k} {len(v)}" for k, v in
                                                         {f'{s}/{k}': q for (s, k), q in out.items()}.items()))


if __name__ == "__main__":
    main()
