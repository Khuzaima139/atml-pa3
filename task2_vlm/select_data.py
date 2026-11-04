"""Task 2 data selection (manual L356-387, L437-438; decisions D1-D9, E1-E2 of the Task 2 plan).

All randomness comes from one rng = np.random.default_rng(6304), drawn in this fixed order:
    val:   vqa_dev   yes/no, number, other      one permutation of the eligible pairs each
           vqa_test  yes/no, number, other
           grounding_dev  presence items, presence partners, counting items, counting partners
           grounding_test presence items, presence partners, counting items, counting partners
           captions_dev   one permutation of the eligible val captions
    train: vqa_train yes/no, number, other
           captions_train one permutation of the eligible images, then one permutation of each selected
                          image's eligible captions (in selection order), first 3 kept
The mismatched-image map (D9) uses its own np.random.default_rng(6304): one permutation per test suite,
vqa_test first. Every set takes the first candidates, in its saved order, that satisfy all constraints
(val images at most once across all val sets, question ids at most once per split).

Run: python -m task2_vlm.select_data [--overwrite]
"""
import argparse
import collections
import gzip
import hashlib
import io
import json
import os

import numpy as np
import yaml

from common.runinfo import run_info
from task2_vlm.vqa_norm import normalize

CONFIG = "configs/task2/data.yaml"
VAL_SETS = ("vqa_dev", "vqa_test", "grounding_dev", "grounding_test", "captions_dev")
TRAIN_SETS = ("vqa_train", "captions_train", "grounding_pools_train")
SPLIT_FILES = ("vqa_train.json", "vqa_dev.json", "vqa_test.json", "captions_train.json", "captions_dev.json",
               "grounding_pools_train.json", "grounding_dev.json", "grounding_test.json", "mismatch_map.json",
               "images_train.json", "images_val.json", "candidate_orders.json.gz")
REPORT = "selection_report.json"
KINDS = ("presence", "counting")


def load_config(path=CONFIG):
    with open(path) as f:
        cfg = yaml.safe_load(f)
    assert 2 * cfg["vqa"]["min_agreement"] > cfg["vqa"]["n_answers"], "consensus must be a strict majority"
    return cfg


def load_categories(path):
    with open(path) as f:
        cats = yaml.safe_load(f)["categories"]
    return {c["id"]: c for c in cats}


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


# ---------------------------------------------------------------- VQA (D1-D4)

def question_consensus(answers, n_answers, min_agreement):
    """D1/D3 for one question. Returns None when no normalized answer reaches min_agreement."""
    answers = sorted(answers, key=lambda a: a["answer_id"])
    assert len(answers) == n_answers, f"expected {n_answers} answers, got {len(answers)}"
    norm = [normalize(a["answer"]) for a in answers]
    top, agreement = collections.Counter(norm).most_common(1)[0]
    if agreement < min_agreement:
        return None
    group = [a["answer"].strip() for a, n in zip(answers, norm) if n == top]
    raw_counts = collections.Counter(group)
    best = max(raw_counts.values())
    target = next(r for r in group if raw_counts[r] == best)  # ties: earliest annotator (answer_id order)
    return {"consensus": top, "agreement": agreement, "target": target,
            "answers": [a["answer"] for a in answers]}


def eligible_vqa_pairs(pairs, questions, annotations, cfg):
    """Released complementary pairs that pass D1, sorted by (qid_a, qid_b), split by answer_type.

    questions / annotations: dicts keyed by question id. Unordered duplicate pairs keep the first one.
    """
    v = cfg["vqa"]
    info = {}
    stats = collections.Counter()
    seen = set()
    out = {t: [] for t in v["types"]}
    for qa, qb in pairs:
        key = frozenset((qa, qb))
        if key in seen:
            stats["duplicate_pair"] += 1
            continue
        seen.add(key)
        for q in (qa, qb):
            if q not in info:
                info[q] = question_consensus(annotations[q]["answers"], v["n_answers"], v["min_agreement"])
        a, b = annotations[qa], annotations[qb]
        if info[qa] is None or info[qb] is None:
            stats["below_agreement"] += 1
            continue
        if info[qa]["consensus"] == info[qb]["consensus"]:
            stats["same_consensus"] += 1
            continue
        if a["answer_type"] != b["answer_type"]:
            stats["answer_type_mismatch"] += 1
            continue
        assert questions[qa]["question"] == questions[qb]["question"], (qa, qb)
        assert questions[qa]["image_id"] != questions[qb]["image_id"], (qa, qb)
        t = a["answer_type"]
        out[t].append({
            "pair_id": f"vqa_{qa}_{qb}", "question_ids": [qa, qb], "question": questions[qa]["question"],
            "image_ids": [questions[qa]["image_id"], questions[qb]["image_id"]], "answer_type": t,
            "question_types": [a["question_type"], b["question_type"]],
            "answers": [info[qa]["answers"], info[qb]["answers"]],
            "consensus": [info[qa]["consensus"], info[qb]["consensus"]],
            "agreement": [info[qa]["agreement"], info[qb]["agreement"]],
            "targets": [info[qa]["target"], info[qb]["target"]]})
        stats[f"eligible_{t}"] += 1
    for t in out:
        out[t].sort(key=lambda r: tuple(r["question_ids"]))
    stats["released_pairs"] = len(pairs)
    return out, dict(sorted(stats.items()))


def select_vqa(cands, n, rng, used_qids, used_images=None):
    """First n candidates of a seeded permutation whose question ids (and, for val, images) are unused."""
    perm = rng.permutation(len(cands))
    order = [cands[i]["pair_id"] for i in perm]
    chosen = []
    for rank, i in enumerate(perm):
        if len(chosen) == n:
            break
        c = cands[i]
        if any(q in used_qids for q in c["question_ids"]):
            continue
        if used_images is not None and any(im in used_images for im in c["image_ids"]):
            continue
        used_qids.update(c["question_ids"])
        if used_images is not None:
            used_images.update(c["image_ids"])
        chosen.append({**c, "candidate_rank": int(rank)})
    assert len(chosen) == n, f"only {len(chosen)} of {n} VQA pairs available"
    return chosen, order


# ---------------------------------------------------------------- grounding (D7, D8, E2)

class GroundingIndex:
    """Per-image category facts from COCO instances.

    cats_any[img]       categories with any annotation (crowd included); only images with >= 1 annotation
    noncrowd[(img,cat)] number of non-crowd instances
    positives           sorted (img, cat): >= 1 non-crowd instance with area >= min_area
    counting[(img,cat)] count, for 1..4 non-crowd instances, no crowd annotation, every instance area >= min_area
    """

    def __init__(self, annotations, cfg):
        g = cfg["grounding"]
        per = collections.defaultdict(lambda: {"crowd": 0, "areas": []})
        for a in annotations:
            d = per[(a["image_id"], a["category_id"])]
            if a["iscrowd"]:
                d["crowd"] += 1
            else:
                d["areas"].append(a["area"])
        cats = collections.defaultdict(set)
        for img, cat in per:
            cats[img].add(cat)
        self.cats_any = {img: frozenset(c) for img, c in cats.items()}
        self.annotated = sorted(self.cats_any)
        self.noncrowd = {k: len(d["areas"]) for k, d in per.items()}
        self.positives = sorted(k for k, d in per.items() if any(x >= g["min_area"] for x in d["areas"]))
        self.counting = {k: len(d["areas"]) for k, d in per.items()
                         if d["crowd"] == 0 and g["count_min"] <= len(d["areas"]) <= g["count_max"]
                         and all(x >= g["min_area"] for x in d["areas"])}

    def items(self, kind):
        if kind == "presence":
            return [list(k) for k in self.positives]
        return [[img, cat, n] for (img, cat), n in sorted(self.counting.items())]

    def presence_partner_ok(self, img, cat):
        return img in self.cats_any and cat not in self.cats_any[img]

    def counting_partner_ok(self, img, cat, n):
        m = self.counting.get((img, cat))
        return m is not None and m != n


def question_text(kind, cat, cfg):
    g = cfg["grounding"]
    if kind == "presence":
        return g["presence_template"].format(presence=cat["presence"])
    return g["counting_template"].format(plural=cat["plural"])


def make_pair(kind, img, cat_id, partner, idx, cats, cfg, rank):
    g = cfg["grounding"]
    if kind == "presence":
        answers = [g["answers"]["positive"], g["answers"]["negative"]]
        counts = [idx.noncrowd[(img, cat_id)], 0]
    else:
        counts = [idx.counting[(img, cat_id)], idx.counting[(partner, cat_id)]]
        answers = [str(c) for c in counts]
    return {"pair_id": f"grd_{kind}_{cat_id}_{img}_{partner}", "type": kind, "category_id": cat_id,
            "category": cats[cat_id]["name"], "question": question_text(kind, cats[cat_id], cfg),
            "image_ids": [img, partner], "answers": answers, "counts": counts, "candidate_rank": int(rank)}


def grounding_orders(idx, kind, rng):
    items = idx.items(kind)
    item_order = [items[i] for i in rng.permutation(len(items))]
    partner_order = [idx.annotated[i] for i in rng.permutation(len(idx.annotated))]
    return {"items": item_order, "partners": partner_order}


def next_grounding_pair(kind, idx, orders, start, used, cats, cfg):
    """First item at rank >= start whose image is unused and that has an unused valid partner; the partner is
    the first such image in the partner order. Returns (pair, rank) or (None, len(items))."""
    items, partners = orders["items"], orders["partners"]
    for rank in range(start, len(items)):
        img, cat = items[rank][0], items[rank][1]
        if img in used:
            continue
        for p in partners:
            if p == img or p in used:
                continue
            ok = (idx.presence_partner_ok(p, cat) if kind == "presence"
                  else idx.counting_partner_ok(p, cat, items[rank][2]))
            if ok:
                return make_pair(kind, img, cat, p, idx, cats, cfg, rank), rank
    return None, len(items)


def select_grounding(kind, n, idx, orders, used, cats, cfg):
    chosen, start = [], 0
    while len(chosen) < n:
        pair, rank = next_grounding_pair(kind, idx, orders, start, used, cats, cfg)
        assert pair is not None, f"only {len(chosen)} of {n} {kind} pairs available"
        used.update(pair["image_ids"])
        chosen.append(pair)
        start = rank + 1
    return chosen


def build_pools(idx, pool_images, cats, cfg):
    """Training item pools (D7): presence positives and counting items with at least one valid partner image
    inside the pool. Partners are drawn at train time from `image_categories` / `counting_items`."""
    pool = sorted(set(pool_images))
    annotated = [i for i in pool if i in idx.cats_any]
    aset = set(annotated)
    n_with_cat = collections.Counter(c for i in annotated for c in idx.cats_any[i])
    positives = [[i, c] for i, c in idx.positives if i in aset and n_with_cat[c] < len(annotated)]
    counts_by_cat = collections.defaultdict(set)
    for (i, c), m in idx.counting.items():
        if i in aset:
            counts_by_cat[c].add(m)
    counting = [[i, c, m] for (i, c), m in sorted(idx.counting.items())
                if i in aset and len(counts_by_cat[c] - {m}) > 0]
    per_cat = {}
    for cid in sorted(cats):
        per_cat[cats[cid]["name"]] = {
            "presence_positives": sum(1 for _, c in positives if c == cid),
            "presence_negatives": len(annotated) - n_with_cat[cid],
            "counting": dict(sorted(collections.Counter(str(m) for _, c, m in counting if c == cid).items()))}
    g = cfg["grounding"]
    return {
        "rules": {"presence_positive": f">=1 non-crowd instance with area >= {g['min_area']}",
                  "presence_negative": "pool image with >=1 annotation and zero annotations (crowd included) of the category",
                  "counting_item": f"{g['count_min']}..{g['count_max']} non-crowd instances, no crowd annotation, "
                                   f"every instance area >= {g['min_area']}",
                  "counting_partner": "pool image with a counting item of the same category and a different count",
                  "images": "caption training images plus VQA training images, images with no annotations excluded"},
        "presence_template": g["presence_template"], "counting_template": g["counting_template"],
        "questions": {str(cid): {"presence": question_text("presence", cats[cid], cfg),
                                 "counting": question_text("counting", cats[cid], cfg)} for cid in sorted(cats)},
        "n_pool_images": len(pool), "n_unannotated_excluded": len(pool) - len(annotated),
        "images": annotated,
        "image_categories": {str(i): sorted(idx.cats_any[i]) for i in annotated},
        "presence_positives": positives, "counting_items": counting, "per_category": per_cat}


# ---------------------------------------------------------------- captions (D5, D6)

def clean_caption(s):
    return " ".join(s.split())


def caption_records(caption_json, tokenizer):
    anns = sorted(caption_json["annotations"], key=lambda a: a["id"])
    texts = [clean_caption(a["caption"]) for a in anns]
    lens = [len(x) for x in tokenizer(texts, add_special_tokens=False)["input_ids"]]
    return [{"caption_id": a["id"], "image_id": a["image_id"], "caption": t, "n_tokens": n}
            for a, t, n in zip(anns, texts, lens)]


def select_caption_dev(records, n, max_tokens, rng, used):
    cands = [r for r in records if r["n_tokens"] <= max_tokens]
    perm = rng.permutation(len(cands))
    chosen = []
    for rank, i in enumerate(perm):
        if len(chosen) == n:
            break
        r = cands[i]
        if r["image_id"] in used:
            continue
        used.add(r["image_id"])
        chosen.append({**r, "candidate_rank": int(rank)})
    assert len(chosen) == n
    return chosen, [cands[i]["caption_id"] for i in perm]


def select_caption_train(records, n_images, per_image, max_tokens, rng):
    by_img = collections.defaultdict(list)
    for r in records:
        if r["n_tokens"] <= max_tokens:
            by_img[r["image_id"]].append(r)
    imgs = sorted(i for i, rs in by_img.items() if len(rs) >= per_image)
    perm = rng.permutation(len(imgs))
    chosen = []
    for rank, i in enumerate(perm[:n_images]):
        rs = by_img[imgs[i]]  # sorted by caption id
        pick = [rs[j] for j in rng.permutation(len(rs))[:per_image]]
        chosen.append({"image_id": imgs[i], "caption_ids": [r["caption_id"] for r in pick],
                       "captions": [r["caption"] for r in pick], "n_tokens": [r["n_tokens"] for r in pick],
                       "candidate_rank": int(rank)})
    assert len(chosen) == n_images
    return chosen, [imgs[i] for i in perm]


# ---------------------------------------------------------------- mismatch map (D9)

def mismatch_map(suites, seed, offset):
    """suites: {name: list of pairs in file order}. Pair at shuffled position p takes the images of the pair at
    position p+offset (cyclic), same slot."""
    rng = np.random.default_rng(seed)
    out = {"rule": f"rng = np.random.default_rng({seed}); per suite in this order a permutation of the pairs in "
                   f"file order; the pair at shuffled position p takes the images of the pair at p+{offset} "
                   "(cyclic), same slot", "suites": {}}
    for name, pairs in suites.items():
        perm = rng.permutation(len(pairs))
        shuffled = [pairs[i] for i in perm]
        mapping = {}
        for p, pair in enumerate(shuffled):
            src = shuffled[(p + offset) % len(shuffled)]
            assert not set(src["image_ids"]) & set(pair["image_ids"]), (pair["pair_id"], src["pair_id"])
            mapping[pair["pair_id"]] = {"source_pair": src["pair_id"], "image_ids": list(src["image_ids"])}
        out["suites"][name] = {"shuffled_order": [x["pair_id"] for x in shuffled], "map": mapping}
    return out


# ---------------------------------------------------------------- image tables, checks, report

def image_ids_of(name, data):
    """Image ids a split file uses (with repeats)."""
    if name in ("captions_train", "captions_dev"):
        return [r["image_id"] for r in data["images"]]
    if name == "grounding_pools_train":
        return list(data["images"])
    return [im for p in data["pairs"] for im in p["image_ids"]]


def image_table(ids, meta, split, cfg):
    return {"split": split, "images": [
        {"image_id": i, "file_name": meta[i], "url": cfg["image_url"].format(split=split, file_name=meta[i])}
        for i in sorted(set(ids))]}


def finalize(sets, train_meta, val_meta, cfg):
    """Image tables and mismatch map from the selected sets; then every assert. Returns (files, isolation)."""
    files = dict(sets)
    train_ids = [im for n in TRAIN_SETS for im in image_ids_of(n, sets[n])]
    val_ids = [im for n in VAL_SETS for im in image_ids_of(n, sets[n])]
    files["images_train"] = image_table(train_ids, train_meta, "train2014", cfg)
    files["images_val"] = image_table(val_ids, val_meta, "val2014", cfg)
    files["mismatch_map"] = mismatch_map({s: sets[s]["pairs"] for s in cfg["mismatch"]["suites"]},
                                         cfg["seed"], cfg["mismatch"]["offset"])
    isolation = check_all(files, set(train_meta), set(val_meta), cfg)
    return files, isolation


def check_all(files, train_universe, val_universe, cfg):
    """Every assert on the split files. Raises AssertionError; returns the isolation summary."""
    T, v, g = cfg["manual_table"], cfg["vqa"], cfg["grounding"]
    train_ids = {im for n in TRAIN_SETS for im in image_ids_of(n, files[n])}
    val_lists = {n: image_ids_of(n, files[n]) for n in VAL_SETS}
    val_all = [im for ids in val_lists.values() for im in ids]
    iso = {
        "train_images": len(train_ids), "val_images": len(set(val_all)),
        "train_images_in_val2014": len(train_ids & val_universe),
        "train_images_outside_train2014": len(train_ids - train_universe),
        "val_images_outside_val2014": len(set(val_all) - val_universe),
        "val_images_used_more_than_once": sum(1 for c in collections.Counter(val_all).values() if c > 1),
        "pairwise_val_overlap": {f"{a}&{b}": len(set(val_lists[a]) & set(val_lists[b]))
                                 for i, a in enumerate(VAL_SETS) for b in VAL_SETS[i + 1:]}}
    assert iso["train_images_in_val2014"] == 0 and iso["train_images_outside_train2014"] == 0
    assert iso["val_images_outside_val2014"] == 0
    assert iso["val_images_used_more_than_once"] == 0, "E1: a val image is used twice"
    assert all(x == 0 for x in iso["pairwise_val_overlap"].values())
    assert {r["image_id"] for r in files["images_train"]["images"]} == train_ids
    assert {r["image_id"] for r in files["images_val"]["images"]} == set(val_all)

    # VQA counts, uniqueness and eligibility
    for split in ("train", "dev", "test"):
        pairs = files[f"vqa_{split}"]["pairs"]
        got = collections.Counter(p["answer_type"] for p in pairs)
        assert dict(got) == {t: n for t, n in v["pairs"][split].items()}, (split, got)
        assert len(pairs) == T["vqa"][f"{split}_pairs"] and 2 * len(pairs) == T["vqa"][f"{split}_questions"]
        for p in pairs:
            assert p["consensus"][0] != p["consensus"][1] and min(p["agreement"]) >= v["min_agreement"]
            assert len(p["answers"][0]) == len(p["answers"][1]) == v["n_answers"]
            assert p["image_ids"][0] != p["image_ids"][1]
            assert [normalize(t) for t in p["targets"]] == p["consensus"], "D3: target outside the consensus group"
    for qids in ([q for p in files["vqa_train"]["pairs"] for q in p["question_ids"]],
                 [q for s in ("dev", "test") for p in files[f"vqa_{s}"]["pairs"] for q in p["question_ids"]]):
        assert len(qids) == len(set(qids)), "D4: question id repeated within a split"

    # grounding
    for split in ("dev", "test"):
        pairs = files[f"grounding_{split}"]["pairs"]
        got = collections.Counter(p["type"] for p in pairs)
        assert dict(got) == g["pairs"][split], (split, got)
        assert len(pairs) == T["grounding"][f"{split}_pairs"] and 2 * len(pairs) == T["grounding"][f"{split}_questions"]
        for p in pairs:
            assert p["answers"][0] != p["answers"][1] and p["image_ids"][0] != p["image_ids"][1]
            if p["type"] == "presence":
                assert p["answers"] == [g["answers"]["positive"], g["answers"]["negative"]] and p["counts"][1] == 0
            else:
                assert all(g["count_min"] <= c <= g["count_max"] for c in p["counts"])
                assert p["answers"] == [str(c) for c in p["counts"]]
    pools = files["grounding_pools_train"]
    assert pools["presence_positives"] and pools["counting_items"]

    # captions
    c = cfg["captions"]
    ct, cd = files["captions_train"]["images"], files["captions_dev"]["images"]
    assert len(ct) == c["train_images"] == T["captions"]["train_images"]
    assert sum(len(r["captions"]) for r in ct) == T["captions"]["train_examples"]
    assert all(len(r["captions"]) == c["train_per_image"] and len(set(r["caption_ids"])) == c["train_per_image"]
               and max(r["n_tokens"]) <= c["max_tokens"] for r in ct)
    assert len({r["image_id"] for r in ct}) == len(ct)
    assert len(cd) == c["dev_images"] == T["captions"]["dev_images"]
    assert all(r["n_tokens"] <= c["max_tokens"] for r in cd)

    # mismatch map (D9)
    mm = files["mismatch_map"]["suites"]
    assert list(mm) == cfg["mismatch"]["suites"]
    for suite, m in mm.items():
        pairs = {p["pair_id"]: p for p in files[suite]["pairs"]}
        assert set(m["map"]) == set(pairs) == set(m["shuffled_order"])
        assert sorted(e["source_pair"] for e in m["map"].values()) == sorted(pairs)
        for pid, e in m["map"].items():
            assert e["source_pair"] != pid and e["image_ids"] == pairs[e["source_pair"]]["image_ids"]
            assert not set(e["image_ids"]) & set(pairs[pid]["image_ids"]), "D9: mismatched image from own pair"
    return iso


def derived_report(files, cfg, cats):
    """Counts, answer distributions and category spread computed from the split files."""
    rep = {"counts": {}, "answer_distributions": {}, "category_spread": {}, "empty_consensus": {}}
    for split in ("train", "dev", "test"):
        pairs = files[f"vqa_{split}"]["pairs"]
        rep["counts"][f"vqa_{split}"] = {"pairs": len(pairs), "questions": 2 * len(pairs),
                                         **collections.Counter(p["answer_type"] for p in pairs)}
        dist = {}
        for t in cfg["vqa"]["types"]:
            cnt = collections.Counter(a for p in pairs if p["answer_type"] == t for a in p["consensus"])
            dist[t] = {"distinct": len(cnt), "top": cnt.most_common(15)}
        rep["answer_distributions"][f"vqa_{split}"] = dist
        empty = [p["pair_id"] for p in pairs if "" in p["consensus"]]
        rep["empty_consensus"][f"vqa_{split}"] = {"n_pairs": len(empty), "pair_ids": empty}
        rep["counts"][f"vqa_{split}"]["targets_differing_from_consensus"] = sum(
            t != c for p in pairs for t, c in zip(p["targets"], p["consensus"]))
    for split in ("dev", "test"):
        pairs = files[f"grounding_{split}"]["pairs"]
        rep["counts"][f"grounding_{split}"] = {"pairs": len(pairs), "questions": 2 * len(pairs),
                                               **collections.Counter(p["type"] for p in pairs)}
        rep["answer_distributions"][f"grounding_{split}"] = {
            k: dict(sorted(collections.Counter(a for p in pairs if p["type"] == k for a in p["answers"]).items()))
            for k in KINDS}
        rep["category_spread"][f"grounding_{split}"] = {
            k: dict(collections.Counter(p["category"] for p in pairs if p["type"] == k).most_common()) for k in KINDS}
    pools = files["grounding_pools_train"]
    rep["counts"]["grounding_pools_train"] = {
        "pool_images": pools["n_pool_images"], "unannotated_excluded": pools["n_unannotated_excluded"],
        "presence_positives": len(pools["presence_positives"]), "counting_items": len(pools["counting_items"]),
        "counting_by_count": dict(sorted(collections.Counter(str(m) for _, _, m in pools["counting_items"]).items())),
        "categories_with_presence": sum(1 for x in pools["per_category"].values() if x["presence_positives"]),
        "categories_with_counting": sum(1 for x in pools["per_category"].values() if x["counting"])}
    rep["category_spread"]["grounding_pools_train"] = pools["per_category"]
    ct, cd = files["captions_train"]["images"], files["captions_dev"]["images"]
    toks = [n for r in ct for n in r["n_tokens"]]
    rep["counts"]["captions_train"] = {"images": len(ct), "examples": len(toks), "tokens_max": max(toks),
                                       "tokens_p50": float(np.median(toks))}
    rep["counts"]["captions_dev"] = {"images": len(cd), "tokens_max": max(r["n_tokens"] for r in cd)}
    rep["counts"]["images_train"] = len(files["images_train"]["images"])
    rep["counts"]["images_val"] = len(files["images_val"]["images"])
    return rep


def dumps(name, data):
    if name.endswith(".gz"):
        buf = io.BytesIO()
        with gzip.GzipFile(filename="", mode="wb", fileobj=buf, mtime=0) as f:
            f.write(json.dumps(data, separators=(",", ":")).encode())
        return buf.getvalue()
    return (json.dumps(data, indent=1) + "\n").encode()


def write_files(out_dir, blobs, overwrite):
    """Write {file name: bytes} atomically; refuse to change an existing split file unless overwrite
    (the report, which carries provenance and a timestamp, is always rewritten)."""
    os.makedirs(out_dir, exist_ok=True)
    if not overwrite:
        for name, b in blobs.items():
            if name == REPORT:
                continue
            p = os.path.join(out_dir, name)
            if os.path.exists(p) and open(p, "rb").read() != b:
                raise SystemExit(f"{p} exists with different contents; pass --overwrite to replace it")
    for name, b in blobs.items():
        p = os.path.join(out_dir, name)
        with open(p + ".tmp", "wb") as f:
            f.write(b)
        os.replace(p + ".tmp", p)


def file_blobs(files, orders):
    blobs = {f"{n}.json": dumps(n, files[n]) for n in files}
    blobs["candidate_orders.json.gz"] = dumps("candidate_orders.json.gz", orders)
    assert sorted(blobs) == sorted(SPLIT_FILES)
    return blobs


def report_blob(files, blobs, cfg, cats, base):
    """selection_report.json: `base` (config, provenance, eligibility, isolation, label flags) plus the
    statistics derived from the files and the sha256 of every split file."""
    rep = {**base, **derived_report(files, cfg, cats)}
    rep["sha256"] = {n: hashlib.sha256(b).hexdigest() for n, b in sorted(blobs.items())}
    return dumps(REPORT, rep)


# ---------------------------------------------------------------- main

def load_raw(cfg, split):
    vd, cd = cfg["raw"]["vqa_dir"], cfg["raw"]["coco_ann_dir"]
    paths = {"questions": f"{vd}/v2_OpenEnded_mscoco_{split}2014_questions.json",
             "annotations": f"{vd}/v2_mscoco_{split}2014_annotations.json",
             "pairs": f"{vd}/v2_mscoco_{split}2014_complementary_pairs.json",
             "captions": f"{cd}/captions_{split}2014.json", "instances": f"{cd}/instances_{split}2014.json"}
    raw = {k: json.load(open(p)) for k, p in paths.items()}
    sha = {os.path.basename(p): sha256_file(p) for p in paths.values()}
    return raw, sha


def load_tokenizer(cfg):
    from transformers import AutoTokenizer
    t = cfg["tokenizer"]
    return AutoTokenizer.from_pretrained(t["name"], revision=t["revision"])


def select(cfg, raw_val, raw_train, tokenizer, cats):
    """The full selection (pure given its inputs). Returns (sets, orders, eligibility, metas, val index)."""
    rng = np.random.default_rng(cfg["seed"])
    v, c, g = cfg["vqa"], cfg["captions"], cfg["grounding"]
    orders, elig, sets = {}, {}, {}
    vqa_cands = {}
    for split, raw in (("val", raw_val), ("train", raw_train)):
        qs = {q["question_id"]: q for q in raw["questions"]["questions"]}
        anns = {a["question_id"]: a for a in raw["annotations"]["annotations"]}
        vqa_cands[split], elig[f"vqa_{split}"] = eligible_vqa_pairs(raw["pairs"], qs, anns, cfg)

    used_val, val_qids = set(), set()
    for s in ("dev", "test"):
        pairs = []
        for t in v["types"]:
            chosen, orders[f"vqa_{s}/{t}"] = select_vqa(vqa_cands["val"][t], v["pairs"][s][t], rng, val_qids, used_val)
            pairs += chosen
        sets[f"vqa_{s}"] = {"split": s, "image_split": "val2014", "pairs": pairs}
    val_idx = GroundingIndex(raw_val["instances"]["annotations"], cfg)
    for s in ("dev", "test"):
        pairs = []
        for k in KINDS:
            orders[f"grounding_{s}/{k}"] = grounding_orders(val_idx, k, rng)
            pairs += select_grounding(k, g["pairs"][s][k], val_idx, orders[f"grounding_{s}/{k}"], used_val, cats, cfg)
        sets[f"grounding_{s}"] = {"split": s, "image_split": "val2014", "pairs": pairs}
    elig["grounding_val"] = {"annotated_images": len(val_idx.annotated), "presence_items": len(val_idx.positives),
                             "counting_items": len(val_idx.counting)}
    val_caps = caption_records(raw_val["captions"], tokenizer)
    chosen, orders["captions_dev"] = select_caption_dev(val_caps, c["dev_images"], c["max_tokens"], rng, used_val)
    sets["captions_dev"] = {"split": "dev", "image_split": "val2014", "instruction": c["instruction"], "images": chosen}
    elig["captions_val"] = {"captions": len(val_caps), "over_max_tokens": sum(r["n_tokens"] > c["max_tokens"] for r in val_caps)}

    train_qids, pairs = set(), []
    for t in v["types"]:
        chosen, orders[f"vqa_train/{t}"] = select_vqa(vqa_cands["train"][t], v["pairs"]["train"][t], rng, train_qids)
        pairs += chosen
    sets["vqa_train"] = {"split": "train", "image_split": "train2014", "pairs": pairs}
    train_caps = caption_records(raw_train["captions"], tokenizer)
    chosen, orders["captions_train"] = select_caption_train(train_caps, c["train_images"], c["train_per_image"],
                                                            c["max_tokens"], rng)
    sets["captions_train"] = {"split": "train", "image_split": "train2014", "instruction": c["instruction"],
                              "images": chosen}
    elig["captions_train"] = {"captions": len(train_caps),
                              "over_max_tokens": sum(r["n_tokens"] > c["max_tokens"] for r in train_caps)}
    train_idx = GroundingIndex(raw_train["instances"]["annotations"], cfg)
    pool_images = image_ids_of("captions_train", sets["captions_train"]) + image_ids_of("vqa_train", sets["vqa_train"])
    sets["grounding_pools_train"] = build_pools(train_idx, pool_images, cats, cfg)
    metas = {split: {im["id"]: im["file_name"] for im in raw["instances"]["images"]}
             for split, raw in (("val", raw_val), ("train", raw_train))}
    for split, raw in (("val", raw_val), ("train", raw_train)):  # apply_label_flags reads the captions file
        assert metas[split] == {im["id"]: im["file_name"] for im in raw["captions"]["images"]}
    return sets, orders, elig, metas, val_idx


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=CONFIG)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args(argv)
    cfg = load_config(args.config)
    cats = load_categories(cfg["categories"])
    raw_val, sha_val = load_raw(cfg, "val")
    raw_train, sha_train = load_raw(cfg, "train")
    sets, orders, elig, metas, _ = select(cfg, raw_val, raw_train, load_tokenizer(cfg), cats)
    files, iso = finalize(sets, metas["train"], metas["val"], cfg)
    blobs = file_blobs(files, orders)
    base = {"config": cfg, "config_sha256": sha256_file(args.config),
            "categories_sha256": sha256_file(cfg["categories"]), "provenance": run_info(),
            "raw_inputs_sha256": {**sha_val, **sha_train}, "eligibility": elig, "isolation": iso,
            "label_flags": None}
    blobs[REPORT] = report_blob(files, blobs, cfg, cats, base)
    write_files(cfg["out_dir"], blobs, args.overwrite)
    rep = json.loads(blobs[REPORT])
    print(json.dumps({"counts": rep["counts"], "isolation": iso,
                      "empty_consensus": {k: x["n_pairs"] for k, x in rep["empty_consensus"].items()}}, indent=1))


if __name__ == "__main__":
    main()
