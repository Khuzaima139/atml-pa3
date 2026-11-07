"""Task 2 data selection (D1-D9, E1-E2) on synthetic inputs, plus checks on the committed split files."""
import copy
import gzip
import hashlib
import json
import os

import numpy as np
import pytest

import task2_synth as T
from task2_vlm import apply_label_flags as A
from task2_vlm import select_data as S

REPO = T.REPO
SPLITS = os.path.join(REPO, "splits/task2")


def ans(*raw):
    return [{"answer": a, "answer_confidence": "yes", "answer_id": i + 1} for i, a in enumerate(raw)]


# ---------------------------------------------------------------- D1 / D3

def test_consensus_needs_eight_of_ten_after_normalization():
    c = S.question_consensus(ans(*["two"] * 4, *["2"] * 4, "3", "4"), 10, 8)
    assert c["consensus"] == "2" and c["agreement"] == 8
    assert S.question_consensus(ans(*["yes"] * 7, "no", "no", "no"), 10, 8) is None


def test_target_most_frequent_raw_then_earliest_annotator():
    c = S.question_consensus(ans("2", "two", "two", "2", "2", "two", "2", "two", "5", "6"), 10, 8)
    assert c["target"] == "2"  # 4 vs 4: "2" is annotator 1
    c = S.question_consensus(ans("none", "0", "0", "0", "none", "none", "none", "0", "0", "x"), 10, 8)
    assert c["consensus"] == "0" and c["target"] == "0"  # 5 "0" vs 4 "none"
    c = S.question_consensus(ans(" t-shirt", *["t shirt"] * 2, *["t-shirt "] * 6, "hat"), 10, 8)
    assert c["consensus"] == "t shirt" and c["target"] == "t-shirt"  # stripped raw strings


def test_empty_consensus_is_kept():
    c = S.question_consensus(ans(*["a"] * 9, "dog"), 10, 8)
    assert c["consensus"] == "" and c["target"] == "a"


def _vqa_inputs(type_b="other", ans_b=None):
    qs = {1: {"question": "what?", "image_id": 10}, 2: {"question": "what?", "image_id": 11}}
    an = {1: {"answer_type": "other", "question_type": "what", "answers": ans(*["red"] * 10)},
          2: {"answer_type": type_b, "question_type": "what", "answers": ans_b or ans(*["blue"] * 10)}}
    return qs, an


def test_eligibility_rules():
    cfg = T.config()
    qs, an = _vqa_inputs()
    out, st = S.eligible_vqa_pairs([[1, 2], [2, 1]], qs, an, cfg)
    assert [p["pair_id"] for p in out["other"]] == ["vqa_1_2"] and st["duplicate_pair"] == 1
    out, st = S.eligible_vqa_pairs([[1, 2]], *_vqa_inputs(type_b="yes/no"), cfg)
    assert st["answer_type_mismatch"] == 1 and not any(out.values())
    out, st = S.eligible_vqa_pairs([[1, 2]], *_vqa_inputs(ans_b=ans(*["red."] * 10)), cfg)
    assert st["same_consensus"] == 1
    out, st = S.eligible_vqa_pairs([[1, 2]], *_vqa_inputs(ans_b=ans(*["blue"] * 7, *["x"] * 3)), cfg)
    assert st["below_agreement"] == 1


# ---------------------------------------------------------------- D7 / E2 / D8

def _inst(img, cat, area=2000.0, crowd=0):
    return {"image_id": img, "category_id": cat, "area": area, "iscrowd": crowd}


def test_grounding_index_rules():
    cfg = T.config()
    anns = [_inst(1, 18), _inst(1, 18), _inst(1, 17, area=500.0),        # dog x2 ok; small cat
            _inst(2, 18), _inst(2, 18, crowd=1),                          # crowd dog
            *[_inst(3, 18) for _ in range(5)],                            # 5 dogs
            _inst(4, 1), _inst(4, 18, area=500.0), _inst(4, 18)]          # one small dog
    idx = S.GroundingIndex(anns, cfg)
    assert idx.counting == {(1, 18): 2, (4, 1): 1}
    assert (1, 17) not in idx.positives and (1, 18) in idx.positives and (4, 18) in idx.positives
    assert (2, 18) in idx.positives and (2, 18) not in idx.counting
    assert idx.presence_partner_ok(3, 17) and not idx.presence_partner_ok(2, 18)  # crowd counts as present
    assert not idx.presence_partner_ok(99, 17)  # unannotated image: never a negative
    assert idx.counting_partner_ok(4, 1, 2) and not idx.counting_partner_ok(4, 1, 1)


def test_question_wording():
    cfg, cats = T.config(), T.categories()
    q = lambda k, c: S.question_text(k, cats[c], cfg)
    assert q("presence", 1) == "Does this image contain a person?"
    assert q("presence", 53) == "Does this image contain an apple?"
    assert q("presence", 56) == "Does this image contain broccoli?"
    assert q("counting", 1) == "How many people can you see?"
    assert q("counting", 74) == "How many mice can you see?"
    assert q("counting", 72) == "How many tvs can you see?"
    assert q("counting", 49) == "How many knives can you see?"
    assert q("counting", 46) == "How many wine glasses can you see?"
    for c in (20, 35, 56, 87):
        assert q("presence", c) == f"Does this image contain {cats[c]['name']}?"
        assert q("counting", c) == f"How many {cats[c]['name']} can you see?"
    assert len(cats) == 80


def test_pools_drop_items_without_partner():
    cfg, cats = T.config(), T.categories()
    anns = [_inst(1, 80), _inst(2, 80), _inst(3, 18), _inst(3, 18), _inst(4, 18), _inst(4, 1)]
    pools = S.build_pools(S.GroundingIndex(anns, cfg), [1, 2, 3, 4, 5], cats, cfg)
    assert pools["n_unannotated_excluded"] == 1 and pools["images"] == [1, 2, 3, 4]
    assert [c for c in pools["counting_items"] if c[1] == 80] == []      # toaster: all count 1
    assert sorted(c for c in pools["counting_items"] if c[1] == 18) == [[3, 18, 2], [4, 18, 1]]
    assert [4, 1] in pools["presence_positives"]


# ---------------------------------------------------------------- whole pipeline on synthetic data

@pytest.fixture(scope="module")
def synth():
    cfg, cats = T.config(), T.categories()
    rv, rt = T.raw_val(), T.raw_train()
    sets, orders, elig, metas, idx = S.select(cfg, rv, rt, T.FakeTokenizer(), cats)
    files, iso = S.finalize(copy.deepcopy(sets), metas["train"], metas["val"], cfg)
    return dict(cfg=cfg, cats=cats, sets=sets, orders=orders, metas=metas, idx=idx, files=files, iso=iso,
                rv=rv, rt=rt)


def test_selection_counts_and_isolation(synth):
    iso = synth["iso"]
    assert iso["val_images_used_more_than_once"] == 0 and iso["train_images_in_val2014"] == 0
    assert set(iso["pairwise_val_overlap"].values()) == {0}
    f = synth["files"]
    assert len(f["captions_train"]["images"]) == 20 and len(f["captions_dev"]["images"]) == 5
    for r in f["captions_train"]["images"]:
        assert all(c == " ".join(c.split()) for c in r["captions"])  # D6 cleaning
        assert max(r["n_tokens"]) <= 8


def test_selection_is_deterministic(synth):
    sets2, orders2, *_ = S.select(synth["cfg"], T.raw_val(), T.raw_train(), T.FakeTokenizer(), synth["cats"])
    assert S.dumps("x.json", sets2) == S.dumps("x.json", synth["sets"])
    assert S.dumps("o.json.gz", orders2) == S.dumps("o.json.gz", synth["orders"])


def test_selection_takes_first_valid_candidates(synth):
    """Each VQA set holds exactly the first permutation entries that pass the constraints."""
    sets, orders = synth["sets"], synth["orders"]
    picked = [p["pair_id"] for p in sets["vqa_dev"]["pairs"] if p["answer_type"] == "yes/no"]
    order = orders["vqa_dev/yes/no"]
    assert picked == [x for x in order if x in set(picked)]
    assert [p["candidate_rank"] for p in sets["vqa_dev"]["pairs"][:2]] == [order.index(x) for x in picked]


def test_check_all_catches_leakage(synth):
    cfg, files = synth["cfg"], copy.deepcopy(synth["files"])
    tr, va = set(synth["metas"]["train"]), set(synth["metas"]["val"])
    files["captions_train"]["images"][0]["image_id"] = files["vqa_dev"]["pairs"][0]["image_ids"][0]
    with pytest.raises(AssertionError):
        S.check_all(files, tr, va, cfg)
    files = copy.deepcopy(synth["files"])
    files["grounding_test"]["pairs"][0]["image_ids"][1] = files["captions_dev"]["images"][0]["image_id"]
    with pytest.raises(AssertionError):
        S.check_all(files, tr, va, cfg)


def test_mismatch_map(synth):
    mm = synth["files"]["mismatch_map"]["suites"]
    for suite, m in mm.items():
        pairs = {p["pair_id"]: p for p in synth["files"][suite]["pairs"]}
        order = m["shuffled_order"]
        for p, pid in enumerate(order):
            src = order[(p + 1) % len(order)]
            assert m["map"][pid] == {"source_pair": src, "image_ids": pairs[src]["image_ids"]}
            assert not set(m["map"][pid]["image_ids"]) & set(pairs[pid]["image_ids"])
    rng = np.random.default_rng(6304)
    perm = rng.permutation(len(synth["files"]["vqa_test"]["pairs"]))
    assert mm["vqa_test"]["shuffled_order"][0] == synth["files"]["vqa_test"]["pairs"][perm[0]]["pair_id"]


def test_apply_label_flags_replaces_with_next_candidate(synth, tmp_path):
    cfg, cats = synth["cfg"], synth["cats"]
    sets = copy.deepcopy(synth["sets"])
    flagged = sets["grounding_test"]["pairs"][1]
    kind = flagged["type"]
    max_rank = max(p["candidate_rank"] for p in sets["grounding_test"]["pairs"] if p["type"] == kind)
    before = {im for n in S.VAL_SETS for im in S.image_ids_of(n, sets[n])}
    recs = A.apply_flags(sets, synth["orders"], synth["idx"], cats, cfg, [flagged["pair_id"]], [])
    new = sets["grounding_test"]["pairs"][1]
    assert recs[0]["flagged"] == flagged and recs[0]["replacement"] == new and new["type"] == kind
    assert new["candidate_rank"] > max_rank
    assert not set(new["image_ids"]) & before  # unused images only
    expected, _ = S.next_grounding_pair(kind, synth["idx"], synth["orders"][f"grounding_test/{kind}"],
                                        max_rank + 1, before, cats, cfg)
    assert new == expected
    # flagged ids already recorded are skipped; flagged images never come back
    assert A.apply_flags(sets, synth["orders"], synth["idx"], cats, cfg, [flagged["pair_id"]], recs) == []
    second = sets["grounding_test"]["pairs"][1]["pair_id"]
    recs2 = A.apply_flags(sets, synth["orders"], synth["idx"], cats, cfg, [second], recs)
    assert not set(flagged["image_ids"]) & set(recs2[0]["replacement"]["image_ids"])
    with pytest.raises(SystemExit):
        A.apply_flags(sets, synth["orders"], synth["idx"], cats, cfg, ["vqa_1_2"], [])


def test_preflagged_candidates_are_skipped_and_recorded(synth):
    cfg, cats = synth["cfg"], synth["cats"]
    sets = copy.deepcopy(synth["sets"])
    flagged = sets["grounding_test"]["pairs"][1]
    kind = flagged["type"]
    seq = A.preview(sets, synth["orders"], synth["idx"], cats, cfg, [], {("grounding_test", kind): 4})[
        ("grounding_test", kind)]
    pre = [seq[0]["pair_id"], seq[2]["pair_id"]]
    recs = A.apply_flags(sets, synth["orders"], synth["idx"], cats, cfg, [flagged["pair_id"]] + pre, [])
    assert [(r["flagged"]["pair_id"], r.get("preflagged", False)) for r in recs] == [
        (flagged["pair_id"], False), (seq[0]["pair_id"], True)]
    assert recs[0]["replacement"] == seq[1] == sets["grounding_test"]["pairs"][1] and recs[1]["replacement"] is None
    assert A.pending_preflags([flagged["pair_id"]] + pre, [], recs, sets) == [seq[2]["pair_id"]]
    # second round: the replacement fails too; the pending pre-flag is consumed, seq[3] is selected
    flags = [flagged["pair_id"]] + pre + [seq[1]["pair_id"]]
    recs2 = A.apply_flags(sets, synth["orders"], synth["idx"], cats, cfg, flags, recs)
    assert [r["flagged"]["pair_id"] for r in recs2] == [seq[1]["pair_id"], seq[2]["pair_id"]]
    assert recs2[0]["replacement"] == seq[3] and recs2[1]["preflagged"]
    assert A.pending_preflags(flags, recs, recs2, sets) == []
    assert A.apply_flags(sets, synth["orders"], synth["idx"], cats, cfg, flags, recs + recs2) == []
    used = {im for n in S.VAL_SETS for im in S.image_ids_of(n, sets[n])}
    assert not used & {im for p in (seq[0], seq[2], flagged, seq[1]) for im in p["image_ids"]}


@pytest.mark.parametrize("seed", range(6))
def test_preview_sequence_does_not_depend_on_verdicts(synth, seed):
    """Three flags in one stream, random verdicts on the previewed candidates: the slots get, in flag order,
    the stream's candidates that are not pre-flagged, and every failing candidate before them is consumed."""
    cfg, cats = synth["cfg"], synth["cats"]
    sets = copy.deepcopy(synth["sets"])
    kind = sets["grounding_test"]["pairs"][0]["type"]
    slots = [i for i, p in enumerate(sets["grounding_test"]["pairs"]) if p["type"] == kind][:3]
    flags = [sets["grounding_test"]["pairs"][i]["pair_id"] for i in slots]
    seq = A.preview(sets, synth["orders"], synth["idx"], cats, cfg, [], {("grounding_test", kind): 9})[
        ("grounding_test", kind)]
    rng = np.random.default_rng(seed)
    fail = [p["pair_id"] for p in seq if rng.random() < 0.4]
    passing = [p for p in seq if p["pair_id"] not in fail]
    if len(passing) < 3:
        pytest.skip("too few passing candidates for this verdict draw")
    recs = A.apply_flags(sets, synth["orders"], synth["idx"], cats, cfg, flags + fail, [])
    assert [sets["grounding_test"]["pairs"][i] for i in slots] == passing[:3]
    ids = [p["pair_id"] for p in seq]
    last = ids.index(passing[2]["pair_id"])
    assert [r["flagged"]["pair_id"] for r in recs if r.get("preflagged")] == [x for x in fail if ids.index(x) < last]


def test_preview_refuses_streams_sharing_images(synth):
    """With few images, two streams can compete for an image; then the interleaving (and so the verdicts)
    could change a sequence, and the preview refuses."""
    with pytest.raises(AssertionError, match="share images"):
        A.preview(copy.deepcopy(synth["sets"]), synth["orders"], synth["idx"], synth["cats"], synth["cfg"], [],
                  {("grounding_test", "presence"): 6, ("grounding_test", "counting"): 6})


def test_apply_label_flags_rewrites_files(synth, tmp_path):
    """End to end through rewrite(): split dir with a fake flag, asserts rerun, mismatch map follows."""
    cfg = copy.deepcopy(synth["cfg"])
    cfg["out_dir"] = str(tmp_path)
    cats, sets = synth["cats"], copy.deepcopy(synth["sets"])
    files, iso = S.finalize(copy.deepcopy(sets), synth["metas"]["train"], synth["metas"]["val"], cfg)
    blobs = S.file_blobs(files, synth["orders"])
    blobs[S.REPORT] = S.report_blob(files, blobs, cfg, cats, {"isolation": iso, "label_flags": None})
    S.write_files(str(tmp_path), blobs, overwrite=False)
    sets2, orders2, report = A.load_split_dir(str(tmp_path))
    fake = sets2["grounding_test"]["pairs"][0]["pair_id"]
    recs = A.apply_flags(sets2, orders2, synth["idx"], cats, cfg, [fake], [])
    A.rewrite(str(tmp_path), sets2, orders2, report, recs, [], synth["metas"], cfg, cats)
    g = json.load(open(tmp_path / "grounding_test.json"))
    new_id = recs[0]["replacement"]["pair_id"]
    assert g["pairs"][0]["pair_id"] == new_id
    mm = json.load(open(tmp_path / "mismatch_map.json"))["suites"]["grounding_test"]
    assert new_id in mm["map"] and fake not in mm["map"]
    old_mm = files["mismatch_map"]["suites"]["grounding_test"]["shuffled_order"]
    assert mm["shuffled_order"] == [new_id if x == fake else x for x in old_mm]  # positions unchanged
    rep = json.load(open(tmp_path / S.REPORT))
    assert rep["label_flags"]["replaced"] == {fake: new_id} and rep["label_flags"]["preflagged_skipped"] == []
    for name, h in rep["sha256"].items():
        assert hashlib.sha256(open(tmp_path / name, "rb").read()).hexdigest() == h
    lf = json.load(open(tmp_path / A.LABEL_FLAGS))
    assert [r["flagged"]["pair_id"] for r in lf["applied"]] == [fake]
    ids = {r["image_id"] for r in json.load(open(tmp_path / "images_val.json"))["images"]}
    assert set(recs[0]["replacement"]["image_ids"]) <= ids
    assert not set(recs[0]["flagged"]["image_ids"]) & ids - set(recs[0]["replacement"]["image_ids"])


def test_preflagged_skips_are_written(synth, tmp_path):
    cfg = copy.deepcopy(synth["cfg"])
    cfg["out_dir"] = str(tmp_path)
    cats, sets = synth["cats"], copy.deepcopy(synth["sets"])
    files, iso = S.finalize(copy.deepcopy(sets), synth["metas"]["train"], synth["metas"]["val"], cfg)
    blobs = S.file_blobs(files, synth["orders"])
    blobs[S.REPORT] = S.report_blob(files, blobs, cfg, cats, {"isolation": iso, "label_flags": None})
    S.write_files(str(tmp_path), blobs, overwrite=False)
    sets2, orders2, report = A.load_split_dir(str(tmp_path))
    fake = sets2["grounding_test"]["pairs"][0]
    seq = A.preview(sets2, orders2, synth["idx"], cats, cfg, [], {("grounding_test", fake["type"]): 2})[
        ("grounding_test", fake["type"])]
    recs = A.apply_flags(sets2, orders2, synth["idx"], cats, cfg, [fake["pair_id"], seq[0]["pair_id"]], [])
    A.rewrite(str(tmp_path), sets2, orders2, report, recs, [], synth["metas"], cfg, cats)
    lf = json.load(open(tmp_path / A.LABEL_FLAGS))["applied"]
    assert [(r["flagged"]["pair_id"], r.get("preflagged", False)) for r in lf] == [
        (fake["pair_id"], False), (seq[0]["pair_id"], True)]
    rep = json.load(open(tmp_path / S.REPORT))["label_flags"]
    assert rep["replaced"] == {fake["pair_id"]: seq[1]["pair_id"]} and rep["preflagged_skipped"] == [seq[0]["pair_id"]]
    ids = {r["image_id"] for r in json.load(open(tmp_path / "images_val.json"))["images"]}
    assert not set(seq[0]["image_ids"]) & ids


def test_write_files_refuses_changes(tmp_path):
    S.write_files(str(tmp_path), {"a.json": b"1"}, overwrite=False)
    with pytest.raises(SystemExit):
        S.write_files(str(tmp_path), {"a.json": b"2"}, overwrite=False)
    S.write_files(str(tmp_path), {S.REPORT: b"x"}, overwrite=False)
    S.write_files(str(tmp_path), {S.REPORT: b"y"}, overwrite=False)  # report always rewritten


def test_gzip_bytes_are_deterministic():
    assert S.dumps("o.json.gz", {"a": [1, 2]}) == S.dumps("o.json.gz", {"a": [1, 2]})


# ---------------------------------------------------------------- committed split files

@pytest.mark.skipif(not os.path.exists(os.path.join(SPLITS, S.REPORT)), reason="splits not built")
def test_committed_splits_match_report_and_asserts():
    rep = json.load(open(os.path.join(SPLITS, S.REPORT)))
    for name, h in rep["sha256"].items():
        with open(os.path.join(SPLITS, name), "rb") as f:
            assert hashlib.sha256(f.read()).hexdigest() == h, name
    cfg = S.load_config(os.path.join(REPO, S.CONFIG))
    files = {n: json.load(open(os.path.join(SPLITS, f"{n}.json")))
             for n in S.VAL_SETS + S.TRAIN_SETS + ("images_train", "images_val", "mismatch_map")}
    tr = {r["image_id"] for r in files["images_train"]["images"]}
    va = {r["image_id"] for r in files["images_val"]["images"]}
    assert not tr & va
    iso = S.check_all(files, tr, va, cfg)
    assert iso == rep["isolation"]
    with gzip.open(os.path.join(SPLITS, "candidate_orders.json.gz")) as f:
        orders = json.load(f)
    for s in ("dev", "test"):
        for p in files[f"grounding_{s}"]["pairs"]:
            assert orders[f"grounding_{s}/{p['type']}"]["items"][p["candidate_rank"]][:2] == \
                [p["image_ids"][0], p["category_id"]]
