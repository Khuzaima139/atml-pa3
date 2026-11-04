"""Task 2 final evaluation of the Stage 3 endpoint only (manual L428-457, L484-487).

Both final-test suites under four image conditions with the question and the original references fixed (L431-438):
    correct     the pair's own image for the slot
    swap        the partner image of the pair (complementary swap)
    mismatched  same slot of the pair given by splits/task2/mismatch_map.json (fixed before generation, D9)
    none        the 196 visual vectors removed (D11)
Greedy decoding: argmax per step with a KV cache, left padding, explicit position ids from the mask, thinking
disabled, 16 new tokens, batch 8, a row stops at <|im_end|>. Reference NLL (dev definition) under all four
conditions. Metrics: grounding exact match (presence / counting, counting also by true count), VQA soft accuracy
(yes/no / number / other), pair accuracy, identical-answer and empty-answer rates, swap follow-partner rate,
generation time per answer, correct-image advantages with complete-pair bootstrap CIs (L455-457).

Run: python -m task2_vlm.evaluate --features_dir /kaggle/working/features [--smoke]
"""
import argparse
import copy
import datetime
import os
import time

import torch

from common.results import read_result, write_result
from common.runinfo import run_info
from common.seed import seed_everything
from task2_vlm import model as M
from task2_vlm import train as T
from task2_vlm.data import Encoder, FeatureStore, check_template, collate, load_splits, pair_target, pair_type, \
    sha256_file
from task2_vlm.scoring import bootstrap_ci, bootstrap_indices, mean, pair_metrics, paired_difference, \
    reference_nll, score_prediction

CORRECT = "correct"


def condition_image(cond, pair, slot, mismatch):
    if cond == "correct":
        return pair["image_ids"][slot]
    if cond == "swap":
        return pair["image_ids"][1 - slot]
    if cond == "mismatched":
        m = mismatch[pair["pair_id"]]
        assert not set(m["image_ids"]) & set(pair["image_ids"]), "mismatched image from the original pair"
        return m["image_ids"][slot]
    assert cond == "none"
    return None


def endpoint_files(doc):
    """Connector and LoRA weights saved at the end of the last stage of a finished run; sha256 checked."""
    assert doc["status"] == "done", f"training run {doc['run_id']} is {doc['status']}"
    last = len(doc["config"]["stages"])
    total = doc["resources"]["updates_total"]
    out = {}
    for kind in ("connector", "lora"):
        es = [e for e in doc["files"] if e["kind"] == kind and e.get("stage") == last]
        assert len(es) == 1 and es[0]["update"] == total, f"{kind}: no single Stage {last} file at update {total}"
        assert sha256_file(es[0]["path"]) == es[0]["sha256"], f"sha256 mismatch: {es[0]['path']}"
        out[kind] = es[0]
    return out


def generation_settings(cfg, tok):
    g = dict(cfg["generation"])
    fe = cfg["final_eval"]
    assert g["max_new_tokens"] == fe["max_new_tokens"] and not g["do_sample"]
    g.update(batch_size=fe["batch_size"], enable_thinking=cfg["model"]["enable_thinking"],
             pad_token_id=cfg["model"]["pad_id"], stop_tokens=tok.convert_ids_to_tokens(g["stop_token_ids"]),
             decoding="argmax of FP32 logits per step (first index on ties), KV cache; position ids = "
                      "cumsum(attention_mask) - 1; a row stops after emitting <|im_end|>; the batch stops when "
                      "every row stopped or after max_new_tokens",
             raw_generation="tokens before <|im_end|>, decoded with skip_special_tokens=False",
             prediction="vqa_norm.normalize(raw_generation)")
    return g


def run_suite(vlm, enc, tok, suite, pairs, mismatch, store, cfg):
    fe = cfg["final_eval"]
    bs, stop, pad = fe["batch_size"], cfg["generation"]["stop_token_ids"], cfg["model"]["pad_id"]
    assert len(stop) == 1
    rows, timing = [], {}
    for cond in fe["conditions"]:
        meta, gen_ex, nll_ex = [], [], []
        for p in pairs:
            for s in (0, 1):
                img = condition_image(cond, p, s, mismatch)
                meta.append((p, s, img))
                gen_ex.append(enc.example(p["question"], None, img, "answer"))
                nll_ex.append(enc.example(p["question"], pair_target(suite, p, s), img, "answer"))
        outs, gen_s = [], 0.0
        for s in range(0, len(gen_ex), bs):
            exs = gen_ex[s:s + bs]
            b = {k: v.to(vlm.device) for k, v in collate(exs, enc.n_vis, enc.pad, "left").items()}
            imgs = [e["image"] for e in exs if e["image"] is not None]
            feats = store.get(imgs).to(vlm.device) if imgs else None
            T._sync(vlm.device)
            t0 = time.perf_counter()
            with vlm.autocast():
                outs += vlm.greedy(b, feats, fe["max_new_tokens"], stop[0], pad)
            T._sync(vlm.device)
            gen_s += time.perf_counter() - t0
        T._sync(vlm.device)
        t1 = time.perf_counter()
        nll = reference_nll(vlm, enc, nll_ex, store, bs, cfg["loss"]["logit_chunk"])
        T._sync(vlm.device)
        timing[cond] = {"generation_s": gen_s, "nll_s": time.perf_counter() - t1, "answers": len(outs),
                        "generation_s_per_answer": gen_s / len(outs)}
        for (p, s, img), o, n in zip(meta, outs, nll):
            raw = tok.decode(o["token_ids"], skip_special_tokens=False)
            r = {"suite": suite, "condition": cond, "pair_id": p["pair_id"], "slot": s, "type": pair_type(suite, p),
                 "question": p["question"], "image_id": img, "pair_image_ids": p["image_ids"],
                 "question_id": p["question_ids"][s] if "question_ids" in p else None,
                 "nll_target": pair_target(suite, p, s), "raw_generation": raw, "token_ids": o["token_ids"],
                 "stopped": o["stopped"], "n_new_tokens": o["n_new_tokens"],
                 **score_prediction(suite, p, s, raw), "reference_nll": n["answer_nll"],
                 "n_target_tokens": n["n_content"]}
            if suite.startswith("vqa"):
                r["references"] = p["answers"][s]
            else:
                r["true_count"] = p["counts"][s]
            rows.append(r)
    return rows, timing


def condition_metrics(rows):
    m = {"n": len(rows), "accuracy": mean(r["accuracy"] for r in rows), "exact_match": mean(r["exact"] for r in rows),
         **pair_metrics(rows), "empty_rate_normalized": mean(r["empty_normalized"] for r in rows),
         "empty_rate_raw": mean(r["empty_raw"] for r in rows),
         "follow_partner_rate": mean(r["equals_partner"] for r in rows),
         "mean_reference_nll": mean(r["reference_nll"] for r in rows),
         "not_stopped": sum(not r["stopped"] for r in rows)}
    return m


def summarize(suite, rows, timing, cfg):
    fe = cfg["final_eval"]
    bt = fe["bootstrap"]
    types = sorted({r["type"] for r in rows})
    out = {"conditions": {}, "bootstrap": {}}
    for c in fe["conditions"]:
        rc = [r for r in rows if r["condition"] == c]
        m = condition_metrics(rc)
        m["generation_s_per_answer"] = timing[c]["generation_s_per_answer"]
        m["by_type"] = {t: condition_metrics([r for r in rc if r["type"] == t]) for t in types}
        if "counting" in types:
            cnt = [r for r in rc if r["type"] == "counting"]
            m["counting_by_true_count"] = {str(k): {"n": sum(r["true_count"] == k for r in cnt),
                                                    "accuracy": mean(r["accuracy"] for r in cnt if r["true_count"] == k)}
                                           for k in (1, 2, 3, 4)}
        out["conditions"][c] = m
    for subset in ["all"] + types:
        sel = [r for r in rows if subset == "all" or r["type"] == subset]
        order = list(dict.fromkeys(r["pair_id"] for r in sel))
        idx = bootstrap_indices(len(order), bt["resamples"], bt["seed"])
        per = {}
        for c in fe["conditions"]:
            by = {}
            for r in sel:
                if r["condition"] == c:
                    by.setdefault(r["pair_id"], []).append(r)
            assert all(len(by[p]) == 2 for p in order)
            per[c] = {"accuracy": [(by[p][0]["accuracy"] + by[p][1]["accuracy"]) / 2 for p in order],
                      "reference_nll": [(by[p][0]["reference_nll"] + by[p][1]["reference_nll"]) / 2 for p in order]}
        pc = bt["percentiles"]
        out["bootstrap"][subset] = {
            "n_pairs": len(order), "resamples": bt["resamples"], "seed": bt["seed"], "percentiles": pc,
            "conditions": {c: {k: bootstrap_ci(v, idx, pc) for k, v in per[c].items()} for c in per},
            "correct_advantage_over": {
                o: {"accuracy": paired_difference(per[CORRECT]["accuracy"], per[o]["accuracy"], idx, pc),
                    "reference_nll": paired_difference(per[o]["reference_nll"], per[CORRECT]["reference_nll"], idx, pc)}
                for o in fe["conditions"] if o != CORRECT}}
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=T.CONFIG)
    ap.add_argument("--features_dir")
    ap.add_argument("--result_dir")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args(argv)
    main_cfg = T.load_config(args.config)
    cfg = T.smoke_config(main_cfg) if args.smoke else copy.deepcopy(main_cfg)
    T.apply_overrides(cfg, args)
    run_dir = cfg["output"]["result_dir"]
    train_doc = read_result(os.path.join(run_dir, "train.json"))
    files = endpoint_files(train_doc)
    device = T.training_device()
    seed_everything(cfg["seed"])
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    kind = "smoke_eval" if args.smoke else "eval"
    out_dir = os.path.join(run_dir, "final")
    doc = {"run_id": f"task2_{kind}_{stamp}", "kind": kind, "status": "running", "config": cfg,
           "provenance": run_info(), "versions": T.versions(), "train_run_id": train_doc["run_id"],
           "endpoint": {k: {x: v[x] for x in ("path", "sha256", "update", "stage")} for k, v in files.items()}}
    eval_path = os.path.join(out_dir, "eval.json")
    write_result(eval_path, doc)
    t_start = time.time()

    t0 = time.perf_counter()
    tok = T.load_tokenizer(cfg)
    check_template(tok, cfg)
    enc = Encoder(tok, cfg)
    d = load_splits(cfg)
    f = cfg["features"]
    store = FeatureStore(f["dir"], "val", f["tokens"], f["dim"], f["verify_sha256"])
    lm = M.load_qwen(cfg, device)
    vlm = M.VLM(cfg, lm, device)
    from safetensors.torch import load_file
    vlm.connector.load_state_dict(load_file(files["connector"]["path"]))
    vlm.add_lora()
    vlm.load_lora_state(load_file(files["lora"]["path"]))
    vlm.eval()
    doc["setup_s"] = time.perf_counter() - t0
    doc["calibration_scale"] = float(vlm.connector.scale)
    assert abs(doc["calibration_scale"] - train_doc["calibration"]["stored_s_float32"]) == 0.0
    doc["dtypes"] = {"backbone": str(next(lm.parameters()).dtype).replace("torch.", ""), "lora": "float32",
                     "connector": "float32", "compute": "float16 autocast" if vlm.amp else "float32"}
    doc["generation"] = generation_settings(cfg, tok)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    n_pairs = cfg["smoke"]["final_pairs_per_suite"] if args.smoke else None
    mm = d["mismatch_map"]["suites"]
    all_rows, doc["suites"], doc["timing"] = [], {}, {}
    for suite in cfg["final_eval"]["suites"]:
        pairs = d[suite]["pairs"][:n_pairs] if n_pairs else d[suite]["pairs"]
        rows, timing = run_suite(vlm, enc, tok, suite, pairs, mm[suite]["map"], store, cfg)
        all_rows += rows
        doc["timing"][suite] = timing
        doc["suites"][suite] = {"n_pairs": len(pairs), **summarize(suite, rows, timing, cfg)}
        c = doc["suites"][suite]["conditions"]
        print(f"{suite}: " + " | ".join(f"{k} acc {v['accuracy']:.3f} pair {v['pair_accuracy']:.3f} nll "
                                        f"{v['mean_reference_nll']:.3f}" for k, v in c.items()), flush=True)
        write_result(eval_path, doc)
    if device.type == "cuda":
        doc["peak_allocated_bytes"] = torch.cuda.max_memory_allocated(device)
        doc["peak_reserved_bytes"] = torch.cuda.max_memory_reserved(device)
    else:
        doc["peak_allocated_bytes"] = doc["peak_reserved_bytes"] = None
    ex_path = os.path.join(out_dir, "examples.json")
    write_result(ex_path, {"run_id": doc["run_id"], "fields": "one row per suite x condition x pair x slot",
                           "rows": all_rows})
    doc["examples_file"] = {"path": ex_path, "sha256": sha256_file(ex_path), "rows": len(all_rows)}
    doc["wall_s"] = time.time() - t_start
    doc["status"] = "done"
    write_result(eval_path, doc)
    print(f"final evaluation done: {len(all_rows)} answers, {doc['wall_s']:.1f}s -> {eval_path}", flush=True)
    return doc


if __name__ == "__main__":
    main()
