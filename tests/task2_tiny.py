"""Tiny Task 2 inputs for CPU tests: subsets of the real split files, synthetic grounding pools, random FP16
features in the cache_features format, a tiny random Qwen3 and the matching config overrides."""
import copy
import json
import os

import torch
import yaml

from task2_vlm.data import sha256_file

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SPLITS = os.path.join(REPO, "splits/task2")
HIDDEN = 64
CONNECTOR_PARAMS = 768 * HIDDEN + HIDDEN + HIDDEN * HIDDEN + HIDDEN          # 53,376
LORA_PARAMS = 2 * 16 * ((64 + 64) + (64 + 32) + (64 + 32) + (64 + 64))     # 2 layers, q/k/v/o: 14,336


def real(name):
    with open(os.path.join(SPLITS, f"{name}.json")) as f:
        return json.load(f)


def tiny_lm(seed=0):
    from transformers import Qwen3Config, Qwen3ForCausalLM
    torch.manual_seed(seed)
    c = Qwen3Config(vocab_size=151936, hidden_size=HIDDEN, intermediate_size=96, num_hidden_layers=2,
                    num_attention_heads=4, num_key_value_heads=2, head_dim=16, tie_word_embeddings=True,
                    max_position_embeddings=512)
    return Qwen3ForCausalLM(c).float()


def tiny_pools(images, questions):
    cats = {}
    for k, i in enumerate(images):
        cats[i] = [[1, 18], [1], [17]][k % 3]
    pos = sorted([i, c] for i in images for c in cats[i])
    cnt = {}
    for k, i in enumerate(images):
        for c in cats[i]:
            cnt[(i, c)] = 1 + (k % 4 if c == 1 else (k // 3) % 2 if c == 18 else k % 2)
    items = [[i, c, m] for (i, c), m in sorted(cnt.items())
             if any(c2 == c and m2 != m for (_, c2), m2 in cnt.items())]
    return {"images": images, "image_categories": {str(i): cats[i] for i in images}, "presence_positives": pos,
            "counting_items": items, "questions": questions}


def write_features(out_dir, split, ids, seed, shard=5):
    from safetensors.torch import save_file
    os.makedirs(out_dir, exist_ok=True)
    g = torch.Generator().manual_seed(seed)
    shards, index = [], {}
    for k, s in enumerate(range(0, len(ids), shard)):
        part = ids[s:s + shard]
        x = (torch.randn(len(part), 196, 768, generator=g) * 2.2).to(torch.float16)
        fn = f"{split}_{k:03d}.safetensors"
        save_file({"features": x}, os.path.join(out_dir, fn), metadata={"image_ids": json.dumps(part)})
        shards.append({"file": fn, "n": len(part), "sha256": sha256_file(os.path.join(out_dir, fn)), "image_ids": part})
        for r, i in enumerate(part):
            index[str(i)] = [k, r]
    with open(os.path.join(out_dir, f"{split}_index.json"), "w") as f:
        json.dump({"split": split, "dtype": "float16", "shape_per_image": [196, 768], "n_images": len(ids),
                   "shards": shards, "index": index}, f)


def cyclic_map(pairs):
    return {p["pair_id"]: {"source_pair": pairs[(k + 1) % len(pairs)]["pair_id"],
                           "image_ids": pairs[(k + 1) % len(pairs)]["image_ids"]} for k, p in enumerate(pairs)}


def build(tmp):
    """Write tiny split files and features under tmp; return the data paths and the features dir."""
    tmp = str(tmp)
    sp = os.path.join(tmp, "splits")
    os.makedirs(sp, exist_ok=True)
    ct, cd = real("captions_train"), real("captions_dev")
    ct["images"], cd["images"] = ct["images"][:4], cd["images"][:4]
    vt, vd, vs = real("vqa_train"), real("vqa_dev"), real("vqa_test")
    vt["pairs"], vd["pairs"] = vt["pairs"][:6], vd["pairs"][:4]
    vs["pairs"] = [p for t in ("yes/no", "number", "other") for p in vs["pairs"] if p["answer_type"] == t][::12][:6]
    gd, gs = real("grounding_dev"), real("grounding_test")
    gd["pairs"] = [p for t in ("presence", "counting") for p in [q for q in gd["pairs"] if q["type"] == t][:2]]
    gs["pairs"] = [p for t in ("presence", "counting") for p in [q for q in gs["pairs"] if q["type"] == t][:3]]
    train_ids = sorted({r["image_id"] for r in ct["images"]} | {i for p in vt["pairs"] for i in p["image_ids"]})
    pools = tiny_pools(train_ids, real("grounding_pools_train")["questions"])
    val_ids = sorted({r["image_id"] for r in cd["images"]} | {i for s in (vd, vs, gd, gs)
                                                              for p in s["pairs"] for i in p["image_ids"]})
    mm = {"rule": "tiny test map: cyclic in file order",
          "suites": {"vqa_test": {"map": cyclic_map(vs["pairs"])}, "grounding_test": {"map": cyclic_map(gs["pairs"])}}}
    url = "http://images.cocodataset.org/{}/{}"
    it = {"split": "train2014", "images": [{"image_id": i, "file_name": f"COCO_train2014_{i:012d}.jpg",
                                            "url": url.format("train2014", i)} for i in train_ids]}
    iv = {"split": "val2014", "images": [{"image_id": i, "file_name": f"COCO_val2014_{i:012d}.jpg",
                                          "url": url.format("val2014", i)} for i in val_ids]}
    files = {"images_train": it, "images_val": iv, "captions_train": ct, "captions_dev": cd, "vqa_train": vt,
             "vqa_dev": vd, "vqa_test": vs, "grounding_pools_train": pools, "grounding_dev": gd,
             "grounding_test": gs, "mismatch_map": mm}
    data = {}
    for k, v in files.items():
        data[k] = os.path.join(sp, f"{k}.json")
        with open(data[k], "w") as f:
            json.dump(v, f)
    feats = os.path.join(tmp, "features")
    write_features(feats, "train", train_ids, 1)
    write_features(feats, "val", val_ids, 2)
    return data, feats


def tiny_cfg(tmp, data, feats):
    with open(os.path.join(REPO, "configs/task2/train.yaml")) as f:
        cfg = yaml.safe_load(f)
    cfg = copy.deepcopy(cfg)
    tmp = str(tmp)
    cfg["data"] = data
    cfg["features"]["dir"] = feats
    cfg["connector"].update(hidden_dim=HIDDEN, out_dim=HIDDEN)
    cfg["expected_trainable_params"] = [CONNECTOR_PARAMS, CONNECTOR_PARAMS + LORA_PARAMS,
                                        CONNECTOR_PARAMS + LORA_PARAMS]
    cfg["batch"].update(microbatch=4, grad_accum=2, effective=8, draws_per_microbatch=2)
    for st, n in zip(cfg["stages"], (3, 3, 2)):
        st["updates"] = n
    cfg["dev_eval"]["updates"] = [0, 2, 3, 6, 8]
    cfg["checkpoints"]["recovery_every"] = 2
    cfg["calibration"]["n_images"] = 4
    cfg["logging"].update(json_every=1, print_every=1)
    cfg["loss"]["logit_chunk"] = 5
    cfg["smoke"].update(updates_per_stage=2, recovery_every=2, final_pairs_per_suite=2)
    cfg["output"] = {"result_dir": os.path.join(tmp, "results/main"), "smoke_result_dir": os.path.join(tmp, "results/smoke"),
                     "keep_dir": os.path.join(tmp, "ckpt/main"), "smoke_keep_dir": os.path.join(tmp, "ckpt/smoke"),
                     "recovery_dir": os.path.join(tmp, "recovery")}
    return cfg


def write_cfg(tmp, cfg):
    p = os.path.join(str(tmp), "train.yaml")
    with open(p, "w") as f:
        yaml.safe_dump(cfg, f)
    return p
