"""Task 2 data for training and evaluation: split files, cached features, prompt encoding and collation.

Prompt layout (decision D10, notes/TASK2_FACTS.md step 20), each part tokenized alone (add_special_tokens=False):
    tok("<|im_start|>user\\n") + [196 visual vectors] + tok("\\n" + question + "<|im_end|>\\n<|im_start|>assistant\\n"
    "<think>\\n\\n</think>\\n\\n") + tok(target) + [<|im_end|>]
No-image inputs (D11) drop the 196 vectors only; the text token ids are unchanged. Visual positions hold the pad
id in input_ids and are replaced by connector outputs (vis_mask). Labels are IGNORE except on target tokens and the
final <|im_end|>; `content` marks target tokens without <|im_end|> (answer NLL, L425-426).
"""
import hashlib
import json
import os

import torch

IGNORE = -100


def load_json(path):
    with open(path) as f:
        return json.load(f)


# ---------------------------------------------------------------- split files

def load_splits(cfg):
    """All split files named in cfg["data"], plus derived example tables. Asserts split isolation (L370-372)."""
    d = {k: load_json(p) for k, p in cfg["data"].items()}
    train_ids = {r["image_id"] for r in d["images_train"]["images"]}
    val_ids = {r["image_id"] for r in d["images_val"]["images"]}
    assert not train_ids & val_ids, "training and evaluation images overlap"
    caps = d["captions_train"]
    d["caption_examples"] = [(r["image_id"], c) for r in caps["images"] for c in r["captions"]]
    d["caption_dev_examples"] = [(r["image_id"], r["caption"]) for r in d["captions_dev"]["images"]]
    d["instruction"] = caps["instruction"]
    assert d["captions_dev"]["instruction"] == d["instruction"]
    used_train = {i for i, _ in d["caption_examples"]}
    used_train |= {i for p in d["vqa_train"]["pairs"] for i in p["image_ids"]}
    used_train |= set(d["grounding_pools_train"]["images"])
    assert used_train <= train_ids, "a training example uses an image outside images_train.json"
    used_val = {i for i, _ in d["caption_dev_examples"]}
    for k in ("vqa_dev", "vqa_test", "grounding_dev", "grounding_test"):
        used_val |= {i for p in d[k]["pairs"] for i in p["image_ids"]}
    for m in d["mismatch_map"]["suites"].values():
        used_val |= {i for v in m["map"].values() for i in v["image_ids"]}
    assert used_val <= val_ids, "an evaluation example uses an image outside images_val.json"
    return d


def pair_type(suite, pair):
    return pair["answer_type"] if suite.startswith("vqa") else pair["type"]


def pair_target(suite, pair, slot):
    """Reference string for NLL and training: VQA training target (D3), grounding canonical answer."""
    return pair["targets"][slot] if suite.startswith("vqa") else pair["answers"][slot]


def pair_canonical(suite, pair, slot):
    """Canonical answer for exact match / pair metrics: VQA normalized consensus (D1), grounding answer."""
    return pair["consensus"][slot] if suite.startswith("vqa") else pair["answers"][slot]


# ---------------------------------------------------------------- cached SigLIP features

def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 24), b""):
            h.update(block)
    return h.hexdigest()


class FeatureStore:
    """FP16 features of one split ({split}_index.json + shards written by task2_vlm.cache_features), in RAM."""

    def __init__(self, features_dir, split, tokens=196, dim=768, verify=True):
        idx = load_json(os.path.join(features_dir, f"{split}_index.json"))
        assert idx["dtype"] == "float16" and idx["shape_per_image"] == [tokens, dim], idx["shape_per_image"]
        from safetensors.torch import load_file
        self.x = torch.empty((idx["n_images"], tokens, dim), dtype=torch.float16)
        self.row, r = {}, 0
        for sh in idx["shards"]:
            path = os.path.join(features_dir, sh["file"])
            if verify:
                assert sha256_file(path) == sh["sha256"], f"sha256 mismatch: {path}"
            t = load_file(path)["features"]
            assert t.dtype == torch.float16 and t.shape[0] == sh["n"] == len(sh["image_ids"])
            self.x[r:r + sh["n"]] = t
            for k, i in enumerate(sh["image_ids"]):
                self.row[int(i)] = r + k
            r += sh["n"]
        assert r == idx["n_images"] == len(self.row)
        self.info = {"dir": features_dir, "split": split, "n_images": r, "verified_sha256": verify,
                     "shards": [{"file": s["file"], "sha256": s["sha256"]} for s in idx["shards"]]}

    def get(self, ids):
        return self.x[[self.row[int(i)] for i in ids]]


# ---------------------------------------------------------------- prompt encoding

class Encoder:
    def __init__(self, tok, cfg):
        self.tok = tok
        self.p = cfg["prompt"]
        self.caps = cfg["caps"]
        self.n_vis = self.p["visual_tokens"]
        self.im_end = cfg["model"]["im_end_id"]
        self.pad = cfg["model"]["pad_id"]
        self.pre = self.ids(self.p["user_prefix"])
        assert self.ids(self.p["end"]) == [self.im_end]
        self._post, self._target = {}, {}

    def ids(self, text):
        return self.tok(text, add_special_tokens=False)["input_ids"]

    def post(self, question):
        if question not in self._post:
            assert len(self.ids(question)) <= self.caps["question"], f"question over cap: {question!r}"
            self._post[question] = self.ids(self.p["after_visual"] + question + self.p["assistant_prefix"])
        return self._post[question]

    def target(self, text, kind):
        key = (text, kind)
        if key not in self._target:
            t = self.ids(text)
            cap = self.caps["caption_target"] if kind == "caption" else self.caps["answer_target"]
            assert len(t) <= cap, f"{kind} target over cap ({len(t)} > {cap}): {text!r}"
            self._target[key] = t + [self.im_end]
        return self._target[key]

    def example(self, question, target, image, kind):
        """image: an image id or None (no-image input). target None: prompt only (generation)."""
        return {"pre": self.pre, "post": self.post(question),
                "target": None if target is None else self.target(target, kind), "image": image}


def check_template(tok, cfg):
    """The prompt strings equal Qwen's chat template with thinking disabled around a placeholder (D10)."""
    p = cfg["prompt"]
    mark = "@@VISUAL@@"
    s = tok.apply_chat_template([{"role": "user", "content": mark + p["after_visual"] + "Q?"}],
                                add_generation_prompt=True, tokenize=False,
                                enable_thinking=cfg["model"]["enable_thinking"])
    expect = p["user_prefix"] + mark + p["after_visual"] + "Q?" + p["assistant_prefix"]
    assert s == expect, f"chat template differs: {s!r} vs {expect!r}"
    return s


def collate(examples, n_vis, pad_id, side="right"):
    """Pad a list of encoded examples. Returns input_ids, vis_mask, attention_mask, position_ids (from the
    mask), labels (IGNORE except target tokens and <|im_end|>), content (target tokens without <|im_end|>)."""
    rows = []
    for e in examples:
        nv = n_vis if e["image"] is not None else 0
        tgt = e["target"] or []
        ids = e["pre"] + [pad_id] * nv + e["post"] + tgt
        n_prompt = len(e["pre"]) + nv + len(e["post"])
        vis = [0] * len(e["pre"]) + [1] * nv + [0] * (len(e["post"]) + len(tgt))
        lab = [IGNORE] * n_prompt + tgt
        con = [0] * n_prompt + [1] * max(len(tgt) - 1, 0) + [0] * min(len(tgt), 1)
        rows.append((ids, vis, lab, con))
    L = max(len(r[0]) for r in rows)
    out = {k: [] for k in ("input_ids", "vis_mask", "attention_mask", "labels", "content")}
    for ids, vis, lab, con in rows:
        n = L - len(ids)
        padded = [(ids, pad_id), (vis, 0), ([1] * len(ids), 0), (lab, IGNORE), (con, 0)]
        for k, (seq, v) in zip(out, padded):
            out[k].append(seq + [v] * n if side == "right" else [v] * n + seq)
    b = {k: torch.tensor(v, dtype=torch.long) for k, v in out.items()}
    b["vis_mask"] = b["vis_mask"].bool()
    b["content"] = b["content"].bool()
    b["position_ids"] = (b["attention_mask"].cumsum(-1) - 1).clamp(min=0)
    return b
