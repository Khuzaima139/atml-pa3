"""Seeded training sampler (manual L468-483). One np.random.default_rng(6304) for the whole run; state is saved.

Stage 1: passes over the 18,000 caption examples, a fresh permutation per pass, consecutive examples per
microbatch. Stages 2-3: each draw picks a component with the stage probabilities (no quotas) and gives two
adjacent examples:
    presence  positive item uniform from the pool, partner uniform among pool images without the category
    counting  item uniform from the pool, partner uniform among pool images of the category with another count
    caption   two caption examples, each uniform over the 18,000
    vqa       one of the 1,536 training pairs, uniform; both questions
Items are references, turned into text by train.py:
    ("caption", idx) | ("vqa", pair_idx, slot) | ("presence", image, category, "yes"/"no") |
    ("counting", image, category, count)
"""
import collections
import copy

import numpy as np


class Sampler:
    def __init__(self, n_captions, n_vqa_pairs, pools, seed):
        self.rng = np.random.default_rng(seed)
        self.n_captions, self.n_vqa = n_captions, n_vqa_pairs
        images = [int(i) for i in pools["images"]]
        cats_of = {int(i): set(c) for i, c in pools["image_categories"].items()}
        self.positives = [(int(i), int(c)) for i, c in pools["presence_positives"]]
        self.negatives = {c: [i for i in images if c not in cats_of[i]] for c in sorted({c for _, c in self.positives})}
        self.counting = [(int(i), int(c), int(m)) for i, c, m in pools["counting_items"]]
        by_cat = collections.defaultdict(list)
        for i, c, m in self.counting:
            by_cat[c].append((i, m))
        self.count_partners = {(c, m): [(i, n) for i, n in by_cat[c] if n != m] for _, c, m in self.counting}
        assert all(self.negatives.values()) and all(self.count_partners.values()), "pool item without a partner"
        self.stage = None

    # -- stages
    def start_stage(self, k, stage_cfg):
        self.stage, self.stage_cfg = k, stage_cfg
        self.kinds = list(stage_cfg.get("mixture") or {})
        self.probs = np.array([stage_cfg["mixture"][x] for x in self.kinds]) if self.kinds else None
        if self.kinds:
            assert abs(self.probs.sum() - 1.0) < 1e-9
        self.perm, self.cursor, self.pass_no = None, 0, 0
        self.counts = {"draws": {x: 0 for x in self.kinds}, "examples": collections.Counter()}

    def _caption_batch(self, size):
        out = []
        while len(out) < size:
            if self.perm is None or self.cursor == len(self.perm):
                self.pass_no += 1
                assert self.pass_no <= self.stage_cfg["caption_passes"], "more caption passes than configured"
                self.perm, self.cursor = self.rng.permutation(self.n_captions), 0
            out.append(("caption", int(self.perm[self.cursor])))
            self.cursor += 1
        return out

    def draw(self):
        kind = self.kinds[int(self.rng.choice(len(self.kinds), p=self.probs))]
        r = self.rng
        if kind == "presence":
            img, cat = self.positives[int(r.integers(len(self.positives)))]
            neg = self.negatives[cat][int(r.integers(len(self.negatives[cat])))]
            pair = [("presence", img, cat, "yes"), ("presence", neg, cat, "no")]
        elif kind == "counting":
            img, cat, m = self.counting[int(r.integers(len(self.counting)))]
            cand = self.count_partners[(cat, m)]
            p, n = cand[int(r.integers(len(cand)))]
            pair = [("counting", img, cat, m), ("counting", p, cat, n)]
        elif kind == "caption":
            pair = [("caption", int(r.integers(self.n_captions))), ("caption", int(r.integers(self.n_captions)))]
        else:
            j = int(r.integers(self.n_vqa))
            pair = [("vqa", j, 0), ("vqa", j, 1)]
        return kind, pair

    def microbatch(self, size, draws):
        """Items of one microbatch and the per-component draw counts it used."""
        if not self.kinds:
            items = self._caption_batch(size)
            mix = {"caption_examples": size}
        else:
            assert size == 2 * draws
            items, mix = [], collections.Counter()
            for _ in range(draws):
                kind, pair = self.draw()
                items += pair
                mix[kind] += 1
                self.counts["draws"][kind] += 1
            mix = dict(mix)
        for it in items:
            self.counts["examples"][it[0]] += 1
        return items, mix

    # -- state
    def state_dict(self):
        return copy.deepcopy({"rng": self.rng.bit_generator.state, "stage": self.stage,
                              "stage_cfg": getattr(self, "stage_cfg", None), "perm": self.perm,
                              "cursor": getattr(self, "cursor", 0), "pass_no": getattr(self, "pass_no", 0),
                              "counts": getattr(self, "counts", None)})

    def load_state_dict(self, s):
        s = copy.deepcopy(s)
        self.rng.bit_generator.state = s["rng"]
        if s["stage"] is not None:
            self.start_stage(s["stage"], s["stage_cfg"])
            self.perm, self.cursor, self.pass_no, self.counts = s["perm"], s["cursor"], s["pass_no"], s["counts"]

    def summary(self):
        return {"draws": dict(self.counts["draws"]), "examples": dict(self.counts["examples"])}
