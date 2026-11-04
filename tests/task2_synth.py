"""Small synthetic VQAv2 / COCO inputs shaped like the real files, for Task 2 selection tests."""
import copy
import os

import numpy as np

from task2_vlm.select_data import load_categories, load_config

CATS = (1, 17, 18)  # person, cat, dog
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WORDS = "a man dog cat sits on the red bench near water with two small birds".split()


class FakeTokenizer:
    """One token per whitespace-separated word."""

    def __call__(self, texts, add_special_tokens=False):
        return {"input_ids": [t.split() for t in texts]}


def config():
    cfg = copy.deepcopy(load_config(os.path.join(REPO, "configs/task2/data.yaml")))
    cfg["vqa"]["pairs"] = {"dev": {"yes/no": 2, "number": 1, "other": 2},
                           "test": {"yes/no": 3, "number": 2, "other": 3},
                           "train": {"yes/no": 5, "number": 3, "other": 5}}
    cfg["grounding"]["pairs"] = {"dev": {"presence": 2, "counting": 2}, "test": {"presence": 3, "counting": 3}}
    cfg["captions"].update(max_tokens=8, dev_images=5, train_images=20, train_per_image=3)
    cfg["manual_table"] = {
        "captions": {"train_images": 20, "train_examples": 60, "dev_images": 5},
        "grounding": {"dev_pairs": 4, "dev_questions": 8, "test_pairs": 6, "test_questions": 12},
        "vqa": {"train_pairs": 13, "train_questions": 26, "dev_pairs": 5, "dev_questions": 10,
                "test_pairs": 8, "test_questions": 16}}
    return cfg


def categories():
    return load_categories(os.path.join(REPO, "configs/task2/categories.yaml"))


def answers(rng, kind, value, agree):
    pool = {"yes/no": ["yes", "no"], "number": ["1", "2", "3", "4"], "other": ["red", "blue", "green"]}[kind]
    others = [a for a in pool if a != value] + ["maybe"]
    raw = [value] * agree + [others[int(rng.integers(len(others)))] for _ in range(10 - agree)]
    rng.shuffle(raw)
    return [{"answer": a, "answer_confidence": "yes", "answer_id": i + 1} for i, a in enumerate(raw)]


def raw_split(split, first_id, n_images, seed):
    rng = np.random.default_rng(seed)
    ids = list(range(first_id, first_id + n_images))
    images = [{"id": i, "file_name": f"COCO_{split}2014_{i:012d}.jpg"} for i in ids]
    inst, aid = [], 1
    for i in ids:
        if i % 17 == 0:
            continue  # unannotated image
        for c in CATS:
            n = int(rng.integers(0, 6))
            for _ in range(n):
                inst.append({"id": aid, "image_id": i, "category_id": c, "iscrowd": 0,
                             "area": float(rng.choice([500.0, 2000.0, 5000.0], p=[0.1, 0.45, 0.45]))})
                aid += 1
            if n == 5 and rng.random() < 0.5:
                inst.append({"id": aid, "image_id": i, "category_id": c, "iscrowd": 1, "area": 9000.0})
                aid += 1
    caps, cid = [], 1
    for i in ids:
        for _ in range(5):
            k = int(rng.integers(3, 12))
            caps.append({"id": cid, "image_id": i, "caption": "  " + " ".join(rng.choice(WORDS, k)) + " \n"})
            cid += 1
    questions, anns, pairs = [], [], []
    qid = first_id * 1000
    kinds = ["yes/no", "number", "other"]
    values = {"yes/no": ["yes", "no"], "number": ["1", "2", "3", "4"], "other": ["red", "blue", "green"]}
    for j in range(0, n_images - 1, 2):
        for t, kind in enumerate(kinds):
            text = f"question {kind} {j}?"
            va, vb = rng.choice(values[kind], 2, replace=False)
            for img, v in ((ids[j], va), (ids[j + 1], vb)):
                agree = int(rng.choice([7, 8, 9, 10]))
                questions.append({"image_id": img, "question": text, "question_id": qid})
                anns.append({"question_id": qid, "image_id": img, "answer_type": kind, "question_type": kind,
                             "multiple_choice_answer": v, "answers": answers(rng, kind, v, agree)})
                qid += 1
            pairs.append([qid - 2, qid - 1])
    pairs.append(list(reversed(pairs[0])))  # unordered duplicate
    return {"questions": {"questions": questions}, "annotations": {"annotations": anns}, "pairs": pairs,
            "captions": {"images": images, "annotations": caps},
            "instances": {"images": images, "annotations": inst}}


def raw_val():
    return raw_split("val", 1, 400, 1)


def raw_train():
    return raw_split("train", 5001, 300, 2)
