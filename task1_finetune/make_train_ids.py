"""Fix the Task 1 training set (manual L111-113): first 40,000 eligible MetaMathQA rows in source order.

Eligible = non-empty query and response (after strip), no N2 match of query or original_question to an
openai/gsm8k main test question, and formatted length (task1_finetune.data format) <= 2,048 tokens.
N2 = lowercase, collapse whitespace, strip, delete string.punctuation, collapse whitespace, strip.
Rows are kept in source order; nothing is truncated or shuffled.

Run: python -m task1_finetune.make_train_ids
"""
import argparse
import json
import os
import re
import string

from task1_finetune.data import (DATA_ID, DATA_REVISION, DATA_SPLIT, MAX_LEN, MODEL_ID, MODEL_REVISION,
                                 TRAIN_IDS, encode_batch, load_tokenizer)

TARGET = 40_000
GSM8K = {"path": "openai/gsm8k", "name": "main", "split": "test"}
_PUNCT = str.maketrans("", "", string.punctuation)
RULE = ("first 40,000 eligible rows in source order; eligible = query and response non-empty after strip, "
        "no N2 match of query or original_question to an openai/gsm8k main test question, "
        "formatted length <= 2048 tokens (prompt_ids from apply_chat_template([user], add_generation_prompt=True, "
        "enable_thinking=False) tokenized alone + tokenizer(response, add_special_tokens=False) + [<|im_end|>]); "
        "exclusion reasons checked in order empty > gsm8k_match > too_long")


def n2(s):
    s = re.sub(r"\s+", " ", (s or "").lower()).strip()
    return re.sub(r"\s+", " ", s.translate(_PUNCT)).strip()


def scan(queries, originals, responses, gsm_questions, tok, target=TARGET, block=2000):
    gsm = {n2(x) for x in gsm_questions}
    kept, excluded, lengths = [], [], []
    i = 0
    while len(kept) < target:
        idx = list(range(i, min(i + block, len(queries))))
        assert idx, "dataset exhausted"
        ok = [j for j in idx if queries[j].strip() and responses[j].strip()]
        L = dict(zip(ok, (len(e["input_ids"]) for e in
                          encode_batch(tok, [queries[j] for j in ok], [responses[j] for j in ok]))))
        for j in idx:
            if j not in L:
                excluded.append({"index": j, "reason": "empty"})
            elif n2(queries[j]) in gsm or n2(originals[j]) in gsm:
                excluded.append({"index": j, "reason": "gsm8k_match"})
            elif L[j] > MAX_LEN:
                excluded.append({"index": j, "reason": "too_long", "length": L[j]})
            else:
                kept.append(j)
                lengths.append(L[j])
                if len(kept) == target:
                    break
        i = idx[-1] + 1
    return kept, excluded, lengths


def build():
    from datasets import load_dataset
    tok = load_tokenizer()
    mm = load_dataset(DATA_ID, split=DATA_SPLIT, revision=DATA_REVISION)
    gsm = load_dataset(GSM8K["path"], GSM8K["name"], split=GSM8K["split"])
    kept, excluded, lengths = scan(list(mm["query"]), list(mm["original_question"]), list(mm["response"]),
                                   list(gsm["question"]), tok)
    return {"source": {"dataset": DATA_ID, "split": DATA_SPLIT, "revision": DATA_REVISION,
                       "rows": len(mm)},
            "tokenizer": {"model": MODEL_ID, "revision": MODEL_REVISION},
            "decontamination": {**GSM8K, "rows": len(gsm), "fingerprint": gsm._fingerprint,
                                "normalization": "N2", "fields": ["query", "original_question"]},
            "rule": RULE, "order": "source order, no shuffle, no truncation",
            "count": len(kept), "first_index": kept[0], "last_index": kept[-1],
            "scanned_rows": kept[-1] + 1, "excluded": excluded,
            "formatted_length": {"max": max(lengths), "min": min(lengths), "total": sum(lengths)},
            "indices": kept}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=TRAIN_IDS)
    args = ap.parse_args()
    data = build()
    if os.path.exists(args.out):
        with open(args.out) as f:
            old = json.load(f)
        if old != data:
            raise SystemExit(f"{args.out} exists with different contents; refusing to overwrite")
        print(f"{args.out} unchanged")
        return
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(data, f)
        f.write("\n")
    print(f"wrote {args.out}: count {data['count']} first {data['first_index']} last {data['last_index']} "
          f"excluded {data['excluded']} max length {data['formatted_length']['max']}")


if __name__ == "__main__":
    main()
