"""Task 1 training data: MetaMathQA in source order, prompt/response formatting and microbatches.

Format (manual L111-116): one user turn (the query) and the supplied assistant response.
    prompt_ids   = tokenizer(apply_chat_template([user: query], add_generation_prompt=True,
                                                 enable_thinking=False))   # tokenized on its own
    response_ids = tokenizer(response, add_special_tokens=False) + [<|im_end|>]
    input_ids    = prompt_ids + response_ids
Labels are -100 on the prompt and on padding; response tokens and <|im_end|> are scored.
Microbatches are right-padded with <|endoftext|> (attention mask 0 on padding).
"""
import hashlib
import json

import numpy as np
import torch

MODEL_ID = "Qwen/Qwen3-0.6B"
MODEL_REVISION = "c1899de289a04d12100db370d81485cdf75e47ca"
DATA_ID = "meta-math/MetaMathQA"
DATA_SPLIT = "train"
DATA_REVISION = "aa4f34d3d2d3231299b5b03d9b3e5a20da45aa18"
TRAIN_IDS = "splits/task1/train_ids.json"

PAD_ID = 151643     # <|endoftext|>
IM_END_ID = 151645  # <|im_end|>
IGNORE = -100
MAX_LEN = 2048


def load_tokenizer(revision=MODEL_REVISION):
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(MODEL_ID, revision=revision)


def prompt_text(tok, query):
    return tok.apply_chat_template([{"role": "user", "content": query}], add_generation_prompt=True,
                                   tokenize=False, enable_thinking=False)


def encode_batch(tok, queries, responses):
    """Encode (query, response) pairs; returns dicts with input_ids and n_prompt."""
    p = tok([prompt_text(tok, q) for q in queries], add_special_tokens=False)["input_ids"]
    r = tok(list(responses), add_special_tokens=False)["input_ids"]
    return [{"input_ids": pi + ri + [IM_END_ID], "n_prompt": len(pi)} for pi, ri in zip(p, r)]


def collate(examples, pad_id=PAD_ID):
    """Right-pad a microbatch to its longest example."""
    T = max(len(e["input_ids"]) for e in examples)
    B = len(examples)
    ids = torch.full((B, T), pad_id, dtype=torch.long)
    att = torch.zeros((B, T), dtype=torch.long)
    lab = torch.full((B, T), IGNORE, dtype=torch.long)
    for i, e in enumerate(examples):
        L, P = len(e["input_ids"]), e["n_prompt"]
        ids[i, :L] = torch.tensor(e["input_ids"], dtype=torch.long)
        att[i, :L] = 1
        lab[i, P:L] = ids[i, P:L]
    return {"input_ids": ids, "attention_mask": att, "labels": lab}


def n_scored(batch):
    """Number of positions that receive a loss (labels shifted by one)."""
    return int((batch["labels"][:, 1:] != IGNORE).sum())


def consecutive_pairs(n, microbatch=2):
    """Microbatches of consecutive examples in source order: (0,1), (2,3), ..."""
    assert n % microbatch == 0
    return [tuple(range(i, i + microbatch)) for i in range(0, n, microbatch)]


def token_fingerprint(examples):
    """sha256 over all input_ids and prompt lengths, to confirm identical tokenization across runs."""
    h = hashlib.sha256()
    for e in examples:
        h.update(np.asarray([e["n_prompt"], len(e["input_ids"])], dtype=np.int64).tobytes())
        h.update(np.asarray(e["input_ids"], dtype=np.int64).tobytes())
    return h.hexdigest()


def load_train_examples(tok, ids_path=TRAIN_IDS, chunk=2000):
    """Load and encode the fixed 40,000 training rows in source order (no shuffle)."""
    from datasets import load_dataset
    with open(ids_path) as f:
        meta = json.load(f)
    assert meta["source"]["revision"] == DATA_REVISION, meta["source"]
    idx = meta["indices"]
    assert len(idx) == meta["count"] == 40_000
    ds = load_dataset(DATA_ID, split=DATA_SPLIT, revision=DATA_REVISION).select(idx)
    out = []
    for s in range(0, len(ds), chunk):
        part = ds[s:s + chunk]
        out.extend(encode_batch(tok, part["query"], part["response"]))
    longest = max(len(e["input_ids"]) for e in out)
    assert longest <= MAX_LEN, longest
    return out, meta
