"""Task 2 scoring shared by development and final evaluation (manual L420-457).

- Reference NLL (teacher forcing, eval mode, FP16 autocast on CUDA): `answer_nll` = mean target-token NLL with
  the end-of-turn token excluded (L425-426); `loss` = mean over target tokens and <|im_end|> (training-loss
  definition, used for the dev caption loss).
- VQA soft accuracy (L444-447) over all ten normalized answers; grounding exact match. Normalization D2.
- Pair metrics and the complete-pair bootstrap (L455-457).
"""
import numpy as np
import torch

from task2_vlm.data import collate
from task2_vlm.model import example_means
from task2_vlm.vqa_norm import normalize


def _sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


@torch.no_grad()
def reference_nll(vlm, enc, examples, store, batch_size, chunk):
    """Per-example {"answer_nll", "loss", "n_content"} for encoded examples, in order (right padding)."""
    out = []
    for s in range(0, len(examples), batch_size):
        exs = examples[s:s + batch_size]
        b = {k: v.to(vlm.device) for k, v in collate(exs, enc.n_vis, enc.pad, "right").items()}
        imgs = [e["image"] for e in exs if e["image"] is not None]
        feats = store.get(imgs).to(vlm.device) if imgs else None
        with vlm.autocast():
            nll, rows, content = vlm.token_nll(b, feats, chunk)
        allm = torch.ones_like(content)
        ans, n = example_means(nll, rows, content, len(exs))
        loss, _ = example_means(nll, rows, allm, len(exs))
        out += [{"answer_nll": float(a), "loss": float(l), "n_content": int(k)}
                for a, l, k in zip(ans.tolist(), loss.tolist(), n.tolist())]
    return out


def soft_accuracy(pred_norm, refs_norm):
    """1/10 sum_i min(1, #{j != i: a_j == pred} / 3) (L446-447); refs already normalized."""
    hits = [r == pred_norm for r in refs_norm]
    tot = sum(hits)
    return sum(min(1.0, (tot - h) / 3.0) for h in hits) / len(refs_norm)


def score_prediction(suite, pair, slot, raw):
    """Per-example metrics for one prediction against the original pair (references fixed, L432)."""
    pred = normalize(raw)
    canon = pair["consensus"][slot] if suite.startswith("vqa") else pair["answers"][slot]
    partner = pair["consensus"][1 - slot] if suite.startswith("vqa") else pair["answers"][1 - slot]
    exact = pred == canon
    if suite.startswith("vqa"):
        acc = soft_accuracy(pred, [normalize(a) for a in pair["answers"][slot]])
    else:
        acc = float(exact)
    return {"prediction": pred, "canonical": canon, "partner_canonical": partner, "accuracy": acc,
            "exact": exact, "equals_partner": pred == partner, "empty_normalized": pred == "",
            "empty_raw": raw == ""}


def pair_rows(rows):
    """Group per-example rows (with pair_id, slot) into complete pairs, in first-seen order."""
    by = {}
    for r in rows:
        by.setdefault(r["pair_id"], [None, None])[r["slot"]] = r
    assert all(a is not None and b is not None for a, b in by.values()), "incomplete pair"
    return list(by.values())


def pair_metrics(rows):
    """Pair accuracy (both exact), identical-answer rate (normalized), over complete pairs."""
    pairs = pair_rows(rows)
    n = len(pairs)
    return {"n_pairs": n,
            "pair_accuracy": sum(a["exact"] and b["exact"] for a, b in pairs) / n if n else None,
            "identical_answer_rate": sum(a["prediction"] == b["prediction"] for a, b in pairs) / n if n else None}


def mean(xs):
    xs = list(xs)
    return float(np.mean(xs)) if xs else None


def bootstrap_indices(n_pairs, resamples, seed):
    return np.random.default_rng(seed).integers(0, n_pairs, size=(resamples, n_pairs))


def bootstrap_ci(values, idx, percentiles):
    """Percentile CI of the mean of per-pair `values` over the resampled pair indices `idx`."""
    v = np.asarray(values, dtype=np.float64)
    stats = v[idx].mean(axis=1)
    lo, hi = np.percentile(stats, percentiles)
    return {"estimate": float(v.mean()), "ci": [float(lo), float(hi)]}


def paired_difference(a, b, idx, percentiles):
    """CI of mean(a - b) over complete pairs; a and b are per-pair values for the same pair order and the
    same resampled pair ids are used for both conditions (L456-457)."""
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    assert a.shape == b.shape
    return bootstrap_ci(a - b, idx, percentiles)
