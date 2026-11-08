"""Task 3 Step 3: what the signals capture, at N = 32 (manual L687-698; prompt b2 item D).

All analyses use all R rollouts of a question (order-independent); VisConf calibration pools all R rollouts.
    Quartile difference  per score, top k = R/4 vs bottom k rollouts by the score, correct rate top minus bottom;
                         ties at a quartile boundary get fractional membership (remaining slots / number tied).
                         Length: top quartile = shortest (score -T). Averaged over all questions (primary);
                         the mean over mixed questions (not all right, not all wrong) is also reported.
    Spearman(T, score)   within question; undefined (constant input) questions counted and excluded.
    Pearson per rollout  (c_t, v_t), (v_t, t), (c_t, t), t = token position; rollouts with T < 2 or a constant
                         input are skipped (counted); averaged per question over the valid rollouts; summarized
                         across questions (mean, median, share > 0, share < 0).
    Identity             Ztok - Z = population Cov_t(c, v) for every rollout.
"""
import numpy as np
from scipy.stats import spearmanr

from task3_ttc.selection import calibrated_tokens


def quartile_weights(x, k):
    """Membership weights of the k largest values of x; boundary ties share the remaining slots."""
    x = np.asarray(x, dtype=np.float64)
    thr = np.sort(x)[::-1][k - 1]
    gt, eq = x > thr, x == thr
    w = gt.astype(np.float64)
    w[eq] = (k - gt.sum()) / eq.sum()
    return w


def quartile_diff(x, correct, k):
    """Correct rate of the top-k minus the bottom-k rollouts by x."""
    c = np.asarray(correct, dtype=np.float64)
    return float(quartile_weights(x, k) @ c / k - quartile_weights(-np.asarray(x, dtype=np.float64), k) @ c / k)


def rollout_pearson(x_segs, y_segs):
    """Per-rollout Pearson(x_t, y_t); NaN where T < 2 or x or y is constant."""
    out = np.full(len(x_segs), np.nan)
    for i, (x, y) in enumerate(zip(x_segs, y_segs)):
        if len(x) < 2 or x.max() == x.min() or y.max() == y.min():
            continue
        xc, yc = x - x.mean(), y - y.mean()
        out[i] = float((xc @ yc) / np.sqrt((xc @ xc) * (yc @ yc)))
    return out


def _summary(per_q):
    v = np.asarray([x for x in per_q if np.isfinite(x)])
    if len(v) == 0:
        return {"questions": 0}
    return {"questions": int(len(v)), "questions_undefined": int(len(per_q) - len(v)), "mean": float(v.mean()),
            "median": float(np.median(v)), "share_positive": float((v > 0).mean()),
            "share_negative": float((v < 0).mean())}


def question_scores(q):
    """Rollout scores for one question with VisConf calibrated on all of its rollouts."""
    s = {"L": q["L"], "K": q["K"]}
    if "V" in q:
        c = calibrated_tokens(q["kl"])
        C = np.array([x.mean() for x in c])
        s["V"] = q["V"]
        s["Z"] = q["V"] * C
        s["Ztok"] = np.array([(ci * vi).mean() for ci, vi in zip(c, q["v"])])
        s["_c"] = c
        s["_C"] = C
    return s


def analyze(qs, mask, k=None):
    """Section D on the questions in mask (list of per-question dicts as for the engine)."""
    qsel = [q for q, m in zip(qs, mask) if m]
    R = len(qsel[0]["T"])
    k = k or R // 4
    vlm = "V" in qsel[0]
    names = ["L", "K"] + (["V", "Z", "Ztok"] if vlm else [])
    qd = {n: [] for n in names + ["length (top = shortest)"]}
    sp = {n: [] for n in names}
    mixed = 0
    pear = {"c_v": [], "v_pos": [], "c_pos": []}
    skipped = {"c_v": 0, "v_pos": 0, "c_pos": 0}
    short = 0
    for q in qsel:
        s = question_scores(q)
        corr = q["correct"]
        mixed += bool(corr.any() and not corr.all())
        for n in names:
            qd[n].append(quartile_diff(s[n], corr, k))
            if q["T"].min() == q["T"].max() or s[n].min() == s[n].max():
                sp[n].append(np.nan)
            else:
                sp[n].append(float(spearmanr(q["T"], s[n]).statistic))
        qd["length (top = shortest)"].append(quartile_diff(-q["T"].astype(np.float64), corr, k))
        if vlm:
            pos = [np.arange(len(x), dtype=np.float64) for x in q["v"]]
            short += int((q["T"] < 2).sum())
            for key, (a, b) in {"c_v": (s["_c"], q["v"]), "v_pos": (q["v"], pos), "c_pos": (s["_c"], pos)}.items():
                r = rollout_pearson(a, b)
                skipped[key] += int(np.isnan(r).sum())
                pear[key].append(float(np.nanmean(r)) if np.isfinite(r).any() else np.nan)
    out = {"n_questions": len(qsel), "quartile_k": k, "mixed_questions": mixed,
           "quartile_difference": {n: {"mean_all_questions": float(np.mean(v)),
                                       "mean_mixed_questions": float(np.sum(v) / mixed) if mixed else None}
                                   for n, v in qd.items()},
           "spearman_T_vs_score": {n: {"mean": float(np.nanmean(v)) if np.isfinite(v).any() else None,
                                       "questions_defined": int(np.isfinite(v).sum()),
                                       "questions_undefined": int((~np.isfinite(v)).sum())}
                                   for n, v in ((n, np.asarray(x)) for n, x in sp.items())}}
    if vlm:
        out["rollout_pearson"] = {
            "rollouts": int(sum(len(q["T"]) for q in qsel)), "rollouts_with_T_lt_2": short,
            "confidence_vision (c_t, v_t)": {"rollouts_skipped": skipped["c_v"], **_summary(pear["c_v"])},
            "vision_position (v_t, t)": {"rollouts_skipped": skipped["v_pos"], **_summary(pear["v_pos"])},
            "confidence_position (c_t, t)": {"rollouts_skipped": skipped["c_pos"], **_summary(pear["c_pos"])}}
    return out


def covariance_identity(qs):
    """max over every rollout of |(Ztok - Z) - Cov_t(c, v)| with calibration on all rollouts of the question."""
    worst = 0.0
    n = 0
    for q in qs:
        s = question_scores(q)
        for ci, vi, z, zt in zip(s["_c"], q["v"], s["Z"], s["Ztok"]):
            cov = float(((ci - ci.mean()) * (vi - vi.mean())).mean())
            worst = max(worst, abs((zt - z) - cov))
            n += 1
    return {"rollouts": n, "max_abs_difference": float(worst), "tolerance": 1e-6, "passed": bool(worst < 1e-6)}
