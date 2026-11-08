"""Task 3 offline selection engine (manual L607-615, L630-686; prompt b2 item C).

For each question, order o (n_orders saved permutations) and budget N, the candidates are orders[q][o][:N]
(positions 0..N-1 in the current order). Groups and correctness come from the rollout CSV; NO_ANSWER is group
-1, an ordinary voting group that is always wrong.

Methods (every rule returns the only candidate at N = 1):
    FirstAns       candidate at position 0; cost = its T.
    SC             one vote per rollout; largest group; ties -> the group whose earliest supporting rollout comes
                   first in the current order.
    Highest-X      argmax of score X, ties -> earliest in order. X in L (mean logp), K (self-certainty) and, for
                   MathVista, V (vision), Z (VisConf), Ztok (tokenwise fusion).
    Shortest       argmin T, ties -> earliest in order (length-only control).
    RWC-X          r = scipy rankdata(-X, method="average") (rank 1 best), w = (N - r) / (N - 1),
                   W(a) = sum of w over group a; argmax W, ties -> earliest supporting rollout. W is accumulated as
                   sum(N - r), which is exact (half-integers), so ties are exact.
    VisConf        at every prefix: pool the token kl_u_p of the N candidates only; Q05, Q95 with
                   np.quantile(method="linear"); c = clip((k - Q05) / max(Q95 - Q05, 1e-8), 0, 1);
                   C_i = mean c, Z_i = V_i * C_i, Ztok_i = mean(c * v). Unrevealed rollouts are never used.
    Cost           sum of T over the N candidates for every method except FirstAns.
For the aggregation rules (SC, RWC-X) the recorded selected rollout is the earliest supporting rollout of the
winning group in the current order.
"""
import numpy as np
from scipy.stats import rankdata

NO_ANSWER = -1
EPS = 1e-8
Q_LO, Q_HI = 0.05, 0.95

BASE_METHODS = ["FirstAns", "SC", "Highest-L", "Highest-K", "RWC-K", "Shortest"]
VLM_METHODS = ["Highest-V", "RWC-V", "Highest-Z", "RWC-Z", "Highest-Ztok", "RWC-Ztok"]


def methods_for(dataset):
    return BASE_METHODS + (VLM_METHODS if dataset == "mathvista" else [])


# ------------------------------------------------------------------------------------------- rules
# Each rule takes arrays in the current candidate order and returns the selected position (0..N-1).

def first_ans(n):
    return 0


def self_consistency(groups):
    """Largest group; ties -> group whose earliest supporting rollout comes first. Returns that rollout's position."""
    groups = np.asarray(groups)
    _, first, counts = np.unique(groups, return_index=True, return_counts=True)
    return int(first[counts == counts.max()].min())


def highest(scores):
    """argmax, ties -> earliest position (np.argmax returns the first maximum)."""
    return int(np.argmax(np.asarray(scores)))


def shortest(lengths):
    return int(np.argmin(np.asarray(lengths)))


def rwc_weights(scores):
    """w_i = (N - r_i) / (N - 1) with descending average ranks r (rank 1 best). N must be > 1."""
    s = np.asarray(scores, dtype=np.float64)
    n = len(s)
    return (n - rankdata(-s, method="average")) / (n - 1)


def rwc_votes(groups, scores):
    """(distinct groups, first position of each, W(a) * (N - 1)) with exact half-integer sums."""
    groups = np.asarray(groups)
    s = np.asarray(scores, dtype=np.float64)
    n = len(s)
    u, first, inv = np.unique(groups, return_index=True, return_inverse=True)
    w2 = n - rankdata(-s, method="average")
    return u, first, np.bincount(inv, weights=w2, minlength=len(u))


def rwc(groups, scores):
    """Rank-weighted consensus; N = 1 returns the only candidate. Returns the winning group's earliest position."""
    if len(groups) == 1:
        return 0
    _, first, W = rwc_votes(groups, scores)
    return int(first[W == W.max()].min())


def visconf(kl_segs, v_segs, V):
    """VisConf on one candidate set: kl_segs / v_segs are per-candidate token arrays (float64), V the rollout
    vision scores. Returns (C, Z, Ztok, (Q05, Q95))."""
    lens = np.array([len(x) for x in kl_segs])
    starts = np.concatenate([[0], np.cumsum(lens)[:-1]])
    pool = np.concatenate(kl_segs)
    q05, q95 = np.quantile(pool, [Q_LO, Q_HI], method="linear")
    c = np.clip((pool - q05) / max(q95 - q05, EPS), 0.0, 1.0)
    C = np.add.reduceat(c, starts) / lens
    Ztok = np.add.reduceat(c * np.concatenate(v_segs), starts) / lens
    return C, np.asarray(V, dtype=np.float64) * C, Ztok, (float(q05), float(q95))


def calibrated_tokens(kl_segs):
    """Per-candidate calibrated confidence c_t (list of arrays), calibration pooled over all given candidates."""
    pool = np.concatenate(kl_segs)
    q05, q95 = np.quantile(pool, [Q_LO, Q_HI], method="linear")
    d = max(q95 - q05, EPS)
    return [np.clip((x - q05) / d, 0.0, 1.0) for x in kl_segs]


# ------------------------------------------------------------------------------------------- engine

def select_question(q, orders, budgets, methods):
    """All selections for one question.

    q: dict of length-R arrays T, L, K, g (group id), correct and, for VLM methods, V plus token lists kl, v
       (list of R float64 arrays). orders: (n_orders, R) rollout ids. Returns dict of arrays:
       sel/group/correct/cost (n_orders, n_budgets, n_methods); oracle, distinct_incl, distinct_excl
       (n_orders, n_budgets).
    """
    n_o, n_b, n_m = len(orders), len(budgets), len(methods)
    sel = np.zeros((n_o, n_b, n_m), dtype=np.int16)
    cost = np.zeros((n_o, n_b, n_m), dtype=np.int64)
    oracle = np.zeros((n_o, n_b), dtype=bool)
    d_incl = np.zeros((n_o, n_b), dtype=np.int16)
    d_excl = np.zeros((n_o, n_b), dtype=np.int16)
    T, g, corr = q["T"], q["g"], q["correct"]
    need_vc = any(m.endswith(("-Z", "-Ztok")) for m in methods)
    for o in range(n_o):
        for b, n in enumerate(budgets):
            idx = np.asarray(orders[o][:n])
            gi, Ti = g[idx], T[idx]
            oracle[o, b] = corr[idx].any()
            d_incl[o, b] = len(np.unique(gi))
            d_excl[o, b] = len(np.unique(gi[gi != NO_ANSWER]))
            total = int(Ti.sum())
            sc = {"L": q["L"][idx], "K": q["K"][idx]}
            if "V" in q:
                sc["V"] = q["V"][idx]
            if need_vc and n > 1:
                _, sc["Z"], sc["Ztok"], _ = visconf([q["kl"][i] for i in idx], [q["v"][i] for i in idx], sc["V"])
            for m, name in enumerate(methods):
                if n == 1 or name == "FirstAns":
                    p = 0
                elif name == "SC":
                    p = self_consistency(gi)
                elif name == "Shortest":
                    p = shortest(Ti)
                elif name.startswith("Highest-"):
                    p = highest(sc[name[8:]])
                elif name.startswith("RWC-"):
                    p = rwc(gi, sc[name[4:]])
                else:
                    raise ValueError(name)
                sel[o, b, m] = idx[p]
                cost[o, b, m] = int(Ti[0]) if name == "FirstAns" else total
    group = g[sel]
    return {"sel": sel, "group": group.astype(np.int16), "correct": corr[sel], "cost": cost,
            "oracle": oracle, "distinct_incl": d_incl, "distinct_excl": d_excl}


def run_engine(qs, orders, budgets, methods):
    """qs: list of per-question dicts; orders: (n_q, n_orders, R). Stacks select_question over questions."""
    outs = [select_question(q, orders[i], budgets, methods) for i, q in enumerate(qs)]
    res = {k: np.stack([o[k] for o in outs]) for k in outs[0]}
    res["orders"] = np.asarray(orders)
    return res


# --------------------------------------------------------------------------------------- aggregation

def per_question(res):
    """Order-averaged per-question metrics. Returns dict of arrays (n_q, n_b, n_m) or (n_q, n_b)."""
    corr = res["correct"].astype(float)
    orc = res["oracle"][..., None]
    na = (res["group"] == NO_ANSWER).astype(float)
    return {
        "accuracy": corr.mean(1),
        "cost": res["cost"].mean(1),
        "no_answer_selected": na.mean(1),
        "sampling_failure": ((~res["correct"]) & ~orc).mean(1),
        "selection_failure": ((~res["correct"]) & orc).mean(1),
        "oracle": res["oracle"].mean(1),
        "distinct_incl": res["distinct_incl"].mean(1),
        "distinct_excl": res["distinct_excl"].mean(1),
    }


def means(pq, mask, methods, budgets):
    """Means over the questions in mask: {method: {N: {...}}} plus {"oracle"/"distinct_*": {N: value}}."""
    out = {"n_questions": int(mask.sum()), "methods": {}, "per_budget": {}}
    for m, name in enumerate(methods):
        out["methods"][name] = {
            str(n): {"accuracy": float(pq["accuracy"][mask, b, m].mean()),
                     "cost": float(pq["cost"][mask, b, m].mean()),
                     "no_answer_selected": float(pq["no_answer_selected"][mask, b, m].mean()),
                     "failure": {"correct": float(pq["accuracy"][mask, b, m].mean()),
                                 "sampling_failure": float(pq["sampling_failure"][mask, b, m].mean()),
                                 "selection_failure": float(pq["selection_failure"][mask, b, m].mean())}}
            for b, n in enumerate(budgets)}
    for b, n in enumerate(budgets):
        out["per_budget"][str(n)] = {"oracle": float(pq["oracle"][mask, b].mean()),
                                     "distinct_groups_incl_no_answer": float(pq["distinct_incl"][mask, b].mean()),
                                     "distinct_groups_excl_no_answer": float(pq["distinct_excl"][mask, b].mean())}
    return out


# ------------------------------------------------------------------------------------ construction checks

def full_scores(q):
    """Rollout scores with all R rollouts revealed (VisConf calibrated on all of them)."""
    s = {"T": q["T"], "L": q["L"], "K": q["K"]}
    if "V" in q:
        s["V"] = q["V"]
        if "kl" in q:
            _, s["Z"], s["Ztok"], _ = visconf(q["kl"], q["v"], q["V"])
    return s


def _tie_info(q, name, scores=None):
    """At N = R (all rollouts): (tie exists, tie spans more than one group) for an order-independent rule."""
    g = q["g"]
    q = scores if scores is not None else full_scores(q)
    if name == "SC":
        u, c = np.unique(g, return_counts=True)
        t = int((c == c.max()).sum()) > 1
        return t, t
    if name == "Shortest" or name.startswith("Highest-"):
        x = q["T"] if name == "Shortest" else q[name[8:]]
        best = x.min() if name == "Shortest" else x.max()
        hit = x == best
        return int(hit.sum()) > 1, len(np.unique(g[hit])) > 1
    if name.startswith("RWC-"):
        _, _, W = rwc_votes(g, q[name[4:]])
        t = int((W == W.max()).sum()) > 1
        return t, t
    raise ValueError(name)


def construction_checks(res, qs, methods, budgets):
    """Checks on engine outputs (res from run_engine); each entry has "passed" and counts. The order-invariance
    check at the largest N covers SC, Highest-*, and also Shortest and RWC-*: a question may vary across orders
    only if the rule has a tie there."""
    out = {}
    sel, grp, cor, cost = res["sel"], res["group"], res["correct"], res["cost"]
    b1 = budgets.index(1)
    out["all_methods_equal_at_N1"] = {"passed": bool((sel[:, :, b1, :] == sel[:, :, b1, :1]).all())}
    if 2 in budgets:
        b2 = budgets.index(2)
        isc, ifa = methods.index("SC"), methods.index("FirstAns")
        out["SC_N2_selects_FirstAns_group"] = {"passed": bool((grp[:, :, b2, isc] == grp[:, :, b2, ifa]).all())}
        ir, ih = methods.index("RWC-K"), methods.index("Highest-K")
        k = np.stack([q["K"] for q in qs])
        first2 = res["orders"][:, :, :2]
        kk = np.take_along_axis(k[:, None, :], first2, axis=2)
        differ = kk[..., 0] != kk[..., 1]
        same = grp[:, :, b2, ir] == grp[:, :, b2, ih]
        out["RWC-K_equals_Highest-K_at_N2_when_K_differ"] = {
            "passed": bool(same[differ].all()), "cases": int(differ.sum()), "K_ties": int((~differ).sum())}
    ifa = methods.index("FirstAns")
    out["FirstAns_identical_across_N"] = {
        "passed": bool((cor[..., ifa] == cor[..., :1, ifa]).all() and (cost[..., ifa] == cost[..., :1, ifa]).all())}
    out["oracle_and_distinct_nondecreasing_in_N"] = {
        "passed": bool((np.diff(res["oracle"].astype(int), axis=2) >= 0).all()
                       and (np.diff(res["distinct_incl"], axis=2) >= 0).all()
                       and (np.diff(res["distinct_excl"], axis=2) >= 0).all())}
    out["accuracy_le_oracle"] = {"passed": bool((cor <= res["oracle"][..., None]).all())}
    bR = len(budgets) - 1
    inv = {}
    full = [full_scores(q) for q in qs]
    for m, name in enumerate(methods):
        if name == "FirstAns":
            continue
        g = grp[:, :, bR, m]
        varies = (g != g[:, :1]).any(1)
        ties = np.array([_tie_info(q, name, f) for q, f in zip(qs, full)])
        bad = varies & ~ties[:, 0]
        inv[name] = {"passed": bool(not bad.any()), "questions_with_tie": int(ties[:, 0].sum()),
                     "questions_with_tie_across_groups": int(ties[:, 1].sum()),
                     "questions_varying_across_orders": int(varies.sum()),
                     "varying_without_tie": int(bad.sum())}
    out[f"order_invariance_at_N{budgets[-1]}"] = {"passed": all(v["passed"] for v in inv.values()), "methods": inv}
    return out
