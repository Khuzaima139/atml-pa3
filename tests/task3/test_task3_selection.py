"""Task 3 selection engine, Step 3 signals and bootstrap on hand-made pools (prompt b2 items C, D, F, G)."""
import numpy as np
import pytest

from common.bootstrap import paired_bootstrap, resample_indices
from task3_ttc import selection as S
from task3_ttc import signals as G

NA = S.NO_ANSWER


# ------------------------------------------------------------------------------------------- rules

def test_planning_chat_example():
    """answers [12, 15, 12, NO_ANSWER], K = [0.8, 2.1, 0.9, 1.0] -> SC 12, Highest-K 15, RWC-K 15."""
    g = np.array([0, 1, 0, NA])           # 12 -> 0, 15 -> 1
    K = np.array([0.8, 2.1, 0.9, 1.0])
    assert g[S.self_consistency(g)] == 0
    assert g[S.highest(K)] == 1
    assert g[S.rwc(g, K)] == 1
    u, _, W2 = S.rwc_votes(g, K)
    W = dict(zip(u.tolist(), (W2 / 3).tolist()))
    assert W[1] == pytest.approx(1) and W[NA] == pytest.approx(2 / 3) and W[0] == pytest.approx(1 / 3)
    assert np.allclose(S.rwc_weights(K), [0, 1, 1 / 3, 2 / 3])


def test_self_consistency_ties_go_to_earliest_supporting_rollout():
    assert S.self_consistency([1, 2, 2, 1]) == 0          # 1 and 2 tie; group 1 appears first
    assert S.self_consistency([3, 2, 2, 1, 1]) == 1        # 2 and 1 tie with 2 votes; group 2 first at position 1
    assert S.self_consistency([NA, NA, 4]) == 0            # NO_ANSWER is an ordinary voting group
    assert S.self_consistency([5]) == 0


def test_highest_and_shortest_ties_earliest():
    assert S.highest([1.0, 3.0, 3.0]) == 1
    assert S.shortest([5, 3, 3, 9]) == 1


def test_rwc_average_rank_ties():
    s = np.array([1.0, 1.0, 0.5])
    assert np.allclose(S.rwc_weights(s), [0.75, 0.75, 0.0])   # ranks 1.5, 1.5, 3
    assert S.rwc([0, 1, 1], s) == 0                           # W = 0.75 each: earliest supporting rollout
    assert S.rwc([0, 1, 2], np.ones(3)) == 0                  # all tied
    assert S.rwc([4, 7, 7], [0.1, 0.2, 0.3]) == 1             # W(7) = 1 + 0.5 > W(4) = 0
    assert S.rwc([9], [0.3]) == 0                             # N = 1


def test_visconf_hand_computed():
    kl = [np.array([0.0, 10.0]), np.array([4.0, 6.0])]
    v = [np.array([0.2, 0.4]), np.array([0.1, 0.3])]
    C, Z, Zt, (q05, q95) = S.visconf(kl, v, np.array([0.3, 0.2]))
    # pooled sorted [0, 4, 6, 10]: Q05 = 0 + 0.15 * 4 = 0.6, Q95 = 6 + 0.85 * 4 = 9.4
    assert q05 == pytest.approx(0.6) and q95 == pytest.approx(9.4)
    cA = np.clip((kl[0] - 0.6) / 8.8, 0, 1)
    cB = np.clip((kl[1] - 0.6) / 8.8, 0, 1)
    assert np.allclose(C, [cA.mean(), cB.mean()])
    assert np.allclose(Z, [0.3 * cA.mean(), 0.2 * cB.mean()])
    assert np.allclose(Zt, [(cA * v[0]).mean(), (cB * v[1]).mean()])


def test_visconf_constant_pool_uses_eps():
    C, _, _, _ = S.visconf([np.ones(3), np.ones(2)], [np.ones(3), np.ones(2)], np.ones(2))
    assert np.allclose(C, 0.0)


def _vlm_question(kl, v, g, correct, T=None):
    T = np.array([len(x) for x in kl]) if T is None else T
    return {"T": T, "L": np.zeros(len(kl)), "K": np.array([x.mean() for x in kl]), "g": np.array(g),
            "correct": np.array(correct), "V": np.array([x.mean() for x in v]), "kl": kl, "v": v}


def test_visconf_calibrates_on_the_prefix_only():
    """With A, B revealed, A wins on Z; including the unrevealed C (extreme kl) would make B win."""
    A, B, C = np.array([1.0, 1, 1, 1]), np.array([0.0, 0, 0, 5]), np.full(4, 100.0)
    kl, v = [A, B, C], [np.full(4, 0.5)] * 3
    q = _vlm_question(kl, v, [0, 1, 2], [True, False, False])
    _, Z2, _, _ = S.visconf(kl[:2], v[:2], q["V"][:2])
    _, Z3, _, _ = S.visconf(kl, v, q["V"])
    assert Z2[0] > Z2[1] and Z3[1] > Z3[0]          # the test discriminates
    m = ["FirstAns", "Highest-Z", "RWC-Z"]
    r = S.select_question(q, np.array([[1, 0, 2]]), [1, 2, 3], m)
    assert r["sel"][0, 1, 1] == 0 and r["sel"][0, 1, 2] == 0     # N = 2: A, from prefix calibration
    assert r["sel"][0, 0].tolist() == [1, 1, 1]                   # N = 1: the only candidate


# ------------------------------------------------------------------------------------------- engine

def _random_question(rng, R=8, vlm=True, n_groups=3):
    T = rng.integers(1, 6, R)
    kl = [rng.random(t) * 10 for t in T]
    v = [rng.random(t) * 0.2 for t in T]
    g = rng.integers(-1, n_groups, R)
    good = rng.integers(0, n_groups)
    q = {"T": T, "L": -rng.random(R), "K": np.array([x.mean() for x in kl]), "g": g, "correct": g == good}
    if vlm:
        q.update({"V": np.array([x.mean() for x in v]), "kl": kl, "v": v})
    return q


def test_select_question_costs_and_records():
    q = {"T": np.array([10, 20, 30, 40]), "L": np.array([-1.0, -0.5, -2.0, -0.1]), "K": np.array([1.0, 2, 3, 4]),
         "g": np.array([0, 1, 0, NA]), "correct": np.array([True, False, True, False])}
    m = S.methods_for("math")
    r = S.select_question(q, np.array([[2, 1, 0, 3]]), [1, 2, 4], m)
    fa, sc, hl, hk, rk, sh = range(6)
    assert r["cost"][0, :, fa].tolist() == [30, 30, 30]
    assert r["cost"][0, :, sc].tolist() == [30, 50, 100]
    assert r["sel"][0, 2, sc] == 2 and r["group"][0, 2, sc] == 0 and r["correct"][0, 2, sc]
    assert r["sel"][0, 2, hl] == 3 and r["group"][0, 2, hl] == NA and not r["correct"][0, 2, hl]
    assert r["sel"][0, 2, hk] == 3 and r["sel"][0, 2, sh] == 0
    # RWC-K at N = 4: ranks K desc: 3 -> 1, 2 -> 2, 1 -> 3, 0 -> 4; W(NA) = 1, W(0) = 2/3 + 0 , W(1) = 1/3
    assert r["group"][0, 2, rk] == NA
    assert r["oracle"][0].tolist() == [True, True, True]
    assert r["distinct_incl"][0].tolist() == [1, 2, 3] and r["distinct_excl"][0].tolist() == [1, 2, 2]


def test_construction_checks_pass_on_random_pools():
    rng = np.random.default_rng(0)
    R, budgets = 8, [1, 2, 4, 8]
    qs = [_random_question(rng, R) for _ in range(20)]
    orders = np.stack([np.stack([rng.permutation(R) for _ in range(5)]) for _ in qs])
    methods = S.methods_for("mathvista")
    res = S.run_engine(qs, orders, budgets, methods)
    ch = S.construction_checks(res, qs, methods, budgets)
    assert all(c["passed"] for c in ch.values()), ch
    assert set(ch) == {"all_methods_equal_at_N1", "SC_N2_selects_FirstAns_group",
                       "RWC-K_equals_Highest-K_at_N2_when_K_differ", "FirstAns_identical_across_N",
                       "oracle_and_distinct_nondecreasing_in_N", "accuracy_le_oracle", "order_invariance_at_N8"}
    # a tampered result is caught
    bad = dict(res, correct=res["correct"].copy())
    bad["correct"][~res["oracle"]] = True
    assert not S.construction_checks(bad, qs, methods, budgets)["accuracy_le_oracle"]["passed"]
    pq = S.per_question(res)
    tot = pq["accuracy"] + pq["sampling_failure"] + pq["selection_failure"]
    assert np.allclose(tot, 1.0)
    mn = S.means(pq, np.ones(20, bool), methods, budgets)
    assert set(mn["methods"]) == set(methods) and set(mn["per_budget"]) == {"1", "2", "4", "8"}


def test_order_invariance_tie_counting():
    q = {"T": np.array([3, 3, 5]), "L": np.zeros(3), "K": np.array([1.0, 1.0, 0.5]), "g": np.array([0, 1, 1]),
         "correct": np.array([True, False, False])}
    assert S._tie_info(q, "Highest-K") == (True, True)
    assert S._tie_info(q, "Shortest") == (True, True)
    assert S._tie_info(q, "SC") == (False, False)
    q2 = dict(q, g=np.array([0, 0, 1]))
    assert S._tie_info(q2, "Highest-K") == (True, False)


# ------------------------------------------------------------------------------------------- signals

def test_quartile_fractional_ties():
    x = np.array([5, 4, 4, 4, 1, 1, 0, 0.0])
    w = G.quartile_weights(x, 2)
    assert np.allclose(w, [1, 1 / 3, 1 / 3, 1 / 3, 0, 0, 0, 0])
    c = np.array([0, 1, 1, 0, 0, 0, 1, 0])
    # top: (0 + 1/3 + 1/3 + 0) / 2 = 1/3; bottom (the two zeros): (1 + 0) / 2 = 1/2
    assert G.quartile_diff(x, c, 2) == pytest.approx(1 / 3 - 1 / 2)
    assert G.quartile_diff(np.ones(8), c, 2) == pytest.approx(0.0)


def test_rollout_pearson_skips():
    r = G.rollout_pearson([np.array([1.0]), np.array([1.0, 1.0]), np.array([1.0, 2, 3])],
                          [np.array([2.0]), np.array([1.0, 2.0]), np.array([2.0, 4, 6.5])])
    assert np.isnan(r[0]) and np.isnan(r[1]) and 0.99 < r[2] <= 1


def test_covariance_identity_and_analyze():
    rng = np.random.default_rng(1)
    qs = [_random_question(rng, 8) for _ in range(6)]
    ci = G.covariance_identity(qs)
    assert ci["passed"] and ci["rollouts"] == 48
    out = G.analyze(qs, np.array([True] * 4 + [False] * 2))
    assert out["n_questions"] == 4 and out["quartile_k"] == 2
    assert set(out["quartile_difference"]) == {"L", "K", "V", "Z", "Ztok", "length (top = shortest)"}
    assert set(out["rollout_pearson"]) >= {"confidence_vision (c_t, v_t)", "vision_position (v_t, t)",
                                           "confidence_position (c_t, t)"}
    math = G.analyze([{k: q[k] for k in ("T", "L", "K", "g", "correct")} for q in qs], np.ones(6, bool))
    assert set(math["quartile_difference"]) == {"L", "K", "length (top = shortest)"} and "rollout_pearson" not in math


# ------------------------------------------------------------------------------------------- bootstrap

def test_paired_bootstrap():
    a = np.array([1.0, 0.0, 1.0, 1.0])
    r = paired_bootstrap(a, a - 0.1)
    assert r["mean"] == pytest.approx(0.1) and r["ci_low"] == pytest.approx(0.1) and r["ci_high"] == pytest.approx(0.1)
    rng = np.random.default_rng(3)
    x, y = rng.random(50), rng.random(50)
    r1, r2 = paired_bootstrap(x, y), paired_bootstrap(x, y)
    assert r1 == r2 and r1["ci_low"] <= r1["mean"] <= r1["ci_high"] and r1["resamples"] == 1000
    idx = resample_indices(50)
    assert np.isclose(np.percentile((x - y)[idx].mean(1), 2.5), r1["ci_low"])
