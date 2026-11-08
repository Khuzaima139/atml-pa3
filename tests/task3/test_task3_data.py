"""Task 3 join/alignment checks and per-rollout summaries on tiny synthetic parquet files."""
import numpy as np
import pytest
import pyarrow.parquet as pq

import task3_synth as synth
from task3_ttc.data import DataError, dataset_files, load_config, load_rollouts, load_token_cache, write_token_cache


def _load(tmp_path, ds, mutate=None):
    cfg = load_config(synth.build(str(tmp_path), mutate))
    return cfg, load_rollouts(cfg, ds)


@pytest.mark.parametrize("ds", ["math", "mathvista"])
def test_join_sorted_and_summaries(tmp_path, ds):
    cfg, d = _load(tmp_path, ds)
    n = 6 * synth.N_ROLLOUTS
    assert len(d["sample_id"]) == n and d["stats"]["n_rollouts"] == n
    keys = list(zip(d["sample_id"], d["rollout_id"]))
    assert keys == sorted(keys)  # sample_id as strings, then rollout_id ("test:10" < "test:2")
    assert d["sample_id"][0] == "test:10" if ds == "math" else d["sample_id"][0] == "1"
    # recompute from the raw files, joined independently by key
    f = dataset_files(cfg, ds)
    gen = {(r["sample_id"], r["rollout_id"]): r for r in pq.read_table(f["generations"]).to_pylist()}
    prob = {(r["sample_id"], r["rollout_id"]): r for r in pq.read_table(f["probability"]).to_pylist()}
    attn = ({(r["sample_id"], r["rollout_id"]): r for r in pq.read_table(f["attention"]).to_pylist()}
            if ds == "mathvista" else None)
    for i, k in enumerate(keys):
        T = len(gen[k]["generated_token_ids"])
        assert d["T"][i] == T
        assert d["capped"][i] == gen[k]["hit_max_new_tokens"]
        assert d["extracted"][i] == gen[k]["extracted_answer"]
        assert d["L"][i] == pytest.approx(np.mean(np.array(prob[k]["logp"], dtype=np.float64)), rel=1e-12)
        assert d["K"][i] == pytest.approx(np.mean(np.array(prob[k]["kl_u_p"], dtype=np.float64)), rel=1e-12)
        o0, o1 = d["offsets"][i], d["offsets"][i + 1]
        np.testing.assert_array_equal(d["kl_tok"][o0:o1], np.array(prob[k]["kl_u_p"], dtype=np.float32))
        if attn is not None:
            a = attn[k]
            img = np.array(a["img_attn_total"], dtype=np.float32).astype(np.float64)
            v = img / (img + np.array(a["prompt_text_attn_total"], dtype=np.float32).astype(np.float64)
                       + np.array(a["generated_text_attn_total"], dtype=np.float32).astype(np.float64))
            assert d["V"][i] == pytest.approx(v.mean(), rel=1e-12)
            np.testing.assert_allclose(d["v_tok"][o0:o1], v.astype(np.float32))
    assert d["L"].dtype == np.float64 and d["K"].dtype == np.float64
    assert (d["V"] is None) == (ds == "math")
    assert d["stats"]["capped"] == 1 and d["stats"]["capped_flag_vs_length_disagreements"] == 0


def test_token_cache_roundtrip(tmp_path):
    cfg, d = _load(tmp_path, "mathvista")
    p = str(tmp_path / "c" / "mathvista_tokens.npz")
    meta = write_token_cache(d, p)
    z = load_token_cache(p)
    np.testing.assert_array_equal(z["kl_u_p"], d["kl_tok"])
    np.testing.assert_array_equal(z["v"], d["v_tok"])
    np.testing.assert_array_equal(z["offsets"], d["offsets"])
    assert z["kl_u_p"].dtype == np.float32 and z["v"].dtype == np.float32
    assert z["meta"]["inputs_sha256"] == meta["inputs_sha256"] and len(z["meta"]["input_sha256"]) == 3


def _drop_prob_row(ds, gen, prob, attn):
    prob.pop(3)


def _dup_gen_key(ds, gen, prob, attn):
    gen[1] = dict(gen[1], rollout_id=gen[0]["rollout_id"])


def _shift_prob_key(ds, gen, prob, attn):
    prob[2] = dict(prob[2], sample_id="test:999" if ds == "math" else "999")


def _short_logp(ds, gen, prob, attn):
    prob[5] = dict(prob[5], logp=prob[5]["logp"][:-1])


def _short_attn(ds, gen, prob, attn):
    if attn:
        attn[4] = dict(attn[4], img_attn_total=attn[4]["img_attn_total"] + [0.1])


def _bad_scalar_len(ds, gen, prob, attn):
    gen[0] = dict(gen[0], n_generated_text_tokens=gen[0]["n_generated_text_tokens"] + 1)


def _missing_rollout_everywhere(ds, gen, prob, attn):
    for rows in (gen, prob, attn):
        if rows:
            rows.pop(0)


def _wrong_dataset(ds, gen, prob, attn):
    gen[0] = dict(gen[0], dataset="other")


@pytest.mark.parametrize("mutate,ds", [
    (_drop_prob_row, "math"), (_dup_gen_key, "math"), (_shift_prob_key, "math"), (_short_logp, "math"),
    (_short_attn, "mathvista"), (_bad_scalar_len, "mathvista"), (_missing_rollout_everywhere, "math"),
    (_wrong_dataset, "mathvista"), (_shift_prob_key, "mathvista"),
])
def test_join_mismatch_fails(tmp_path, mutate, ds):
    cfg = load_config(synth.build(str(tmp_path), mutate))
    with pytest.raises(DataError):
        load_rollouts(cfg, ds)
