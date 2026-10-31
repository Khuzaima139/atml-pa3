"""Task 1 evaluation pipeline tests: tiny random Qwen3 on CPU, real tokenizer, real lm-eval task definitions."""
import copy
import glob
import gzip
import json
import os
import re

import pytest
import torch
import yaml
from safetensors.torch import save_file

from task1_finetune import build as B
from task1_finetune import data as D
from task1_finetune import diagnostics as G
from task1_finetune import evaluate as E
from task1_finetune import qualitative as Q
from task1_finetune import status
from task1_finetune.adapters import TARGETS, AdapterLinear, adapter_state, export_effective, inject_adapters, \
    leading_bases, target_linears
from task1_finetune.tracking_lm import TrackingHFLM, end_of_generation

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CPU = torch.device("cpu")
LORA = {"r": 4, "alpha": 8, "dropout": 0.05, "project": False, "k": None}
OPLORA = {"r": 4, "alpha": 8, "dropout": 0.05, "project": True, "k": 8}


def tiny_model(seed=0, gen_config=True):
    """Parameters in bf16 and the RoPE buffers in FP32, like from_pretrained(dtype=bf16) on the real checkpoint."""
    from transformers import Qwen3Config, Qwen3ForCausalLM
    torch.manual_seed(seed)
    cfg = Qwen3Config(vocab_size=151936, hidden_size=64, intermediate_size=96, num_hidden_layers=2,
                      num_attention_heads=4, num_key_value_heads=2, head_dim=16, tie_word_embeddings=True,
                      max_position_embeddings=2048)
    m = Qwen3ForCausalLM(cfg)
    for p in m.parameters():
        p.data = p.data.to(torch.bfloat16)
    if gen_config:  # the checkpoint's generation_config.json
        g = m.generation_config
        g.do_sample, g.temperature, g.top_k, g.top_p = True, 0.6, 20, 0.95
        g.eos_token_id, g.pad_token_id, g.bos_token_id = [151645, 151643], 151643, 151643
    return m.eval()


@pytest.fixture(scope="module")
def tok():
    return D.load_tokenizer()


def random_factors(model, r, seed=1):
    g = torch.Generator().manual_seed(seed)
    out = {}
    for n, lin in target_linears(model):
        out[f"{n}.lora_A"] = torch.randn(r, lin.in_features, generator=g) * 0.1
        out[f"{n}.lora_B"] = torch.randn(lin.out_features, r, generator=g) * 0.1
    return out


def make_lm(model, tok, batch=2, **kw):
    return TrackingHFLM(pretrained=model, tokenizer=tok, batch_size=batch, enable_thinking=False, **kw)


def instances(contexts, until=("Q:",), max_gen_toks=8):
    from lm_eval.api.instance import Instance
    gk = {"until": list(until), "do_sample": False, "max_gen_toks": max_gen_toks}
    return [Instance("generate_until", {}, (c, dict(gk)), idx=0, metadata=("gsm8k_cot", i, 1))
            for i, c in enumerate(contexts)]


# ---------------------------------------------------------------- merge

def unmerged_and_merged(tmp_path, ad, dtype=torch.float32):
    """Trained-like adapters saved as in training, then the real build path for both models."""
    model = tiny_model()
    bases = None
    if ad["project"]:
        bases = {n: leading_bases(l.weight.detach().float(), ad["k"]) for n, l in target_linears(model)}
    for p in model.parameters():
        p.data = p.data.float()
    mods = inject_adapters(model, ad["r"], ad["alpha"], ad["dropout"], 6304, ad["project"], bases)
    fac = random_factors(tiny_model(), ad["r"])
    with torch.no_grad():
        for n, m in mods.items():
            m.lora_A.copy_(fac[f"{n}.lora_A"])
            m.lora_B.copy_(fac[f"{n}.lora_B"])
    save_file(adapter_state(mods), tmp_path / "a.safetensors")
    index = {"a.safetensors": str(tmp_path / "a.safetensors")}
    cond = {"name": "x", "kind": "adapter", "adapter": ad, "seed": 6304, "file": "a.safetensors"}
    if ad["project"]:
        save_file(export_effective(mods), tmp_path / "e.safetensors")
        index["e.safetensors"] = str(tmp_path / "e.safetensors")
        cond["eff_file"] = "e.safetensors"
    return cond, index, mods


@pytest.fixture
def patched_base(monkeypatch):
    calls = []

    def from_pretrained(name, revision, dtype):
        calls.append((name, revision, dtype))
        return tiny_model()

    import transformers
    monkeypatch.setattr(transformers.AutoModelForCausalLM, "from_pretrained", from_pretrained)
    return calls


CFG = {"model": {"name": "Qwen/Qwen3-0.6B", "revision": D.MODEL_REVISION}}


@pytest.mark.parametrize("ad", [LORA, OPLORA], ids=["lora", "oplora"])
def test_merged_equals_unmerged_in_fp32(tmp_path, tok, ad, patched_base):
    cond, index, _ = unmerged_and_merged(tmp_path, ad)
    merged, info = B.build_model(cond, CFG, index, CPU, torch.float32)
    un = B.build_unmerged(cond, CFG, index, CPU, torch.float32)
    assert any(isinstance(m, AdapterLinear) for m in un.modules()) and not un.training
    assert info["targets_merged"] == 14 and info["scaling"] == 2.0 and info["merge_s"] > 0
    assert not any(isinstance(m, AdapterLinear) for m in merged.modules()) and not merged.training
    assert all(p.dtype == torch.float32 for p in merged.parameters())
    items = [{"domain": "x", "id": 0, "input_ids": tok("The quick brown fox")["input_ids"]},
             {"domain": "x", "id": 1, "input_ids": tok("Two plus two is four, so")["input_ids"]}]
    r = E.compare_models(un, merged, items, CPU, unmerged_autocast=False)
    assert r["max_abs_diff"] < 2e-4 and r["argmax_agreement"] == 1.0 and r["non_finite"] == 0
    base, _ = B.build_model({"name": "b", "kind": "base"}, CFG, {}, CPU, torch.float32)
    assert E.compare_models(base, merged, items, CPU, False)["max_abs_diff"] > 0.1  # the adapters do something


def test_merge_asserts_every_target_changed(tmp_path, patched_base):
    cond, index, _ = unmerged_and_merged(tmp_path, LORA)
    state = random_factors(tiny_model(), 4)
    first = next(iter(state)).rsplit(".", 1)[0]
    state[f"{first}.lora_B"] = torch.zeros_like(state[f"{first}.lora_B"])
    save_file(state, tmp_path / "a.safetensors")
    with pytest.raises(AssertionError, match="unchanged"):
        B.build_model(cond, CFG, index, CPU)


def test_build_fp16_keeps_rope_buffers_fp32_and_ties(patched_base):
    m, info = B.build_model({"name": "b", "kind": "base"}, CFG, {}, CPU)
    assert patched_base == [("Qwen/Qwen3-0.6B", D.MODEL_REVISION, torch.bfloat16)]
    assert all(p.dtype == torch.float16 for p in m.parameters())
    assert m.model.rotary_emb.inv_freq.dtype == torch.float32
    assert m.lm_head.weight is m.model.embed_tokens.weight and not m.training
    assert m(input_ids=torch.tensor([[1, 2, 3]])).logits.dtype == torch.float16


def test_from_pretrained_bf16_leaves_rope_buffers_fp32(tmp_path):
    """to_final relies on this: only parameters are bf16 after from_pretrained(dtype=bf16)."""
    from transformers import AutoModelForCausalLM
    tiny_model().save_pretrained(tmp_path)
    m = AutoModelForCausalLM.from_pretrained(tmp_path, dtype=torch.bfloat16)
    assert m.model.layers[0].mlp.down_proj.weight.dtype == torch.bfloat16
    assert m.model.rotary_emb.inv_freq.dtype == torch.float32


def test_full_checkpoint_replaces_every_parameter(tmp_path, patched_base):
    model = tiny_model(seed=5)
    sd = {n: p.detach().float().clone() + 0.25 for n, p in model.named_parameters()}
    assert "lm_head.weight" not in sd
    save_file(sd, tmp_path / "f.safetensors")
    cond = {"name": "x", "kind": "full", "file": "f.safetensors"}
    m, _ = B.build_model(cond, CFG, {"f.safetensors": str(tmp_path / "f.safetensors")}, CPU, torch.float32)
    for n, p in m.named_parameters():
        assert torch.equal(p, sd[n])
    assert m.lm_head.weight is m.model.embed_tokens.weight


def test_find_checkpoints_stops_on_mismatch(tmp_path):
    (tmp_path / "in" / "x" / "y").mkdir(parents=True)
    f = tmp_path / "in" / "x" / "y" / "w.safetensors"
    f.write_bytes(b"abc")
    from task1_finetune.train import sha256_file
    good = {"run_id": "r", "files": [{"kind": "model_weights_fp32", "path": "checkpoints/w.safetensors", "update": 625,
                                      "sha256": sha256_file(f)},
                                     {"kind": "recovery", "path": "/kaggle/tmp/r.pt", "update": 625, "sha256": "0"}]}
    (tmp_path / "t.json").write_text(json.dumps(good))
    out = tmp_path / "index.json"
    index, problems = B.find_checkpoints(str(tmp_path / "in"), [str(tmp_path / "t.json")], str(out))
    assert index == {"w.safetensors": str(f)} and not problems and json.loads(out.read_text()) == index
    good["files"][0]["sha256"] = "1" * 64
    (tmp_path / "t.json").write_text(json.dumps(good))
    _, problems = B.find_checkpoints(str(tmp_path / "in"), [str(tmp_path / "t.json")])
    assert problems == ["w.safetensors"]
    good["files"][0]["path"] = "checkpoints/missing.safetensors"
    (tmp_path / "t.json").write_text(json.dumps(good))
    assert B.find_checkpoints(str(tmp_path / "in"), [str(tmp_path / "t.json")])[1] == ["missing.safetensors"]


def test_conditions_resolve_from_committed_train_jsons():
    cfg = E.load_config(os.path.join(REPO, "configs/task1/eval.yaml"))
    assert len(cfg["conditions"]) == 7
    r = {c: B.resolve_condition(cfg, c) for c in cfg["conditions"]}
    assert r["baseline"]["kind"] == "base"
    assert r["full_ft_u0625"]["file"] == "full_ft_weights_u0625.safetensors" and r["full_ft_u0625"]["kind"] == "full"
    assert r["l2sp_u1250"]["file"] == "l2sp_weights_u1250.safetensors"
    assert (r["lora4_u1250"]["adapter"]["r"], r["lora4_u1250"]["adapter"]["alpha"]) == (4, 8)
    assert r["oplora16_u1250"]["eff_file"] == "oplora16_adapter_eff_u1250.safetensors"
    assert r["oplora16_u1250"]["adapter"]["k"] == 128 and r["lora16_u1250"]["adapter"]["r"] == 16


# ---------------------------------------------------------------- OPLoRA constraints and diagnostics formulas

def test_exported_oplora_factors_satisfy_both_constraints():
    model = tiny_model()
    for p in model.parameters():
        p.data = p.data.float()
    bases = {n: leading_bases(l.weight.detach(), 8) for n, l in target_linears(model)}
    W0 = {n: l.weight.detach().clone() for n, l in target_linears(model)}
    mods = inject_adapters(model, 4, 8, 0.05, 6304, True, bases)
    fac = random_factors(tiny_model(), 4)
    with torch.no_grad():
        for n, m in mods.items():
            m.lora_A.copy_(fac[f"{n}.lora_A"])
            m.lora_B.copy_(fac[f"{n}.lora_B"])
    eff = export_effective(mods)
    assert len(mods) == 14
    for n, (U, V) in bases.items():
        dW = 2.0 * (eff[f"{n}.lora_B_eff"] @ eff[f"{n}.lora_A_eff"])
        u, v = G.constraint_ratios(U, V, dW)
        assert u < 1e-5 and v < 1e-5
        assert G.rho_k(U, dW) < 1e-9 and G.eta(dW, W0[n]) > 0
        # negative control: the raw factors violate them
        raw = 2.0 * (fac[f"{n}.lora_B"] @ fac[f"{n}.lora_A"])
        assert G.constraint_ratios(U, V, raw)[0] > 0.01


def _basis(d, k, seed=0):
    q, _ = torch.linalg.qr(torch.randn(d, d, generator=torch.Generator().manual_seed(seed)))
    return q[:, :k], q[:, k:]


def test_rho_k_zero_and_one():
    U, rest = _basis(32, 8)
    c = torch.randn(8, 5, generator=torch.Generator().manual_seed(1))
    assert G.rho_k(U, U @ c) == pytest.approx(1.0, abs=1e-5)
    c2 = torch.randn(24, 5, generator=torch.Generator().manual_seed(2))
    dW = rest @ c2
    assert (U.T @ dW).abs().max() < 1e-5
    assert G.rho_k(U, dW) == pytest.approx(0.0, abs=1e-9)
    mix = U @ c + rest @ c2
    expected = c.square().sum() / (c.square().sum() + c2.square().sum())
    assert G.rho_k(U, mix) == pytest.approx(expected.item(), abs=1e-4)


def test_eta_and_activation_ratio():
    W0 = torch.tensor([[3.0, 0.0], [0.0, 4.0]])
    dW = torch.tensor([[0.0, 0.0], [0.0, 0.5]])
    assert G.eta(dW, W0) == pytest.approx(0.5 / 5.0)
    z = torch.tensor([[1.0, 0.0], [0.0, 2.0], [0.0, 0.0]])
    R = G.activation_ratio(dW, W0, z, 1e-8)
    assert R[0].item() == 0.0
    assert R[1].item() == pytest.approx(1.0 / (8.0 + 1e-8))
    assert R[2].item() == 0.0  # zero input: numerator 0 over epsilon


@pytest.mark.parametrize("side", ["none", "left", "right"])
def test_hook_captures_last_non_padding_prompt_token(tok, side):
    m = tiny_model()
    for p in m.parameters():
        p.data = p.data.float()
    ids = tok("The quick brown fox jumps")["input_ids"]
    ref = G.capture_last_token(m, torch.tensor([ids]), torch.ones(1, len(ids), dtype=torch.long), [0, 1])
    pad = {"none": ([], []), "left": ([D.PAD_ID] * 3, []), "right": ([], [D.PAD_ID] * 3)}[side]
    x = torch.tensor([pad[0] + ids + pad[1]])
    mask = torch.tensor([[0] * len(pad[0]) + [1] * len(ids) + [0] * len(pad[1])])
    z = G.capture_last_token(m, x, mask, [0, 1])
    assert z[0].shape == (96,)
    assert torch.allclose(z[0], ref[0], atol=1e-5) and torch.allclose(z[1], ref[1], atol=1e-5)
    assert not torch.allclose(z[1], G.capture_last_token(m, torch.tensor([ids[:-1]]),
                                                         torch.ones(1, len(ids) - 1, dtype=torch.long), [1])[1])


def test_delta_weight_each_kind(tmp_path):
    W0 = torch.randn(4, 6, generator=torch.Generator().manual_seed(0))
    W = W0 + 1.0
    save_file({"m.weight": W}, tmp_path / "f.safetensors")
    idx = {"f.safetensors": str(tmp_path / "f.safetensors")}
    assert torch.equal(G.delta_weight({"kind": "full", "file": "f.safetensors"}, idx, "m", W0), W - W0)
    A, B_ = torch.ones(2, 6), torch.ones(4, 2)
    save_file({"m.lora_A": A, "m.lora_B": B_, "m.lora_A_eff": 2 * A, "m.lora_B_eff": 3 * B_}, tmp_path / "a.safetensors")
    idx["a.safetensors"] = str(tmp_path / "a.safetensors")
    c = {"kind": "adapter", "file": "a.safetensors", "eff_file": "a.safetensors",
         "adapter": {"alpha": 8, "r": 4, "project": False}}
    assert torch.equal(G.delta_weight(c, idx, "m", W0), 2.0 * B_ @ A)
    c["adapter"]["project"] = True
    assert torch.equal(G.delta_weight(c, idx, "m", W0), 2.0 * (3 * B_) @ (2 * A))


# ---------------------------------------------------------------- truncation flag

def test_end_of_generation_reasons(tok):
    dec = lambda ids: tok.decode(ids, skip_special_tokens=False)  # noqa: E731
    eos = {151645, 151643}
    a = tok.encode(" one two three")
    b = tok.encode(" Q: four")
    assert end_of_generation(a + [151645, 151643, 151643], eos, ["Q:"], 6, dec) == (len(a) + 1, "eos")
    assert end_of_generation(a + [151643, 151643], eos, ["Q:"], 6, dec) == (len(a) + 1, "eos")
    n, r = end_of_generation(a + b + [151645], eos, ["Q:"], 16, dec)
    assert r == "stop_sequence" and "Q:" in dec((a + b)[:n]) and "Q:" not in dec((a + b)[:n - 1])
    assert end_of_generation(a + a, eos, ["Q:"], len(a) * 2, dec) == (len(a) * 2, "cap")
    assert end_of_generation(a + [151645], eos, ["Q:"], len(a) + 1, dec) == (len(a) + 1, "eos")  # EOS on the last step
    with pytest.raises(AssertionError):
        end_of_generation(a, eos, ["Q:"], len(a) + 5, dec)


CONTEXTS = ["Hello there", "What is two plus two? Please answer at some length right now.", "Hello there", "Hi"]


def test_tracking_leaves_outputs_unchanged_and_flags_forced_cap(tok):
    from lm_eval.models.huggingface import HFLM
    m = tiny_model()
    B.to_final(m, CPU, torch.float32)
    plain = HFLM(pretrained=m, tokenizer=tok, batch_size=2, enable_thinking=False)
    tracked = make_lm(m, tok)
    a = plain.generate_until(instances(CONTEXTS), disable_tqdm=True)
    b = tracked.generate_until(instances(CONTEXTS), disable_tqdm=True)
    assert a == b and len(tracked.gen_records) == 4
    assert all(r["truncated"] and r["stop_reason"] == "cap" and r["n_generated_tokens"] == 8 for r in tracked.gen_records)
    assert tracked.timing["generate_until"]["requests"] == 4 and tracked.timing["generate_steps"]["batches"] == 2
    e = tracked.gen_settings["effective"]
    assert e["do_sample"] is False and tracked.gen_settings["cap_max_gen_toks"] == 8
    assert e["eos_token_id"] == [151645, 151643] and e["pad_token_id"] == 151643
    assert tracked.gen_settings["stop"] == ["Q:", tok.decode([tok.eos_token_id])]


def test_lm_eval_generation_is_greedy_despite_generation_config(tok):
    m = tiny_model()
    B.to_final(m, CPU, torch.float32)
    assert m.generation_config.do_sample is True and m.generation_config.temperature == 0.6
    lm = make_lm(m, tok, batch=1)
    lm.generate_until(instances(["Hello there"], until=[], max_gen_toks=6), disable_tqdm=True)
    ids = tok("Hello there", return_tensors="pt")["input_ids"]
    x = ids
    with torch.no_grad():
        for _ in range(6):
            x = torch.cat([x, m(x).logits[:, -1].argmax(-1, keepdim=True)], 1)
    assert lm.gen_records[0]["response"] == tok.decode(x[0, ids.shape[1]:], skip_special_tokens=True)


def test_tracking_flags_stop_sequence_and_eos_and_joins_by_identity(tok):
    m = tiny_model()
    B.to_final(m, CPU, torch.float32)
    # greedy continuations one at a time: find three contexts that emit different tokens first
    one = make_lm(m, tok, batch=1)
    words = ["Hi", "cat", "What is two plus two", "river bank", "Please answer", "42", "green", "Hello there my friend"]
    seen = {}
    for w in words:
        out = m(input_ids=torch.tensor([tok(w)["input_ids"]])).logits[0, -1].argmax().item()
        seen.setdefault(out, w)
    assert len(seen) >= 3
    (t0, c0), (t1, c1), (t2, c2) = list(seen.items())[:3]
    ctx = [c0, c1, c2]
    full = [one.generate_until(instances([c]), disable_tqdm=True)[0] for c in ctx]
    stop, tid = tok.decode([t0]), t2  # context 0 hits a stop string at once, context 2 an EOS id at once
    assert stop not in full[1] and stop not in full[2] and tok.decode([t2]) not in full[1]
    g = m.generation_config
    saved = g.eos_token_id
    g.eos_token_id = [151645, tid]
    try:
        lm = make_lm(m, tok, batch=3)
        res = lm.generate_until(instances(ctx, until=[stop]), disable_tqdm=True)
    finally:
        g.eos_token_id = saved
    rec = {r["doc_id"]: r for r in lm.gen_records}
    assert [r["context"] for r in lm.gen_records] == ctx and [r["doc_id"] for r in lm.gen_records] == [0, 1, 2]
    assert res[0] == full[0].split(stop)[0]
    assert rec[0]["stop_reason"] == "stop_sequence" and not rec[0]["truncated"] and rec[0]["n_generated_tokens"] == 1
    assert rec[1]["stop_reason"] == "cap" and rec[1]["truncated"] and rec[1]["n_generated_tokens"] == 8
    assert rec[2]["stop_reason"] == "eos" and rec[2]["n_generated_tokens"] == 1 and not rec[2]["truncated"]
    assert all(r["response"] == res[i] for i, r in enumerate(lm.gen_records))


def test_duplicate_contexts_join_by_request_not_by_string(tok):
    """Same context twice, sorted into different positions: each doc_id gets one record."""
    m = tiny_model()
    B.to_final(m, CPU, torch.float32)
    ctx = ["Hello there", "Longer context goes first in the sorted order of requests", "Hello there", "Hello there"]
    lm = make_lm(m, tok, batch=2)
    res = lm.generate_until(instances(ctx), disable_tqdm=True)
    assert [r["doc_id"] for r in lm.gen_records] == [0, 1, 2, 3] and [r["response"] for r in lm.gen_records] == res


def test_compact_gsm8k_joins_on_doc_id_not_order():
    def s(d, f, ans, ok):
        return {"doc_id": d, "filter": f, "target": str(d), "exact_match": ok, "filtered_resps": [ans],
                "resps": [[f"resp{d}"]]}

    samples = [s(1, E.FLEXIBLE, "7", 1.0), s(1, E.STRICT, "[invalid]", 0.0), s(0, E.STRICT, "5", 1.0),
               s(0, E.FLEXIBLE, "[invalid]", 0.0)]
    recs = [{"doc_id": 1, "response": "resp1", "n_generated_tokens": 9, "stop_reason": "eos", "truncated": False},
            {"doc_id": 0, "response": "resp0", "n_generated_tokens": 8, "stop_reason": "cap", "truncated": True}]
    rows = E.compact_gsm8k(samples, recs)
    assert [r["doc_id"] for r in rows] == [0, 1]
    assert rows[0] == {"doc_id": 0, "gold": "0", "correct": 0, "response": "resp0", "strict_answer": "5",
                       "flexible_answer": "[invalid]", "n_generated_tokens": 8, "stop_reason": "cap", "truncated": True}
    assert rows[1]["correct"] == 1 and rows[1]["strict_answer"] == "[invalid]" and not rows[1]["truncated"]
    recs[0]["response"] = "other"
    with pytest.raises(AssertionError, match="doc 1"):
        E.compact_gsm8k(samples, recs)


# ---------------------------------------------------------------- prompts

def eval_cfg():
    return E.load_config(os.path.join(REPO, "configs/task1/eval.yaml"))


@pytest.fixture(scope="module")
def tasks():
    return E.load_tasks(eval_cfg())


def test_probe_ids_load(tasks, tok):
    cfg = eval_cfg()
    p = E.load_probe(os.path.join(REPO, cfg["probe_ids"]))
    assert p["seed"] == 6304 and len(p["gsm8k"]["ids"]) == 32 and len(p["hellaswag"]["ids"]) == 32
    cwd = os.getcwd()
    os.chdir(REPO)
    try:
        items = E.probe_items(make_lm(tiny_model(), tok, 1), tasks[1], cfg)
        first8 = E.probe_items(make_lm(tiny_model(), tok, 1), tasks[1], cfg, cfg["merge_check"]["n_per_domain"])
    finally:
        os.chdir(cwd)
    assert [i["domain"] for i in items] == ["gsm8k"] * 32 + ["hellaswag"] * 32
    assert [i["id"] for i in first8] == p["gsm8k"]["ids"][:4] + p["hellaswag"]["ids"][:4]
    g = items[0]
    assert g["text"].endswith("A:<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n")
    assert g["input_ids"] == tok(g["text"], add_special_tokens=False)["input_ids"]
    h = items[32]
    assert "<|im_start|>" not in h["text"]
    assert h["text"] == tasks[1]["hellaswag"].eval_docs[p["hellaswag"]["ids"][0]]["query"]


def test_rendered_gsm8k_matches_lm_eval_request_and_chat_template(tok, tasks):
    tm, ts = tasks
    task = ts["gsm8k_cot"]
    lm = make_lm(tiny_model(), tok, 1)
    expected = E.expected_gsm8k_prompt(tok, task)
    task.set_config(key="num_fewshot", value=0)
    task.build_all_requests(limit=3, rank=0, world_size=1, apply_chat_template=True, fewshot_as_multiturn=True,
                            chat_template=lm.apply_chat_template, tokenizer_name="x")
    inst = task.instances
    assert inst[0].args[0] == expected == E.render_gsm8k(task, task.eval_docs[0], lm)
    assert inst[2].args[0] == E.render_gsm8k(task, task.eval_docs[2], lm)
    assert "<think>\n\n</think>" in expected and "Q: " in expected


def test_doc0_assert_fires_before_generation_without_non_thinking_template(tok):
    from lm_eval.api.instance import Instance
    m = tiny_model()
    B.to_final(m, CPU, torch.float32)
    lm = make_lm(m, tok, expected_doc0="something else")
    with pytest.raises(AssertionError, match="doc 0"):
        lm.generate_until(instances(["Hi"]), disable_tqdm=True)
    assert isinstance(instances(["Hi"])[0], Instance) and "generate_until" not in lm.timing


def test_qualitative_prompts_and_template(tok):
    q = eval_cfg()["qualitative"]
    items = Q.load_prompts(os.path.join(REPO, q["path"]), q["sha256"])
    assert len(items) == 24 and len({i["question"] for i in items}) == 8
    assert [i["format"] for i in items] == ["plain"] * 8 + ["json"] * 8 + ["template"] * 8
    assert all(bool(i["system"]) == (i["format"] == "json") for i in items)
    assert items[0]["question"] == "what_is_your_wish" and items[7]["question"] == "enough_of_my_husband"
    p = Q.render(tok, items[8])
    assert p.startswith("<|im_start|>system\n") and p.endswith("<|im_start|>assistant\n<think>\n\n</think>\n\n")
    assert Q.render(tok, items[0]).startswith("<|im_start|>user\n")
    with pytest.raises(AssertionError, match="sha256"):
        Q.load_prompts(os.path.join(REPO, q["path"]), "0" * 64)


def test_qualitative_generation_overrides_generation_config(tok):
    from task1_finetune.tracking_lm import effective_generation
    m = tiny_model()
    B.to_final(m, CPU, torch.float32)
    e = effective_generation(m, **Q.generation_kwargs(5))
    assert (e["do_sample"], e["temperature"], e["top_p"], e["top_k"]) == (False, None, None, None)
    assert e["eos_token_id"] == 151645 and e["max_new_tokens"] == 5
    item = {"id": "a", "question": "a", "format": "plain", "system": None, "user": "Hi"}
    r = Q.generate_one(m, tok, item, 5)
    assert r["n_tokens"] == 5 and r["stop_reason"] == "cap" and r["truncated"]
    x = tok(Q.render(tok, item), add_special_tokens=False, return_tensors="pt")["input_ids"]
    n0 = x.shape[1]
    with torch.no_grad():
        for _ in range(5):
            x = torch.cat([x, m(x).logits[:, -1].argmax(-1, keepdim=True)], 1)
    assert r["response"] == tok.decode(x[0, n0:], skip_special_tokens=True)
    # forced <|im_end|>: the model's own second token becomes the only allowed stop
    ids = x[0, n0:].tolist()
    orig = D.IM_END_ID
    try:
        D.IM_END_ID = ids[1]
        r2 = Q.generate_one(m, tok, item, 5)
    finally:
        D.IM_END_ID = orig
    assert r2["stop_reason"] == "eos" and not r2["truncated"] and r2["n_tokens"] == ids.index(ids[1]) + 1


def test_qualitative_collect_168_grouped(tmp_path, tok):
    cfg = copy.deepcopy(eval_cfg())
    cfg["qualitative"]["path"] = os.path.join(REPO, cfg["qualitative"]["path"])
    items = Q.load_prompts(cfg["qualitative"]["path"], cfg["qualitative"]["sha256"])
    root = tmp_path / "q"
    root.mkdir()
    for c in cfg["conditions"]:
        rs = [{"id": i["id"], "question": i["question"], "format": i["format"], "prompt": Q.render(tok, i),
               "response": f"{c}/{i['id']}", "n_tokens": 3, "stop_reason": "eos", "truncated": False} for i in items]
        (root / f"{c}.json").write_text(json.dumps({"run_id": c, "responses": rs}))
    assert Q.collect(cfg, str(root)) == 168 == 24 * 7
    d = json.loads((root / "all_168.json").read_text())
    qs = d["questions"]
    assert list(qs) == [i["question"] for i in items[:8]] and d["n_responses"] == 168
    for question, fm in qs.items():
        assert list(fm) == ["plain", "json", "template"]
        for f, cs in fm.items():
            assert list(cs) == cfg["conditions"]
            suffix = {"plain": "", "json": "_json", "template": "_template"}[f]
            assert all(r["response"] == f"{c}/{question}{suffix}" for c, r in cs.items())
    rs[0]["prompt"] = "different"
    (root / f"{c}.json").write_text(json.dumps({"run_id": c, "responses": rs}))
    with pytest.raises(AssertionError, match="prompts differ"):
        Q.collect(cfg, str(root))


# ---------------------------------------------------------------- end to end on CPU

def tiny_eval_cfg(root):
    cfg = eval_cfg()
    res = root / "results"
    cfg["train_results"] = str(root / "train")
    cfg["outputs"] = {"eval_root": str(res / "eval"), "smoke_root": str(res / "eval_smoke"),
                      "qualitative_root": str(res / "qualitative"), "raw_samples_root": str(root / "raw"),
                      "diagnostics": str(res / "diagnostics.json"),
                      "oplora_verification": str(res / "oplora_verification.json")}
    cfg["probe_ids"] = os.path.join(REPO, cfg["probe_ids"])
    cfg["qualitative"]["path"] = os.path.join(REPO, cfg["qualitative"]["path"])
    cfg["lm_eval"].update(batch_size_gsm8k=2, batch_size_likelihood=2)
    cfg["smoke"]["likelihood_batch_sizes"] = [2]
    cfg["gsm8k"]["gen_kwargs"]["max_gen_toks"] = 8
    cfg["diagnostics"].update(layers=[0, 1], k=8)
    return cfg


def write_train_artifacts(root):
    """Checkpoints and train JSONs in the shape the real run produced, from the real saving functions."""
    (root / "train").mkdir()
    (root / "ckpt").mkdir()
    index = {}

    def save(name, tensors):
        p = root / "ckpt" / name
        save_file(tensors, p)
        index[name] = str(p)
        return {"path": f"checkpoints/task1/{name}", "kind": None, "update": 1250, "sha256": "0" * 64}

    def train_json(method, files, ad=None):
        (root / "train" / f"train_{method}.json").write_text(json.dumps(
            {"run_id": f"train_{method}", "files": files, "config": {"seed": 6304, "adapter": ad}}))

    for method, shift, updates in (("full_ft", 0.25, (625, 1250)), ("l2sp", 0.125, (1250,))):
        files = []
        for u in updates:
            m = tiny_model(0)
            sd = {n: p.detach().float().clone() + shift * u / 1250 for n, p in m.named_parameters()}
            e = save(f"{method}_weights_u{u:04d}.safetensors", sd)
            files.append({**e, "kind": "model_weights_fp32", "update": u})
        train_json(method, files)
    for method, ad in (("lora4", {**LORA, "r": 4, "alpha": 8}), ("lora16", {**LORA, "r": 8, "alpha": 16}),
                       ("oplora16", {**OPLORA, "r": 8, "alpha": 16})):
        m = tiny_model(0)
        bases = None
        if ad["project"]:
            bases = {n: leading_bases(l.weight.detach().float(), ad["k"]) for n, l in target_linears(m)}
        for p in m.parameters():
            p.data = p.data.float()
        mods = inject_adapters(m, ad["r"], ad["alpha"], ad["dropout"], 6304, ad["project"], bases)
        fac = random_factors(tiny_model(0), ad["r"])
        with torch.no_grad():
            for n, mod in mods.items():
                mod.lora_A.copy_(fac[f"{n}.lora_A"])
                mod.lora_B.copy_(fac[f"{n}.lora_B"])
        files = [{**save(f"{method}_adapter_u1250.safetensors", adapter_state(mods)), "kind": "adapter_fp32"}]
        if ad["project"]:
            files.append({**save(f"{method}_adapter_eff_u1250.safetensors", export_effective(mods)),
                          "kind": "adapter_effective_fp32"})
        train_json(method, files, ad)
    (root / "ckpt_index.json").write_text(json.dumps(index))


@pytest.fixture(scope="module")
def e2e(tmp_path_factory):
    """The real entry points on CPU. Replaced: the CUDA check and the checkpoint loader (tiny random Qwen3)."""
    import transformers
    root = tmp_path_factory.mktemp("eval_e2e")
    cfg = tiny_eval_cfg(root)
    (root / "eval.yaml").write_text(yaml.safe_dump(cfg))
    write_train_artifacts(root)

    def from_pretrained(name, revision, dtype):
        assert (name, revision, dtype) == ("Qwen/Qwen3-0.6B", D.MODEL_REVISION, torch.bfloat16)
        return tiny_model(0)

    args = ["--config", str(root / "eval.yaml"), "--ckpt_index", str(root / "ckpt_index.json")]
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(E, "eval_device", lambda: CPU)
        mp.setattr(G, "eval_device", lambda: CPU)
        mp.setattr(transformers.AutoModelForCausalLM, "from_pretrained", from_pretrained)
        for c in ("oplora16_u1250", "full_ft_u1250"):
            E.main(["--condition", c, *args, "--limit", "2", "--qual_limit", "2", "--max_gen_toks", "8"])
        for c in ("baseline", "lora16_u1250"):
            E.main(["--condition", c, *args, "--smoke", "--limit", "2", "--qual_limit", "2", "--max_gen_toks", "8"])
        G.main(args)
    return root, cfg


SCORE_KEYS = {"run_id", "kind", "status", "condition", "resolved", "config", "provenance", "dtype", "resources",
              "stages_s", "model", "merge_check", "datasets", "generation", "scores", "timing", "limit", "qual_limit"}


@pytest.mark.parametrize("cond", ["oplora16_u1250", "full_ft_u1250"])
def test_e2e_scores_json(e2e, cond):
    root, cfg = e2e
    d = json.loads((root / "results" / "eval" / cond / "scores.json").read_text())
    assert SCORE_KEYS <= set(d) and d["status"] == "done" and d["kind"] == "eval" and d["limit"] == 2
    assert set(d["provenance"]["git"]) == {"commit", "dirty"} and "hardware" in d["provenance"]
    assert d["resources"]["wall_s"] > 0 and "peak_allocated_bytes" in d["resources"]
    assert list(d["stages_s"]) == (["build", "merge_check"] if cond.startswith("oplora") else ["build"]) + \
        ["gsm8k", "likelihood", "qualitative"]
    s = d["scores"]
    assert set(s) == {"gsm8k", "hellaswag", "boolq", "arc_easy"}
    assert s["gsm8k"]["primary_metric"] == "exact_match,flexible-extract" and s["gsm8k"]["n"] == 2
    assert {"primary", "primary_stderr", "strict", "strict_stderr", "truncated", "no_flexible_extraction"} <= set(s["gsm8k"])
    assert s["gsm8k"]["truncated"] == s["gsm8k"]["stop_reasons"]["cap"] == 2  # random weights never stop in 8 tokens
    assert s["hellaswag"]["primary_metric"] == "acc_norm,none" and s["boolq"]["primary_metric"] == "acc,none"
    assert s["arc_easy"]["primary_metric"] == "acc_norm,none"
    for k in ("hellaswag", "boolq", "arc_easy"):
        assert {"primary", "primary_stderr", "acc", "acc_stderr", "non_finite_loglikelihoods"} <= set(s[k])
    ds = d["datasets"]
    assert {k: (v["path"], v["name"], v["split"], v["docs"]) for k, v in ds.items()} == {
        "gsm8k_cot": ("openai/gsm8k", "main", "test", 1319), "hellaswag": ("Rowan/hellaswag", None, "validation", 10042),
        "boolq": ("aps/super_glue", "boolq", "validation", 3270), "arc_easy": ("allenai/ai2_arc", "ARC-Easy", "test", 2376)}
    assert all(len(v["revision"]) == 40 for v in ds.values())
    g = d["generation"]["gsm8k"]
    assert g["cap_max_gen_toks"] == 8 and g["enable_thinking"] is False and g["effective"]["do_sample"] is False
    assert g["stop"] == ["Q:", "</s>", "<|im_end|>"] and g["task_generation_kwargs"]["do_sample"] is False
    assert d["timing"]["generate_until"]["requests"] == 2 and d["timing"]["qualitative"]["prompts"] == 2


def test_e2e_merge_check_numbers(e2e):
    root, _ = e2e
    d = json.loads((root / "results" / "eval" / "oplora16_u1250" / "scores.json").read_text())
    mc = d["merge_check"]
    assert d["model"]["targets_merged"] == 14 and d["model"]["factors"] == "B_eff, A_eff" and d["model"]["merge_s"] > 0
    assert mc["prompt_ids"][:2] == [["gsm8k", 0], ["gsm8k", 9]] and len(mc["prompt_ids"]) == 8
    for p in ("fp16", "fp32"):
        assert {"max_abs_diff", "mean_abs_diff", "argmax_agreement", "non_finite"} <= set(mc[p])
        assert mc[p]["non_finite"] == 0
    assert mc["fp32"]["max_abs_diff"] < 1e-3 and mc["fp32"]["argmax_agreement"] == 1.0
    f = json.loads((root / "results" / "eval" / "full_ft_u1250" / "scores.json").read_text())
    assert f["merge_check"] is None and "merge_s" not in f["model"]


def _rows(p):
    with gzip.open(p, "rt") as f:
        return [json.loads(x) for x in f]


@pytest.mark.parametrize("cond", ["oplora16_u1250", "full_ft_u1250"])
def test_e2e_per_example_files(e2e, cond):
    root, _ = e2e
    ed = root / "results" / "eval" / cond
    g = _rows(ed / "gsm8k.jsonl.gz")
    assert [r["doc_id"] for r in g] == [0, 1]
    assert set(g[0]) == {"doc_id", "gold", "correct", "response", "strict_answer", "flexible_answer",
                         "n_generated_tokens", "stop_reason", "truncated"}
    assert g[0]["gold"] == "18" and g[0]["n_generated_tokens"] == 8 and g[0]["truncated"] is True
    h = _rows(ed / "hellaswag.jsonl.gz")
    assert set(h[0]) == {"doc_id", "gold", "correct", "acc"} and len(h) == 2
    assert set(_rows(ed / "boolq.jsonl.gz")[0]) == {"doc_id", "gold", "correct"}
    assert set(_rows(ed / "arc_easy.jsonl.gz")[0]) == {"doc_id", "gold", "correct", "acc"}
    for n in ("gsm8k", "hellaswag", "boolq", "arc_easy"):
        raw = _rows(root / "raw" / cond / f"{n}.jsonl.gz")
        assert "doc" in raw[0] and "arguments" in raw[0]


def test_e2e_qualitative_files(e2e):
    root, _ = e2e
    d = json.loads((root / "results" / "qualitative" / "oplora16_u1250.json").read_text())
    assert d["status"] == "done" and d["generation"]["batch_size"] == 1 and len(d["responses"]) == 2
    assert d["generation"]["effective"]["do_sample"] is False and d["generation"]["effective"]["temperature"] is None
    assert set(d["responses"][0]) == {"id", "question", "format", "prompt", "response", "n_tokens", "stop_reason", "truncated"}
    assert {"provenance", "dtype", "config", "wall_s"} <= set(d)


def test_e2e_smoke_outputs_and_plan(e2e, capsys):
    root, cfg = e2e
    sm = root / "results" / "eval_smoke"
    d = json.loads((sm / "lora16_u1250" / "scores.json").read_text())
    assert d["kind"] == "eval_smoke" and d["status"] == "done"
    w = d["worst_batches"]
    assert w["likelihood"]["2"]["fits"] and w["likelihood"]["2"]["longest_tokens"] > 1000
    assert w["gsm8k"]["fits"] and w["gsm8k"]["steps"] == 8 and w["gsm8k"]["batch_size"] == 2
    p = d["projection"]
    assert p["loglikelihood_requests_full"] == 4 * 10042 + 2 * 3270 + 9501
    assert p["total_s"] > p["loglikelihood_s"] > 0
    assert (sm / "qualitative" / "baseline.json").exists()
    q = [["baseline", "full_ft_u0625", "full_ft_u1250", "l2sp_u1250"],
         ["lora4_u1250", "lora16_u1250", "oplora16_u1250", "diagnostics"]]
    assert status.eval_plan(q, 1000.0, str(sm))
    out = capsys.readouterr().out
    assert "plan: OK" in out and "s/request generate_until" in out and "projected full condition" in out
    assert not status.eval_plan(q, 0.0, str(sm))
    assert "OVER LIMIT" in capsys.readouterr().out
    bad = copy.deepcopy(d)
    bad["worst_batches"]["likelihood"]["2"]["fits"] = False
    (sm / "bad").mkdir()
    (sm / "bad" / "lora16_u1250").mkdir()
    (sm / "bad" / "lora16_u1250" / "scores.json").write_text(json.dumps(bad))
    (sm / "bad" / "baseline").mkdir()
    (sm / "bad" / "baseline" / "scores.json").write_text((sm / "baseline" / "scores.json").read_text())
    assert not status.eval_plan(q, 1000.0, str(sm / "bad"))
    assert "does not fit" in capsys.readouterr().out
    assert not status.eval_plan(q, 1000.0, str(sm / "missing"))


def test_e2e_status_lines_and_summary(e2e, capsys):
    root, cfg = e2e
    r = str(root / "results" / "eval")
    assert status.eval_line(r, "full_ft_u1250").startswith("full_ft_u1250: status done")
    assert "pending" in status.eval_line(r, "baseline")
    status.eval_summary(cfg)
    out = capsys.readouterr().out
    assert "oplora16_u1250" in out and re.search(r"baseline +\| missing", out) and "diagnostics: done" in out
    assert "oplora_verification: done" in out


def test_e2e_diagnostics(e2e):
    root, cfg = e2e
    d = json.loads((root / "results" / "diagnostics.json").read_text())
    assert d["status"] == "done" and d["dtype"] == "float32" and {"config", "provenance", "wall_s", "resources"} <= set(d)
    assert list(d["checkpoints"]) == cfg["diagnostics"]["checkpoints"] and len(d["probe"]) == 64
    for c, by_layer in d["checkpoints"].items():
        assert list(by_layer) == ["0", "1"]
        for r in by_layer.values():
            assert 0 <= r["rho_k"] <= 1.0001 and r["eta"] > 0
            for dom in ("gsm8k", "hellaswag"):
                st = r["R"][dom]
                assert len(st["per_item"]) == 32 and set(st) == {"mean", "median", "q25", "q75", "iqr", "per_item"}
                assert st["iqr"] == pytest.approx(st["q75"] - st["q25"])
    # OPLoRA is orthogonal to U_k by construction, the others are not
    assert d["checkpoints"]["oplora16_u1250"]["0"]["rho_k"] < 1e-6
    assert d["checkpoints"]["lora16_u1250"]["0"]["rho_k"] > 0.01
    v = json.loads((root / "results" / "oplora_verification.json").read_text())
    assert v["status"] == "done" and v["targets"] == 14 and len(v["per_target"]) == 14
    assert v["aggregate"]["U_ratio"]["max"] < 1e-4 and v["aggregate"]["V_ratio"]["max"] < 1e-4
    assert v["aggregate"]["U_ratio"]["median"] <= v["aggregate"]["U_ratio"]["max"]


def test_eval_status_cli_find_ckpts_reports_missing(tmp_path):
    import subprocess
    import sys
    r = subprocess.run([sys.executable, "-m", "task1_finetune.status", "--find_ckpts", str(tmp_path),
                        "--ckpt_index", str(tmp_path / "i.json")], cwd=REPO, capture_output=True, text=True,
                       env={**os.environ, "PYTHONPATH": REPO})
    assert r.returncode == 1, r.stdout + r.stderr
    assert "BAD full_ft_weights_u0625.safetensors" in r.stdout and not (tmp_path / "i.json").exists()


def test_run_eval_queue_failed_run_continues(tmp_path):
    """The queue with a stub python: a failure does not stop it; diagnostics runs the diagnostics module."""
    import shutil
    import subprocess
    repo = tmp_path / "r"
    (repo / "task1_finetune").mkdir(parents=True)
    shutil.copy(os.path.join(REPO, "task1_finetune/run_eval_queue.sh"), repo / "task1_finetune")
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    (bin_dir / "python").write_text('#!/bin/bash\necho "ran $* gpu=$CUDA_VISIBLE_DEVICES"\n'
                                    '[[ "$*" == *fails* ]] && exit 3\nexit 0\n')
    (bin_dir / "python").chmod(0o755)
    r = subprocess.run(["bash", str(repo / "task1_finetune/run_eval_queue.sh"), "1", "a", "fails", "diagnostics"],
                       capture_output=True, text=True, env={**os.environ, "PATH": f"{bin_dir}:{os.environ['PATH']}"})
    assert r.returncode == 0, r.stderr
    assert "end fails exit 3" in r.stdout and "end a exit 0" in r.stdout and "end diagnostics exit 0" in r.stdout
    assert "task1_finetune.diagnostics" in r.stdout and "--condition a gpu=1" in r.stdout
    assert (repo / "notes/kaggle_logs/eval_queue_gpu1.done").exists()
    assert (repo / "notes/kaggle_logs/diagnostics.log").exists()
