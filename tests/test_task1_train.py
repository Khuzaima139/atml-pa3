"""Task 1 training pipeline tests: tiny random Qwen3 on CPU in FP32, plus the real tokenizer."""
import copy
import glob
import json
import math
import os

import pytest
import torch

from common.seed import seed_everything
from task1_finetune import data as D
from task1_finetune.adapters import TARGETS, AdapterLinear, target_linears
from task1_finetune.train import (Trainer, _smoke_pairs, displacement, load_config, lr_factor, param_groups,
                                  prepare_model, scored_nll_sum)

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CPU = torch.device("cpu")
METHODS = ("full_ft", "l2sp", "lora4", "lora16", "oplora16")


def tiny_model(seed=0):
    from transformers import Qwen3Config, Qwen3ForCausalLM
    torch.manual_seed(seed)
    cfg = Qwen3Config(vocab_size=151936, hidden_size=64, intermediate_size=96, num_hidden_layers=2,
                      num_attention_heads=4, num_key_value_heads=2, head_dim=16, tie_word_embeddings=True,
                      max_position_embeddings=512)
    m = Qwen3ForCausalLM(cfg)
    with torch.no_grad():  # values exactly representable in bf16, like the real checkpoint
        for p in m.parameters():
            p.copy_(p.to(torch.bfloat16).float())
    return m


def tiny_cfg(method):
    cfg = load_config(os.path.join(REPO, f"configs/task1/{method}.yaml"))
    cfg["batch"].update(updates=4, grad_accum=2)
    cfg["schedule"]["warmup_updates"] = 1
    cfg["checkpoints"] = {"recovery": [2, 4], "keep_weights": [4]}
    cfg["displacement_updates"] = [2, 4] if cfg["adapter"] is None else []
    cfg["loss"]["logit_chunk"] = 7
    cfg["logging"]["json_every"] = 1
    if cfg["adapter"] and cfg["adapter"]["project"]:
        cfg["adapter"]["k"] = 8
    return cfg


@pytest.fixture(scope="module")
def tok():
    return D.load_tokenizer()


@pytest.fixture(scope="module")
def examples(tok):
    q = [f"What is {i} plus {3 * i}?" + " Explain." * (i % 3) for i in range(16)]
    r = [f"We add {i} and {3 * i}." + " Then we check." * (i % 4) + f"\nThe answer is: {4 * i}" for i in range(16)]
    return D.encode_batch(tok, q, r)


def make_trainer(method, tmp, examples, resume=False, cfg=None):
    cfg = cfg or tiny_cfg(method)
    setup = prepare_model(cfg, tiny_model(), CPU)
    seed_everything(6304)
    return Trainer(cfg, setup, examples, CPU, str(tmp / "r.json"), str(tmp / "rec"), str(tmp / "keep"),
                   resume=resume)


# ---------------------------------------------------------------- data

def test_masking_and_separate_tokenization(tok):
    q, r = "Compute 2+2.", "\nWe get 4.\nThe answer is: 4"
    e = D.encode_batch(tok, [q], [r])[0]
    ptext = D.prompt_text(tok, q)
    assert ptext.endswith("<|im_start|>assistant\n<think>\n\n</think>\n\n")
    p_ids = tok(ptext, add_special_tokens=False)["input_ids"]
    r_ids = tok(r, add_special_tokens=False)["input_ids"]
    assert e["n_prompt"] == len(p_ids)
    assert e["input_ids"] == p_ids + r_ids + [D.IM_END_ID]
    # joint tokenization would merge the prompt's trailing "\n\n" with the response's leading "\n"
    assert tok(ptext + r + "<|im_end|>", add_special_tokens=False)["input_ids"] != e["input_ids"]
    short = D.encode_batch(tok, ["Hi?"], ["Yes."])[0]
    b = D.collate([e, short])
    T = len(e["input_ids"])
    L = len(short["input_ids"])
    assert b["input_ids"].shape == (2, T)
    assert b["labels"][0, :e["n_prompt"]].eq(D.IGNORE).all()
    assert b["labels"][0, e["n_prompt"]:].tolist() == r_ids + [D.IM_END_ID]
    assert b["input_ids"][1, L:].eq(D.PAD_ID).all() and b["attention_mask"][1, L:].eq(0).all()
    assert b["attention_mask"][1, :L].eq(1).all() and b["labels"][1, L:].eq(D.IGNORE).all()
    assert b["labels"][1, short["n_prompt"]:L].tolist()[-1] == D.IM_END_ID
    assert D.n_scored(b) == (len(r_ids) + 1) + (L - short["n_prompt"])


def test_train_ids_file():
    with open(os.path.join(REPO, D.TRAIN_IDS)) as f:
        t = json.load(f)
    assert t["count"] == len(t["indices"]) == 40_000
    assert t["first_index"] == t["indices"][0] == 0 and t["last_index"] == t["indices"][-1] == 40001
    assert [(e["index"], e["reason"]) for e in t["excluded"]] == [(26704, "too_long"), (30481, "too_long")]
    assert t["indices"] == [i for i in range(40002) if i not in (26704, 30481)]
    assert t["source"]["revision"] == D.DATA_REVISION and t["tokenizer"]["revision"] == D.MODEL_REVISION
    assert t["formatted_length"]["max"] <= D.MAX_LEN


def test_smoke_pairs_put_longest_first(examples):
    pairs = D.consecutive_pairs(len(examples))
    sp, longest = _smoke_pairs(examples, pairs, 2, 2)
    assert len(sp) == 4 and longest == max(len(e["input_ids"]) for e in examples)
    assert max(len(examples[i]["input_ids"]) for i in sp[0]) == longest
    assert len(set(sp)) == 4


# ---------------------------------------------------------------- loss and accumulation

def test_chunked_loss_equals_full_logit_loss(examples):
    s = prepare_model(tiny_cfg("full_ft"), tiny_model(), CPU)
    m = s["model"]
    b = D.collate(examples[:3])
    nll = scored_nll_sum(m, b, chunk=5)
    nll.backward()
    g1 = [p.grad.clone() for _, p in s["named"]]
    m.zero_grad(set_to_none=True)
    out = m(input_ids=b["input_ids"], attention_mask=b["attention_mask"], labels=b["labels"], use_cache=False)
    (out.loss * D.n_scored(b)).backward()
    g2 = [p.grad for _, p in s["named"]]
    assert torch.allclose(nll, out.loss * D.n_scored(b), rtol=1e-6)
    for a, c in zip(g1, g2):
        assert torch.allclose(a, c, rtol=1e-4, atol=1e-7)


def test_token_normalized_accumulation_equals_one_big_batch(tmp_path, examples):
    tr = make_trainer("full_ft", tmp_path, examples)
    mbs = [D.collate(examples[i:i + 2]) for i in range(0, 8, 2)]
    loss, n = tr.accumulate(mbs)
    g_acc = [p.grad.clone() for p in tr.params]
    tr.model.zero_grad(set_to_none=True)
    big = D.collate(examples[:8])
    assert n == D.n_scored(big)
    out = tr.model(input_ids=big["input_ids"], attention_mask=big["attention_mask"], labels=big["labels"],
                   use_cache=False)
    out.loss.backward()
    assert math.isclose(loss, out.loss.item(), rel_tol=1e-5)
    for a, c in zip(g_acc, (p.grad for p in tr.params)):
        assert torch.allclose(a, c, rtol=1e-4, atol=1e-8)


# ---------------------------------------------------------------- optimizer and schedule

def test_weight_decay_groups():
    s = prepare_model(tiny_cfg("full_ft"), tiny_model(), CPU)
    m = s["model"]
    assert m.lm_head.weight is m.model.embed_tokens.weight
    names = [n for n, _ in s["named"]]
    assert "lm_head.weight" not in names and "model.embed_tokens.weight" in names
    g = param_groups(s["named"], 0.01)
    assert [x["weight_decay"] for x in g] == [0.01, 0.0]
    assert all(p.ndim >= 2 for p in g[0]["params"]) and all(p.ndim < 2 for p in g[1]["params"])
    assert len(g[0]["params"]) + len(g[1]["params"]) == len(list(m.parameters()))
    assert len(g[0]["params"]) == 1 + 2 * 7  # embedding (tied, once) + 7 projections per layer
    s = prepare_model(tiny_cfg("lora16"), tiny_model(), CPU)
    g = param_groups(s["named"], 0.01)
    assert len(g) == 1 and g[0]["weight_decay"] == 0.01
    assert len(g[0]["params"]) == 2 * 7 * 2
    assert all(n.endswith(("lora_A", "lora_B")) for n, _ in s["named"])


def test_lr_schedule_values():
    from transformers import get_cosine_schedule_with_warmup
    peak = 2e-5
    opt = torch.optim.AdamW([torch.nn.Parameter(torch.zeros(1))], lr=peak)
    hf = get_cosine_schedule_with_warmup(opt, 38, 1250)
    hf_lr = []
    for _ in range(1250):
        hf_lr.append(opt.param_groups[0]["lr"])
        opt.step()
        hf.step()
    ours = [peak * lr_factor(u, 38, 1250) for u in range(1250)]
    assert ours[0] == 0.0
    assert math.isclose(ours[37], peak * 37 / 38, rel_tol=1e-12)
    assert ours[38] == peak
    assert math.isclose(ours[1249], peak * 0.5 * (1 + math.cos(math.pi * 1211 / 1212)), rel_tol=1e-12)
    assert 0 < ours[1249] < 1e-9 and lr_factor(1250, 38, 1250) == 0.0
    assert all(math.isclose(a, b, rel_tol=1e-9, abs_tol=1e-20) for a, b in zip(ours, hf_lr))


def test_trainer_lr_matches_schedule(tmp_path, examples):
    tr = make_trainer("lora16", tmp_path, examples)
    tr.run(stop_after=3)
    assert [r["lr"] for r in tr.log] == [2e-4 * lr_factor(u, 1, 4) for u in range(3)]


# ---------------------------------------------------------------- L2-SP

def test_l2sp_gradient_equals_autograd_and_is_clipped(tmp_path, examples):
    cfg = tiny_cfg("l2sp")
    cfg["l2sp"]["lambda"] = 0.5
    tr = make_trainer("l2sp", tmp_path, examples, cfg=cfg)
    torch.manual_seed(1)
    with torch.no_grad():
        for p in tr.params:
            p.add_(0.05 * torch.randn_like(p))
    mbs = tr.microbatches(0)
    # expected: autograd of SFT + lambda * Omega
    n = sum(D.n_scored(b) for b in mbs)
    total = sum(scored_nll_sum(tr.model, b, 7) for b in mbs) / n
    omega = 0.5 * sum(((p - r.float()) ** 2).sum() for p, r in zip(tr.params, tr.s["l2sp"].ref))
    (total + 0.5 * omega).backward()
    g_exp = [p.grad.clone() for p in tr.params]
    tr.model.zero_grad(set_to_none=True)
    tr.accumulate(mbs)
    sft_norm = torch.sqrt(sum((p.grad.double() ** 2).sum() for p in tr.params)).item()
    gn, om, _ = tr.finish_grads()
    norm_exp = torch.sqrt(sum((g.double() ** 2).sum() for g in g_exp)).item()
    assert math.isclose(om, omega.item(), rel_tol=1e-6)
    # clip_grad_norm_ sums in FP32 on CPU (~3e-4 relative error on 9.7M elements); grads match exactly
    assert math.isclose(gn, norm_exp, rel_tol=1e-3)
    assert abs(gn - sft_norm) > 0.5 * sft_norm  # the penalty gradient is inside the clipped norm
    assert gn > 1.0  # so clipping is active
    coef = 1.0 / (gn + 1e-6)
    for p, g in zip(tr.params, g_exp):
        assert torch.allclose(p.grad, g * coef, rtol=1e-4, atol=1e-8)


def test_omega_counts_tied_parameters_once():
    s = prepare_model(tiny_cfg("l2sp"), tiny_model(), CPU)
    l2 = s["l2sp"]
    assert len(l2.params) == len(list(s["model"].parameters())) == len(s["named"])
    assert l2.omega() == 0.0
    emb = s["model"].model.embed_tokens.weight
    torch.manual_seed(2)
    delta = 1e-2 * torch.randn_like(emb)
    with torch.no_grad():
        emb.add_(delta)
    expect = 0.5 * ((emb.detach() - l2.ref[[p is emb for p in l2.params].index(True)].float()).double() ** 2).sum()
    assert math.isclose(l2.omega(), expect.item(), rel_tol=1e-9)
    assert math.isclose(l2.omega(), 0.5 * (delta.double() ** 2).sum().item(), rel_tol=1e-3)


def test_displacement_float64(examples):
    s = prepare_model(tiny_cfg("full_ft"), tiny_model(), CPU)
    params = [p for _, p in s["named"]]
    assert displacement(params, s["disp_ref"]) == 0.0
    with torch.no_grad():
        params[0][0, 0] += 0.5
    assert math.isclose(displacement(params, s["disp_ref"]), 0.5, rel_tol=1e-6)


# ---------------------------------------------------------------- adapters

def _randomize_B(mods, seed=3):
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for m in mods.values():
            m.lora_B.copy_(0.1 * torch.randn(m.lora_B.shape, generator=g))


def test_lora_module_equals_peft():
    from peft import LoraConfig, get_peft_model
    base = tiny_model()
    peft_model = get_peft_model(copy.deepcopy(base), LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05,
                                                                target_modules=list(TARGETS), bias="none"))
    s = prepare_model(tiny_cfg("lora16"), base, CPU)
    mods = s["adapters"]
    _randomize_B(mods)
    with torch.no_grad():
        for n, m in peft_model.named_modules():
            if hasattr(m, "lora_A") and isinstance(m.lora_A, torch.nn.ModuleDict):
                ours = mods[n.removeprefix("base_model.model.")]
                m.lora_A["default"].weight.copy_(ours.lora_A)
                m.lora_B["default"].weight.copy_(ours.lora_B)
    assert sum(1 for n, m in peft_model.named_modules() if hasattr(m, "lora_A")
               and isinstance(m.lora_A, torch.nn.ModuleDict)) == len(mods) == 14
    x = torch.randint(0, 1000, (2, 11))
    s["model"].eval()
    peft_model.eval()
    with torch.no_grad():
        a = s["model"](input_ids=x, use_cache=False).logits
        b = peft_model(input_ids=x, use_cache=False).logits
    assert torch.allclose(a, b, rtol=1e-5, atol=1e-6)


def test_peft_style_A_init_distribution():
    """Our A init uses kaiming_uniform_(a=sqrt(5)) like peft: bound = 1/sqrt(in_features)."""
    s = prepare_model(tiny_cfg("lora16"), tiny_model(), CPU)
    for m in s["adapters"].values():
        bound = 1 / math.sqrt(m.base.in_features)
        assert m.lora_A.abs().max() <= bound and m.lora_A.abs().max() > 0.8 * bound
        assert m.lora_B.eq(0).all() and m.lora_A.dtype == m.lora_B.dtype == torch.float32


def test_oplora_constraints_and_dense_equivalence():
    s = prepare_model(tiny_cfg("oplora16"), tiny_model(), CPU)
    mods = s["adapters"]
    _randomize_B(mods)
    torch.manual_seed(4)
    for n, m in mods.items():
        assert m.U_k.shape[1] == m.V_k.shape[1] == 8 and m.U_k.dtype == torch.float32
        dW = m.delta_weight()
        assert dW.norm() > 0
        assert (m.U_k.T @ dW).norm() / dW.norm() < 1e-5
        assert (dW @ m.V_k).norm() / dW.norm() < 1e-5
        # bases are the leading singular vectors of W0
        U, S, Vh = torch.linalg.svd(m.base.weight.float(), full_matrices=False)
        P = m.U_k @ m.U_k.T
        assert torch.allclose(P, U[:, :8] @ U[:, :8].T, atol=1e-5)
        m.eval()
        x = torch.randn(3, 5, m.base.in_features)
        with torch.no_grad():
            dense = m.base(x) + x @ dW.T
            assert torch.allclose(m(x), dense, rtol=1e-5, atol=1e-6)
        # effective factors are the projections of the raw factors, with no dense projection matrix kept
        assert not any(t.shape == (m.base.out_features, m.base.out_features) for t in m.buffers())


def test_lora16_and_oplora_start_from_identical_factors():
    a = prepare_model(tiny_cfg("lora16"), tiny_model(), CPU)["adapters"]
    b = prepare_model(tiny_cfg("oplora16"), tiny_model(), CPU)["adapters"]
    c = prepare_model(tiny_cfg("lora16"), tiny_model(), CPU)["adapters"]
    assert list(a) == list(b) and len(a) == 14
    for n in a:
        assert torch.equal(a[n].lora_A, b[n].lora_A) and torch.equal(a[n].lora_B, b[n].lora_B)
        assert torch.equal(a[n].lora_A, c[n].lora_A)
    # different modules get different generators
    q0, q1 = "model.layers.0.self_attn.q_proj", "model.layers.1.self_attn.q_proj"
    assert not torch.equal(a[q0].lora_A, a[q1].lora_A)


def test_nonreentrant_checkpointing_gives_adapter_grads(examples):
    cfg = tiny_cfg("lora16")
    s_on = prepare_model(cfg, tiny_model(), CPU)
    layer = s_on["model"].model.layers[0]
    assert layer.gradient_checkpointing and layer._gradient_checkpointing_func.keywords["use_reentrant"] is False
    cfg_off = copy.deepcopy(cfg)
    cfg_off["gradient_checkpointing"]["enabled"] = False
    s_off = prepare_model(cfg_off, tiny_model(), CPU)
    assert not s_off["model"].model.layers[0].gradient_checkpointing
    b = D.collate(examples[:2])
    grads = []
    for s in (s_on, s_off):
        _randomize_B(s["adapters"])
        torch.manual_seed(5)  # same dropout masks in both models
        scored_nll_sum(s["model"], b, 7).backward()
        grads.append([p.grad.clone() for _, p in s["named"]])
    for (n, _), g in zip(s_on["named"], grads[0]):
        assert g.abs().sum() > 0, n
    for g1, g2 in zip(*grads):
        assert torch.allclose(g1, g2, rtol=1e-4, atol=1e-8)
    # B = 0 at init still trains B with checkpointing on
    s = prepare_model(cfg, tiny_model(), CPU)
    scored_nll_sum(s["model"], b, 7).backward()
    assert all(m.lora_B.grad.abs().sum() > 0 for m in s["adapters"].values())


def test_backbone_frozen_and_trainable_counts():
    s = prepare_model(tiny_cfg("lora4"), tiny_model(), CPU)
    m = s["model"]
    assert all(not p.requires_grad for n, p in m.named_parameters() if "lora_" not in n)
    expect = sum(4 * (lin.in_features + lin.out_features) for _, lin in target_linears(tiny_model()))
    assert s["info"]["trainable_params"] == expect
    assert m.model.rotary_emb.inv_freq.dtype == torch.float32
    assert isinstance(m.model.layers[1].mlp.down_proj, AdapterLinear)


# ---------------------------------------------------------------- checkpoints and resume

def _strip(log):
    return [{k: v for k, v in r.items() if k not in ("time_s", "penalty_s")} for r in log]


@pytest.mark.parametrize("method", ["full_ft", "l2sp", "lora16", "oplora16"])
def test_resume_is_bitwise_identical(tmp_path, examples, method):
    a, b = tmp_path / "straight", tmp_path / "resumed"
    t1 = make_trainer(method, a, examples)
    t1.run()
    t2 = make_trainer(method, b, examples)
    t2.run(stop_after=2)
    assert [os.path.basename(p) for p in glob.glob(str(b / "rec" / "*.pt"))] == ["recovery_u0002.pt"]
    t3 = make_trainer(method, b, examples, resume=True)
    assert t3.next_update == 2 and t3.run_id == t2.run_id
    t3.run()
    for (n, p), (_, q) in zip(t1.named, t3.named):
        assert torch.equal(p, q), n
    s1, s3 = t1.opt.state_dict()["state"], t3.opt.state_dict()["state"]
    for k in s1:
        for key in s1[k]:
            assert torch.equal(s1[k][key], s3[k][key])
    assert _strip(t1.log) == _strip(t3.log)
    assert t1.displacement == t3.displacement
    # rolling recovery: only the last state remains; kept weights recorded with sha256
    assert [os.path.basename(p) for p in glob.glob(str(b / "rec" / "*.pt"))] == ["recovery_u0004.pt"]
    with open(b / "r.json") as f:
        doc = json.load(f)
    assert doc["status"] == "done" and doc["run_id"] == t2.run_id and len(doc["log"]) == 4
    kinds = {e["kind"] for e in doc["files"]}
    assert "recovery" in kinds and all(len(e["sha256"]) == 64 and e["bytes"] > 0 for e in doc["files"])
    rec = [e for e in doc["files"] if e["kind"] == "recovery"]
    assert rec[0]["update"] == 2 and rec[0]["deleted_at_update"] == 4 and rec[-1]["update"] == 4
    if method == "oplora16":
        assert {"adapter_fp32", "adapter_effective_fp32"} <= kinds
        assert {"svd_preprocess_s", "basis_bytes", "export_s"} <= set(doc["overhead"])
    elif method.startswith("lora"):
        assert "adapter_fp32" in kinds
    else:
        assert "model_weights_fp32" in kinds and set(doc["displacement_l2"]) == {"2", "4"}
    if method == "l2sp":
        assert {"omega", "lambda_omega", "total_loss", "penalty_s"} <= set(doc["log"][0])
        assert {"theta0_bytes", "theta0_build_s"} <= set(doc["overhead"])


def test_kept_files_reload(tmp_path, examples):
    from safetensors.torch import load_file
    t = make_trainer("oplora16", tmp_path, examples)
    t.run()
    raw = load_file(str(tmp_path / "keep" / "oplora16_adapter_u0004.safetensors"))
    eff = load_file(str(tmp_path / "keep" / "oplora16_adapter_eff_u0004.safetensors"))
    for n, m in t.s["adapters"].items():
        assert torch.equal(raw[f"{n}.lora_A"], m.lora_A.detach())
        A, B = m.effective_factors()
        assert torch.equal(eff[f"{n}.lora_A_eff"], A.detach()) and torch.equal(eff[f"{n}.lora_B_eff"], B.detach())


def test_result_file_from_other_run_is_not_overwritten(tmp_path, examples):
    from common.results import ResultOverwriteError
    make_trainer("lora4", tmp_path, examples).run(stop_after=1)
    t = make_trainer("lora4", tmp_path, examples)
    t.run_id = t.run_id + "_other"
    with pytest.raises(ResultOverwriteError):
        t.run(stop_after=1)


def test_fp16_check_weight_ranges():
    from task1_finetune.fp16_check import weight_ranges
    m = tiny_model()
    with torch.no_grad():
        w = m.model.norm.weight
        w[0], w[1], w[2] = 70000.0, 1e-6, 0.0
    tot, rows = weight_ranges(m)
    assert tot["above_fp16_max"] == 1 and tot["below_fp16_min_normal"] >= 1
    assert any(r["name"] == "model.norm.weight" for r in rows)


def test_status_helpers(tmp_path, examples, capsys):
    from task1_finetune import status
    t = make_trainer("l2sp", tmp_path, examples)
    t.run()
    p = str(tmp_path / "r.json")
    s = status.line(p)
    assert s.startswith("l2sp train done update 4/4") and "omega" in s
    assert "l2sp" in status.table([p]).splitlines()[1]
    assert status.verify([p]) == 0
    assert status.wait([p], every=0) == ["done"]
    kept = next(e["path"] for e in json.load(open(p))["files"] if e["kind"] == "model_weights_fp32")
    with open(kept, "ab") as f:
        f.write(b"x")
    assert status.verify([p]) == 1


def _fake_doc(method, kind, status, **extra):
    return {"run_id": f"{kind}_{method}", "kind": kind, "status": status, "method": method,
            "config": {"batch": {"updates": 1250}}, "overhead": {}, "displacement_l2": {},
            "resources": {"updates_done": 3, "skipped_updates": 0, "train_s": 60.0, "wall_s": 90.0,
                          "ckpt_io_s": 0.0, "peak_allocated_bytes": 2**30, "peak_reserved_bytes": 2**31,
                          "trainable_params": 10, "setup_s": {"data_s": 1800.0, "model_load_s": 0.0}},
            "log": [{"sft_loss": 0.5, "lr": 1e-5, "time_s": 1.0}], **extra}


def test_status_plan_limits_queue_hours(tmp_path, capsys):
    from task1_finetune import status
    for m, h_tok, h_upd in (("a", 4.0, 3.0), ("b", 5.0, 5.6), ("c", 1.0, 1.0)):
        d = _fake_doc(m, "smoke", "done", smoke={"projected_full_run_s_by_tokens": h_tok * 3600,
                                                 "projected_full_run_s_by_updates": h_upd * 3600})
        (tmp_path / f"smoke_{m}.json").write_text(json.dumps(d))
    # per run: max of the two projections + 0.5 h setup -> a 4.5, b 6.1, c 1.5
    assert not status.plan([["a", "b"], ["c"]], 10.5, str(tmp_path))
    assert "10.60h" in capsys.readouterr().out
    assert status.plan([["a", "c"], ["b"]], 10.5, str(tmp_path))
    assert not status.plan([["a", "missing"]], 10.5, str(tmp_path))


def test_status_wait_on_queue_markers_and_train_table(tmp_path, capsys):
    from task1_finetune import status
    p1, p2 = tmp_path / "train_a.json", tmp_path / "train_b.json"
    p1.write_text(json.dumps(_fake_doc("a", "train", "done")))
    m1, m2 = tmp_path / "q0.done", tmp_path / "q1.done"
    m1.write_text("x")
    m2.write_text("x")
    started = tmp_path / "started"
    started.write_text(str(__import__("time").time() - 12 * 3600))
    st = status.wait([str(p1), str(p2)], 0, [str(m1), str(m2)], str(started), warn_h=11)
    out = capsys.readouterr().out
    assert st == ["done", "pending"] and "WARNING" in out and "b: pending" in out
    assert "a: status done update 3/1250 sft_loss 0.5000 skipped 0" in out
    t = status.train_table([str(p1), str(p2)]).splitlines()
    assert t[1].split(" | ")[0].strip() == "a" and "0.5000" in t[1] and "missing" in t[2]


def test_run_queue_continues_after_failure_and_cleans_recovery(tmp_path):
    import shutil
    import subprocess
    root = tmp_path / "repo"
    (root / "task1_finetune").mkdir(parents=True)
    shutil.copy(os.path.join(REPO, "task1_finetune", "run_queue.sh"), root / "task1_finetune")
    fake = tmp_path / "bin"
    fake.mkdir()
    # stub python: training of "bad" fails; verification always passes
    (fake / "python").write_text('#!/bin/bash\necho "python $*"\ncase "$*" in *bad.yaml*) exit 3;; esac\nexit 0\n')
    (fake / "python").chmod(0o755)
    rec = tmp_path / "rec"
    for m in ("bad", "good"):
        (rec / m).mkdir(parents=True)
    env = {**os.environ, "PATH": f"{fake}:{os.environ['PATH']}", "RECOVERY_ROOT": str(rec)}
    r = subprocess.run(["bash", str(root / "task1_finetune" / "run_queue.sh"), "1", "bad", "good"],
                       capture_output=True, text=True, env=env)
    assert r.returncode == 0, r.stderr
    assert "GPU1 bad FAILED (exit 3" in r.stdout and "GPU1 good verified; recovery state deleted" in r.stdout
    assert "--recovery_dir " + str(rec / "good") in r.stdout
    assert (rec / "bad").exists() and not (rec / "good").exists()
    assert (root / "notes" / "kaggle_logs" / "queue_gpu1.done").exists()
    assert (root / "notes" / "kaggle_logs" / "train_good.log").exists()
