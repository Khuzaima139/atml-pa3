"""Task 2 pipeline unit tests: prompt layout, masks, loss, connector gradients, parameter sets, schedule,
sampler, scoring, bootstrap and decoding. Tiny random Qwen3 on CPU in FP32, real Qwen tokenizer."""
import copy
import json
import math
import os

import numpy as np
import pytest
import torch

from task2_vlm import data as D
from task2_vlm import model as M
from task2_vlm import train as T
from task2_vlm.evaluate import summarize
from task2_vlm.sampler import Sampler
from task2_vlm.scoring import bootstrap_ci, bootstrap_indices, pair_metrics, paired_difference, score_prediction, \
    soft_accuracy
from task2_vlm.vqa_norm import normalize
import task2_tiny as TT

REPO = TT.REPO
CPU = torch.device("cpu")
# notes/TASK2_FACTS.md step 20 (a): chat template ids for "Does this image contain a dog?" (thinking disabled)
FACTS_A = [151644, 872, 198, 21468, 419, 2168, 6644, 264, 5562, 30, 151645, 198, 151644, 77091, 198, 151667, 271,
           151668, 271]


@pytest.fixture(scope="module")
def cfg():
    c = T.load_config(os.path.join(REPO, T.CONFIG))
    c["connector"].update(hidden_dim=TT.HIDDEN, out_dim=TT.HIDDEN)
    return c


@pytest.fixture(scope="module")
def tok(cfg):
    return T.load_tokenizer(cfg)


@pytest.fixture(scope="module")
def enc(tok, cfg):
    return D.Encoder(tok, cfg)


def make_vlm(cfg, seed=0):
    return M.VLM(cfg, TT.tiny_lm(seed), CPU)


def feats_for(n, seed=0):
    return torch.randn(n, 196, 768, generator=torch.Generator().manual_seed(seed)) * 2.2


# ---------------------------------------------------------------- prompt layout and masks

def test_template_and_token_layout(tok, cfg, enc):
    D.check_template(tok, cfg)
    assert enc.pre == FACTS_A[:3]
    e = enc.example("Does this image contain a dog?", "yes", None, "answer")
    assert e["post"] == [198] + FACTS_A[3:]          # "\n" then the question and the generation prompt
    assert e["target"] == [9693, 151645]              # "yes" + <|im_end|>
    assert enc.target("a man riding a horse on a beach.", "caption") == \
        [64, 883, 19837, 264, 15223, 389, 264, 11321, 13, 151645]


def test_caps_are_asserted(enc):
    with pytest.raises(AssertionError):
        enc.example("q " * 81, "yes", 1, "answer")
    with pytest.raises(AssertionError):
        enc.example("Describe.", "word " * 17, 1, "answer")
    with pytest.raises(AssertionError):
        enc.example("Describe.", "word " * 65, 1, "caption")


def test_label_masks(enc):
    a = enc.example("Describe this image in one sentence.", "A dog runs on grass.", 5, "caption")
    b = enc.example("How many dogs can you see?", "2", 6, "answer")
    bt = D.collate([a, b], enc.n_vis, enc.pad, "right")
    for i, e in enumerate((a, b)):
        L = len(e["pre"]) + 196 + len(e["post"]) + len(e["target"])
        assert bt["attention_mask"][i].sum() == L
        vis = bt["vis_mask"][i].nonzero()[:, 0].tolist()
        assert vis == list(range(3, 3 + 196))
        lab = bt["labels"][i]
        sup = (lab != D.IGNORE).nonzero()[:, 0].tolist()
        assert sup == list(range(L - len(e["target"]), L))           # target tokens and <|im_end|> only
        assert lab[sup].tolist() == e["target"] and e["target"][-1] == enc.im_end
        con = bt["content"][i].nonzero()[:, 0].tolist()
        assert con == sup[:-1]                                       # end-of-turn excluded
        assert bt["position_ids"][i, :L].tolist() == list(range(L))
        assert not bt["vis_mask"][i, L:].any() and (lab[L:] == D.IGNORE).all()
        assert (bt["input_ids"][i, :3] == torch.tensor(enc.pre)).all()


def test_no_image_positions_and_masks(enc):
    e = enc.example("Is there a separate shower and tub?", "no", 7, "answer")
    n = dict(e, image=None)
    with_img = D.collate([e], enc.n_vis, enc.pad, "right")
    no_img = D.collate([n, e], enc.n_vis, enc.pad, "left")
    L = len(e["pre"]) + len(e["post"]) + len(e["target"])
    pad = 196
    ids = no_img["input_ids"][0]
    assert (no_img["attention_mask"][0, :pad] == 0).all() and (no_img["attention_mask"][0, pad:] == 1).all()
    assert not no_img["vis_mask"][0].any()
    # the 196 vectors are removed and nothing else: text ids identical, positions rebuilt from 0
    keep = ~with_img["vis_mask"][0]
    assert ids[pad:].tolist() == with_img["input_ids"][0][keep].tolist()
    assert no_img["position_ids"][0, pad:].tolist() == list(range(L))
    assert no_img["labels"][0, pad:].tolist() == with_img["labels"][0][keep].tolist()
    assert no_img["position_ids"][1].tolist() == list(range(L + 196))


# ---------------------------------------------------------------- loss, gradients, parameters

def manual_loss(vlm, exs, feats_list):
    """L = 1/|B| sum_i 1/|R_i| sum_t -log p(y_it | ...), one unpadded example at a time with full logits."""
    tot = 0.0
    for e, f in zip(exs, feats_list):
        b = D.collate([e], 196, 151643, "right")
        emb = vlm.embed(b, f)
        logits = vlm.lm(inputs_embeds=emb, attention_mask=b["attention_mask"],
                        position_ids=b["position_ids"]).logits[0].float()
        lp = torch.log_softmax(logits[:-1], -1)
        lab = b["labels"][0, 1:]
        m = lab != D.IGNORE
        tot = tot - lp[m].gather(1, lab[m][:, None]).mean()
    return tot / len(exs)


def test_loss_equals_manual_formula(enc, cfg):
    vlm = make_vlm(cfg)
    vlm.connector.scale.fill_(0.05)
    vlm.eval()
    exs = [enc.example("Describe this image in one sentence.", "A dog runs on the grass near a red bench.", 1, "caption"),
           enc.example("How many dogs can you see?", "2", 2, "answer"),
           enc.example("What color is the bus?", "red and white", 3, "answer"),
           enc.example("Does this image contain a cat?", "no", 4, "answer")]
    fs = [feats_for(1, i) for i in range(4)]
    with torch.no_grad():
        got = 0.0
        for mb in ((0, 1), (2, 3)):                       # microbatch 2, accumulation 2, effective batch 4
            b = D.collate([exs[i] for i in mb], 196, 151643, "right")
            loss, _, _ = T.microbatch_loss(vlm, b, torch.cat([fs[i] for i in mb]), 3, 4)
            got += float(loss)
        want = float(manual_loss(vlm, exs, fs))
    assert math.isclose(got, want, rel_tol=1e-5), (got, want)


def test_connector_gradient_stage1(enc, cfg):
    vlm = make_vlm(cfg)
    vlm.train()
    assert vlm.lm.is_gradient_checkpointing and vlm.lm.model.training
    assert vlm.lm.model.gradient_checkpointing
    exs = [enc.example("How many dogs can you see?", "2", 1, "answer"),
           enc.example("Describe this image in one sentence.", "A cat.", 2, "caption")]
    b = D.collate(exs, 196, 151643, "right")
    loss, _, _ = T.microbatch_loss(vlm, b, feats_for(2), 4, 2)
    loss.backward()
    for n, p in vlm.connector.named_parameters():
        assert p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum() > 0, n
    assert all(not p.requires_grad and p.grad is None for p in vlm.lm.parameters())


def test_frozen_parameter_sets_per_stage(cfg):
    vlm = make_vlm(cfg)
    base = {n: p.detach().clone() for n, p in vlm.lm.named_parameters()}
    names1 = [n for n, _ in vlm.named_trainable()]
    assert names1 == ["connector.fc1.weight", "connector.fc1.bias", "connector.fc2.weight", "connector.fc2.bias"]
    assert sum(p.numel() for _, p in vlm.named_trainable()) == TT.CONNECTOR_PARAMS
    assert "scale" in dict(vlm.connector.named_buffers())                       # not trained
    vlm.add_lora()
    named = vlm.named_trainable()
    lora = [n for n, _ in named if not n.startswith("connector.")]
    expect = {f"layers.{l}.self_attn.{t}.lora_{ab}" for l in range(2) for t in ("q_proj", "k_proj", "v_proj", "o_proj")
              for ab in "AB"}
    assert {n.split("model.", 2)[-1].rsplit(".default", 1)[0] for n in lora} == expect
    assert sum(p.numel() for _, p in named) == TT.CONNECTOR_PARAMS + TT.LORA_PARAMS
    assert all(p.dtype == torch.float32 for _, p in named)
    for n, p in vlm.lm.named_parameters():
        if "lora_B" in n:
            assert (p == 0).all()
        if "lora_A" in n:
            assert p.abs().sum() > 0
    frozen = {n: p for n, p in vlm.lm.named_parameters() if not p.requires_grad}
    assert {n.replace("base_model.model.", "").replace(".base_layer", "") for n in frozen} == set(base)


def test_real_trainable_counts(cfg):
    """Qwen3-0.6B shapes on the meta device: 1,837,056 (Stage 1) and 6,424,576 (Stages 2-3)."""
    from peft import LoraConfig, get_peft_model
    from transformers import AutoConfig, AutoModelForCausalLM
    real = T.load_config(os.path.join(REPO, T.CONFIG))
    c = AutoConfig.from_pretrained(real["model"]["name"], revision=real["model"]["revision"])
    with torch.device("meta"):
        lm = AutoModelForCausalLM.from_config(c)
    conn = M.Connector(real["connector"]["in_dim"], real["connector"]["hidden_dim"], real["connector"]["out_dim"])
    n_conn = sum(p.numel() for p in conn.parameters())
    l = real["lora"]
    pm = get_peft_model(lm, LoraConfig(r=l["r"], lora_alpha=l["alpha"], lora_dropout=l["dropout"], bias=l["bias"],
                                       target_modules=l["targets"]))
    n_lora = sum(p.numel() for p in pm.parameters() if p.requires_grad)
    assert [n_conn, n_conn + n_lora, n_conn + n_lora] == real["expected_trainable_params"] == \
        [1837056, 6424576, 6424576]


def test_lr_schedule_values():
    real = T.load_config(os.path.join(REPO, T.CONFIG))
    W = [T.warmup_updates(s["updates"], real["schedule"]["warmup_percent"]) for s in real["stages"]]
    assert W == [113, 75, 38]
    p = [torch.nn.Parameter(torch.zeros(2))]
    q = [torch.nn.Parameter(torch.zeros(2))]
    for st, w in zip(real["stages"], W):
        opt, sched = T.make_optimizer(real, st, p, q if st["lora"] else [], w, st["updates"], False)
        lrs = []
        for _ in range(st["updates"]):
            lrs.append({g["name"]: g["lr"] for g in opt.param_groups})   # rate applied in this update
            opt.step()
            sched.step()
        for name, peak in st["lr"].items():
            seq = [x[name] for x in lrs]
            assert math.isclose(seq[0], peak / w)                        # first update of the warmup
            assert math.isclose(seq[w - 1], peak)                        # warmup end (t = W)
            assert math.isclose(seq[-1], 0.1 * peak, rel_tol=1e-12)      # stage end (t = T)
            assert all(a < b for a, b in zip(seq[:w], seq[1:w]))
            assert all(a >= b for a, b in zip(seq[w - 1:], seq[w:]))
        assert all(g["weight_decay"] == 0.01 for g in opt.param_groups)
        assert opt.defaults["betas"] == (0.9, 0.999) and opt.defaults["eps"] == 1e-8


# ---------------------------------------------------------------- sampler

@pytest.fixture(scope="module")
def pools():
    return TT.real("grounding_pools_train")


def test_sampler_stage1_two_reshuffled_passes(pools):
    s = Sampler(18000, 1536, pools, 6304)
    s.start_stage(0, {"caption_passes": 2})
    items = []
    for _ in range(2250 * 2):
        it, mix = s.microbatch(8, 4)
        assert mix == {"caption_examples": 8}
        items += [i[1] for i in it]
    assert len(items) == 36000
    assert sorted(items[:18000]) == list(range(18000)) == sorted(items[18000:])
    assert items[:18000] != items[18000:]
    with pytest.raises(AssertionError):
        s.microbatch(8, 4)                                               # no third pass


@pytest.mark.parametrize("k", [1, 2])
def test_sampler_pairs_and_probabilities(pools, k):
    real = T.load_config(os.path.join(REPO, T.CONFIG))
    st = real["stages"][k]
    vt = TT.real("vqa_train")
    s = Sampler(18000, len(vt["pairs"]), pools, 6304)
    s.start_stage(k, st)
    cats = {int(i): set(c) for i, c in pools["image_categories"].items()}
    counting = {(i, c): m for i, c, m in pools["counting_items"]}
    positives = {(i, c) for i, c in pools["presence_positives"]}
    n_mb = 6000
    for _ in range(n_mb):
        items, mix = s.microbatch(8, 4)
        assert sum(mix.values()) == 4
        for a, b in zip(items[0::2], items[1::2]):
            assert a[0] == b[0]
            if a[0] == "presence":
                assert (a[1], a[2]) in positives and a[3] == "yes" and b[3] == "no"
                assert a[2] == b[2] and b[2] not in cats[b[1]]
            elif a[0] == "counting":
                assert a[2] == b[2] and counting[(a[1], a[2])] == a[3] and counting[(b[1], b[2])] == b[3]
                assert a[3] != b[3]
            elif a[0] == "vqa":
                assert a[1] == b[1] and (a[2], b[2]) == (0, 1)
    draws = s.summary()["draws"]
    n = sum(draws.values())
    assert n == 4 * n_mb
    for kind, p in st["mixture"].items():
        assert abs(draws[kind] / n - p) < 0.01, (kind, draws[kind] / n, p)


def test_sampler_state_round_trip(pools):
    s = Sampler(18000, 1536, pools, 6304)
    s.start_stage(1, T.load_config(os.path.join(REPO, T.CONFIG))["stages"][1])
    s.microbatch(8, 4)
    st = copy.deepcopy(s.state_dict())
    a = [s.microbatch(8, 4) for _ in range(5)]
    s2 = Sampler(18000, 1536, pools, 1)
    s2.load_state_dict(st)
    assert [s2.microbatch(8, 4) for _ in range(5)] == a
    assert s2.summary() == s.summary()


# ---------------------------------------------------------------- scoring

def test_soft_accuracy():
    refs = ["2"] * 8 + ["3"] * 2
    assert soft_accuracy("2", refs) == 1.0
    assert math.isclose(soft_accuracy("3", refs), 0.6)
    assert soft_accuracy("4", refs) == 0.0
    pair = {"answers": [["two"] * 8 + ["3"] * 2, ["1"] * 10], "consensus": ["2", "1"]}
    r = score_prediction("vqa_test", pair, 0, "Two.")
    assert r["prediction"] == "2" and r["accuracy"] == 1.0 and r["exact"] and not r["equals_partner"]


def test_consensus_answers_are_normalized():
    for name in ("vqa_dev", "vqa_test"):
        for p in TT.real(name)["pairs"]:
            assert all(normalize(c) == c for c in p["consensus"])


def test_pair_metrics():
    def row(pid, slot, pred, canon):
        return {"pair_id": pid, "slot": slot, "prediction": pred, "exact": pred == canon}
    rows = [row("a", 0, "yes", "yes"), row("a", 1, "no", "no"),       # both correct
            row("b", 1, "yes", "yes"), row("b", 0, "yes", "no"),      # identical answers, one wrong
            row("c", 0, "", "2"), row("c", 1, "3", "3")]
    m = pair_metrics(rows)
    assert m == {"n_pairs": 3, "pair_accuracy": 1 / 3, "identical_answer_rate": 1 / 3}
    with pytest.raises(AssertionError):
        pair_metrics(rows[:-1])


def test_bootstrap_pairing():
    rng = np.random.default_rng(0)
    b = rng.random(40)
    a = b + 0.5
    idx = bootstrap_indices(40, 4000, 6304)
    assert idx.shape == (4000, 40) and (idx == bootstrap_indices(40, 4000, 6304)).all()
    d = paired_difference(a, b, idx, [2.5, 97.5])
    assert math.isclose(d["estimate"], 0.5) and np.allclose(d["ci"], [0.5, 0.5])
    # unpaired resampling would not collapse: independent indices for the two conditions
    idx2 = bootstrap_indices(40, 4000, 1)
    unpaired = a[idx].mean(1) - b[idx2].mean(1)
    assert np.percentile(unpaired, 97.5) - np.percentile(unpaired, 2.5) > 0.05
    assert bootstrap_ci(b, idx, [2.5, 97.5])["ci"][0] < b.mean() < bootstrap_ci(b, idx, [2.5, 97.5])["ci"][1]


def test_summarize_uses_matched_pairs():
    real = T.load_config(os.path.join(REPO, T.CONFIG))
    rows = []
    for p in range(10):
        for s in (0, 1):
            acc = float((p + s) % 3 == 0)
            for c in real["final_eval"]["conditions"]:
                rows.append({"suite": "grounding_test", "condition": c, "pair_id": f"p{p}", "slot": s,
                             "type": "presence" if p < 5 else "counting", "true_count": 1 + p % 4,
                             "accuracy": acc, "exact": bool(acc), "prediction": "yes" if acc else "no",
                             "equals_partner": False, "empty_normalized": False, "empty_raw": False,
                             "reference_nll": 1.0 + p + (0.3 if c != "correct" else 0.0), "stopped": True})
    timing = {c: {"generation_s_per_answer": 0.1} for c in real["final_eval"]["conditions"]}
    out = summarize("grounding_test", rows, timing, real)
    for sub in ("all", "presence", "counting"):
        for o, a in out["bootstrap"][sub]["correct_advantage_over"].items():
            assert a["accuracy"]["ci"] == [0.0, 0.0]
            assert np.allclose(a["reference_nll"]["ci"], [0.3, 0.3]) and math.isclose(a["reference_nll"]["estimate"], 0.3)
    assert out["bootstrap"]["presence"]["n_pairs"] == 5
    assert set(out["conditions"]["swap"]["counting_by_true_count"]) == {"1", "2", "3", "4"}


# ---------------------------------------------------------------- decoding

def test_batched_left_padded_greedy_equals_single(enc, cfg):
    vlm = make_vlm(cfg, seed=3)
    vlm.connector.scale.fill_(0.05)
    vlm.eval()
    qs = ["Is there a dog?", "How many people can you see in this picture right now?", "What color is it?",
          "Does this image contain an umbrella?", "Why?"]
    imgs = [1, None, 2, 3, None]
    exs = [enc.example(q, None, i, "answer") for q, i in zip(qs, imgs)]
    feats = {1: feats_for(1, 1), 2: feats_for(1, 2), 3: feats_for(1, 3)}

    def run(batch_exs):
        b = D.collate(batch_exs, 196, 151643, "left")
        f = [feats[e["image"]] for e in batch_exs if e["image"] is not None]
        return vlm.greedy(b, torch.cat(f) if f else None, 16, 151645, 151643)

    batched = run(exs)
    single = [run([e])[0] for e in exs]
    assert [x["token_ids"] for x in batched] == [x["token_ids"] for x in single]
    assert all(len(x["token_ids"]) > 0 for x in single)


def test_greedy_matches_hf_generate_text_only(enc, cfg):
    vlm = make_vlm(cfg, seed=4)
    vlm.eval()
    e = enc.example("Does this image contain a dog?", None, None, "answer")
    b = D.collate([e], 196, 151643, "left")
    mine = vlm.greedy(b, None, 16, 151645, 151643)[0]
    out = vlm.lm.generate(input_ids=b["input_ids"], attention_mask=b["attention_mask"], do_sample=False,
                          max_new_tokens=16, eos_token_id=151645, pad_token_id=151643, temperature=None,
                          top_p=None, top_k=None)
    hf = out[0, b["input_ids"].shape[1]:].tolist()
    if 151645 in hf:
        hf = hf[:hf.index(151645)]
    assert mine["token_ids"] == hf


# ---------------------------------------------------------------- features and splits

def test_feature_store_checks_sha256(tmp_path):
    TT.write_features(str(tmp_path), "val", [5, 6, 7], 0, shard=2)
    st = D.FeatureStore(str(tmp_path), "val")
    assert st.get([7, 5]).shape == (2, 196, 768) and st.get([7]).dtype == torch.float16
    with open(tmp_path / "val_001.safetensors", "r+b") as f:
        f.seek(-4, 2)
        f.write(b"\x00\x01\x02\x03")
    with pytest.raises(AssertionError, match="sha256"):
        D.FeatureStore(str(tmp_path), "val")


def test_load_splits_isolation(tmp_path):
    data, _ = TT.build(tmp_path)
    cfg = {"data": data}
    d = D.load_splits(cfg)
    assert len(d["caption_examples"]) == 12
    vd = json.load(open(data["vqa_dev"]))
    vd["pairs"][0]["image_ids"][0] = d["images_train"]["images"][0]["image_id"]
    json.dump(vd, open(data["vqa_dev"], "w"))
    with pytest.raises(AssertionError):
        D.load_splits(cfg)


def test_real_splits_respect_caps_and_isolation(tok):
    real = T.load_config(os.path.join(REPO, T.CONFIG))
    d = D.load_splits(real)
    enc = D.Encoder(tok, real)
    for _, c in d["caption_examples"][:3000] + d["caption_dev_examples"]:
        enc.target(c, "caption")
    for k in ("vqa_train", "vqa_dev", "vqa_test", "grounding_dev", "grounding_test"):
        for p in d[k]["pairs"]:
            enc.post(p["question"])
            for s in (0, 1):
                enc.target(D.pair_target(k, p, s), "answer")
