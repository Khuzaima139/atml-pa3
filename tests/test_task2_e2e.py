"""Task 2 end to end on CPU through the real entry points (train, evaluate, status) with a tiny random Qwen3,
tiny split files and random cached features: dev evaluation, stage weights, resumable states, resume, smoke,
plan, final evaluation under four conditions. Every output file and key is checked."""
import json
import os

import numpy as np
import pytest
import torch

import task2_tiny as TT
from common.results import read_result
from task2_vlm import evaluate as E
from task2_vlm import model as M
from task2_vlm import status as S
from task2_vlm import train as T
from task2_vlm.data import sha256_file

STAGE_ENDS = (3, 6, 8)


@pytest.fixture(autouse=True)
def cpu_and_tiny_model(monkeypatch):
    monkeypatch.setattr(T, "training_device", lambda: torch.device("cpu"))
    monkeypatch.setattr(M, "load_qwen", lambda cfg, device: TT.tiny_lm())


def setup(tmp, sub):
    d = tmp / sub
    d.mkdir()
    data, feats = TT.build(d)
    cfg = TT.tiny_cfg(d, data, feats)
    return d, cfg, TT.write_cfg(d, cfg)


@pytest.fixture(scope="module")
def tinydir(tmp_path_factory):
    return tmp_path_factory.mktemp("t2e2e")


@pytest.fixture(scope="module")
def reference(tinydir):
    """One uninterrupted run (train then final evaluation)."""
    mp = pytest.MonkeyPatch()
    mp.setattr(T, "training_device", lambda: torch.device("cpu"))
    mp.setattr(M, "load_qwen", lambda cfg, device: TT.tiny_lm())
    d, cfg, p = setup(tinydir, "ref")
    T.main(["--config", p])
    E.main(["--config", p])
    mp.undo()
    return d, cfg, p


def load_tensors(path):
    from safetensors.torch import load_file
    return load_file(path)


# ---------------------------------------------------------------- training outputs

def test_train_json_keys(reference):
    d, cfg, _ = reference
    tr = read_result(os.path.join(cfg["output"]["result_dir"], "train.json"))
    assert tr["status"] == "done" and tr["kind"] == "train" and tr["config"] == cfg
    for k in ("run_id", "segments", "dtypes", "calibration", "data", "features", "lm_config", "resources", "stages",
              "dev", "files", "log_fields", "log"):
        assert tr[k] is not None, k
    seg = tr["segments"][0]
    assert set(seg["provenance"]["git"]) == {"commit", "dirty"} and "hardware" in seg["provenance"]
    assert {"torch", "torch_build", "transformers", "peft", "accelerate", "torchao", "cuda"} <= set(seg["versions"])
    r = tr["resources"]
    assert r["updates_done"] == r["updates_total"] == 8 and r["skipped_updates"] == 0
    for k in ("train_s", "dev_eval_s", "ckpt_io_s", "calibration_s", "wall_s", "peak_allocated_bytes",
              "peak_reserved_bytes", "setup_s"):
        assert k in r
    assert [s["status"] for s in tr["stages"]] == ["done"] * 3
    assert [s["trainable_params"] for s in tr["stages"]] == cfg["expected_trainable_params"]
    assert [s["last_update"] for s in tr["stages"]] == list(STAGE_ENDS)
    assert tr["stages"][1]["trainable_groups"] == {"connector": TT.CONNECTOR_PARAMS, "lora": TT.LORA_PARAMS}
    mix2, mix3 = tr["stages"][1]["mixture"], tr["stages"][2]["mixture"]
    assert set(mix2["draws"]) == {"presence", "counting", "caption"} and sum(mix2["draws"].values()) == 3 * 2 * 2
    assert set(mix3["draws"]) == {"vqa", "presence", "counting", "caption"}
    assert tr["stages"][0]["mixture"]["examples"] == {"caption": 24}
    assert len(tr["log"]) == 8
    for i, rec in enumerate(tr["log"], 1):
        assert rec["update"] == i and rec["loss"] is not None and rec["grad_norm_preclip"] is not None
        assert set(rec["lr"]) == ({"connector"} if i <= 3 else {"connector", "lora"})
        assert rec["loss_scale"] == 1.0 and rec["skipped"] is False and sum(rec["mixture"].values()) > 0
    for k, e in enumerate(STAGE_ENDS):
        st = cfg["stages"][k]
        for name, peak in st["lr"].items():
            assert np.isclose(tr["log"][e - 1]["lr"][name], 0.1 * peak)       # stage end: 10% of peak


def test_calibration_rule(reference):
    d, cfg, _ = reference
    tr = read_result(os.path.join(cfg["output"]["result_dir"], "train.json"))
    c = tr["calibration"]
    assert c["rule"] == cfg["calibration"]["rule"] and c["n_images"] == 4 and len(c["image_ids"]) == 4
    train_ids = {r["image_id"] for r in json.load(open(cfg["data"]["images_train"]))["images"]}
    assert set(c["image_ids"]) <= train_ids
    assert set(c["sources"]) == {"captions_train", "vqa_train_questions", "grounding_templates"}
    assert c["sources"]["grounding_templates"]["texts"] == 160
    assert c["sources"]["vqa_train_questions"]["texts"] == 12 and c["sources"]["captions_train"]["texts"] == 12
    assert np.isclose(c["s"], c["text_mean_l2"] / c["connector_mean_l2_at_init"])
    # recompute the connector side independently on the sampled features
    from task2_vlm.data import FeatureStore
    conn = M.build_connector(cfg)
    f = FeatureStore(cfg["features"]["dir"], "train").get(c["image_ids"]).float()
    with torch.no_grad():
        m = conn.raw(f).norm(dim=-1).mean().item()
    assert np.isclose(m, c["connector_mean_l2_at_init"], rtol=1e-5)
    w = load_tensors([e for e in tr["files"] if e["kind"] == "connector"][0]["path"])
    assert float(w["scale"]) == c["stored_s_float32"]


def test_dev_files(reference):
    d, cfg, _ = reference
    tr = read_result(os.path.join(cfg["output"]["result_dir"], "train.json"))
    assert [x["update"] for x in tr["dev"]] == cfg["dev_eval"]["updates"]
    assert [x["stage"] for x in tr["dev"]] == [1, 1, 1, 2, 3]
    for s in tr["dev"]:
        f = read_result(s["file"])
        assert f["run_id"] == tr["run_id"] and f["update"] == s["update"]
        rows = f["examples"]
        assert sum(r["set"] == "captions_dev" for r in rows) == 4
        for suite, n, types in (("vqa_dev", 8, None), ("grounding_dev", 8, {"presence", "counting"})):
            sr = [r for r in rows if r["set"] == suite]
            assert len(sr) == n
            for r in sr:
                assert set(r["nll"]) == {"correct", "complementary", "none"}
                assert np.isclose(r["dnll"], r["nll"]["complementary"] - r["nll"]["correct"])
            assert s[suite]["by_type"] and set(s[suite]["nll"]) == {"correct", "complementary", "none"}
            if types:
                assert set(s[suite]["by_type"]) == types
        assert np.isclose(s["caption_loss"], np.mean([r["loss"] for r in rows if r["set"] == "captions_dev"]))


def test_kept_weights_and_recovery(reference):
    d, cfg, _ = reference
    tr = read_result(os.path.join(cfg["output"]["result_dir"], "train.json"))
    kept = [e for e in tr["files"] if e["kind"] in ("connector", "lora")]
    assert [(e["kind"], e["stage"], e["update"]) for e in kept] == [
        ("connector", 1, 3), ("connector", 2, 6), ("lora", 2, 6), ("connector", 3, 8), ("lora", 3, 8)]
    for e in kept:
        assert sha256_file(e["path"]) == e["sha256"]
    lora = load_tensors(kept[-1]["path"])
    assert len(lora) == 2 * 4 * 2 and all(v.dtype == torch.float32 for v in lora.values())
    assert set(load_tensors(kept[0]["path"])) == {"fc1.weight", "fc1.bias", "fc2.weight", "fc2.bias", "scale"}
    rec = [e for e in tr["files"] if e["kind"] == "recovery"]
    assert [e["update"] for e in rec] == [2, 4, 6, 8]
    assert all("deleted_at_update" in e for e in rec[:-1])
    assert sorted(os.listdir(cfg["output"]["recovery_dir"])) == ["recovery_u0008.pt"]


# ---------------------------------------------------------------- resume

@pytest.mark.parametrize("stop", [4, 6])
def test_resume_reproduces_updates(reference, tinydir, stop):
    """Stop after `stop` (mid Stage 2 / Stage 2-3 boundary), resume from the saved state: every later update
    and the final weights equal the uninterrupted run exactly."""
    _, rcfg, _ = reference
    d, cfg, p = setup(tinydir, f"resume{stop}")
    T.main(["--config", p, "--stop_after", str(stop)])
    part = read_result(os.path.join(cfg["output"]["result_dir"], "train.json"))
    assert part["status"] == "stopped" and part["resources"]["updates_done"] == stop
    T.main(["--config", p, "--resume"])
    a = read_result(os.path.join(rcfg["output"]["result_dir"], "train.json"))
    b = read_result(os.path.join(cfg["output"]["result_dir"], "train.json"))
    assert b["status"] == "done" and b["run_id"] == part["run_id"] and len(b["segments"]) == 2
    keys = ("loss", "grad_norm_preclip", "lr", "mixture", "supervised_tokens", "padded_tokens")
    for x, y in zip(a["log"], b["log"]):
        assert {k: x[k] for k in keys} == {k: y[k] for k in keys}, x["update"]
    for x, y in zip(a["dev"], b["dev"]):
        assert x["caption_loss"] == y["caption_loss"] and x["vqa_dev"]["nll"] == y["vqa_dev"]["nll"]
    ka = {(e["kind"], e["stage"]): load_tensors(e["path"]) for e in a["files"] if e["kind"] in ("connector", "lora")}
    kb = {(e["kind"], e["stage"]): load_tensors(e["path"]) for e in b["files"] if e["kind"] in ("connector", "lora")}
    assert ka.keys() == kb.keys()
    for k in ka:
        assert all(torch.equal(ka[k][n], kb[k][n]) for n in ka[k]), k
    assert a["stages"][1]["mixture"] == b["stages"][1]["mixture"]


def test_resume_refuses_changed_config(reference, tinydir):
    d, cfg, p = setup(tinydir, "refuse")
    T.main(["--config", p, "--stop_after", "2"])
    cfg["optimizer"]["weight_decay"] = 0.02
    p2 = TT.write_cfg(d, cfg)
    with pytest.raises(SystemExit, match="config differs"):
        T.main(["--config", p2, "--resume"])


# ---------------------------------------------------------------- final evaluation

def test_final_eval_outputs(reference):
    d, cfg, _ = reference
    ev = read_result(os.path.join(cfg["output"]["result_dir"], "final/eval.json"))
    tr = read_result(os.path.join(cfg["output"]["result_dir"], "train.json"))
    assert ev["status"] == "done" and ev["train_run_id"] == tr["run_id"] and ev["kind"] == "eval"
    assert ev["endpoint"]["connector"]["update"] == 8 and ev["endpoint"]["lora"]["stage"] == 3
    for k in ("provenance", "versions", "dtypes", "generation", "timing", "suites", "wall_s", "setup_s",
              "peak_allocated_bytes", "examples_file", "calibration_scale"):
        assert k in ev, k
    g = ev["generation"]
    assert g["do_sample"] is False and g["temperature"] is None and g["top_p"] is None and g["top_k"] is None
    assert g["max_new_tokens"] == 16 and g["batch_size"] == 8 and g["stop_tokens"] == ["<|im_end|>"]
    assert g["enable_thinking"] is False
    conds = ["correct", "swap", "mismatched", "none"]
    for suite, n_pairs, types in (("vqa_test", 6, {"yes/no", "number", "other"}),
                                  ("grounding_test", 6, {"presence", "counting"})):
        s = ev["suites"][suite]
        assert s["n_pairs"] == n_pairs and list(s["conditions"]) == conds
        assert set(ev["timing"][suite]) == set(conds)
        for c, m in s["conditions"].items():
            for k in ("n", "accuracy", "exact_match", "n_pairs", "pair_accuracy", "identical_answer_rate",
                      "empty_rate_normalized", "empty_rate_raw", "follow_partner_rate", "mean_reference_nll",
                      "generation_s_per_answer", "by_type", "not_stopped"):
                assert k in m, (suite, c, k)
            assert m["n"] == 2 * n_pairs and set(m["by_type"]) == types
            if suite == "grounding_test":
                assert set(m["counting_by_true_count"]) == {"1", "2", "3", "4"}
        assert set(s["bootstrap"]) == {"all"} | types
        for sub, b in s["bootstrap"].items():
            assert b["resamples"] == 4000 and b["seed"] == 6304 and set(b["conditions"]) == set(conds)
            assert set(b["correct_advantage_over"]) == {"swap", "mismatched", "none"}
            for a in b["correct_advantage_over"].values():
                for k in ("accuracy", "reference_nll"):
                    lo, hi = a[k]["ci"]
                    assert lo <= hi
    assert sha256_file(ev["examples_file"]["path"]) == ev["examples_file"]["sha256"]


def test_final_eval_examples(reference):
    d, cfg, _ = reference
    ex = read_result(os.path.join(cfg["output"]["result_dir"], "final/examples.json"))
    rows = ex["rows"]
    assert len(rows) == (6 + 6) * 2 * 4
    mm = json.load(open(cfg["data"]["mismatch_map"]))["suites"]
    for r in rows:
        for k in ("suite", "condition", "pair_id", "slot", "type", "question", "image_id", "raw_generation",
                  "token_ids", "prediction", "canonical", "partner_canonical", "accuracy", "exact",
                  "equals_partner", "empty_normalized", "empty_raw", "reference_nll", "nll_target", "stopped"):
            assert k in r, k
        assert ("references" in r) == r["suite"].startswith("vqa") and len(r.get("references", [0] * 10)) == 10
        pi = r["pair_image_ids"]
        if r["condition"] == "correct":
            assert r["image_id"] == pi[r["slot"]]
        elif r["condition"] == "swap":
            assert r["image_id"] == pi[1 - r["slot"]]
        elif r["condition"] == "mismatched":
            assert r["image_id"] not in pi
            assert r["image_id"] == mm[r["suite"]]["map"][r["pair_id"]]["image_ids"][r["slot"]]
        else:
            assert r["image_id"] is None
        assert len(r["token_ids"]) <= 16
    # questions and references are fixed across conditions; NLL target is the training-style reference
    by = {}
    for r in rows:
        by.setdefault((r["pair_id"], r["slot"]), set()).add((r["question"], r["canonical"], r["nll_target"]))
    assert all(len(v) == 1 for v in by.values())


def test_eval_refuses_unfinished_run(tinydir):
    d, cfg, p = setup(tinydir, "unfinished")
    T.main(["--config", p, "--stop_after", "2"])
    with pytest.raises(AssertionError, match="stopped"):
        E.main(["--config", p])


def test_result_files_are_not_overwritten_by_another_run(reference):
    from common.results import ResultOverwriteError
    d, cfg, p = reference
    with pytest.raises(ResultOverwriteError):
        T.main(["--config", p])


# ---------------------------------------------------------------- smoke, plan, status

def test_smoke_plan_and_status(tinydir, capsys):
    d, cfg, p = setup(tinydir, "smoke")
    T.main(["--config", p, "--smoke"])
    E.main(["--config", p, "--smoke"])
    log = capsys.readouterr().out
    tr = read_result(os.path.join(cfg["output"]["smoke_result_dir"], "train.json"))
    ev = read_result(os.path.join(cfg["output"]["smoke_result_dir"], "final/eval.json"))
    assert tr["kind"] == "smoke" and tr["resources"]["updates_done"] == 6 and [x["update"] for x in tr["dev"]] == [0]
    sm = tr["smoke"]
    assert sm["main_stage_updates"] == [3, 3, 2] and sm["main_dev_points"] == 5 and sm["main_recovery_saves"] == 4
    assert set(sm["mean_s_per_update_excl_first"]) == {"1", "2", "3"} and len(sm["recovery_save_s"]) == 3
    assert ev["kind"] == "smoke_eval" and all(s["n_pairs"] == 2 for s in ev["suites"].values())
    assert os.listdir(cfg["output"]["smoke_keep_dir"]) and not os.path.exists(cfg["output"]["keep_dir"])
    logp = d / "smoke.log"
    logp.write_text(log)
    last = S.progress(str(logp))
    assert all(x in last for x in S.PATTERNS)
    with pytest.raises(SystemExit) as found:
        S.main(["--progress", str(logp)])
    assert found.value.code == 0
    cache = d / "cache.json"
    cache.write_text(json.dumps({"run_id": "c", "status": "done", "wall_s": 100.0, "download_s": 60.0}))
    args = ["--plan", "--smoke_dir", cfg["output"]["smoke_result_dir"], "--cache", str(cache), "--config", p]
    with pytest.raises(SystemExit) as ok:
        S.main(args + ["--limit_h", "10.5"])
    assert ok.value.code == 0
    with pytest.raises(SystemExit) as over:
        S.main(args + ["--limit_h", "0.0001"])
    assert over.value.code == 1
    out = capsys.readouterr().out
    assert "PLAN OK" in out and "STOP: projected total" in out and "projected session total" in out


def test_summary_runs(reference, capsys):
    d, cfg, _ = reference
    S.main(["--summary", "--run_dir", cfg["output"]["result_dir"]])
    out = capsys.readouterr().out
    assert "calibration s =" in out and "correct - swap" in out and "grounding_test (6 pairs)" in out



def test_lora_preflight(tinydir, capsys):
    from task2_vlm import preflight
    d, cfg, p = setup(tinydir, "preflight")
    assert preflight.main(["--config", p]) == TT.LORA_PARAMS
    assert "lora preflight OK: LoRA trainable 14,336" in capsys.readouterr().out
