"""CPU end-to-end tests of task3_ttc.live (real entry point, tiny random Qwen2 with the real Qwen2.5 tokenizer and
chat template), the resume logic, task3_ttc.live_status, and task3_ttc.live_analysis on those outputs."""
import csv
import json
import os
import subprocess
import sys

import pytest

import task3_tiny_lm as tiny
from common.runinfo import REPO_ROOT
from task3_ttc import live, live_problems
from task3_ttc.extract import last_boxed

pytestmark = pytest.mark.skipif(tiny.real_tokenizer() is None,
                                reason="Qwen2.5-3B-Instruct tokenizer not in the local HF cache")

CAP_SCALE = 0.01
N_Q = 2
RECORD_KEYS = {"run_id", "sample_id", "q", "strategy", "call", "seed", "max_new_tokens", "prompt_text",
               "input_tokens", "output_token_ids", "output_text", "output_tokens", "stop_token_id", "finished_by_eos",
               "truncated", "extracted_answer", "wall_s", "peak_mem_allocated_bytes", "peak_mem_reserved_bytes",
               "draft_call", "session", "timestamp"}
RUN_KEYS = {"run_id", "status", "config", "effective_generation_config", "logits_processors", "sessions", "totals",
            "updated", "wall_s_sessions"}
SESSION_KEYS = {"started", "provenance", "libraries", "argv", "records_at_start", "partial_bytes_removed", "model",
                "torch", "warmup", "elapsed_s"}


def _live(out, model, *extra, env=None):
    e = dict(os.environ, **(env or {}))
    return subprocess.run([sys.executable, "-m", "task3_ttc.live", "--out", out, "--questions", str(N_Q),
                           "--cap-scale", str(CAP_SCALE), "--model", model, "--device", "cpu", *extra],
                          cwd=REPO_ROOT, capture_output=True, text=True, env=e)


def _calls(out):
    with open(os.path.join(out, "calls.jsonl")) as f:
        return [json.loads(x) for x in f]


def _run_json(out):
    with open(os.path.join(out, "run.json")) as f:
        return json.load(f)


@pytest.fixture(scope="module")
def env(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("t3live")
    model = tiny.build(str(tmp / "tiny"), eos_scale=1.2)
    full = str(tmp / "full")
    p = _live(full, model)
    assert p.returncode == 0, p.stderr[-3000:]
    return {"tmp": tmp, "model": model, "full": full, "log": p.stdout}


def test_records_order_seeds_keys(env):
    recs = _calls(env["full"])
    assert len(recs) == N_Q * 5
    assert [(r["q"], r["call"], r["strategy"]) for r in recs] == \
        [(q, c, s) for q in range(N_Q) for c, s, _ in live.CALLS]
    caps = live.caps(CAP_SCALE)
    assert caps == {"single": 31, "p1": 10, "p2": 10, "p3": 10, "refine": 20}
    probs = live_problems.load()["problems"]
    for r in recs:
        assert set(r) == RECORD_KEYS
        assert r["seed"] == 6304 * 1000 + 10 * r["q"] + r["call"] != live.WARMUP_SEED
        assert r["sample_id"] == probs[r["q"]]["sample_id"]
        assert r["max_new_tokens"] == caps[r["strategy"]]
        assert len(r["output_token_ids"]) == r["output_tokens"] <= r["max_new_tokens"]
        assert r["extracted_answer"] == last_boxed(r["output_text"])
        assert r["wall_s"] > 0 and r["peak_mem_allocated_bytes"] is None   # CPU: no CUDA memory
        assert r["draft_call"] == (1 if r["strategy"] == "refine" else None)


def test_prompts_and_refine_draft(env):
    tok = tiny.real_tokenizer()
    probs = live_problems.load()["problems"]
    recs = {(r["q"], r["strategy"]): r for r in _calls(env["full"])}
    q0 = recs[(0, "single")]
    assert q0["input_tokens"] == 106                         # TASK3_FACTS item 12
    assert q0["prompt_text"].startswith("<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n")
    assert q0["prompt_text"].endswith("\n\nThink step by step inside <think> </think> tags, then give your final "
                                      "answer inside \\boxed{}.<|im_end|>\n<|im_start|>assistant\n")
    for q in range(N_Q):
        p = probs[q]["problem"]
        for s in ("single", "p1", "p2", "p3"):
            assert recs[(q, s)]["prompt_text"] == live.render(tok, live.build_messages(p))
        p1, ref = recs[(q, "p1")], recs[(q, "refine")]
        assert p1["output_text"] == tok.decode(p1["output_token_ids"], skip_special_tokens=True)
        assert ref["prompt_text"] == live.render(tok, live.build_messages(p, p1["output_text"]))
        assert ("<|im_start|>assistant\n" + p1["output_text"] + "<|im_end|>\n<|im_start|>user\n" + live.REVIEW
                + "<|im_end|>\n<|im_start|>assistant\n") in ref["prompt_text"]
        assert ref["input_tokens"] == len(tok(ref["prompt_text"], add_special_tokens=False)["input_ids"])
    assert recs[(0, "p1")]["truncated"]                      # a truncated draft is still appended


def test_truncation_flags(env):
    recs = _calls(env["full"])
    for r in recs:
        if r["finished_by_eos"]:
            assert r["stop_token_id"] in live.EOS_IDS and not r["truncated"]
        else:
            assert r["stop_token_id"] is None and r["truncated"] and r["output_tokens"] == r["max_new_tokens"]
        assert live.EOS_IDS[0] not in r["output_token_ids"] and live.EOS_IDS[1] not in r["output_token_ids"]
    assert any(r["truncated"] for r in recs) and any(r["finished_by_eos"] for r in recs)


def test_run_json_and_effective_sampling(env):
    with open(os.path.join(env["model"], "generation_config.json")) as f:
        assert json.load(f)["repetition_penalty"] == 1.05   # the checkpoint default that must not leak in
    run = _run_json(env["full"])
    assert set(run) == RUN_KEYS and run["status"] == "complete"
    assert len(run["sessions"]) == 1 and set(run["sessions"][0]) == SESSION_KEYS
    s = run["sessions"][0]
    assert {"git", "hardware"} <= set(s["provenance"]) and s["libraries"]["transformers"]
    assert s["model"]["param_dtypes"] == ["torch.float16"] and s["model"]["load_s"] > 0
    assert s["torch"]["device"] == "cpu" and s["warmup"]["seed"] == live.WARMUP_SEED
    for strat, cap in live.caps(CAP_SCALE).items():
        e = run["effective_generation_config"][strat]
        assert e["repetition_penalty"] == 1.0
        assert {k: e[k] for k in live.SAMPLING} == live.SAMPLING
        assert e["eos_token_id"] == [151645, 151643] and e["pad_token_id"] == 151643 and e["max_new_tokens"] == cap
    assert sorted(run["logits_processors"]) == sorted(live.EXPECTED_PROCESSORS)
    assert "RepetitionPenaltyLogitsProcessor" not in run["logits_processors"]
    t = run["totals"]
    assert t["calls_done"] == t["calls_expected"] == 10 and t["questions_done"] == 2
    assert sum(d["calls"] for d in t["by_strategy"].values()) == 10
    assert run["config"]["sampling"] == live.SAMPLING and run["config"]["n_questions"] == N_Q
    assert "warm-up" in env["log"] and "complete: 10/10 calls" in env["log"]


def test_repetition_penalty_processor_never_built(env, monkeypatch):
    """In process: generate() through the live path never constructs a RepetitionPenaltyLogitsProcessor even though
    the checkpoint's generation_config.json says 1.05."""
    import torch
    import transformers.generation.utils as gu
    built = []
    orig = gu.RepetitionPenaltyLogitsProcessor

    class Spy(orig):
        def __init__(self, *a, **k):
            built.append((a, k))
            super().__init__(*a, **k)
    monkeypatch.setattr(gu, "RepetitionPenaltyLogitsProcessor", Spy)
    model, tok, _ = live.load_model(env["model"], torch.device("cpu"))
    model.generation_config = live.generation_config(1)
    spy = live.ProcessorSpy(model)
    live.generate(model, tok, torch.device("cpu"), live.build_messages("1+1?"), 5, 1, spy)
    assert built == [] and "RepetitionPenaltyLogitsProcessor" not in spy.last
    # control: the checkpoint's own config would have built it
    from transformers import GenerationConfig
    model.generation_config = GenerationConfig.from_pretrained(env["model"])
    ids = tok("hi", return_tensors="pt")["input_ids"]
    model.generate(input_ids=ids, attention_mask=torch.ones_like(ids), max_new_tokens=2)
    assert len(built) == 1


def _same_outputs(a, b):
    key = ["q", "call", "strategy", "seed", "prompt_text", "input_tokens", "output_token_ids", "output_text",
           "finished_by_eos", "truncated", "extracted_answer"]
    return [{k: r[k] for k in key} for r in a] == [{k: r[k] for k in key} for r in b]


def test_resume_after_kill_first_question(env):
    out = str(env["tmp"] / "resume5")
    p = _live(out, env["model"], env={live.KILL_ENV: "5"})
    assert p.returncode == -9, (p.returncode, p.stderr[-2000:])
    assert len(_calls(out)) == 5 and _run_json(out)["status"] == "running"
    p = _live(out, env["model"])
    assert p.returncode == 0, p.stderr[-3000:]
    recs = _calls(out)
    assert len(recs) == 10 and len({(r["q"], r["call"]) for r in recs}) == 10
    run = _run_json(out)
    assert run["status"] == "complete" and len(run["sessions"]) == 2
    assert run["sessions"][1]["records_at_start"] == 5 and [r["session"] for r in recs] == [0] * 5 + [1] * 5
    assert "5 calls already present" in p.stdout
    assert _same_outputs(recs, _calls(env["full"]))        # per-call seeds: same outputs as the uninterrupted run
    p = _live(out, env["model"])                             # a third start has nothing to do
    assert p.returncode == 0 and len(_calls(out)) == 10 and "warm-up" not in p.stdout


def test_resume_mid_question_with_partial_line(env):
    out = str(env["tmp"] / "resume7")
    p = _live(out, env["model"], env={live.KILL_ENV: "7"})   # q1 single and p1 written, refine not yet
    assert p.returncode == -9
    with open(os.path.join(out, "calls.jsonl"), "a") as f:
        f.write('{"run_id": "partial", "q": 1, "ca')         # a line cut by the kill
    p = _live(out, env["model"])
    assert p.returncode == 0, p.stderr[-3000:]
    recs = _calls(out)
    assert len(recs) == 10 and len({(r["q"], r["call"]) for r in recs}) == 10
    run = _run_json(out)
    assert run["sessions"][1]["partial_bytes_removed"] == len('{"run_id": "partial", "q": 1, "ca')
    by = {(r["q"], r["strategy"]): r for r in recs}
    assert by[(1, "p1")]["session"] == 0 and by[(1, "refine")]["session"] == 1
    assert by[(1, "p1")]["output_text"] in by[(1, "refine")]["prompt_text"]
    assert _same_outputs(recs, _calls(env["full"]))


def test_refuses_different_config(env):
    p = subprocess.run([sys.executable, "-m", "task3_ttc.live", "--out", env["full"], "--questions", str(N_Q),
                        "--cap-scale", "0.02", "--model", env["model"], "--device", "cpu"],
                       cwd=REPO_ROOT, capture_output=True, text=True)
    assert p.returncode != 0 and "different config" in p.stderr
    assert len(_calls(env["full"])) == 10


def test_status_plan_summary(env):
    tmp = env["tmp"]
    smoke = str(tmp / "smoke")
    p = subprocess.run([sys.executable, "-m", "task3_ttc.live", "--out", smoke, "--questions", "1", "--cap-scale",
                        str(CAP_SCALE), "--model", env["model"], "--device", "cpu"], cwd=REPO_ROOT,
                       capture_output=True, text=True)
    assert p.returncode == 0, p.stderr[-2000:]
    pre = tmp / "pre"
    pre.mkdir()
    (pre / "preflight.json").write_text(json.dumps({"forwards": [{"prefill_tokens_per_s": 2000.0},
                                                                 {"prefill_tokens_per_s": 1500.0}]}))
    run = lambda *a: subprocess.run([sys.executable, "-m", "task3_ttc.live_status", *a], cwd=REPO_ROOT,
                                    capture_output=True, text=True)
    p = run("--plan", "--smoke", smoke, "--preflight", str(pre), "--limit-h", "10.5")
    assert p.returncode == 0 and "plan OK" in p.stdout and "161,335 output tokens" in p.stdout, p.stdout + p.stderr
    p = run("--plan", "--smoke", smoke, "--preflight", str(pre), "--limit-h", "0")
    assert p.returncode == 1 and "STOP" in p.stdout
    p = run("--summary", "--run", env["full"])
    assert p.returncode == 0 and "calls 10/10" in p.stdout and "repetition_penalty': 1.0" in p.stdout, p.stdout
    log, done = tmp / "x.log", tmp / "x.done"
    log.write_text("a\ncall q=1 last line\n")
    done.write_text("0\n")
    pid = tmp / "x.pid"
    pid.write_text(f"{os.getpid()}\n")
    p = run("--wait", "--run", env["full"], "--log", str(log), "--done", str(done), "--pid", str(pid),
            "--exit", str(tmp / "x.exit"), "--every", "0", "--poll", "0")
    assert p.returncode == 0 and "questions 2/50, calls 10/250" in p.stdout and "call q=1 last line" in p.stdout


def test_live_preflight_cpu(env):
    out = str(env["tmp"] / "preflight")
    p = subprocess.run([sys.executable, "-m", "task3_ttc.live", "--preflight", "--out", out, "--model", env["model"],
                        "--device", "cpu", "--long-tokens", "300"], cwd=REPO_ROOT, capture_output=True, text=True)
    assert p.returncode == 0, p.stderr[-3000:]
    with open(os.path.join(out, "preflight.json")) as f:
        d = json.load(f)
    assert d["status"] == "ok" and [x["name"] for x in d["forwards"]] == ["first_live_prompt", "long_refine_prompt"]
    assert all(x["all_finite"] for x in d["forwards"]) and d["forwards"][1]["input_tokens"] > 106 + 300
    assert d["forwards"][0]["input_tokens"] == 106 and d["generation"]["output_tokens"] <= 64
    assert d["effective_generation_config"]["repetition_penalty"] == 1.0


# ------------------------------------------------------------------------------------------------ analysis

ANALYSIS_KEYS = {"run_id", "provenance", "config", "inputs", "scorer_version", "scorer_timeouts", "definitions",
                 "run_settings", "strategies", "calls", "refine_transitions", "bootstrap"}
STRATEGY_KEYS = {"n_questions", "accuracy", "n_correct", "answer_calls", "cost_calls", "extraction_rate",
                 "truncation_rate", "n_responses", "input_tokens_per_question", "output_tokens_per_question",
                 "total_tokens_per_question", "input_tokens_total", "output_tokens_total", "runtime_s_per_question",
                 "runtime_s_total", "peak_mem_allocated_bytes", "peak_mem_reserved_bytes"}


def test_live_analysis_cli(env):
    tmp = env["tmp"]
    ids = [p["sample_id"] for p in live_problems.load()["problems"][:N_Q]]
    gold = tmp / "gold.json"
    gold.write_text(json.dumps({ids[0]: "14", ids[1]: "\\frac{1}{2}"}))
    out_dir = str(tmp / "analysis")
    cmd = [sys.executable, "-m", "task3_ttc.live_analysis", "--run", env["full"], "--out-dir", out_dir,
           "--gold", str(gold)]
    p = subprocess.run(cmd, cwd=REPO_ROOT, capture_output=True, text=True)
    assert p.returncode == 0, p.stderr[-3000:]
    with open(os.path.join(out_dir, "analysis.json")) as f:
        a = json.load(f)
    assert set(a) == ANALYSIS_KEYS
    assert set(a["strategies"]) == {"single", "parallel", "refine", "initial"}
    for s, m in a["strategies"].items():
        assert STRATEGY_KEYS <= set(m) and m["n_questions"] == N_Q
    assert "selected_no_answer_rate" in a["strategies"]["parallel"]
    assert a["strategies"]["parallel"]["n_responses"] == 3 * N_Q
    recs = _calls(env["full"])
    tot = lambda strats: sum(r["output_tokens"] for r in recs if r["strategy"] in strats)
    assert a["strategies"]["refine"]["output_tokens_total"] == tot({"p1", "refine"})
    assert a["strategies"]["parallel"]["output_tokens_total"] == tot({"p1", "p2", "p3"})
    assert a["strategies"]["single"]["output_tokens_total"] == tot({"single"})
    assert abs(a["strategies"]["refine"]["runtime_s_total"]
               - sum(r["wall_s"] for r in recs if r["strategy"] in ("p1", "refine"))) < 1e-9
    assert set(a["calls"]) == {"single", "p1", "p2", "p3", "refine"}
    t = a["refine_transitions"]
    assert t["wrong_to_correct"] + t["correct_to_wrong"] + t["both_correct"] + t["both_wrong"] == N_Q
    assert set(a["bootstrap"]) == {"parallel-single", "refine-single", "refine-parallel", "single-initial",
                                   "refine-initial"}
    assert all(b["resamples"] == 1000 and b["seed"] == 6304 and b["n"] == N_Q for b in a["bootstrap"].values())
    assert a["inputs"]["gold"]["source"] == str(gold) and a["scorer_version"]["id"]
    with open(os.path.join(out_dir, "per_question.csv")) as f:
        rows = list(csv.DictReader(f))
    assert [r["sample_id"] for r in rows] == ids
    for col in ("single_answer", "parallel_correct", "refine_output_tokens", "initial_wall_s", "parallel_groups",
                "parallel_selected_call", "p1_truncated", "refine_extracted", "refine_answer_changed", "gold"):
        assert col in rows[0]
    p = subprocess.run(cmd, cwd=REPO_ROOT, capture_output=True, text=True)
    assert p.returncode != 0 and "--replace" in p.stderr
    p = subprocess.run(cmd + ["--replace"], cwd=REPO_ROOT, capture_output=True, text=True)
    assert p.returncode == 0, p.stderr[-2000:]


def _synthetic(answers, truncated=None):
    """run dict + records with chosen extracted answers per question: answers[q] = {strategy: str | None}."""
    n = len(answers)
    probs = live_problems.load()["problems"][:n]
    run = {"run_id": "r", "config": {"n_questions": n, "sample_ids": [p["sample_id"] for p in probs]}}
    recs = []
    for q, ans in enumerate(answers):
        for c, s, _ in live.CALLS:
            recs.append({"run_id": "r", "q": q, "call": c, "strategy": s, "seed": live.call_seed(q, c),
                         "sample_id": probs[q]["sample_id"], "extracted_answer": ans[s],
                         "truncated": bool(truncated and truncated[q].get(s)), "finished_by_eos": True,
                         "input_tokens": 100 + c, "output_tokens": 10 * (c + 1), "wall_s": 1.0 + c,
                         "peak_mem_allocated_bytes": 1000 + c, "peak_mem_reserved_bytes": 2000 + c})
    return run, recs, {p["sample_id"]: g for p, g in zip(probs, ["14", "\\frac{1}{2}", "3", "7"][:n])}


def test_analyse_rules():
    from task3_ttc import live_analysis as LA
    from task3_ttc.answers import MathScorer
    answers = [
        # q0 gold 14: all parallel distinct -> p1 (wrong); refine fixes it (wrong -> correct)
        {"single": "14", "p1": "13", "p2": "12", "p3": "14", "refine": "14"},
        # q1 gold 1/2: p2 and p3 equivalent (0.5 and \frac{1}{2}) -> parallel correct via grouping; refine breaks it
        {"single": None, "p1": "\\frac{1}{2}", "p2": "0.5", "p3": "2", "refine": "1"},
        # q2 gold 3: NO_ANSWER majority wins the vote -> parallel wrong; refine keeps a correct draft
        {"single": "3", "p1": "3", "p2": None, "p3": None, "refine": "3.0"},
        # q3 gold 7: everything wrong; refine keeps the same wrong answer
        {"single": None, "p1": "8", "p2": "8", "p3": "7", "refine": "8"},
    ]
    run, recs, gold = _synthetic(answers, truncated=[{}, {"p2": True}, {"p2": True, "p3": True}, {}])
    res, rows = LA.analyse(run, recs, gold, MathScorer())
    acc = {s: [r[f"{s}_correct"] for r in rows] for s in LA.STRATEGIES}
    assert acc == {"single": [1, 0, 1, 0], "parallel": [0, 1, 0, 0], "refine": [1, 0, 1, 0], "initial": [0, 1, 1, 0]}
    assert [r["parallel_selected_call"] for r in rows] == ["p1", "p1", "p2", "p1"]
    assert rows[2]["parallel_answer"] is None
    t = res["refine_transitions"]
    assert (t["wrong_to_correct"], t["correct_to_wrong"], t["both_correct"], t["both_wrong"]) == (1, 1, 1, 1)
    assert t["answer_changed"] == 2                          # q0 13->14, q1 1/2->1; q2 3 vs 3.0 and q3 8 vs 8 same
    s = res["strategies"]
    assert s["parallel"]["selected_no_answer_rate"] == 0.25
    assert s["parallel"]["extraction_rate"] == 10 / 12 and s["parallel"]["truncation_rate"] == 3 / 12
    assert s["single"]["extraction_rate"] == 0.5
    assert s["refine"]["output_tokens_per_question"] == 20 + 50 and s["parallel"]["output_tokens_per_question"] == 90
    assert s["refine"]["runtime_s_per_question"] == 2.0 + 5.0 and s["parallel"]["input_tokens_per_question"] == 306
    assert s["refine"]["peak_mem_allocated_bytes"] == 1004
    b = res["bootstrap"]["refine-initial"]
    assert b["mean"] == 0.0 and b["n"] == 4


def test_analyse_rejects_incomplete():
    from task3_ttc import live_analysis as LA
    from task3_ttc.answers import MathScorer
    run, recs, gold = _synthetic([{"single": "1", "p1": "1", "p2": "1", "p3": "1", "refine": "1"}])
    with pytest.raises(AssertionError):
        LA.analyse(run, recs[:-1], gold, MathScorer())
    with pytest.raises(AssertionError):
        LA.analyse(run, recs + recs[-1:], gold, MathScorer())


def test_resolve_model_snapshot_commit(monkeypatch, tmp_path):
    """Regression: transformers 5.x leaves config._commit_hash None, so the pinned revision is checked through the
    resolved snapshot folder instead (the first version asserted on _commit_hash and would fail on Kaggle)."""
    import huggingface_hub
    calls = []

    def fake(repo, revision=None, **k):
        calls.append((repo, revision))
        return str(tmp_path / "snapshots" / live.REVISION)
    monkeypatch.setattr(huggingface_hub, "snapshot_download", fake)
    assert live.resolve_model(live.MODEL_ID) == (str(tmp_path / "snapshots" / live.REVISION), live.REVISION)
    assert calls == [(live.MODEL_ID, live.REVISION)]
    monkeypatch.setattr(huggingface_hub, "snapshot_download", lambda repo, revision=None, **k: str(tmp_path / "main"))
    with pytest.raises(RuntimeError, match="expected"):
        live.resolve_model(live.MODEL_ID)
    assert live.resolve_model("/some/local/dir") == ("/some/local/dir", None)


def test_resolve_model_real_cache():
    """With the real snapshot in the local HF cache (config only, no weights loaded), the commit matches."""
    import huggingface_hub
    try:
        path = huggingface_hub.snapshot_download(live.MODEL_ID, revision=live.REVISION, local_files_only=True)
    except Exception:
        pytest.skip("model snapshot not cached")
    assert os.path.basename(path) == live.REVISION and os.path.exists(os.path.join(path, "config.json"))
