"""Task 1 analysis tests: toy statistics, then the real committed result files."""
import copy
import csv
import json
import os
import shutil
import subprocess

import numpy as np
import pytest

from task1_finetune import analyze as A

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@pytest.fixture(scope="module", autouse=True)
def repo_root():
    old = os.getcwd()
    os.chdir(REPO)
    yield
    os.chdir(old)


@pytest.fixture(scope="module")
def run(tmp_path_factory):
    d = tmp_path_factory.mktemp("analysis")
    t, text, rd = A.analyze(out=str(d / "out"), figures=str(d / "figures"), sheet=str(d / "sheet.md"), annotations=str(d / "ann.csv"))
    return d, t, text, rd


@pytest.fixture(scope="module")
def ev():
    rd = A.Reader()
    return A.load_eval(rd, "results/task1", rd.yaml("configs/task1/eval.yaml")["conditions"])


def tracked():
    return set(subprocess.run(["git", "ls-files"], capture_output=True, text=True, check=True).stdout.split("\n"))


# ---------------------------------------------------------------- bootstrap

def test_bootstrap_all_flipped_has_no_spread():
    r = A.paired_bootstrap(np.ones(50), np.zeros(50), n_boot=200)
    assert (r["diff_pp"], r["ci_low_pp"], r["ci_high_pp"]) == (100.0, 100.0, 100.0)
    assert (r["n_a_only_correct"], r["n_b_only_correct"]) == (50, 0)


def test_bootstrap_identical_flags_is_zero():
    x = np.array([0, 1] * 30)
    r = A.paired_bootstrap(x, x, n_boot=200)
    assert (r["diff_pp"], r["ci_low_pp"], r["ci_high_pp"]) == (0.0, 0.0, 0.0)


def test_bootstrap_matches_binomial_interval():
    # 100 of 1,000 items flip one way: the difference is a binomial mean, 10 +- 1.96 sqrt(0.09 / 1000) = 10 +- 1.859 pp
    a, b = np.zeros(1000), np.zeros(1000)
    a[:100] = 1
    r = A.paired_bootstrap(a, b)
    assert r["diff_pp"] == pytest.approx(10.0)
    assert r["ci_low_pp"] == pytest.approx(10 - 1.859, abs=0.15)
    assert r["ci_high_pp"] == pytest.approx(10 + 1.859, abs=0.15)


def test_bootstrap_pairs_items():
    # correlated flags: every item equal in a and b except one, so the pairing removes all the spread of a and b
    a = np.array([0, 1] * 500)
    b = a.copy()
    b[0] = 1
    r = A.paired_bootstrap(a, b)
    assert r["ci_high_pp"] - r["ci_low_pp"] < 1.0
    assert (r["n_a_only_correct"], r["n_b_only_correct"]) == (0, 1)


def test_bootstrap_is_reproducible():
    a = np.array([1, 0, 1, 1, 0, 0, 1, 0] * 20)
    b = a[::-1]
    assert A.paired_bootstrap(a, b, n_boot=300) == A.paired_bootstrap(a, b, n_boot=300)


# ---------------------------------------------------------------- scores and extraction

def test_exact_match_norm_follows_stock_regexes():
    assert A.exact_match_norm("$150,000.") == "150000"
    assert A.exact_match_norm("$$") == ""
    assert A.exact_match_norm("working\n#### 18") == "18"


def test_scores_reproduced_from_per_example_files(ev):
    out = A.check_scores(ev)
    assert set(out) == set(ev)
    assert all(rec[b]["n"] == n for rec in out.values() for b, n in zip(A.BENCHMARKS, (1319, 10042, 3270, 2376)))


@pytest.mark.parametrize("benchmark", A.BENCHMARKS)
def test_score_check_catches_a_flipped_flag(ev, benchmark):
    bad = {"baseline": copy.deepcopy(ev["baseline"])}
    bad["baseline"][benchmark][0]["correct"] = 1 - bad["baseline"][benchmark][0]["correct"]
    with pytest.raises(AssertionError):
        A.check_scores(bad)


def test_last_number_and_no_digit_extraction():
    assert A.last_number("so \\boxed{1,200}.\n$$") == "1200"
    assert A.last_number("none") is None
    row = {"response": "$$\n\\boxed{18}\n$$", "flexible_answer": "$$", "gold": "18"}
    assert A.no_digit_extraction(row) and A.gold_is_last_number(row)
    assert not A.no_digit_extraction({"flexible_answer": "$18"})
    assert not A.no_digit_extraction({"flexible_answer": A.NO_EXTRACTION})


@pytest.mark.parametrize("response,answer", [("$$\\boxed{18}$$", "18"), ("The answer is: 18", "18"), ("18.", "18"), ("18 ,,", "18"),
                                             ("so\n$$\n\\boxed{1,200}\n$$", "1200"), ("7 then $18.\n$$", "18")])
def test_adjusted_rule_on_toy_strings(response, answer):
    assert A.exact_match_norm(A.adjusted_answer(response)) == answer


@pytest.mark.parametrize("response", [",,", "$$ no numbers $$", ".. ,, $$", "none"])
def test_adjusted_rule_ignores_matches_without_digits(response):
    assert A.adjusted_answer(response) == A.NO_EXTRACTION


def test_adjusted_answer_never_sees_the_gold():
    # two rows differing only in gold get the same answer, so only the flag differs
    rows = [{"response": "\\boxed{18}\n$$", "gold": g} for g in ("18", "19")]
    assert A.adjusted_flags(rows).tolist() == [1, 0]


def test_extraction_changed():
    assert A.extraction_changed({"doc_id": 0, "response": "\\boxed{18}\n$$", "flexible_answer": "$$"})
    assert not A.extraction_changed({"doc_id": 0, "response": "paid $18", "flexible_answer": "$18"})
    with pytest.raises(AssertionError):      # a stock answer with a digit that the rule does not reproduce
        A.extraction_changed({"doc_id": 0, "response": "paid $18", "flexible_answer": "$19"})


def test_numeric_equal_only():
    assert A.numeric_equal_only({"response": "$16.00", "gold": "16"})
    assert not A.numeric_equal_only({"response": "$16", "gold": "16"})      # already correct
    assert not A.numeric_equal_only({"response": "17", "gold": "16"})
    assert not A.numeric_equal_only({"response": "none", "gold": "16"})


def test_adjusted_scores_on_committed_files(run):
    _, t, *_ = run
    for r in t["adjusted"]:
        assert r["adjusted_correct"] == r["stock_correct"] + r["flips_to_correct"]
        assert r["flips_to_correct"] <= r["extraction_changed"]
        assert r["adjusted_acc"] == r["adjusted_correct"] / 1319
        assert r["stock_change_pp"] == pytest.approx(r["adjusted_change_pp"] + r["extraction_only_change_pp"], abs=1e-12)
    for r, e in zip(t["adjusted"], t["extraction"]):
        assert r["extraction_changed"] == e["wrong_no_digit_extraction"]
    assert t["adjusted"][0]["flips_to_correct"] > 0
    assert all(r["extraction_changed"] == 0 for r in t["adjusted"][1:])
    assert t["benchmarks"][0]["gsm8k"]["primary"] == t["adjusted"][0]["stock_acc"]      # stock primary untouched


def test_adjusted_bootstrap_covers_the_eleven_comparisons(run):
    _, t, *_ = run
    assert [(r["a"], r["b"]) for r in t["adjusted_comparisons"]] == A.COMPARISONS
    assert len(A.COMPARISONS) == 11
    stock = {(r["a"], r["b"]): r for r in t["comparisons"] if r["benchmark"] == "gsm8k"}
    for r in t["adjusted_comparisons"]:
        if r["b"] != "baseline":      # the rule changes the baseline only, so these equal the stock bootstrap
            assert r == stock[(r["a"], r["b"])]


def test_summary_labels_adjusted_as_secondary(run):
    *_, text, _ = run
    assert "gsm8k adjusted % (secondary)" in text and "stock % (primary)" in text


def test_extraction_counts_add_up(run):
    _, t, _, _ = run
    for r in t["extraction"]:
        assert r["correct"] + r["wrong"] == r["n"] == 1319
        assert r["wrong_extracted"] + r["wrong_unextractable"] == r["wrong"]
        assert r["truncated_wrong"] <= r["truncated"]
    for r in t["gain_loss"]:
        c = next(x for x in t["extraction"] if x["condition"] == r["condition"])
        assert t["extraction"][0]["correct"] + r["gained"] - r["lost"] == c["correct"]


def test_boolq_balance_on_toy_rows():
    rows = [{"gold": 1, "correct": 1}, {"gold": 1, "correct": 0}, {"gold": 0, "correct": 1}, {"gold": 0, "correct": 0}]
    r = A.boolq_balance({"c": {"boolq": rows}})[0]
    assert (r["gold_yes_share"], r["pred_yes_share"], r["acc_gold_yes"], r["acc_gold_no"]) == (0.5, 0.5, 0.5, 0.5)
    r = A.boolq_balance({"c": {"boolq": rows[:2] + rows[2:3]}})[0]
    assert r["pred_yes_share"] == pytest.approx(1 / 3)


def test_moving_average_window():
    assert A.moving_average(np.arange(30.0)).tolist() == [i + 12.0 for i in range(6)]


# ---------------------------------------------------------------- qualitative

@pytest.mark.parametrize("text,ok", [('{"a": 1}', True), ('```json\n{"a": 1}\n```', True), ('```\n{"a": 1}\n```\n', True),
                                     ('[1, 2]', False), ('```json\n{"a": 1', False), ('```json\n{"a": 1}\n```\nmore', False),
                                     ('```json\n```json\n{"a": 1}\n```\n```', False), ('not json', False)])
def test_json_object_check(text, ok):
    assert A.is_json_object(text) is ok


def test_annotation_csv_has_42_rows(run):
    d, *_ = run
    with open(d / "ann.csv", newline="") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 42
    assert tuple(rows[0]) == A.ANNOTATION_COLUMNS
    assert {r["question"] for r in rows} == set(A.ANNOTATION_QUESTIONS)
    assert len({(r["question"], r["format"], r["condition"]) for r in rows}) == 42
    assert len({r["id"] for r in rows}) == 42
    for r in rows:
        assert not any(r[k] for k in ("coherence", "repetition", "math_framing", "inappropriate", "excerpt", "notes"))
        assert r["truncated"] in ("y", "n")
        assert r["format_compliance"] in {"plain": ("n/a",), "template": ("",), "json": ("yes", "no")}[r["format"]]


def test_annotation_sheet_blocks(run):
    d, *_ = run
    text = (d / "sheet.md").read_text()
    assert text.count("\n## ") == 6
    assert text.count("\n### ") == 42
    assert text.count("Prompt:") == 6
    first = text.index("### baseline")
    order = [text.index(f"### {c}", first) for c in ("baseline", "full_ft_u0625", "full_ft_u1250", "l2sp_u1250", "lora4_u1250", "lora16_u1250", "oplora16_u1250")]
    assert order == sorted(order)


def test_filled_annotations_are_never_replaced(tmp_path):
    p = tmp_path / "a.csv"
    p.write_text("student,work\n")
    assert A.write_annotations(str(p), []) is False
    assert p.read_text() == "student,work\n"


def test_qualitative_counts(run):
    _, t, *_ = run
    assert sum(r["responses"] for r in t["qualitative"]) == 168
    assert all(r["json_prompts"] == 8 for r in t["qualitative"])


def test_summary_tables_are_well_formed(run):
    # a cell holding "||" once split a table row into extra columns
    *_, text, _ = run
    width = None
    for line in text.splitlines():
        if not line.startswith("|"):
            width = None
            continue
        cells = line.count("|")
        assert width in (None, cells), line
        width = cells


# ---------------------------------------------------------------- reproducibility

def test_every_figure_regenerates(run):
    d, *_ = run
    for name in A.FIGURES:
        assert (d / "figures" / name).stat().st_size > 10000


def test_panel_titles_fit_the_figure(monkeypatch, tmp_path):
    # a long title on the narrow L2-SP panel was cut off at the figure edge
    rd = A.Reader()
    train = {m: rd.json(f"results/task1/train_{m}.json") for m in A.METHODS}
    seen = []
    monkeypatch.setattr(A, "_save", lambda fig, folder, name: seen.append(fig))
    A.figure_learning(train, str(tmp_path))
    fig = seen[0]
    fig.canvas.draw()
    width = fig.bbox.width
    for ax in fig.axes:
        assert ax._left_title.get_window_extent().x1 <= width, ax._left_title.get_text()


def test_benchmark_titles_do_not_overlap(run, monkeypatch, tmp_path):
    # the GSM8K title with the adjusted score once ran into the HellaSwag title
    _, t, *_ = run
    seen = []
    monkeypatch.setattr(A, "_save", lambda fig, folder, name: seen.append(fig))
    A.figure_benchmarks(t["benchmarks"], t["comparisons"], (t["adjusted"], t["adjusted_comparisons"]), str(tmp_path))
    seen[0].canvas.draw()
    spans = [ax._left_title.get_window_extent() for ax in seen[0].axes]
    assert all(a.x1 < b.x0 for a, b in zip(spans, spans[1:]))
    assert len(seen[0].axes) == 5
    assert "secondary" in seen[0].axes[1]._left_title.get_text()
    assert seen[0].axes[0].get_xlim() == seen[0].axes[1].get_xlim()


def test_inputs_are_committed_results_only(run):
    _, _, _, rd = run
    missing = set(rd.read) - tracked()
    assert not missing, missing
    assert not [p for p in rd.read if p.startswith(("outputs/", "scratch/", "notes/"))]


def test_analysis_json_matches_committed_outputs(run):
    d, *_ = run
    committed = [f for f in os.listdir("results/task1/analysis") if f.endswith(".json")] if os.path.isdir("results/task1/analysis") else []
    if f"results/task1/analysis/{committed[0] if committed else ''}" not in tracked():
        pytest.skip("analysis outputs not committed yet")
    for name in committed:
        with open(f"results/task1/analysis/{name}") as f:
            old = json.load(f)
        with open(d / "out" / name) as f:
            new = json.load(f)
        for doc in (old, new):
            doc.pop("provenance")
            doc.pop("figures", None)
        assert old == new, name
    assert (d / "out" / "summary.md").read_text() == open("results/task1/analysis/summary.md").read()


def test_provenance_is_taken_before_outputs_are_written(monkeypatch, tmp_path):
    # rewriting tracked figures first made a clean run record dirty True
    calls = []
    real = A.run_info
    monkeypatch.setattr(A, "run_info", lambda: calls.append("provenance") or real())
    monkeypatch.setattr(A, "figure_learning", lambda *a: calls.append("figure"))
    monkeypatch.setattr(A, "figure_benchmarks", lambda *a: None)
    monkeypatch.setattr(A, "figure_diagnostics", lambda *a: None)
    A.analyze(out=str(tmp_path / "out"), figures=str(tmp_path / "figures"), sheet=str(tmp_path / "sheet.md"), annotations=str(tmp_path / "ann.csv"))
    assert calls == ["provenance", "figure"]


def test_analyze_output_is_deterministic(run, tmp_path):
    d, *_ = run
    shutil.copy(d / "ann.csv", tmp_path / "ann.csv")
    A.analyze(out=str(tmp_path / "out"), figures=str(tmp_path / "figures"), sheet=str(tmp_path / "sheet.md"), annotations=str(tmp_path / "ann.csv"))
    assert (tmp_path / "out" / "summary.md").read_text() == (d / "out" / "summary.md").read_text()
    for name in A.FIGURES:
        assert (tmp_path / "figures" / name).read_bytes() == (d / "figures" / name).read_bytes()
