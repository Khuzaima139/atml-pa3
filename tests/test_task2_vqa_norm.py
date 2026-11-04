"""VQA answer normalization (decision D2): official vqaEval.py steps, applied unconditionally."""
import ast
import os

import pytest

from task2_vlm import vqa_norm as V
from task2_vlm.vqa_norm import normalize

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OFFICIAL = os.path.join(REPO, "data/task2/raw/vqa_code/PythonEvaluationTools/vqaEvaluation/vqaEval.py")


@pytest.mark.parametrize("a,b,expected", [
    ("t-shirt", "t shirt", "t shirt"),
    ("none", "0", "0"),
    ("zero", "0", "0"),
    ("two", "2", "2"),
    ("ten", "10", "10"),
    ("a dog", "dog", "dog"),
    ("the man", "man", "man"),
    ("an apple", "apple", "apple"),
    ("Yes", "yes", "yes"),
    ("yes.", "yes", "yes"),
    ("yes!", "yes", "yes"),
    ("dont", "don't", "don't"),
    ("1,000", "1000", "1000"),
    ("left side.", "left side", "left side"),
    ("red/white", "red white", "red white"),
    ("(1)", "1", "1"),
    ("2.", "2", "2"),
])
def test_merges(a, b, expected):
    assert normalize(a) == normalize(b) == expected


@pytest.mark.parametrize("a,b", [
    ("tshirt", "t-shirt"), ("dog", "dogs"), ("eleven", "11"), ("grey", "gray"),
    ("black & white", "black and white"), ("1.0", "1"), ("3 dogs", "3"),
])
def test_does_not_merge(a, b):
    assert normalize(a) != normalize(b)


def test_punctuation_deleted_when_touching_a_space():
    assert normalize("yes ,no") == "yes no"      # ' ,' occurs: ',' deleted
    assert normalize("red,white") == "red white"  # no space next to ',': replaced by a space
    assert normalize("3.5") == "3.5"              # period before a digit kept
    assert normalize("it's") == "it's"            # apostrophe is not in the list


def test_whitespace_and_edge_cases():
    assert normalize("  yes\n") == "yes"
    assert normalize("a\tdog") == "dog"
    assert normalize("a") == "" and normalize("the") == ""
    assert normalize("no one") == "no 1"
    assert normalize("A Dog") == "dog"


def test_unconditional_rule():
    """The official code normalizes nothing when all ten references are identical; here every string is
    normalized on its own, so a prediction is treated the same whatever the references are."""
    refs = ["yes"] * 10
    assert all(normalize(r) == "yes" for r in refs)
    assert normalize("Yes") == "yes" and normalize("2.") == "2"
    assert normalize(normalize("The T-Shirt.")) == normalize("The T-Shirt.") == "t shirt"


def test_period_count_cap_matches_official():
    """re.UNICODE (=32) is the count argument in the official sub(): at most 32 periods removed."""
    assert normalize("." * 40 + "x") == "." * 8 + "x"


def _official_block(src, name, opener, closer):
    i = src.index(opener, src.index("self." + name))
    depth = 0
    for j in range(i, len(src)):
        depth += src[j] == opener
        depth -= src[j] == closer
        if depth == 0:
            return ast.literal_eval(src[i:j + 1])


@pytest.mark.skipif(not os.path.exists(OFFICIAL), reason="official vqaEval.py not downloaded")
def test_tables_equal_official_file():
    src = open(OFFICIAL).read().replace("\\\n", "\n")
    assert _official_block(src, "contractions", "{", "}") == V.CONTRACTIONS
    assert _official_block(src, "manualMap", "{", "}") == V.MANUAL_MAP
    assert _official_block(src, "articles", "[", "]") == V.ARTICLES
    assert _official_block(src, "punct", "[", "]") == V.PUNCT
    assert 'periodStrip  = re.compile("(?!<=\\d)(\\.)(?!\\d)")' in src
