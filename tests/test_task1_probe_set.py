import json
import os

from task1_finetune.make_probe_set import OUT, build, sample_probe_ids

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def test_sampling_is_deterministic():
    assert sample_probe_ids(1319, 10042) == sample_probe_ids(1319, 10042)


def test_ids_sorted_unique_in_range():
    g, h = sample_probe_ids(1319, 10042)
    assert len(g) == len(h) == 32
    assert g == sorted(set(g)) and h == sorted(set(h))
    assert 0 <= g[0] and g[-1] < 1319 and 0 <= h[0] and h[-1] < 10042


def test_hellaswag_draw_uses_same_rng_after_gsm8k():
    import numpy as np
    rng = np.random.default_rng(6304)
    rng.choice(1319, 32, replace=False)
    expected = sorted(int(i) for i in rng.choice(10042, 32, replace=False))
    assert sample_probe_ids(1319, 10042)[1] == expected


def test_saved_file_matches_rerun():
    with open(os.path.join(REPO, OUT)) as f:
        saved = json.load(f)
    assert saved == build(saved["gsm8k"]["split_size"], saved["hellaswag"]["split_size"], saved["seed"])
    assert saved["seed"] == 6304
    assert (saved["gsm8k"]["split_size"], saved["hellaswag"]["split_size"]) == (1319, 10042)
