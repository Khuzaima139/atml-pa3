"""SigLIP feature cache: CPU end to end on 4 real COCO images through the real entry point."""
import hashlib
import json
import os
import shutil

import pytest
import torch

from task2_vlm import cache_features as C
from task2_vlm.images import open_rgb

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOCAL = os.path.join(REPO, "data/task2/raw/imgs")
IMAGES = {"train2014": [9, 86], "val2014": [42, 73]}  # 86 is single-channel (mode L)


def table(split, ids):
    fn = lambda i: f"COCO_{split}_{i:012d}.jpg"
    return {"split": split, "images": [{"image_id": i, "file_name": fn(i),
                                        "url": f"http://images.cocodataset.org/{split}/{fn(i)}"} for i in ids]}


@pytest.fixture(scope="module")
def run(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("cache")
    tables = []
    for name, split in (("train", "train2014"), ("val", "val2014")):
        p = tmp / f"images_{name}.json"
        p.write_text(json.dumps(table(split, IMAGES[split])))
        tables.append(f"{name}={p}")
        os.makedirs(tmp / "images" / split)
        for i in IMAGES[split]:  # reuse local copies; the rest is downloaded
            src = os.path.join(LOCAL, f"COCO_{split}_{i:012d}.jpg")
            if os.path.exists(src):
                shutil.copy(src, tmp / "images" / split)
    cfg = C.load_config(os.path.join(REPO, C.CONFIG))
    cfg["batch_size"], cfg["shard_size"] = 1, 1  # exercise batching and several shards
    cfg_path = tmp / "features.yaml"
    import yaml
    cfg_path.write_text(yaml.safe_dump(cfg))
    doc = C.main(["--config", str(cfg_path), "--device", "cpu", "--tables", *tables,
                  "--images-dir", str(tmp / "images"), "--out-dir", str(tmp / "out"),
                  "--result", str(tmp / "run.json")])
    return tmp, cfg, doc


def load_shard(path):
    from safetensors import safe_open
    with safe_open(str(path), "pt") as f:
        return f.get_tensor("features"), json.loads(f.metadata()["image_ids"])


def test_run_json(run):
    tmp, cfg, doc = run
    saved = json.load(open(tmp / "run.json"))
    assert saved["run_id"] == doc["run_id"] and saved["status"] == "done"
    assert saved["image_counts"] == {"train": 2, "val": 2}
    assert saved["processor_class"] == "SiglipImageProcessorPil"
    assert saved["config"]["model"]["revision"] == "7fd15f0689c79d79e38b1c2e2e2370a7bf2761ed"
    for k in ("torch", "transformers", "Pillow", "numpy", "torchvision"):
        assert k in saved["versions"]
    for k in ("download_s", "encode_s", "batch_size", "peak_gpu_allocated_bytes", "provenance"):
        assert k in saved
    assert saved["download"]["train"]["images"] == 2


def test_shards_index_and_hashes(run):
    tmp, cfg, doc = run
    for name, split in (("train", "train2014"), ("val", "val2014")):
        idx = json.load(open(tmp / "out" / f"{name}_index.json"))
        assert idx["n_images"] == 2 and sorted(int(i) for i in idx["index"]) == IMAGES[split]
        assert idx["dtype"] == "float16" and idx["shape_per_image"] == [196, 768]
        for k, s in enumerate(idx["shards"]):
            path = tmp / "out" / s["file"]
            assert hashlib.sha256(path.read_bytes()).hexdigest() == s["sha256"]
            x, ids = load_shard(path)
            assert ids == s["image_ids"] and x.dtype == torch.float16 and tuple(x.shape) == (s["n"], 196, 768)
            assert torch.isfinite(x).all()
            for row, i in enumerate(ids):
                assert idx["index"][str(i)] == [k, row]


def test_features_are_last_hidden_state_with_one_layernorm(run):
    tmp, cfg, doc = run
    proc, model = C.load_encoder(cfg, torch.device("cpu"))
    img = open_rgb(tmp / "images" / "train2014" / "COCO_train2014_000000000086.jpg")
    px = proc(images=[img], return_tensors="pt")["pixel_values"]
    assert torch.equal(px[0, 0], px[0, 1]) and torch.equal(px[0, 1], px[0, 2])  # grayscale -> RGB
    with torch.no_grad():
        out = model(pixel_values=px, output_hidden_states=True)
    ln_once = model.post_layernorm(out.hidden_states[-1])
    assert torch.allclose(out.last_hidden_state, ln_once, atol=1e-5)
    idx = json.load(open(tmp / "out" / "train_index.json"))
    k, row = idx["index"]["86"]
    x, _ = load_shard(tmp / "out" / idx["shards"][k]["file"])
    assert torch.equal(x[row], out.last_hidden_state[0].to(torch.float16))
    assert (x[row].float() - out.hidden_states[-1][0]).abs().max() > 1.0  # not the pre-LN state


def test_unreadable_file_is_rejected(tmp_path):
    p = tmp_path / "bad.jpg"
    p.write_bytes(b"not an image")
    with pytest.raises(Exception):
        open_rgb(p)


def test_download_progress_report(tmp_path, monkeypatch, capsys):
    """Kaggle cell 2: rate and projected download time after the first `report_after` images."""
    from task2_vlm import images

    def fake_fetch(url, dest, retries, timeout):
        open(dest, "wb").write(b"x")
        return 1
    monkeypatch.setattr(images, "_fetch", fake_fetch)
    table = {"split": "val2014", "images": [{"image_id": i, "file_name": f"{i}.jpg", "url": f"u{i}"} for i in range(7)]}
    r = images.download_images(table, str(tmp_path), workers=2, report_after=3, pending_total=20)
    assert r["downloaded"] == 7 and r["progress"]["images"] == 3 and r["progress"]["pending_total"] == 20
    assert r["progress"]["projected_total_s"] > 0
    assert "download: first 3 images" in capsys.readouterr().out
