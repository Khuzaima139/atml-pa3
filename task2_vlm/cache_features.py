"""Cache frozen SigLIP patch features for every selected image (manual L345-355).

For images_train.json and images_val.json: download (parallel, retries; every file must decode), convert to RGB,
preprocess with the slow PIL processor (use_fast=False: resize to 224x224 bicubic, rescale, normalize to [-1, 1]),
run SiglipVisionModel and keep last_hidden_state only: the 196 patch tokens in 14 x 14 order after the final
post_layernorm, applied once by the model (notes/TASK2_FACTS.md step 17); no pooling head output, nothing else.
Features are stored as FP16 safetensors shards ([n, 196, 768], key "features") with an image-id index and the
sha256 of every shard. The encoder is freed at the end.

Run: python -m task2_vlm.cache_features [--splits train val] [--device cuda]
"""
import argparse
import datetime
import gc
import hashlib
import json
import os
import time

import numpy as np
import torch
import yaml

from common.results import write_result
from common.runinfo import run_info
from task2_vlm.images import download_images, image_path, load_image_table, open_rgb

CONFIG = "configs/task2/features.yaml"


def load_config(path=CONFIG):
    with open(path) as f:
        return yaml.safe_load(f)


def versions():
    import PIL
    import safetensors
    import transformers
    v = {"torch": torch.__version__, "transformers": transformers.__version__, "Pillow": PIL.__version__,
         "numpy": np.__version__, "safetensors": safetensors.__version__}
    try:
        import torchvision
        v["torchvision"] = torchvision.__version__
    except ImportError:
        v["torchvision"] = None
    return v


def load_encoder(cfg, device):
    from transformers import AutoImageProcessor, SiglipVisionModel
    m = cfg["model"]
    proc = AutoImageProcessor.from_pretrained(m["name"], revision=m["revision"], use_fast=cfg["processor"]["use_fast"])
    assert type(proc).__name__ == cfg["processor"]["expected_class"], type(proc).__name__
    model = SiglipVisionModel.from_pretrained(m["name"], revision=m["revision"],
                                              dtype=getattr(torch, cfg["compute_dtype"]))
    model.to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return proc, model


@torch.no_grad()
def encode(images, proc, model, cfg, device):
    """List of RGB PIL images -> FP32 last_hidden_state [n, 196, 768] on CPU (checked shape, finite)."""
    px = proc(images=images, return_tensors="pt")["pixel_values"]
    px = px.to(device, getattr(torch, cfg["compute_dtype"]))
    out = model(pixel_values=px).last_hidden_state
    assert tuple(out.shape) == (len(images), cfg["tokens"], cfg["dim"]), tuple(out.shape)
    assert torch.isfinite(out).all(), "non-finite SigLIP features"
    return out.float().cpu()


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def cache_split(name, table, proc, model, cfg, device, images_dir, out_dir):
    from safetensors.torch import save_file
    store = getattr(torch, cfg["store_dtype"])
    recs = table["images"]
    ids = [r["image_id"] for r in recs]
    assert len(ids) == len(set(ids))
    os.makedirs(out_dir, exist_ok=True)
    shards, index = [], {}
    norm_sum, norm_n, max_abs = 0.0, 0, 0.0
    for k, s in enumerate(range(0, len(recs), cfg["shard_size"])):
        part = recs[s:s + cfg["shard_size"]]
        feats = []
        for b in range(0, len(part), cfg["batch_size"]):
            ims = [open_rgb(image_path(images_dir, table["split"], r["file_name"]))
                   for r in part[b:b + cfg["batch_size"]]]
            f = encode(ims, proc, model, cfg, device)
            norm = f.norm(dim=-1)
            norm_sum, norm_n = norm_sum + norm.sum().item(), norm_n + norm.numel()
            max_abs = max(max_abs, f.abs().max().item())
            feats.append(f.to(store))
        x = torch.cat(feats).contiguous()
        assert torch.isfinite(x).all(), f"non-finite after the {cfg['store_dtype']} cast"
        part_ids = [r["image_id"] for r in part]
        fn = f"{name}_{k:03d}.safetensors"
        path = os.path.join(out_dir, fn)
        save_file({"features": x}, path, metadata={"image_ids": json.dumps(part_ids)})
        shards.append({"file": fn, "n": len(part_ids), "sha256": sha256_file(path), "image_ids": part_ids})
        for row, i in enumerate(part_ids):
            index[str(i)] = [k, row]
    idx = {"split": name, "image_split": table["split"], "model": cfg["model"], "processor_class": type(proc).__name__,
           "dtype": cfg["store_dtype"], "shape_per_image": [cfg["tokens"], cfg["dim"]], "n_images": len(ids),
           "shards": shards, "index": index}
    with open(os.path.join(out_dir, f"{name}_index.json"), "w") as f:
        json.dump(idx, f, indent=1)
    assert len(index) == len(ids)
    return {"n_images": len(ids), "n_shards": len(shards), "token_l2_norm_mean": norm_sum / max(norm_n, 1),
            "max_abs": max_abs, "index": f"{name}_index.json",
            "shards": [{k: v for k, v in s.items() if k != "image_ids"} for s in shards]}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=CONFIG)
    ap.add_argument("--splits", nargs="+", default=["train", "val"])
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--tables", nargs="*", default=None, help="split=path overrides of the image tables")
    ap.add_argument("--images-dir", default=None)
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--result", default=None)
    args = ap.parse_args(argv)
    cfg = load_config(args.config)
    tables = dict(cfg["image_tables"], **dict(t.split("=", 1) for t in (args.tables or [])))
    images_dir = args.images_dir or cfg["images_dir"]
    out_dir = args.out_dir or cfg["out_dir"]
    result = args.result or cfg["result"]
    device = torch.device(args.device)
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    doc = {"run_id": f"cache_features_{stamp}", "status": "running", "config": cfg, "provenance": run_info(),
           "versions": versions(), "device": str(device), "splits": args.splits,
           "tables": {s: {"path": tables[s], "sha256": sha256_file(tables[s])} for s in args.splits},
           "batch_size": cfg["batch_size"], "compute_dtype": cfg["compute_dtype"], "store_dtype": cfg["store_dtype"]}
    write_result(result, doc)

    t0 = time.time()
    doc["download"] = {}
    loaded = {s: load_image_table(tables[s]) for s in args.splits}
    pending = sum(not os.path.exists(image_path(images_dir, loaded[s]["split"], r["file_name"]))
                  for s in args.splits for r in loaded[s]["images"])
    d = cfg["download"]
    print(f"download: {pending} images missing, {d['workers']} workers, {d['retries']} attempts each", flush=True)
    reported = False
    for s in args.splits:
        doc["download"][s] = download_images(loaded[s], images_dir, d["workers"], d["retries"], d["timeout_s"],
                                             None if reported else d["report_after"], pending)
        reported = reported or doc["download"][s]["progress"] is not None
    doc["download_s"] = time.time() - t0
    print(f"download: done in {doc['download_s']:.1f}s", flush=True)
    doc["image_counts"] = {s: len(loaded[s]["images"]) for s in args.splits}
    write_result(result, doc)

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    t1 = time.time()
    proc, model = load_encoder(cfg, device)
    doc["processor_class"] = type(proc).__name__
    doc["processor"] = {k: v for k, v in proc.to_dict().items() if isinstance(v, (int, float, str, bool, list))}
    doc["encoder_parameters"] = sum(p.numel() for p in model.parameters())
    doc["load_s"] = time.time() - t1
    t2 = time.time()
    doc["features"] = {s: cache_split(s, loaded[s], proc, model, cfg, device, images_dir, out_dir)
                       for s in args.splits}
    doc["encode_s"] = time.time() - t2
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        doc["peak_gpu_allocated_bytes"] = torch.cuda.max_memory_allocated(device)
        doc["peak_gpu_reserved_bytes"] = torch.cuda.max_memory_reserved(device)
    else:
        doc["peak_gpu_allocated_bytes"] = doc["peak_gpu_reserved_bytes"] = None
    del model, proc
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
        doc["gpu_allocated_after_free_bytes"] = torch.cuda.memory_allocated(device)
    doc["out_dir"] = out_dir
    doc["wall_s"] = time.time() - t0
    doc["status"] = "done"
    write_result(result, doc)
    print(f"cache_features: {doc['image_counts']} download {doc['download_s']:.1f}s encode {doc['encode_s']:.1f}s "
          f"peak_gpu {doc['peak_gpu_allocated_bytes']} -> {result}")
    return doc


if __name__ == "__main__":
    main()
