"""Task 2 training: caption alignment, visual instruction tuning, natural VQA specialization (manual L388-427,
L460-483; decisions D10-D14). One continuous run of 4,500 updates with development evaluation inside it.

Order at every update: train update -> development evaluation (if listed) -> stage-end weights (connector, and
LoRA from Stage 2) -> resumable state (every 250 updates, rolling). Calibration (D12) runs before update 0.

Run (CUDA only; never on the Mac):
    python -m task2_vlm.train --features_dir /kaggle/working/features --recovery_dir /kaggle/tmp/task2_recovery
    python -m task2_vlm.train --smoke ...       # 20 updates per stage, dev evaluation at update 0, smoke dirs
    python -m task2_vlm.train --resume ...      # continue from the latest state in --recovery_dir
"""
import argparse
import collections
import copy
import datetime
import glob
import math
import os
import time

import numpy as np
import torch
import yaml

from common.results import write_result
from common.runinfo import run_info
from common.seed import rng_state, seed_everything, set_rng_state
from task2_vlm import model as M
from task2_vlm.data import Encoder, FeatureStore, check_template, collate, load_splits, pair_target, pair_type, \
    sha256_file
from task2_vlm.sampler import Sampler
from task2_vlm.scoring import mean, reference_nll

CONFIG = "configs/task2/train.yaml"
LOG_FIELDS = ("update is the global 1-based update just completed, stage_update its 1-based index in the stage; "
              "lr per optimizer group is the rate applied in that update; loss = mean over the 16 examples of "
              "each example's mean supervised-token NLL; grad_norm_preclip is the global norm over all trainable "
              "parameters after unscale_ and before clipping; loss_scale is the scale used for that backward; "
              "skipped = the scale dropped (inf/nan gradients, optimizer step skipped); mixture = draws per "
              "component in this update (Stage 1: caption examples)")


def load_config(path=CONFIG):
    with open(path) as f:
        return yaml.safe_load(f)


def _sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _num(x):
    x = float(x)
    return x if math.isfinite(x) else None


def versions():
    import accelerate
    import peft
    import safetensors
    import transformers
    return {"torch": torch.__version__, "transformers": transformers.__version__, "peft": peft.__version__,
            "accelerate": accelerate.__version__, "safetensors": safetensors.__version__, "numpy": np.__version__,
            "cuda": torch.version.cuda}


# ---------------------------------------------------------------- schedule (D14)

def warmup_updates(T, percent):
    """round-half-up(percent% of T) in integer arithmetic: 2250 -> 113, 1500 -> 75, 750 -> 38."""
    return (T * percent * 2 + 100) // 200


def lr_factor(t, W, T, final=0.1):
    """Multiplier for the t-th update of a stage (1-based): t/W up to W, then cosine to `final` at t = T."""
    if t <= W:
        return t / W
    return final + (1 - final) / 2 * (1 + math.cos(math.pi * (t - W) / (T - W)))


def make_optimizer(cfg, stage, conn, lora, W, T, amp):
    """Fresh AdamW for one stage (D13; weight decay on every trainable tensor) and its LambdaLR (D14)."""
    o = cfg["optimizer"]
    groups = [{"params": conn, "lr": stage["lr"]["connector"], "weight_decay": o["weight_decay"], "name": "connector"}]
    if lora:
        groups.append({"params": lora, "lr": stage["lr"]["lora"], "weight_decay": o["weight_decay"], "name": "lora"})
    extra = {"fused": True} if amp and o["fused_on_cuda"] else {}
    opt = torch.optim.AdamW(groups, betas=tuple(o["betas"]), eps=o["eps"], **extra)
    fin = cfg["schedule"]["final_fraction"]
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda e: lr_factor(e + 1, W, T, fin))
    return opt, sched


def microbatch_loss(vlm, batch, feats, chunk, effective):
    """sum over the microbatch of each example's mean supervised-token NLL, divided by the effective batch size
    (L401-408); backward of these terms over the accumulated microbatches gives the manual's loss. Also returns
    the per-example means and supervised-token counts."""
    with vlm.autocast():
        nll, rows, content = vlm.token_nll(batch, feats, chunk)
    means, n = M.example_means(nll, rows, torch.ones_like(content), batch["input_ids"].shape[0])
    return means.sum() / effective, means.detach(), n


# ---------------------------------------------------------------- calibration (D12)

@torch.no_grad()
def calibrate(vlm, enc, d, store, cfg):
    c = cfg["calibration"]
    pools = d["grounding_pools_train"]
    texts = {"captions_train": [cap for _, cap in d["caption_examples"]],
             "vqa_train_questions": [p["question"] for p in d["vqa_train"]["pairs"] for _ in p["question_ids"]],
             "grounding_templates": [q[k] for _, q in sorted(pools["questions"].items(), key=lambda x: int(x[0]))
                                     for k in ("presence", "counting")]}
    W = vlm.lm.get_input_embeddings().weight
    row_norm = W.detach().float().norm(dim=-1).double().cpu()
    counts = torch.zeros_like(row_norm)
    sources = {}
    for name, ts in texts.items():
        flat = torch.tensor([i for x in enc.tok(ts, add_special_tokens=False)["input_ids"] for i in x])
        sources[name] = {"texts": len(ts), "tokens": int(flat.numel()), "mean_l2": float(row_norm[flat].mean())}
        counts += torch.bincount(flat, minlength=row_norm.numel()).double()
    text_mean = float((counts * row_norm).sum() / counts.sum())
    ids = sorted(r["image_id"] for r in d["images_train"]["images"])
    sample = [int(i) for i in np.random.default_rng(c["seed"]).choice(ids, size=c["n_images"], replace=False)]
    tot, n = 0.0, 0
    vlm.connector.scale.fill_(1.0)
    for s in range(0, len(sample), 64):
        f = store.get(sample[s:s + 64]).to(vlm.device).float()
        with torch.autocast(vlm.device.type, enabled=False):
            norms = vlm.connector.raw(f).norm(dim=-1)
        tot, n = tot + norms.double().sum().item(), n + norms.numel()
    conn_mean = tot / n
    scale = text_mean / conn_mean
    vlm.connector.scale.fill_(scale)
    return {"rule": c["rule"], "text_mean_l2": text_mean, "text_tokens": int(counts.sum()), "sources": sources,
            "connector_mean_l2_at_init": conn_mean, "connector_tokens": n, "n_images": len(sample),
            "seed": c["seed"], "image_ids": sample, "s": scale,
            "stored_s_float32": float(vlm.connector.scale)}


# ---------------------------------------------------------------- examples

def make_example(item, d, enc):
    kind = item[0]
    if kind == "caption":
        img, cap = d["caption_examples"][item[1]]
        return enc.example(d["instruction"], cap, img, "caption")
    if kind == "vqa":
        p = d["vqa_train"]["pairs"][item[1]]
        return enc.example(p["question"], p["targets"][item[2]], p["image_ids"][item[2]], "answer")
    _, img, cat, ans = item
    q = d["grounding_pools_train"]["questions"][str(cat)][kind]
    return enc.example(q, str(ans), img, "answer")


def dev_sets(d, enc):
    """Development examples: 96 captions; VQA and grounding dev pairs under correct / complementary / no image."""
    out = {"captions_dev": {"meta": [{"image_id": i} for i, _ in d["caption_dev_examples"]],
                            "examples": [enc.example(d["instruction"], c, i, "caption")
                                         for i, c in d["caption_dev_examples"]]}}
    for suite in ("vqa_dev", "grounding_dev"):
        meta, ex = [], {"correct": [], "complementary": [], "none": []}
        for p in d[suite]["pairs"]:
            for s in (0, 1):
                tgt = pair_target(suite, p, s)
                meta.append({"pair_id": p["pair_id"], "slot": s, "type": pair_type(suite, p), "target": tgt,
                             "image_correct": p["image_ids"][s], "image_complementary": p["image_ids"][1 - s]})
                ex["correct"].append(enc.example(p["question"], tgt, p["image_ids"][s], "answer"))
                ex["complementary"].append(enc.example(p["question"], tgt, p["image_ids"][1 - s], "answer"))
                ex["none"].append(enc.example(p["question"], tgt, None, "answer"))
        out[suite] = {"meta": meta, "examples": ex}
    return out


# ---------------------------------------------------------------- trainer

class Trainer:
    def __init__(self, cfg, vlm, d, enc, stores, dirs, kind="train", resume=False, info=None, main_cfg=None):
        self.cfg, self.vlm, self.d, self.enc, self.stores, self.dirs = cfg, vlm, d, enc, stores, dirs
        self.device, self.kind, self.info = vlm.device, kind, info or {}
        self.main_cfg = main_cfg or cfg
        self.stages = cfg["stages"]
        self.T = [s["updates"] for s in self.stages]
        self.ends = list(np.cumsum(self.T))
        self.total = int(self.ends[-1])
        self.W = [warmup_updates(t, cfg["schedule"]["warmup_percent"]) for t in self.T]
        b = cfg["batch"]
        self.mb, self.accum, self.draws = b["microbatch"], b["grad_accum"], b["draws_per_microbatch"]
        self.eff = self.mb * self.accum
        assert self.eff == b["effective"]
        assert self.T[0] * self.eff <= self.stages[0]["caption_passes"] * len(d["caption_examples"])
        self.chunk = cfg["loss"]["logit_chunk"]
        self.dev_updates = list(cfg["dev_eval"]["updates"])
        self.scaler = torch.amp.GradScaler(self.device.type, enabled=vlm.amp,
                                           init_scale=cfg["precision"]["init_scale"])
        self.sampler = Sampler(len(d["caption_examples"]), len(d["vqa_train"]["pairs"]),
                               d["grounding_pools_train"], cfg["seed"])
        self.dev = dev_sets(d, enc)
        self.done, self.stage_k, self.stage_t = 0, None, 0
        self.opt = self.sched = None
        self.log, self.dev_summaries, self.stage_records, self.files = [], [], [], []
        self.acc = {"train_s": 0.0, "dev_eval_s": 0.0, "ckpt_io_s": 0.0, "calibration_s": 0.0, "wall_s_prev": 0.0,
                    "skipped": 0, "peak_allocated_bytes": 0, "peak_reserved_bytes": 0, "recovery_save_s": []}
        stamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        self.run_id = f"task2_{kind}_{stamp}"
        self.segments = []
        self.t_start = time.time()
        if resume:
            self._resume()
        else:
            _sync(self.device)
            t0 = time.perf_counter()
            self.calibration = calibrate(vlm, enc, d, stores["train"], cfg)
            _sync(self.device)
            self.acc["calibration_s"] = time.perf_counter() - t0
            print(f"calibration: text mean L2 {self.calibration['text_mean_l2']:.6f} / connector mean L2 "
                  f"{self.calibration['connector_mean_l2_at_init']:.6f} -> s = {self.calibration['s']:.6f}",
                  flush=True)
        self.segments.append({"from_update": self.done, "provenance": run_info(), "versions": versions()})

    # -- stages
    def _stage_record(self, k):
        for r in self.stage_records:
            if r["stage"] == k + 1:
                return r
        r = {"stage": k + 1, "name": self.stages[k]["name"], "updates": self.T[k],
             "first_update": int(self.ends[k] - self.T[k] + 1), "last_update": int(self.ends[k]),
             "warmup_updates": self.W[k], "peak_lr": dict(self.stages[k]["lr"]), "status": "running",
             "train_s": 0.0, "dev_eval_s": 0.0, "ckpt_io_s": 0.0, "skipped": 0,
             "peak_allocated_bytes": 0, "peak_reserved_bytes": 0}
        self.stage_records.append(r)
        return r

    def _fold_peak(self):
        if self.device.type != "cuda" or self.stage_k is None:
            return
        r = self._stage_record(self.stage_k)
        a, v = torch.cuda.max_memory_allocated(self.device), torch.cuda.max_memory_reserved(self.device)
        r["peak_allocated_bytes"] = max(r["peak_allocated_bytes"], a)
        r["peak_reserved_bytes"] = max(r["peak_reserved_bytes"], v)
        self.acc["peak_allocated_bytes"] = max(self.acc["peak_allocated_bytes"], a)
        self.acc["peak_reserved_bytes"] = max(self.acc["peak_reserved_bytes"], v)

    def _begin_stage(self, k, fresh):
        st = self.stages[k]
        self._fold_peak()
        if st["lora"] and self.vlm.peft is None:
            self.vlm.add_lora()
        named = self.vlm.named_trainable()
        n_params = sum(p.numel() for _, p in named)
        expected = self.cfg["expected_trainable_params"][k]
        assert expected is None or n_params == expected, f"stage {k + 1}: {n_params} trainable, expected {expected}"
        assert all(p.dtype == torch.float32 for _, p in named), "trainable parameters must be FP32"
        conn = [p for n, p in named if n.startswith("connector.")]
        lora = [p for n, p in named if not n.startswith("connector.")]
        assert bool(lora) == bool(st["lora"])
        W, T = self.W[k], self.T[k]
        self.opt, self.sched = make_optimizer(self.cfg, st, conn, lora, W, T, self.vlm.amp)
        groups = self.opt.param_groups
        self.params = [p for _, p in named]
        self.stage_k = k
        if fresh:
            self.stage_t = 0
            self.sampler.start_stage(k, st)
        r = self._stage_record(k)
        r.update(trainable_params=n_params, trainable_tensors=len(named),
                 trainable_groups={g["name"]: sum(p.numel() for p in g["params"]) for g in groups})
        if self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(self.device)
        print(f"stage {k + 1} ({st['name']}): {'start' if fresh else 'resume'} at update {self.done}, trainable "
              f"{n_params:,}, warmup {W} of {T}, peak lr {st['lr']}", flush=True)

    def _end_stage(self, k):
        os.makedirs(self.dirs["keep"], exist_ok=True)
        t0 = time.perf_counter()
        meta = {"run_id": self.run_id, "stage": str(k + 1), "update": str(self.done)}
        path = os.path.join(self.dirs["keep"], f"connector_stage{k + 1}_u{self.done:04d}.safetensors")
        self._save_tensors(self.vlm.connector_state(), path, {**meta, "scale_s": repr(self.calibration["s"])})
        self._record(path, "connector", stage=k + 1)
        if self.stages[k]["lora"]:
            c = self.cfg["lora"]
            path = os.path.join(self.dirs["keep"], f"lora_stage{k + 1}_u{self.done:04d}.safetensors")
            self._save_tensors(self.vlm.lora_state(), path,
                               {**meta, "r": str(c["r"]), "alpha": str(c["alpha"]), "targets": ",".join(c["targets"])})
            self._record(path, "lora", stage=k + 1)
        dt = time.perf_counter() - t0
        self.acc["ckpt_io_s"] += dt
        self._fold_peak()
        r = self._stage_record(k)
        r["ckpt_io_s"] += dt
        r.update(status="done", mixture=self.sampler.summary())
        print(f"stage {k + 1} done at update {self.done}: train {r['train_s'] / 3600:.2f}h, peak alloc "
              f"{r['peak_allocated_bytes'] / 2**30:.2f}GiB, mixture {r['mixture']}", flush=True)

    # -- one update
    def _update(self):
        k = self.stage_k
        _sync(self.device)
        t0 = time.perf_counter()
        lrs = {g["name"]: g["lr"] for g in self.opt.param_groups}
        loss_sum, mix, sup, padded = 0.0, collections.Counter(), 0, 0
        for _ in range(self.accum):
            items, m = self.sampler.microbatch(self.mb, self.draws)
            mix.update(m)
            exs = [make_example(it, self.d, self.enc) for it in items]
            b = {k_: v.to(self.device) for k_, v in collate(exs, self.enc.n_vis, self.enc.pad, "right").items()}
            feats = self.stores["train"].get([e["image"] for e in exs]).to(self.device)
            loss, means, n = microbatch_loss(self.vlm, b, feats, self.chunk, self.eff)
            self.scaler.scale(loss).backward()
            loss_sum += float(means.sum())
            sup += int(n.sum())
            padded += b["input_ids"].numel()
        self.scaler.unscale_(self.opt)
        gn = torch.nn.utils.clip_grad_norm_(self.params, self.cfg["clip_grad_norm"])
        scale = self.scaler.get_scale()
        self.scaler.step(self.opt)
        self.scaler.update()
        skipped = self.scaler.get_scale() < scale
        self.opt.zero_grad(set_to_none=True)
        self.sched.step()
        _sync(self.device)
        dt = time.perf_counter() - t0
        rec = {"update": self.done + 1, "stage": k + 1, "stage_update": self.stage_t + 1,
               "loss": _num(loss_sum / self.eff), "lr": lrs, "grad_norm_preclip": _num(gn), "loss_scale": scale,
               "skipped": bool(skipped), "mixture": dict(mix), "supervised_tokens": sup, "padded_tokens": padded,
               "time_s": dt}
        r = self._stage_record(k)
        r["train_s"] += dt
        self.acc["train_s"] += dt
        if skipped:
            self.acc["skipped"] += 1
            r["skipped"] += 1
            print(f"SKIPPED update {self.done + 1}: loss scale {scale:g} -> {self.scaler.get_scale():g}", flush=True)
        self.done += 1
        self.stage_t += 1
        return rec

    # -- development evaluation
    def _dev_eval(self, g):
        _sync(self.device)
        t0 = time.perf_counter()
        self.vlm.eval()
        bs, store, chunk = self.cfg["dev_eval"]["batch_size"], self.stores["val"], self.chunk
        cap = reference_nll(self.vlm, self.enc, self.dev["captions_dev"]["examples"], store, bs, chunk)
        rows = [{"set": "captions_dev", **m, "loss": c["loss"], "n_tokens": c["n_content"] + 1}
                for m, c in zip(self.dev["captions_dev"]["meta"], cap)]
        summary = {"update": g, "stage": self._stage_of(g), "caption_loss": mean(r["loss"] for r in rows)}
        for suite in ("vqa_dev", "grounding_dev"):
            ds = self.dev[suite]
            res = {c: reference_nll(self.vlm, self.enc, ds["examples"][c], store, bs, chunk)
                   for c in self.cfg["dev_eval"]["conditions"]}
            srows = []
            for i, m in enumerate(ds["meta"]):
                nll = {c: res[c][i]["answer_nll"] for c in res}
                srows.append({"set": suite, **m, "n_target_tokens": res["correct"][i]["n_content"], "nll": nll,
                              "dnll": nll["complementary"] - nll["correct"]})
            rows += srows
            types = sorted({r["type"] for r in srows})
            summary[suite] = {"nll": {c: mean(r["nll"][c] for r in srows) for c in res},
                              "dnll": mean(r["dnll"] for r in srows),
                              "by_type": {t: {"n": sum(r["type"] == t for r in srows),
                                              "nll": {c: mean(r["nll"][c] for r in srows if r["type"] == t)
                                                      for c in res},
                                              "dnll": mean(r["dnll"] for r in srows if r["type"] == t)}
                                          for t in types}}
        self.vlm.train()
        _sync(self.device)
        dt = time.perf_counter() - t0
        summary["time_s"] = dt
        self.acc["dev_eval_s"] += dt
        if self.stage_k is not None:
            self._stage_record(self.stage_k)["dev_eval_s"] += dt
        path = os.path.join(self.dirs["result"], "dev", f"dev_u{g:04d}.json")
        write_result(path, {"run_id": self.run_id, "update": g, "summary": summary, "examples": rows})
        summary["file"] = path
        self.dev_summaries.append(summary)
        v, gr = summary["vqa_dev"], summary["grounding_dev"]
        print(f"dev u{g}: caption_loss {summary['caption_loss']:.4f} | vqa nll c/p/n "
              f"{v['nll']['correct']:.3f}/{v['nll']['complementary']:.3f}/{v['nll']['none']:.3f} dNLL "
              f"{v['dnll']:.3f} | grounding nll c/p/n {gr['nll']['correct']:.3f}/{gr['nll']['complementary']:.3f}/"
              f"{gr['nll']['none']:.3f} dNLL {gr['dnll']:.3f} | {dt:.1f}s", flush=True)

    def _stage_of(self, g):
        return next(k + 1 for k, e in enumerate(self.ends) if g <= e)

    # -- loop
    def run(self, stop_after=None):
        self._write("running")
        if 0 in self.dev_updates and not any(s["update"] == 0 for s in self.dev_summaries):
            if self.stage_k is None:
                self._begin_stage(0, fresh=True)
            self._dev_eval(0)
            self._write("running")
        self.vlm.train()
        lg = self.cfg["logging"]
        for k in range(len(self.stages)):
            if self.done >= self.ends[k]:
                continue
            if self.stage_k != k:
                self._begin_stage(k, fresh=True)
            while self.done < self.ends[k]:
                rec = self._update()
                self.log.append(rec)
                g = self.done
                at = False
                if g in self.dev_updates:
                    self._dev_eval(g)
                    at = True
                if self.stage_t == self.T[k]:
                    self._end_stage(k)
                    at = True
                if g % self.cfg["checkpoints"]["recovery_every"] == 0:
                    self._save_recovery()
                    at = True
                if g % lg["print_every"] == 0 or g == 1 or at:
                    el = self.acc["train_s"] / g
                    print(f"update {g}/{self.total} stage {k + 1} loss {rec['loss']} lr "
                          + " ".join(f"{n} {v:.3e}" for n, v in rec["lr"].items())
                          + f" gn {rec['grad_norm_preclip']} scale {rec['loss_scale']:g} skipped_total "
                          f"{self.acc['skipped']} {rec['time_s']:.2f}s eta_train {el * (self.total - g) / 3600:.2f}h",
                          flush=True)
                if g % lg["json_every"] == 0 or g == 1 or at:
                    self._write("running")
                if stop_after is not None and g >= stop_after:
                    self._write("stopped")
                    return
        self._write("done")

    # -- files
    def _record(self, path, kind, **extra):
        e = {"path": path, "kind": kind, "update": self.done, "bytes": os.path.getsize(path),
             "sha256": sha256_file(path), **extra}
        self.files.append(e)
        return e

    def _save_tensors(self, tensors, path, metadata):
        from safetensors.torch import save_file
        save_file(tensors, path + ".tmp", metadata=metadata)
        os.replace(path + ".tmp", path)

    def _core_cfg(self):
        c = copy.deepcopy(self.cfg)
        c.pop("output", None)
        c["features"].pop("dir", None)
        return c

    def _save_recovery(self):
        os.makedirs(self.dirs["recovery"], exist_ok=True)
        t0 = time.perf_counter()
        self._fold_peak()
        st = {"format": 1, "run_id": self.run_id, "kind": self.kind, "config": self._core_cfg(),
              "done": self.done, "stage_k": self.stage_k, "stage_t": self.stage_t,
              "connector": self.vlm.connector_state(),
              "lora": self.vlm.lora_state() if self.vlm.peft is not None else None,
              "optimizer": self.opt.state_dict(), "scheduler": self.sched.state_dict(),
              "scaler": self.scaler.state_dict(), "sampler": self.sampler.state_dict(), "rng": rng_state(),
              "log": self.log, "dev_summaries": self.dev_summaries, "stage_records": self.stage_records,
              "files": self.files, "calibration": self.calibration, "segments": self.segments,
              "acc": {**self.acc, "wall_s_prev": self.acc["wall_s_prev"] + time.time() - self.t_start}}
        path = os.path.join(self.dirs["recovery"], f"recovery_u{self.done:04d}.pt")
        torch.save(st, path + ".tmp")
        os.replace(path + ".tmp", path)
        for old in glob.glob(os.path.join(self.dirs["recovery"], "recovery_u*.pt")):
            if os.path.abspath(old) != os.path.abspath(path):
                os.remove(old)
                for e in self.files:
                    if e["kind"] == "recovery" and os.path.abspath(e["path"]) == os.path.abspath(old):
                        e["deleted_at_update"] = self.done
        self._record(path, "recovery")
        dt = time.perf_counter() - t0
        self.acc["ckpt_io_s"] += dt
        self.acc["recovery_save_s"].append(dt)
        self._stage_record(self.stage_k)["ckpt_io_s"] += dt

    def _resume(self):
        files = sorted(glob.glob(os.path.join(self.dirs["recovery"], "recovery_u*.pt")))
        if not files:
            raise SystemExit(f"--resume: no recovery state in {self.dirs['recovery']}")
        path = files[-1]
        st = torch.load(path, map_location="cpu", weights_only=False)
        if st["config"] != self._core_cfg():
            raise SystemExit(f"--resume: config differs from the one stored in {path}")
        self.vlm.connector.load_state_dict(st["connector"])
        if st["lora"] is not None:
            self.vlm.add_lora()
            self.vlm.load_lora_state(st["lora"])
        self.run_id, self.done = st["run_id"], st["done"]
        self.log, self.dev_summaries, self.stage_records = st["log"], st["dev_summaries"], st["stage_records"]
        self.files, self.calibration, self.segments, self.acc = st["files"], st["calibration"], st["segments"], st["acc"]
        self.scaler.load_state_dict(st["scaler"])
        k, t = st["stage_k"], st["stage_t"]
        self.stage_t = t
        if t < self.T[k]:
            self._begin_stage(k, fresh=False)
            self.opt.load_state_dict(st["optimizer"])
            self.sched.load_state_dict(st["scheduler"])
        else:
            self.stage_k = k
        self.sampler.load_state_dict(st["sampler"])
        print(f"resumed from {path} at update {self.done} (stage {k + 1}, {t} of {self.T[k]} done)", flush=True)
        set_rng_state(st["rng"])  # last: nothing after this may consume random numbers

    # -- result JSON
    def _smoke_summary(self):
        by = collections.defaultdict(list)
        for r in self.log:
            by[r["stage"]].append(r["time_s"])
        steady = {k: (v[1:] or v) for k, v in by.items()}
        main_T = [s["updates"] for s in self.main_cfg["stages"]]
        return {"time_per_update_s_by_stage": {str(k): v for k, v in by.items()},
                "mean_s_per_update_excl_first": {str(k): float(np.mean(v)) for k, v in steady.items()},
                "dev_eval_s": [s["time_s"] for s in self.dev_summaries],
                "recovery_save_s": self.acc["recovery_save_s"],
                "main_stage_updates": main_T, "main_dev_points": len(self.main_cfg["dev_eval"]["updates"]),
                "main_recovery_saves": sum(main_T) // self.main_cfg["checkpoints"]["recovery_every"]}

    def _write(self, status):
        self._fold_peak()
        res = {"updates_done": self.done, "updates_total": self.total, "skipped_updates": self.acc["skipped"],
               "train_s": self.acc["train_s"], "dev_eval_s": self.acc["dev_eval_s"],
               "ckpt_io_s": self.acc["ckpt_io_s"], "calibration_s": self.acc["calibration_s"],
               "wall_s": self.acc["wall_s_prev"] + time.time() - self.t_start,
               "peak_allocated_bytes": self.acc["peak_allocated_bytes"],
               "peak_reserved_bytes": self.acc["peak_reserved_bytes"], **self.info.get("resources", {})}
        doc = {"run_id": self.run_id, "kind": self.kind, "status": status, "config": self.cfg,
               "segments": self.segments, "dtypes": self.info.get("dtypes"), "calibration": self.calibration,
               "data": self.info.get("data"), "features": self.info.get("features"),
               "lm_config": self.info.get("lm_config"), "resources": res, "stages": self.stage_records,
               "dev": self.dev_summaries, "files": self.files, "log_fields": LOG_FIELDS, "log": self.log}
        if self.kind == "smoke":
            doc["smoke"] = self._smoke_summary()
        write_result(os.path.join(self.dirs["result"], "train.json"), doc)


# ---------------------------------------------------------------- entry point

def training_device():
    """The only CUDA requirement of main(); the CPU end-to-end test replaces this function."""
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required (training never runs on the Mac)")
    return torch.device("cuda")


def load_tokenizer(cfg):
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(cfg["model"]["name"], revision=cfg["model"]["revision"])


def smoke_config(cfg):
    c = copy.deepcopy(cfg)
    s = cfg["smoke"]
    for st in c["stages"]:
        st["updates"] = s["updates_per_stage"]
    c["dev_eval"]["updates"] = list(s["dev_eval_updates"])
    c["checkpoints"]["recovery_every"] = s["recovery_every"]
    c["output"]["result_dir"] = c["output"]["smoke_result_dir"]
    c["output"]["keep_dir"] = c["output"]["smoke_keep_dir"]
    return c


def apply_overrides(cfg, args):
    if args.features_dir:
        cfg["features"]["dir"] = args.features_dir
    for k in ("recovery_dir", "keep_dir", "result_dir"):
        if getattr(args, k, None):
            cfg["output"][k] = getattr(args, k)
    return cfg


def data_info(cfg, d):
    return {"files": {k: {"path": p, "sha256": sha256_file(p)} for k, p in cfg["data"].items()},
            "caption_examples": len(d["caption_examples"]), "caption_dev": len(d["caption_dev_examples"]),
            "vqa_train_pairs": len(d["vqa_train"]["pairs"]), "vqa_dev_pairs": len(d["vqa_dev"]["pairs"]),
            "grounding_dev_pairs": len(d["grounding_dev"]["pairs"]),
            "pool_images": len(d["grounding_pools_train"]["images"]),
            "presence_positives": len(d["grounding_pools_train"]["presence_positives"]),
            "counting_items": len(d["grounding_pools_train"]["counting_items"]), "instruction": d["instruction"]}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=CONFIG)
    ap.add_argument("--features_dir")
    ap.add_argument("--recovery_dir")
    ap.add_argument("--keep_dir")
    ap.add_argument("--result_dir")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--stop_after", type=int, default=None, help="stop after this global update (tests)")
    args = ap.parse_args(argv)
    main_cfg = load_config(args.config)
    cfg = smoke_config(main_cfg) if args.smoke else copy.deepcopy(main_cfg)
    apply_overrides(cfg, args)
    device = training_device()
    seed_everything(cfg["seed"])
    t0 = time.perf_counter()
    tok = load_tokenizer(cfg)
    check_template(tok, cfg)
    enc = Encoder(tok, cfg)
    d = load_splits(cfg)
    t_data = time.perf_counter() - t0
    t1 = time.perf_counter()
    f = cfg["features"]
    stores = {s: FeatureStore(f["dir"], s, f["tokens"], f["dim"], f["verify_sha256"]) for s in ("train", "val")}
    assert {r["image_id"] for r in d["images_train"]["images"]} <= set(stores["train"].row)
    assert {r["image_id"] for r in d["images_val"]["images"]} <= set(stores["val"].row)
    assert not set(stores["train"].row) & set(stores["val"].row)
    t_feat = time.perf_counter() - t1
    t2 = time.perf_counter()
    lm = M.load_qwen(cfg, device)
    vlm = M.VLM(cfg, lm, device)
    t_model = time.perf_counter() - t2
    info = {"data": data_info(cfg, d), "features": {s: stores[s].info for s in stores},
            "lm_config": {k: getattr(lm.config, k, None) for k in ("hidden_size", "num_hidden_layers",
                                                                     "attention_dropout", "vocab_size",
                                                                     "tie_word_embeddings")},
            "dtypes": {"backbone": str(next(lm.parameters()).dtype).replace("torch.", ""),
                       "connector": "float32 parameters and compute, output cast to the embedding dtype",
                       "lora": "float32", "optimizer_states": "float32",
                       "compute": "float16 autocast" if vlm.amp else "float32",
                       "features": "float16 stored, float32 into the connector"},
            "resources": {"setup_s": {"data_s": t_data, "features_s": t_feat, "model_load_s": t_model}}}
    print(f"setup: data {t_data:.1f}s features {t_feat:.1f}s model {t_model:.1f}s; "
          f"{len(d['caption_examples'])} captions, {len(d['vqa_train']['pairs'])} VQA pairs", flush=True)
    out = cfg["output"]
    dirs = {"result": out["result_dir"], "keep": out["keep_dir"], "recovery": out["recovery_dir"]}
    tr = Trainer(cfg, vlm, d, enc, stores, dirs, kind="smoke" if args.smoke else "train", resume=args.resume,
                 info=info, main_cfg=main_cfg)
    tr.run(stop_after=args.stop_after)
    print(f"finished {tr.kind}: {tr.done}/{tr.total} updates, skipped {tr.acc['skipped']}, "
          f"result {os.path.join(dirs['result'], 'train.json')}", flush=True)
    return tr


if __name__ == "__main__":
    main()
