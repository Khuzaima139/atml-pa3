"""Task 1 training: Full FT, Full FT + L2-SP, LoRA-4, LoRA-16 and OPLoRA-16.

Manual: model and data L108-116, shared settings L117-127, precision L129-131, checkpoints L156-158,
steps L176-211. One config per run (configs/task1/*.yaml); one result JSON per run.

Run (CUDA only; never on the Mac):
    python -m task1_finetune.train --config configs/task1/full_ft.yaml [--resume]
    python -m task1_finetune.train --config configs/task1/full_ft.yaml --smoke_updates 3
"""
import argparse
import contextlib
import copy
import datetime
import glob
import hashlib
import math
import os
import random
import time

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch.utils.checkpoint import checkpoint

from common.results import write_result
from common.runinfo import run_info
from common.seed import seed_everything
from task1_finetune import data as D
from task1_finetune.adapters import adapter_state, export_effective, inject_adapters, leading_bases, target_linears

CHUNK = 1 << 24  # elements per chunk for penalty / displacement arithmetic
LOG_FIELDS = ("update is 1-based (the update just completed); lr is the rate applied in that update; "
              "sft_loss = summed token NLL over the 16 microbatches / scored_tokens; grad_norm_preclip is the "
              "global norm after unscale_ (and after the L2-SP gradient) and before clipping; loss_scale is the "
              "GradScaler scale used for that update's backward; skipped = the scale dropped (inf/nan grads)")


def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)


def _sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _num(x):
    x = float(x)
    return x if math.isfinite(x) else None


# ---------------------------------------------------------------- schedule and optimizer

def lr_factor(u, warmup, total):
    """LR multiplier for 0-based update u: linear warmup u/warmup, then cosine decay to 0 at `total`."""
    if u < warmup:
        return u / warmup
    return 0.5 * (1.0 + math.cos(math.pi * (u - warmup) / (total - warmup)))


def unique_trainable(model):
    """Trainable parameters, tied tensors listed once (named_parameters de-duplicates)."""
    return [(n, p) for n, p in model.named_parameters() if p.requires_grad]


def param_groups(named, weight_decay):
    """Weight decay on tensors with >= 2 dims, none on the rest (L123; the model has no biases)."""
    decay = [p for _, p in named if p.ndim >= 2]
    other = [p for _, p in named if p.ndim < 2]
    groups = [{"params": decay, "weight_decay": weight_decay}]
    if other:
        groups.append({"params": other, "weight_decay": 0.0})
    return groups


# ---------------------------------------------------------------- loss

def _ce_sum(lm_head, h, t):
    return F.cross_entropy(lm_head(h).float(), t, reduction="sum")


def scored_nll_sum(model, batch, chunk):
    """Summed next-token NLL over scored positions (FP32 CE).

    Logits are formed only at scored positions, `chunk` rows at a time, from the final hidden states with
    the (tied) lm_head; each chunk is recomputed in backward so only one chunk of logits is alive.
    """
    out = model.model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"], use_cache=False)
    h = out.last_hidden_state[:, :-1]
    tgt = batch["labels"][:, 1:]
    m = tgt != D.IGNORE
    h, tgt = h[m], tgt[m]
    total = torch.zeros((), dtype=torch.float32, device=h.device)
    for s in range(0, h.shape[0], chunk):
        total = total + checkpoint(_ce_sum, model.lm_head, h[s:s + chunk], tgt[s:s + chunk], use_reentrant=False)
    return total


# ---------------------------------------------------------------- L2-SP and displacement

def _chunks(t):
    return t.reshape(-1).split(CHUNK)


class L2SP:
    """Omega = 0.5 * sum over unique parameters of (theta - theta_0)^2 (L190-196), theta_0 frozen."""

    def __init__(self, named, lam, ref_dtype=torch.bfloat16):
        self.lam = lam
        self.params = [p for _, p in named]
        self.ref = [p.detach().to(ref_dtype).clone() for p in self.params]

    def ref_bytes(self):
        return sum(r.numel() * r.element_size() for r in self.ref)

    @torch.no_grad()
    def omega(self):
        om = torch.zeros((), dtype=torch.float64, device=self.params[0].device)
        for p, p0 in zip(self.params, self.ref):
            for pc, rc in zip(_chunks(p), _chunks(p0)):
                om += (pc - rc.float()).double().square().sum()
        return 0.5 * om.item()

    @torch.no_grad()
    def add_grad(self):
        """grad += lam * (theta - theta_0) for every parameter; returns Omega at the current theta."""
        om = torch.zeros((), dtype=torch.float64, device=self.params[0].device)
        for p, p0 in zip(self.params, self.ref):
            assert p.grad is not None
            for pc, rc, gc in zip(_chunks(p), _chunks(p0), _chunks(p.grad)):
                d = pc - rc.float()
                om += d.double().square().sum()
                gc.add_(d, alpha=self.lam)
        return 0.5 * om.item()


@torch.no_grad()
def displacement(params, refs):
    """||theta - theta_0||_2 with float64 accumulation; refs may live on another device."""
    s = torch.zeros((), dtype=torch.float64, device=params[0].device)
    for p, p0 in zip(params, refs):
        for pc, rc in zip(_chunks(p), _chunks(p0)):
            s += (pc.double() - rc.to(pc.device).double()).square().sum()
    return math.sqrt(s.item())


# ---------------------------------------------------------------- model setup

def prepare_model(cfg, model, device):
    """Turn a model holding the bf16 checkpoint weights (CPU) into the method's training model on `device`.

    Full FT variants: FP32 master weights (bf16 -> fp32 is exact). Adapter variants: frozen FP16 backbone
    (parameters only; the FP32 RoPE buffers stay FP32) plus FP32 adapters. On CPU (tests) everything is FP32.
    """
    amp = device.type == "cuda"
    info = {"overhead": {}}
    ad = cfg.get("adapter")
    l2sp = disp_ref = mods = None
    if ad is None:
        model.to(device=device, dtype=torch.float32)
        named = unique_trainable(model)
        lam = cfg["l2sp"]["lambda"]
        if lam > 0:
            _sync(device)
            t0 = time.perf_counter()
            l2sp = L2SP(named, lam, getattr(torch, cfg["l2sp"]["ref_dtype"]))
            _sync(device)
            info["overhead"].update(theta0_build_s=time.perf_counter() - t0, theta0_bytes=l2sp.ref_bytes(),
                                    theta0_dtype=cfg["l2sp"]["ref_dtype"], theta0_device=str(device),
                                    theta0_transfer="built on the training device; no per-update transfer")
            disp_ref = l2sp.ref
        elif cfg["displacement_updates"]:
            disp_ref = [p.detach().to("cpu", torch.bfloat16) for _, p in named]
        if disp_ref is not None:
            for (n, p), r in zip(named, disp_ref):
                assert torch.equal(r.to(p.device).float(), p), f"theta_0 not an exact copy for {n}"
        info["dtypes"] = {"master": "float32", "compute": "float16 autocast" if amp else "float32",
                          "optimizer_states": "float32"}
    else:
        project = ad["project"]
        bases = None
        if project:
            _sync(device)
            t0 = time.perf_counter()
            bases = {n: leading_bases(lin.weight.detach().to(device, torch.float32), ad["k"])
                     for n, lin in target_linears(model)}
            _sync(device)
            info["overhead"].update(
                svd_preprocess_s=time.perf_counter() - t0,
                basis_bytes=sum(U.numel() * U.element_size() + V.numel() * V.element_size()
                                for U, V in bases.values()),
                basis_dtype="float32", k=ad["k"])
        bb = torch.float16 if amp else torch.float32
        for p in model.parameters():
            p.requires_grad_(False)
            p.data = p.data.to(device, bb)
        model.to(device)
        mods = inject_adapters(model, ad["r"], ad["alpha"], ad["dropout"], cfg["seed"], project, bases)
        named = unique_trainable(model)
        info["dtypes"] = {"backbone": str(bb).replace("torch.", ""), "adapters": "float32",
                          "compute": "float16 autocast" if amp else "float32", "optimizer_states": "float32",
                          "effective_factors": "float32, autocast disabled" if project else None}
    gc = cfg["gradient_checkpointing"]
    if gc["enabled"]:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": gc["use_reentrant"]})
    model.config.use_cache = False
    model.train()
    info["trainable_params"] = sum(p.numel() for _, p in named)
    info["trainable_tensors"] = len(named)
    return {"model": model, "named": named, "l2sp": l2sp, "disp_ref": disp_ref, "adapters": mods, "info": info}


# ---------------------------------------------------------------- files and RNG

def sha256_file(path, block=1 << 24):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while b := f.read(block):
            h.update(b)
    return h.hexdigest()


def _fsync_replace(tmp, path):
    with open(tmp, "rb") as f:
        os.fsync(f.fileno())
    os.replace(tmp, path)


def rng_state():
    s = {"python": random.getstate(), "numpy": np.random.get_state(), "torch": torch.get_rng_state()}
    if torch.cuda.is_available():
        s["cuda"] = torch.cuda.get_rng_state_all()
    return s


def set_rng_state(s):
    random.setstate(s["python"])
    np.random.set_state(s["numpy"])
    torch.set_rng_state(s["torch"])
    if "cuda" in s:
        torch.cuda.set_rng_state_all(s["cuda"])


def _without_output(cfg):
    c = copy.deepcopy(cfg)
    c.pop("output", None)
    return c


# ---------------------------------------------------------------- trainer

class Trainer:
    def __init__(self, cfg, setup, examples, device, result_path, recovery_dir, keep_dir, pairs=None,
                 run_updates=None, kind="train", resume=False, data_info=None):
        self.cfg, self.s, self.examples, self.device = cfg, setup, examples, device
        self.model = setup["model"]
        self.named = setup["named"]
        self.params = [p for _, p in self.named]
        self.result_path, self.recovery_dir, self.keep_dir = result_path, recovery_dir, keep_dir
        self.kind = kind
        self.data_info = data_info or {}
        b = cfg["batch"]
        self.accum, self.updates = b["grad_accum"], b["updates"]
        self.pairs = pairs if pairs is not None else D.consecutive_pairs(len(examples), b["microbatch"])
        self.run_updates = run_updates or self.updates
        assert len(self.pairs) >= self.run_updates * self.accum
        self.chunk = cfg["loss"]["logit_chunk"]
        self.amp = device.type == "cuda"
        o, sc = cfg["optimizer"], cfg["schedule"]
        extra = {"fused": True} if self.amp and o["fused_on_cuda"] else {}
        self.opt = torch.optim.AdamW(param_groups(self.named, o["weight_decay"]), lr=sc["peak_lr"],
                                     betas=tuple(o["betas"]), eps=o["eps"], **extra)
        w, total = sc["warmup_updates"], self.updates
        self.sched = torch.optim.lr_scheduler.LambdaLR(self.opt, lambda u: lr_factor(u, w, total))
        self.scaler = torch.amp.GradScaler(device.type, enabled=self.amp,
                                           init_scale=cfg["precision"]["init_scale"])
        self.next_update = 0
        self.log, self.files, self.displacement, self.segments = [], [], {}, []
        self.acc = {"train_s": 0.0, "ckpt_io_s": 0.0, "displacement_s": 0.0, "wall_s_prev": 0.0,
                    "skipped": 0, "peak_allocated_bytes": 0, "peak_reserved_bytes": 0}
        stamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        self.run_id = f"{kind}_{cfg['method']}_{stamp}"
        if resume:
            self._resume()
        self.segments.append({"from_update": self.next_update, "provenance": run_info()})
        self.t_start = time.time()

    # -- one optimizer update
    def _autocast(self):
        return torch.autocast("cuda", dtype=torch.float16) if self.amp else contextlib.nullcontext()

    def microbatches(self, u):
        ps = self.pairs[u * self.accum:(u + 1) * self.accum]
        return [D.collate([self.examples[i] for i in p]) for p in ps]

    def accumulate(self, mbs):
        """Backward of sum_mb NLL_mb / N, N = scored tokens of the whole effective batch (counted first)."""
        n = sum(D.n_scored(b) for b in mbs)
        tot = torch.zeros((), dtype=torch.float64, device=self.device)
        for b in mbs:
            b = {k: v.to(self.device, non_blocking=True) for k, v in b.items()}
            with self._autocast():
                s = scored_nll_sum(self.model, b, self.chunk)
            self.scaler.scale(s / n).backward()
            tot += s.detach().double()
        return tot.item() / n, n

    def finish_grads(self):
        """unscale_, then the L2-SP gradient, then global-norm clipping; returns pre-clip norm."""
        self.scaler.unscale_(self.opt)
        omega = pen_s = None
        if self.s["l2sp"] is not None:
            _sync(self.device)
            t0 = time.perf_counter()
            omega = self.s["l2sp"].add_grad()
            _sync(self.device)
            pen_s = time.perf_counter() - t0
        gn = torch.nn.utils.clip_grad_norm_(self.params, self.cfg["clip_grad_norm"])
        return float(gn), omega, pen_s

    def train_update(self, u):
        _sync(self.device)
        t0 = time.perf_counter()
        lr = self.opt.param_groups[0]["lr"]
        mbs = self.microbatches(u)
        loss, n = self.accumulate(mbs)
        gn, omega, pen_s = self.finish_grads()
        scale = self.scaler.get_scale()
        self.scaler.step(self.opt)
        self.scaler.update()
        new_scale = self.scaler.get_scale()
        skipped = new_scale < scale
        self.opt.zero_grad(set_to_none=True)
        self.sched.step()
        _sync(self.device)
        rec = {"update": u + 1, "sft_loss": _num(loss), "lr": lr, "grad_norm_preclip": _num(gn),
               "loss_scale": scale, "skipped": bool(skipped), "scored_tokens": n,
               "padded_tokens": sum(b["input_ids"].numel() for b in mbs), "time_s": time.perf_counter() - t0}
        if self.s["l2sp"] is not None:
            lam = self.s["l2sp"].lam
            rec.update(omega=omega, lambda_omega=lam * omega,
                       total_loss=None if rec["sft_loss"] is None else rec["sft_loss"] + lam * omega,
                       penalty_s=pen_s)
        if skipped:
            self.acc["skipped"] += 1
            print(f"SKIPPED update {u + 1}: loss scale {scale:g} -> {new_scale:g} (inf/nan gradients)", flush=True)
        return rec

    # -- loop
    def run(self, stop_after=None):
        self._write("running")
        ck = self.cfg["checkpoints"]
        for u in range(self.next_update, self.run_updates):
            rec = self.train_update(u)
            self.log.append(rec)
            self.acc["train_s"] += rec["time_s"]
            done = self.next_update = u + 1
            at_ck = False
            if done in self.cfg["displacement_updates"]:
                self._displacement(done)
                at_ck = True
            if done in ck["keep_weights"]:
                self._save_kept(done)
                at_ck = True
            if done in ck["recovery"]:
                self._save_recovery(done)
                at_ck = True
            if done % self.cfg["logging"]["json_every"] == 0 or at_ck or done == 1:
                self._write("running")
                r = rec
                print(f"update {done}/{self.run_updates} loss {r['sft_loss']} lr {r['lr']:.3e} "
                      f"gn {r['grad_norm_preclip']} scale {r['loss_scale']:g} skipped_total {self.acc['skipped']} "
                      f"{r['time_s']:.2f}s", flush=True)
            if stop_after is not None and done >= stop_after:
                self._write("stopped")
                return
        self._write("done")

    def _displacement(self, done):
        _sync(self.device)
        t0 = time.perf_counter()
        v = displacement(self.params, self.s["disp_ref"])
        self.acc["displacement_s"] += time.perf_counter() - t0
        self.displacement[str(done)] = v
        print(f"||theta - theta_0||_2 at update {done}: {v:.6f}", flush=True)

    # -- files
    def _record(self, path, kind, done, **extra):
        e = {"path": path, "kind": kind, "update": done, "bytes": os.path.getsize(path),
             "sha256": sha256_file(path), **extra}
        self.files.append(e)
        return e

    def _save_tensors(self, tensors, path, metadata):
        from safetensors.torch import save_file
        tmp = path + ".tmp"
        save_file(tensors, tmp, metadata=metadata)
        _fsync_replace(tmp, path)

    def _save_kept(self, done):
        os.makedirs(self.keep_dir, exist_ok=True)
        t0 = time.perf_counter()
        meth, ad = self.cfg["method"], self.cfg.get("adapter")
        meta = {"run_id": self.run_id, "method": meth, "update": str(done)}
        if ad is None:
            path = os.path.join(self.keep_dir, f"{meth}_weights_u{done:04d}.safetensors")
            tensors = {n: p.detach().float().cpu().contiguous() for n, p in self.named}
            self._save_tensors(tensors, path, {**meta, "note": "unique parameters; lm_head.weight is tied to "
                                                               "model.embed_tokens.weight"})
            self._record(path, "model_weights_fp32", done)
        else:
            meta.update(r=str(ad["r"]), alpha=str(ad["alpha"]), targets=",".join(ad["targets"]))
            path = os.path.join(self.keep_dir, f"{meth}_adapter_u{done:04d}.safetensors")
            self._save_tensors(adapter_state(self.s["adapters"]), path, {**meta, "factors": "A, B (unprojected)"})
            self._record(path, "adapter_fp32", done)
            if ad["project"]:
                _sync(self.device)
                t1 = time.perf_counter()
                eff = export_effective(self.s["adapters"])
                _sync(self.device)
                t_compute = time.perf_counter() - t1
                path = os.path.join(self.keep_dir, f"{meth}_adapter_eff_u{done:04d}.safetensors")
                self._save_tensors(eff, path, {**meta, "k": str(ad["k"]), "factors": "A_eff, B_eff (projected)"})
                self.s["info"]["overhead"].update(export_compute_s=t_compute,
                                                  export_s=time.perf_counter() - t1)
                self._record(path, "adapter_effective_fp32", done)
        self.acc["ckpt_io_s"] += time.perf_counter() - t0

    def _save_recovery(self, done):
        os.makedirs(self.recovery_dir, exist_ok=True)
        t0 = time.perf_counter()
        state = {"format": 1, "run_id": self.run_id, "method": self.cfg["method"], "next_update": done,
                 "config": _without_output(self.cfg),
                 "trainable": {n: p.detach() for n, p in self.named},
                 "optimizer": self.opt.state_dict(), "scheduler": self.sched.state_dict(),
                 "scaler": self.scaler.state_dict(), "rng": rng_state(), "log": self.log,
                 "acc": {**self.acc, "wall_s_prev": self.acc["wall_s_prev"] + time.time() - self.t_start},
                 "files": self.files, "displacement": self.displacement, "segments": self.segments,
                 "overhead": self.s["info"]["overhead"]}
        if self.s["adapters"] is not None and self.cfg["adapter"]["project"]:
            state["bases"] = {n: (m.U_k, m.V_k) for n, m in self.s["adapters"].items()}
        path = os.path.join(self.recovery_dir, f"recovery_u{done:04d}.pt")
        torch.save(state, path + ".tmp")
        _fsync_replace(path + ".tmp", path)
        for old in glob.glob(os.path.join(self.recovery_dir, "recovery_u*.pt")):
            if os.path.abspath(old) != os.path.abspath(path):
                os.remove(old)
                for e in self.files:
                    if e["kind"] == "recovery" and os.path.abspath(e["path"]) == os.path.abspath(old):
                        e["deleted_at_update"] = done
        self._record(path, "recovery", done)
        self.acc["ckpt_io_s"] += time.perf_counter() - t0

    def _resume(self):
        files = sorted(glob.glob(os.path.join(self.recovery_dir, "recovery_u*.pt")))
        if not files:
            raise SystemExit(f"--resume: no recovery state in {self.recovery_dir}")
        path = files[-1]
        st = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
        if st["config"] != _without_output(self.cfg):
            raise SystemExit(f"--resume: config differs from the one stored in {path}")
        with torch.no_grad():
            for n, p in self.named:
                p.copy_(st["trainable"][n])
            if "bases" in st:
                for n, m in self.s["adapters"].items():
                    m.U_k.copy_(st["bases"][n][0])
                    m.V_k.copy_(st["bases"][n][1])
        self.opt.load_state_dict(st["optimizer"])
        self.sched.load_state_dict(st["scheduler"])
        self.scaler.load_state_dict(st["scaler"])
        self.run_id, self.next_update = st["run_id"], st["next_update"]
        self.log, self.acc, self.files = st["log"], st["acc"], st["files"]
        self.displacement, self.segments = st["displacement"], st["segments"]
        self.s["info"]["overhead"] = st["overhead"]
        if not any(e["kind"] == "recovery" and os.path.abspath(e["path"]) == os.path.abspath(path)
                   for e in self.files):
            self._record(path, "recovery", self.next_update, recorded_on_resume=True)
        print(f"resumed from {path} at update {self.next_update}", flush=True)
        set_rng_state(st["rng"])  # last: nothing after this may consume random numbers

    # -- result JSON
    def _resources(self):
        if self.device.type == "cuda":
            self.acc["peak_allocated_bytes"] = max(self.acc["peak_allocated_bytes"],
                                                   torch.cuda.max_memory_allocated(self.device))
            self.acc["peak_reserved_bytes"] = max(self.acc["peak_reserved_bytes"],
                                                  torch.cuda.max_memory_reserved(self.device))
        info = self.s["info"]
        return {"trainable_params": info["trainable_params"], "trainable_tensors": info["trainable_tensors"],
                "updates_done": self.next_update, "skipped_updates": self.acc["skipped"],
                "train_s": self.acc["train_s"], "ckpt_io_s": self.acc["ckpt_io_s"],
                "displacement_s": self.acc["displacement_s"],
                "wall_s": self.acc["wall_s_prev"] + time.time() - self.t_start,
                "setup_s": info.get("setup_s"),
                "peak_allocated_bytes": self.acc["peak_allocated_bytes"],
                "peak_reserved_bytes": self.acc["peak_reserved_bytes"],
                "checkpoint_bytes": {e["kind"]: e["bytes"] for e in self.files}}

    def _smoke_summary(self):
        steady = self.log[1:] or self.log
        t = [r["time_s"] for r in steady]
        tok = [r["padded_tokens"] for r in steady]
        epoch_tok = self.data_info.get("padded_tokens_epoch")
        return {"time_per_update_s": [r["time_s"] for r in self.log],
                "padded_tokens_per_update": [r["padded_tokens"] for r in self.log],
                "skipped_updates": self.acc["skipped"],
                "projected_full_run_s_by_updates": float(np.mean(t)) * self.updates,
                "projected_full_run_s_by_tokens": (sum(t) / sum(tok) * epoch_tok) if epoch_tok else None,
                "projection_note": "updates after the first; excludes checkpoint I/O and setup"}

    def _write(self, status):
        doc = {"run_id": self.run_id, "kind": self.kind, "status": status, "method": self.cfg["method"],
               "config": self.cfg, "segments": self.segments, "dtypes": self.s["info"]["dtypes"],
               "data": self.data_info, "overhead": self.s["info"]["overhead"], "resources": self._resources(),
               "displacement_l2": self.displacement, "files": self.files, "generation": None,
               "log_fields": LOG_FIELDS, "log": self.log}
        if self.kind == "smoke":
            doc["smoke"] = self._smoke_summary()
        write_result(self.result_path, doc)


def _smoke_pairs(examples, pairs, n_updates, accum):
    """Longest microbatch first, then the epoch's microbatches in order (L117-127 recipe unchanged)."""
    lens = [max(len(examples[i]["input_ids"]) for i in p) for p in pairs]
    j = int(np.argmax(lens))
    rest = [p for i, p in enumerate(pairs) if i != j]
    return [pairs[j]] + rest[:n_updates * accum - 1], lens[j]


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--smoke_updates", type=int, default=0)
    ap.add_argument("--recovery_dir")
    ap.add_argument("--keep_dir")
    args = ap.parse_args(argv)
    cfg = load_config(args.config)
    if args.recovery_dir:
        cfg["output"]["recovery_dir"] = args.recovery_dir
    if args.keep_dir:
        cfg["output"]["keep_dir"] = args.keep_dir
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required (training never runs on the Mac)")
    device = torch.device("cuda")
    seed_everything(cfg["seed"])
    from transformers import AutoModelForCausalLM

    t0 = time.perf_counter()
    m = cfg["model"]
    tok = D.load_tokenizer(m["revision"])
    examples, meta = D.load_train_examples(tok, cfg["data"]["train_ids"])
    t_data = time.perf_counter() - t0
    pairs = D.consecutive_pairs(len(examples), cfg["batch"]["microbatch"])
    data_info = {"train_ids": cfg["data"]["train_ids"], "count": len(examples), "first_index": meta["first_index"],
                 "last_index": meta["last_index"], "token_fingerprint": D.token_fingerprint(examples),
                 "microbatches": len(pairs),
                 "padded_tokens_epoch": sum(len(p) * max(len(examples[i]["input_ids"]) for i in p) for p in pairs),
                 "longest_example": max(len(e["input_ids"]) for e in examples)}
    t1 = time.perf_counter()
    model = AutoModelForCausalLM.from_pretrained(m["name"], revision=m["revision"], dtype=torch.bfloat16)
    t_load = time.perf_counter() - t1
    t2 = time.perf_counter()
    setup = prepare_model(cfg, model, device)
    t_prep = time.perf_counter() - t2
    setup["info"]["setup_s"] = {"data_s": t_data, "model_load_s": t_load, "prepare_s": t_prep}
    print(f"method {cfg['method']}: trainable params {setup['info']['trainable_params']:,}, "
          f"overhead {setup['info']['overhead']}, setup {setup['info']['setup_s']}", flush=True)
    out = cfg["output"]
    kind, run_updates, result = "train", None, out["result"]
    if args.smoke_updates:
        kind, run_updates, result = "smoke", args.smoke_updates, out["smoke_result"]
        pairs, longest = _smoke_pairs(examples, pairs, args.smoke_updates, cfg["batch"]["grad_accum"])
        data_info["smoke_first_microbatch_len"] = longest
        cfg = copy.deepcopy(cfg)
        cfg["displacement_updates"] = []
        cfg["checkpoints"] = {"recovery": [], "keep_weights": []}
    seed_everything(cfg["seed"])
    tr = Trainer(cfg, setup, examples, device, result, out["recovery_dir"], out["keep_dir"], pairs=pairs,
                 run_updates=run_updates, kind=kind, resume=args.resume, data_info=data_info)
    tr.run()
    print(f"finished {kind} {cfg['method']}: {tr.next_update} updates, skipped {tr.acc['skipped']}, "
          f"result {result}", flush=True)


if __name__ == "__main__":
    main()
