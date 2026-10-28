"""LoRA and OPLoRA adapters for Task 1 (manual L80-90, L197-211).

One module for both: project=False is LoRA, project=True is OPLoRA.
    W_eff x = W0 x + (alpha / r) * B_eff (A_eff (dropout(x)))
LoRA: A_eff = A, B_eff = B.  OPLoRA: B_eff = B - U_k (U_k^T B), A_eff = A - (A V_k) V_k^T with the frozen
leading-k singular bases of W0 (computed once in FP32); no dense projection matrix is formed.
A is initialised as peft does (kaiming_uniform_, a=sqrt(5)) from a generator seeded by (seed, module name);
B starts at zero. Adapter parameters are FP32.
"""
import hashlib
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

TARGETS = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")


def module_seed(seed, name):
    """Deterministic 63-bit seed from (seed, module name)."""
    d = hashlib.sha256(f"{seed}:{name}".encode()).digest()
    return int.from_bytes(d[:8], "little") & (2**63 - 1)


def target_linears(model, targets=TARGETS):
    """(name, nn.Linear) for every target projection in every decoder layer, in module order."""
    return [(n, m) for n, m in model.named_modules()
            if isinstance(m, nn.Linear) and ".layers." in n and n.rsplit(".", 1)[-1] in targets]


def leading_bases(W, k):
    """Leading-k left/right singular vectors of W (out, in): U_k (out, k), V_k (in, k), FP32."""
    U, _, Vh = torch.linalg.svd(W.to(torch.float32), full_matrices=False)
    return U[:, :k].contiguous(), Vh[:k].T.contiguous()


class AdapterLinear(nn.Module):
    def __init__(self, base, name, r, alpha, dropout, seed, project=False, U_k=None, V_k=None):
        super().__init__()
        self.base = base
        self.name = name
        self.r, self.alpha, self.scaling = r, alpha, alpha / r
        self.project = project
        self.dropout = nn.Dropout(dropout)
        dev = base.weight.device
        A = torch.empty(r, base.in_features, dtype=torch.float32)
        g = torch.Generator().manual_seed(module_seed(seed, name))
        nn.init.kaiming_uniform_(A, a=math.sqrt(5), generator=g)
        self.lora_A = nn.Parameter(A.to(dev))
        self.lora_B = nn.Parameter(torch.zeros(base.out_features, r, dtype=torch.float32, device=dev))
        if project:
            self.register_buffer("U_k", U_k.to(dev, torch.float32), persistent=False)
            self.register_buffer("V_k", V_k.to(dev, torch.float32), persistent=False)

    def effective_factors(self):
        if not self.project:
            return self.lora_A, self.lora_B
        with torch.autocast(device_type=self.lora_A.device.type, enabled=False):
            B = self.lora_B - self.U_k @ (self.U_k.T @ self.lora_B)
            A = self.lora_A - (self.lora_A @ self.V_k) @ self.V_k.T
        return A, B

    def delta_weight(self):
        """Dense (alpha/r) B_eff A_eff in FP32 (for tests and diagnostics, not used in training)."""
        A, B = self.effective_factors()
        return self.scaling * (B.float() @ A.float())

    def forward(self, x):
        out = self.base(x)
        A, B = self.effective_factors()
        delta = F.linear(F.linear(self.dropout(x.to(A.dtype)), A), B) * self.scaling
        return out + delta.to(out.dtype)


def inject_adapters(model, r, alpha, dropout, seed, project=False, bases=None, targets=TARGETS):
    """Replace every target nn.Linear with an AdapterLinear in place; returns {name: module}."""
    mods = {}
    for name, lin in target_linears(model, targets):
        U_k, V_k = bases[name] if project else (None, None)
        m = AdapterLinear(lin, name, r, alpha, dropout, seed, project, U_k, V_k)
        parent_name, child = name.rsplit(".", 1)
        setattr(model.get_submodule(parent_name), child, m)
        mods[name] = m
    return mods


def adapter_state(mods):
    """Trainable factors A, B (FP32, CPU)."""
    out = {}
    for n, m in mods.items():
        out[f"{n}.lora_A"] = m.lora_A.detach().float().cpu().contiguous()
        out[f"{n}.lora_B"] = m.lora_B.detach().float().cpu().contiguous()
    return out


@torch.no_grad()
def export_effective(mods):
    """Projected factors A_eff, B_eff (FP32, CPU) for inference and diagnostics."""
    out = {}
    for n, m in mods.items():
        A, B = m.effective_factors()
        out[f"{n}.lora_A_eff"] = A.detach().float().cpu().contiguous()
        out[f"{n}.lora_B_eff"] = B.detach().float().cpu().contiguous()
    return out
