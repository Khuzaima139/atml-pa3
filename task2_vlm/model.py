"""SigLIP-feature -> Qwen3 VLM (manual L308-355, L461-476): connector, visual-token insertion, LoRA, loss, decoding.

- Qwen3-0.6B, all original parameters frozen, FP16 on CUDA (FP32 on CPU: tests only). Dropout settings as
  loaded (attention_dropout 0.0); LoRA dropout is active in training mode only.
- Connector Linear(768,1024)-GELU-Linear(1024,1024), FP32 parameters and FP32 compute (autocast off), times the
  calibration scale s (buffer, D12), cast to the embedding dtype, then scattered into the visual positions.
- From Stage 2: peft LoRA r=16, alpha=32, dropout 0.05, q/k/v/o in all layers, FP32 adapter weights.
- Gradient checkpointing (non-reentrant) is active in training mode; the Qwen forward is never under no_grad in
  training, so gradients reach the connector through the frozen backbone (L501-502).
- Loss: logits only at supervised positions, `chunk` rows at a time, CE in FP32 (one row per supervised token).
"""
import contextlib

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from task2_vlm.data import IGNORE


class Connector(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, out_dim)
        self.register_buffer("scale", torch.ones((), dtype=torch.float32))

    def raw(self, x):
        return self.fc2(self.act(self.fc1(x)))

    def forward(self, x):
        return self.raw(x) * self.scale


def build_connector(cfg):
    c = cfg["connector"]
    torch.manual_seed(c["init_seed"])
    return Connector(c["in_dim"], c["hidden_dim"], c["out_dim"]).float()


def load_qwen(cfg, device):
    """The frozen language model (replaced by a tiny random Qwen3 in the CPU tests)."""
    from transformers import AutoModelForCausalLM
    m = cfg["model"]
    dtype = getattr(torch, m["dtype"]) if device.type == "cuda" else torch.float32
    return AutoModelForCausalLM.from_pretrained(m["name"], revision=m["revision"], dtype=dtype,
                                                attn_implementation=m["attn_implementation"])


def _ce_rows(lm_head, h, t):
    return F.cross_entropy(lm_head(h).float(), t, reduction="none")


def example_means(nll, rows, sel, n_examples):
    """Mean of nll[sel] per example (rows = example index of each supervised token) and token counts."""
    s = torch.zeros(n_examples, dtype=torch.float32, device=nll.device).index_add(0, rows[sel], nll[sel])
    n = torch.zeros(n_examples, dtype=torch.float32, device=nll.device).index_add(
        0, rows[sel], torch.ones_like(nll[sel]))
    return s / n, n


class VLM:
    def __init__(self, cfg, lm, device):
        self.cfg, self.lm, self.device = cfg, lm, device
        self.amp = device.type == "cuda"
        for p in lm.parameters():
            p.requires_grad_(False)
        lm.to(device)
        gc = cfg["gradient_checkpointing"]
        if gc["enabled"]:
            lm.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": gc["use_reentrant"]})
        lm.config.use_cache = False
        self.connector = build_connector(cfg).to(device)
        self.peft = None
        self.lora_names = []

    # -- parameters
    def add_lora(self):
        from peft import LoraConfig, get_peft_model
        c = self.cfg["lora"]
        torch.manual_seed(c["init_seed"])
        pc = LoraConfig(r=c["r"], lora_alpha=c["alpha"], lora_dropout=c["dropout"], bias=c["bias"],
                        target_modules=list(c["targets"]))
        self.peft = get_peft_model(self.lm, pc)
        for n, p in self.lm.named_parameters():
            if p.requires_grad:
                p.data = p.data.float()  # FP32 adapter weights (L399)
        self.lora_names = [n for n, p in self.lm.named_parameters() if p.requires_grad]
        assert self.lora_names and all("lora_A" in n or "lora_B" in n for n in self.lora_names)

    def named_trainable(self):
        return ([("connector." + n, p) for n, p in self.connector.named_parameters()]
                + [(n, p) for n, p in self.lm.named_parameters() if p.requires_grad])

    def lora_state(self):
        return {n: p.detach().float().cpu().contiguous() for n, p in self.lm.named_parameters() if p.requires_grad}

    def load_lora_state(self, state):
        params = dict(self.lm.named_parameters())
        assert set(state) == set(self.lora_names), "LoRA tensor names differ"
        with torch.no_grad():
            for n, t in state.items():
                params[n].copy_(t)

    def connector_state(self):
        return {k: v.detach().float().cpu().contiguous() for k, v in self.connector.state_dict().items()}

    def train(self):
        self.lm.train()
        self.connector.train()

    def eval(self):
        self.lm.eval()
        self.connector.eval()

    def autocast(self):
        return torch.autocast("cuda", dtype=torch.float16) if self.amp else contextlib.nullcontext()

    # -- forward
    def embed(self, batch, feats):
        emb = self.lm.get_input_embeddings()(batch["input_ids"])
        vm = batch["vis_mask"]
        n = int(vm.sum())
        if n:
            assert feats is not None and feats.shape[0] * feats.shape[1] == n, "visual positions != features"
            with torch.autocast(self.device.type, enabled=False):
                v = self.connector(feats.float())
            emb = emb.masked_scatter(vm.unsqueeze(-1), v.to(emb.dtype))
        return emb

    def hidden(self, batch, feats):
        # peft injects the adapters into self.lm in place, so this path includes LoRA from Stage 2
        out = self.lm.model(inputs_embeds=self.embed(batch, feats), attention_mask=batch["attention_mask"],
                            position_ids=batch["position_ids"], use_cache=False)
        return out.last_hidden_state

    def lm_head(self):
        return self.lm.get_output_embeddings()

    def token_nll(self, batch, feats, chunk):
        """Next-token NLL (FP32) at every supervised position; returns nll, example row, content flag."""
        h = self.hidden(batch, feats)[:, :-1]
        lab = batch["labels"][:, 1:]
        m = lab != IGNORE
        rows = m.nonzero()[:, 0]
        hs, ts = h[m], lab[m]
        head = self.lm_head()
        parts = []
        for s in range(0, hs.shape[0], chunk):
            if torch.is_grad_enabled():
                parts.append(checkpoint(_ce_rows, head, hs[s:s + chunk], ts[s:s + chunk], use_reentrant=False))
            else:
                parts.append(_ce_rows(head, hs[s:s + chunk], ts[s:s + chunk]))
        nll = torch.cat(parts) if parts else hs.new_zeros(0, dtype=torch.float32)
        return nll, rows, batch["content"][:, 1:][m]

    # -- greedy decoding (left padding, explicit positions, KV cache)
    @torch.no_grad()
    def greedy(self, batch, feats, max_new_tokens, stop_id, pad_id):
        from transformers import DynamicCache
        emb = self.embed(batch, feats)
        attn, pos = batch["attention_mask"], batch["position_ids"]
        B = emb.shape[0]
        cache = DynamicCache()
        out = self.lm(inputs_embeds=emb, attention_mask=attn, position_ids=pos, past_key_values=cache,
                      use_cache=True, logits_to_keep=1)
        assert torch.isfinite(out.logits).all(), "non-finite logits after the left-padded prefill"
        nxt_pos = pos[:, -1:] + 1
        done = torch.zeros(B, dtype=torch.bool, device=emb.device)
        steps = []
        for _ in range(max_new_tokens):
            tok = out.logits[:, -1].float().argmax(-1)
            tok = torch.where(done, torch.full_like(tok, pad_id), tok)
            steps.append(tok)
            done = done | (tok == stop_id)
            if bool(done.all()) or len(steps) == max_new_tokens:
                break
            attn = torch.cat([attn, torch.ones_like(attn[:, :1])], dim=1)
            out = self.lm(inputs_embeds=self.lm.get_input_embeddings()(tok[:, None]), attention_mask=attn,
                          position_ids=nxt_pos, past_key_values=out.past_key_values, use_cache=True)
            nxt_pos = nxt_pos + 1
        gen = torch.stack(steps, dim=1).tolist()
        res = []
        for row in gen:
            stopped = stop_id in row
            res.append({"token_ids": row[:row.index(stop_id)] if stopped else row, "stopped": stopped,
                        "n_new_tokens": row.index(stop_id) + 1 if stopped else len(row)})
        return res
