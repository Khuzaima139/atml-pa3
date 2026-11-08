"""A tiny random Qwen2 causal LM saved with the real Qwen2.5-3B-Instruct tokenizer and chat template
(real vocabulary, hidden size 32, 2 layers), for CPU end-to-end tests of task3_ttc.live.

The checkpoint's generation_config.json copies the real one (repetition_penalty 1.05), so a test can check that
nothing leaks into the live sampling settings. The <|im_end|> output row is set so that EOS is sampled now and then,
giving both EOS-finished and truncated calls at small caps.
"""
import json
import os

from task3_ttc.live import MODEL_ID, REVISION

REAL_GENERATION_CONFIG = {"bos_token_id": 151643, "pad_token_id": 151643, "do_sample": True,
                          "eos_token_id": [151645, 151643], "repetition_penalty": 1.05, "temperature": 0.7,
                          "top_p": 0.8, "top_k": 20}


def real_tokenizer():
    """The cached real tokenizer, or None when it is not in the local HF cache."""
    from transformers import AutoTokenizer
    try:
        return AutoTokenizer.from_pretrained(MODEL_ID, revision=REVISION, local_files_only=True)
    except Exception:
        return None


def build(path, eos_scale=4.0, seed=0):
    import torch
    from transformers import Qwen2Config, Qwen2ForCausalLM
    tok = real_tokenizer()
    assert tok is not None
    torch.manual_seed(seed)
    cfg = Qwen2Config(vocab_size=151936, hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                      num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=8192,
                      tie_word_embeddings=False, bos_token_id=151643, eos_token_id=151645)
    m = Qwen2ForCausalLM(cfg)
    with torch.no_grad():
        w = m.lm_head.weight
        w.normal_(0.0, 0.02)
        u = torch.randn(cfg.hidden_size)
        w[151645] = eos_scale * u / u.norm()
    m.save_pretrained(path)
    tok.save_pretrained(path)
    with open(os.path.join(path, "generation_config.json"), "w") as f:
        json.dump(REAL_GENERATION_CONFIG, f)
    return path
