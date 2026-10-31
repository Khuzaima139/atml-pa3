"""HFLM subclass that only records (manual L184-186, L212-215): per GSM8K request the number of generated
tokens and how generation ended; per request type the time spent. Generation is unchanged.

Stop reasons: "eos" (a model EOS id, which includes <|im_end|>), "stop_sequence" (a stock `until` string that is
not an EOS token), "cap" (neither within max_gen_toks). truncated = stop reason "cap". n_generated_tokens counts up
to and including the terminating token, or the token completing the stop string. lm-eval only stops a batch when
every row is done, so a row may run past its own stop: the reason is taken from the first stop in that row.

Records are joined to requests by identity: each request's context is a str subclass carrying the request's
index, so the join survives lm-eval's sorting and identical contexts.
"""
import bisect
import time

from lm_eval.models.huggingface import HFLM

SETTINGS = ("do_sample", "temperature", "top_p", "top_k", "repetition_penalty", "num_beams", "eos_token_id",
            "pad_token_id", "max_new_tokens", "max_length", "use_cache")


def effective_generation(model, **kwargs):
    """The settings transformers resolves for a generate call (model.generation_config overridden by kwargs)."""
    gc = model._prepare_generation_config(None, **kwargs)[0]
    return {k: getattr(gc, k, None) for k in SETTINGS}


class _Context(str):
    request_index = None


class _Request:
    def __init__(self, index, args):
        context = _Context(args[0])
        context.request_index = index
        self.args = (context, *args[1:])


def as_list(x):
    return list(x) if isinstance(x, (list, tuple)) else [x]


def end_of_generation(tokens, eos_ids, stops, cap, decode):
    """(n_generated_tokens, stop_reason) of one row's generated tokens (see module docstring)."""
    eos_at = next((i for i, t in enumerate(tokens) if t in eos_ids), None)
    body = tokens if eos_at is None else tokens[:eos_at]

    def stopped(n):
        text = decode(body[:n])
        return any(s in text for s in stops)

    if stops and stopped(len(body)):
        return bisect.bisect_left(range(len(body) + 1), True, key=stopped), "stop_sequence"
    if eos_at is not None:
        return eos_at + 1, "eos"
    assert len(tokens) == cap, f"row ended at {len(tokens)} tokens with no stop and cap {cap}"
    return cap, "cap"


class TrackingHFLM(HFLM):
    def __init__(self, *args, expected_doc0=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.expected_doc0 = expected_doc0  # rendered GSM8K doc 0, asserted before any generation
        self.gen_records = []               # one per generate_until request, in request order
        self.gen_settings = None
        self.timing = {}
        self._batch_tags = None
        self._by_index = None

    def _timed(self, kind, n_requests, fn):
        t0 = time.perf_counter()
        res = fn()
        t = self.timing.setdefault(kind, {"requests": 0, "seconds": 0.0})
        t["requests"] += n_requests
        t["seconds"] += time.perf_counter() - t0
        return res

    def loglikelihood(self, requests, disable_tqdm=False):
        return self._timed("loglikelihood", len(requests), lambda: super(TrackingHFLM, self).loglikelihood(
            requests, disable_tqdm=disable_tqdm))

    def generate_until(self, requests, disable_tqdm=False):
        if self.expected_doc0 is not None:
            first = next((r for r in requests if r.doc_id == 0), None)
            assert first is not None and first.args[0] == self.expected_doc0, \
                "rendered GSM8K doc 0 differs from the non-thinking chat template of doc_to_text(doc 0)"
        self._by_index = {}
        tagged = [_Request(i, r.args) for i, r in enumerate(requests)]
        res = self._timed("generate_until", len(requests), lambda: super(TrackingHFLM, self).generate_until(
            tagged, disable_tqdm=disable_tqdm))
        assert len(res) == len(requests) and sorted(self._by_index) == list(range(len(requests)))
        for i, (r, s) in enumerate(zip(requests, res, strict=True)):
            self.gen_records.append({"doc_id": r.doc_id, "context": r.args[0], "response": s, **self._by_index[i]})
        return res

    def tok_batch_encode(self, strings, *args, **kwargs):
        self._batch_tags = [getattr(s, "request_index", None) for s in strings]
        return super().tok_batch_encode(strings, *args, **kwargs)

    def _model_generate(self, context, max_length, stop, **generation_kwargs):
        cap = max_length - context.shape[1]
        kw = {k: v for k, v in generation_kwargs.items() if k != "attention_mask"}
        if kw.get("do_sample") is False and not kw.get("temperature"):  # what HFLM passes to generate
            kw.pop("temperature", None)
        out = super()._model_generate(context, max_length, stop, **generation_kwargs)
        eos_ids = set(as_list(self.model.generation_config.eos_token_id))
        eos_text = {self.tokenizer.decode([i], skip_special_tokens=False) for i in eos_ids}
        stops = [s for s in stop if s and s not in eos_text]
        decode = lambda ids: self.tokenizer.decode(ids, skip_special_tokens=False)  # noqa: E731
        generated = out[:, context.shape[1]:].tolist()
        for tag, row in zip(self._batch_tags, generated, strict=True):
            n, reason = end_of_generation(row, eos_ids, stops, cap, decode)
            self._by_index[tag] = {"n_generated_tokens": n, "stop_reason": reason, "truncated": reason == "cap"}
        t = self.timing.setdefault("generate_steps", {"batches": 0, "steps": 0})
        t["batches"] += 1
        t["steps"] += len(generated[0])
        if self.gen_settings is None:
            self.gen_settings = {"cap_max_gen_toks": cap, "stop": list(stop), "generate_kwargs": dict(kw),
                                 "effective": effective_generation(self.model, max_length=max_length,
                                                                   pad_token_id=self.tokenizer.pad_token_id,
                                                                   use_cache=True, **kw)}
        return out
