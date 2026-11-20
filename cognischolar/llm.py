"""LLM backends.

Two interchangeable backends share one interface, ``complete(prompt, meta) -> str``:

* ``OpenAICompatBackend`` – any OpenAI-compatible chat endpoint (OpenAI,
  Ollama ``http://localhost:11434/v1``, vLLM, LM Studio, Together ...).
  Uses only the prompt text; ``meta`` is ignored.

* ``SimulatedBackend`` – a seeded, offline *stochastic LLM surrogate* used for
  reproducible experiments and for the demo when no API key is available. It
  reads the structured ``meta`` the agent attaches to every call and answers
  with the same JSON a real model is asked to produce. Its behaviour is
  governed by a small set of explicit parameters (see ``SimParams``):
  how often it ranks a tempting-but-wrong strategy first, how often a sampled
  interpretation drifts away from the committed strategy, and how often it
  misreads evidence. Token usage is counted from the real prompt text.

All calls are metered (tokens in/out, number of calls, latency) in ``Meter``.
"""
from __future__ import annotations

import json
import os
import random
import re
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Optional


def count_tokens(text: str) -> int:
    """Cheap, model-agnostic token estimate (~0.75 words per token)."""
    return max(1, int(len(re.findall(r"\S+", text)) * 1.33))


@dataclass
class Meter:
    calls: int = 0
    tokens_in: int = 0
    tokens_out: int = 0
    latency_s: float = 0.0
    by_phase: Dict[str, int] = field(default_factory=dict)

    def add(self, phase: str, tin: int, tout: int, lat: float):
        self.calls += 1
        self.tokens_in += tin
        self.tokens_out += tout
        self.latency_s += lat
        self.by_phase[phase] = self.by_phase.get(phase, 0) + tin + tout

    @property
    def tokens(self):
        return self.tokens_in + self.tokens_out

    def snapshot(self):
        return dict(calls=self.calls, tokens=self.tokens, latency_s=round(self.latency_s, 4))


class LLMBackend:
    name = "base"

    def __init__(self):
        self.meter = Meter()

    def _complete(self, prompt: str, meta: Dict[str, Any], temperature: float) -> str:
        raise NotImplementedError

    def complete(self, prompt: str, meta: Optional[Dict[str, Any]] = None,
                 temperature: float = 0.7, phase: str = "other") -> str:
        meta = meta or {}
        t0 = time.perf_counter()
        out = self._complete(prompt, meta, temperature)
        lat = time.perf_counter() - t0
        if isinstance(self, SimulatedBackend):
            lat += self.sim_latency(prompt, out)
        self.meter.add(phase, count_tokens(prompt), count_tokens(out), lat)
        return out


class OpenAICompatBackend(LLMBackend):
    name = "openai-compat"

    def __init__(self, model: str = None, base_url: str = None, api_key: str = None):
        super().__init__()
        try:
            from openai import OpenAI
        except ImportError as e:  # pragma: no cover
            raise RuntimeError("pip install openai  (also works for Ollama/vLLM)") from e
        self.model = model or os.getenv("COGNI_MODEL", "gpt-4o-mini")
        self.client = OpenAI(base_url=base_url or os.getenv("COGNI_BASE_URL") or None,
                             api_key=api_key or os.getenv("OPENAI_API_KEY", "ollama"))

    def _complete(self, prompt, meta, temperature):
        r = self.client.chat.completions.create(
            model=self.model, temperature=temperature, response_format={"type": "json_object"},
            messages=[{"role": "system", "content": "You are a careful scientific analyst. Reply with valid JSON only."},
                      {"role": "user", "content": prompt}])
        return r.choices[0].message.content or "{}"


@dataclass
class SimParams:
    # probability that the correct strategy/hypothesis is ranked FIRST at assertion time
    p_rank_easy: float = 0.80
    p_rank_hard: float = 0.35
    # probability a sampled interpretation drifts from the committed strategy
    drift_from_wrong: float = 0.45   # wrong strategy -> model "notices" the right reading
    drift_from_right: float = 0.08   # right strategy -> sloppy alternative reading
    # probability of misreading evidence in a single sample
    misread_obvious: float = 0.05
    misread_subtle: float = 0.18
    ms_per_token: float = 0.02       # simulated generation latency


class SimulatedBackend(LLMBackend):
    """Stochastic surrogate. The *competence* comes from the real deterministic
    analyzers in reasoning.py (passed in via meta['oracle'] callables); the
    surrogate only decides which strategy the model would pick and injects
    sampling noise, so its mistakes are structured rather than arbitrary."""
    name = "simulated"

    def __init__(self, seed: int = 0, params: SimParams = None):
        super().__init__()
        self.rng = random.Random(seed)
        self.p = params or SimParams()

    def sim_latency(self, prompt, out):
        return (count_tokens(out) * self.p.ms_per_token + 25) / 1000.0

    # ---------------------------------------------------------------- dispatch
    def _complete(self, prompt, meta, temperature):
        kind = meta.get("kind")
        fn = getattr(self, f"_k_{kind}", None)
        if fn is None:
            return json.dumps({"note": "no-op"})
        return json.dumps(fn(meta, temperature))

    # ------------------------------------------------------------ assertion
    def _k_assert(self, meta, temperature):
        """Return a ranked list of 3-5 candidate strategies/hypotheses."""
        cands = list(meta["candidates"])
        gold = meta.get("gold_strategy")
        hard = meta.get("difficulty", "easy") == "hard"
        tempting = meta.get("tempting")
        p_first = self.p.p_rank_hard if hard else self.p.p_rank_easy
        self.rng.shuffle(cands)
        if gold in cands:
            cands.remove(gold)
            if self.rng.random() < p_first:
                cands.insert(0, gold)
            else:
                if tempting in cands:
                    cands.remove(tempting)
                    cands.insert(0, tempting)
                cands.insert(self.rng.randint(1, min(2, len(cands))), gold)
        n = min(len(cands), self.rng.randint(3, 5))
        if gold in cands and cands.index(gold) >= n:
            n = cands.index(gold) + 1
        return {"candidates": cands[:n]}

    def _k_commit(self, meta, temperature):
        # the selection prompt binds the top candidate unless told to skip
        cands = [c for c in meta["candidates"] if c not in meta.get("excluded", [])]
        return {"intention": cands[0] if cands else None}

    # ------------------------------------------------------------ execution
    def _k_execute(self, meta, temperature):
        """Run the committed strategy once (one sample)."""
        strategy = meta["strategy"]
        gold = meta.get("gold_strategy")
        fits = strategy in meta.get("fit_set", ())
        alt = meta.get("alternative") or gold
        drift = self.p.drift_from_wrong if not fits else self.p.drift_from_right
        effective = strategy
        if temperature > 0 and self.rng.random() < drift:
            effective = alt if not fits else meta.get("tempting", strategy)
        result = meta["oracle"](effective)
        subtle = meta.get("difficulty") == "hard"
        if temperature > 0 and self.rng.random() < (self.p.misread_subtle if subtle else self.p.misread_obvious):
            result = meta["perturb"](result, self.rng)
        return {"result": result}

    def _k_cot(self, meta, temperature):
        ranked = self._k_assert(meta, temperature)["candidates"]
        m = dict(meta, strategy=ranked[0])
        out = self._k_execute(m, temperature)
        out["strategy"] = ranked[0]
        return out

    def _k_compile(self, meta, temperature):
        return {"template": meta.get("summary", "")}

    def _k_react_revise(self, meta, temperature):
        return {"revise": self.rng.random() < meta.get("p_revise", 0.3)}


def make_backend(kind: str = "simulated", seed: int = 0, **kw) -> LLMBackend:
    if kind == "simulated":
        return SimulatedBackend(seed=seed, params=kw.get("params"))
    return OpenAICompatBackend(model=kw.get("model"), base_url=kw.get("base_url"))


def parse_json(text: str) -> Dict[str, Any]:
    """Robust JSON extraction for real-model output; always returns a dict."""
    obj = None
    try:
        obj = json.loads(text)
    except Exception:
        for pat in (r"\{.*\}", r"\[.*\]"):
            m = re.search(pat, text or "", re.S)
            if m:
                try:
                    obj = json.loads(m.group(0))
                    break
                except Exception:
                    pass
    if isinstance(obj, dict):
        return obj
    if isinstance(obj, list):
        return {"result": obj, "candidates": obj}
    return {}
