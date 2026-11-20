"""Memory systems.

* WorkingMemory   – bounded buffer holding the Active Intention (BDI committed
                    goal), the remaining candidates, retracted intentions and
                    the current checkpoint statistics.
* SessionLog      – episodic/historical record of every cognitive event
                    (assert, commit, checkpoint, retract, compile, prune).
* ProceduralMemory – compiled reasoning templates (Soar-style chunks) kept in a
                    small persistent vector store and scored by utility.
"""
from __future__ import annotations

import hashlib
import json
import math
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np


# ============================================================ working memory
@dataclass
class WorkingMemory:
    task: str = ""
    active_intention: Optional[str] = None
    candidates: List[str] = field(default_factory=list)
    retracted: List[str] = field(default_factory=list)
    last_entropy: Optional[float] = None
    last_verdict: Optional[str] = None
    source: str = "assert"           # 'assert' | 'template'
    capacity_tokens: int = 1200      # budget for WM contents in a prompt

    def bind(self, intention):
        self.active_intention = intention

    def retract(self):
        if self.active_intention:
            self.retracted.append(self.active_intention)
        self.active_intention = None

    def remaining(self):
        return [c for c in self.candidates if c not in self.retracted]

    def snapshot(self):
        return asdict(self)


# =============================================================== session log
class SessionLog:
    def __init__(self):
        self.events: List[Dict] = []

    def log(self, kind, **kw):
        e = {"t": round(time.time(), 3), "event": kind, **kw}
        self.events.append(e)
        return e

    def failures(self):
        return [e for e in self.events if e["event"] == "retract"]

    def tail(self, n=3):
        return self.events[-n:]


# ====================================================== embedding (no deps)
class HashingEmbedder:
    """Signed feature-hashing of word uni/bi-grams; L2-normalised.
    Replace with sentence-transformers via ``ProceduralMemory(embedder=...)``."""

    def __init__(self, dim=512):
        self.dim = dim

    def __call__(self, text: str) -> np.ndarray:
        v = np.zeros(self.dim, dtype=np.float32)
        toks = text.lower().split()
        grams = toks + [a + "_" + b for a, b in zip(toks, toks[1:])]
        for g in grams:
            h = int(hashlib.md5(g.encode()).hexdigest(), 16)
            v[h % self.dim] += 1.0 if (h >> 9) & 1 else -1.0
        n = np.linalg.norm(v)
        return v / n if n else v


# ======================================================== procedural memory
@dataclass
class Template:
    tid: str
    task: str
    signature: str
    strategy: str
    abandoned: List[str]
    procedure: str
    created_session: int
    compile_tokens: int
    uses: int = 0
    successes: int = 0
    tokens_saved: float = 0.0
    latency_ms: float = 0.0
    pruned: bool = False
    # evaluation-only bookkeeping (shadow tracking after pruning)
    shadow_hits: int = 0
    shadow_fits: int = 0

    def utility(self, session_idx: int) -> float:
        """U = (Reuse_Frequency x Tokens_Saved) / Retrieval_Latency."""
        age = max(1, session_idx - self.created_session)
        freq = self.uses / age
        saved = self.tokens_saved / max(1, self.uses)
        lat = (self.latency_ms / max(1, self.uses)) if self.uses else 1.0
        return freq * saved / max(lat, 1e-3)


class ProceduralMemory:
    def __init__(self, threshold=0.55, embedder=None, path: Optional[str] = None,
                 prune_every=10, min_age=10, horizon=20):
        self.threshold = threshold
        self.embed = embedder or HashingEmbedder()
        self.templates: List[Template] = []
        self.vecs: List[np.ndarray] = []
        self.path = Path(path) if path else None
        self.prune_every, self.min_age, self.horizon = prune_every, min_age, horizon
        self.pruned: List[Template] = []
        self.pruned_vecs: List[np.ndarray] = []

    # ---------------------------------------------------------------- write
    def compile(self, task, sig, strategy, abandoned, procedure, session_idx, compile_tokens) -> Template:
        tid = "T%03d" % (len(self.templates) + len(self.pruned) + 1)
        t = Template(tid, task, sig, strategy, abandoned, procedure, session_idx, compile_tokens)
        self.templates.append(t)
        self.vecs.append(self.embed(sig))
        return t

    # ----------------------------------------------------------------- read
    def retrieve(self, task, sig):
        t0 = time.perf_counter()
        q = self.embed(sig)
        best, score = None, -1.0
        for t, v in zip(self.templates, self.vecs):
            if t.task != task:
                continue
            s = float(q @ v)
            if s > score:
                best, score = t, s
        lat_ms = (time.perf_counter() - t0) * 1000 + 2.0  # + simulated store I/O
        # shadow lookup over pruned templates (evaluation only)
        shadow, sscore = None, -1.0
        for t, v in zip(self.pruned, self.pruned_vecs):
            if t.task == task:
                s = float(q @ v)
                if s > sscore:
                    shadow, sscore = t, s
        if shadow is not None and sscore >= self.threshold and sscore > score:
            shadow.shadow_hits += 1
        else:
            shadow = None
        if best is not None and score >= self.threshold:
            return best, score, lat_ms, shadow
        return None, score, lat_ms, shadow

    def record_use(self, t: Template, success: bool, tokens_saved: float, lat_ms: float):
        t.uses += 1
        t.successes += int(success)
        t.tokens_saved += max(0.0, tokens_saved)
        t.latency_ms += lat_ms

    # ---------------------------------------------------------------- prune
    def maybe_prune(self, session_idx, log: SessionLog = None):
        if session_idx == 0 or session_idx % self.prune_every:
            return []
        out = []
        keep_t, keep_v = [], []
        for t, v in zip(self.templates, self.vecs):
            age = session_idx - t.created_session
            # break-even: a template must repay its compilation cost within the horizon
            lat = (t.latency_ms / t.uses) if t.uses else 2.0
            breakeven = t.compile_tokens / (self.horizon * max(lat, 1e-3))
            harmful = t.uses >= 2 and t.successes / t.uses < 0.5
            if age >= self.min_age and (t.utility(session_idx) < breakeven or harmful):
                t.pruned = True
                self.pruned.append(t)
                self.pruned_vecs.append(v)
                out.append(t)
                if log:
                    log.log("prune", template=t.tid, utility=round(t.utility(session_idx), 3),
                            breakeven=round(breakeven, 3), harmful=harmful)
            else:
                keep_t.append(t)
                keep_v.append(v)
        self.templates, self.vecs = keep_t, keep_v
        return out

    # ---------------------------------------------------------- persistence
    def save(self, path=None):
        p = Path(path or self.path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps([asdict(t) for t in self.templates], indent=1))

    def load(self, path=None):
        p = Path(path or self.path)
        if p.exists():
            for d in json.loads(p.read_text()):
                t = Template(**d)
                self.templates.append(t)
                self.vecs.append(self.embed(t.signature))


# ============================================================ metacognition
def semantic_entropy(canon_samples: List[str]) -> float:
    """Normalised entropy over semantic-equivalence clusters (0 = all agree)."""
    k = len(canon_samples)
    if k <= 1:
        return 0.0
    counts: Dict[str, int] = {}
    for c in canon_samples:
        counts[c] = counts.get(c, 0) + 1
    h = -sum((n / k) * math.log(n / k) for n in counts.values())
    return max(0.0, h / math.log(k)) + 0.0
