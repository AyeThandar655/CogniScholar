"""CogniScholar agent: observe -> decide -> act with
  (1) assertion of candidate intentions,
  (2) commitment of exactly one Active Intention in working memory,
  (3) a reconsideration interlock at analytical checkpoints
      (k self-consistency samples -> semantic entropy -> retract/backtrack),
  (4) knowledge compilation of solved episodes into procedural templates,
      utility-scored and pruned below break-even.

Ablation switches in ``AgentConfig`` give baselines B4 (no compilation) and
B5 (no reconsideration).
"""
from __future__ import annotations

import time
from collections import Counter
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

from . import reasoning as R
from .llm import LLMBackend, SimulatedBackend, parse_json, count_tokens
from .memory import ProceduralMemory, SessionLog, WorkingMemory, semantic_entropy
from .perception import Percept, perceive


@dataclass
class AgentConfig:
    k: int = 5                      # samples per checkpoint
    tau: float = 0.50               # semantic-entropy threshold for retraction
    max_retractions: int = 4
    reconsideration: bool = True
    compilation: bool = True
    template_threshold: float = 0.55
    temperature: float = 0.7
    name: str = "CogniScholar"


@dataclass
class SessionResult:
    item_id: str
    task: str
    answer: object
    trace: List[Dict]
    first_intention: Optional[str]
    final_intention: Optional[str]
    switches: int
    used_template: Optional[str]
    compiled: Optional[str]
    tokens: int
    calls: int
    latency_s: float
    wm_tokens_peak: int
    entropies: List[float] = field(default_factory=list)


class CogniScholarAgent:
    def __init__(self, llm: LLMBackend, cfg: AgentConfig = None, memory: ProceduralMemory = None,
                 on_event: Callable[[Dict], None] = None):
        self.llm = llm
        self.cfg = cfg or AgentConfig()
        self.pm = memory or ProceduralMemory(threshold=self.cfg.template_threshold)
        self.session_idx = 0
        self.on_event = on_event
        self._avg_tokens: Dict[str, List[int]] = {"contradiction": [], "flaw": []}
        self.simulated = isinstance(llm, SimulatedBackend)

    # ----------------------------------------------------------- helpers
    def _emit(self, log: SessionLog, kind, **kw):
        e = log.log(kind, **kw)
        if self.on_event:
            self.on_event(e)

    def _meta(self, kind, task, percepts, item, **kw):
        sim = item.get("sim") or R.default_sim(task, percepts)
        oracle = (lambda s: R.analyze_contradiction(s, percepts)) if task == "contradiction" \
            else (lambda s: R.analyze_flaw(s, percepts))
        m = dict(kind=kind, gold_strategy=sim.get("gold_strategy"), tempting=sim.get("tempting"),
                 difficulty=sim.get("difficulty", "easy"), fit_set=set(sim.get("fit_set", [])),
                 oracle=oracle, perturb=R.perturb(task), alternative=sim.get("gold_strategy"))
        m.update(kw)
        return m

    def _strategies(self, task):
        return R.CONTRA_STRATEGIES if task == "contradiction" else R.FLAW_HYPOTHESES

    def _sample(self, task, percepts, item, intention, hint=None, temperature=None):
        prompt = R.prompt_execute(task, percepts, intention, hint)
        out = self.llm.complete(prompt, self._meta("execute", task, percepts, item, strategy=intention),
                                temperature=self.cfg.temperature if temperature is None else temperature,
                                phase="execute")
        obj = parse_json(out)
        if self.simulated:
            return obj.get("result")
        return R.parse_result(task, obj)

    # -------------------------------------------------------------- main
    def perceive(self, item) -> List[Percept]:
        return [perceive(p["text"], p["id"], p.get("title", "")) for p in item["papers"]]

    def run(self, item) -> SessionResult:
        cfg, task = self.cfg, item["task"]
        log = SessionLog()
        m0 = self.llm.meter.snapshot()
        wm = WorkingMemory(task=task)
        wm_peak = 0

        # ---------- OBSERVE: perception layer
        percepts = self.perceive(item)
        for p in percepts:
            self._emit(log, "perceive", paper=p.paper_id, summary=p.summary(),
                       results=[f"{r.method}|{r.dataset}|{r.split}|{r.metric}={r.value}" for r in p.results],
                       method_statements=p.method_statements[:6])

        # ---------- DECIDE (a): procedural memory lookup
        sig = R.signature(task, percepts)
        template, shadow, t_lat = None, None, 0.0
        if cfg.compilation:
            template, score, t_lat, shadow = self.pm.retrieve(task, sig)
            if template:
                wm.candidates = [template.strategy]
                wm.source = "template"
                self._emit(log, "retrieve", template=template.tid, score=round(score, 3),
                           strategy=template.strategy, latency_ms=round(t_lat, 2))
            else:
                self._emit(log, "retrieve", template=None, score=round(score, 3))
            if shadow is not None and self.simulated:
                shadow.shadow_fits += int(shadow.strategy in item.get("sim", {}).get("fit_set", []))

        asserted = False

        def do_assert():
            cands = self._strategies(task)
            out = self.llm.complete(R.prompt_assert(task, percepts, cands),
                                    self._meta("assert", task, percepts, item, candidates=list(cands)),
                                    temperature=cfg.temperature, phase="assert")
            got = R.as_names(parse_json(out).get("candidates", []), list(cands)) or list(cands)
            wm.source = "assert"
            for c in got:
                if c not in wm.candidates:
                    wm.candidates.append(c)
            self._emit(log, "assert", candidates=got)

        if not wm.candidates:
            do_assert()
            asserted = True

        first_intention, switches, answer, best = None, 0, None, None
        entropies = []
        while True:
            remaining = wm.remaining()
            if not remaining:
                if not asserted and cfg.reconsideration:
                    do_assert()
                    asserted = True
                    remaining = wm.remaining()
                if not remaining:
                    break
            # ---------- DECIDE (b): commitment
            if wm.source == "template" and not wm.retracted:
                intention = remaining[0]          # chunk fires directly, no selection call
            else:
                out = self.llm.complete(R.prompt_commit(task, remaining, wm.retracted,
                                                        [e.get("intention") for e in log.failures()]),
                                        self._meta("commit", task, percepts, item, candidates=remaining,
                                                   excluded=wm.retracted),
                                        temperature=0.0, phase="commit")
                intention = R.as_name(parse_json(out).get("intention"), remaining)
                if intention not in remaining:
                    intention = remaining[0]
            wm.bind(intention)
            first_intention = first_intention or intention
            self._emit(log, "commit", intention=intention, source=wm.source,
                       wm=dict(active=intention, retracted=list(wm.retracted)))

            # ---------- ACT + checkpoint (reconsideration interlock)
            k = cfg.k if cfg.reconsideration else 1
            hint = template.procedure if (template and intention == template.strategy) else None
            samples = [self._sample(task, percepts, item, intention, hint,
                                    temperature=cfg.temperature if k > 1 else 0.0) for _ in range(k)]
            canons = [R.canon(task, s) for s in samples]
            H = semantic_entropy(canons)
            entropies.append(H)
            top, cnt = Counter(canons).most_common(1)[0]
            verdict = samples[canons.index(top)]
            wm.last_entropy, wm.last_verdict = H, top
            wm_peak = max(wm_peak, count_tokens(str(wm.snapshot())) + count_tokens(str(log.tail(3))))
            rejected = task == "flaw" and not verdict.get("confirmed")
            self._emit(log, "checkpoint", intention=intention, k=k, entropy=round(H, 3),
                       clusters=dict(Counter(canons)), verdict=top, tau=cfg.tau)

            if best is None or H < best[0]:
                best = (H, intention, verdict)
            if cfg.reconsideration and (H > cfg.tau or rejected) and switches < cfg.max_retractions:
                reason = "high semantic entropy" if H > cfg.tau else "hypothesis not supported"
                wm.retract()
                switches += 1
                self._emit(log, "retract", intention=intention, reason=reason, entropy=round(H, 3))
                continue
            if task == "flaw":
                answer = intention if verdict.get("confirmed") else "none"
            else:
                answer = verdict
            break

        if answer is None:  # exhausted all candidates
            if task == "flaw":
                answer = "none"
            else:
                answer = best[2] if best else []
                wm.bind(best[1] if best else None)
            self._emit(log, "exhausted", fallback=str(answer))

        m1 = self.llm.meter.snapshot()
        tokens = m1["tokens"] - m0["tokens"]

        # ---------- LEARN: knowledge compilation & utility bookkeeping
        compiled = None
        confident = (wm.last_entropy is not None and wm.last_entropy <= cfg.tau)
        if cfg.compilation:
            if template is not None:
                base = self._avg_tokens[task]
                saved = (sum(base) / len(base) - tokens) if base else 0.0
                success = (wm.active_intention == template.strategy) and confident
                self.pm.record_use(template, success, saved, t_lat)
                self._emit(log, "template_use", template=template.tid, success=success,
                           tokens_saved=round(saved, 1))
            elif confident and wm.active_intention and (answer not in ("none",) or task == "contradiction"):
                proc = (f"For problems shaped like this, go directly to '{wm.active_intention}'. "
                        f"Abandoned before: {wm.retracted}. Steps: {self._strategies(task)[wm.active_intention]}")
                out = self.llm.complete("COMPILE: summarise the successful reasoning as a reusable template.\n" + proc,
                                        self._meta("compile", task, percepts, item, summary=proc),
                                        temperature=0.0, phase="compile")
                ct = count_tokens(proc) + count_tokens(out)
                t = self.pm.compile(task, sig, wm.active_intention, list(wm.retracted), proc,
                                    self.session_idx, ct)
                compiled = t.tid
                self._emit(log, "compile", template=t.tid, strategy=t.strategy, abandoned=t.abandoned)
            self.pm.maybe_prune(self.session_idx + 1, log)
        if template is None:
            self._avg_tokens[task].append(self.llm.meter.snapshot()["tokens"] - m0["tokens"])

        m2 = self.llm.meter.snapshot()
        self._emit(log, "answer", answer=answer if task == "flaw" else [list(a) for a in answer],
                   intention=wm.active_intention, entropy=wm.last_entropy)
        self.session_idx += 1
        return SessionResult(item["id"], task, answer, log.events, first_intention, wm.active_intention,
                             switches, template.tid if template else None, compiled,
                             m2["tokens"] - m0["tokens"], m2["calls"] - m0["calls"],
                             m2["latency_s"] - m0["latency_s"], wm_peak, entropies)
