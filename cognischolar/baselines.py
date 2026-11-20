"""Baselines B1-B3 (B4/B5 are ablations of CogniScholarAgent via AgentConfig).

B1 CoT     – one chain-of-thought pass; first interpretation is final.
B2 ReAct   – Thought/Action/Observation loop with analysis tools; the whole
             trajectory is re-sent each step (single growing context).
B3 SC-CoT  – k independent CoT samples + majority vote; no commitment and
             no backtracking.
"""
from __future__ import annotations

from collections import Counter

from . import reasoning as R
from .agent import SessionResult
from .llm import LLMBackend, SimulatedBackend, parse_json, count_tokens
from .perception import perceive


class _Base:
    name = "base"

    def __init__(self, llm: LLMBackend, k=5, temperature=0.7):
        self.llm, self.k, self.temperature = llm, k, temperature
        self.simulated = isinstance(llm, SimulatedBackend)

    def _percepts(self, item):
        return [perceive(p["text"], p["id"], p.get("title", "")) for p in item["papers"]]

    def _meta(self, kind, task, percepts, item, **kw):
        sim = item.get("sim") or R.default_sim(task, percepts)
        oracle = (lambda s: R.analyze_contradiction(s, percepts)) if task == "contradiction" \
            else (lambda s: R.analyze_flaw(s, percepts))
        cands = R.CONTRA_STRATEGIES if task == "contradiction" else R.FLAW_HYPOTHESES
        m = dict(kind=kind, candidates=list(cands), gold_strategy=sim.get("gold_strategy"),
                 tempting=sim.get("tempting"), difficulty=sim.get("difficulty", "easy"),
                 fit_set=set(sim.get("fit_set", [])), oracle=oracle, perturb=R.perturb(task),
                 alternative=sim.get("gold_strategy"))
        m.update(kw)
        return m

    def _cot_once(self, item, percepts, temperature):
        task = item["task"]
        out = parse_json(self.llm.complete(R.prompt_cot(task, percepts, real=not self.simulated),
                                           self._meta("cot", task, percepts, item),
                                           temperature=temperature, phase="cot"))
        strat = out.get("strategy")
        res = out.get("result") if self.simulated else R.parse_result(task, out)
        if task == "flaw":
            if self.simulated:
                ans = strat if (res or {}).get("confirmed") else "none"
            else:
                ans = R.as_name(out.get("label") or out.get("result") or "none", R.FLAW_LABELS) or "none"
            return ans, strat
        return res or [], strat

    def _wrap(self, item, ans, first, final, switches, m0, wm_peak=0):
        m1 = self.llm.meter.snapshot()
        return SessionResult(item["id"], item["task"], ans, [], first, final, switches, None, None,
                             m1["tokens"] - m0["tokens"], m1["calls"] - m0["calls"],
                             m1["latency_s"] - m0["latency_s"], wm_peak)


class CoTBaseline(_Base):
    name = "B1 CoT"

    def run(self, item):
        m0 = self.llm.meter.snapshot()
        p = self._percepts(item)
        ans, strat = self._cot_once(item, p, self.temperature)
        return self._wrap(item, ans, strat, strat, 0, m0)


class SelfConsistencyBaseline(_Base):
    name = "B3 SC-CoT"

    def run(self, item):
        m0 = self.llm.meter.snapshot()
        p = self._percepts(item)
        outs = [self._cot_once(item, p, self.temperature) for _ in range(self.k)]
        key = lambda a: a if item["task"] == "flaw" else R.canon("contradiction", a)
        votes = Counter(key(a) for a, _ in outs)
        top = votes.most_common(1)[0][0]
        ans, strat = next((a, s) for a, s in outs if key(a) == top)
        return self._wrap(item, ans, outs[0][1], strat, len(set(s for _, s in outs)) - 1, m0)


class ReActBaseline(_Base):
    name = "B2 ReAct"

    def __init__(self, llm, max_steps=5, p_revise=0.35, **kw):
        super().__init__(llm, **kw)
        self.max_steps, self.p_revise = max_steps, p_revise

    def run(self, item):
        m0 = self.llm.meter.snapshot()
        task = item["task"]
        p = self._percepts(item)
        ranked = parse_json(self.llm.complete(
            "ReAct. Thought: which approach first?\n" + R.prompt_assert(task, p, R.CONTRA_STRATEGIES if task == "contradiction" else R.FLAW_HYPOTHESES),
            self._meta("assert", task, p, item), temperature=self.temperature, phase="react")).get("candidates", [])
        valid = list(R.CONTRA_STRATEGIES if task == "contradiction" else R.FLAW_HYPOTHESES)
        ranked = R.as_names(ranked, valid) or valid
        context = R.prompt_cot(task, p, real=not self.simulated)
        first, switches, ans, peak = ranked[0], 0, None, 0
        for step in range(min(self.max_steps, len(ranked))):
            strat = ranked[step]
            prompt = context + f"\nThought {step+1}: try {strat}\nAction: analyze[{strat}]"
            if not self.simulated:  # tell a real model what the observation must look like
                prompt += "\n" + R.prompt_execute(task, p, strat, template_hint="(ReAct step)")
            peak = max(peak, count_tokens(prompt))
            obs = parse_json(self.llm.complete(prompt, self._meta("execute", task, p, item, strategy=strat),
                                               temperature=self.temperature, phase="react"))
            res = obs.get("result") if self.simulated else R.parse_result(task, obs)
            context = prompt + f"\nObservation {step+1}: {res}"
            if task == "flaw":
                if (res or {}).get("confirmed"):
                    ans = strat
                    break
                ans = "none"
            else:
                ans = res or []
            rev = parse_json(self.llm.complete(context + "\nThought: is the observation sufficient, or revise?",
                                               self._meta("react_revise", task, p, item, p_revise=self.p_revise if task == "contradiction" else 0.6),
                                               temperature=self.temperature, phase="react")).get("revise", False)
            rev = R.as_bool(rev)
            if not rev:
                break
            switches += 1
        final = ranked[min(switches, len(ranked) - 1)]
        return self._wrap(item, ans, first, final, switches, m0, peak)
