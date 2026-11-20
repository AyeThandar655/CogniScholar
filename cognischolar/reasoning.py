"""Task knowledge: strategies (candidate intentions), deterministic analyzers,
prompt builders, answer canonicalisation and scoring helpers.

Two analysis tasks are implemented:

* ``contradiction`` – find conflicting empirical results across papers
  (e.g. two papers report different scores for the same baseline on the same
  dataset/split/metric).
* ``flaw`` – identify a methodology flaw in one paper
  (data leakage, test-set tuning, split mismatch, unfair baseline, single run).

Each task exposes a set of *strategies* (for contradiction) or *hypotheses*
(for flaw).  These are the candidate intentions the agent asserts, commits
to and possibly retracts.
"""
from __future__ import annotations

import re
from collections import defaultdict
from typing import Dict, List, Tuple

from .perception import Percept, ResultTuple

# ============================================================== normalisation
_DS_ALIASES = {"sst2": "sst-2", "sst-2": "sst-2", "squad1.1": "squad", "squad": "squad",
               "mnli-m": "mnli", "mnli": "mnli", "cifar10": "cifar-10", "cifar-10": "cifar-10",
               "imagenet-1k": "imagenet", "imagenet": "imagenet", "conll03": "conll-2003",
               "conll-2003": "conll-2003", "wmt14": "wmt14", "wmt'14": "wmt14"}


def norm_method(m: str) -> str:
    m = m.lower().strip()
    m = re.sub(r"[\s_]+", "-", m)
    m = re.sub(r"\((\w+)\)", r"-\1", m).replace("--", "-").strip("-")
    return m


def norm_dataset(d: str) -> str:
    d = d.lower().strip().rstrip(".,")
    key = d.replace(" ", "")
    return _DS_ALIASES.get(key, key)


def norm_value(v: float) -> float:
    return v * 100 if v <= 1.0 else v


def conflict_key(method, dataset, metric) -> Tuple[str, str, str]:
    return (norm_method(str(method)), norm_dataset(str(dataset)), str(metric).lower())


# ======================================================== contradiction task
CONTRA_STRATEGIES = {
    "naive_name_match": "Group reported numbers by the literal method name and flag any two papers whose numbers differ.",
    "aligned_tuple": "Align results on (method, dataset, split, metric) and flag pairs whose values differ by more than the tolerance.",
    "normalized_aligned": "Normalise method/dataset aliases and value scales (0-1 vs %), then align on (method, dataset, split, metric) and flag differences above tolerance.",
    "claim_polarity_nli": "Compare qualitative claim sentences for opposite polarity (outperforms vs. underperforms) using NLI.",
    "leaderboard_max": "Compare every reported value with the best value reported for the same dataset, regardless of method.",
}
TOL = 1.0


def _cross_paper_pairs(groups):
    for key, items in groups.items():
        for i in range(len(items)):
            for j in range(i + 1, len(items)):
                a, b = items[i], items[j]
                if a.paper_id != b.paper_id:
                    yield key, a, b


def analyze_contradiction(strategy: str, percepts: List[Percept]) -> List[Tuple[str, str, str]]:
    res: List[ResultTuple] = [r for p in percepts for r in p.results]
    found = set()
    if strategy == "naive_name_match":
        g = defaultdict(list)
        for r in res:
            g[r.method].append(r)
        for _, a, b in _cross_paper_pairs(g):
            if abs(a.value - b.value) > 0.5:
                found.add(conflict_key(a.method, a.dataset, a.metric))
    elif strategy in ("aligned_tuple", "normalized_aligned"):
        norm = strategy == "normalized_aligned"
        g = defaultdict(list)
        for r in res:
            k = (norm_method(r.method), norm_dataset(r.dataset)) if norm else (r.method, r.dataset)
            g[k + (r.split, r.metric)].append(r)
        for _, a, b in _cross_paper_pairs(g):
            va, vb = (norm_value(a.value), norm_value(b.value)) if norm else (a.value, b.value)
            if abs(va - vb) > TOL:
                found.add(conflict_key(a.method, a.dataset, a.metric))
    elif strategy == "claim_polarity_nli":
        pos = re.compile(r"\b(outperform\w*|surpass\w*|better than|beats)\b", re.I)
        neg = re.compile(r"\b(underperform\w*|worse than|fails to (?:match|beat)|falls short)\b", re.I)
        cl = [(p.paper_id, c) for p in percepts for c in p.claims]
        for pa, ca in cl:
            for pb, cb in cl:
                if pa < pb and ((pos.search(ca) and neg.search(cb)) or (neg.search(ca) and pos.search(cb))):
                    ma = re.findall(r"[A-Z][\w\-]+", ca)
                    if ma and ma[0] in cb:
                        found.add((norm_method(ma[0]), "*", "claim"))
    elif strategy == "leaderboard_max":
        g = defaultdict(list)
        for r in res:
            g[(norm_dataset(r.dataset), r.metric)].append(r)
        for _, a, b in _cross_paper_pairs(g):
            if abs(norm_value(a.value) - norm_value(b.value)) > 3 * TOL:
                found.add(conflict_key(a.method, a.dataset, a.metric))
    return sorted(found)


# ================================================================= flaw task
FLAW_HYPOTHESES = {
    "data_leakage": "Information from the evaluation data leaks into training (preprocessing/statistics/duplicates/overlap before or across the split).",
    "test_set_tuning": "Hyperparameters, checkpoints or early stopping were selected using the test set.",
    "split_mismatch": "The training and evaluation splits/protocols do not match the claim or the compared baselines.",
    "unfair_baseline": "Baselines received less tuning, compute or a different protocol than the proposed method.",
    "single_run": "Conclusions rest on a single run/seed with no variance estimate despite small margins.",
}
FLAW_LABELS = list(FLAW_HYPOTHESES) + ["none"]

_FLAW_PATTERNS: Dict[str, List[str]] = {
    "data_leakage": [
        r"(normali[sz]\w*|standardi[sz]\w*|scal\w*|vocabulary|feature selection|statistics|tf-idf|pca|imput\w*)[^.]*(entire|full|whole|complete) (data ?set|corpus|data)",
        r"(before|prior to) (the )?(train[- ]test )?split",
        r"(entire|full|whole) (data ?set|corpus)[^.]*(before|prior to)[^.]*split",
        r"(same|overlapping) (patients|users|subjects|speakers|documents|images|authors)[^.]*(both|train\w* and test)",
        r"(duplicat\w*|near-duplicates?)[^.]*(not removed|remain|across|both)",
        r"(oversampl\w*|smote|augment\w*)[^.]*(before|prior to)[^.]*split",
        r"(pre-?train\w*)[^.]*(includ\w*|contain\w*)[^.]*(test|evaluation) (set|data|split)",
        r"(future|later) (data|time ?steps|observations)[^.]*(used|included)[^.]*(train|features)",
    ],
    "test_set_tuning": [
        r"(tun\w*|select\w*|chose|choose|picked|early[- ]stopping|best checkpoint)[^.]*(on|using|based on|against) the test (set|split|data|accuracy|f1|score|loss|performance)",
        r"(maximi[sz]\w*|best)[^.]*test (accuracy|f1|score|performance)[^.]*(select|chos|pick|tun)",
        r"test (set|split)[^.]*(guide|select|choose|tune)[^.]*(hyperparameter|checkpoint|epoch)",
    ],
    "split_mismatch": [
        r"(evaluat\w*|report\w*)[^.]*(dev|development|validation) (set|split)[^.]*(while|whereas|but)[^.]*(baselines?|prior work)[^.]*test",
        r"(random|shuffled) split[^.]*(while|whereas|but|although)[^.]*(temporal|chronological|official|standard) split",
        r"(compare|comparison)[^.]*(numbers|results|scores) (reported|published)[^.]*(different|another|non-standard) (split|test set|protocol)",
        r"(own|custom|re-?created|resampled) (?:\w+ )?(test )?split[^.]*(compar\w*|against)[^.]*(official|published|reported)",
        r"(trained|train) on[^.]*(in-domain|source)[^.]*claim[^.]*(out-of-domain|generali[sz])",
    ],
    "unfair_baseline": [
        r"baselines?[^.]*(default|untuned|without tuning|not tuned|original) hyper-?parameters",
        r"(extensive|exhaustive|large) (hyper-?parameter )?(search|tuning|sweep)[^.]*(our|proposed)[^.]*(baselines?|compared)[^.]*(default|not|fixed|single)",
        r"baselines?[^.]*(fewer|less|smaller|reduced) (epochs|steps|compute|budget|data)",
        r"(numbers|results) for (the )?baselines?[^.]*(copied|taken) from[^.]*(different|other) (preprocessing|tokeni[sz]ation|setup|settings)",
    ],
    "single_run": [
        r"(single|one) (run|seed|random seed|training run)",
        r"(no|without) (variance|standard deviation|confidence interval|error bars|significance test)",
        r"(results|numbers) (are )?from (one|a single) (run|trial)",
    ],
}
_NEGATIONS = [r"(only|exclusively) (on|using|from) the (training|train) (set|split|data)",
              r"(validation|development|dev) set[^.]*(select|tun|early)",
              r"averag\w* (over|across) (\d+|three|five|ten) (seeds|runs)",
              r"(same|identical) (hyper-?parameter )?(search|budget|tuning)[^.]*(all|every|each|baselines)"]


def detect_flaw(hypothesis: str, percept: Percept) -> Tuple[bool, str]:
    for s in percept.method_statements + percept.sentences:
        low = s.lower()
        if any(re.search(n, low) for n in _NEGATIONS) and not re.search(r"\bbut\b|\bwhereas\b|\bwhile\b", low):
            continue
        for pat in _FLAW_PATTERNS.get(hypothesis, []):
            if re.search(pat, low):
                return True, s
    return False, ""


def analyze_flaw(hypothesis: str, percepts: List[Percept]) -> Dict:
    if hypothesis is None:
        return {"confirmed": False, "evidence": ""}
    ok, ev = detect_flaw(hypothesis, percepts[0])
    return {"confirmed": ok, "evidence": ev}


# ============================================================ prompt builders
def _papers_block(percepts: List[Percept], max_sent=40) -> str:
    out = []
    for p in percepts:
        out.append(f"### Paper {p.paper_id}: {p.title}\n" + " ".join(p.sentences[:max_sent]))
    return "\n\n".join(out)


def _results_block(percepts):
    rows = [f"- [{r.paper_id}] {r.method} | {r.dataset} | {r.split} | {r.metric} | {r.value}"
            for p in percepts for r in p.results]
    return "\n".join(rows) or "(none extracted)"


def prompt_assert(task, percepts, candidates: Dict[str, str]) -> str:
    opts = "\n".join(f"- {k}: {v}" for k, v in candidates.items())
    goal = ("identify conflicting empirical results across the papers" if task == "contradiction"
            else "identify the methodology flaw (if any) in the paper")
    return (f"TASK: {goal}.\nPropose 3-5 candidate {'strategies' if task=='contradiction' else 'hypotheses'} "
            f"ranked by plausibility. Choose from:\n{opts}\n\n{_papers_block(percepts)}\n\n"
            "Return JSON: {\"candidates\": [names...]}")


def prompt_commit(task, candidates, excluded, failed_log) -> str:
    return ("SELECTION: bind exactly ONE candidate as the Active Intention. Do not choose an excluded one.\n"
            f"Candidates: {candidates}\nExcluded (retracted): {excluded}\nFailure log: {failed_log[-3:]}\n"
            "Return JSON: {\"intention\": name}")


def prompt_execute(task, percepts, intention, template_hint=None) -> str:
    if task == "contradiction":
        instr = CONTRA_STRATEGIES[intention]
        fmt = '{"result": [[method, dataset, metric], ...]}'
        ctx = f"Extracted result tuples:\n{_results_block(percepts)}"
    else:
        instr = f"Test the hypothesis: {FLAW_HYPOTHESES[intention]} Quote the evidence sentence."
        fmt = '{"result": {"confirmed": true|false, "evidence": "..."}}'
        ctx = "Method statements:\n" + "\n".join("- " + s for s in percepts[0].method_statements)
    hint = f"\nCompiled procedure (reuse): {template_hint}\n" if template_hint else ""
    body = _papers_block(percepts) if not template_hint else ""
    return f"ACTIVE INTENTION: {intention}\nProcedure: {instr}{hint}\n{ctx}\n\n{body}\nReturn JSON: {fmt}"


def prompt_cot(task, percepts, real=False) -> str:
    q = ("Which reported results conflict across these papers?" if task == "contradiction"
         else "Does this paper contain a methodology flaw? Which one?")
    if not real:  # simulated surrogate: original short prompt (keeps reported token counts)
        return f"{q} Let's think step by step.\n\n{_papers_block(percepts)}\n\nReturn JSON with key 'result'."
    if task == "contradiction":
        fmt = 'Return JSON: {"reasoning": "...", "result": [[method, dataset, metric], ...]} (empty list if no conflict).'
    else:
        fmt = 'Return JSON: {"reasoning": "...", "label": one of ' + str(FLAW_LABELS) + '}.'
    return f"{q} Let's think step by step.\n\n{_papers_block(percepts)}\n\n{fmt}"


# ======================================================= canonical answers
def canon(task, result) -> str:
    """Canonical form used to cluster samples for semantic entropy."""
    if task == "contradiction":
        return "|".join(sorted("/".join(map(str, k)) for k in (result or [])))
    return "Y" if (result or {}).get("confirmed") else "N"


def perturb(task):
    def _p(result, rng):
        if task == "contradiction":
            res = list(result or [])
            if res and rng.random() < 0.5:
                res.pop(rng.randrange(len(res)))
            else:
                res.append(("spurious-" + str(rng.randint(0, 9)), "unknown", "accuracy"))
            return sorted(set(map(tuple, res)))
        r = dict(result or {})
        r["confirmed"] = not r.get("confirmed", False)
        return r
    return _p


def parse_result(task, obj):
    """Coerce real-LLM JSON into the canonical python structure (never raises)."""
    try:
        return _parse_result(task, obj)
    except Exception:
        return [] if task == "contradiction" else {"confirmed": False, "evidence": ""}


def _is_num(x):
    if isinstance(x, (int, float)):
        return True
    try:
        float(str(x).strip().rstrip("%"))
        return True
    except ValueError:
        return False


def _parse_result(task, obj):
    r = obj.get("result", obj) if isinstance(obj, dict) else obj
    if task == "contradiction":
        if isinstance(r, dict):
            r = r.get("conflicts", [r] if "method" in r else [])
        out = []
        for x in (r if isinstance(r, list) else []):
            if isinstance(x, dict):
                x = [x.get("method", ""), x.get("dataset", ""), x.get("metric", "")]
            if isinstance(x, (list, tuple)):
                words = [str(v).strip() for v in x if v is not None and not _is_num(v) and str(v).strip()]
                if len(words) >= 3:
                    out.append(conflict_key(*words[:3]))
        return sorted(set(out))
    if isinstance(r, list) and r:
        r = r[0]
    if isinstance(r, dict):
        return {"confirmed": as_bool(r.get("confirmed")), "evidence": str(r.get("evidence", ""))}
    return {"confirmed": as_bool(r) if isinstance(r, (str, bool)) else False, "evidence": ""}

def variablize(text: str) -> str:
    """Soar-style generalisation: replace constants with typed variables."""
    t = re.sub(r"\d+(?:\.\d+)?\s*%?", "<NUM>", text)
    t = re.sub(r"\b[A-Z][A-Za-z]*[\-\d][\w\-\.]*\b", "<ENT>", t)
    t = re.sub(r"\b[A-Z]{2,}[\w\-]*\b", "<ENT>", t)
    return t.lower()


def signature(task, percepts: List[Percept]) -> str:
    if task == "contradiction":
        rows = [f"{r.split} {r.metric} {'frac' if r.value <= 1 else 'pct'}" for p in percepts for r in p.results]
        txt = " ".join(c for p in percepts for c in p.claims)
        return variablize(txt) + " || " + " ".join(rows)
    return variablize(" ".join(percepts[0].method_statements))


def default_sim(task, percepts):
    """Surrogate 'competence' for unlabeled user input (simulated backend only):
    the model's best reading is the strategy/hypothesis the analyzers support."""
    if task == "contradiction":
        return dict(gold_strategy="normalized_aligned", fit_set=["normalized_aligned"],
                    tempting="naive_name_match", difficulty="easy")
    fired = [h for h in FLAW_HYPOTHESES if analyze_flaw(h, percepts)["confirmed"]]
    g = fired[0] if fired else None
    return dict(gold_strategy=g, fit_set=[g] if g else [], tempting=None, difficulty="easy")


# ------------------------------------------------- robust parsing of real-LLM output
_NAME_KEYS = ("name", "strategy", "hypothesis", "candidate", "intention", "label", "id", "type", "flaw")


def as_name(x, valid):
    """Map a model's mention of a strategy/hypothesis (str, dict, list) to a valid name or None."""
    if isinstance(x, (list, tuple)):
        for y in x:
            n = as_name(y, valid)
            if n:
                return n
        return None
    if isinstance(x, dict):
        for k in _NAME_KEYS:
            if k in x:
                n = as_name(x[k], valid)
                if n:
                    return n
        for v in x.values():
            n = as_name(v, valid) if isinstance(v, str) else None
            if n:
                return n
        return None
    if not isinstance(x, str):
        return None
    t = x.strip().lower().replace("-", "_").replace(" ", "_").strip("`'\".:")
    if t in valid:
        return t
    for v in valid:  # e.g. "1. normalized_aligned: normalise names ..."
        if v in t:
            return v
    return None


def as_names(xs, valid):
    if not isinstance(xs, (list, tuple)):
        xs = [xs]
    out = []
    for x in xs:
        n = as_name(x, valid)
        if n and n not in out:
            out.append(n)
    return out


def as_bool(v):
    if isinstance(v, str):
        return v.strip().lower() in ("true", "yes", "y", "1", "confirmed", "supported")
    return bool(v)
