"""Runs every experiment reported in the paper and writes results/*.json|csv
and figures/*.png.

    python experiments/run_all.py                    # simulated surrogate, 5 seeds
    python experiments/run_all.py --backend openai --model gpt-4o-mini --seeds 1
    python experiments/run_all.py --backend openai --base-url http://localhost:11434/v1 --model llama3.1:8b --seeds 1

Steps
  1. Calibrate tau on the 20 calibration items (grid search).
  2. Main comparison on the 80 test items: B1 CoT, B2 ReAct, B3 SC-CoT,
     B4 no-compilation, B5 no-reconsideration, CogniScholar.
  3. Sensitivity of F1 / tokens / thrashing to tau.
  4. Long-session context growth (window saturation).
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import statistics as st
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from cognischolar import AgentConfig, CogniScholarAgent, make_backend          # noqa: E402
from cognischolar.agent import SessionResult  # noqa: E402
from cognischolar.baselines import CoTBaseline, ReActBaseline, SelfConsistencyBaseline  # noqa: E402

RES = ROOT / "results"
FIG = ROOT / "figures"
TAU_GRID = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]


def load(split=None):
    items = []
    for f in ("contradiction", "flaw"):
        for line in open(ROOT / "data" / f"{f}.jsonl"):
            it = json.loads(line)
            if split is None or it["split"] == split:
                items.append(it)
    return items


def correct(it, ans):
    if it["task"] == "flaw":
        return ans == it["gold"]
    return sorted(tuple(x) for x in ans) == sorted(tuple(x) for x in it["gold"])


def prf(tp, fp, fn):
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return p, r, f


def score(pairs):
    out = {}
    for task in ("contradiction", "flaw"):
        tp = fp = fn = 0
        sub = [(it, r) for it, r in pairs if it["task"] == task]
        for it, r in sub:
            if task == "contradiction":
                g = set(map(tuple, it["gold"]))
                p = set(map(tuple, r.answer))
                tp += len(g & p)
                fp += len(p - g)
                fn += len(g - p)
            else:
                g, p = it["gold"], r.answer
                if p != "none" and p == g:
                    tp += 1
                else:
                    fp += int(p != "none")
                    fn += int(g != "none")
        P, Rr, F = prf(tp, fp, fn)
        out[task] = dict(P=P, R=Rr, F1=F, acc=sum(correct(i, r.answer) for i, r in sub) / max(1, len(sub)))
    # robustness
    wrong_start = [(it, r) for it, r in pairs
                   if r.first_intention is not None and it["sim"]["gold_strategy"] is not None
                   and r.first_intention not in it["sim"]["fit_set"]]
    out["error_recovery"] = (sum(correct(i, r.answer) for i, r in wrong_start) / len(wrong_start)) if wrong_start else 0.0
    out["n_wrong_start"] = len(wrong_start)
    out["switches"] = st.mean(r.switches for _, r in pairs)
    out["thrash_rate"] = sum(r.switches >= 3 for _, r in pairs) / len(pairs)
    out["tokens"] = st.mean(r.tokens for _, r in pairs)
    out["calls"] = st.mean(r.calls for _, r in pairs)
    out["latency_s"] = st.mean(r.latency_s for _, r in pairs)
    out["macro_F1"] = (out["contradiction"]["F1"] + out["flaw"]["F1"]) / 2
    return out


def make_system(name, seed, backend, tau, model=None, base_url=None):
    llm = make_backend(backend, seed=seed, model=model, base_url=base_url)
    if name == "B1 CoT":
        return CoTBaseline(llm)
    if name == "B2 ReAct":
        return ReActBaseline(llm)
    if name == "B3 SC-CoT":
        return SelfConsistencyBaseline(llm, k=5)
    cfg = AgentConfig(tau=tau)
    if name == "B4 No-Compilation":
        cfg.compilation = False
    if name == "B5 No-Reconsideration":
        cfg.reconsideration = False
    return CogniScholarAgent(llm, cfg)


VERBOSE = False


def safe_run(system, it):
    """One bad model reply must not kill a multi-hour run: log it and score the item as failed."""
    try:
        return system.run(it)
    except Exception as e:  # noqa: BLE001
        print(f"  ! {it['id']}: {type(e).__name__}: {e} (counted as wrong)", flush=True)
        return SessionResult(it["id"], it["task"], [] if it["task"] == "contradiction" else "none", [],
                             None, None, 0, None, None, 0, 0, 0.0, 0)


def run_stream(system, items, seed, label=None):
    order = items[:]
    random.Random(1000 + seed).shuffle(order)
    out = []
    for n, it in enumerate(order, 1):
        out.append((it, safe_run(system, it)))
        if VERBOSE:
            print(f"  {label or getattr(system, 'name', None) or system.cfg.name}: {n}/{len(order)} {it['id']}", flush=True)
    return out


def agg(runs, keypath):
    vals = []
    for r in runs:
        v = r
        for k in keypath:
            v = v[k]
        vals.append(v)
    return st.mean(vals), (st.stdev(vals) if len(vals) > 1 else 0.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", default="simulated")
    ap.add_argument("--model")
    ap.add_argument("--base-url")
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--skip-sensitivity", action="store_true")
    ap.add_argument("--tau", type=float, help="use this tau and skip calibration (saves many LLM calls)")
    ap.add_argument("--limit", type=int, help="only use the first N test items (quick real-LLM check)")
    a = ap.parse_args()
    global VERBOSE
    VERBOSE = a.backend != "simulated"
    RES.mkdir(exist_ok=True)
    seeds = list(range(a.seeds))
    calib, test = load("calibration"), load("test")
    print(f"calibration={len(calib)} test={len(test)} backend={a.backend}")

    # 1. calibration of tau ------------------------------------------------
    if a.limit:
        test = [i for i in test if i["task"] == "contradiction"][:a.limit // 2] + [i for i in test if i["task"] == "flaw"][:a.limit - a.limit // 2]
    calib_curve = []
    for tau in ([] if a.tau is not None else TAU_GRID):
        f = st.mean(score(run_stream(make_system("CogniScholar", s, a.backend, tau, a.model, a.base_url), calib, s))["macro_F1"]
                    for s in seeds)
        calib_curve.append((tau, f))
    best_tau = a.tau if a.tau is not None else max(calib_curve, key=lambda x: (round(x[1], 3), -abs(x[0] - 0.5)))[0]
    print("calibration:", [(t, round(f, 3)) for t, f in calib_curve], "-> tau*", best_tau)

    # 2. main comparison ---------------------------------------------------
    systems = ["B1 CoT", "B2 ReAct", "B3 SC-CoT", "B4 No-Compilation", "B5 No-Reconsideration", "CogniScholar"]
    main_rows, comp_stats, curves, ctx = {}, {}, {}, {}
    for name in systems:
        runs, comp = [], []
        for s in seeds:
            sysm = make_system(name, s, a.backend, best_tau, a.model, a.base_url)
            pairs = run_stream(sysm, test, s, label=name)
            runs.append(score(pairs))
            curves.setdefault(name, []).append([r.tokens for _, r in pairs])
            if name == "B2 ReAct":
                ctx.setdefault(name, []).append([sum(1.33 * len(p["text"].split()) for p in it["papers"]) + r.tokens * 0.85 for it, r in pairs])
            if name == "CogniScholar":
                ctx.setdefault(name, []).append([sum(1.33 * len(p['text'].split()) for p in it['papers']) + r.wm_tokens_peak for it, r in pairs])
            if isinstance(sysm, CogniScholarAgent) and sysm.cfg.compilation:
                pm = sysm.pm
                allt = pm.templates + pm.pruned
                used = sum(bool(r.used_template) for _, r in pairs)
                pruned = pm.pruned
                good_prune = sum(1 for t in pruned if t.shadow_hits == 0 or t.shadow_fits / t.shadow_hits < 0.5)
                succ = sum(t.successes for t in allt)
                uses = sum(t.uses for t in allt)
                comp.append(dict(compiled=len(allt), active=len(pm.templates), pruned=len(pruned),
                                 reuse_rate=used / len(pairs), tokens_saved=sum(t.tokens_saved for t in allt),
                                 reuse_success=succ / uses if uses else 0.0,
                                 prune_acc=(good_prune / len(pruned)) if pruned else float("nan")))
                if s == 0:
                    (RES / f"templates_{name.split()[0]}.json").write_text(json.dumps(
                        [dict(tid=t.tid, task=t.task, strategy=t.strategy, uses=t.uses, successes=t.successes,
                              tokens_saved=round(t.tokens_saved), pruned=t.pruned,
                              utility=round(t.utility(len(pairs)), 3)) for t in allt], indent=1))
        main_rows[name] = runs
        if comp:
            comp_stats[name] = {k: st.mean(c[k] for c in comp if c[k] == c[k]) if any(c[k] == c[k] for c in comp) else float("nan")
                                for k in comp[0]}
        sys.stdout.flush()
        print(f"{name:24s} macroF1={agg(runs,['macro_F1'])[0]:.3f} "
              f"conF1={agg(runs,['contradiction','F1'])[0]:.3f} flawF1={agg(runs,['flaw','F1'])[0]:.3f} "
              f"tok={agg(runs,['tokens'])[0]:.0f} ERR={agg(runs,['error_recovery'])[0]:.3f} sw={agg(runs,['switches'])[0]:.2f}")

    # table export
    cols = [("contradiction", "P"), ("contradiction", "R"), ("contradiction", "F1"),
            ("flaw", "P"), ("flaw", "R"), ("flaw", "F1"), ("macro_F1",), ("error_recovery",),
            ("switches",), ("thrash_rate",), ("tokens",), ("calls",), ("latency_s",)]
    table = []
    for name, runs in main_rows.items():
        row = {"system": name}
        for c in cols:
            mu, sd = agg(runs, list(c))
            row["_".join(c)] = mu
            row["_".join(c) + "_sd"] = sd
        table.append(row)
    with open(RES / "main_results.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(table[0]))
        w.writeheader()
        w.writerows(table)

    # 3. sensitivity -------------------------------------------------------
    sens = []
    if not a.skip_sensitivity:
        for tau in TAU_GRID:
            runs = [score(run_stream(make_system("CogniScholar", s, a.backend, tau, a.model, a.base_url), test, s)) for s in seeds]
            sens.append(dict(tau=tau, macro_F1=agg(runs, ["macro_F1"])[0], macro_F1_sd=agg(runs, ["macro_F1"])[1],
                             tokens=agg(runs, ["tokens"])[0], switches=agg(runs, ["switches"])[0],
                             error_recovery=agg(runs, ["error_recovery"])[0]))
        print("sensitivity:", [(d["tau"], round(d["macro_F1"], 3), round(d["tokens"])) for d in sens])

    json.dump(dict(backend=a.backend, model=a.model, seeds=seeds, best_tau=best_tau,
                   calibration=calib_curve, table=table, compilation=comp_stats, sensitivity=sens,
                   token_curves={k: [st.mean(x) for x in zip(*v)] for k, v in curves.items()},
                   context={k: [st.mean(x) for x in zip(*v)] for k, v in ctx.items()}),
              open(RES / "results.json", "w"), indent=1)
    print("compilation:", json.dumps(comp_stats, indent=1))
    print("wrote", RES / "results.json")


if __name__ == "__main__":
    main()
