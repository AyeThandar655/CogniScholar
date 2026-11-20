"""Per-item-kind accuracy breakdown (writes results/breakdown.json)."""
import json, sys, statistics as st
from collections import defaultdict
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "experiments"))
from run_all import load, make_system, run_stream, correct
R = json.load(open(ROOT / "results" / "results.json"))
test = load("test"); out = {}
for name in ["B1 CoT", "B3 SC-CoT", "B4 No-Compilation", "CogniScholar"]:
    acc = defaultdict(list)
    for s in range(5):
        for it, r in run_stream(make_system(name, s, "simulated", R["best_tau"]), test, s):
            acc[it["kind"]].append(correct(it, r.answer))
    out[name] = {k: round(sum(v) / len(v), 3) for k, v in sorted(acc.items())}
json.dump(out, open(ROOT / "results" / "breakdown.json", "w"), indent=1)
for n, d in out.items(): print(n.ljust(18), d)
