"""Terminal demo: prints the cognitive trace step by step (good for recording).

    python demo/cli_demo.py                      # C12 (recovery) then C13 (template reuse)
    python demo/cli_demo.py --seed 2 --item F00 F02   # flaw: recovery then reuse
    python demo/cli_demo.py --text paperA.pdf paperB.pdf --task contradiction
"""
import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from cognischolar import AgentConfig, CogniScholarAgent, make_backend   # noqa: E402
from cognischolar.perception import load_text                           # noqa: E402

C = dict(h="\033[1;36m", ok="\033[32m", warn="\033[33m", bad="\033[31m", dim="\033[2m", b="\033[1m", x="\033[0m")


def printer(delay):
    def show(e):
        k = e["event"]
        if k == "perceive":
            print(f"{C['h']}[PERCEIVE]{C['x']} {e['summary']}")
            for r in e["results"]:
                print(f"   {C['dim']}result: {r}{C['x']}")
        elif k == "retrieve":
            print(f"{C['h']}[PROCEDURAL MEMORY]{C['x']} " + (f"hit {e['template']} sim={e['score']} -> {e['strategy']}"
                                                             if e.get("template") else f"miss (best sim={e['score']})"))
        elif k == "assert":
            print(f"{C['h']}[ASSERT]{C['x']} candidates: {e['candidates']}")
        elif k == "commit":
            print(f"{C['b']}[COMMIT]{C['x']} Active Intention := {e['intention']}  {C['dim']}WM={e['wm']}{C['x']}")
        elif k == "checkpoint":
            col = C["ok"] if e["entropy"] <= e["tau"] else C["warn"]
            print(f"{C['h']}[CHECKPOINT]{C['x']} k={e['k']} clusters={e['clusters']}  "
                  f"{col}H={e['entropy']:.2f} (tau={e['tau']}){C['x']}")
        elif k == "retract":
            print(f"{C['warn']}[RETRACT]{C['x']} {e['intention']} - {e['reason']} -> backtrack")
        elif k == "compile":
            print(f"{C['ok']}[COMPILE]{C['x']} {e['template']} strategy={e['strategy']} abandoned={e['abandoned']}")
        elif k == "template_use":
            print(f"{C['ok']}[REUSE]{C['x']} {e['template']} success={e['success']} tokens_saved={e['tokens_saved']}")
        elif k == "prune":
            print(f"{C['dim']}[PRUNE] {e['template']} utility={e['utility']} < {e['breakeven']}{C['x']}")
        elif k == "answer":
            print(f"{C['b']}[ANSWER]{C['x']} {e['answer']}  (intention={e['intention']}, H={e['entropy']})")
        time.sleep(delay)
    return show


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--item", nargs="*", default=["C12", "C13"])
    ap.add_argument("--text", nargs="*")
    ap.add_argument("--task", default="contradiction")
    ap.add_argument("--warmup", type=int, default=0)
    ap.add_argument("--tau", type=float, default=0.4)
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--delay", type=float, default=0.3)
    ap.add_argument("--backend", default="simulated")
    ap.add_argument("--model")
    ap.add_argument("--base-url")
    a = ap.parse_args()
    items = {}
    for f in ("contradiction", "flaw"):
        for l in open(ROOT / "data" / f"{f}.jsonl"):
            it = json.loads(l)
            items[it["id"]] = it
    agent = CogniScholarAgent(make_backend(a.backend, seed=a.seed, model=a.model, base_url=a.base_url),
                              AgentConfig(tau=a.tau))
    if a.warmup:
        print(f"{C['dim']}warming up procedural memory with {a.warmup} sessions...{C['x']}")
        for it in [i for i in items.values() if i["split"] == "calibration"][:a.warmup]:
            agent.run(it)
        print(f"{C['dim']}{len(agent.pm.templates)} active templates{C['x']}\n")
    agent.on_event = printer(a.delay)
    queue = []
    if a.text:
        queue.append(dict(id="user", task=a.task, gold=None,
                          papers=[dict(id=f"P{i}", title=Path(t).name, text=load_text(t)) for i, t in enumerate(a.text)]))
    else:
        queue = [items[i] for i in a.item]
    for it in queue:
        print(f"\n{C['b']}=== Session {it['id']} ({it['task']}) ==={C['x']}")
        r = agent.run(it)
        if it.get("gold") is not None:
            print(f"{C['dim']}gold={it['gold']}{C['x']}")
        print(f"{C['dim']}tokens={r.tokens} calls={r.calls} switches={r.switches}{C['x']}")


if __name__ == "__main__":
    main()
