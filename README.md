# CogniScholar

**A cognitive agent with commitment–reconsideration and knowledge compilation for multi-step analysis of scientific papers.**
CPE 633 Cognitive Computing — term project.

CogniScholar applies two cognitive design patterns (Wray, Kirk & Laird, 2025) to LLM-based paper analysis:

1. **Commitment & reconsideration** — the agent *asserts* 3–5 candidate strategies/hypotheses, *commits* to exactly one
   Active Intention held in working memory (BDI-style), and at each analytical checkpoint samples *k* interpretations,
   measures **semantic entropy**, and *retracts & backtracks* when entropy exceeds τ or the hypothesis is not supported.
2. **Knowledge compilation** (Soar-style chunking) — solved episodes are variablized into reusable templates stored in a
   vector store (procedural memory), retrieved for structurally similar problems, scored by
   `U = (Reuse_Frequency × Tokens_Saved) / Retrieval_Latency`, and pruned below break-even.

Two analysis tasks are implemented: **cross-paper contradiction detection** (conflicting reported results) and
**methodology-flaw detection** (data leakage, test-set tuning, split mismatch, unfair baseline, single run).

## Repository layout

```
cognischolar/
  perception.py   input & perception layer: PDF/text -> sentences, claims, result tuples, method statements
  reasoning.py    strategies/hypotheses, deterministic analyzers, prompts, canonicalisation, variablization
  memory.py       WorkingMemory, SessionLog, ProceduralMemory (vector store + utility + pruning), semantic entropy
  agent.py        CogniScholarAgent: assert -> commit -> act -> checkpoint -> retract/backtrack -> compile
  baselines.py    B1 CoT, B2 ReAct, B3 Self-consistency CoT   (B4/B5 = AgentConfig ablations)
  llm.py          OpenAI-compatible backend (OpenAI / Ollama / vLLM) + seeded simulated LLM surrogate
data/
  build_benchmark.py   builds the custom test set (50 contradiction pairs + 50 flaw papers; 20 calibration, 80 test)
  contradiction.jsonl, flaw.jsonl
experiments/
  run_all.py       calibration of τ, main comparison (5 seeds), τ sensitivity, context growth
  make_figures.py  paper figures -> figures/
demo/
  app.py           Streamlit UI (perception | cognitive trace | outputs & procedural memory)
  cli_demo.py      terminal trace, good for screen recording
tests/test_core.py
results/           main_results.csv, results.json, templates_*.json
figures/           PNG figures used in the paper
```

## Setup

Requirements: Python ≥ 3.9.

```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python tests/test_core.py                               # 7 checks, < 2 s
```

## Reproduce the paper results (offline, simulated LLM surrogate)

```bash
python data/build_benchmark.py          # optional: regenerates data/*.jsonl (fixed seed 663)
python experiments/run_all.py           # ~10 s; writes results/main_results.csv and results/results.json
python experiments/make_figures.py      # writes figures/*.png
```

## Run with a real LLM

Any OpenAI-compatible endpoint works.

```bash
# OpenAI
export OPENAI_API_KEY=sk-...
python experiments/run_all.py --backend openai --model gpt-4o-mini --seeds 1 --skip-sensitivity

# Local Ollama (free):  ollama pull llama3.1:8b && ollama serve
python experiments/run_all.py --backend openai --base-url http://localhost:11434/v1 --model llama3.1:8b --seeds 1
```

With a real model the `sim` fields in the data are ignored by the model; they are used only by evaluation
(error-recovery rate, pruning accuracy).

## Demo

```bash
streamlit run demo/app.py               # web UI at http://localhost:8501
python demo/cli_demo.py                 # C12: retract naive strategy -> recover;  C13: compiled template reused
python demo/cli_demo.py --seed 2 --item F00 F02            # flaw detection: backtracking, then reuse
python demo/cli_demo.py --text paperA.pdf paperB.pdf --task contradiction   # your own papers
```

**Suggested recording script (≈ 90 s).** In the Streamlit app: seed 13, τ = 0.40, k = 5 →
pick *contradiction / C12* → show the Perception panel (extracted result tuples, note `0.848` vs `88.7`) → Run →
the trace shows the naive strategy committed, entropy 0.66 > τ, retraction, backtrack to `normalized_aligned`,
entropy 0.00, compilation of T001 → pick *C13* → Run → procedural-memory hit, no assertion step, ~75 % fewer tokens.
Finish on the procedural-memory table (uses, tokens saved, utility).

## Main results (simulated surrogate, 80 test items, 5 seeds)

| System | Contradiction F1 | Flaw F1 | Error recovery | Tokens/session |
|---|---|---|---|---|
| B1 CoT | 0.563 | 0.533 | 0.206 | 158 |
| B2 ReAct | 0.621 | 0.558 | 0.363 | 704 |
| B3 SC-CoT | 0.703 | 0.614 | 0.378 | 792 |
| B4 No compilation | 0.769 | 0.708 | 0.624 | 2794 |
| B5 No reconsideration | 0.600 | 0.655 | 0.000 | 297 |
| **CogniScholar** | **0.873** | **0.723** | **0.624** | 2198 |

These numbers come from the seeded **LLM surrogate** (see `cognischolar/llm.py::SimParams`); they characterise the
behaviour of the cognitive mechanisms, not the accuracy of any particular LLM. Re-run with `--backend openai` to obtain
real-model numbers.

## Real-LLM pilot (Llama-3.1-8B via Ollama, 80 test items, 1 seed, τ = 0.4)

| System | Contradiction F1 | Flaw F1 | Macro-F1 | Tokens/session |
|---|---|---|---|---|
| B1 CoT | 0.241 | 0.568 | 0.404 | 189 |
| B2 ReAct | 0.252 | 0.286 | 0.269 | 891 |
| **B3 SC-CoT** | 0.242 | **0.730** | **0.486** | 961 |
| B4 No compilation | 0.165 | 0.154 | 0.159 | 3050 |
| B5 No reconsideration | 0.165 | 0.000 | 0.082 | 398 |
| CogniScholar | 0.165 | 0.174 | 0.169 | 2376 |

The accuracy gains of the surrogate study did not transfer: the model's errors were consistent (all samples agree,
entropy ≈ 0), so reconsideration could not catch them. Compilation still saved 22 % of tokens. Raw files:
`results_llama31_8b/`. See paper Section IV-J.

## License
MIT
