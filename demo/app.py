"""CogniScholar interactive demo (Streamlit).

    streamlit run demo/app.py

Panels
  1. Input & Perception  – paste text / upload PDFs or pick a benchmark item;
                           shows extracted claims, result tuples, method statements.
  2. Cognitive Processing – live trace of assert -> commit -> checkpoint
                           (semantic entropy vs tau) -> retract/backtrack ->
                           compile, with the working-memory state at each step.
  3. Output & Evaluation  – final answer, confidence (1 - entropy, vote share),
                           cost (tokens, calls), and the procedural-memory store
                           with template utilities.
"""
import json
import sys
import time
from collections import Counter
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from cognischolar import AgentConfig, CogniScholarAgent, make_backend   # noqa: E402
from cognischolar.perception import load_text, perceive                 # noqa: E402

st.set_page_config(page_title="CogniScholar", layout="wide")


def show_df(df, **kw):
    """Full-width dataframe on both new (width=) and older (use_container_width=) Streamlit."""
    try:
        st.dataframe(df, width="stretch", **kw)
    except TypeError:
        st.dataframe(df, use_container_width=True, **kw)


@st.cache_data
def load_items():
    out = []
    for f in ("contradiction", "flaw"):
        out += [json.loads(l) for l in open(ROOT / "data" / f"{f}.jsonl")]
    return out


items = load_items()
by_id = {i["id"]: i for i in items}

# ------------------------------------------------------------------ sidebar
st.sidebar.title("CogniScholar")
backend = st.sidebar.selectbox("LLM backend", ["simulated", "openai-compatible"])
model = base_url = None
if backend != "simulated":
    model = st.sidebar.text_input("Model", "gpt-4o-mini")
    base_url = st.sidebar.text_input("Base URL (blank = OpenAI; Ollama: http://localhost:11434/v1)", "")
seed = st.sidebar.number_input("Seed", 0, 999, 7)
k = st.sidebar.slider("k samples per checkpoint", 1, 7, 5)
tau = st.sidebar.slider("Reconsideration threshold tau", 0.0, 1.0, 0.40, 0.05)
recon = st.sidebar.checkbox("Reconsideration interlock", True)
comp = st.sidebar.checkbox("Knowledge compilation", True)
delay = st.sidebar.slider("Animation delay (s)", 0.0, 1.5, 0.4, 0.1)

if "agent" not in st.session_state or st.sidebar.button("Reset agent (clear memory)"):
    st.session_state.agent = None
    st.session_state.history = []


def get_agent():
    cfg = AgentConfig(k=k, tau=tau, reconsideration=recon, compilation=comp)
    a = st.session_state.agent
    if a is None:
        llm = make_backend("simulated" if backend == "simulated" else "openai", seed=int(seed),
                           model=model, base_url=base_url or None)
        a = CogniScholarAgent(llm, cfg)
        st.session_state.agent = a
    a.cfg = cfg
    return a


st.sidebar.markdown("---")
warm = st.sidebar.number_input("Warm-up sessions (build procedural memory)", 0, 60, 15)
if st.sidebar.button("Run warm-up"):
    a = get_agent()
    pool = [i for i in items if i["split"] == "calibration"] + [i for i in items if i["split"] == "test"]
    for it in pool[:warm]:
        a.run(it)
    st.sidebar.success(f"Ran {warm} sessions; {len(a.pm.templates)} active templates.")

# ------------------------------------------------------------------ input
st.title("CogniScholar — commitment, reconsideration & knowledge compilation")
c1, c2 = st.columns([1, 1])
with c1:
    mode = st.radio("Input", ["Benchmark item", "Your own papers"], horizontal=True)
    if mode == "Benchmark item":
        task_f = st.selectbox("Task", ["contradiction", "flaw"])
        ids = sorted(i["id"] for i in items if i["task"] == task_f)
        iid = st.selectbox("Item", ids, format_func=lambda x: f"{x} · {by_id[x]['kind']} · {by_id[x]['split']}")
        item = by_id[iid]
    else:
        task_f = st.selectbox("Task", ["contradiction", "flaw"])
        ups = st.file_uploader("Upload 1-3 papers (PDF or .txt)", accept_multiple_files=True)
        txt = st.text_area("...or paste paper text (separate papers with a line containing ---)", height=150)
        papers = []
        for j, u in enumerate(ups or []):
            p = ROOT / "results" / f"_upload_{j}_{u.name}"
            p.write_bytes(u.read())
            papers.append(dict(id=f"U{j}", title=u.name, text=load_text(str(p))))
        if txt.strip():
            for j, chunk in enumerate(txt.split("\n---\n")):
                papers.append(dict(id=f"P{j}", title=f"Pasted {j}", text=chunk))
        item = dict(id="user", task=task_f, papers=papers, gold=None, sim={})
with c2:
    st.subheader("1 · Input & Perception")
    for p in item["papers"]:
        pc = perceive(p["text"], p["id"], p.get("title", ""))
        with st.expander(f"{p['id']} — {pc.summary()}", expanded=True):
            st.caption(p["text"][:600] + ("…" if len(p["text"]) > 600 else ""))
            if pc.results:
                show_df(pd.DataFrame([dict(method=r.method, dataset=r.dataset, split=r.split,
                                                metric=r.metric, value=r.value) for r in pc.results]),
                             hide_index=True)
            if pc.method_statements and item["task"] == "flaw":
                st.markdown("**Method statements**")
                for s_ in pc.method_statements:
                    st.markdown(f"- {s_}")

run = st.button("▶ Run CogniScholar", type="primary", disabled=not item["papers"])

# ------------------------------------------------------------------ run
ICON = {"perceive": "👁️", "retrieve": "📚", "assert": "💡", "commit": "📌", "checkpoint": "🔍",
        "retract": "↩️", "exhausted": "⚠️", "compile": "🧩", "template_use": "♻️", "prune": "✂️", "answer": "✅"}
if run:
    a = get_agent()
    st.subheader("2 · Cognitive Processing")
    wm_box = st.empty()
    feed = st.container()
    ent_rows = []

    def show(e):
        kind = e["event"]
        if kind == "perceive":
            return
        with feed:
            if kind == "assert":
                st.markdown(f"{ICON[kind]} **Assertion** — candidates: `{', '.join(e['candidates'])}`")
            elif kind == "retrieve":
                if e.get("template"):
                    st.markdown(f"{ICON[kind]} **Procedural memory hit** {e['template']} (sim={e['score']}) "
                                f"→ fire compiled strategy `{e['strategy']}`")
                else:
                    st.markdown(f"{ICON[kind]} Procedural memory: no template above threshold (best sim={e['score']})")
            elif kind == "commit":
                st.markdown(f"{ICON[kind]} **Commit** Active Intention = `{e['intention']}` (source: {e['source']})")
                wm_box.info(f"**Working memory** · Active Intention: `{e['wm']['active']}` · "
                            f"Retracted: {e['wm']['retracted'] or '—'}")
            elif kind == "checkpoint":
                ok = e["entropy"] <= e["tau"]
                st.markdown(f"{ICON[kind]} **Checkpoint** k={e['k']} samples → clusters {e['clusters']}  \n"
                            f"semantic entropy **{e['entropy']:.2f}** {'≤' if ok else '>'} τ={e['tau']:.2f}")
                ent_rows.append(dict(step=f"{len(ent_rows)+1}:{e['intention']}", entropy=e["entropy"]))
            elif kind == "retract":
                st.warning(f"{ICON[kind]} **Retract** `{e['intention']}` — {e['reason']}; backtracking to next candidate")
            elif kind == "compile":
                st.success(f"{ICON[kind]} **Compiled** {e['template']}: strategy `{e['strategy']}`, abandoned {e['abandoned']}")
            elif kind == "template_use":
                st.markdown(f"{ICON[kind]} Template {e['template']} reuse — success={e['success']}, tokens saved≈{e['tokens_saved']}")
            elif kind == "prune":
                st.markdown(f"{ICON[kind]} Pruned {e['template']} (utility {e['utility']} < break-even {e['breakeven']})")
            elif kind == "exhausted":
                st.error(f"{ICON[kind]} All candidates exhausted — fallback: {e['fallback']}")
        time.sleep(delay)

    a.on_event = show
    res = a.run(item)
    a.on_event = None
    if ent_rows:
        st.bar_chart(pd.DataFrame(ent_rows).set_index("step"))

    st.subheader("3 · Output & Evaluation")
    last = [e for e in res.trace if e["event"] == "checkpoint"][-1:]
    H = last[0]["entropy"] if last else 0.0
    share = (max(last[0]["clusters"].values()) / sum(last[0]["clusters"].values())) if last else 1.0
    o1, o2, o3, o4, o5 = st.columns(5)
    o1.metric("Confidence (1−H)", f"{1-H:.2f}")
    o2.metric("Vote share", f"{share:.0%}")
    o3.metric("Intention switches", res.switches)
    o4.metric("Tokens", res.tokens)
    o5.metric("LLM calls", res.calls)
    if item["task"] == "flaw":
        st.markdown(f"### Verdict: **{res.answer}**")
    else:
        st.markdown("### Conflicts found")
        st.table(pd.DataFrame(res.answer, columns=["method", "dataset", "metric"]) if res.answer else
                 pd.DataFrame([{"result": "no conflicting results"}]))
    if item.get("gold") is not None:
        gold = item["gold"]
        ok = (res.answer == gold) if item["task"] == "flaw" else \
            sorted(map(tuple, res.answer)) == sorted(map(tuple, gold))
        (st.success if ok else st.error)(f"Gold: {gold} → {'correct' if ok else 'incorrect'}")
    st.session_state.history.append(dict(item=item["id"], answer=str(res.answer), switches=res.switches,
                                         tokens=res.tokens, template=res.used_template or "",
                                         compiled=res.compiled or ""))

# ------------------------------------------------------------------ memory
st.subheader("Procedural memory (compiled templates)")
a = st.session_state.agent
if a and (a.pm.templates or a.pm.pruned):
    rows = [dict(id=t.tid, task=t.task, strategy=t.strategy, uses=t.uses, successes=t.successes,
                 tokens_saved=round(t.tokens_saved), utility=round(t.utility(a.session_idx), 3),
                 status="pruned" if t.pruned else "active") for t in a.pm.templates + a.pm.pruned]
    show_df(pd.DataFrame(rows), hide_index=True)
else:
    st.caption("Empty — run a few sessions or the warm-up to compile templates.")
if st.session_state.get("history"):
    st.subheader("Session history")
    show_df(pd.DataFrame(st.session_state.history), hide_index=True)
