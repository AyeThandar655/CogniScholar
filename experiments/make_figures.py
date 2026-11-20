"""Generates the paper figures from results/results.json.
    python experiments/make_figures.py
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "figures"
FIG.mkdir(exist_ok=True)
BLUE, ORANGE, AQUA, GRAY, INK, MUTED = "#2a78d6", "#eb6834", "#1baf7a", "#9a9a94", "#1f1f1e", "#6b6b66"
plt.rcParams.update({"font.family": ["Times New Roman", "Liberation Serif", "DejaVu Serif"], "font.size": 9.5, "axes.edgecolor": "#c9c8c2",
                     "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED,
                     "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.color": "#ecebe6", "grid.linewidth": 0.8, "axes.axisbelow": True})


def save(fig, name):
    fig.savefig(FIG / name, dpi=300, bbox_inches="tight")
    plt.close(fig)


def box(ax, x, y, w, h, text, fc, ec, fs=6.2, bold=False):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.08", fc=fc, ec=ec, lw=1.2))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, color=INK,
            fontweight="bold" if bold else "normal", wrap=True)


def arrow(ax, x1, y1, x2, y2, txt=None, color=MUTED, style="-|>", rad=0.0):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle=style, color=color, lw=1.2, connectionstyle=f"arc3,rad={rad}"))
    if txt:
        ax.text((x1 + x2) / 2, (y1 + y2) / 2 + 0.12, txt, fontsize=6.5, color=MUTED, ha="center")


def architecture():
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    ax.set_xlim(0, 14); ax.set_ylim(0, 7.4); ax.axis("off"); ax.grid(False)
    O, B, G, R_ = ("#fff3ec", ORANGE), ("#eef4fc", BLUE), ("#e9f8f2", AQUA), ("#fdecec", "#c44444")
    N = ("#f3f2ee", MUTED)
    box(ax, 0.1, 3.0, 2.5, 2.2, "INPUT &\nPERCEPTION\n\nPDF/text → sentences,\nclaims, result tuples,\nmethod statements", *B, fs=6.2)
    box(ax, 0.1, 0.5, 2.5, 1.9, "OUTPUT & EVAL\nanswer, confidence\n(1−H), full trace incl.\nabandoned strategies", *B, fs=6.2)
    ax.add_patch(FancyBboxPatch((3.0, 0.25), 7.2, 6.65, boxstyle="round,pad=0.02,rounding_size=0.1", fc="#fbfbf9", ec="#c9c8c2", lw=1, ls="--"))
    ax.text(6.6, 6.62, "Cognitive cycle:  observe → decide → act → learn", ha="center", fontsize=7.5, color=MUTED)
    box(ax, 3.25, 4.85, 2.05, 1.35, "① ASSERT\n3–5 candidate\nintentions", *O, fs=6.2)
    box(ax, 5.6, 4.85, 2.05, 1.35, "② COMMIT\nbind ONE\nActive Intention", *O, fs=6.2)
    box(ax, 7.95, 4.85, 2.05, 1.35, "③ ACT\nexecute strategy\n(k samples)", *O, fs=6.2)
    box(ax, 7.95, 2.85, 2.05, 1.45, "④ CHECKPOINT\nsemantic entropy\nH > τ  or\nunsupported?", *O, fs=6.2)
    box(ax, 3.25, 2.85, 2.05, 1.45, "RETRACT &\nBACKTRACK\nlog failure,\nnext candidate", *R_, fs=6.2)
    box(ax, 7.95, 0.75, 2.05, 1.45, "⑤ COMPILE\nvariablize the\nsolved episode\n→ template", *G, fs=6.2)
    box(ax, 10.7, 4.75, 3.2, 1.55, "WORKING MEMORY\nActive Intention, candidates,\nretracted, last H & verdict", *N, fs=6.2)
    box(ax, 10.7, 2.8, 3.2, 1.55, "SESSION LOG (episodic)\nassert · commit · retract ·\ncompile · prune events", *N, fs=6.2)
    box(ax, 10.7, 0.35, 3.2, 2.0, "PROCEDURAL MEMORY\nvector store of templates\nU = freq × tokens saved\n÷ retrieval latency;\nprune if U < break-even", *G, fs=6.2)
    arrow(ax, 2.6, 4.6, 3.25, 5.3)
    arrow(ax, 5.3, 5.52, 5.6, 5.52); arrow(ax, 7.65, 5.52, 7.95, 5.52); arrow(ax, 8.97, 4.85, 8.97, 4.3)
    arrow(ax, 7.95, 3.7, 5.3, 3.7); ax.text(5.95, 3.35, "yes → retract", fontsize=6.0, color="#c44444", ha="center")
    arrow(ax, 4.27, 4.3, 5.9, 4.85, rad=-0.25)
    arrow(ax, 8.97, 2.85, 8.97, 2.2); ax.text(9.07, 2.45, "no → accept", fontsize=6.5, color=MUTED)
    arrow(ax, 10.0, 5.52, 10.7, 5.52, style="<|-|>"); arrow(ax, 10.0, 3.55, 10.7, 3.55)
    arrow(ax, 10.0, 1.45, 10.7, 1.45)
    # retrieval path (procedural memory -> commit)
    ax.plot([10.7, 6.62, 6.62], [0.6, 0.6, 2.6], color=AQUA, lw=1.4)
    arrow(ax, 6.62, 2.55, 6.62, 4.85, color=AQUA)
    ax.text(6.5, 0.75, "retrieve (sim ≥ θ):\nfire chunk, skip assert", fontsize=6.2, color=AQUA, ha="right")
    ax.plot([7.95, 7.3], [1.9, 1.9], color=MUTED, lw=1.2)
    arrow(ax, 6.0, 1.9, 2.6, 1.45, color=MUTED)
    ax.plot([7.3, 6.0], [1.9, 1.9], color=MUTED, lw=1.2)
    save(fig, "fig_architecture.png")


def main_bars(R):
    tab = R["table"]
    names = [t["system"].replace("No-Reconsideration", "No-Reconsid.").replace("No-Compilation", "No-Compil.") for t in tab]
    c = [t["contradiction_F1"] for t in tab]; cs = [t["contradiction_F1_sd"] for t in tab]
    f = [t["flaw_F1"] for t in tab]; fs_ = [t["flaw_F1_sd"] for t in tab]
    x = np.arange(len(names)); w = 0.36
    fig, ax = plt.subplots(figsize=(3.5, 2.5))
    ax.bar(x - w / 2 - 0.01, c, w, yerr=cs, color=BLUE, label="Contradiction F1", error_kw=dict(lw=0.8, capsize=2, ecolor=MUTED))
    ax.bar(x + w / 2 + 0.01, f, w, yerr=fs_, color=ORANGE, label="Flaw F1", error_kw=dict(lw=0.8, capsize=2, ecolor=MUTED))
    ax.set_xticks(x); ax.set_xticklabels(names, rotation=30, ha="right", fontsize=7)
    ax.set_ylim(0, 1); ax.set_ylabel("F1 (mean ± sd, 5 seeds)")
    ax.legend(fontsize=7, frameon=False, loc="upper left", ncol=2)
    ax.grid(axis="x", visible=False)
    save(fig, "fig_main_f1.png")


def sensitivity(R):
    S = R["sensitivity"]
    t = [d["tau"] for d in S]
    fig, axs = plt.subplots(1, 3, figsize=(7.2, 2.0))
    axs[0].errorbar(t, [d["macro_F1"] for d in S], yerr=[d["macro_F1_sd"] for d in S], color=BLUE, lw=2, marker="o", ms=4, capsize=2)
    axs[0].set_title("Macro-F1", fontsize=8.5, color=INK); axs[0].set_ylim(0.5, 0.9)
    axs[1].plot(t, [d["tokens"] for d in S], color=ORANGE, lw=2, marker="o", ms=4)
    axs[1].set_title("Tokens / session", fontsize=8.5, color=INK)
    axs[2].plot(t, [d["switches"] for d in S], color=AQUA, lw=2, marker="o", ms=4)
    axs[2].set_title("Intention switches / session", fontsize=8.5, color=INK)
    for a in axs:
        a.set_xlabel("reconsideration threshold τ")
        a.axvline(R["best_tau"], color=GRAY, lw=1, ls="--")
    axs[0].text(R["best_tau"] + 0.02, 0.52, f"τ*={R['best_tau']}", fontsize=7, color=MUTED)
    fig.tight_layout()
    save(fig, "fig_sensitivity.png")


def token_curve(R):
    tc = R["token_curves"]
    fig, ax = plt.subplots(figsize=(3.5, 2.3))
    def roll(v, w=10):
        v = np.array(v, float); return np.convolve(v, np.ones(w) / w, mode="valid")
    for name, col, lab in (("B4 No-Compilation", ORANGE, "B4 no compilation"), ("CogniScholar", BLUE, "CogniScholar (full)")):
        y = roll(tc[name]); ax.plot(np.arange(len(y)) + 10, y, color=col, lw=2, label=lab)
    ax.set_xlabel("session index (stream of 80 test items)"); ax.set_ylabel("tokens / session\n(10-session rolling mean)")
    ax.legend(fontsize=7, frameon=False)
    save(fig, "fig_tokens_curve.png")


def context(R):
    cx = R["context"]
    react = np.cumsum(cx["B2 ReAct"]); ours = np.array(cx["CogniScholar"])
    fig, ax = plt.subplots(figsize=(3.5, 2.3))
    ax.plot(np.arange(1, len(react) + 1), react / 1000, color=ORANGE, lw=2, label="single growing context (ReAct)")
    ax.plot(np.arange(1, len(ours) + 1), ours / 1000, color=BLUE, lw=2, label="CogniScholar (bounded WM)")
    for lim in (8, 32):
        ax.axhline(lim, color=GRAY, lw=1, ls="--"); ax.text(1, lim + 0.8, f"{lim}k window", fontsize=7, color=MUTED)
    ax.set_xlabel("papers/sessions in one long session"); ax.set_ylabel("context tokens (×1000)")
    ax.legend(fontsize=7, frameon=False, loc="upper left", bbox_to_anchor=(0.0, 0.88))
    save(fig, "fig_context.png")
    s8 = int(np.argmax(react > 8000)) + 1 if (react > 8000).any() else None
    s32 = int(np.argmax(react > 32000)) + 1 if (react > 32000).any() else None
    return dict(react_sat_8k=s8, react_sat_32k=s32, ours_peak=float(ours.max()))


if __name__ == "__main__":
    R = json.load(open(ROOT / "results" / "results.json"))
    architecture(); main_bars(R); sensitivity(R); token_curve(R)
    sat = context(R)
    R["saturation"] = sat
    json.dump(R, open(ROOT / "results" / "results.json", "w"), indent=1)
    print("figures ->", FIG, sat)
