"""Builds the CogniScholar custom test set (hand-written templates,
instantiated with a fixed seed).

  data/contradiction.jsonl  50 paper pairs  (10 calibration + 40 test)
  data/flaw.jsonl           50 papers       (10 calibration + 40 test)

Every item stores the gold answer.  The ``sim`` block (fit_set, tempting,
difficulty) is *derived* by running each strategy's analyzer against the gold
answer; it is consumed only by the simulated LLM surrogate and by the
evaluation scripts, never by the agent's decision logic.

Usage:  python data/build_benchmark.py
"""
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from cognischolar import reasoning as R            # noqa: E402
from cognischolar.perception import perceive       # noqa: E402

rng = random.Random(663)

METHODS = ["BERT-base", "RoBERTa-large", "XLNet-base", "ELECTRA-small", "DeBERTa-v3",
           "T5-base", "ALBERT-xl", "DistilBERT", "ResNet-50", "ViT-B16", "BiLSTM-CRF", "GPT-2"]
TASKS = [("SST-2", "accuracy"), ("MNLI", "accuracy"), ("SQuAD", "F1"), ("CoNLL-2003", "F1"),
         ("CIFAR-10", "accuracy"), ("ImageNet", "top-1 accuracy"), ("QNLI", "accuracy"), ("WMT14", "BLEU")]
TOPICS = ["parameter-efficient fine-tuning", "curriculum learning", "data augmentation for text",
          "robust optimisation", "knowledge distillation", "contrastive pre-training",
          "label-noise robustness", "efficient attention"]
FILLER = [
    "We study {topic} and its effect on downstream performance.",
    "Our approach builds on prior work in {topic}.",
    "We follow the standard experimental protocol of earlier studies.",
    "Section 4 describes the training details.",
    "We release our code to support reproducibility.",
    "Results are summarised in Table 2.",
]


def fmt_val(v, scale="pct"):
    return f"{v/100:.3f}" if scale == "frac" else f"{v:.1f}"


def result_sentence(m, d, metric, v, split="test", scale="pct", style=None):
    style = style or rng.choice(["a", "b", "c"])
    val = fmt_val(v, scale)
    pct = "%" if (scale == "pct" and metric != "BLEU" and style == "a" and rng.random() < .5) else ""
    sp = "" if split == "test" and rng.random() < .5 else f" ({split} set)"
    if scale == "frac":
        return f"{m} achieves an {metric} of {val} on {d}{sp}."
    if style == "a":
        return f"{m} achieves {val}{pct} {metric} on {d}{sp}."
    if style == "b":
        return f"On {d}, {m} reaches {val} {metric}{sp}."
    return f"The {m} baseline obtains {val} {metric} on {d}{sp} in our runs."


def paper_text(pid, results, topic):
    s = [rng.choice(FILLER).format(topic=topic), rng.choice(FILLER).format(topic=topic)]
    for r in results:
        s.append(result_sentence(**r))
    s.append(rng.choice(FILLER).format(topic=topic))
    return " ".join(s)


def build_contradiction(n_per):
    items = []
    kinds = (["true_conflict"] * n_per[0] + ["hidden_conflict"] * n_per[1] + ["distractor_dataset"] * n_per[2] +
             ["distractor_split"] * n_per[3] + ["distractor_rounding"] * n_per[4] + ["distractor_metric"] * n_per[5])
    for i, kind in enumerate(kinds):
        m, *others = rng.sample(METHODS, 5)
        (d, metric), (d2, metric2), (d3, metric3) = rng.sample(TASKS, 3)
        v = round(rng.uniform(70, 94), 1)
        A = [dict(m=m, d=d, metric=metric, v=v)]
        B = []
        gold = []
        if kind == "true_conflict":
            B.append(dict(m=m, d=d, metric=metric, v=round(v - rng.uniform(2.0, 6.0), 1)))
            gold = [R.conflict_key(m, d, metric)]
        elif kind == "hidden_conflict":
            alias = {"SST-2": "SST2", "CIFAR-10": "CIFAR10", "CoNLL-2003": "CoNLL03"}.get(d, d)
            mv = m.upper() if rng.random() < .5 else m.replace("-", " ").title().replace(" ", "-")
            if metric == "BLEU":
                B.append(dict(m=mv, d=alias, metric=metric, v=round(v - rng.uniform(2, 5), 1)))
            else:
                B.append(dict(m=mv, d=alias, metric=metric, v=round(v - rng.uniform(2.5, 6), 1), scale="frac"))
            gold = [R.conflict_key(m, d, metric)]
        elif kind == "distractor_dataset":
            B.append(dict(m=m, d=d2, metric=metric2, v=round(v - rng.uniform(2, 8), 1)))
        elif kind == "distractor_split":
            A[0]["split"] = "test"
            B.append(dict(m=m, d=d, metric=metric, v=round(v + rng.uniform(1.5, 4), 1), split="validation"))
        elif kind == "distractor_rounding":
            B.append(dict(m=m, d=d, metric=metric, v=round(v + rng.choice([-.4, -.2, .3, .5]), 1)))
        elif kind == "distractor_metric":
            alt = "F1" if metric != "F1" else "accuracy"
            if metric == "BLEU":
                alt = "ROUGE-L"
            B.append(dict(m=m, d=d, metric=alt, v=round(v - rng.uniform(3, 9), 1)))
        # unrelated noise results (distinct methods per paper)
        A.append(dict(m=others[0], d=d3, metric=metric3, v=round(rng.uniform(60, 95), 1)))
        B.append(dict(m=others[1], d=d3, metric=metric3, v=round(rng.uniform(60, 95), 1)))
        if rng.random() < .5:
            A.append(dict(m=others[2], d=d2, metric=metric2, v=round(rng.uniform(60, 95), 1)))
        rng.shuffle(A)
        rng.shuffle(B)
        topic = rng.choice(TOPICS)
        papers = [dict(id=f"C{i:02d}a", title=f"On {topic} (A)", text=paper_text("a", A, topic)),
                  dict(id=f"C{i:02d}b", title=f"Revisiting {topic} (B)", text=paper_text("b", B, topic))]
        items.append(dict(id=f"C{i:02d}", task="contradiction", kind=kind, papers=papers,
                          gold=[list(g) for g in sorted(gold)]))
    return items


FLAWS = {
    "data_leakage": [
        ("easy", "Features were standardised using statistics computed on the entire dataset before the train-test split."),
        ("easy", "We applied SMOTE oversampling to the full data prior to the split into training and test sets."),
        ("hard", "Because the same patients contribute several scans, some patients appear in both the training and test partitions."),
        ("hard", "The tokenizer vocabulary and TF-IDF weights were fitted on the complete corpus, including evaluation documents."),
        ("hard", "Near-duplicates of test questions were not removed from the crawled training data."),
    ],
    "test_set_tuning": [
        ("easy", "Hyperparameters were tuned on the test set to obtain the best configuration."),
        ("easy", "We selected the best checkpoint based on the test accuracy at each epoch."),
        ("hard", "Early stopping was triggered using the test split, since no separate validation data was available."),
        ("hard", "The learning rate that maximised test F1 was chosen for all reported runs."),
    ],
    "split_mismatch": [
        ("easy", "We report results on our own random split and compare against numbers reported by prior work on the official split."),
        ("hard", "We evaluate on the development set, whereas the baselines in Table 3 report test results."),
        ("hard", "Our experiments use a random split of the time series, while previous studies use a chronological split."),
    ],
    "unfair_baseline": [
        ("easy", "The baselines were trained with their default hyperparameters, whereas our model received an extensive hyperparameter search."),
        ("easy", "Due to resource limits, baselines were trained for fewer epochs than the proposed model."),
        ("hard", "Results for the baselines were copied from the original papers, which used different preprocessing and tokenization."),
    ],
    "single_run": [
        ("easy", "All numbers come from a single run with one random seed."),
        ("hard", "Our method improves accuracy by 0.3 points; results are from a single training run."),
        ("hard", "We do not report variance across runs, and no significance test was performed."),
    ],
}
BENIGN = [
    "Hyperparameters were selected on the validation set using a grid search.",
    "We average all results over five seeds and report the standard deviation.",
    "Normalisation statistics were computed only on the training set.",
    "We use the official train, validation and test splits released with the dataset.",
    "All models, including baselines, received the same hyperparameter search budget.",
    "Training uses the AdamW optimiser with a linear warm-up schedule.",
    "We train for 20 epochs with a batch size of 32 on a single GPU.",
    "Input text is lower-cased and truncated to 256 tokens.",
    "Data augmentation is applied to training batches only.",
    "Early stopping monitors the validation loss with a patience of three epochs.",
]
FLAW_INTRO = ["We propose a new model for {topic}.", "Our method sets a new state of the art on two benchmarks.",
              "We show that our approach outperforms strong baselines."]


def build_flaw(n_flawed_per, n_none):
    items, i = [], 0
    plan = [(lab, j) for lab in FLAWS for j in range(n_flawed_per)] + [("none", j) for j in range(n_none)]
    for lab, j in plan:
        topic = rng.choice(TOPICS)
        benign = rng.sample(BENIGN, 4)
        if lab != "none":
            diff, sent = FLAWS[lab][j % len(FLAWS[lab])]
            # drop benign sentences that would contradict the flaw statement
            benign = [b for b in benign if not (lab == "single_run" and "seeds" in b)
                      and not (lab == "test_set_tuning" and ("validation set" in b or "Early stopping" in b))
                      and not (lab == "unfair_baseline" and "same hyperparameter" in b)
                      and not (lab == "data_leakage" and "only on the training" in b)
                      and not (lab == "split_mismatch" and "official train" in b)][:3]
            body = benign[:]
            body.insert(rng.randint(0, len(body)), sent)
        else:
            diff = "hard" if j % 2 else "easy"
            body = benign
        intro = [s.format(topic=topic) for s in rng.sample(FLAW_INTRO, 2)]
        text = " ".join(intro + body + ["Section 5 reports the main results."])
        items.append(dict(id=f"F{i:02d}", task="flaw", kind=lab, difficulty=diff,
                          papers=[dict(id=f"F{i:02d}", title=f"A study of {topic}", text=text)], gold=lab))
        i += 1
    return items


def annotate(items):
    bad = 0
    for it in items:
        p = [perceive(x["text"], x["id"], x["title"]) for x in it["papers"]]
        if it["task"] == "contradiction":
            gold = sorted(tuple(g) for g in it["gold"])
            fit = [s for s in R.CONTRA_STRATEGIES if R.analyze_contradiction(s, p) == gold]
            gs = "normalized_aligned"
            pref = ["naive_name_match", "aligned_tuple", "leaderboard_max", "claim_polarity_nli"]
            tempting = next((s for s in pref if s not in fit), "naive_name_match")
            diff = "hard" if it["kind"] in ("hidden_conflict", "distractor_split", "distractor_rounding") else "easy"
            ok = gs in fit
        else:
            fires = {h: R.analyze_flaw(h, p)["confirmed"] for h in R.FLAW_HYPOTHESES}
            gs = it["gold"] if it["gold"] != "none" else None
            fit = [h for h, f in fires.items() if f and h == gs]
            ok = (fires.get(gs, False) if gs else True) and sum(fires.values()) == (1 if gs else 0)
            tempting = rng.choice([h for h in R.FLAW_HYPOTHESES if h != gs])
            diff = it.get("difficulty", "easy")
            if not ok:
                print("  ambiguous:", it["id"], it["kind"], {h: f for h, f in fires.items() if f})
        if not ok:
            bad += 1
        it["sim"] = dict(gold_strategy=gs, fit_set=fit, tempting=tempting, difficulty=diff)
        it["valid"] = ok
    return bad


def main():
    con = build_contradiction([12, 8, 8, 8, 7, 7])
    fl = build_flaw(8, 10)
    for name, items in (("contradiction", con), ("flaw", fl)):
        bad = annotate(items)
        rng.shuffle(items)
        for j, it in enumerate(items):
            it["split"] = "calibration" if j < 10 else "test"
        out = ROOT / "data" / f"{name}.jsonl"
        out.write_text("\n".join(json.dumps(x) for x in items) + "\n")
        print(f"{name}: {len(items)} items -> {out.name}  (invalid: {bad})")


if __name__ == "__main__":
    main()
