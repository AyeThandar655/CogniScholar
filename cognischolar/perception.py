"""Input & perception layer.

Turns raw paper text (or a PDF) into a structured percept:
  * sections and sentences
  * candidate claims (sentences carrying claim cues)
  * empirical result tuples  (method, dataset, split, metric, value)
  * methodology statements    (sentences about data, splits, tuning, seeds)

Everything here is deterministic so that the cognitive layer above it can be
studied in isolation. An LLM-based claim extractor can be swapped in through
`Reasoner.extract_claims` (see reasoning.py).
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import List, Optional

CLAIM_CUES = (
    "we show", "we find", "we demonstrate", "outperform", "achieves", "achieve",
    "reaches", "obtains", "improves", "state-of-the-art", "surpasses", "reports",
    "results in", "yields", "scores", "attains",
)
METHOD_CUES = (
    "split", "train", "test", "validation", "dev set", "held-out", "tuned",
    "hyperparameter", "seed", "run", "preprocess", "normaliz", "deduplicat",
    "augment", "feature", "early stopping", "cross-validation", "fold",
)
METRICS = ("accuracy", "f1", "macro-f1", "micro-f1", "bleu", "rouge-l", "rouge",
           "exact match", "em", "auc", "map", "mrr", "precision", "recall",
           "perplexity", "top-1 accuracy", "top-1")
SPLITS = ("test", "dev", "validation", "val", "train")

_NUM = r"(?P<value>\d{1,3}(?:\.\d+)?)\s*(?:%|points)?"


@dataclass
class ResultTuple:
    method: str
    dataset: str
    split: str
    metric: str
    value: float
    sentence: str
    paper_id: str = ""

    def key(self):
        return (self.method.lower(), self.dataset.lower(), self.split.lower(), self.metric.lower())


@dataclass
class Percept:
    paper_id: str
    title: str
    sentences: List[str]
    claims: List[str] = field(default_factory=list)
    results: List[ResultTuple] = field(default_factory=list)
    method_statements: List[str] = field(default_factory=list)

    def to_dict(self):
        d = asdict(self)
        return d

    def summary(self) -> str:
        return (f"{self.paper_id}: {len(self.sentences)} sentences, {len(self.claims)} claims, "
                f"{len(self.results)} result tuples, {len(self.method_statements)} method statements")


def load_text(path: str) -> str:
    p = Path(path)
    if p.suffix.lower() == ".pdf":
        try:
            from pypdf import PdfReader
        except ImportError as e:  # pragma: no cover
            raise RuntimeError("pip install pypdf to read PDFs") from e
        reader = PdfReader(str(p))
        return "\n".join((pg.extract_text() or "") for pg in reader.pages)
    return p.read_text(encoding="utf-8", errors="ignore")


def split_sentences(text: str) -> List[str]:
    text = re.sub(r"\s+", " ", text).strip()
    # protect common abbreviations / decimals
    text = re.sub(r"(e\.g|i\.e|et al|vs|Fig|Tab|Sec)\.", lambda m: m.group(1) + "<DOT>", text)
    parts = re.split(r"(?<=[.!?])\s+(?=[A-Z(\[])", text)
    return [p.replace("<DOT>", ".").strip() for p in parts if len(p.strip()) > 3]


def _find_metric(s: str) -> Optional[str]:
    low = s.lower()
    for m in sorted(METRICS, key=len, reverse=True):
        if re.search(r"(?<![a-z])" + re.escape(m) + r"(?![a-z])", low):
            return {"em": "exact match", "top-1": "top-1 accuracy"}.get(m, m)
    return None


def _find_split(s: str) -> str:
    low = s.lower()
    for sp in SPLITS:
        if re.search(r"\b" + sp + r"\b", low):
            return {"val": "validation", "dev": "validation"}.get(sp, sp)
    return "test"


_RESULT_PATTERNS = [
    # "<Method> achieves 91.2% accuracy on SST-2 (test)"
    re.compile(r"(?P<method>[A-Z][\w\-\+\.]*(?:\s[A-Z][\w\-\+\.]*)?)\s+(?:achieves|achieved|obtains|obtained|reaches|reached|scores|scored|attains|attained|reports|reported|gets|yields)\s+(?:an?\s+)?(?:[\w\-]+\s+){0,2}?(?:of\s+)?"
               + _NUM + r"\s*(?P<metric>[\w\-\s]{0,20}?)\s+on\s+(?:the\s+)?(?P<dataset>[A-Z][\w\-\.]*(?:\s[A-Z0-9][\w\-\.]*)?)"),
    # "On SQuAD, BERT-base reaches 88.5 F1"
    re.compile(r"[Oo]n\s+(?:the\s+)?(?P<dataset>[A-Z][\w\-\.]*)[^,]*,\s+(?P<method>[A-Z][\w\-\+\.]*)\s+(?:achieves|reaches|obtains|scores|gets|attains)\s+(?:an?\s+)?" + _NUM + r"\s*(?P<metric>[\w\-\s]{0,20})"),
    # "the <Method> baseline ... <value> <metric> on <dataset>"
    re.compile(r"(?:the\s+)?(?P<method>[A-Z][\w\-\+\.]*)\s+baseline[^.]*?" + _NUM + r"\s*(?P<metric>[\w\-\s]{0,20}?)\s+on\s+(?:the\s+)?(?P<dataset>[A-Z][\w\-\.]*)"),
]


def extract_results(sentence: str, paper_id: str = "") -> List[ResultTuple]:
    out = []
    for pat in _RESULT_PATTERNS:
        for m in pat.finditer(sentence):
            metric = _find_metric(m.group("metric") or "") or _find_metric(sentence)
            if not metric:
                continue
            val = float(m.group("value"))
            ds = m.group("dataset").strip().rstrip(".,")
            method = m.group("method").strip()
            if method.lower() in ("we", "our", "this", "the", "it", "on"):
                continue
            out.append(ResultTuple(method, ds, _find_split(sentence), metric, val, sentence, paper_id))
        if out:
            break
    return out


def perceive(text: str, paper_id: str = "paper", title: str = "") -> Percept:
    sents = split_sentences(text)
    claims, results, methods = [], [], []
    for s in sents:
        low = s.lower()
        r = extract_results(s, paper_id)
        if r:
            results.extend(r)
        if r or any(c in low for c in CLAIM_CUES):
            claims.append(s)
        if any(c in low for c in METHOD_CUES):
            methods.append(s)
    return Percept(paper_id, title or paper_id, sents, claims, results, methods)
