"""Run with:  python -m pytest -q   (or  python tests/test_core.py)"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from cognischolar import AgentConfig, CogniScholarAgent, make_backend
from cognischolar import reasoning as R
from cognischolar.memory import ProceduralMemory, semantic_entropy
from cognischolar.perception import perceive


def test_perception_extracts_tuples():
    p = perceive("BERT-base achieves 92.7% accuracy on SST-2. On SQuAD, RoBERTa reaches 88.5 F1 (validation set).", "p")
    keys = {(r.method, r.dataset, r.split, r.metric, r.value) for r in p.results}
    assert ("BERT-base", "SST-2", "test", "accuracy", 92.7) in keys
    assert ("RoBERTa", "SQuAD", "validation", "f1", 88.5) in keys


def test_contradiction_strategies_differ():
    a = perceive("BERT-base achieves 90.0 accuracy on SST-2.", "a")
    b = perceive("BERT-base achieves 85.0 accuracy on MNLI.", "b")
    assert R.analyze_contradiction("naive_name_match", [a, b])          # spurious conflict
    assert not R.analyze_contradiction("normalized_aligned", [a, b])    # correct: different datasets


def test_scale_normalisation():
    a = perceive("BERT-base achieves 90.0 accuracy on SST-2.", "a")
    b = perceive("Bert-Base achieves an accuracy of 0.850 on SST2.", "b")
    assert R.analyze_contradiction("normalized_aligned", [a, b]) == [("bert-base", "sst-2", "accuracy")]
    assert not R.analyze_contradiction("aligned_tuple", [a, b])


def test_flaw_detector_and_negation():
    bad = perceive("Hyperparameters were tuned on the test set.", "x")
    good = perceive("Hyperparameters were selected on the validation set using a grid search.", "y")
    assert R.analyze_flaw("test_set_tuning", [bad])["confirmed"]
    assert not R.analyze_flaw("test_set_tuning", [good])["confirmed"]


def test_semantic_entropy_bounds():
    assert semantic_entropy(["a"] * 5) == 0.0
    assert abs(semantic_entropy(list("abcde")) - 1.0) < 1e-9


def test_template_store_roundtrip(tmp_path=Path("/tmp")):
    pm = ProceduralMemory(threshold=0.3)
    pm.compile("flaw", "tuned on the test set", "test_set_tuning", [], "proc", 0, 50)
    t, s, lat, _ = pm.retrieve("flaw", "hyperparameters tuned on the test set")
    assert t is not None and t.strategy == "test_set_tuning"
    pm.save(tmp_path / "tmpl.json")
    pm2 = ProceduralMemory()
    pm2.load(tmp_path / "tmpl.json")
    assert len(pm2.templates) == 1


def test_agent_end_to_end_and_ablation():
    items = [json.loads(l) for l in open(ROOT / "data" / "flaw.jsonl")][:5]
    full = CogniScholarAgent(make_backend("simulated", seed=0), AgentConfig())
    nore = CogniScholarAgent(make_backend("simulated", seed=0), AgentConfig(reconsideration=False))
    for it in items:
        r = full.run(it)
        assert r.answer in R.FLAW_LABELS and r.tokens > 0
        assert nore.run(it).switches == 0


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
