"""Unit and integration tests for Brief B Knowledge Graph Study components.

Covers:
1. EntityLinker (medication and concept graph linking)
2. RuleEngine & VerdictEngine (Excludes1 and hierarchy redundancy)
3. Hierarchical metrics (Hierarchical Precision, Recall, F1)
4. make_tables.py generator
"""

import sys
from pathlib import Path
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from linker import EntityLinker
from rules import RuleEngine
from kg import Hierarchy
from metrics import (
    expand_ancestors,
    hierarchical_precision_recall,
    hierarchical_f1,
    rule_violation_rate,
)


def test_entity_linker():
    linker = EntityLinker()
    # Test medication linking
    med_cands = linker.link_medications(["metformin 500 mg BID", "lisinopril 20 mg daily"])
    codes = {c["code"] for c in med_cands}
    assert "E11.9" in codes
    assert "I10" in codes

    # Test clinical concept text linking
    text_cands = linker.link_text("Patient with severe essential hypertension.")
    text_codes = {c["code"] for c in text_cands}
    assert "I10" in text_codes

    # Test link_record
    record = {
        "id": "rec_01",
        "text": "History of hypertension and chest pain.",
        "medications": ["metformin 500 mg daily"],
    }
    rec_cands = linker.link_record(record)
    rec_codes = {c["code"] for c in rec_cands}
    assert "E11.9" in rec_codes
    assert "I10" in rec_codes


def test_rule_engine_excludes1_detection_and_verdict():
    engine = RuleEngine()

    # Conflicting codes: Type 1 diabetes (E10.9) and Type 2 diabetes (E11.9)
    conflicting = ["E10.9", "E11.9", "I10"]
    violations = engine.check_violations(conflicting)
    assert len(violations) >= 1
    assert any(v["rule_type"] == "excludes1" for v in violations)

    # Verdict with scores
    scores = {"E10.9": 0.4, "E11.9": 0.95, "I10": 0.8}
    resolved, audit = engine.apply_verdict(conflicting, candidate_scores=scores)
    assert "E11.9" in resolved
    assert "I10" in resolved
    assert "E10.9" not in resolved  # lower confidence conflicting code dropped
    assert len(audit) >= 1


def test_rule_engine_hierarchy_redundancy():
    engine = RuleEngine()
    h = Hierarchy()
    h.add("E11", "Type 2 diabetes mellitus", None)
    h.add("E11.9", "Type 2 diabetes mellitus without complications", "E11")

    redundant = ["E11", "E11.9", "I10"]
    violations = engine.check_violations(redundant, hierarchy=h)
    assert len(violations) == 1
    assert violations[0]["rule_type"] == "hierarchy_redundancy"

    resolved, audit = engine.apply_verdict(redundant, hierarchy=h)
    assert "E11.9" in resolved
    assert "I10" in resolved
    assert "E11" not in resolved  # redundant category ancestor dropped
    assert len(audit) == 1


def test_hierarchical_metrics_partial_match():
    h = Hierarchy()
    h.add("E11", "Type 2 diabetes mellitus", None)
    h.add("E11.9", "Type 2 diabetes mellitus without complications", "E11")
    h.add("E11.65", "Type 2 diabetes mellitus with hyperglycemia", "E11")

    # Pred: E11.9, Gold: E11.65 (share ancestor E11)
    pred_sets = [{"E11.9"}]
    gold_sets = [{"E11.65"}]

    # Exact match is 0.0
    from metrics import micro_f1
    assert micro_f1(pred_sets, gold_sets) == 0.0

    # Hierarchical match is positive because both expand to include category E11
    hp, hr = hierarchical_precision_recall(pred_sets, gold_sets, h)
    hf1 = hierarchical_f1(pred_sets, gold_sets, h)
    assert hp > 0.0
    assert hr > 0.0
    assert hf1 > 0.0


def test_rule_violation_rate_calculation():
    engine = RuleEngine()
    pred_lists = [
        ["E10.9", "E11.9"],  # 1 violation (Excludes1)
        ["I10", "E11.9"],    # 0 violations
        ["I10", "I15.0"],    # 1 violation (Excludes1)
        ["J44.9"],           # 0 violations
    ]
    rate = rule_violation_rate(pred_lists, engine)
    assert rate == 0.5  # 2 out of 4 instances have violations


def test_make_tables_smoke(tmp_path):
    from make_tables import generate_tables, load_gold, load_vocab, load_hierarchy

    fixture_dir = REPO_ROOT / "tests" / "fixtures" / "baseline_v0"
    gold = load_gold(fixture_dir / "eval.jsonl")
    _, vocab_set = load_vocab(fixture_dir / "vocab.jsonl")
    hierarchy = load_hierarchy(fixture_dir / "hierarchy.jsonl")

    out_reports = tmp_path / "reports_test"
    generate_tables(
        results_dir=fixture_dir / "results",
        gold_records=gold,
        vocab_valid_set=vocab_set,
        hierarchy=hierarchy,
        out_dir=out_reports,
    )

    for i in range(3, 10):
        md_file = out_reports / f"table{i}.md"
        csv_file = out_reports / f"table{i}.csv"
        assert md_file.exists(), f"Missing table{i}.md"
        assert csv_file.exists(), f"Missing table{i}.csv"
        assert md_file.stat().st_size > 0

