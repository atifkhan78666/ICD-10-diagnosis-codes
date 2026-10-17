"""
Phase 1 Baseline Characterisation Test.

Asserts that evaluating the baseline v0 mock fixture outputs on the 20-record fixture
reproduces the exact frozen baseline metrics.
"""

import json
import math
import sys
from pathlib import Path
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from vocab import Vocabulary
from metrics import compute_all_metrics
from parser import parse_output


def test_baseline_v0_characterisation():
    fixture_dir = REPO_ROOT / "tests" / "fixtures" / "baseline_v0"
    vocab_path = fixture_dir / "vocab.jsonl"
    eval_path = fixture_dir / "eval.jsonl"
    results_dir = fixture_dir / "results"
    frozen_table2_path = fixture_dir / "table2.json"

    assert fixture_dir.exists()
    assert vocab_path.exists()
    assert eval_path.exists()
    assert results_dir.exists()
    assert frozen_table2_path.exists()

    # Load frozen expected metrics
    with open(frozen_table2_path, "r", encoding="utf-8") as f:
        frozen_data = json.load(f)
    frozen_rows = {row["Strategy"]: row for row in frozen_data["rows"]}

    # Load vocab
    vocab_codes = []
    with open(vocab_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                vocab_codes.append(json.loads(line))
    vocab = Vocabulary(vocab_codes)
    vocab_valid_set = set(vocab.all_codes())

    # Load gold eval dataset
    gold_records = {}
    with open(eval_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rec = json.loads(line)
                gold_records[rec["id"]] = rec

    strategy_files = {
        "Zero-Shot": "mock__zero_shot.jsonl",
        "Few-Shot (k=5)": "mock__few_shot.jsonl",
        "RAG": "mock__rag.jsonl",
        "KG": "mock__kg.jsonl",
    }

    for strat_label, fname in strategy_files.items():
        res_file = results_dir / fname
        assert res_file.exists(), f"Missing result file {fname}"

        records = []
        with open(res_file, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    records.append(json.loads(line))

        assert len(records) == len(gold_records) == 20
        records.sort(key=lambda r: r["id"])

        pred_lists = []
        for r in records:
            if "parsed_codes" not in r and "raw_output" in r:
                p_codes, _ = parse_output(r["raw_output"])
                pred_lists.append(p_codes)
            else:
                pred_lists.append(r.get("parsed_codes", []))

        gold_sets = [set(gold_records[r["id"]]["gold_codes"]) for r in records]
        metrics = compute_all_metrics(pred_lists, gold_sets, vocab_valid_set)

        expected = frozen_rows[strat_label]
        assert metrics["micro_precision"] == pytest.approx(expected["Precision"], rel=1e-4)
        assert metrics["micro_recall"] == pytest.approx(expected["Recall"], rel=1e-4)
        assert metrics["micro_f1"] == pytest.approx(expected["Micro-F1"], rel=1e-4)
        assert metrics["macro_f1"] == pytest.approx(expected["Macro-F1"], rel=1e-4)
        assert metrics["invalid_code_rate"] == pytest.approx(expected["Invalid Code Rate"], rel=1e-4)
