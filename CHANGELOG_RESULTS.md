# Results Changelog & Baseline Freeze Log

This document records the provenance, schema versions, and frozen benchmark baselines for the repository `ICD-10-diagnosis-codes` to strictly prevent mixing of results across code versions per Brief C §0.4.

---

## 1. Baseline Freeze v0 (Pre-Upgrade Freeze)

- **Date:** 2026-10-17
- **Git Branch:** `research/kg-upgrade` (branched off `research/four-strategy-pilot`)
- **Base Commit:** `e786770` (Phase 0 Audit completion)
- **Schema Version:** `v0.1.0`
- **Fixture Directory:** `tests/fixtures/baseline_v0/`
- **Execution Target:** 20 synthetic clinical records (`eval.jsonl`), 5-record dev pool (`dev.jsonl`), 5-record few-shot pool (`fewshot_pool.jsonl`), 84-code vocabulary (`vocab.jsonl`), 8-code hierarchy (`hierarchy.jsonl`).
- **Backend:** `mock` (deterministic ICD-10 mock model)
- **Characterisation Suite:** `tests/test_characterisation.py`

### Frozen Baseline Metrics (`table2.json` Summary)

Evaluated with `python src/evaluate.py --results-dir tests/fixtures/baseline_v0/results --data tests/fixtures/baseline_v0/eval.jsonl --vocab tests/fixtures/baseline_v0/vocab.jsonl`:

| Model | Strategy | n | Precision | Recall | Micro-F1 | Macro-F1 | P@5 | P@8 | Invalid Code Rate |
|---|---|---|---|---|---|---|---|---|---|
| mock | Zero-Shot | 20 | 0.100 | 0.065 | 0.078 | 0.011 | n/a | n/a | 0.0% |
| mock | Few-Shot (k=5) | 20 | 0.100 | 0.065 | 0.078 | 0.011 | n/a | n/a | 0.0% |
| mock | RAG | 20 | 0.100 | 0.065 | 0.078 | 0.011 | n/a | n/a | 0.0% |
| mock | KG | 20 | 0.100 | 0.065 | 0.078 | 0.011 | n/a | n/a | 0.0% |

### Supplementary Diagnostic Rates (`table_s1.md` Summary)

| Strategy | Unsupported Code Rate | Parse Failure Rate | Overflow Rate | Truncation Rate | Mean Codes Emitted | Candidate Recall@k |
|---|---|---|---|---|---|---|
| Zero-Shot | 90.0% | 0.0% | 0.0% | 0.0% | 3.00 | n/a |
| Few-Shot (k=5) | 90.0% | 0.0% | 0.0% | 0.0% | 3.00 | n/a |
| RAG | 90.0% | 0.0% | 0.0% | 0.0% | 3.00 | 42.8% |
| KG | 90.0% | 0.0% | 0.0% | 0.0% | 3.00 | 23.2% |

---

## 2. Result Tagging and Mixing Prevention Policy

1. **Tagging:** Every experiment run output must include:
   - `git_commit`: Hash of the code used to generate the output.
   - `schema_version`: Data schema version (e.g. `v0.1.0` for legacy, `v1.0.0` for Brief A/B target).
   - `backend`: Backend identifier (`mock`, `hf`, `ollama`, `openai`, `gemini`, etc.).
2. **Directory Separation:**
   - Any prior outputs produced by legacy code are sequestered under `results_legacy/`.
   - Mock outputs are saved under `results_mock/` and clearly stamped `MOCK DATA`.
   - Real model evaluation runs are stored under `results/`.
3. **Guardrails:** `src/evaluate.py` will actively abort if mock and real outputs are present in the same results directory.

---

## 3. Upgrades and Schema Changes (Brief C Phase 2)

- **Date:** 2026-10-17
- **Git Branch:** `research/kg-upgrade`
- **Schema Version:** `v1.0.0`
- **Key Enhancements:**
  1. **System A3 Implementation (`kg_rules`):** Integrated `src/rules.py` with official Excludes1 conflict detection and hierarchy redundancy resolution, auditing all diagnostic decisions.
  2. **EntityLinker (`src/linker.py`):** Added pharmacology-to-condition linkage from `src/templates/conditions.yaml` to seed and expand KG retrieval.
  3. **Hierarchical Metrics (`src/metrics.py`):** Implemented `hierarchical_f1`, `hierarchical_precision`, `hierarchical_recall`, and `rule_violation_rate`.
  4. **Official Paper Tables Generator (`src/make_tables.py`):** Automated generation of Brief B Tables III through IX in Markdown and CSV formats.
  5. **Chapter U Code Support (`src/codes.py`, `src/parser.py`):** Expanded regex `^[A-TV-Z]` to `^[A-Z]` to accept official CDC FY2024 Chapter 22 emergency codes (`U07.0`, `U07.1`, `U09.9`).
  6. **Ablation Split Isolation (`src/ablations.py`):** Targeted `--split dev` by default to preserve holdout integrity on `eval.jsonl`.
  7. **Clinical Privacy Guard (`src/data_prep.py`):** Added automated validation ensuring destination paths for clinical data are covered by `.gitignore`.
  8. **Reproducibility Lockfile:** Committed `requirements-lock.txt` with exact package pins.
