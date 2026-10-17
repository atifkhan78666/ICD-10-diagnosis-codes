.PHONY: setup vocab data index kg-build smoke phase1 evaluate tables test phase2 significance ablations clean

PYTHON ?= python
CONFIG ?= configs/phase1.yaml
SYNTH_DIR ?= data/synthetic/run01
RESULTS_DIR ?= results/exp01
RESULTS_MOCK_DIR ?= results_mock/exp01

# ── Setup ──────────────────────────────────────────────────────────────────
setup:
	$(PYTHON) -m pip install -r requirements.txt
	@echo "✓ Dependencies installed"

# ── M1: Vocabulary ─────────────────────────────────────────────────────────
vocab:
	$(PYTHON) src/vocab.py --config $(CONFIG)
	@echo "✓ Vocabulary processed"

# ── M2: Synthetic data ─────────────────────────────────────────────────────
data:
	$(PYTHON) src/synth_data.py \
		--n-eval 200 --n-fewshot-pool 50 --n-dev 30 \
		--implicit-rate 0.45 --seed 42 --max-words 200 \
		--out $(SYNTH_DIR)
	@echo "✓ Synthetic data generated"

# ── M3: Retrieval index & KG Hierarchy ────────────────────────────────────
index:
	@echo "Index is built on first RAG run. To pre-build:"
	@echo "  python -c 'from src.retrieval import build_index; from src.vocab import Vocabulary; v=Vocabulary.load_or_download({}); build_index(\"sentence-transformers/all-MiniLM-L6-v2\", v, \"data/index/sentence-transformers_all-MiniLM-L6-v2\")'"

kg-build:
	$(PYTHON) src/kg.py build \
		--zip data/raw/icd10cm-Table-and-Index-2024.zip \
		--xml data/raw/icd10cm_tabular_2024.xml \
		--vocab data/vocab/icd10cm_2024.jsonl \
		--out data/kg/icd10cm_2024_hierarchy.jsonl
	@echo "✓ Knowledge Graph hierarchy built"

# ── M4: Smoke test (mock) ─────────────────────────────────────────────────
smoke:
	$(PYTHON) src/run_experiment.py \
		--config $(CONFIG) \
		--backend mock \
		--strategies zero_shot few_shot rag kg \
		--limit 20 \
		--out $(RESULTS_MOCK_DIR)
	$(PYTHON) src/evaluate.py \
		--results-dir $(RESULTS_MOCK_DIR)_smoke \
		--data $(SYNTH_DIR)/eval.jsonl \
		--vocab data/vocab/icd10cm_2024.jsonl \
		--allow-partial \
		--dump-invalid 5
	@echo "✓ Smoke test complete (MOCK DATA)"

# ── M5/M6: Full Phase 1 run (requires GPU) ────────────────────────────────
phase1:
	$(PYTHON) src/run_experiment.py \
		--config $(CONFIG) \
		--models llama3 biomistral \
		--strategies zero_shot few_shot rag kg \
		--split eval \
		--out $(RESULTS_DIR)
	@echo "✓ Phase 1 inference complete"

# ── Evaluation ─────────────────────────────────────────────────────────────
evaluate:
	$(PYTHON) src/evaluate.py \
		--results-dir $(RESULTS_DIR) \
		--data $(SYNTH_DIR)/eval.jsonl \
		--vocab data/vocab/icd10cm_2024.jsonl \
		--dump-invalid 5
	@echo "✓ Evaluation complete. See reports/"

tables:
	$(PYTHON) src/make_tables.py \
		--results-dir $(RESULTS_DIR) \
		--data $(SYNTH_DIR)/eval.jsonl \
		--vocab data/vocab/icd10cm_2024.jsonl \
		--hierarchy data/kg/icd10cm_2024_hierarchy.jsonl \
		--out reports/
	@echo "✓ Brief B Tables III-IX generated in reports/"

# ── Tests ──────────────────────────────────────────────────────────────────
test:
	$(PYTHON) -m pytest tests/ -v --tb=short
	@echo "✓ All tests passed"

# ── Phase 2 ────────────────────────────────────────────────────────────────
significance:
	$(PYTHON) src/significance.py \
		--results-dir $(RESULTS_DIR) \
		--data $(SYNTH_DIR)/eval.jsonl \
		--seed 42 --B 10000
	@echo "✓ Significance testing complete"

ablations:
	$(PYTHON) src/ablations.py --config configs/ablation_embedder.yaml --type embedder
	$(PYTHON) src/ablations.py --config configs/ablation_k.yaml --type k
	$(PYTHON) src/ablations.py --config configs/ablation_implicit.yaml --type implicit
	@echo "✓ Ablation experiments complete"

phase2: significance ablations
	@echo "✓ Phase 2 complete"

# ── Cleanup ────────────────────────────────────────────────────────────────
clean:
	rm -rf results_mock/ reports/ __pycache__ src/__pycache__ tests/__pycache__
	@echo "✓ Cleaned"

