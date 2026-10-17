"""Brief B Tables III–IX Generator for the Knowledge Graph Study.

Computes and formats the official evaluation tables:
- Table III: Comparative Coding Performance Across Systems A0, A1, A2, A3
- Table IV: Retrieval & Candidate Recall@k
- Table V: Exact vs Hierarchical Metrics Alignment
- Table VI: Clinical Coding Rule Violations & Verdict Resolutions
- Table VII: Computational Efficiency & Latency
- Table VIII: Implicit vs Explicit Diagnostic Signal Resilience
- Table IX: Qualitative Error Taxonomy
"""

import argparse
import glob
import json
import logging
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import pandas as pd

from codes import normalize
from evaluate import map_model_name, sort_key
from kg import Hierarchy
from metrics import (
    compute_all_metrics,
    hierarchical_f1,
    hierarchical_precision_recall,
    rule_violation_rate,
)
from parser import parse_output
from rules import RuleEngine
from vocab import Vocabulary

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger(__name__)


SYSTEM_MAP = {
    "zero_shot": "System A0 (Zero-Shot)",
    "few_shot": "Baseline (Few-Shot)",
    "rag": "System A1 (Flat RAG)",
    "kg": "System A2 (KG Hybrid)",
    "kg_rules": "System A3 (KG + Rule Verdict)",
}


def load_vocab(vocab_path: Path) -> Tuple[Optional[Vocabulary], Set[str]]:
    if not vocab_path.exists():
        return None, set()
    codes = []
    with open(vocab_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                codes.append(json.loads(line))
    v = Vocabulary(codes)
    return v, set(v.all_codes())


def load_gold(gold_path: Path) -> Dict[str, Dict[str, Any]]:
    records = {}
    with open(gold_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                records[r["id"]] = r
    return records


def load_hierarchy(hierarchy_path: Optional[Path]) -> Optional[Hierarchy]:
    if hierarchy_path and hierarchy_path.exists():
        return Hierarchy.from_jsonl(str(hierarchy_path))
    return None


def generate_tables(
    results_dir: Path,
    gold_records: Dict[str, Dict[str, Any]],
    vocab_valid_set: Set[str],
    hierarchy: Optional[Hierarchy],
    out_dir: Path,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    rule_engine = RuleEngine()

    result_files = list(results_dir.glob("*.jsonl"))
    if not result_files:
        logger.error(f"No result files found in {results_dir}")
        return

    results_by_config = defaultdict(list)
    for rf in result_files:
        stem = rf.stem
        parts = stem.split("__")
        model_raw = parts[0] if len(parts) >= 2 else stem
        strategy_raw = parts[1] if len(parts) >= 2 else "unknown"

        model = map_model_name(model_raw)
        system_label = SYSTEM_MAP.get(strategy_raw, strategy_raw)

        with open(rf, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    rec = json.loads(line)
                    if "parsed_codes" not in rec and "raw_output" in rec:
                        rec["parsed_codes"], rec["parse_mode"] = parse_output(rec["raw_output"])
                    results_by_config[(model, system_label, strategy_raw)].append(rec)

    # ── Table III: Main Comparative Table ─────────────────────────────────────
    t3_rows = []
    t4_rows = []
    t5_rows = []
    t6_rows = []
    t7_rows = []
    t8_rows = []
    t9_rows = []

    for (model, sys_label, strat_raw), records in sorted(results_by_config.items()):
        records.sort(key=lambda r: r["id"])
        aligned = [r for r in records if r["id"] in gold_records]
        if not aligned:
            continue

        n = len(aligned)
        pred_lists = [r.get("parsed_codes", []) for r in aligned]
        pred_sets = [set(p) for p in pred_lists]
        gold_sets = [set(gold_records[r["id"]].get("gold_codes", [])) for r in aligned]

        metrics = compute_all_metrics(pred_lists, gold_sets, vocab_valid_set)
        h_f1 = hierarchical_f1(pred_sets, gold_sets, hierarchy) if hierarchy else None
        h_p, h_r = hierarchical_precision_recall(pred_sets, gold_sets, hierarchy) if hierarchy else (None, None)
        rvr = rule_violation_rate(pred_lists, rule_engine, hierarchy)

        # Candidate recall
        candidate_recalls = []
        for r, g in zip(aligned, gold_sets):
            raw_cands = r.get("retrieved", [])
            cands = {c["code"] if isinstance(c, dict) else str(c) for c in raw_cands}
            if g:
                candidate_recalls.append(len(cands & g) / len(g))
        cand_recall = sum(candidate_recalls) / len(candidate_recalls) if candidate_recalls else None

        # Table III
        t3_rows.append({
            "Model": model,
            "System": sys_label,
            "n": n,
            "Micro-F1": round(metrics["micro_f1"], 3),
            "Macro-F1": round(metrics["macro_f1"], 3),
            "Hierarchical-F1": round(h_f1, 3) if h_f1 is not None else "n/a",
            "Precision": round(metrics["micro_precision"], 3),
            "Recall": round(metrics["micro_recall"], 3),
            "Invalid Code Rate": f"{metrics['invalid_code_rate']*100:.1f}%",
            "Rule Violation Rate": f"{rvr*100:.1f}%",
        })

        # Table IV: Retrieval Quality
        if cand_recall is not None:
            mean_cands = sum(len(r.get("retrieved", [])) for r in aligned) / n if n else 0
            t4_rows.append({
                "Model": model,
                "System": sys_label,
                "Mean Candidates Offered": round(mean_cands, 1),
                "Candidate Recall@k": f"{cand_recall*100:.1f}%",
                "Diagnostic Precision": round(metrics["micro_precision"], 3),
            })

        # Table V: Exact vs Hierarchical Metrics Alignment
        t5_rows.append({
            "Model": model,
            "System": sys_label,
            "Exact Micro-F1": round(metrics["micro_f1"], 3),
            "Hierarchical Micro-F1": round(h_f1, 3) if h_f1 is not None else "n/a",
            "Δ (Hierarchical - Exact)": f"{h_f1 - metrics['micro_f1']:+.3f}" if h_f1 is not None else "n/a",
            "Hierarchical Precision": round(h_p, 3) if h_p is not None else "n/a",
            "Hierarchical Recall": round(h_r, 3) if h_r is not None else "n/a",
        })

        # Table VI: Coding Rule Audits
        all_violations = []
        all_resolutions = []
        for r in aligned:
            raw_codes = r.get("raw_parsed_codes", r.get("parsed_codes", []))
            viols = rule_engine.check_violations(raw_codes, hierarchy)
            all_violations.extend(viols)
            all_resolutions.extend(r.get("verdict_audit", []))

        e1_count = sum(1 for v in all_violations if v.get("rule_type") == "excludes1")
        red_count = sum(1 for v in all_violations if v.get("rule_type") == "hierarchy_redundancy")

        t6_rows.append({
            "Model": model,
            "System": sys_label,
            "Total Violations Detected": len(all_violations),
            "Excludes1 Conflicts": e1_count,
            "Hierarchy Redundancies": red_count,
            "Verdict Actions Applied": len(all_resolutions),
            "Post-Verdict Violation Rate": f"{rvr*100:.1f}%",
        })

        # Table VII: Efficiency and Latency
        latencies = [r.get("latency_s", 0) for r in aligned if r.get("latency_s") is not None]
        mean_lat = sum(latencies) / len(latencies) if latencies else 0.0
        prompt_toks = [r.get("prompt_tokens", 0) for r in aligned if r.get("prompt_tokens") is not None]
        mean_p_tok = sum(prompt_toks) / len(prompt_toks) if prompt_toks else 0.0
        new_toks = [r.get("new_tokens", 0) for r in aligned if r.get("new_tokens") is not None]
        mean_n_tok = sum(new_toks) / len(new_toks) if new_toks else 0.0
        overflow_pct = sum(1 for r in aligned if r.get("overflow")) / n if n else 0.0

        t7_rows.append({
            "Model": model,
            "System": sys_label,
            "Mean Prompt Tokens": round(mean_p_tok, 1),
            "Mean Generated Tokens": round(mean_n_tok, 1),
            "Mean Latency (s)": round(mean_lat, 2),
            "Overflow Rate": f"{overflow_pct*100:.1f}%",
            "GPU": aligned[0].get("gpu", "unknown"),
        })

        # Table VIII: Implicit vs Explicit Subgroup Performance
        implicit_recs = [r for r in aligned if gold_records[r["id"]].get("has_implicit", False)]
        explicit_recs = [r for r in aligned if not gold_records[r["id"]].get("has_implicit", False)]

        def _subgroup_f1(recs):
            if not recs:
                return "n/a"
            p = [r.get("parsed_codes", []) for r in recs]
            g = [set(gold_records[r["id"]].get("gold_codes", [])) for r in recs]
            m = compute_all_metrics(p, g, vocab_valid_set)
            return round(m["micro_f1"], 3)

        t8_rows.append({
            "Model": model,
            "System": sys_label,
            "All Instances Micro-F1": round(metrics["micro_f1"], 3),
            "Explicit Only Micro-F1": _subgroup_f1(explicit_recs),
            "Implicit Cohort Micro-F1": _subgroup_f1(implicit_recs),
        })

        # Table IX: Qualitative Error Taxonomy
        pf_count = sum(1 for r in aligned if r.get("parse_mode") in ("empty", "failure"))
        trunc_count = sum(1 for r in aligned if r.get("generation_truncated"))

        t9_rows.append({
            "Model": model,
            "System": sys_label,
            "Invalid Code Rate": f"{metrics['invalid_code_rate']*100:.1f}%",
            "Unsupported Code Rate": f"{metrics['unsupported_code_rate']*100:.1f}%",
            "Parse Failure Rate": f"{(pf_count/n)*100:.1f}%",
            "Truncation Rate": f"{(trunc_count/n)*100:.1f}%",
        })

    tables = {
        "table3": ("Table III: Comparative Coding Performance Across Systems A0–A3", t3_rows),
        "table4": ("Table IV: Retrieval Quality and Candidate Recall@k", t4_rows),
        "table5": ("Table V: Exact vs Hierarchical Metrics Alignment", t5_rows),
        "table6": ("Table VI: Clinical Coding Rule Violations and Verdict Resolutions", t6_rows),
        "table7": ("Table VII: Computational Efficiency and Latency", t7_rows),
        "table8": ("Table VIII: Implicit vs Explicit Evidence Robustness", t8_rows),
        "table9": ("Table IX: Qualitative Error Taxonomy", t9_rows),
    }

    for name, (title, rows) in tables.items():
        if not rows:
            continue
        df = pd.DataFrame(rows)
        # Markdown
        md_file = out_dir / f"{name}.md"
        with open(md_file, "w", encoding="utf-8") as f:
            f.write(f"# {title}\n\n")
            f.write(df.to_markdown(index=False))
            f.write("\n")

        # CSV
        csv_file = out_dir / f"{name}.csv"
        df.to_csv(csv_file, index=False)

        logger.info(f"Generated {name} -> {md_file} & {csv_file}")
        print("\n" + "=" * 80)
        print(title)
        print("=" * 80)
        print(df.to_markdown(index=False))
        print("=" * 80 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Generate Brief B Tables III–IX")
    parser.add_argument("--results-dir", required=True, help="Directory containing result JSONL files")
    parser.add_argument("--data", required=True, help="Path to gold dataset JSONL")
    parser.add_argument("--vocab", required=True, help="Path to vocabulary JSONL")
    parser.add_argument("--hierarchy", help="Path to hierarchy JSONL (optional)")
    parser.add_argument("--out", default="reports", help="Output directory for reports")
    args = parser.parse_args()

    vocab, vocab_set = load_vocab(Path(args.vocab))
    gold = load_gold(Path(args.data))
    hierarchy = load_hierarchy(Path(args.hierarchy)) if args.hierarchy else None

    generate_tables(
        results_dir=Path(args.results_dir),
        gold_records=gold,
        vocab_valid_set=vocab_set,
        hierarchy=hierarchy,
        out_dir=Path(args.out),
    )


if __name__ == "__main__":
    main()

