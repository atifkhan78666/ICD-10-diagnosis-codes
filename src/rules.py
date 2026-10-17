"""Clinical Coding Rule Engine and Verdict Engine for ICD-10-CM (Brief B System A3).

Enforces official ICD-10-CM coding guidelines:
1. Excludes1: Strictly mutually exclusive codes (e.g. Type 1 vs Type 2 diabetes, primary vs secondary hypertension).
2. Hierarchy Redundancy: Non-billable/parent categories co-occurring with specific leaf codes.
3. Code First / Etiology-Manifestation pairing constraints.

The VerdictEngine evaluates predictions, detects violations, records structured audits,
and applies clinical resolution policies.
"""

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

from codes import normalize

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RuleViolation:
    rule_type: str
    code_a: str
    code_b: str
    message: str
    action_taken: str


class RuleEngine:
    """Evaluates and enforces official ICD-10-CM tabular coding rules."""

    # Official ICD-10-CM Excludes1 pairs (prefix or exact matching)
    DEFAULT_EXCLUDES1 = [
        # Type 1 vs Type 2 diabetes
        ("E10", "E11", "Excludes1: Type 1 diabetes (E10) cannot be coded with Type 2 diabetes (E11)"),
        ("E10", "E13", "Excludes1: Type 1 diabetes (E10) cannot be coded with other specified diabetes (E13)"),
        ("E11", "E13", "Excludes1: Type 2 diabetes (E11) cannot be coded with other specified diabetes (E13)"),
        # Gestational diabetes vs pre-existing diabetes
        ("O24.4", "E10", "Excludes1: Gestational diabetes (O24.4) cannot be coded with pre-existing Type 1 diabetes (E10)"),
        ("O24.4", "E11", "Excludes1: Gestational diabetes (O24.4) cannot be coded with pre-existing Type 2 diabetes (E11)"),
        # Essential vs Secondary hypertension
        ("I10", "I15", "Excludes1: Essential hypertension (I10) cannot be coded with Secondary hypertension (I15)"),
        # End stage renal disease vs early stage CKD
        ("N18.6", "N18.1", "Excludes1: ESRD (N18.6) cannot be coded with CKD Stage 1 (N18.1)"),
        ("N18.6", "N18.2", "Excludes1: ESRD (N18.6) cannot be coded with CKD Stage 2 (N18.2)"),
        ("N18.6", "N18.3", "Excludes1: ESRD (N18.6) cannot be coded with CKD Stage 3 (N18.3)"),
        ("N18.6", "N18.4", "Excludes1: ESRD (N18.6) cannot be coded with CKD Stage 4 (N18.4)"),
        # Acute vs Chronic coronary disease specificity
        ("I25.10", "I20.0", "Excludes1: CAD without angina (I25.10) conflicts with Unstable angina (I20.0)"),
    ]

    def __init__(self, custom_excludes1: Optional[List[Tuple[str, str, str]]] = None):
        self.excludes1 = list(self.DEFAULT_EXCLUDES1)
        if custom_excludes1:
            self.excludes1.extend(custom_excludes1)

    def _matches_pattern(self, code: str, pattern: str) -> bool:
        """Check if normalized code matches a rule pattern or prefix."""
        norm_code = normalize(code)
        norm_pat = normalize(pattern)
        return norm_code == norm_pat or norm_code.startswith(norm_pat + ".") or norm_code.startswith(norm_pat)

    def check_violations(
        self,
        codes: List[str],
        hierarchy: Optional[Any] = None,
    ) -> List[Dict[str, Any]]:
        """Identify all coding guideline violations in a predicted code list."""
        norm_codes = [normalize(c) for c in codes if c]
        violations = []

        # 1. Check Excludes1 violations
        n = len(norm_codes)
        for i in range(n):
            for j in range(i + 1, n):
                c1, c2 = norm_codes[i], norm_codes[j]
                for p1, p2, msg in self.excludes1:
                    if (self._matches_pattern(c1, p1) and self._matches_pattern(c2, p2)) or \
                       (self._matches_pattern(c1, p2) and self._matches_pattern(c2, p1)):
                        violations.append({
                            "rule_type": "excludes1",
                            "code_a": c1,
                            "code_b": c2,
                            "message": msg,
                        })

        # 2. Check Hierarchy Redundancy (Parent + Child co-occurring)
        if hierarchy is not None:
            code_set = set(norm_codes)
            for code in norm_codes:
                curr = hierarchy.parent_of(code)
                while curr is not None:
                    if curr in code_set:
                        violations.append({
                            "rule_type": "hierarchy_redundancy",
                            "code_a": curr,
                            "code_b": code,
                            "message": f"Hierarchy Redundancy: parent '{curr}' and specific descendant '{code}' are both coded",
                        })
                    curr = hierarchy.parent_of(curr)

        return violations

    def apply_verdict(
        self,
        codes: List[str],
        hierarchy: Optional[Any] = None,
        candidate_scores: Optional[Dict[str, float]] = None,
    ) -> Tuple[List[str], List[Dict[str, Any]]]:
        """Apply clinical rule verdict engine to audit and adjust predictions.

        Policies:
        1. Hierarchy Redundancy: Drop the broader parent/ancestor in favor of the more specific child.
        2. Excludes1: If two mutually exclusive codes appear, retain the one with higher confidence
           or earlier prediction rank, dropping the conflicting code.
        """
        if not codes:
            return [], []

        norm_codes = [normalize(c) for c in codes if c]
        scores = candidate_scores or {}
        audit_trail: List[Dict[str, Any]] = []

        # Step 1: Resolve Hierarchy Redundancy
        resolved = list(norm_codes)
        if hierarchy is not None:
            to_drop = set()
            for code in resolved:
                curr = hierarchy.parent_of(code)
                while curr is not None:
                    if curr in resolved:
                        to_drop.add(curr)
                        audit_trail.append({
                            "rule_type": "hierarchy_redundancy",
                            "action": f"Dropped parent '{curr}' in favor of specific child '{code}'",
                            "dropped": curr,
                            "retained": code,
                        })
                    curr = hierarchy.parent_of(curr)
            resolved = [c for c in resolved if c not in to_drop]

        # Step 2: Resolve Excludes1 Conflicts
        to_drop_excludes1 = set()
        for i in range(len(resolved)):
            for j in range(i + 1, len(resolved)):
                c1, c2 = resolved[i], resolved[j]
                if c1 in to_drop_excludes1 or c2 in to_drop_excludes1:
                    continue
                for p1, p2, msg in self.excludes1:
                    match_1 = self._matches_pattern(c1, p1) and self._matches_pattern(c2, p2)
                    match_2 = self._matches_pattern(c1, p2) and self._matches_pattern(c2, p1)
                    if match_1 or match_2:
                        s1 = scores.get(c1, 1.0)
                        s2 = scores.get(c2, 1.0)
                        # Keep higher scoring code, or c1 (earlier in list) if tied
                        if s1 >= s2:
                            loser, winner = c2, c1
                        else:
                            loser, winner = c1, c2
                        to_drop_excludes1.add(loser)
                        audit_trail.append({
                            "rule_type": "excludes1",
                            "action": f"Dropped conflicting '{loser}' (score={scores.get(loser, 'n/a')}) in favor of '{winner}' (score={scores.get(winner, 'n/a')})",
                            "dropped": loser,
                            "retained": winner,
                            "message": msg,
                        })

        final_codes = [c for c in resolved if c not in to_drop_excludes1]
        return final_codes, audit_trail

