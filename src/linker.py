"""Entity Linker: Maps clinical mentions (medications and concepts) to ICD-10 diagnosis candidates.

Used by Knowledge Graph System A2 and System A3 to supplement semantic retrieval
with deterministic clinical pharmacology and ontology associations.
Strictly respects retrieval_view: never consumes gold diagnosis labels.
"""

import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple
import yaml

from codes import normalize

logger = logging.getLogger(__name__)


class EntityLinker:
    """Clinical entity and medication-to-condition knowledge linker."""

    def __init__(self, templates_path: Optional[str] = None):
        if templates_path is None:
            templates_path = str(Path(__file__).resolve().parent / "templates" / "conditions.yaml")
        self.templates_path = Path(templates_path)
        self.med_to_codes: Dict[str, List[Tuple[str, float]]] = {}
        self.phrase_to_codes: Dict[str, List[Tuple[str, float]]] = {}
        self._load_templates()

    def _extract_drug_name(self, med_str: str) -> str:
        """Extract canonical base drug name from dosage string (e.g. 'metformin 500 mg BID' -> 'metformin')."""
        clean = re.sub(r'\d+.*$', '', med_str.strip().lower()).strip()
        # Remove common trailing words
        clean = re.sub(r'\b(tablet|capsule|injection|solution|cream|patch|inhaler)\b', '', clean).strip()
        return clean.split('/')[0].strip()

    def _load_templates(self) -> None:
        if not self.templates_path.exists():
            logger.warning(f"EntityLinker template file not found at {self.templates_path}")
            return

        with open(self.templates_path, "r", encoding="utf-8") as f:
            conditions = yaml.safe_load(f)

        if not conditions or not isinstance(conditions, list):
            return

        for item in conditions:
            code = normalize(item.get("code", ""))
            if not code:
                continue

            # Link medications from implicit signals
            signals = item.get("implicit_signals", {}) or {}
            meds = signals.get("medications", []) or []
            distractors = item.get("distractor_meds", []) or []

            for med in meds:
                base_drug = self._extract_drug_name(med)
                if base_drug and len(base_drug) >= 3:
                    self.med_to_codes.setdefault(base_drug, []).append((code, 0.95))

            for dist in distractors:
                if dist:
                    base_drug = self._extract_drug_name(dist)
                    if base_drug and len(base_drug) >= 3:
                        self.med_to_codes.setdefault(base_drug, []).append((code, 0.75))

            # Link explicit phrases
            phrases = item.get("explicit_phrases", []) or []
            for phrase in phrases:
                phrase_clean = phrase.strip().lower()
                if len(phrase_clean) >= 3:
                    self.phrase_to_codes.setdefault(phrase_clean, []).append((code, 0.90))

        logger.info(
            f"EntityLinker loaded: {len(self.med_to_codes)} drug keywords, "
            f"{len(self.phrase_to_codes)} clinical phrases."
        )

    def link_medications(self, medications: List[str]) -> List[Dict[str, Any]]:
        """Link a patient record's medication list to candidate diagnosis codes."""
        candidates = []
        seen = set()

        for med in medications:
            med_lower = med.lower()
            for drug_kw, code_list in self.med_to_codes.items():
                if re.search(r'\b' + re.escape(drug_kw) + r'\b', med_lower):
                    for code, conf in code_list:
                        if (code, drug_kw) not in seen:
                            seen.add((code, drug_kw))
                            candidates.append({
                                "code": code,
                                "source_drug": drug_kw,
                                "score": conf,
                                "source_query": f"linker_med:{drug_kw}",
                            })
        return candidates

    def link_text(self, text: str) -> List[Dict[str, Any]]:
        """Identify explicit condition concept mentions in clinical text."""
        candidates = []
        seen = set()
        text_lower = text.lower()

        for phrase, code_list in self.phrase_to_codes.items():
            if re.search(r'\b' + re.escape(phrase) + r'\b', text_lower):
                for code, conf in code_list:
                    if (code, phrase) not in seen:
                        seen.add((code, phrase))
                        candidates.append({
                            "code": code,
                            "source_phrase": phrase,
                            "score": conf,
                            "source_query": f"linker_text:{phrase}",
                        })
        return candidates

    def link_record(self, record: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract all candidate diagnoses via pharmacology and entity mentions."""
        meds = record.get("medications", [])
        text = record.get("text", "")
        med_cands = self.link_medications(meds)
        text_cands = self.link_text(text)

        # Merge, deduplicating by code and keeping highest score
        merged: Dict[str, Dict[str, Any]] = {}
        for c in med_cands + text_cands:
            code = c["code"]
            if code not in merged or c["score"] > merged[code]["score"]:
                merged[code] = c

        return list(merged.values())

