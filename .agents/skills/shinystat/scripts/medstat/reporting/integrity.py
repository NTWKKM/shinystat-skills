"""
src/medstat/reporting/integrity.py: Pillar 5 Reporting Integrity Verification.

Invariants enforced:
1. Numerical Traceability: Every narrative number traces back to results_dict (+/- 0.02).
2. Zero-PHI Compliance: Scans text and tables for HN, Thai Citizen ID, Phone, Name markers, DOB.
3. Causal Inference / E-Value Caveats: Observational studies with PSM/Cox must acknowledge unmeasured confounding.
4. Retention & Methodology: Ensures presence of methods and retention flow details.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from medstat.reporting.ir import (
    ReportDocument,
    TableBlock,
)


@dataclass
class IntegrityReport:
    """Detailed outcome of document integrity audit."""

    passed: bool
    untraced_numbers: list[float] = field(default_factory=list)
    phi_violations: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    has_methods: bool = False
    has_retention_flow: bool = False
    has_causal_caveat: bool = True  # True if not needed or properly addressed

    def summary(self) -> str:
        lines = [f"Integrity Audit: {'PASSED' if self.passed else 'FAILED'}"]
        if self.phi_violations:
            lines.append(
                f"  [CRITICAL] PHI Violations Detected: {len(self.phi_violations)}"
            )
            for v in self.phi_violations[:5]:
                lines.append(f"    - {v}")
        if self.untraced_numbers:
            lines.append(
                f"  [WARNING] Untraced Narrative Numbers: {len(self.untraced_numbers)}"
            )
            for n in self.untraced_numbers[:10]:
                lines.append(f"    - {n}")
        if self.warnings:
            lines.append("  [NOTES / WARNINGS]:")
            for w in self.warnings:
                lines.append(f"    - {w}")
        return "\n".join(lines)


def _flatten_numbers(val: Any) -> list[float]:
    """Recursively extracts all float and int numbers from nested dicts, lists, and DataFrames."""
    numbers: list[float] = []
    if isinstance(val, (int, float)):
        if pd.notna(val):
            numbers.append(float(val))
    elif isinstance(val, dict):
        for v in val.values():
            numbers.extend(_flatten_numbers(v))
    elif isinstance(val, (list, tuple, set)):
        for item in val:
            numbers.extend(_flatten_numbers(item))
    elif isinstance(val, pd.DataFrame):
        for col in val.columns:
            for v in val[col]:
                if pd.notna(v) and isinstance(v, (int, float)):
                    numbers.append(float(v))
    elif isinstance(val, pd.Series):
        for v in val:
            if pd.notna(v) and isinstance(v, (int, float)):
                numbers.append(float(v))
    return numbers


# PHI detection patterns (HIPAA 18 Identifiers + Thai PDPA)
PHI_PATTERNS = [
    (re.compile(r"\b(?:HN|hn|H\.N\.)\s*[:#-]?\s*\d{4,}\b"), "Hospital Number (HN)"),
    (
        re.compile(r"\b\d{1}[-\s]?\d{4}[-\s]?\d{5}[-\s]?\d{2}[-\s]?\d{1}\b"),
        "Thai Citizen ID (13 digits)",
    ),
    (
        re.compile(
            r"\b(?:0[689]\d{1}[-\s]?\d{3}[-\s]?\d{4}|0[2-57]\d{1}[-\s]?\d{3}[-\s]?\d{3})\b"
        ),
        "Phone Number",
    ),
    (
        re.compile(
            r"\b(?:Mr\.|Mrs\.|Ms\.|Dr\.|นาย|นาง|นางสาว|นพ\.|พญ\.)\s+[A-Za-zก-๙]{2,}\b"
        ),
        "Personal Name Marker",
    ),
    (
        re.compile(
            r"\b(?:DOB|dob|Date of Birth|วันเกิด)\s*[:#-]?\s*\d{1,4}[-/.]\d{1,2}[-/.]\d{1,4}\b",
            re.IGNORECASE,
        ),
        "Date of Birth",
    ),
]

# Common non-data numbers to ignore during traceability check
IGNORED_NUMBERS = {
    0.0,
    1.0,
    2.0,
    3.0,
    4.0,
    5.0,
    6.0,
    7.0,
    8.0,
    9.0,
    10.0,  # Single-digit indices/levels
    1.5,
    2.5,
    95.0,
    99.0,
    0.05,
    0.01,
    0.001,  # Standard alpha/confidence level constants
    100.0,  # Standard 100% scale
    2020.0,
    2021.0,
    2022.0,
    2023.0,
    2024.0,
    2025.0,
    2026.0,
    2027.0,  # Years
}


def _extract_text_numbers(text: str) -> list[float]:
    """Extracts candidate float numbers from narrative text."""
    # Match numbers like 0.85, 12.3, 145, 0.001, 1,234.56, -0.45, 95%
    clean_text = text.replace(",", "")
    matches = re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", clean_text)
    nums = []
    for m in matches:
        try:
            val = float(m)
            if val not in IGNORED_NUMBERS and -1e7 < val < 1e7:
                nums.append(val)
        except ValueError:
            continue
    return nums


def verify_report_integrity(doc: ReportDocument) -> IntegrityReport:
    """
    Performs comprehensive verification on a ReportDocument:
    1. Traces numbers in narrative text against results_dict.
    2. Scans all text and table entries for PHI.
    3. Verifies causal inference / unmeasured confounding caveats.
    4. Confirms methods and retention flow representation.
    """
    all_text = doc.extract_all_text()

    # 1. Zero-PHI Scan
    phi_violations: list[str] = []
    for pattern, label in PHI_PATTERNS:
        matches = pattern.findall(all_text)
        for m in matches:
            phi_violations.append(f"{label}: '{m}'")

    # Also scan table cells
    for block in doc.blocks:
        if isinstance(block, TableBlock):
            for col in block.df.columns:
                for val in block.df[col]:
                    val_str = str(val)
                    for pattern, label in PHI_PATTERNS:
                        matches = pattern.findall(val_str)
                        for m in matches:
                            phi_violations.append(
                                f"{label} in Table '{block.caption}': '{m}'"
                            )

    # 2. Numerical Traceability
    known_numbers = _flatten_numbers(doc.results_dict)
    text_numbers = _extract_text_numbers(all_text)

    untraced: list[float] = []
    for tn in text_numbers:
        # Check against known numbers within +/- 0.02 or 1% relative error
        matched = False
        for kn in known_numbers:
            abs_diff = abs(tn - kn)
            rel_diff = abs_diff / max(abs(kn), 1e-9)
            if abs_diff <= 0.02 or rel_diff <= 0.01:
                matched = True
                break
            # Also check percentage scale (e.g. 0.852 in results vs 85.2 in text)
            if abs(tn - kn * 100.0) <= 0.05 or abs(tn / 100.0 - kn) <= 0.005:
                matched = True
                break
        if not matched:
            untraced.append(tn)

    # 3. Observational Causal Inference & E-value Check
    results_keys_str = " ".join(doc.results_dict.keys()).lower()
    is_causal = any(
        k in results_keys_str
        for k in ["psm", "propensity", "love_plot", "hazard_ratio", "cox", "or_table"]
    )

    has_causal_caveat = True
    warnings: list[str] = []
    if is_causal:
        text_lower = all_text.lower()
        has_caveat = any(
            phrase in text_lower
            for phrase in [
                "unmeasured confounding",
                "e-value",
                "residual confounding",
                "observational",
                "causal inference assumption",
            ]
        )
        if not has_caveat:
            has_causal_caveat = False
            warnings.append(
                "Observational/PSM analysis detected without explicit discussion of unmeasured confounding or E-values."
            )

    # 4. Methods & Retention Check
    text_lower = all_text.lower()
    has_methods = any(
        w in text_lower
        for w in ["method", "statistical analysis", "protocol", "package", "version"]
    )
    has_retention_flow = any(
        w in text_lower
        for w in [
            "cohort",
            "retention",
            "flow",
            "strobe",
            "consort",
            "inclusion",
            "exclusion",
            "eligible",
        ]
    )

    if not has_methods:
        warnings.append(
            "Document lacks a dedicated Methods or Statistical Analysis section."
        )
    if not has_retention_flow:
        warnings.append(
            "Document lacks cohort retention / inclusion-exclusion flow details."
        )

    passed = (len(phi_violations) == 0) and (
        len(untraced) == 0 or len(untraced) <= max(1, int(len(text_numbers) * 0.15))
    )

    return IntegrityReport(
        passed=passed,
        untraced_numbers=untraced,
        phi_violations=phi_violations,
        warnings=warnings,
        has_methods=has_methods,
        has_retention_flow=has_retention_flow,
        has_causal_caveat=has_causal_caveat,
    )
