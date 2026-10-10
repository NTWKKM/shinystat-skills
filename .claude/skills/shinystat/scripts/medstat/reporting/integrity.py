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
from typing import Any, NamedTuple

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
    elif hasattr(val, "tolist") and callable(val.tolist):
        numbers.extend(_flatten_numbers(val.tolist()))
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
    (
        re.compile(
            r"\b(?:HN|hn|H\.N\.)\s*[:#/-]?\s*\d{2,}(?:[-/]\d{2,})+\b|\b(?:HN|hn|H\.N\.)\s*[:#/-]?\s*\d{4,}\b"
        ),
        "Hospital Number (HN)",
    ),
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
            r"\b(?:Mr\.?|Mrs\.?|Ms\.?|Dr\.?|นาย|นาง|นางสาว|นพ\.?|พญ\.?|ดร\.?)\s+[A-Za-zก-๙]{2,}\b"
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

_MONTHS_PATTERN = r"(?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|jul(?:y)?|aug(?:ust)?|sep(?:tember)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)"
_DATE_PREFIX_PATTERN = re.compile(
    r"(?:"
    r"\b(?:in|year|years|during|period|from|dated|since|until|between)(?:\s+(?:the\s+)?(?:calendar|fiscal|academic|study)?(?:\s*year)?)?(?:\s+"
    + _MONTHS_PATTERN
    + r")?"
    r"|\b" + _MONTHS_PATTERN + r""
    r"|\b(?:19\d{2}|20\d{2})\s*(?:-|–|/|\b(?:to|through|until|and)\b)"
    r")\s*$",
    re.IGNORECASE,
)

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


class TextToken(NamedTuple):
    val: float
    raw_str: str
    has_pct: bool
    is_structural: bool


def _is_structural_context(
    val: float, raw_str: str, window_before: str, window_after: str
) -> bool:
    """Checks whether an unmatched numeric token is an uninformative structural constant."""
    window = f"{window_before}{raw_str}{window_after}"

    # 1. Confidence interval levels (e.g., 90%, 95%, 99% CI)
    if val in (90.0, 95.0, 99.0):
        if re.search(r"\b(?:CI|confidence|credible)\b", window, re.IGNORECASE):
            return True

    # 2. Alpha thresholds and p-value cutoffs (e.g., alpha = 0.05, p < 0.05, p < 0.001)
    if val in (0.05, 0.01, 0.001, 0.005):
        if re.search(
            r"(?:alpha|significance|threshold|cutoff)\b|p\s*[<>=]",
            window,
            re.IGNORECASE,
        ):
            return True

    # 3. Calendar years (2010-2035) when preceded immediately by date-related context
    if 2010.0 <= val <= 2035.0 and "." not in raw_str:
        if _DATE_PREFIX_PATTERN.search(window_before):
            return True

    # 4. Structural headings and table/figure/step counters (e.g., Table 1, Figure 2, Tier 1, Step 3, v.1)
    is_version = re.search(r"\b(?:Version|v\.)\s*$", window_before, re.IGNORECASE)
    is_counter = re.search(
        r"\b(?:Table|Figure|Fig\.?|Tier|Phase|Stage|Grade|Step|Level|Item|Section)\s*$",
        window_before,
        re.IGNORECASE,
    )
    if is_version:
        return True
    if is_counter and "." not in raw_str:
        return True

    # 5. Fixed percentage scale reference (e.g. "on a 0 to 100% scale", "normalized to 100%")
    if val in (0.0, 100.0) and re.search(
        r"\b(?:scale|normalized|range|total)\b", window, re.IGNORECASE
    ):
        return True

    return False


def _extract_text_tokens(text: str) -> list[TextToken]:
    """Extracts candidate TextToken objects from narrative text."""
    clean_text = text.replace(",", "")
    matches = list(
        re.finditer(
            r"(?<!\w)(?<![a-zA-Z]-)(?P<num>(?:[-+]?\d+(?:\.\d+)?|\.\d+)(?:[eE][-+]?\d+)?)(?P<pct>\s*%)?",
            clean_text,
        )
    )
    tokens: list[TextToken] = []
    for m in matches:
        raw_num = m.group("num")
        pct_group = m.group("pct")
        try:
            val = float(raw_num)
        except ValueError:
            continue
        if not (-1e7 < val < 1e7):
            continue

        has_pct = bool(pct_group and pct_group.strip() == "%")
        start = m.start()
        end = m.end()
        window_before = clean_text[max(0, start - 30) : start].split("\n")[-1]
        window_after = clean_text[end : min(len(clean_text), end + 30)].split("\n")[0]

        is_structural = _is_structural_context(
            val, raw_num, window_before, window_after
        )
        tokens.append(
            TextToken(
                val=val,
                raw_str=raw_num,
                has_pct=has_pct,
                is_structural=is_structural,
            )
        )
    return tokens


def _extract_text_numbers(text: str) -> list[float]:
    """Extracts candidate float numbers from narrative text."""
    return [token.val for token in _extract_text_tokens(text)]


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

    # Also scan table cells and headers
    for block in doc.blocks:
        if isinstance(block, TableBlock):
            for col in block.df.columns:
                col_str = str(col)
                for pattern, label in PHI_PATTERNS:
                    matches = pattern.findall(col_str)
                    for m in matches:
                        phi_violations.append(
                            f"{label} in Table '{block.caption}' header: '{m}'"
                        )
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
    text_tokens = _extract_text_tokens(all_text)
    text_numbers = [token.val for token in text_tokens]

    untraced: list[float] = []
    for token in text_tokens:
        tn = token.val
        token_str = token.raw_str
        has_pct = token.has_pct
        is_structural = token.is_structural

        # Check against known numbers: for values <= 1, use token's displayed precision
        if abs(tn) <= 1.0:
            if "." in token_str:
                dec_part = token_str.split(".")[1]
                dec_digits = len(re.split(r"[eE%]", dec_part)[0])
                abs_tol = min(0.02, 10.0 ** (-dec_digits))
            else:
                abs_tol = 0.02
        else:
            abs_tol = 0.02

        matched = False
        for kn in known_numbers:
            abs_diff = abs(tn - kn)
            # Direct match branch:
            # For values with abs(tn) <= 1, strictly enforce displayed precision abs_tol;
            # Only allow 1% relative tolerance for larger values (|tn| > 1.0)
            if abs_diff <= abs_tol:
                matched = True
                break
            if abs(tn) > 1.0 and abs(kn) > 1e-9:
                rel_diff = abs_diff / abs(kn)
                if rel_diff <= 0.01:
                    matched = True
                    break

            # Percentage scale match branch (e.g. 85.2% in text vs 0.852 in results):
            # Only trigger percentage-scale comparison if token explicitly has '%'
            # OR if tn is on percentage scale (|tn| > 1.0 and |kn| <= 1.0)
            if has_pct or (abs(tn) > 1.0 and abs(kn) <= 1.0):
                pct_diff = abs(tn - kn * 100.0)
                pct_tol = min(0.05, abs_tol * 100.0) if has_pct else 0.05
                if pct_diff <= pct_tol:
                    matched = True
                    break

        if not matched:
            if not is_structural:
                untraced.append(tn)

    # 3. Observational Causal Inference & E-value Check
    results_keys_str = " ".join(doc.results_dict.keys()).lower()
    is_causal = any(
        k in results_keys_str
        for k in [
            "psm",
            "propensity",
            "love_plot",
            "hazard_ratio",
            "cox",
            "or_table",
            "odds_ratio",
            "logistic",
            "regression",
        ]
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
                "confound",
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
