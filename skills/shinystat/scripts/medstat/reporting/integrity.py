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

from medstat._phi import PHI_PATTERNS
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
    citations_count: int = 0
    agent_remediation: list[str] = field(default_factory=list)

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


_P_KEY_RE = re.compile(
    r"^(?:p|p[-_]?(?:val|value|adj|overall|non[-_]?linear|omnibus|q|lrt|wald|trend|interaction))s?$",
    re.IGNORECASE,
)


def _detect_p_qualifier(key_str: str) -> str | None:
    """Infers specialized p-value qualifier from key or context string."""
    k_lower = key_str.lower()
    if "interaction" in k_lower:
        return "interaction"
    if "trend" in k_lower:
        return "trend"
    if "non" in k_lower and "linear" in k_lower:
        return "nonlinear"
    if "omnibus" in k_lower:
        return "omnibus"
    if "lrt" in k_lower or "likelihood" in k_lower:
        return "lrt"
    if "wald" in k_lower:
        return "wald"
    return None


def _extract_qualified_p_values(
    val: Any, parent_qualifier: str | None = None
) -> list[tuple[str | None, float]]:
    """Recursively extracts (qualifier, p_value) pairs from dictionary keys or DataFrame columns.

    For dictionary-valued records, recurses into their p-value fields instead of passing
    the entire record to _flatten_numbers, ensuring non-p-value fields (e.g. estimates,
    test statistics) are not treated as p-values.
    """
    pvals: list[tuple[str | None, float]] = []
    if isinstance(val, dict):
        for k, v in val.items():
            k_str = str(k)
            k_qual = _detect_p_qualifier(k_str) or parent_qualifier
            if _P_KEY_RE.match(k_str):
                if isinstance(v, dict):
                    pvals.extend(
                        _extract_qualified_p_values(v, parent_qualifier=k_qual)
                    )
                elif isinstance(v, pd.DataFrame):
                    pvals.extend(
                        _extract_qualified_p_values(v, parent_qualifier=k_qual)
                    )
                else:
                    for n in _flatten_numbers(v):
                        if 0.0 <= n <= 1.0:
                            pvals.append((k_qual, float(n)))
            else:
                pvals.extend(_extract_qualified_p_values(v, parent_qualifier=k_qual))
    elif isinstance(val, (list, tuple, set)):
        for item in val:
            pvals.extend(
                _extract_qualified_p_values(item, parent_qualifier=parent_qualifier)
            )
    elif isinstance(val, pd.DataFrame):
        for col in val.columns:
            col_str = str(col)
            col_qual = _detect_p_qualifier(col_str) or parent_qualifier
            if _P_KEY_RE.match(col_str):
                for n in _flatten_numbers(val[col]):
                    if 0.0 <= n <= 1.0:
                        pvals.append((col_qual, float(n)))
    return pvals


def _flatten_p_values(val: Any) -> list[float]:
    """Recursively extracts numeric p-values from dictionary keys or DataFrame columns."""
    return [pv for _, pv in _extract_qualified_p_values(val)]


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
    is_p_floor: bool = False
    p_inequality: str | None = None
    p_qualifier: str | None = None


def _is_structural_context(
    val: float,
    raw_str: str,
    window_before: str,
    window_after: str,
    doc_has_references: bool = False,
) -> bool:
    """Checks whether an unmatched numeric token is an uninformative structural constant.

    The windows provide surrounding same-line context for ``raw_str``.
    Recognized context includes citation markers, exponents, and calendar or
    publication years from 1900 through 2035. ``doc_has_references`` enables
    additional exemptions for author-style bibliography entries numbered 1–100;
    it indicates a reference heading anywhere in the document, not that the
    token itself is in a reference section.
    """
    window = f"{window_before}{raw_str}{window_after}"

    # 1. Confidence interval levels (e.g., 90%, 95%, 99% CI)
    if val in (90.0, 95.0, 99.0):
        if re.search(r"\b(?:CI|confidence|credible)\b", window, re.IGNORECASE):
            return True

    # 2. Alpha thresholds and p-value cutoffs (e.g., alpha = 0.05, significance threshold 0.05)
    if val in (0.05, 0.01, 0.001, 0.005, 0.0001):
        has_threshold_kw = bool(
            re.search(
                r"\b(?:alpha|significance\s+(?:level|threshold|cutoff)|threshold|cutoff)\b",
                window,
                re.IGNORECASE,
            )
        )
        is_direct_p_eq = bool(
            re.search(r"\bp\s*(?:-\s*value\s*)?=\s*$", window_before, re.IGNORECASE)
            and not re.search(
                r"\b(?:alpha|significance\s+(?:level|threshold|cutoff)|threshold|cutoff)\b",
                window_before,
                re.IGNORECASE,
            )
        )
        is_p_ineq = bool(
            re.search(
                r"\bp(?=[-_\s(\<>=≤≥])\s*(?:[-_\s]*(?:value|val|\(?for\s+[a-zA-Z_-]+\)?|\(?[a-zA-Z_-]+\)?))?\s*([<>]=?|[≤≥])\s*$",
                window_before,
                re.IGNORECASE,
            )
        )
        is_threshold_def = bool(
            re.search(
                r"\b(?:alpha|significance\s+(?:level|threshold|cutoff)|threshold|cutoff|prespecified(?:\s+(?:level|threshold|cutoff))?|defined\s+as)"
                r"(?:\s*(?:[:=]|\b(?:was|were|is)?\s*(?:set\s+at|defined\s+as|of|at)?\b))?\s*"
                r"\bp(?=[-_\s(\<>=≤≥])\s*(?:[-_\s]*(?:value|val|\(?for\s+[a-zA-Z_-]+\)?|\(?[a-zA-Z_-]+\)?))?\s*([<>]=?|[≤≥])\s*$",
                window_before,
                re.IGNORECASE,
            )
        )
        if (
            has_threshold_kw
            and not is_direct_p_eq
            and (not is_p_ineq or is_threshold_def)
        ):
            return True

    # 3. Calendar & Publication years (1900-2035) when preceded by date or journal/citation context
    if 1900.0 <= val <= 2035.0 and "." not in raw_str:
        if _DATE_PREFIX_PATTERN.search(window_before):
            return True
        if re.search(
            r"(?:\b(?:et\s+al\.?|JAMA|NEJM|Lancet|BMJ|Crit\s+Care|doi|vol\.?)\b|\bpp?\.)",
            window_before,
            re.IGNORECASE,
        ):
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

    # 6. Mathematical or unit exponents (e.g. R^2, m^2, x^2, cm^2)
    if "." not in raw_str and re.search(r"[a-zA-Z\)]\^$", window_before):
        return True

    # 7. Superscript citations and bibliographic reference markers (e.g., <sup>1</sup>, <sup>12,13</sup>, ^[1], [^1], [1], Ref 1, 1. Author)
    if "." not in raw_str:
        # 7a. Inside HTML superscript tags: <sup>...</sup>
        if "<sup" in window_before and "</sup>" in window_after:
            last_sup_open = window_before.rfind("<sup")
            last_sup_close = window_before.rfind("</sup>")
            if last_sup_open > last_sup_close:
                return True

        # 6b. Footnote / caret citation reference: [^1] or ^[1]
        if re.search(r"\[\^\s*$", window_before) and window_after.startswith("]"):
            return True
        if re.search(r"\^\[(?:[\d\s,]+|[\d\s,]*[-–]\s*)*$", window_before) and re.match(
            r"^(?:[\s,\-–]+\d+)*\s*\](?![a-zA-Z])", window_after
        ):
            return True

        # 6c. Bracketed citation reference: [1], [1, 2], [1-3]
        # Bounded by plausible citation range (1 <= val <= 100) and clean bracket closure
        if (
            1.0 <= val <= 100.0
            and re.search(r"\[(?:[\d\s,]+|[\d\s,]*[-–]\s*)*$", window_before)
            and re.match(
                r"^(?:[\s,\-–]+\d+)*\s*\](?![a-zA-Z])",
                window_after,
            )
        ):
            return True

        # 6d. Reference or citation label: Ref. 1, Reference 1, Citation 2
        if re.search(
            r"\b(?:Ref\.?|Refs\.?|Reference|References|Citation|Citations)\s*#?$",
            window_before,
            re.IGNORECASE,
        ):
            return True

        # 6e. Bibliographic reference list item at start of paragraph/line: "1. Smith J...", "1 Smith J...", "1 O'Neil J...", "[1] McDonald A..."
        # Requires document to have a References/Bibliography section, plausible range (1 <= val <= 100),
        # and author/bibliographic format
        if (
            doc_has_references
            and 1.0 <= val <= 100.0
            and re.match(r"^\s*(?:\[\s*)?$", window_before)
            and re.match(
                r"^\s*(?:\]\s*)?[\.\)]?\s+[A-Z](?:['’]?[a-zA-Z]+)+(?:\s+[A-Z]{1,3}\b|\s*,\s*[A-Z]{1,3}\b|\s+et\s+al)",
                window_after,
            )
        ):
            return True

    return False


def _extract_text_tokens(text: str) -> list[TextToken]:
    """Extracts candidate TextToken objects from narrative text.

    Normalizes thousands separators and preserves token order and duplicates.
    Percentages retain their displayed scale (85% yields ``val=85.0`` with
    ``has_pct=True``). Only values strictly between -1e7 and 1e7 are returned,
    with raw numeric strings and flags identifying structural context.
    """
    clean_text = re.sub(r"(?<=\d),(?=\d{3}\b)", "", text)
    doc_has_references = bool(
        re.search(
            r"(?:^|\n)\s*(?:#{1,4}\s*)?(?:References|Bibliography)\b",
            text,
            re.IGNORECASE,
        )
    )
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
        window_before = clean_text[max(0, start - 80) : start].split("\n")[-1]
        window_after = clean_text[end : min(len(clean_text), end + 80)].split("\n")[0]

        is_structural = _is_structural_context(
            val,
            raw_num,
            window_before,
            window_after,
            doc_has_references=doc_has_references,
        )
        p_ineq_match = re.search(
            r"(?:((?:[a-zA-Z_-]+(?:\s+(?:ratio\s+)?(?:test|score|statistic))?|[a-zA-Z_-]+))\s*,?\s+)?\bp(?=[-_\s(\<>=≤≥])\s*(?:[-_\s]*(?:value|val|\(?for\s+([a-zA-Z_-]+)\)?|\(?([a-zA-Z_-]+)\)?))?\s*([<>]=?|[≤≥])\s*$",
            window_before,
            re.IGNORECASE,
        )
        p_val = val / 100.0 if has_pct else val
        p_inequality: str | None = None
        p_qualifier: str | None = None
        if 0.0 <= p_val <= 1.0 and not is_structural and p_ineq_match:
            qual_raw = (
                p_ineq_match.group(2) or p_ineq_match.group(3) or p_ineq_match.group(1)
            )
            raw_op = p_ineq_match.group(4)
            p_inequality = (
                "<=" if raw_op == "≤" else (">=" if raw_op == "≥" else raw_op)
            )
            p_qualifier = _detect_p_qualifier(qual_raw) if qual_raw else None

        is_p_floor = bool(p_inequality in ("<", "<="))
        tokens.append(
            TextToken(
                val=val,
                raw_str=raw_num,
                has_pct=has_pct,
                is_structural=is_structural,
                is_p_floor=is_p_floor,
                p_inequality=p_inequality,
                p_qualifier=p_qualifier,
            )
        )
    return tokens


def _extract_text_numbers(text: str) -> list[float]:
    """Extracts candidate float numbers from narrative text."""
    return [token.val for token in _extract_text_tokens(text)]


def verify_report_integrity(
    doc: ReportDocument,
    allowed_untraced: set[float] | list[float] | tuple[float, ...] | None = None,
) -> IntegrityReport:
    """
    Audits a ReportDocument and returns an IntegrityReport:
    1. Traces numbers in narrative text against results_dict.
    2. Scans all text and table entries for PHI.
    3. Checks confounding caveats when results_dict keys suggest causal analyses.
    4. Checks for methods and retention keywords.

    Structural constants and values explicitly listed in ``allowed_untraced``
    are exempt from numerical tracing. ``passed`` requires no PHI matches,
    any required confounding caveat, and zero untraced numbers
    (``len(untraced) == 0``). Missing methods or retention keywords only add warnings.

    Numerical matching uses a half-unit tolerance based on claimed precision:
    ``abs(val - kn) <= 0.5 * 10**(-d) + 1e-9`` where ``d`` is the number of
    decimal digits displayed in the narrative text (d=0 for integers).
    For example, P=0.01 passes when true P=0.012 (correct rounding to 2 decimals),
    whereas 0.02 and 0.013 are rejected. Percentage-scale matching against kn*100
    is only permitted when the token explicitly carries a '%' sign.
    The ``allowed_untraced`` whitelist enforces the identical half-unit tolerance.

    ``citations_count`` counts matched citation groups, not unique references;
    bracketed bibliography labels can also count. Citation sources and their
    agreement with the narrative are not verified.
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
    qualified_p_values = _extract_qualified_p_values(doc.results_dict)
    # Require an actual identified p-value in results_dict to support p-value inequalities
    allowed_list = (
        [float(x) for x in allowed_untraced] if allowed_untraced is not None else []
    )
    text_tokens = _extract_text_tokens(all_text)

    untraced: list[float] = []
    for token in text_tokens:
        tn = token.val
        token_str = token.raw_str
        has_pct = token.has_pct
        is_structural = token.is_structural
        is_p_floor = token.is_p_floor
        p_inequality = getattr(token, "p_inequality", None)
        token_qual = getattr(token, "p_qualifier", None)

        if p_inequality is not None or is_p_floor:
            p_bound = tn / 100.0 if has_pct else tn
            op = p_inequality if p_inequality is not None else "<"

            # Filter candidate p-values to preserve association between claim and its relevant p-value
            if token_qual is not None:
                candidate_pvals = [
                    pv for qual, pv in qualified_p_values if qual == token_qual
                ]
            else:
                candidate_pvals = [
                    pv for qual, pv in qualified_p_values if qual is None
                ]

            if not candidate_pvals:
                matched = False
            elif op == "<":
                matched = any(0.0 <= pv < p_bound for pv in candidate_pvals)
            elif op == "<=":
                matched = any(0.0 <= pv <= p_bound for pv in candidate_pvals)
            elif op == ">":
                matched = any(p_bound < pv <= 1.0 for pv in candidate_pvals)
            elif op == ">=":
                matched = any(p_bound <= pv <= 1.0 for pv in candidate_pvals)
            else:
                matched = False

            if not matched and not is_structural:
                untraced.append(tn)
            continue

        # Determine decimal precision d from raw token string:
        # For scientific notation (e.g. 1.2e-3), derive d from mantissa decimals minus exponent.
        # For standard decimals (e.g. 1.45), d is the count of fractional digits.
        # For integers (e.g. 412), d = 0.
        if re.search(r"[eE]", token_str):
            parts = re.split(r"[eE]", token_str)
            mantissa = parts[0]
            try:
                exp_str = re.split(r"[%]", parts[1])[0]
                exp = int(exp_str)
            except (IndexError, ValueError):
                exp = 0
            mantissa_dec = len(mantissa.split(".")[1]) if "." in mantissa else 0
            d = max(0, mantissa_dec - exp)
        elif "." in token_str:
            dec_part = token_str.split(".")[1]
            d = len(re.split(r"[%]", dec_part)[0])
        else:
            d = 0

        # Half-unit tolerance based on claimed precision: abs(val - kn) <= 0.5 * 10^(-d) + 1e-9.
        # Intended behavior: P=0.01 passes when true P=0.012 (correct 2-decimal rounding),
        # whereas 0.02 and 0.013 are rejected.
        tol = 0.5 * (10.0 ** (-d)) + 1e-9

        matched = False
        for kn in known_numbers:
            # 1. Direct match:
            if abs(tn - kn) <= tol:
                matched = True
                break

            # 2. Percentage scale match: only permitted when token explicitly has '%' attached
            if has_pct:
                if abs(tn - kn * 100.0) <= tol:
                    matched = True
                    break

        # 3. Allowed untraced whitelist: uses the exact same half-unit precision tolerance
        if not matched and allowed_list:
            if any(abs(tn - aw) <= tol for aw in allowed_list):
                matched = True

        if not matched:
            if not is_structural:
                untraced.append(tn)

    # 3. Observational Causal Inference & E-value Check
    results_keys_str = " ".join(doc.results_dict.keys()).lower()
    text_lower = all_text.lower()
    has_causal_key = any(
        k in results_keys_str
        for k in [
            "psm",
            "propensity",
            "love_plot",
        ]
    )
    has_regression_key = any(
        k in results_keys_str
        for k in [
            "cox",
            "logistic",
            "regression",
            "hazard_ratio",
            "or_table",
            "odds_ratio",
        ]
    )
    has_causal_claims = bool(
        re.search(
            r"\b(?:causal|causes?|causation|effect of|impact of|etiolog\w*)\b",
            text_lower,
        )
    )
    is_non_causal_prediction = (
        bool(
            re.search(
                r"\b(?:prediction|predictive|tripod|risk score|discrimination|calibration|c-statistic|c-index|auroc)\b",
                text_lower,
            )
        )
        and not has_causal_claims
    )

    is_causal = (
        has_causal_key
        or (has_regression_key and not is_non_causal_prediction)
        or has_causal_claims
    )

    has_causal_caveat = True
    warnings: list[str] = []
    if is_causal:
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

    # 5. Citation Count
    citation_matches = re.findall(
        r"<sup[^>]*>.*?</sup>|\[\^\d+\]|\^\[\d+(?:[,\-–]\s*\d+)*\]|\[(?:100|\d{1,2})(?:\s*[,\-–]\s*(?:100|\d{1,2}))*\]|\b(?:Ref\.?|Citation)\s*#?\d+",
        all_text,
        re.IGNORECASE,
    )
    citations_count = len(citation_matches)

    passed = (len(phi_violations) == 0) and has_causal_caveat and (len(untraced) == 0)

    remediation_hints: list[str] = []
    if phi_violations:
        remediation_hints.append(
            "PHI detected. Remediation: Strip or mask patient identifiers (HN, Thai ID, phone, patient names) before publishing."
        )
    if untraced:
        remediation_hints.append(
            "Untraced numbers detected. Remediation: Ensure narrative numbers match values in results_dict within half-unit precision tolerance "
            "(|val - known| <= 0.5 * 10^(-d) + 1e-9, where d is the count of displayed fractional decimal digits, d=0 for integers). "
            "Percentage-scale matching (kn * 100) applies only when the narrative token includes an explicit '%' sign. "
            "P-value inequality expressions (e.g. P < 0.05, P ≤ 0.05, P > 0.05, P ≥ 0.05) must satisfy recorded p-values."
        )
    if not has_causal_caveat:
        remediation_hints.append(
            "Observational analysis lacks causal/unmeasured confounding caveat. Remediation: Add VanderWeele E-value or unmeasured confounding caveat."
        )
    if not has_methods:
        remediation_hints.append(
            "Lacks Methods section. Remediation: Include a dedicated Methods section in the ReportDocument."
        )
    if not has_retention_flow:
        remediation_hints.append(
            "Lacks cohort retention flow. Remediation: Include participant retention flow (initial -> excluded -> analyzed) in the report."
        )

    return IntegrityReport(
        passed=passed,
        untraced_numbers=untraced,
        phi_violations=phi_violations,
        warnings=warnings,
        has_methods=has_methods,
        has_retention_flow=has_retention_flow,
        has_causal_caveat=has_causal_caveat,
        citations_count=citations_count,
        agent_remediation=remediation_hints,
    )
