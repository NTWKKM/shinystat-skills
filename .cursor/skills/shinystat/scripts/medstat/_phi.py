"""
src/medstat/_phi.py: Single Source of Truth for Protected Health Information (PHI)
patterns, zero-PHI scanning, and validation guards (HIPAA 18 Identifiers + Thai PDPA).
"""

from __future__ import annotations

import re
from typing import Sequence

# PHI detection patterns (HIPAA 18 Identifiers + Thai PDPA)
PHI_PATTERNS: list[tuple[re.Pattern, str]] = [
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


def scan_phi_violations(text: str) -> list[str]:
    """Scans text against all PHI patterns and returns list of violation messages."""
    violations: list[str] = []
    if not text:
        return violations
    for pattern, label in PHI_PATTERNS:
        matches = pattern.findall(text)
        for m in matches:
            violations.append(f"{label}: '{m}'")
    return violations


def assert_zero_phi_text(texts: Sequence[str | None]) -> None:
    """
    Enforces Pillar 5 Zero-PHI safety invariant on text sequences (titles, labels, captions).
    Raises ValueError on any PHI detection.
    """
    for t in texts:
        if not t:
            continue
        t_str = str(t)
        for pattern, label in PHI_PATTERNS:
            match = pattern.search(t_str)
            if match:
                raise ValueError(
                    f"Zero-PHI Violation detected ({label}) in text: '{match.group()}'"
                )
