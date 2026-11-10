"""
Shared input sanitation for binary-outcome diagnostic metrics (ADR 73).

Missing outcomes must be excluded, never coded as non-events: comparing NaN with a
positive label (or casting NaN to int) silently turns a missing outcome into 0.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from medstat.logging import get_logger

logger = get_logger(__name__)


def binary_outcome(y_true: Any, pos_label: Any = 1) -> tuple[np.ndarray, np.ndarray]:
    """Return (y01, observed): y01 is 1 where y == pos_label else 0; observed is False for missing y."""
    y = pd.Series(np.asarray(y_true, dtype=object).ravel())
    observed = (~pd.isna(y)).to_numpy()
    y01 = (y == pos_label).to_numpy().astype(int)
    return y01, observed


def numeric_vector(x: Any) -> np.ndarray:
    """Float vector with missing values as NaN."""
    return pd.to_numeric(
        pd.Series(np.asarray(x, dtype=object).ravel()), errors="coerce"
    ).to_numpy(dtype=float)


def complete_cases(
    func_name: str, y_true: Any, *scores: Any, pos_label: Any = 1
) -> tuple[np.ndarray, ...]:
    """Drop rows with a missing outcome or score; log the excluded count (Zero Silent Deletion)."""
    y01, observed = binary_outcome(y_true, pos_label=pos_label)
    arrays = [numeric_vector(s) for s in scores]
    for a in arrays:
        if len(a) != len(y01):
            raise ValueError(
                f"{func_name}: outcome and score vectors must have equal length."
            )
    keep = observed.copy()
    for a in arrays:
        keep &= np.isfinite(a)
    n_drop = int(len(keep) - keep.sum())
    if n_drop:
        logger.warning(
            f"{func_name}: excluded {n_drop} of {len(keep)} rows with missing outcome or score."
        )
    return (y01[keep], *(a[keep] for a in arrays))
