"""
Baseline Patient Characteristics Table 1 Generator.

Constructs comprehensive, stratified baseline characteristics tables
with automatic parametric/non-parametric hypothesis testing and
Standardized Mean Differences (SMDs).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import scipy.stats as stats

from medstat.causal.balance import calculate_smd


def generate_table_one(
    df: pd.DataFrame,
    strata: str | None = None,
    continuous_vars: list[str] | None = None,
    categorical_vars: list[str] | None = None,
    nonnormal_vars: list[str] | None = None,
    include_smd: bool = True,
    include_p: bool = True,
    digits: int = 1,
) -> pd.DataFrame:
    """
    Generate a publication-grade Table 1 summary.

    Parameters:
        df: Input clinical DataFrame.
        strata: Optional categorical column to stratify by (e.g. 'treatment').
        continuous_vars: List of continuous variable names.
        categorical_vars: List of categorical variable names.
        nonnormal_vars: Subset of continuous variables to format as Median [IQR].
        include_smd: Whether to calculate SMDs (if strata has exactly 2 levels).
        include_p: Whether to calculate p-values across strata.
        digits: Number of decimal places to format.

    Returns:
        pd.DataFrame formatted as Table 1.
    """
    data = df.copy()
    nonnormal = set(nonnormal_vars or [])

    # Auto-detect if lists are not provided
    if continuous_vars is None and categorical_vars is None:
        continuous_vars = []
        categorical_vars = []
        for col in data.columns:
            if col == strata:
                continue
            if pd.api.types.is_numeric_dtype(data[col]) and data[col].nunique() > 10:
                continuous_vars.append(col)
            else:
                categorical_vars.append(col)
    else:
        continuous_vars = continuous_vars or []
        categorical_vars = categorical_vars or []

    # Identify strata levels
    if strata is not None and strata in data.columns:
        strata_levels = [s for s in data[strata].dropna().unique()]
        strata_levels.sort(key=lambda x: str(x))
    else:
        strata_levels = []

    rows = []

    # 1. Overall Sample Size row
    n_total = len(data)
    header_row = {"Characteristic": "Total Patients (N)", "Overall": f"{n_total}"}
    for s in strata_levels:
        n_s = len(data[data[strata] == s])
        pct_s = (n_s / n_total) * 100.0 if n_total > 0 else 0.0
        header_row[f"{s}"] = f"{n_s} ({pct_s:.1f}%)"
    if include_p and strata_levels:
        header_row["P-value"] = ""
    if include_smd and len(strata_levels) == 2:
        header_row["SMD"] = ""
    rows.append(header_row)

    # 2. Continuous Variables
    for var in continuous_vars:
        if var not in data.columns:
            continue

        s_all = data[var].dropna()
        is_nn = var in nonnormal

        if is_nn:
            med = s_all.median()
            q25 = s_all.quantile(0.25)
            q75 = s_all.quantile(0.75)
            overall_str = f"{med:.{digits}f} [{q25:.{digits}f}, {q75:.{digits}f}]"
            label = f"{var}, median [IQR]"
        else:
            mean = s_all.mean()
            sd = s_all.std(ddof=1)
            overall_str = f"{mean:.{digits}f} ± {sd:.{digits}f}"
            label = f"{var}, mean ± SD"

        row = {"Characteristic": label, "Overall": overall_str}

        group_series = []
        for s in strata_levels:
            s_grp = data[data[strata] == s][var].dropna()
            group_series.append(s_grp)
            if is_nn:
                m = s_grp.median()
                q1 = s_grp.quantile(0.25)
                q3 = s_grp.quantile(0.75)
                row[f"{s}"] = f"{m:.{digits}f} [{q1:.{digits}f}, {q3:.{digits}f}]"
            else:
                m = s_grp.mean()
                sd = s_grp.std(ddof=1)
                row[f"{s}"] = f"{m:.{digits}f} ± {sd:.{digits}f}"

        # Hypothesis test
        p_val_str = ""
        if include_p and len(group_series) >= 2:
            try:
                valid_groups = [g.values for g in group_series if len(g) > 0]
                if len(valid_groups) == 2:
                    if is_nn:
                        _, p = stats.mannwhitneyu(valid_groups[0], valid_groups[1])
                    else:
                        _, p = stats.ttest_ind(
                            valid_groups[0], valid_groups[1], equal_var=False
                        )
                elif len(valid_groups) > 2:
                    if is_nn:
                        _, p = stats.kruskal(*valid_groups)
                    else:
                        _, p = stats.f_oneway(*valid_groups)
                else:
                    p = np.nan
                p_val_str = (
                    "<0.001" if p < 0.001 else f"{p:.3f}" if not np.isnan(p) else ""
                )
            except Exception:
                p_val_str = "—"
            row["P-value"] = p_val_str

        # SMD
        if include_smd and len(group_series) == 2:
            try:
                smd = calculate_smd(group_series[0], group_series[1])
                row["SMD"] = f"{abs(smd):.3f}" if not np.isnan(smd) else "—"
            except Exception:
                row["SMD"] = "—"

        rows.append(row)

    # 3. Categorical Variables
    for var in categorical_vars:
        if var not in data.columns:
            continue

        clean_var = data[var].dropna()
        cats = [c for c in clean_var.unique()]
        cats.sort(key=lambda x: str(x))

        header_cat = {"Characteristic": f"{var}, n (%)", "Overall": ""}
        for s in strata_levels:
            header_cat[f"{s}"] = ""

        # Precompute Chi-square across all levels
        p_cat_str = ""
        smd_cat_str = ""
        if strata_levels:
            try:
                ct = pd.crosstab(data[var], data[strata])
                if ct.size > 0:
                    chi2, p, _, _ = stats.chi2_contingency(ct)
                    p_cat_str = "<0.001" if p < 0.001 else f"{p:.3f}"
            except Exception:
                p_cat_str = "—"

        if include_p and strata_levels:
            header_cat["P-value"] = p_cat_str
        if include_smd and len(strata_levels) == 2:
            header_cat["SMD"] = smd_cat_str
        rows.append(header_cat)

        # Iterate over individual categories
        for cat in cats:
            n_cat = int(np.sum(data[var] == cat))
            pct_cat = (n_cat / n_total * 100.0) if n_total > 0 else 0.0
            cat_row = {
                "Characteristic": f"   {cat}",
                "Overall": f"{n_cat} ({pct_cat:.1f}%)",
            }
            grp_props = []
            for s in strata_levels:
                sub_grp = data[data[strata] == s]
                n_grp = len(sub_grp)
                n_c = int(np.sum(sub_grp[var] == cat))
                pct_c = (n_c / n_grp * 100.0) if n_grp > 0 else 0.0
                cat_row[f"{s}"] = f"{n_c} ({pct_c:.1f}%)"
                grp_props.append((sub_grp[var] == cat).astype(int))

            if include_p and strata_levels:
                cat_row["P-value"] = ""

            if include_smd and len(grp_props) == 2:
                try:
                    smd = calculate_smd(grp_props[0], grp_props[1])
                    cat_row["SMD"] = f"{abs(smd):.3f}" if not np.isnan(smd) else ""
                except Exception:
                    cat_row["SMD"] = ""

            rows.append(cat_row)

    return pd.DataFrame(rows)
