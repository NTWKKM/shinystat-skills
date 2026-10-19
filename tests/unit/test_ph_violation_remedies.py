"""
tests/unit/test_ph_violation_remedies.py

Unit tests verifying proportional hazards (PH) violation handling and remedy functions:
1. refit_cox_with_strata matches lifelines strata= HRs for non-strata covariates,
   omits strata variables, and raises ValueError if exposure is in strata.
2. Simulated data (n=3000, fixed seed; true HR 0.5 before t=2, 1.0 after):
   piecewise_hr recovers each period's HR within 15%.
3. Plain Cox exposure term on same data is flagged (P <= 0.05) and appears in exactly
   one of strong_violations or weak_violations.
4. Multiplicity Bonferroni rule: every covariate with P <= 0.05 appears in exactly
   one of the two lists, and in neither when P > 0.05.
5. Recipe 11 code block equals src functions (normalized source text and AST).
"""

import ast
import re
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from lifelines import CoxPHFitter

from medstat.models.survival import (
    fit_survival_analysis_suite,
    piecewise_hr,
    refit_cox_with_strata,
    schoenfeld_diagnostics,
)


def _generate_piecewise_dataset(n: int = 3000, seed: int = 42) -> pd.DataFrame:
    """Generate survival data with true HR=0.5 before t=2 and HR=1.0 after t=2."""
    np.random.seed(seed)
    x = np.random.binomial(1, 0.5, size=n)
    u = np.random.uniform(0, 1, size=n)
    neg_log_u = -np.log(u)

    # Base hazard rate h0 = 0.1
    # x=0: H(t) = 0.1 * t
    # x=1: H(t) = 0.05 * t for t <= 2 (HR=0.5); H(t) = 0.1 + 0.1 * (t - 2) for t > 2 (HR=1.0)
    t = np.zeros(n)
    for i in range(n):
        if x[i] == 0:
            t[i] = neg_log_u[i] / 0.1
        else:
            if neg_log_u[i] <= 0.1:
                t[i] = neg_log_u[i] / 0.05
            else:
                t[i] = 2.0 + (neg_log_u[i] - 0.1) / 0.1

    censor_time = np.random.uniform(2.5, 6.0, size=n)
    duration = np.minimum(t, censor_time)
    event = (t <= censor_time).astype(int)

    return pd.DataFrame({"duration": duration, "event": event, "exposure": x})


def test_refit_cox_with_strata_invariants():
    """Verify refit_cox_with_strata HR equivalence, strata omission, and exposure validation."""
    rng = np.random.default_rng(123)
    n = 200
    df = pd.DataFrame(
        {
            "time": rng.exponential(10, size=n) + 1,
            "event": rng.binomial(1, 0.5, size=n),
            "exposure": rng.binomial(1, 0.5, size=n),
            "age": rng.normal(50, 10, size=n),
            "site": rng.choice(["Hospital A", "Hospital B", "Hospital C"], size=n),
        }
    )

    # 1. ValueError when exposure is in strata_cols
    with pytest.raises(ValueError, match="cannot be in strata_cols"):
        refit_cox_with_strata(
            df=df,
            duration_col="time",
            event_col="event",
            covariates=["exposure", "age"],
            strata_cols=["exposure"],
            exposure="exposure",
        )

    # 2. Fit with strata on site
    res_df = refit_cox_with_strata(
        df=df,
        duration_col="time",
        event_col="event",
        covariates=["exposure", "age"],
        strata_cols=["site"],
        exposure="exposure",
    )

    # 3. Direct lifelines reference
    cph_ref = CoxPHFitter()
    cph_ref.fit(
        df[["time", "event", "exposure", "age", "site"]],
        duration_col="time",
        event_col="event",
        strata=["site"],
    )

    # 4. HRs equal lifelines strata= HRs for non-strata covariates (rtol 1e-8)
    for cov in ["exposure", "age"]:
        assert cov in res_df.index
        assert np.isclose(
            res_df.loc[cov, "hazard_ratio"],
            cph_ref.summary.loc[cov, "exp(coef)"],
            rtol=1e-8,
        )
        assert np.isclose(
            res_df.loc[cov, "exp(coef)"],
            cph_ref.summary.loc[cov, "exp(coef)"],
            rtol=1e-8,
        )

    # 5. Strata covariates are absent from the table
    assert "site" not in res_df.index


def test_piecewise_hr_recovers_true_hrs_within_15_percent():
    """Simulated data (n=3000, seed 42): piecewise_hr recovers each period's HR within 15%."""
    df = _generate_piecewise_dataset(n=3000, seed=42)

    pw_res = piecewise_hr(
        df=df,
        duration_col="duration",
        event_col="event",
        exposure="exposure",
        cut_points=2.0,
    )

    assert len(pw_res) == 2

    # Period 1 (t < 2.0): true HR = 0.5
    hr_period_1 = float(pw_res.iloc[0]["hazard_ratio"])
    rel_error_1 = abs(hr_period_1 - 0.5) / 0.5
    assert rel_error_1 < 0.15, (
        f"Period 1 HR {hr_period_1} exceeds 15% error (rel error {rel_error_1:.1%})"
    )

    # Period 2 (t >= 2.0): true HR = 1.0
    hr_period_2 = float(pw_res.iloc[1]["hazard_ratio"])
    rel_error_2 = abs(hr_period_2 - 1.0) / 1.0
    assert rel_error_2 < 0.15, (
        f"Period 2 HR {hr_period_2} exceeds 15% error (rel error {rel_error_2:.1%})"
    )


def test_plain_cox_violates_ph_and_flagged_in_exactly_one_list():
    """On simulated data, plain Cox exposure term is flagged (P <= 0.05) and appears in exactly one of strong/weak violations."""
    df = _generate_piecewise_dataset(n=3000, seed=42)

    cph = CoxPHFitter()
    cph.fit(df, duration_col="duration", event_col="event")

    diag = schoenfeld_diagnostics(cph, df)

    p_exposure = diag["p_values"]["exposure"]
    assert p_exposure <= 0.05, (
        f"Expected exposure PH violation (p <= 0.05), got {p_exposure}"
    )

    in_strong = "exposure" in diag["strong_violations"]
    in_weak = "exposure" in diag["weak_violations"]

    assert in_strong ^ in_weak, (
        f"Exposure must appear in exactly one list (strong={in_strong}, weak={in_weak})"
    )


def test_schoenfeld_multiplicity_partition_invariant():
    """Every covariate with P <= 0.05 appears in exactly one list, and in neither when P > 0.05."""
    df = _generate_piecewise_dataset(n=3000, seed=42)
    rng = np.random.default_rng(42)

    # Add extra covariates with known varying properties
    df["z_ph_pass"] = rng.normal(0, 1, size=len(df))
    df["z_ph_binary"] = rng.binomial(1, 0.4, size=len(df))

    cph = CoxPHFitter()
    cph.fit(df, duration_col="duration", event_col="event")

    diag = schoenfeld_diagnostics(cph, df)
    k = len(diag["p_values"])
    bonferroni_alpha = 0.05 / k

    assert diag["bonferroni_alpha"] == bonferroni_alpha

    for var, p in diag["p_values"].items():
        in_strong = var in diag["strong_violations"]
        in_weak = var in diag["weak_violations"]

        if p <= 0.05:
            # Must appear in exactly one list
            assert in_strong ^ in_weak, (
                f"{var} (p={p}) must appear in exactly one violation list"
            )
            if p <= bonferroni_alpha:
                assert in_strong and not in_weak, (
                    f"{var} (p={p} <= {bonferroni_alpha}) should be strong"
                )
            else:
                assert in_weak and not in_strong, (
                    f"{var} ({bonferroni_alpha} < p={p} <= 0.05) should be weak"
                )
        else:
            # Must appear in neither list
            assert not in_strong and not in_weak, (
                f"{var} (p={p} > 0.05) must not appear in any violation list"
            )


def test_recipe_11_equals_src_functions():
    """Recipe 11 code block equals src functions (normalized source text and AST)."""
    repo_root = Path(__file__).resolve().parent.parent.parent
    recipe_path = (
        repo_root / "skills" / "shinystat-cloud" / "references" / "python-recipes.md"
    )
    src_path = repo_root / "src" / "medstat" / "models" / "survival.py"

    content = recipe_path.read_text(encoding="utf-8")
    m = re.search(
        r"## 11\. Survival Analysis.*?```python\n(.*?)\n```", content, re.DOTALL
    )
    assert m, "Recipe 11 code block not found in python-recipes.md"
    recipe_code = m.group(1)

    recipe_tree = ast.parse(recipe_code)
    recipe_funcs = {
        node.name: node
        for node in ast.walk(recipe_tree)
        if isinstance(node, ast.FunctionDef)
    }

    src_code = src_path.read_text(encoding="utf-8")
    src_tree = ast.parse(src_code)
    src_funcs = {
        node.name: node
        for node in ast.walk(src_tree)
        if isinstance(node, ast.FunctionDef)
    }

    target_funcs = [
        "schoenfeld_diagnostics",
        "refit_cox_with_strata",
        "piecewise_hr",
        "fit_survival_analysis_suite",
    ]

    for func_name in target_funcs:
        assert func_name in recipe_funcs, f"Function {func_name} missing from Recipe 11"
        assert func_name in src_funcs, (
            f"Function {func_name} missing from src/medstat/models/survival.py"
        )

        # AST equivalence
        recipe_ast = ast.dump(recipe_funcs[func_name])
        src_ast = ast.dump(src_funcs[func_name])
        assert recipe_ast == src_ast, f"AST mismatch for {func_name}"

        # Source text equivalence
        recipe_text = ast.get_source_segment(
            recipe_code, recipe_funcs[func_name]
        ).strip()
        src_text = ast.get_source_segment(src_code, src_funcs[func_name]).strip()
        assert recipe_text == src_text, (
            f"Normalized source text mismatch for {func_name}"
        )


def test_fit_survival_analysis_suite_remedy_integration():
    """Verify fit_survival_analysis_suite integrates schoenfeld_diagnostics with Bonferroni fields."""
    df = _generate_piecewise_dataset(n=300, seed=42)
    res = fit_survival_analysis_suite(
        df=df,
        duration_col="duration",
        event_col="event",
        covariates=["exposure"],
    )
    schoen = res["schoenfeld_diagnostics"]
    assert "bonferroni_alpha" in schoen
    assert "strong_violations" in schoen
    assert "weak_violations" in schoen
    assert "indeterminate_tests" in schoen
    assert "p_values" in schoen
    assert "exposure" in schoen["p_values"]


def test_schoenfeld_indeterminate_p_values_prevent_pass(monkeypatch):
    """Verify non-finite Schoenfeld p-values populate indeterminate_tests, prevent pass, and update note."""
    df = _generate_piecewise_dataset(n=100, seed=42)
    cph = CoxPHFitter()
    cph.fit(df, duration_col="duration", event_col="event")

    # Mock proportional_hazard_test to return a NaN p-value for exposure
    class MockPropSummary:
        summary = pd.DataFrame(
            {
                "test_statistic": [1.23, 0.45],
                "p": [np.nan, 0.85],
            },
            index=["exposure", "covar2"],
        )

    def mock_prop_test(*args, **kwargs):
        return MockPropSummary()

    monkeypatch.setattr(
        "medstat.models.survival.proportional_hazard_test", mock_prop_test
    )

    diag = schoenfeld_diagnostics(cph, df)

    assert "exposure" in diag["indeterminate_tests"]
    assert (
        diag["passed"] is False
        if "passed" in diag
        else diag["no_ph_violation_detected"] is False
    )
    assert diag["PH_Assumptions_Passed"] is False
    assert "Indeterminate PH test result" in diag["PH_Note"]
    assert "exposure" in diag["PH_Note"]


def test_schoenfeld_indeterminate_and_violations_both_reported(monkeypatch):
    """Verify PH_Note includes both significant violations and indeterminate tests when co-occurring."""
    df = _generate_piecewise_dataset(n=100, seed=42)
    cph = CoxPHFitter()
    cph.fit(df, duration_col="duration", event_col="event")

    class MockCoOccurringPropSummary:
        summary = pd.DataFrame(
            {
                "test_statistic": [10.5, 1.23, 0.45],
                "p": [0.001, np.nan, 0.85],
            },
            index=["strong_var", "indet_var", "normal_var"],
        )

    def mock_prop_test(*args, **kwargs):
        return MockCoOccurringPropSummary()

    monkeypatch.setattr(
        "medstat.models.survival.proportional_hazard_test", mock_prop_test
    )

    diag = schoenfeld_diagnostics(cph, df)

    assert "strong_var" in diag["strong_violations"]
    assert "indet_var" in diag["indeterminate_tests"]
    assert diag["PH_Assumptions_Passed"] is False
    assert "PH violation detected: 1 strong" in diag["PH_Note"]
    assert (
        "Indeterminate PH test result: non-finite p-value for ['indet_var']."
        in diag["PH_Note"]
    )


def test_refit_cox_with_strata_retention_records():
    """Verify refit_cox_with_strata records N_initial, N_excluded, and itemized reasons in attrs."""
    rng = np.random.default_rng(99)
    n = 150
    df = pd.DataFrame(
        {
            "time": rng.exponential(10, size=n) + 1,
            "event": rng.binomial(1, 0.5, size=n),
            "exposure": rng.binomial(1, 0.5, size=n),
            "site": rng.choice(["Hospital A", "Hospital B"], size=n),
            "biomarker": rng.normal(10, 2, size=n),
        }
    )
    # Inject missing values
    df.loc[0:4, "biomarker"] = np.nan  # 5 missing
    df.loc[3:7, "site"] = np.nan  # 5 missing (indices 3, 4 overlap)

    res_df = refit_cox_with_strata(
        df=df,
        duration_col="time",
        event_col="event",
        covariates=["exposure", "biomarker"],
        strata_cols=["site"],
        exposure="exposure",
    )

    retention = res_df.attrs["retention"]
    assert retention["n_initial"] == 150
    assert retention["n_analyzed"] == 142
    assert retention["n_excluded"] == 8
    assert retention["exclusion_reasons"] == {
        "Missing biomarker": 5,
        "Missing site": 5,
    }


def test_piecewise_hr_cut_points_validation_and_period_names():
    """Verify cut_points validation (positive, finite, unique, < max duration) and (lo, hi] period names."""
    df = _generate_piecewise_dataset(n=500, seed=42)
    max_t = float(df["duration"].max())

    # Invalid: empty cut_points
    with pytest.raises(ValueError, match="at least one cut time"):
        piecewise_hr(df, "duration", "event", "exposure", cut_points=[])

    # Invalid: non-finite cut_points
    with pytest.raises(ValueError, match="finite numbers"):
        piecewise_hr(df, "duration", "event", "exposure", cut_points=[np.nan])

    with pytest.raises(ValueError, match="finite numbers"):
        piecewise_hr(df, "duration", "event", "exposure", cut_points=[np.inf])

    # Invalid: non-positive cut_points
    with pytest.raises(ValueError, match="strictly positive"):
        piecewise_hr(df, "duration", "event", "exposure", cut_points=0.0)

    with pytest.raises(ValueError, match="strictly positive"):
        piecewise_hr(df, "duration", "event", "exposure", cut_points=[-1.5])

    # Invalid: cut_point >= max duration
    with pytest.raises(
        ValueError, match="strictly less than maximum follow-up duration"
    ):
        piecewise_hr(df, "duration", "event", "exposure", cut_points=max_t + 1.0)

    with pytest.raises(
        ValueError, match="strictly less than maximum follow-up duration"
    ):
        piecewise_hr(df, "duration", "event", "exposure", cut_points=max_t)

    # Valid multi-cut: verify period names use (lo, hi] notation
    pw_res = piecewise_hr(df, "duration", "event", "exposure", cut_points=[1.5, 3.0])
    assert list(pw_res["period"]) == ["t <= 1.5", "1.5 < t <= 3.0", "t > 3.0"]


def test_piecewise_hr_zero_duration_filtering():
    """Verify zero and negative durations are excluded and recorded in attrs."""
    df = _generate_piecewise_dataset(n=300, seed=42)
    # Inject 4 zero or negative durations
    df.loc[0, "duration"] = 0.0
    df.loc[1, "duration"] = 0.0
    df.loc[2, "duration"] = -0.5
    df.loc[3, "duration"] = -2.0

    pw_res = piecewise_hr(df, "duration", "event", "exposure", cut_points=2.0)
    assert pw_res.attrs["n_zero_duration_excluded"] == 4
    assert len(pw_res) == 2
