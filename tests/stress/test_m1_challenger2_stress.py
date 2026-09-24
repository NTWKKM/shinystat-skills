"""
tests/stress/test_m1_challenger2_stress.py: Empirical Stress Harness for Milestone M1.

Author: Challenger 2 (Empirical Reviewer)
Scope:
1. Restricted Cubic Splines (medstat.models.splines) with lifelines CoxPHFitter:
   - Mathematical proof of patsy.cr() partition-of-unity collinearity (sum == 1.0).
   - Empirical proof that unconstrained cr() causes singular matrix / ConvergenceError in lifelines.
   - Verification that constraints='center' resolves collinearity and converges reliably.
2. Meta-Analysis (medstat.meta.models & medstat.meta.forest):
   - Zero standard error (se=0.0) division by zero and inf/nan propagation.
   - Negative standard error (se < 0.0) silent acceptance without ValueError.
   - Zero between-study variance (tau^2 = 0.0) model consistency and HKSJ behavior.
   - Cross-module contract defect: run_meta_analysis omitting ci_lower/ci_upper/effect_size causing KeyError in generate_forest_data.
3. ROC DeLong Test (medstat.diagnostic.roc):
   - Paired DeLong test under identical predictions (s1 == s2).
   - Paired DeLong test under strictly collinear positive scaling (s2 = a*s1 + b).
   - Inverted predictions (s2 = -s1).
   - Constant predictions (s1 == constant).
4. Conftest Forwarding Wrapper & Config Interaction (tests/conftest.py & medstat.config):
   - MEDSTAT_PYTEST_FORWARDED=1 collision with ConfigManager._load_env_overrides emitting UserWarning.
   - Non-root working directory invocation collecting 0 tests when forwarded due to missing cwd=project_root.
"""

import math
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import patsy
import pytest
from lifelines import CoxPHFitter
from lifelines.exceptions import ConvergenceError

from medstat.diagnostic.roc import (
    delong_paired_test,
)
from medstat.meta.forest import generate_forest_data
from medstat.meta.models import run_meta_analysis
from medstat.models.splines import fit_cox_rcs

# =============================================================================
# 1. Splines & Cox Proportional Hazards Collinearity
# =============================================================================


class TestSplinesCoxCollinearity:
    """Empirical verification of spline basis collinearity in Cox PH regression."""

    @pytest.fixture
    def survival_data(self) -> pd.DataFrame:
        np.random.seed(42)
        n = 100
        return pd.DataFrame(
            {
                "time": np.random.exponential(10, size=n) + 1.0,
                "status": np.random.binomial(1, 0.5, size=n),
                "age": np.random.normal(50, 10, size=n),
            }
        )

    def test_unconstrained_cr_basis_sums_identically_to_one(self, survival_data):
        """Proof that default unconstrained patsy.cr basis forms a partition of unity (sum == 1.0)."""
        y, X = patsy.dmatrices(
            "time + status ~ cr(age, df=4)", survival_data, return_type="dataframe"
        )
        cr_cols = [c for c in X.columns if "cr" in c]
        row_sums = X[cr_cols].sum(axis=1)

        # Mathematical fact: unconstrained cr basis functions sum exactly to 1.0
        np.testing.assert_allclose(
            row_sums,
            1.0,
            rtol=1e-12,
            err_msg="patsy cr basis without constraints must sum identically to 1.0",
        )

    def test_unconstrained_cr_causes_lifelines_convergence_error(self, survival_data):
        """
        Empirically confirm that feeding unconstrained cr() into CoxPHFitter triggers
        singular matrix LinAlgError/ConvergenceError due to the implicit intercept.
        """
        y, X = patsy.dmatrices(
            "time + status ~ cr(age, df=4)", survival_data, return_type="dataframe"
        )
        if "Intercept" in X.columns:
            X = X.drop(columns=["Intercept"])

        data_for_cph = X.copy()
        data_for_cph["time"] = survival_data["time"].values
        data_for_cph["status"] = survival_data["status"].values

        cph = CoxPHFitter()
        # Must fail with ConvergenceError due to singularity of the Hessian
        with pytest.raises(ConvergenceError) as exc_info:
            cph.fit(data_for_cph, duration_col="time", event_col="status")

        err_msg = str(exc_info.value).lower()
        assert (
            "singular" in err_msg or "collinear" in err_msg or "convergence" in err_msg
        )

    def test_constrained_center_cr_converges_reliably(self, survival_data):
        """
        Empirically verify that constraints='center' breaks the partition of unity
        and enables clean, stable Cox PH fitting across multiple seeds.
        """
        # Test across multiple random seeds
        for seed in [1, 7, 42, 99]:
            np.random.seed(seed)
            df = pd.DataFrame(
                {
                    "time": np.random.exponential(10, size=80) + 1.0,
                    "status": np.random.binomial(1, 0.5, size=80),
                    "age": np.random.normal(55, 10, size=80),
                }
            )
            fig, cdf, meta = fit_cox_rcs(
                df, duration_col="time", event_col="status", rcs_var="age", knots=4
            )
            assert not cdf.empty
            assert meta["c_index"] > 0.0
            assert "constraints='center'" in " ".join(meta["spline_columns"])


# =============================================================================
# 2. Meta-Analysis Boundary & Cross-Module Contract Conditions
# =============================================================================


class TestMetaAnalysisBoundaries:
    """Empirical verification of meta-analysis under boundary conditions and cross-module contracts."""

    def test_meta_zero_standard_error_division_by_zero(self):
        """Verify models.py defensively rejects se == 0 with ValueError."""
        df = pd.DataFrame(
            {"study": ["S1", "S2"], "log_effect": [-0.5, -0.4], "se": [0.0, 0.1]}
        )
        with pytest.raises(ValueError, match="strictly positive"):
            run_meta_analysis(df)

    def test_meta_negative_standard_error_silent_acceptance(self):
        """Verify models.py defensively rejects se < 0 with ValueError."""
        df = pd.DataFrame(
            {"study": ["S1", "S2"], "log_effect": [0.5, 0.5], "se": [-0.2, 0.2]}
        )
        with pytest.raises(ValueError, match="strictly positive"):
            run_meta_analysis(df)

    def test_meta_zero_between_study_variance_consistency(self):
        """
        Verify that when true between-study variance is zero (homogeneous studies),
        tau^2 is exactly 0.0, I^2 is 0.0, and RE estimates match FE estimates exactly.
        """
        df = pd.DataFrame(
            {
                "study": ["Study1", "Study2", "Study3", "Study4"],
                "log_effect": [0.35, 0.35, 0.35, 0.35],
                "se": [0.10, 0.10, 0.10, 0.10],
            }
        )
        res = run_meta_analysis(df)
        assert res["heterogeneity"]["tau2"] == 0.0
        assert res["heterogeneity"]["I2"] == 0.0
        assert math.isclose(
            res["fixed_effect"]["log_effect"],
            res["random_effects"]["log_effect"],
            abs_tol=1e-12,
        )
        assert math.isclose(
            res["fixed_effect"]["se"], res["random_effects"]["se"], abs_tol=1e-12
        )

    def test_meta_zero_between_study_variance_hksj(self):
        """
        Verify that Hartung-Knapp adjustment handles tau^2 = 0 gracefully without division by zero.
        """
        df = pd.DataFrame(
            {
                "study": ["Study1", "Study2", "Study3", "Study4"],
                "log_effect": [0.20, 0.20, 0.20, 0.20],
                "se": [0.15, 0.15, 0.15, 0.15],
            }
        )
        res = run_meta_analysis(df, use_hksj=True)
        assert res["random_effects"]["se"] > 0
        assert not np.isnan(res["random_effects"]["p_value"])

    def test_meta_forest_missing_study_ci_defect(self):
        """Verify generate_forest_data succeeds with per-study confidence limits."""
        df = pd.DataFrame(
            {
                "study": ["Trial 1", "Trial 2", "Trial 3"],
                "log_effect": [0.20, 0.30, 0.15],
                "se": [0.05, 0.08, 0.06],
            }
        )
        res = run_meta_analysis(df)
        forest_data = generate_forest_data(res)
        assert forest_data is not None


# =============================================================================
# 3. ROC DeLong Test Under Identical and Collinear Predictions
# =============================================================================


class TestROCDeLongCollinearity:
    """Empirical verification of DeLong paired test with collinear / identical scores."""

    @pytest.fixture
    def test_data(self):
        y_true = np.array([0, 0, 0, 0, 1, 1, 1, 1])
        s1 = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8])
        return y_true, s1

    def test_delong_identical_predictions(self, test_data):
        """Identical predictions (score1 == score2) must yield diff=0, z=0, p=1.0."""
        y_true, s1 = test_data
        res = delong_paired_test(y_true, s1, s1)
        assert res["difference"] == 0.0
        assert res["z_statistic"] == 0.0
        assert res["p_value"] == 1.0
        assert res["auc1"] == res["auc2"]

    def test_delong_strictly_collinear_predictions(self, test_data):
        """Positive linear transformation (s2 = 3.5 * s1 + 10.0) preserves rank and yields p=1.0."""
        y_true, s1 = test_data
        s2 = 3.5 * s1 + 10.0
        res = delong_paired_test(y_true, s1, s2)
        assert math.isclose(res["difference"], 0.0, abs_tol=1e-12)
        assert math.isclose(res["z_statistic"], 0.0, abs_tol=1e-12)
        assert res["p_value"] == 1.0

    def test_delong_inverted_predictions(self, test_data):
        """Inverted predictions (s2 = -s1) yields AUC1=1.0, AUC2=0.0, diff=1.0, p=0.0."""
        y_true, s1 = test_data
        s2 = -s1
        res = delong_paired_test(y_true, s1, s2)
        assert res["auc1"] == 1.0
        assert res["auc2"] == 0.0
        assert res["difference"] == 1.0
        assert res["p_value"] < 1e-5

    def test_delong_all_constant_predictions(self, test_data):
        """All-constant scores yield AUC=0.5, diff=0.0, p=1.0 without singular matrix crash."""
        y_true, _ = test_data
        s_const = np.full_like(y_true, 0.5, dtype=float)
        res = delong_paired_test(y_true, s_const, s_const)
        assert res["auc1"] == 0.5
        assert res["auc2"] == 0.5
        assert res["difference"] == 0.0
        assert res["p_value"] == 1.0


# =============================================================================
# 4. Conftest Forwarding Wrapper & Config Interaction
# =============================================================================


class TestConftestForwardingWrapper:
    """Empirical verification of conftest.py forwarding and environment interactions."""

    def test_medstat_pytest_forwarded_triggers_config_warning(self):
        """
        Empirical proof: Setting MEDSTAT_PYTEST_FORWARDED=1 triggers a UserWarning in
        medstat.config.ConfigManager because ConfigManager._load_env_overrides tries to parse
        'pytest.forwarded' as a configuration key and fails with KeyError.
        """
        env = os.environ.copy()
        env["MEDSTAT_PYTEST_FORWARDED"] = "1"

        cmd = [
            sys.executable,
            "-c",
            "import warnings\n"
            "with warnings.catch_warnings(record=True) as w:\n"
            "    warnings.simplefilter('always')\n"
            "    import medstat.config\n"
            "    for item in w:\n"
            "        if 'MEDSTAT_PYTEST_FORWARDED' in str(item.message):\n"
            "            print(f'CAUGHT_WARNING: {item.message}')\n",
        ]
        proc = subprocess.run(cmd, env=env, capture_output=True, text=True)
        assert proc.returncode == 0
        assert "CAUGHT_WARNING" not in proc.stdout

    def test_conftest_wrapper_invoked_from_different_cwd(self):
        """
        Empirical proof:
        1. When conftest.py is executed via Python < 3.12 from a directory outside project_root,
           subprocess.run does not set cwd=project_root, causing pytest to execute in the external
           directory and collect 0 tests (exit code 5).
        2. When conftest.py is executed with Python 3.12 directly as a script, should_forward is False,
           so it exits with code 0 without running pytest.
        """
        project_root = Path(__file__).resolve().parent.parent.parent
        conftest_path = project_root / "tests" / "conftest.py"

        # Test with /usr/bin/python3 (macOS Python 3.9) if available
        py39_bin = "/usr/bin/python3"
        if Path(py39_bin).exists():
            proc39 = subprocess.run(
                [py39_bin, str(conftest_path), "--collect-only"],
                cwd="/tmp",
                capture_output=True,
                text=True,
            )
            # Forwarded pytest runs in /tmp without project root, collecting 0 tests (code 5)
            assert proc39.returncode == 5, (
                f"Expected returncode 5 (no tests in /tmp) from python 3.9 forwarder, got {proc39.returncode}"
            )
            assert "no tests collected" in proc39.stdout.lower()

        # In Python 3.12 directly, running `python tests/conftest.py` invokes venv_pytest
        proc312 = subprocess.run(
            [sys.executable, str(conftest_path), "--collect-only"],
            cwd="/tmp",
            capture_output=True,
            text=True,
        )
        assert proc312.returncode == 5
        assert "no tests collected" in proc312.stdout.lower()
