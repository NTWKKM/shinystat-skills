"""
tests/fixtures/synthetic_clinical.py: Deterministic, Zero-PHI Synthetic Clinical Datasets.

Provides reproducible cohorts for testing all 13 biostatistical recipes,
figure generation, and multi-format document reporting.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def generate_synthetic_clinical_cohort(n: int = 500, seed: int = 42) -> pd.DataFrame:
    """
    Generates a deterministic synthetic clinical cohort with Zero-PHI.
    Includes binary endpoints, time-to-event survival, diagnostic biomarkers,
    paired blood pressure measurements, and baseline confounders.
    """
    rng = np.random.default_rng(seed)

    patient_ids = [f"PATIENT_MOCK_{i:04d}" for i in range(1, n + 1)]
    age = rng.normal(loc=62.0, scale=12.0, size=n).clip(25.0, 95.0)
    sex = rng.choice([0, 1], size=n, p=[0.48, 0.52])  # 0: Female, 1: Male
    bmi = rng.normal(loc=26.5, scale=4.5, size=n).clip(16.0, 48.0)
    egfr = rng.normal(loc=72.0, scale=20.0, size=n).clip(10.0, 130.0)
    hypertension = rng.choice([0, 1], size=n, p=[0.55, 0.45])
    diabetes = rng.choice([0, 1], size=n, p=[0.70, 0.30])

    # Propensity for treatment
    logit_ps = (
        -1.2
        + 0.02 * (age - 60)
        + 0.4 * sex
        + 0.5 * hypertension
        + 0.6 * diabetes
        - 0.015 * (egfr - 70)
    )
    ps = 1.0 / (1.0 + np.exp(-logit_ps))
    treatment_arm = rng.binomial(1, ps)

    # Diagnostic biomarker (Troponin for AMI diagnosis)
    # AMI status prevalence ~25%
    ami_prob = 1.0 / (1.0 + np.exp(-(-2.0 + 0.03 * age + 0.8 * diabetes)))
    gold_standard_ami = rng.binomial(1, ami_prob)
    # Biomarker values conditioned on AMI status
    troponin_level = np.where(
        gold_standard_ami == 1,
        rng.lognormal(mean=2.8, sigma=0.6, size=n),
        rng.lognormal(mean=1.1, sigma=0.5, size=n),
    ).clip(0.1, 150.0)

    # Paired measurements (Device A vs Device B SBP)
    true_sbp = rng.normal(loc=132.0, scale=15.0, size=n)
    sbp_device_a = true_sbp + rng.normal(loc=0.0, scale=4.0, size=n)
    sbp_device_b = (
        true_sbp + 2.5 + rng.normal(loc=0.0, scale=5.0, size=n)
    )  # Mean bias ~2.5 mmHg

    # Inter-rater ordinal assessments (0: Normal, 1: Mild, 2: Moderate, 3: Severe)
    latent_severity = 0.02 * age + 0.5 * diabetes + rng.normal(0, 1, size=n)
    rater_1 = np.digitize(latent_severity, bins=[-0.5, 0.8, 2.0]).clip(0, 3)
    rater_2 = np.digitize(
        latent_severity + rng.normal(0, 0.4, size=n), bins=[-0.5, 0.8, 2.0]
    ).clip(0, 3)

    # Binary outcome: 30-day mortality
    logit_mort = (
        -3.0
        + 0.04 * (age - 60)
        + 0.7 * gold_standard_ami
        + 0.5 * diabetes
        - 0.6 * treatment_arm
    )
    prob_mort = 1.0 / (1.0 + np.exp(-logit_mort))
    mortality_30d = rng.binomial(1, prob_mort)

    # Survival data: Time-to-event (days, max 365)
    baseline_hazard = 0.001
    hr = np.exp(0.03 * (age - 60) + 0.6 * diabetes - 0.4 * treatment_arm)
    surv_time = rng.exponential(scale=1.0 / (baseline_hazard * hr))
    censoring_time = rng.uniform(30.0, 365.0, size=n)
    time_to_event = np.minimum(surv_time, censoring_time).clip(1.0, 365.0)
    event_death = (surv_time <= censoring_time).astype(int)

    # Continuous lab biomarker with MCAR missingness (for Little's MCAR test & MICE)
    lab_crp = rng.lognormal(mean=1.2, sigma=0.8, size=n)
    lab_ldl = rng.normal(loc=115.0, scale=30.0, size=n).clip(40.0, 240.0)

    # Inject ~8% MCAR missingness into crp and ldl
    crp_missing = rng.binomial(1, 0.08, size=n).astype(bool)
    ldl_missing = rng.binomial(1, 0.07, size=n).astype(bool)
    lab_crp[crp_missing] = np.nan
    lab_ldl[ldl_missing] = np.nan

    df = pd.DataFrame(
        {
            "patient_id": patient_ids,
            "age": np.round(age, 1),
            "sex": sex,
            "bmi": np.round(bmi, 1),
            "egfr": np.round(egfr, 1),
            "hypertension": hypertension,
            "diabetes": diabetes,
            "treatment_arm": treatment_arm,
            "gold_standard_ami": gold_standard_ami,
            "troponin_level": np.round(troponin_level, 2),
            "sbp_device_a": np.round(sbp_device_a, 1),
            "sbp_device_b": np.round(sbp_device_b, 1),
            "rater_1": rater_1,
            "rater_2": rater_2,
            "mortality_30d": mortality_30d,
            "time_to_event": np.round(time_to_event, 1),
            "event_death": event_death,
            "lab_crp": np.round(lab_crp, 2),
            "lab_ldl": np.round(lab_ldl, 1),
        }
    )

    return df


def generate_synthetic_meta_analysis_data() -> pd.DataFrame:
    """Generates synthetic multi-study data for forest plots and meta-analyses."""
    return pd.DataFrame(
        {
            "study": [
                "Smith et al. (2019)",
                "Chen et al. (2020)",
                "Rodriguez et al. (2021)",
                "Kumar et al. (2021)",
                "Takahashi et al. (2022)",
                "Muller et al. (2023)",
                "Patel et al. (2023)",
                "Johnson et al. (2024)",
            ],
            "estimate": [1.42, 1.85, 1.30, 2.10, 1.65, 1.15, 1.72, 1.58],
            "ci_lower": [1.05, 1.32, 0.95, 1.45, 1.18, 0.82, 1.25, 1.14],
            "ci_upper": [1.92, 2.59, 1.78, 3.04, 2.31, 1.61, 2.37, 2.19],
            "p_value": [0.023, 0.001, 0.104, 0.0005, 0.004, 0.420, 0.001, 0.006],
            "weight": [12.4, 15.2, 10.1, 11.8, 14.0, 9.5, 13.6, 13.4],
            "scale": ["OR"] * 8,
        }
    )
