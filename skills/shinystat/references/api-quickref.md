# medstat API Quick Reference

Engine call signatures, held equal to `inspect.signature` by `test_api_quickref_matches_engine`. Call these as written; open `scripts/medstat/` source only for a function not listed here.

```text
medstat.data.clean.audit_missingness(df, var_meta=None, missing_codes=None) -> MissingnessAudit
medstat.data.clean.detect_outliers(series, method='iqr', threshold=None) -> tuple[pd.Series, dict[str, Any]]
medstat.data.clean.littles_mcar_test(df, cols=None, max_iter=100, tol=1e-05, ridge_eps=1e-06) -> LittlesMCARResult
medstat.data.retention.SampleFlowTracker(initial_n=None, n_initial=None, initial_name='Initial Enrolled / Assessed', design=<FlowDesign.STROBE: 'STROBE'>)
medstat.data.retention.SampleFlowTracker.record_stage(stage_name, n_remaining, n_excluded=None, reason='', details=None) -> FlowStage
medstat.models.firth.check_separation(y, X) -> dict[str, Any]
medstat.models.firth.fit_firth_logistic(y, X, feature_names=None, fit_intercept=True, penalty_weight=0.5, ci_method='pl', alpha=0.05) -> dict[str, Any]
medstat.models.splines.fit_cox_rcs(df, duration_col, event_col, rcs_var=None, adjust_cols=None, knots=None, ref_value=None, n_knots=None, **kwargs) -> CoxRCSResult
medstat.models.splines.fit_logistic_rcs(df, outcome, spline_var, covariates=None, n_knots=4, ref_value=None, **kwargs) -> dict[str, Any]
medstat.models.sensitivity.calculate_e_value(estimate, lower=None, upper=None, estimate_type='RR', rare_outcome=False) -> dict[str, Any]
medstat.causal.iptw.compute_iptw_weights(df, treatment_col=None, covariates=None, estimand='ATE', stabilized=True, truncate_quantiles=(0.01, 0.99), ps_clip=(0.0001, 0.9999), treatment=None, truncation=None, ps_method='logistic') -> IPTWResult
medstat.causal.iptw.fit_iptw_outcome_model(df, outcome_col, treatment_col=None, weights_col='iptw_weight', covariates=None, outcome_type='continuous', duration_col=None, robust_cov_type='HC1', treatment=None, model_type=None, event_col=None) -> dict[str, Any]
medstat.reporting.integrity.verify_report_integrity(doc, allowed_untraced=None, study_design=None, delivered_text=None, exported_files=None) -> IntegrityReport
medstat.reporting.renderers.render_html(doc, out_path) -> str
medstat.reporting.renderers.render_docx(doc, out_path) -> str
medstat.reporting.renderers.render_markdown(doc, out_path, assets_dir_name='figures') -> str
```

## Return shapes

- `detect_outliers` returns a tuple `(outlier_mask, info)`; unpack it.
- `check_separation(y, X)` takes `X` without an intercept column (the detector adds one; a constant column raises "rank deficient") and returns `{"is_separated": bool, "recommendation": str}`.
- `compute_iptw_weights` and `medstat.causal.psm.propensity_score_match` raise `ValueError` on a rank-deficient or separated propensity model and never substitute an estimator. After Gate #3 option A pass `ps_method="firth"`; the estimator used is in `diagnostics["ps_method"]` / `info["ps_method"]`.
- `SampleFlowTracker.final_n` is a property; `record_stage` records the N remaining after each exclusion step, ending at the primary analytic N.
- `fit_cox_rcs` returns a tuple `(fig, contrast_df, stats)` that also supports `res["non_linear_pvalue"]`, `res["p_overall"]`, `res["knot_locations"]`; `fit_logistic_rcs` returns a dict with the same keys.
- `verify_report_integrity`: `delivered_text` is the In-Chat Summary reply, audited with the document; `exported_files` lists data files written beside the report (.csv, .tsv, .xlsx or .parquet; an identifier column such as `patient_id` or any other format fails the audit).
- `IntegrityReport` fields: `passed`, `untraced_numbers`, `phi_violations`, `warnings`, `has_methods`, `has_retention_flow`, `has_causal_caveat`, `citations_count`, `agent_remediation`; `summary()` returns text. It is a dataclass: use `dataclasses.asdict(report)` for JSON.
