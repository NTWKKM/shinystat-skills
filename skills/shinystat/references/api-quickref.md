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
medstat.diagnostic.accuracy.calculate_2x2_metrics(tp, fp, fn, tn, ci=0.95) -> dict[str, Any]
medstat.diagnostic.roc.calculate_roc_curve(y_true, y_score, pos_label=1) -> dict[str, Any]
medstat.diagnostic.roc.find_optimal_threshold(fpr, tpr, thresholds, method='youden') -> dict[str, Any]
medstat.diagnostic.roc.auc_ci_delong(y_true, y_score, pos_label=1, alpha=0.05) -> dict[str, Any]
medstat.diagnostic.roc.delong_paired_test(y_true, score1, score2, pos_label=1, alpha=0.05) -> dict[str, Any]
medstat.diagnostic.roc.compute_reclassification_metrics(y_true, p_old, p_new, thresholds=None, n_bootstrap=0, random_state=42) -> dict[str, Any]
medstat.models.glm.fit_standard_logistic(y, X, add_constant=True, maxiter=100) -> dict[str, Any]
medstat.models.survival.fit_cox_ph(df, duration_col, event_col, covariates, penalizer=0.0, check_assumptions=True) -> dict[str, Any]
medstat.models.survival.fit_survival_analysis_suite(df, duration_col, event_col, strata_col=None, covariates=None, penalizer=0.0, standardize_continuous=False, continuous_covariates=None) -> dict[str, Any]
medstat.models.survival.schoenfeld_diagnostics(cph, df, time_transform='rank') -> dict[str, Any]
medstat.data.missing.MICEImputer(n_imputations=5, max_iter=10, random_state=42)
medstat.data.missing.pool_regression_results(models, model_type='linear', get_params=None, alpha=0.05) -> PooledRegressionResults
medstat.reporting.table1.generate_table_one(df, strata=None, continuous_vars=None, categorical_vars=None, nonnormal_vars=None, include_smd=True, include_p=True, digits=1) -> pd.DataFrame
medstat.agreement.icc.calculate_icc(df, targets, raters, ratings, icc_type=None, alpha=0.05) -> pd.DataFrame
medstat.agreement.bland_altman.calculate_bland_altman(df, method1, method2, ci=0.95) -> dict[str, Any]
medstat.meta.models.run_meta_analysis(df, method_re='DL', use_hksj=False, alpha=0.05, effect_col=None, se_col=None, study_col=None, model=None, method=None, exponentiate=None, **kwargs) -> dict[str, Any]
medstat.figures.forest.plot_forest(data, scale='OR', title='Forest Plot of Effect Estimates', out_path='figures/forest_plot.png', target='docx', journal=None, canvas='manuscript', export_vector=True, null_value=None, favor_left_label=None, favor_right_label=None) -> FigureResult
medstat.figures.html.forest.plot_dual_forest(data, scale='OR', title='Forest Plot of Effect Estimates', out_path='figures/forest_plot.png', target='docx', journal=None, canvas='manuscript', favor_left_label=None, favor_right_label=None) -> DualFigureResult
medstat.figures.html.survival.plot_dual_kaplan_meier(km_curves, risk_table=None, log_rank_p=None, time_unit='Days', title='Kaplan-Meier Overall Survival', out_path='figures/kaplan_meier.png', target='docx', journal=None, canvas='manuscript') -> DualFigureResult
medstat.figures.agreement.plot_bland_altman(ba_data, units='mmHg', title='Bland-Altman Limits of Agreement', out_path='figures/bland_altman.png', target='docx', journal=None, canvas='manuscript', export_vector=True) -> FigureResult
medstat.figures.roc.plot_roc_curve(roc_data, paired_roc_data=None, paired_p_value=None, title='Receiver Operating Characteristic (ROC) Curve', out_path='figures/roc_curve.png', target='docx', journal=None, canvas='manuscript', export_vector=True) -> FigureResult
medstat.figures.retention.plot_retention_flow(retention_data, title='Participant Retention Flow Diagram (STROBE / CONSORT)', out_path='figures/retention_flow.png', target='docx', journal=None, canvas='manuscript') -> FigureResult
medstat.reporting.ir.ReportDocument(title, authors=<factory>, date='', blocks=<factory>, results_dict=<factory>, institution='', study_design=None)
medstat.reporting.renderers.render_pptx(doc, out_path) -> str
medstat.reporting.renderers.render_pdf(doc, out_path) -> str
medstat.reporting.ir.ReportDocument.add_heading(text, level=1) -> ReportDocument
medstat.reporting.ir.ReportDocument.add_paragraph(text) -> ReportDocument
medstat.reporting.ir.ReportDocument.add_table(df, caption, footnote=None, style='nejm') -> ReportDocument
medstat.reporting.ir.ReportDocument.add_figure(figure) -> ReportDocument
medstat.reporting.ir.ReportDocument.add_callout(text, level='info') -> ReportDocument
medstat.causal.psm.propensity_score_match(df, treatment_col, covariates, caliper=0.2, ratio=1, ps_method='logistic') -> tuple[pd.DataFrame, dict[str, Any]]
medstat.causal.balance.compute_balance_table(df, treatment, covariates, matched_df=None, weights_col=None) -> pd.DataFrame
medstat.figures.balance.plot_love(love_data, threshold=0.1, title='Covariate Balance Before and After Propensity Score Matching', out_path='figures/love_plot.png', target='docx', journal=None, canvas='manuscript', export_vector=True) -> FigureResult
medstat.figures.flow.cohort_flow_from_tracker(tracker, groups=None, out_path='figures/cohort.png', journal='NEJM', width='one_half_col', title='Cohort selection flow diagram', seed=1) -> FigureResult
medstat.models.multilevel.fit_gee(y, X, cluster_ids, family='binomial', cov_struct='exchangeable', add_constant=True, time=None) -> dict[str, Any]
```

## Return shapes

- `detect_outliers` returns a tuple `(outlier_mask, info)`; unpack it.
- `check_separation(y, X)` takes `X` without an intercept column (the detector adds one; a constant column raises "rank deficient") and returns `{"is_separated": bool, "recommendation": str}`.
- `compute_iptw_weights` and `medstat.causal.psm.propensity_score_match` raise `ValueError` on a rank-deficient propensity design, and the default `ps_method="logistic"` also raises on separation or non-convergence; neither substitutes an estimator. The explicit `ps_method="firth"` may return scores, subject to its own convergence check. After Gate #3 option A pass `ps_method="firth"`; the estimator used is in `diagnostics["ps_method"]` / `info["ps_method"]`.
- `SampleFlowTracker.final_n` is a property; `record_stage` records the N remaining after each exclusion step, ending at the primary analytic N.
- `fit_cox_rcs` returns a tuple `(fig, contrast_df, stats)` that also supports `res["non_linear_pvalue"]`, `res["p_overall"]`, `res["knot_locations"]`; `fit_logistic_rcs` returns a dict with the same keys.
- `verify_report_integrity`: `delivered_text` is the In-Chat Summary reply, audited with the document; `exported_files` lists data files written beside the report (.csv, .tsv, .xlsx or .parquet; an identifier column such as `patient_id` or any other format fails the audit).
- `IntegrityReport` fields: `passed`, `untraced_numbers`, `phi_violations`, `warnings`, `has_methods`, `has_retention_flow`, `has_causal_caveat`, `citations_count`, `agent_remediation`; `summary()` returns text. It is a dataclass: use `dataclasses.asdict(report)` for JSON.

## Analysis and report notes

- **Minimal report**: `doc = ReportDocument(title=..., results_dict=..., study_design=...)`, then `add_heading` / `add_paragraph` (Methods, Results, retention counts) / `add_table` / `add_figure`. For a file format call the matching `render_*`; for an In-Chat Summary render nothing. Then `verify_report_integrity(doc, exported_files=..., delivered_text=<the chat reply>)` and post that same reply. `SampleFlowTracker` has no slot in the document: write its counts as a retention paragraph and, when the cohort changes, a `plot_retention_flow` figure.
- `results_dict`: store every number the text quotes; a P-value key must contain the token `p`, `p_value`, `p_adj` etc. (`age_p_value`, `re_p` work), and `P < 0.001` traces only if some stored P is below 0.001. State the significance level with a threshold word ("a two-sided P < .05 threshold"); a bare "two-sided P < .05" or "P < .05 was considered significant" is traced as a result.
- `results_dict` must also hold the counts the text quotes (non-events, parameter count, SD used for a unit); list figure files for `exported_files` from `FigureResult.csv_path` (the file is named `<figure stem>.csv`, not `_source.csv`). A `DualFigureResult` carries `qa` at top level.
- `generate_table_one` labels strata by their raw values (`0`, `1`) and lists the outcome as a row: stratify by exposure, drop the outcome row from the covariate list, and rename the strata columns before `add_table`.
- `fit_standard_logistic(y, X)` returns `summary_df` indexed by `X` column names (`x0`, `x1` for an ndarray) with `odds_ratio`, `or_ci_lower`, `or_ci_upper`, `p_value`; `X` has no intercept column.
- Cox: `fit_cox_ph` and `fit_survival_analysis_suite` need numeric covariates; dummy-encode factors against a stated reference level and divide continuous predictors by their unit (10 years, 1 SD) yourself before the fit (the suite's `standardize_continuous` applies one SD scale to all of them). Use the suite for KM curves, the risk table and Schoenfeld tests, and `fit_cox_ph` for the numeric hazard ratios. `fit_cox_ph` returns the numeric `summary_df`; the suite adds KM curves, a risk table and Schoenfeld tests but its `cox_table` holds formatted strings.
- `fit_gee` needs a reset index on a matched frame, and its summary keys differ from `fit_standard_logistic`: print the returned dict's keys before formatting. No helper gives RR or RD for matched or weighted fits; compute them with a Poisson/binomial GLM and sandwich variance and store the values in `results_dict`.
- NRI/IDI: fit a baseline and an extended `fit_standard_logistic` (same rows), pass both predicted-risk vectors to `compute_reclassification_metrics`, and state that the estimates are apparent (in-sample). The dict also holds the input arrays: print only the metrics.
- ROC: `find_optimal_threshold` returns the Youden cutoff as a `>=` rule; `auc_ci_delong` clips its limits to [0, 1].
- Missing data: `MICEImputer` imputes numeric columns only (encode binary predictors 0/1 first); pool the per-imputation fits with `pool_regression_results`.
- `plot_bland_altman` fails its text-overlap QA at some `target` values; try `target="docx"` and shorten the title before accepting a QA failure. `plot_forest` draws pooled rows as ordinary rows. `medstat.meta.create_funnel_plot` is Plotly only; build a static funnel with `apply_journal_theme` and `save_figure_bundle`.
- `render_pdf` needs Playwright or soffice; when neither is installed, print the `render_html` output with headless Chrome.
