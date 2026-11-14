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
medstat.models.survival.fit_competing_risks_cif(durations, event_observed, event_of_interest=1, competing_event=2, horizons=None, alpha=0.05) -> CompetingRisksCIFResult
medstat.models.survival.fit_competing_risks_regression(df, duration_col, event_col, covariates, event_of_interest=1, competing_event=2, id_col=None, alpha=0.05, se_method='sandwich', n_boot=200, seed=0, score_gap_limit=0.2) -> dict[str, Any]
medstat.models.survival.compare_survival_curves(durations, groups, event_observed) -> dict[str, Any]
medstat.models.ordinal.fit_proportional_odds(y, X, distr='logit', method='bfgs', maxiter=200) -> dict[str, Any]
medstat.models.ordinal.test_proportional_odds(y, X) -> dict[str, Any]
medstat.models.multilevel.calculate_design_effect(y, cluster_ids) -> dict[str, Any]
medstat.diagnostic.roc.compute_time_dependent_roc(durations, events, scores=None, time_horizons=None, competing_event_code=None, n_bootstrap=200, random_state=42, scores_old=None, scores_new=None) -> dict[str, Any]
medstat.diagnostic.calibration.calculate_brier_score(y_true, y_pred) -> dict[str, Any]
medstat.diagnostic.calibration.calculate_calibration_slope_and_intercept(y_true, y_pred) -> dict[str, Any]
medstat.diagnostic.calibration.calculate_ici(y_true, y_pred) -> dict[str, Any]
medstat.diagnostic.calibration.hosmer_lemeshow_test(y_true, y_pred, g=10) -> dict[str, Any]
medstat.diagnostic.dca.calculate_dca(y_true, y_pred, thresholds=None, model_name='Model') -> pd.DataFrame
medstat.diagnostic.validation.bootstrap_internal_validation(df, outcome, predictors, n_bootstrap=200, penalty='none', random_state=42) -> dict[str, Any]
medstat.diagnostic.validation.recalibrate_prediction_model(y_true, p_pred, method='logistic', clip_eps=1e-06) -> dict[str, Any]
medstat.power.sample_size.calculate_riley_sample_size(n_parameters, prevalence, c_statistic=None, cox_snell_r2=None, nagelkerke_r2=None, shrinkage_target=0.9, intercept_margin=0.05, optimism_margin=0.05, mape_target=None) -> dict[str, Any]
medstat.power.sample_size.calculate_diagnostic_auroc_sample_size(auroc_expected, auroc_null=0.5, prevalence=0.5, alpha=0.05, power=0.8, two_sided=True) -> dict[str, Any]
medstat.meta.eggers_test(df_or_theta, se_or_effect_col=None, se_col=None) -> dict[str, Any]
medstat.figures.survival.plot_cif(cif_data, primary_label='Event of Interest', competing_label='Competing Event', show_naive_km=True, time_unit='Months', title='Cumulative Incidence Function Under Competing Risks', out_path='figures/cif_plot.png', target='docx', journal=None, canvas='manuscript', export_vector=True) -> FigureResult
medstat.figures.roc.plot_time_dependent_roc(td_roc_data, mode='curves', title=None, out_path='figures/td_roc.png', target='docx', journal=None, canvas='manuscript', export_vector=True) -> FigureResult
medstat.figures.roc.plot_reclassification(reclass_data=None, p_old=None, p_new=None, y_true=None, thresholds=None, title='Risk Reclassification (Baseline vs Updated Model)', out_path='figures/reclassification.png', target='docx', journal=None, canvas='manuscript', export_vector=True) -> FigureResult
medstat.figures.calibration.plot_calibration(calib_bins, calib_curve=None, brier_score=None, ici=None, title='Model Calibration Curve', out_path='figures/calibration_plot.png', target='docx', journal=None, canvas='manuscript', export_vector=True) -> FigureResult
medstat.figures.calibration.plot_bootstrap_optimism(validation_res, title='Bootstrap Internal Validation (Optimism)', out_path=None, output_path=None, target='docx', journal=None, canvas='manuscript', export_vector=True) -> FigureResult
medstat.figures.calibration.plot_recalibration_comparison(y_true, p_orig=None, p_recal=None, n_bins=10, title='Model Recalibration Comparison', out_path=None, output_path=None, target='docx', journal=None, canvas='manuscript', export_vector=True) -> FigureResult
medstat.figures.dca.plot_dca(dca_data, title='Decision Curve Analysis (Clinical Net Benefit)', out_path='figures/dca_plot.png', target='docx', journal=None, canvas='manuscript', export_vector=True) -> FigureResult
medstat.figures.html.roc.plot_dual_roc_curve(roc_data, paired_roc_data=None, paired_p_value=None, title='Receiver Operating Characteristic (ROC) Curve', out_path='figures/roc_curve.png', target='docx', journal=None, canvas='manuscript') -> DualFigureResult
medstat.figures.html.roc.plot_dual_time_dependent_roc(td_roc_data, mode='curves', title=None, out_path='figures/td_roc.png', target='docx', journal=None, canvas='manuscript') -> DualFigureResult
medstat.figures.html.roc.plot_dual_reclassification(reclass_data=None, p_old=None, p_new=None, y_true=None, thresholds=None, title='Risk Reclassification (Baseline vs Updated Model)', out_path='figures/reclassification.png', target='docx', journal=None, canvas='manuscript') -> DualFigureResult
medstat.figures.html.calibration.plot_dual_calibration(calib_bins, calib_curve=None, brier_score=None, ici=None, title='Model Calibration Curve', out_path='figures/calibration_plot.png', target='docx', journal=None, canvas='manuscript', export_vector=True) -> DualFigureResult
medstat.figures.html.calibration.plot_dual_bootstrap_optimism(validation_res, title='Bootstrap Internal Validation (Optimism)', out_path=None, output_path=None, target='docx', journal=None, canvas='manuscript', export_vector=True) -> DualFigureResult
medstat.figures.html.calibration.plot_dual_recalibration_comparison(y_true, p_orig=None, p_recal=None, n_bins=10, title='Model Recalibration Comparison', out_path=None, output_path=None, target='docx', journal=None, canvas='manuscript', export_vector=True) -> DualFigureResult
medstat.figures.html.dca.plot_dual_dca(dca_data=None, dca_df=None, title='Decision Curve Analysis (Clinical Net Benefit)', out_path='figures/dca_curve.png', target='docx', journal=None, canvas='manuscript') -> DualFigureResult
medstat.figures.illustration.Scene(width='double_col', aspect=0.6, journal='NEJM', canvas='manuscript', seed=7)
medstat.figures.illustration.Scene.vessel_longitudinal(x0, x1, y_floor, lumen_h=14.0, wall=3.0, tissue_h=18.0) -> dict[str, Any]
medstat.figures.illustration.Scene.artery_cross_section(cx, cy, R=14.0, stenosis=0.6, lipid_core=True, rupture=False, thrombus=0.0) -> dict[str, Any]
medstat.figures.illustration.Scene.skin(x0, x1, ytop, thick=(3.0, 11.0, 9.0, 9.0), rete=0.9) -> dict[str, Any]
medstat.figures.illustration.Scene.rbc(cx, cy, r=1.3, rot=0.0, tilt=0.55)
medstat.figures.illustration.Scene.neutrophil(cx, cy, r=2.2, lobes=3, squeeze=1.0, rot=0.0)
medstat.figures.illustration.Scene.macrophage(cx, cy, r=3.2, vacuoles=4, engulfed=0)
medstat.figures.illustration.Scene.lymphocyte(cx, cy, r=1.8)
medstat.figures.illustration.Scene.platelet(cx, cy, r=0.7)
medstat.figures.illustration.Scene.fibroblast(cx, cy, L=5.0, rot=0.0)
medstat.figures.illustration.Scene.eukaryote(cx, cy, r=9.0, organelles=True)
medstat.figures.illustration.Scene.text(x, y, s, size=7.0, ha='center', va='center', weight='normal', color='#222222', z=None, **kw) -> mpl.text.Text
medstat.figures.illustration.Scene.callout(s, target, text_xy, size=6.5, ha=None, dot=True) -> mpl.text.Text
medstat.figures.illustration.Scene.label_column(items, x, y_lo, y_hi, side='right', size=6.5, gap=3.4)
medstat.figures.illustration.Scene.legend_swatches(entries, x, y, dy=3.6, size=6.5, hatch_map=None)
medstat.figures.illustration.Scene.save(out_path, caption_title, alt_text, notes='Schematic illustration; not to scale; not derived from patient images.', export_vector=True, export_svg=True, dpi=600, strict=True) -> FigureResult
```

## Return shapes

- `detect_outliers` returns a tuple `(outlier_mask, info)`; unpack it.
- `check_separation(y, X)` takes `X` without an intercept column (the detector adds one; a constant column raises "rank deficient") and returns `{"is_separated": bool, "recommendation": str}`.
- `compute_iptw_weights` and `medstat.causal.psm.propensity_score_match` raise `ValueError` on a rank-deficient propensity design, and the default `ps_method="logistic"` also raises on separation or non-convergence; neither substitutes an estimator. The explicit `ps_method="firth"` may return scores, subject to its own convergence check. After Gate #3 option A pass `ps_method="firth"`; the estimator used is in `diagnostics["ps_method"]` / `info["ps_method"]`.
- `SampleFlowTracker`: `final_n` is a property; `record_stage` records the N remaining after each exclusion step; call `render_ascii_flow()` (not `to_ascii()`), `to_dict()`, `to_json()`, `to_mermaid()`, or `get_flow_summary()`. `FlowStage.reasons` is a `list[dict]` (not `.reason`).
- `fit_cox_rcs` returns a tuple/mapping subclass `CoxRCSResult(fig, contrast_df, stats_meta)` (no `.fig`, `.contrast_df`, or `.stats` attributes): unpack `fig, contrast_df, stats = fit_cox_rcs(...)` and use `res["non_linear_pvalue"]`, `res["p_overall"]`, `res["knot_locations"]`, `res["ref_value"]`; `fit_logistic_rcs` returns a dict with the same keys. For custom log-scale HR axes, clear minor ticks with `ax.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())` so tick labels do not collide with curve annotations.
- `fit_competing_risks_cif` returns `CompetingRisksCIFResult` with fields `ajf_primary`, `ajf_competing`, `km_naive`, `cif_table`, `horizon_summary`, `event_of_interest`, `competing_event`, `probability_conservation_max_error`, `clinical_warnings`. Note precision consistency: `horizon_summary["cif_primary"]` rounds to 4 decimals whereas `cif_primary_ci` formats the unrounded value to 3 decimals; format percentages (`* 100`) and proportions from the same precision across text, tables, and slides (avoid mixing e.g. `0.567` and `56.8%` from rounded `0.5675`).
- `fit_competing_risks_regression` returns keys `model_cause_specific`, `model_fine_gray`, `comparison_table` (not `summary_df`; columns include `covariate`, `cause_specific_hr`, `cs_95_ci`, `cs_p_value`, `fine_gray_shr`, `fg_95_ci`, `fg_p_value`, `divergent`), `n_original`, `n_expanded`, `fg_se_method`, `fg_score_gap`, `event_distribution`, `interpretation` (singular, not `interpretations`).
- `calculate_e_value` returns `{'original_estimate', 'estimate_type', 'rare_outcome', 'e_value_estimate', 'e_value_ci_limit', 'interpretation'}` (use `e_value_estimate` and `e_value_ci_limit`, not `e_value`).
- `compute_time_dependent_roc` returns `{'time_horizons', 'results_by_horizon', 'trajectory', 'ci_method', 'warnings', 'warning'}` with `results_by_horizon[t]` using `'optimal_threshold'` (plus `'auroc'`, `'ci_lower'`, `'ci_upper'`, `'se'`, `'ci_method'`, `'sensitivity'`, `'specificity'`), whereas `find_optimal_threshold` returns `{'threshold', 'sensitivity', 'specificity', 'fpr', 'tpr', 'index'}`.
- ROC & calibration return dicts: `calculate_roc_curve` -> `['auc', 'fpr', 'tpr', 'thresholds', 'n_pos', 'n_neg']`; `auc_ci_delong` -> `['auc', 'se', 'ci_lower', 'ci_upper', 'n_pos', 'n_neg']`; `delong_paired_test` -> `['auc1', 'auc2', 'difference', 'se_difference', 'ci_lower', 'ci_upper', 'z_statistic', 'p_value']`; `calculate_brier_score` -> `['brier_score', 'brier_scaled', 'brier_reference', 'interpretation', 'n']`; `calculate_calibration_slope_and_intercept` -> `['calibration_slope', 'slope_ci', 'slope_pvalue', 'calibration_intercept', 'intercept_ci', 'intercept_pvalue']`; `calculate_ici` -> `['ici', 'e50', 'e90', 'emax']`; `hosmer_lemeshow_test` -> `['test', 'statistic', 'df', 'p_value', 'is_calibrated']`; `recalibrate_prediction_model` -> `['method', 'n', 'y_true', 'p_orig', 'pre', 'post', 'recalibration_parameters', 'p_recalibrated', 'p_recalibrated_intercept_only', 'p_recalibrated_logistic', 'warnings']`; `bootstrap_internal_validation` -> `['n_total', 'n_events', 'n_predictors', 'events_per_predictor', 'n_bootstrap', 'n_successful_replicates', 'n_regularized_replicates', 'apparent_metrics', 'metrics', 'shrinkage_factor', 'shrunk_coefficients', 'warnings']`.
- Power, ordinal & clustering return dicts: `calculate_riley_sample_size` -> `['n_total', 'n_events', 'n_parameters', 'prevalence', 'events_per_parameter', 'cox_snell_r2', 'nagelkerke_r2', 'r2_max', 'shrinkage_target', 'binding_criterion', 'criteria']`; `fit_proportional_odds` -> `['model', 'summary_df', 'converged', 'predictor_df', 'threshold_df', 'categories', 'k_categories', 'log_likelihood', 'pseudo_r2', 'aic', 'bic', 'nobs']`; `test_proportional_odds` -> `['test', 'omnibus', 'variables', 'interpretation']`; `calculate_design_effect` -> `['n_clusters', 'n_total', 'mean_cluster_size', 'cluster_size_min', 'cluster_size_max', 'icc_cluster', 'design_effect', 'effective_sample_size', 'interpretation']`.
- `Scene` takes `(width='double_col', aspect=0.6, journal='NEJM', canvas='manuscript', seed=7)` (not `w_mm`/`h_mm`); use `Scene.label_column(...)` for stacked labels and `Scene.vessel_longitudinal` / `Scene.artery_cross_section` (there is no `endothelium` helper).
- `ReportDocument` methods: `add_heading`, `add_paragraph`, `add_table`, `add_figure`, `add_callout`, `extract_all_text`.
- `verify_report_integrity`: `delivered_text` is the In-Chat Summary reply, audited with the document; `exported_files` lists data files written beside the report (.csv, .tsv, .xlsx or .parquet; an identifier column such as `patient_id` or any other format fails the audit).
- `IntegrityReport` fields: `passed`, `untraced_numbers`, `phi_violations`, `warnings`, `has_methods`, `has_retention_flow`, `has_causal_caveat`, `citations_count`, `agent_remediation`; `summary()` returns text. It is a dataclass: use `dataclasses.asdict(report)` for JSON.

## Analysis and report notes

- **Minimal report**: `doc = ReportDocument(title=..., results_dict=..., study_design=...)`, then `add_heading` / `add_paragraph` (Methods, Results, retention counts) / `add_table` / `add_figure`. For a file format call the matching `render_*`; for an In-Chat Summary render nothing. Then `verify_report_integrity(doc, exported_files=..., delivered_text=<the chat reply>)` and post that same reply. `SampleFlowTracker` has no slot in the document: write its counts as a retention paragraph and, when the cohort changes, a `plot_retention_flow` figure.
- `results_dict`: store every number the text quotes; a P-value key must contain the token `p`, `p_value`, `p_adj` etc. (`age_p_value`, `re_p` work), and `P < 0.001` traces only if some stored P is below 0.001. State the significance level with a threshold word ("a two-sided P < .05 threshold"); a bare "two-sided P < .05" or "P < .05 was considered significant" is traced as a result.
- `results_dict` must also hold the counts the text quotes (non-events, parameter count, SD used for a unit); list figure files for `exported_files` from `FigureResult.csv_path` (the file is named `<figure stem>.csv`, not `_source.csv`). A `DualFigureResult` carries `qa` at top level.
- `generate_table_one` labels strata by their raw values (`0`, `1`) and lists the outcome as a row: stratify by exposure, drop the outcome row from the covariate list, and rename the strata columns before `add_table`.
- `fit_standard_logistic(y, X)` returns `summary_df` indexed by `X` column names (`x0`, `x1` for an ndarray), `n_obs` (not `n`), and `model` (`odds_ratio`, `or_ci_lower`, `or_ci_upper`, `p_value`); `X` has no intercept column. Pass `X` as a `pd.DataFrame` with named columns so `model.params` is named, or map index positions in `get_params` for `pool_regression_results`.
- `bootstrap_internal_validation`: signature uses `(df, outcome, predictors, ...)` (not `outcome_col`); check the exact signature above.
- `fit_cox_rcs`: returns `(fig, contrast_df, stats)` and takes no `out_path` parameter; save via `save_figure_bundle` if needed.
- Cox: `fit_cox_ph` and `fit_survival_analysis_suite` need numeric covariates; dummy-encode factors against a stated reference level and divide continuous predictors by their unit (10 years, 1 SD) yourself before the fit (the suite's `standardize_continuous` applies one SD scale to all of them). Use the suite for KM curves, the risk table and Schoenfeld tests, and `fit_cox_ph` for the numeric hazard ratios. `fit_cox_ph` returns the numeric `summary_df`; the suite adds KM curves, a risk table and Schoenfeld tests but its `cox_table` holds formatted strings.
- `fit_gee` needs a reset index on a matched frame, and its summary keys differ from `fit_standard_logistic`: print the returned dict's keys before formatting. No helper gives RR or RD for matched or weighted fits; compute them with a Poisson/binomial GLM and sandwich variance and store the values in `results_dict`.
- NRI/IDI: fit a baseline and an extended `fit_standard_logistic` (same rows), pass both predicted-risk vectors to `compute_reclassification_metrics`, and state that the estimates are apparent (in-sample). The dict also holds the input arrays: print only the metrics.
- ROC: `find_optimal_threshold` returns the Youden cutoff as a `>=` rule; `auc_ci_delong` clips its limits to [0, 1].
- Missing data: `MICEImputer` imputes numeric columns only (encode binary predictors 0/1 first); pool the per-imputation fits with `pool_regression_results`.
- `plot_bland_altman` fails its text-overlap QA at some `target` values; try `target="docx"` and shorten the title before accepting a QA failure. `plot_forest` draws pooled rows as ordinary rows. `medstat.meta.create_funnel_plot` is Plotly only; build a static funnel with `apply_journal_theme` and `save_figure_bundle`. `plot_schoenfeld_residuals` plots one covariate without `journal`; for multi-covariate journal-styled panels build with `apply_journal_theme` + `save_figure_bundle`.
- `medstat/reporting/renderers/` is a package directory (`renderers/__init__.py`, `docx.py`, `html.py`, `markdown.py`, `pptx.py`, `pdf.py`), not `renderers.py`. `render_pdf` needs Playwright or soffice; when neither is installed, print the `render_html` output with headless Chrome.
