"""
Publication Table Renderers (NEJM, JAMA, APA 7).

Formats biostatistical results and regression estimates into publication-standard
HTML tables and clean ASCII text.
"""

from __future__ import annotations

import html
import math
from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np
import pandas as pd


@dataclass
class Estimate:
    term: str
    label: str
    estimate: float
    ci_lower: float
    ci_upper: float
    p_value: float
    scale: str = "OR"  # "OR", "HR", "RR", "Beta", "IRR", "MD"
    reference: bool = False
    ref_label: str = "Reference"
    std_error: float | None = None
    n: int | None = None
    events: int | None = None


@dataclass
class ModelMeta:
    estimator: str
    n_total: int
    n_events: int | None = None
    outcome_name: str | None = None
    exposure_name: str | None = None
    adjusted_for: list[str] = field(default_factory=list)
    software_version: str = (
        "medstat-core (Python 3.12+ statsmodels/lifelines/firthmodels)"
    )
    seed: int | None = None
    notes: str | None = None


@dataclass
class EstimateTable:
    title: str
    rows: list[Estimate]
    meta: ModelMeta | None = None
    confidence_level: float = 0.95


def format_journal_p_value(p: float, style: str = "NEJM") -> str:
    """
    Format p-value per journal conventions.
    """
    if p is None or math.isnan(p):
        return "—"

    style_upper = style.upper()
    if style_upper == "NEJM":
        if p < 0.001:
            return "P<0.001"
        elif p < 0.01:
            return f"P={p:.3f}"
        elif p > 0.99:
            return "P>0.99"
        else:
            return f"P={p:.2f}"
    elif style_upper == "JAMA":
        # JAMA: no leading zero before decimal
        if p < 0.001:
            return "P<.001"
        elif p < 0.01:
            val_str = f"{p:.3f}".lstrip("0")
            return f"P={val_str}"
        elif p > 0.99:
            return "P>.99"
        else:
            val_str = f"{p:.2f}".lstrip("0")
            return f"P={val_str}"
    else:  # APA 7
        if p < 0.001:
            return "< .001"
        elif p < 0.01:
            return f"{p:.3f}".lstrip("0")
        elif p > 0.99:
            return "> .99"
        else:
            return f"{p:.2f}".lstrip("0")


class PublicationRenderer:
    """
    Renders EstimateTable instances into publication-grade HTML tables.
    """

    @classmethod
    def render_html(
        cls, table: EstimateTable, style: Literal["NEJM", "JAMA", "APA7"] = "NEJM"
    ) -> str:
        s = style.upper()
        if s == "APA7" or s == "APA":
            return cls._render_apa7(table)
        elif s == "JAMA":
            return cls._render_jama(table)
        else:
            return cls._render_nejm(table)

    @classmethod
    def _render_nejm(cls, table: EstimateTable) -> str:
        pct_ci = f"{int(table.confidence_level * 100)}%"
        scale_label = table.rows[0].scale if table.rows else "Estimate"

        html_lines = [
            "<table style='border-collapse: collapse; width: 100%; font-family: -apple-system, sans-serif; font-size: 14px;'>",
            f"  <caption style='caption-side: top; text-align: left; font-weight: bold; margin-bottom: 8px;'>{html.escape(table.title)}</caption>",
            "  <thead>",
            "    <tr style='border-top: 2px solid #000; border-bottom: 1px solid #000;'>",
            "      <th style='text-align: left; padding: 6px 12px;'>Variable</th>",
            f"      <th style='text-align: right; padding: 6px 12px;'>{scale_label} ({pct_ci} CI)</th>",
            "      <th style='text-align: right; padding: 6px 12px;'>P Value</th>",
            "    </tr>",
            "  </thead>",
            "  <tbody>",
        ]

        for r in table.rows:
            if r.reference:
                est_str = r.ref_label
                p_str = "—"
            else:
                if math.isnan(r.ci_lower) or math.isnan(r.ci_upper):
                    est_str = f"{r.estimate:.2f} (—)"
                else:
                    est_str = f"{r.estimate:.2f} ({r.ci_lower:.2f}–{r.ci_upper:.2f})"
                p_str = format_journal_p_value(r.p_value, style="NEJM")

            html_lines.append(
                f"    <tr>\n"
                f"      <td style='padding: 6px 12px;'>{html.escape(r.label)}</td>\n"
                f"      <td style='text-align: right; padding: 6px 12px;'>{est_str}</td>\n"
                f"      <td style='text-align: right; padding: 6px 12px;'>{p_str}</td>\n"
                f"    </tr>"
            )

        html_lines.extend(
            [
                "  </tbody>",
                "  <tfoot>",
                "    <tr style='border-top: 2px solid #000;'>",
                "      <td colspan='3' style='font-size: 12px; color: #555; padding-top: 6px;'>",
            ]
        )

        if table.meta:
            m = table.meta
            foot = f"Model: {m.estimator}, Total N = {m.n_total}"
            if m.n_events is not None:
                foot += f", Events = {m.n_events}"
            if m.adjusted_for:
                foot += f". Adjusted for: {', '.join(m.adjusted_for)}"
            html_lines.append(f"        {html.escape(foot)}")

        html_lines.extend(["      </td>", "    </tr>", "  </tfoot>", "</table>"])
        return "\n".join(html_lines)

    @classmethod
    def _render_jama(cls, table: EstimateTable) -> str:
        # JAMA uses clean borders and slightly compact typography
        return cls._render_nejm(table).replace(
            "caption-side: top", "caption-side: top; color: #111;"
        )

    @classmethod
    def _render_apa7(cls, table: EstimateTable) -> str:
        pct_ci = f"{int(table.confidence_level * 100)}%"
        scale_label = table.rows[0].scale if table.rows else "Estimate"

        html_lines = [
            "<table style='border-collapse: collapse; width: 100%; font-family: Times New Roman, serif; font-size: 14px;'>",
            f"  <caption style='caption-side: top; text-align: left; font-style: italic; margin-bottom: 8px;'>{html.escape(table.title)}</caption>",
            "  <thead>",
            "    <tr style='border-top: 1px solid #000; border-bottom: 1px solid #000;'>",
            "      <th style='text-align: left; padding: 6px 12px; font-weight: normal;'>Predictor</th>",
            f"      <th style='text-align: right; padding: 6px 12px; font-weight: normal;'><em>{scale_label}</em></th>",
            f"      <th style='text-align: right; padding: 6px 12px; font-weight: normal;'>{pct_ci} CI</th>",
            "      <th style='text-align: right; padding: 6px 12px; font-weight: normal;'><em>p</em></th>",
            "    </tr>",
            "  </thead>",
            "  <tbody>",
        ]

        for r in table.rows:
            if r.reference:
                est_str = r.ref_label
                ci_str = "—"
                p_str = "—"
            else:
                est_str = f"{r.estimate:.2f}"
                if math.isnan(r.ci_lower) or math.isnan(r.ci_upper):
                    ci_str = "—"
                else:
                    ci_str = f"[{r.ci_lower:.2f}, {r.ci_upper:.2f}]"
                p_str = format_journal_p_value(r.p_value, style="APA7")

            html_lines.append(
                f"    <tr>\n"
                f"      <td style='padding: 6px 12px;'>{html.escape(r.label)}</td>\n"
                f"      <td style='text-align: right; padding: 6px 12px;'>{est_str}</td>\n"
                f"      <td style='text-align: right; padding: 6px 12px;'>{ci_str}</td>\n"
                f"      <td style='text-align: right; padding: 6px 12px;'>{p_str}</td>\n"
                f"    </tr>"
            )

        html_lines.extend(
            [
                "  </tbody>",
                "  <tfoot>",
                "    <tr style='border-top: 1px solid #000;'>",
                "      <td colspan='4' style='font-size: 12px; font-style: italic; padding-top: 6px;'>",
                "        <em>Note</em>. CI = confidence interval.",
                "      </td>",
                "    </tr>",
                "  </tfoot>",
                "</table>",
            ]
        )
        return "\n".join(html_lines)


def render_nejm_table(table: EstimateTable) -> str:
    """Render table in NEJM publication style HTML."""
    return PublicationRenderer.render_html(table, style="NEJM")


def render_jama_table(table: EstimateTable) -> str:
    """Render table in JAMA publication style HTML."""
    return PublicationRenderer.render_html(table, style="JAMA")


def render_apa_table(table: EstimateTable) -> str:
    """Render table in APA 7 publication style HTML."""
    return PublicationRenderer.render_html(table, style="APA7")


def render_ascii_table(table: EstimateTable) -> str:
    """Render table as plain text ASCII table."""
    headers = [
        "Variable",
        f"{table.rows[0].scale if table.rows else 'Est'} (95% CI)",
        "P-Value",
    ]
    rows = []
    for r in table.rows:
        if r.reference:
            rows.append([r.label, r.ref_label, "—"])
        else:
            if math.isnan(r.ci_lower) or math.isnan(r.ci_upper):
                ci_formatted = f"{r.estimate:.2f} (—)"
            else:
                ci_formatted = f"{r.estimate:.2f} ({r.ci_lower:.2f}, {r.ci_upper:.2f})"
            rows.append(
                [
                    r.label,
                    ci_formatted,
                    format_journal_p_value(r.p_value),
                ]
            )

    col_widths = [max(len(str(x)) for x in col) for col in zip(*([headers] + rows))]
    line = "+-" + "-+-".join("-" * w for w in col_widths) + "-+"

    output = [table.title, line]
    hdr_str = "| " + " | ".join(f"{h:<{w}}" for h, w in zip(headers, col_widths)) + " |"
    output.extend([hdr_str, line])

    for row in rows:
        row_str = (
            "| " + " | ".join(f"{val:<{w}}" for val, w in zip(row, col_widths)) + " |"
        )
        output.append(row_str)

    output.append(line)
    return "\n".join(output)


def render_records_table(
    title: str,
    records: list[dict[str, Any]],
    style: Literal["NEJM", "JAMA", "APA7"] = "NEJM",
    note: str | None = None,
) -> str:
    """Render a list of dictionary records (e.g. Table 1, ICC) into a publication-grade HTML table."""
    if not records:
        return f"<table style='border-collapse: collapse; width: 100%; font-family: -apple-system, sans-serif;'><caption>{html.escape(title)}</caption><tbody><tr><td>No records</td></tr></tbody></table>"

    headers = list(records[0].keys())
    is_apa = style.upper() in ("APA", "APA7")
    font_fam = (
        "Times New Roman, serif"
        if is_apa
        else "-apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif"
    )
    border_rule = "1px solid #000" if is_apa else "2px solid #000"
    caption_style = (
        "caption-side: top; text-align: left; font-style: italic; margin-bottom: 8px;"
        if is_apa
        else "caption-side: top; text-align: left; font-weight: bold; margin-bottom: 8px;"
    )

    lines = [
        f"<table style='border-collapse: collapse; width: 100%; font-family: {font_fam}; font-size: 14px;'>",
        f"  <caption style='{caption_style}'>{html.escape(title)}</caption>",
        "  <thead>",
        f"    <tr style='border-top: {border_rule}; border-bottom: 1px solid #000;'>",
    ]

    for idx, h in enumerate(headers):
        align = "left" if idx == 0 else "right"
        lines.append(
            f"      <th style='text-align: {align}; padding: 6px 12px; font-weight: {'normal' if is_apa else 'bold'};'>{html.escape(str(h))}</th>"
        )

    lines.extend(
        [
            "    </tr>",
            "  </thead>",
            "  <tbody>",
        ]
    )

    for row in records:
        lines.append("    <tr>")
        for idx, h in enumerate(headers):
            align = "left" if idx == 0 else "right"
            val = row.get(h, "")
            val_str = f"{val:.3f}" if isinstance(val, float) else str(val)
            lines.append(
                f"      <td style='text-align: {align}; padding: 6px 12px;'>{html.escape(val_str)}</td>"
            )
        lines.append("    </tr>")

    lines.extend(
        [
            "  </tbody>",
            "  <tfoot>",
            f"    <tr style='border-top: {border_rule};'>",
            f"      <td colspan='{len(headers)}' style='font-size: 12px; color: #555; padding-top: 6px; font-style: {'italic' if is_apa else 'normal'};'>",
        ]
    )
    if note:
        lines.append(f"        {html.escape(note)}")
    lines.extend(["      </td>", "    </tr>", "  </tfoot>", "</table>"])
    return "\n".join(lines)


def render_diagnostic_table(
    title: str,
    diag_data: dict[str, Any],
    style: Literal["NEJM", "JAMA", "APA7"] = "NEJM",
) -> str:
    """Render diagnostic test accuracy, ROC AUC, and calibration as a publication-grade HTML table."""
    records: list[dict[str, Any]] = []

    # Accuracy metrics at cutoff
    acc = diag_data.get("accuracy_at_cutoff", diag_data.get("accuracy", {}))
    if isinstance(acc, dict):

        def _fmt_ci(
            est: float | dict | None, ci: tuple | list | None, is_pct: bool = True
        ) -> str:
            if isinstance(est, dict):
                if ci is None and "ci_lower" in est and "ci_upper" in est:
                    ci = [est["ci_lower"], est["ci_upper"]]
                elif ci is None and "ci" in est:
                    ci = est["ci"]
                est = est.get("estimate", est.get("value"))
            if est is None:
                return "—"
            try:
                est_val = float(est)
            except (ValueError, TypeError):
                return "—"
            if is_pct:
                val = f"{est_val * 100:.1f}%"
                if ci and len(ci) == 2:
                    return f"{val} ({ci[0] * 100:.1f}–{ci[1] * 100:.1f}%)"
                return val
            else:
                val = f"{est_val:.2f}"
                if ci and len(ci) == 2:
                    return f"{val} ({ci[0]:.2f}–{ci[1]:.2f})"
                return val

        if "sensitivity" in acc:
            records.append(
                {
                    "Diagnostic Metric": "Sensitivity",
                    "Estimate (95% CI)": _fmt_ci(
                        acc.get("sensitivity"), acc.get("sensitivity_ci"), is_pct=True
                    ),
                }
            )
        if "specificity" in acc:
            records.append(
                {
                    "Diagnostic Metric": "Specificity",
                    "Estimate (95% CI)": _fmt_ci(
                        acc.get("specificity"), acc.get("specificity_ci"), is_pct=True
                    ),
                }
            )
        if "ppv" in acc:
            records.append(
                {
                    "Diagnostic Metric": "Positive Predictive Value (PPV)",
                    "Estimate (95% CI)": _fmt_ci(
                        acc.get("ppv"), acc.get("ppv_ci"), is_pct=True
                    ),
                }
            )
        if "npv" in acc:
            records.append(
                {
                    "Diagnostic Metric": "Negative Predictive Value (NPV)",
                    "Estimate (95% CI)": _fmt_ci(
                        acc.get("npv"), acc.get("npv_ci"), is_pct=True
                    ),
                }
            )
        if "positive_likelihood_ratio" in acc or "lr_positive" in acc:
            lr_pos = acc.get("positive_likelihood_ratio", acc.get("lr_positive"))
            records.append(
                {
                    "Diagnostic Metric": "Positive Likelihood Ratio (LR+)",
                    "Estimate (95% CI)": _fmt_ci(
                        lr_pos, acc.get("lr_positive_ci"), is_pct=False
                    ),
                }
            )
        if "negative_likelihood_ratio" in acc or "lr_negative" in acc:
            lr_neg = acc.get("negative_likelihood_ratio", acc.get("lr_negative"))
            records.append(
                {
                    "Diagnostic Metric": "Negative Likelihood Ratio (LR-)",
                    "Estimate (95% CI)": _fmt_ci(
                        lr_neg, acc.get("lr_negative_ci"), is_pct=False
                    ),
                }
            )
        if "diagnostic_odds_ratio" in acc or "dor" in acc:
            dor_val = acc.get("diagnostic_odds_ratio", acc.get("dor"))
            records.append(
                {
                    "Diagnostic Metric": "Diagnostic Odds Ratio (DOR)",
                    "Estimate (95% CI)": _fmt_ci(
                        dor_val, acc.get("dor_ci"), is_pct=False
                    ),
                }
            )

    # ROC AUC
    roc = diag_data.get("roc", {})
    if isinstance(roc, dict) and "auc" in roc:
        auc_val = float(roc["auc"])
        auc_ci = roc.get("auc_ci", [roc.get("ci_lower"), roc.get("ci_upper")])
        ci_str = (
            f" ({auc_ci[0]:.3f}–{auc_ci[1]:.3f})"
            if auc_ci and auc_ci[0] is not None and auc_ci[1] is not None
            else ""
        )
        records.append(
            {
                "Diagnostic Metric": "Area Under ROC Curve (AUC)",
                "Estimate (95% CI)": f"{auc_val:.3f}{ci_str}",
            }
        )

    # Calibration
    cal = diag_data.get("calibration", {})
    if isinstance(cal, dict) and "brier" in cal:
        brier_entry = cal["brier"]
        brier_val = None
        if isinstance(brier_entry, dict):
            brier_val = brier_entry.get("brier_score")
        elif isinstance(brier_entry, (int, float, np.number)):
            brier_val = float(brier_entry)
        elif brier_entry is not None:
            try:
                brier_val = float(brier_entry)
            except (ValueError, TypeError):
                brier_val = None

        if brier_val is not None:
            prob_source = diag_data.get("probability_source") or cal.get(
                "probability_source"
            )
            brier_label = (
                "Brier Score (Calibration - in-sample apparent estimate)"
                if prob_source
                else "Brier Score (Calibration)"
            )
            records.append(
                {
                    "Diagnostic Metric": brier_label,
                    "Estimate (95% CI)": f"{float(brier_val):.4f}",
                }
            )

    note = "Confidence intervals for proportions are calculated via Wilson score method; AUC confidence interval via DeLong test."
    apparent_note = diag_data.get("apparent_estimates_note") or (
        cal.get("apparent_estimates_note") if isinstance(cal, dict) else None
    )
    if apparent_note:
        note += f" Note: {apparent_note}"
    return render_records_table(title, records, style=style, note=note)


def render_bland_altman_table(
    title: str,
    ba_data: dict[str, Any],
    style: Literal["NEJM", "JAMA", "APA7"] = "NEJM",
) -> str:
    """Render Bland-Altman mean difference and Limits of Agreement as a publication-grade HTML table."""
    records: list[dict[str, Any]] = []

    mean_diff = ba_data.get(
        "mean_difference", ba_data.get("bias", ba_data.get("mean_diff"))
    )
    ci_mean = ba_data.get(
        "mean_diff_ci", ba_data.get("bias_ci", ba_data.get("ci_mean_diff"))
    )
    loa_lower = ba_data.get("lower_loa", ba_data.get("loa_lower"))
    ci_loa_l = ba_data.get(
        "lower_loa_ci", ba_data.get("loa_lower_ci", ba_data.get("ci_lower_loa"))
    )
    loa_upper = ba_data.get("upper_loa", ba_data.get("loa_upper"))
    ci_loa_u = ba_data.get(
        "upper_loa_ci", ba_data.get("loa_upper_ci", ba_data.get("ci_upper_loa"))
    )

    def _fmt(val: float | None, ci: tuple | list | None) -> str:
        if val is None:
            return "—"
        res = f"{val:.3f}"
        if ci and len(ci) == 2 and ci[0] is not None and ci[1] is not None:
            res += f" ({ci[0]:.3f} to {ci[1]:.3f})"
        return res

    if mean_diff is not None:
        records.append(
            {
                "Agreement Parameter": "Mean Difference (Bias)",
                "Estimate (95% CI)": _fmt(mean_diff, ci_mean),
            }
        )
    if loa_lower is not None:
        records.append(
            {
                "Agreement Parameter": "Lower Limit of Agreement (LoA)",
                "Estimate (95% CI)": _fmt(loa_lower, ci_loa_l),
            }
        )
    if loa_upper is not None:
        records.append(
            {
                "Agreement Parameter": "Upper Limit of Agreement (LoA)",
                "Estimate (95% CI)": _fmt(loa_upper, ci_loa_u),
            }
        )

    note = "Limits of agreement computed as mean difference ± 1.96 standard deviations. CIs based on Bland & Altman (1999) large-sample variance."
    return render_records_table(title, records, style=style, note=note)


def render_balance_table(
    title: str,
    bal_data: dict[str, Any],
    style: Literal["NEJM", "JAMA", "APA7"] = "NEJM",
) -> str:
    """Render Austin 2009 Covariate Balance table (Pre/Post/Weighted SMD) as a publication-grade HTML table."""
    covariates = bal_data.get("covariates", [])
    smd_pre = bal_data.get("smd_pre", bal_data.get("smd_raw", []))
    smd_post = bal_data.get(
        "smd_post", bal_data.get("smd_matched", bal_data.get("post_smd", []))
    )
    smd_weighted = bal_data.get(
        "smd_weighted",
        bal_data.get("weighted_smd", bal_data.get("Post_Weight_SMD", [])),
    )

    has_weighted = len(smd_weighted) > 0
    include_post_matching = (not has_weighted) or any(
        (v is not None and math.isfinite(v)) for v in smd_post
    )

    records: list[dict[str, Any]] = []
    for idx, cov in enumerate(covariates):
        pre_val = smd_pre[idx] if idx < len(smd_pre) else None
        post_val = smd_post[idx] if idx < len(smd_post) else None
        weight_val = smd_weighted[idx] if idx < len(smd_weighted) else None

        pre_valid = pre_val is not None and math.isfinite(pre_val)
        post_valid = post_val is not None and math.isfinite(post_val)
        weight_valid = weight_val is not None and math.isfinite(weight_val)

        pre_str = f"{pre_val:.3f}" if pre_valid else "—"
        post_str = f"{post_val:.3f}" if post_valid else "—"

        eval_val = weight_val if weight_valid else post_val
        if eval_val is not None and math.isfinite(eval_val):
            status = "Balanced (SMD < 0.10)" if abs(eval_val) < 0.10 else "Imbalanced"
        else:
            status = "—"

        rec = {
            "Covariate": str(cov),
            "Pre-Matching SMD": pre_str,
        }
        if include_post_matching:
            rec["Post-Matching SMD"] = post_str
        if has_weighted:
            rec["IPTW Weighted SMD"] = f"{weight_val:.3f}" if weight_valid else "—"
        rec["Balance Status"] = status
        records.append(rec)

    note = "Standardized Mean Difference (SMD) evaluates covariate balance; SMD < 0.10 indicates negligible imbalance (Austin 2009)."
    return render_records_table(title, records, style=style, note=note)


def render_competing_risks_table(
    title: str,
    cr_data: dict[str, Any] | pd.DataFrame,
    style: Literal["NEJM", "JAMA", "APA7"] = "NEJM",
) -> str:
    """
    Render dual Cause-Specific HR vs Fine-Gray SHR table as a publication-grade HTML table.
    """
    if hasattr(cr_data, "comparison_table"):
        df = cr_data.comparison_table
    elif isinstance(cr_data, dict) and "comparison_table" in cr_data:
        df = cr_data["comparison_table"]
    elif isinstance(cr_data, pd.DataFrame):
        df = cr_data
    else:
        raise ValueError("cr_data must contain comparison_table DataFrame.")

    records: list[dict[str, Any]] = []
    for _, row in df.iterrows():
        cov = str(row["covariate"])
        cs_ci = (
            row.get("cs_95_ci")
            or f"{row['cause_specific_hr']:.3f} ({row['cs_ci_lower']:.3f} to {row['cs_ci_upper']:.3f})"
        )
        cs_p = format_journal_p_value(row["cs_p_value"], style=style)
        fg_ci = (
            row.get("fg_95_ci")
            or f"{row['fine_gray_shr']:.3f} ({row['fg_ci_lower']:.3f} to {row['fg_ci_upper']:.3f})"
        )
        fg_p = format_journal_p_value(row["fg_p_value"], style=style)

        divergent = row.get("divergent", False)
        status_note = "Divergent (Etiology ≠ Prognosis)" if divergent else "Concordant"

        records.append(
            {
                "Covariate": cov,
                "Cause-Specific HR (95% CI)": cs_ci,
                "csHR P-value": cs_p,
                "Fine-Gray SHR (95% CI)": fg_ci,
                "SHR P-value": fg_p,
                "Association Profile": status_note,
            }
        )

    note = (
        "Cause-specific hazard ratios (csHR) estimate relative event rates among patients currently event-free (etiology). "
        "Fine-Gray subdistribution hazard ratios (SHR) estimate relative effects on cumulative incidence over time (clinical prognosis)."
    )
    return render_records_table(title, records, style=style, note=note)


def format_reclassification_table(
    reclass_data: dict[str, Any],
    title: str = "Table. Risk Reclassification and Discrimination Metrics",
    style: Literal["NEJM", "JAMA", "APA7"] = "NEJM",
) -> str:
    """
    Renders publication-grade NEJM/JAMA model reclassification table with:
    - Model AUROCs and paired DeLong ΔAUROC
    - Continuous Net Reclassification Improvement (NRI_total, NRI_events, NRI_nonevents)
    - Categorical NRI across prespecified risk strata
    - Integrated Discrimination Improvement (IDI) and Relative IDI
    - Reclassification contingency matrix sub-table
    """
    if "reclassification" in reclass_data:
        reclass = reclass_data["reclassification"]
        delong = reclass_data.get("delong")
        old_m = reclass_data.get("model_old")
        new_m = reclass_data.get("model_new")
    else:
        reclass = reclass_data
        delong = None
        old_m = None
        new_m = None

    records: list[dict[str, Any]] = []

    if old_m:
        ci_l = old_m.get("ci_lower")
        ci_u = old_m.get("ci_upper")
        ci_str = (
            f"{old_m['auc']:.3f} ({ci_l:.3f} to {ci_u:.3f})"
            if ci_l is not None and ci_u is not None
            else f"{old_m['auc']:.3f}"
        )
        records.append(
            {
                "Model / Metric": "Baseline Model AUROC",
                "Estimate (95% CI)": ci_str,
                "P-Value": "—",
            }
        )

    if new_m:
        ci_l = new_m.get("ci_lower")
        ci_u = new_m.get("ci_upper")
        ci_str = (
            f"{new_m['auc']:.3f} ({ci_l:.3f} to {ci_u:.3f})"
            if ci_l is not None and ci_u is not None
            else f"{new_m['auc']:.3f}"
        )
        records.append(
            {
                "Model / Metric": "Updated Model AUROC",
                "Estimate (95% CI)": ci_str,
                "P-Value": "—",
            }
        )

    if delong:
        records.append(
            {
                "Model / Metric": "ΔAUROC (Paired DeLong)",
                "Estimate (95% CI)": f"{delong['difference']:.3f} ({delong['ci_lower']:.3f} to {delong['ci_upper']:.3f})",
                "P-Value": format_journal_p_value(delong["p_value"], style=style),
            }
        )

    # Continuous NRI
    nri_c = reclass.get("nri_continuous", {})
    if nri_c:
        tot = nri_c.get("total", {})
        ev = nri_c.get("events", {})
        nev = nri_c.get("nonevents", {})
        if tot:
            records.append(
                {
                    "Model / Metric": "Continuous NRI (Total)",
                    "Estimate (95% CI)": f"{tot['estimate']:.3f} ({tot['ci_lower']:.3f} to {tot['ci_upper']:.3f})",
                    "P-Value": format_journal_p_value(tot["p_value"], style=style),
                }
            )
        if ev:
            records.append(
                {
                    "Model / Metric": "  Event NRI (P[up] – P[down])",
                    "Estimate (95% CI)": f"{ev['estimate']:.3f} ({ev['ci_lower']:.3f} to {ev['ci_upper']:.3f})",
                    "P-Value": format_journal_p_value(ev["p_value"], style=style),
                }
            )
        if nev:
            records.append(
                {
                    "Model / Metric": "  Non-event NRI (P[down] – P[up])",
                    "Estimate (95% CI)": f"{nev['estimate']:.3f} ({nev['ci_lower']:.3f} to {nev['ci_upper']:.3f})",
                    "P-Value": format_journal_p_value(nev["p_value"], style=style),
                }
            )

    # Categorical NRI
    nri_cat = reclass.get("nri_categorical")
    if nri_cat:
        tot_cat = nri_cat.get("total", {})
        ev_cat = nri_cat.get("events", {})
        nev_cat = nri_cat.get("nonevents", {})
        th_str = ", ".join(f"{t:g}" for t in nri_cat.get("thresholds", []))
        if tot_cat:
            records.append(
                {
                    "Model / Metric": f"Categorical NRI ({th_str})",
                    "Estimate (95% CI)": f"{tot_cat['estimate']:.3f} ({tot_cat['ci_lower']:.3f} to {tot_cat['ci_upper']:.3f})",
                    "P-Value": format_journal_p_value(tot_cat["p_value"], style=style),
                }
            )
        if ev_cat:
            records.append(
                {
                    "Model / Metric": "  Event Categorical NRI",
                    "Estimate (95% CI)": f"{ev_cat['estimate']:.3f} ({ev_cat['ci_lower']:.3f} to {ev_cat['ci_upper']:.3f})",
                    "P-Value": format_journal_p_value(ev_cat["p_value"], style=style),
                }
            )
        if nev_cat:
            records.append(
                {
                    "Model / Metric": "  Non-event Categorical NRI",
                    "Estimate (95% CI)": f"{nev_cat['estimate']:.3f} ({nev_cat['ci_lower']:.3f} to {nev_cat['ci_upper']:.3f})",
                    "P-Value": format_journal_p_value(nev_cat["p_value"], style=style),
                }
            )

    # IDI
    idi_d = reclass.get("idi", {})
    if idi_d:
        records.append(
            {
                "Model / Metric": "Integrated Discrimination Improvement (IDI)",
                "Estimate (95% CI)": f"{idi_d['estimate']:.3f} ({idi_d['ci_lower']:.3f} to {idi_d['ci_upper']:.3f})",
                "P-Value": format_journal_p_value(idi_d["p_value"], style=style),
            }
        )
        if idi_d.get("relative_idi") is not None and not math.isnan(
            idi_d["relative_idi"]
        ):
            records.append(
                {
                    "Model / Metric": "  Relative IDI",
                    "Estimate (95% CI)": f"{idi_d['relative_idi'] * 100.0:.1f}%",
                    "P-Value": "—",
                }
            )

    note = (
        "Net Reclassification Improvement (NRI) and Integrated Discrimination Improvement (IDI) "
        "evaluate risk reclassification per Pencina et al. (2008, 2011). Event and Non-event components are reported "
        "separately per Kerr et al. (2014) guidelines."
    )
    main_html = render_records_table(title, records, style=style, note=note)

    # Render companion contingency matrix if categorical NRI present
    if nri_cat and "contingency_matrices" in nri_cat:
        cats = nri_cat.get("categories", [])
        m_comb = nri_cat["contingency_matrices"].get("combined", [])
        m_ev = nri_cat["contingency_matrices"].get("events", [])
        m_nev = nri_cat["contingency_matrices"].get("nonevents", [])

        matrix_records = []
        for r_idx, cat_name in enumerate(cats):
            row_dict = {"Baseline Category": cat_name}
            for c_idx, c_name in enumerate(cats):
                comb_val = (
                    m_comb[r_idx][c_idx]
                    if r_idx < len(m_comb) and c_idx < len(m_comb[r_idx])
                    else 0
                )
                ev_val = (
                    m_ev[r_idx][c_idx]
                    if r_idx < len(m_ev) and c_idx < len(m_ev[r_idx])
                    else 0
                )
                nev_val = (
                    m_nev[r_idx][c_idx]
                    if r_idx < len(m_nev) and c_idx < len(m_nev[r_idx])
                    else 0
                )
                row_dict[f"Updated: {c_name}"] = (
                    f"{comb_val} (E: {ev_val}, NE: {nev_val})"
                )
            matrix_records.append(row_dict)

        matrix_title = f"{title} — Reclassification Contingency Matrix (Combined [E: Events, NE: Non-Events])"
        matrix_note = "Rows represent baseline risk categories; columns represent updated model risk categories."
        matrix_html = render_records_table(
            matrix_title, matrix_records, style=style, note=matrix_note
        )
        return f"{main_html}<br>\n{matrix_html}"

    return main_html


def format_time_dependent_roc_table(
    td_roc_data: dict[str, Any],
    title: str = "Table. Time-Dependent ROC Performance Across Clinical Follow-up Horizons",
    style: Literal["NEJM", "JAMA", "APA7"] = "NEJM",
) -> str:
    """
    Renders publication-grade Time-Dependent ROC table across follow-up horizons t.
    Reports Horizon t, N Cases, N Controls, Censored, AUROC(t) (95% CI), Optimal Threshold,
    Sensitivity, and Specificity.
    """
    horizons = td_roc_data.get("time_horizons", [])
    results_by_horizon = td_roc_data.get("results_by_horizon", {})
    comp_dict = td_roc_data.get("comparison", {})

    records: list[dict[str, Any]] = []

    for t in horizons:
        res_h = results_by_horizon.get(t, results_by_horizon.get(str(t), {}))
        if not res_h:
            continue

        auc_val = res_h.get("auroc")
        ci_l = res_h.get("ci_lower")
        ci_u = res_h.get("ci_upper")

        if (
            auc_val is not None
            and not math.isnan(auc_val)
            and ci_l is not None
            and not math.isnan(ci_l)
            and ci_u is not None
            and not math.isnan(ci_u)
        ):
            auc_str = f"{auc_val:.3f} ({ci_l:.3f} to {ci_u:.3f})"
        elif auc_val is not None and not math.isnan(auc_val):
            auc_str = f"{auc_val:.3f} (CI unavail.)"
        else:
            auc_str = "—"

        opt_cut = res_h.get("optimal_threshold")
        opt_str = (
            f"{opt_cut:.3f}" if opt_cut is not None and math.isfinite(opt_cut) else "—"
        )
        sens = res_h.get("sensitivity")
        sens_str = (
            f"{sens * 100.0:.1f}%" if sens is not None and math.isfinite(sens) else "—"
        )
        spec = res_h.get("specificity")
        spec_str = (
            f"{spec * 100.0:.1f}%" if spec is not None and math.isfinite(spec) else "—"
        )

        rec = {
            "Horizon (t)": f"{t:g}",
            "Cases (N)": str(res_h.get("n_cases", 0)),
            "Controls (N)": str(res_h.get("n_controls", 0)),
            "Censored (N)": str(res_h.get("n_censored", 0)),
            "AUROC(t) (95% CI)": auc_str,
            "Optimal Cutoff": opt_str,
            "Sensitivity": sens_str,
            "Specificity": spec_str,
        }

        if comp_dict and t in comp_dict:
            comp_h = comp_dict[t]
            delta = comp_h.get("delta_auroc")
            p_val = comp_h.get("p_value")
            rec["ΔAUROC(t)"] = (
                f"{delta:.3f}" if delta is not None and not math.isnan(delta) else "—"
            )
            rec["Comparison P-Value"] = (
                format_journal_p_value(p_val, style=style)
                if p_val is not None and not math.isnan(p_val)
                else "—"
            )

        records.append(rec)

    ci_method = td_roc_data.get("ci_method")
    if ci_method == "bootstrap":
        ci_str = "Confidence intervals (95%) calculated via bootstrap resampling."
    elif ci_method == "hanley_mcneil":
        ci_str = "Confidence intervals (95%) calculated via unweighted Hanley-McNeil approximation."
    else:
        ci_str = "Confidence intervals (95%) calculated where risk sets allow."

    note = (
        "Time-dependent receiver operating characteristic curves and AUROC(t) estimated using the "
        "cumulative/dynamic inverse probability of censoring weighting (IPCW) estimator (Uno et al., 2007; Heagerty et al., 2000). "
        f"{ci_str}"
    )
    return render_records_table(title, records, style=style, note=note)
