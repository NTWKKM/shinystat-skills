"""
Reporting Checklists Module (STROBE, CONSORT, TRIPOD, PRISMA).

Provides structured audit checklists for medical research publications.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class ChecklistStatus(Enum):
    NOT_APPLICABLE = "N/A"
    NOT_DONE = "Not addressed"
    PARTIAL = "Partially addressed"
    COMPLETE = "Complete"


@dataclass
class ChecklistItem:
    number: str
    item: str
    description: str
    section: str
    status: ChecklistStatus = ChecklistStatus.NOT_DONE
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "number": self.number,
            "item": self.item,
            "description": self.description,
            "section": self.section,
            "status": self.status.value,
            "notes": self.notes,
        }


@dataclass
class ReportingChecklist:
    name: str
    guideline: str
    items: list[ChecklistItem] = field(default_factory=list)

    def to_markdown(self) -> str:
        lines = [
            f"# {self.name} Compliance Checklist ({self.guideline})",
            "",
            "| Item | Section | Topic | Status | Notes |",
            "| :--- | :--- | :--- | :--- | :--- |",
        ]
        for item in self.items:
            lines.append(
                f"| {item.number} | {item.section} | {item.item} — {item.description} | {item.status.value} | {item.notes} |"
            )
        lines.append("")
        return "\n".join(lines)


def get_strobe_checklist() -> ReportingChecklist:
    items = [
        ChecklistItem(
            "1a",
            "Title",
            "Indicate study design with a commonly used term in title",
            "Title and Abstract",
        ),
        ChecklistItem(
            "1b",
            "Abstract",
            "Provide informative and balanced summary of what was done and found",
            "Title and Abstract",
        ),
        ChecklistItem(
            "2",
            "Background",
            "Explain scientific background and rationale for investigation",
            "Introduction",
        ),
        ChecklistItem(
            "3",
            "Objectives",
            "State specific objectives, including any prespecified hypotheses",
            "Introduction",
        ),
        ChecklistItem(
            "4",
            "Study design",
            "Present key elements of study design early in paper",
            "Methods",
        ),
        ChecklistItem(
            "5",
            "Setting",
            "Describe setting, locations, relevant dates, recruitment periods",
            "Methods",
        ),
        ChecklistItem(
            "6",
            "Participants",
            "Give eligibility criteria, sources and methods of participant selection",
            "Methods",
        ),
        ChecklistItem(
            "7",
            "Variables",
            "Clearly define all outcomes, exposures, predictors, potential confounders",
            "Methods",
        ),
        ChecklistItem(
            "8",
            "Data sources",
            "For each variable of interest, give sources of data and measurement methods",
            "Methods",
        ),
        ChecklistItem(
            "9",
            "Bias",
            "Describe any efforts to address potential sources of bias",
            "Methods",
        ),
        ChecklistItem(
            "10", "Study size", "Explain how study size was arrived at", "Methods"
        ),
        ChecklistItem(
            "11",
            "Quantitative variables",
            "Explain how quantitative variables were handled in analyses",
            "Methods",
        ),
        ChecklistItem(
            "12",
            "Statistical methods",
            "Describe all statistical methods, including confounder adjustment and missing data",
            "Methods",
        ),
        ChecklistItem(
            "13",
            "Participants flow",
            "Report numbers of individuals at each stage of study",
            "Results",
        ),
        ChecklistItem(
            "14",
            "Descriptive data",
            "Give characteristics of study participants (e.g. Table 1)",
            "Results",
        ),
        ChecklistItem(
            "15",
            "Outcome data",
            "Report numbers of outcome events or summary measures",
            "Results",
        ),
        ChecklistItem(
            "16",
            "Main results",
            "Give unadjusted and confounder-adjusted estimates and 95% CIs",
            "Results",
        ),
        ChecklistItem(
            "17",
            "Other analyses",
            "Report other analyses done (subgroups, sensitivity, E-values)",
            "Results",
        ),
        ChecklistItem(
            "18",
            "Key results",
            "Summarize key results with reference to study objectives",
            "Discussion",
        ),
        ChecklistItem(
            "19",
            "Limitations",
            "Discuss limitations of study, taking into account sources of bias",
            "Discussion",
        ),
        ChecklistItem(
            "20",
            "Interpretation",
            "Give cautious overall interpretation of results considering objectives",
            "Discussion",
        ),
        ChecklistItem(
            "21",
            "Generalisability",
            "Discuss external validity (generalisability) of study findings",
            "Discussion",
        ),
        ChecklistItem(
            "22",
            "Funding",
            "Give source of funding and role of funders",
            "Other Information",
        ),
    ]
    return ReportingChecklist("STROBE", "Observational Studies in Epidemiology", items)


def get_consort_checklist() -> ReportingChecklist:
    items = [
        ChecklistItem(
            "1a",
            "Title",
            "Identification as randomized trial in title",
            "Title and Abstract",
        ),
        ChecklistItem(
            "1b",
            "Abstract",
            "Structured summary of trial design, methods, results, conclusions",
            "Title and Abstract",
        ),
        ChecklistItem(
            "2a",
            "Background",
            "Scientific background and explanation of rationale",
            "Introduction",
        ),
        ChecklistItem(
            "2b", "Objectives", "Specific objectives or hypotheses", "Introduction"
        ),
        ChecklistItem(
            "3a",
            "Trial design",
            "Description of trial design (parallel, factorial) and allocation ratio",
            "Methods",
        ),
        ChecklistItem(
            "4a", "Participants", "Eligibility criteria for participants", "Methods"
        ),
        ChecklistItem(
            "5",
            "Interventions",
            "Interventions for each group with specific details",
            "Methods",
        ),
        ChecklistItem(
            "6a",
            "Outcomes",
            "Completely defined prespecified primary and secondary outcome measures",
            "Methods",
        ),
        ChecklistItem("7a", "Sample size", "How sample size was determined", "Methods"),
        ChecklistItem(
            "8a",
            "Sequence generation",
            "Method used to generate random allocation sequence",
            "Methods",
        ),
        ChecklistItem(
            "9",
            "Allocation concealment",
            "Mechanism used to implement random allocation sequence",
            "Methods",
        ),
        ChecklistItem(
            "10",
            "Implementation",
            "Who generated allocation sequence, enrolled, and assigned participants",
            "Methods",
        ),
        ChecklistItem(
            "11a",
            "Blinding",
            "If done, who was blinded after assignment to interventions",
            "Methods",
        ),
        ChecklistItem(
            "12a",
            "Statistical methods",
            "Statistical methods used to compare groups for primary outcomes",
            "Methods",
        ),
        ChecklistItem(
            "13a",
            "Participant flow",
            "Flow diagram of participants through each stage (CONSORT flow)",
            "Results",
        ),
        ChecklistItem(
            "14a",
            "Recruitment",
            "Dates defining periods of recruitment and follow-up",
            "Results",
        ),
        ChecklistItem(
            "15",
            "Baseline data",
            "Table showing baseline demographic and clinical characteristics",
            "Results",
        ),
        ChecklistItem(
            "16",
            "Numbers analyzed",
            "Number of participants included in each analysis (intention-to-treat)",
            "Results",
        ),
        ChecklistItem(
            "17a",
            "Outcomes and estimation",
            "For each primary outcome, results for each group, effect size, and 95% CI",
            "Results",
        ),
        ChecklistItem(
            "18",
            "Ancillary analyses",
            "Results of any other analyses, such as subgroup and adjusted analyses",
            "Results",
        ),
        ChecklistItem(
            "19",
            "Harms",
            "All important harms or unintended effects in each group",
            "Results",
        ),
        ChecklistItem(
            "20",
            "Limitations",
            "Trial limitations, addressing sources of potential bias and imprecision",
            "Discussion",
        ),
        ChecklistItem(
            "21",
            "Generalisability",
            "Generalisability (external validity) of the trial findings",
            "Discussion",
        ),
        ChecklistItem(
            "22",
            "Interpretation",
            "Interpretation consistent with results, balancing benefits and harms",
            "Discussion",
        ),
    ]
    return ReportingChecklist(
        "CONSORT", "Consolidated Standards of Reporting Trials", items
    )


def get_tripod_checklist() -> ReportingChecklist:
    items = [
        ChecklistItem(
            "1",
            "Title",
            "Identify study as developing and/or validating a multivariable prediction model",
            "Title and Abstract",
        ),
        ChecklistItem(
            "2",
            "Abstract",
            "Structured summary of objectives, study design, setting, participants, model performance",
            "Title and Abstract",
        ),
        ChecklistItem(
            "3a",
            "Background",
            "Explain medical context and rationale for prediction model",
            "Introduction",
        ),
        ChecklistItem(
            "3b",
            "Objectives",
            "Specify objectives (development, validation, or both)",
            "Introduction",
        ),
        ChecklistItem(
            "4a",
            "Source of data",
            "Describe source of data (prospective cohort, registry, trial)",
            "Methods",
        ),
        ChecklistItem(
            "5a",
            "Participants",
            "Eligibility criteria for study participants",
            "Methods",
        ),
        ChecklistItem(
            "6a",
            "Outcome",
            "Clearly define the outcome predicted and how it was assessed",
            "Methods",
        ),
        ChecklistItem(
            "7a",
            "Predictors",
            "Clearly define all candidate predictors and blind assessment",
            "Methods",
        ),
        ChecklistItem(
            "8",
            "Sample size",
            "Explain how study sample size was determined (events per variable)",
            "Methods",
        ),
        ChecklistItem(
            "9",
            "Missing data",
            "Describe how missing data were handled (complete-case, MICE)",
            "Methods",
        ),
        ChecklistItem(
            "10a",
            "Statistical analysis",
            "Describe how predictors were handled (splines, transformations)",
            "Methods",
        ),
        ChecklistItem(
            "10b",
            "Model development",
            "Specify type of model, predictor selection procedures, shrinkage",
            "Methods",
        ),
        ChecklistItem(
            "10c",
            "Model performance",
            "State performance measures: discrimination (AUC/ROC) and calibration (ICI, slope)",
            "Methods",
        ),
        ChecklistItem(
            "10d",
            "Clinical utility",
            "Report Decision Curve Analysis (DCA) net benefit across threshold probabilities",
            "Methods",
        ),
        ChecklistItem(
            "13a",
            "Participant flow",
            "Flow of participants through study (N_init -> N_excl -> N_anal)",
            "Results",
        ),
        ChecklistItem(
            "14a",
            "Participant characteristics",
            "Baseline characteristics of study population (Table 1)",
            "Results",
        ),
        ChecklistItem(
            "15a",
            "Model specification",
            "Present full prediction model (coefficients, intercept, baseline hazard)",
            "Results",
        ),
        ChecklistItem(
            "16",
            "Model performance",
            "Report discrimination (AUC with DeLong 95% CIs) and calibration",
            "Results",
        ),
        ChecklistItem(
            "17",
            "Decision curve analysis",
            "Net benefit curves across clinical threshold probabilities",
            "Results",
        ),
        ChecklistItem(
            "18",
            "Limitations",
            "Discuss study limitations and potential sources of bias",
            "Discussion",
        ),
        ChecklistItem(
            "19a",
            "Interpretation",
            "Interpretation of results in context of existing prediction models",
            "Discussion",
        ),
        ChecklistItem(
            "20",
            "Implications",
            "Discuss potential clinical use and practical implications",
            "Discussion",
        ),
    ]
    return ReportingChecklist(
        "TRIPOD", "Transparent Reporting of multivariable prediction models", items
    )


def get_stard_checklist() -> ReportingChecklist:
    """Standards for Reporting Diagnostic Accuracy Studies (STARD 2015)."""
    items = [
        ChecklistItem(
            "1",
            "Title",
            "Identify study as a study of diagnostic accuracy using at least one measure of accuracy",
            "Title and Abstract",
        ),
        ChecklistItem(
            "2",
            "Abstract",
            "Structured summary of study design, methods, results, and conclusions",
            "Title and Abstract",
        ),
        ChecklistItem(
            "3",
            "Background",
            "Scientific and clinical background, including intended clinical role of index test",
            "Introduction",
        ),
        ChecklistItem(
            "4",
            "Objectives",
            "Study objectives and prespecified hypotheses",
            "Introduction",
        ),
        ChecklistItem(
            "5",
            "Protocol",
            "Whether a study protocol was prepared and where it can be accessed",
            "Methods",
        ),
        ChecklistItem(
            "6",
            "Eligibility criteria",
            "Eligibility criteria for participants",
            "Methods",
        ),
        ChecklistItem(
            "7",
            "Setting",
            "Where and when potentially eligible participants were identified",
            "Methods",
        ),
        ChecklistItem(
            "8",
            "Participant recruitment",
            "Whether participants formed a consecutive, random, or convenience series",
            "Methods",
        ),
        ChecklistItem(
            "9",
            "Participant sampling",
            "Whether data collection was prospective or retrospective",
            "Methods",
        ),
        ChecklistItem(
            "10a",
            "Index test",
            "Index test details, including how it was executed and interpreted",
            "Methods",
        ),
        ChecklistItem(
            "10b",
            "Cut-offs",
            "Prespecified or post-hoc cut-offs for index test positivity and test direction",
            "Methods",
        ),
        ChecklistItem(
            "11",
            "Reference standard",
            "Reference standard specifications, execution, and clinical rationale",
            "Methods",
        ),
        ChecklistItem(
            "12",
            "Blinding",
            "Blinding of index test readers to reference standard, and vice versa",
            "Methods",
        ),
        ChecklistItem(
            "13a",
            "Statistical methods",
            "Methods for estimating diagnostic accuracy (sensitivity, specificity, AUC) and 95% CIs",
            "Methods",
        ),
        ChecklistItem(
            "13b",
            "Indeterminate results",
            "How missing, indeterminate, or outlier results were handled",
            "Methods",
        ),
        ChecklistItem(
            "14",
            "Participant flow",
            "Flow of participants through study (CONSORT/STARD flow diagram)",
            "Results",
        ),
        ChecklistItem(
            "15",
            "Baseline characteristics",
            "Baseline demographic and clinical characteristics of study cohort",
            "Results",
        ),
        ChecklistItem(
            "16",
            "Disease severity",
            "Distribution of severity of disease or alternative diagnoses in cohort",
            "Results",
        ),
        ChecklistItem(
            "17",
            "Time interval",
            "Time interval and clinical interventions between index test and reference standard",
            "Results",
        ),
        ChecklistItem(
            "18",
            "Cross-tabulation",
            "Cross-tabulation of index test results (2x2 matrix) by reference standard",
            "Results",
        ),
        ChecklistItem(
            "19",
            "Diagnostic accuracy",
            "Estimates of sensitivity, specificity, PPV, NPV, likelihood ratios, AUC with DeLong CIs",
            "Results",
        ),
        ChecklistItem(
            "20",
            "Decision curve analysis",
            "Net benefit assessment across clinical decision thresholds",
            "Results",
        ),
        ChecklistItem(
            "21",
            "Adverse events",
            "Any adverse events from performing index test or reference standard",
            "Results",
        ),
        ChecklistItem(
            "22",
            "Limitations",
            "Study limitations, sources of potential bias, and statistical uncertainty",
            "Discussion",
        ),
        ChecklistItem(
            "23",
            "Implications",
            "Implications for clinical practice and patient health outcomes",
            "Discussion",
        ),
        ChecklistItem(
            "24",
            "Registration",
            "Registration number and name of trial registry",
            "Other information",
        ),
        ChecklistItem(
            "25",
            "Funding",
            "Sources of funding and role of funders",
            "Other information",
        ),
    ]
    return ReportingChecklist(
        "STARD", "Standards for Reporting Diagnostic Accuracy Studies", items
    )


def get_prisma_checklist() -> ReportingChecklist:
    """Preferred Reporting Items for Systematic Reviews and Meta-Analyses (PRISMA 2020)."""
    items = [
        ChecklistItem(
            "1",
            "Title",
            "Identify the report as a systematic review and/or meta-analysis",
            "Title and Abstract",
        ),
        ChecklistItem(
            "2",
            "Abstract",
            "Structured summary including background, methods, results, and conclusions",
            "Title and Abstract",
        ),
        ChecklistItem(
            "3",
            "Rationale",
            "Rationale for review in context of existing clinical evidence",
            "Introduction",
        ),
        ChecklistItem(
            "4",
            "Objectives",
            "Explicit statement of clinical questions addressed (PICO/PECO format)",
            "Introduction",
        ),
        ChecklistItem(
            "5",
            "Eligibility criteria",
            "Inclusion and exclusion criteria and study design characteristics",
            "Methods",
        ),
        ChecklistItem(
            "6",
            "Information sources",
            "All bibliographic databases, registers, and websites searched and dates",
            "Methods",
        ),
        ChecklistItem(
            "7",
            "Search strategy",
            "Full electronic search strategy for at least one database",
            "Methods",
        ),
        ChecklistItem(
            "8",
            "Selection process",
            "Methods used to screen and select studies, independent reviewers",
            "Methods",
        ),
        ChecklistItem(
            "9",
            "Data collection",
            "Methods used to extract data from included reports",
            "Methods",
        ),
        ChecklistItem(
            "10a",
            "Data items",
            "Outcomes sought, effect measures (RR, OR, HR, MD, SMD), definitions",
            "Methods",
        ),
        ChecklistItem(
            "10b",
            "Data assumptions",
            "Assumptions made regarding missing or unclear information",
            "Methods",
        ),
        ChecklistItem(
            "11",
            "Risk of bias",
            "Methods used to assess risk of bias in included studies (RoB 2, ROBINS-I)",
            "Methods",
        ),
        ChecklistItem(
            "12",
            "Effect measures",
            "Effect measures used in synthesis (OR, RR, HR, SMD)",
            "Methods",
        ),
        ChecklistItem(
            "13",
            "Synthesis methods",
            "Statistical synthesis methods, fixed/random effects, DerSimonian-Laird, tau2, I2",
            "Methods",
        ),
        ChecklistItem(
            "14",
            "Reporting bias",
            "Methods used to assess reporting/publication bias (Egger's test, funnel plots)",
            "Methods",
        ),
        ChecklistItem(
            "15",
            "Certainty assessment",
            "Methods used to assess certainty of evidence (GRADE framework)",
            "Methods",
        ),
        ChecklistItem(
            "16",
            "Study selection",
            "Numbers of studies screened, assessed for eligibility, and included (PRISMA flow)",
            "Results",
        ),
        ChecklistItem(
            "17",
            "Study characteristics",
            "Citations of included studies and summary table of characteristics",
            "Results",
        ),
        ChecklistItem(
            "18",
            "Risk of bias in studies",
            "Risk of bias assessments for each included study",
            "Results",
        ),
        ChecklistItem(
            "19",
            "Synthesis results",
            "Pooled estimates, forest plots, confidence intervals, heterogeneity I2 and tau2",
            "Results",
        ),
        ChecklistItem(
            "20",
            "Reporting biases",
            "Assessments of publication bias (Egger's regression intercept and p-value)",
            "Results",
        ),
        ChecklistItem(
            "21",
            "Certainty of evidence",
            "Assessments of certainty of evidence for each primary clinical outcome",
            "Results",
        ),
        ChecklistItem(
            "22",
            "Interpretation",
            "General interpretation of findings in context of existing evidence",
            "Discussion",
        ),
        ChecklistItem(
            "23",
            "Limitations",
            "Limitations of evidence included and review methodology",
            "Discussion",
        ),
        ChecklistItem(
            "24",
            "Implications",
            "Implications for clinical practice, policy, and future clinical trials",
            "Discussion",
        ),
        ChecklistItem(
            "25",
            "Registration",
            "Registration number and register name (PROSPERO), protocol access",
            "Other information",
        ),
        ChecklistItem(
            "26",
            "Support",
            "Sources of financial and non-financial support for the review",
            "Other information",
        ),
        ChecklistItem(
            "27",
            "Competing interests",
            "Financial or non-financial competing interests of review authors",
            "Other information",
        ),
    ]
    return ReportingChecklist(
        "PRISMA",
        "Preferred Reporting Items for Systematic Reviews and Meta-Analyses",
        items,
    )


def get_checklist(name: str) -> ReportingChecklist:
    """Retrieve checklist by name (strobe, consort, tripod, stard, prisma)."""
    key = name.lower().strip()
    if "strobe" in key:
        return get_strobe_checklist()
    elif "consort" in key:
        return get_consort_checklist()
    elif "tripod" in key:
        return get_tripod_checklist()
    elif "stard" in key:
        return get_stard_checklist()
    elif "prisma" in key:
        return get_prisma_checklist()
    else:
        raise ValueError(
            f"Unknown reporting checklist: '{name}'. Choose 'strobe', 'consort', 'tripod', 'stard', or 'prisma'."
        )
