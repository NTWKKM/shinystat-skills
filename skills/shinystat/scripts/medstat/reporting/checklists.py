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
            "Identification as a study of diagnostic accuracy using at least one measure of accuracy",
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
            "Scientific and clinical background",
            "Scientific and clinical background, including the intended use and clinical role of the index test",
            "Introduction",
        ),
        ChecklistItem(
            "4",
            "Objectives and hypotheses",
            "Study objectives and prespecified hypotheses",
            "Introduction",
        ),
        ChecklistItem(
            "5",
            "Study design",
            "Whether data collection was planned before index test and reference standard were performed (prospective) or after (retrospective)",
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
            "Participant identification",
            "On what basis potentially eligible participants were identified (symptoms, prior test results, registry)",
            "Methods",
        ),
        ChecklistItem(
            "8",
            "Setting and location",
            "Where and when potentially eligible participants were identified (setting, location, and dates)",
            "Methods",
        ),
        ChecklistItem(
            "9",
            "Participant sampling",
            "Whether participants formed a consecutive, random, or convenience series",
            "Methods",
        ),
        ChecklistItem(
            "10a",
            "Index test details",
            "Index test, in sufficient detail to allow replication",
            "Methods",
        ),
        ChecklistItem(
            "10b",
            "Reference standard details",
            "Reference standard, in sufficient detail to allow replication",
            "Methods",
        ),
        ChecklistItem(
            "11",
            "Reference standard rationale",
            "Rationale for choosing the reference standard (if alternatives exist)",
            "Methods",
        ),
        ChecklistItem(
            "12a",
            "Index test cut-offs",
            "Definition of and rationale for test positivity cut-offs or result categories of the index test, distinguishing prespecified from exploratory",
            "Methods",
        ),
        ChecklistItem(
            "12b",
            "Reference standard cut-offs",
            "Definition of and rationale for test positivity cut-offs or result categories of the reference standard, distinguishing prespecified from exploratory",
            "Methods",
        ),
        ChecklistItem(
            "13a",
            "Blinding of index test readers",
            "Whether clinical information and reference standard results were available to performers/readers of index test",
            "Methods",
        ),
        ChecklistItem(
            "13b",
            "Blinding of reference standard assessors",
            "Whether clinical information and index test results were available to assessors of reference standard",
            "Methods",
        ),
        ChecklistItem(
            "14",
            "Statistical methods",
            "Methods for estimating or comparing measures of diagnostic accuracy",
            "Methods",
        ),
        ChecklistItem(
            "15",
            "Indeterminate results handling",
            "How indeterminate index test or reference standard results were handled",
            "Methods",
        ),
        ChecklistItem(
            "16",
            "Missing data handling",
            "How missing data on the index test and reference standard were handled",
            "Methods",
        ),
        ChecklistItem(
            "17",
            "Variability analyses",
            "Any analyses of variability in diagnostic accuracy, distinguishing prespecified from exploratory",
            "Methods",
        ),
        ChecklistItem(
            "18",
            "Sample size determination",
            "Intended sample size and how it was determined",
            "Methods",
        ),
        ChecklistItem(
            "19",
            "Participant flow",
            "Flow of participants, using a diagram",
            "Results",
        ),
        ChecklistItem(
            "20",
            "Baseline characteristics",
            "Baseline demographic and clinical characteristics of participants",
            "Results",
        ),
        ChecklistItem(
            "21a",
            "Disease severity distribution",
            "Distribution of severity of disease in those with the target condition",
            "Results",
        ),
        ChecklistItem(
            "21b",
            "Alternative diagnoses distribution",
            "Distribution of alternative diagnoses in those without the target condition",
            "Results",
        ),
        ChecklistItem(
            "22",
            "Time interval and interventions",
            "Time interval and any clinical interventions between index test and reference standard",
            "Results",
        ),
        ChecklistItem(
            "23",
            "Cross-tabulation",
            "Cross tabulation of index test results (or their distribution) by results of reference standard",
            "Results",
        ),
        ChecklistItem(
            "24",
            "Diagnostic accuracy estimates",
            "Estimates of diagnostic accuracy and their precision (such as 95% confidence intervals)",
            "Results",
        ),
        ChecklistItem(
            "25",
            "Adverse events",
            "Any adverse events from performing the index test or the reference standard",
            "Results",
        ),
        ChecklistItem(
            "26",
            "Study limitations",
            "Study limitations, including sources of potential bias, statistical uncertainty, and generalisability",
            "Discussion",
        ),
        ChecklistItem(
            "27",
            "Implications for practice",
            "Implications for practice, including the intended use and clinical role of the index test",
            "Discussion",
        ),
        ChecklistItem(
            "28",
            "Study registration",
            "Registration number and name of registry",
            "Other information",
        ),
        ChecklistItem(
            "29",
            "Protocol access",
            "Where the full study protocol can be accessed",
            "Other information",
        ),
        ChecklistItem(
            "30",
            "Funding and support",
            "Sources of funding and other support; role of funders",
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
            "Identify the report as a systematic review",
            "Title and Abstract",
        ),
        ChecklistItem(
            "2",
            "Abstract",
            "Structured summary (background, objectives, eligibility, methods, results, limitations, conclusions)",
            "Title and Abstract",
        ),
        ChecklistItem(
            "3",
            "Rationale",
            "Describe the rationale for the review in the context of existing knowledge",
            "Introduction",
        ),
        ChecklistItem(
            "4",
            "Objectives",
            "Provide an explicit statement of the objective(s) or question(s) the review addresses",
            "Introduction",
        ),
        ChecklistItem(
            "5",
            "Eligibility criteria",
            "Specify inclusion and exclusion criteria and how studies were grouped for syntheses",
            "Methods",
        ),
        ChecklistItem(
            "6",
            "Information sources",
            "Specify all databases, registers, websites, and other sources searched, and dates last searched",
            "Methods",
        ),
        ChecklistItem(
            "7",
            "Search strategy",
            "Present full electronic search strategy for at least one database, including filters and limits",
            "Methods",
        ),
        ChecklistItem(
            "8",
            "Selection process",
            "Methods used to decide which studies were eligible, including reviewers and automation tools",
            "Methods",
        ),
        ChecklistItem(
            "9",
            "Data collection process",
            "Methods used to extract data from included reports, including independent reviewers and confirmation",
            "Methods",
        ),
        ChecklistItem(
            "10a",
            "Data items - outcomes",
            "List and define all outcomes for which data were sought, and compatible outcome domains",
            "Methods",
        ),
        ChecklistItem(
            "10b",
            "Data items - other variables",
            "List and define all other variables sought (e.g. participant/intervention features, funding, assumptions)",
            "Methods",
        ),
        ChecklistItem(
            "11",
            "Study risk of bias assessment",
            "Methods used to assess risk of bias in included studies (e.g. RoB 2, ROBINS-I)",
            "Methods",
        ),
        ChecklistItem(
            "12",
            "Effect measures",
            "Specify for each outcome the effect measure(s) used in synthesis (OR, RR, HR, MD, SMD)",
            "Methods",
        ),
        ChecklistItem(
            "13a",
            "Synthesis methods - eligibility",
            "Describe processes used to decide which studies were eligible for each synthesis",
            "Methods",
        ),
        ChecklistItem(
            "13b",
            "Synthesis methods - preparation",
            "Describe methods required to prepare data for synthesis (missing statistics, data conversions)",
            "Methods",
        ),
        ChecklistItem(
            "13c",
            "Synthesis methods - tabulation and display",
            "Describe methods used to tabulate or visually display results of individual studies and syntheses",
            "Methods",
        ),
        ChecklistItem(
            "13d",
            "Synthesis methods - statistical synthesis",
            "Describe statistical synthesis methods, models, heterogeneity assessment (tau2, I2), and software",
            "Methods",
        ),
        ChecklistItem(
            "13e",
            "Synthesis methods - heterogeneity",
            "Describe methods used to explore possible causes of heterogeneity (subgroup analysis, meta-regression)",
            "Methods",
        ),
        ChecklistItem(
            "13f",
            "Synthesis methods - sensitivity analyses",
            "Describe sensitivity analyses conducted to assess robustness of synthesized results",
            "Methods",
        ),
        ChecklistItem(
            "14",
            "Reporting bias assessment",
            "Methods used to assess risk of bias due to missing results / reporting biases (Egger's test, funnel plots)",
            "Methods",
        ),
        ChecklistItem(
            "15",
            "Certainty assessment",
            "Methods used to assess certainty (or confidence) in the body of evidence for an outcome (e.g. GRADE)",
            "Methods",
        ),
        ChecklistItem(
            "16a",
            "Study selection - results",
            "Results of search and selection process (records screened, eligible, included, PRISMA flow diagram)",
            "Results",
        ),
        ChecklistItem(
            "16b",
            "Study selection - excluded studies",
            "Cite studies that might appear to meet the inclusion criteria, but which were excluded, and explain why they were excluded",
            "Results",
        ),
        ChecklistItem(
            "17",
            "Study characteristics",
            "Cite each included study and present its demographic and clinical characteristics",
            "Results",
        ),
        ChecklistItem(
            "18",
            "Risk of bias in studies",
            "Present assessments of risk of bias for each included study",
            "Results",
        ),
        ChecklistItem(
            "19",
            "Results of individual studies",
            "Summary statistics for each group and effect estimates with precision for each included study",
            "Results",
        ),
        ChecklistItem(
            "20a",
            "Results of syntheses - characteristics",
            "Briefly summarise characteristics and risk of bias among studies contributing to each synthesis",
            "Results",
        ),
        ChecklistItem(
            "20b",
            "Results of syntheses - statistical results",
            "Present results of all statistical syntheses conducted, pooled estimates, precision, and heterogeneity",
            "Results",
        ),
        ChecklistItem(
            "20c",
            "Results of syntheses - heterogeneity investigations",
            "Present results of all investigations of possible causes of statistical heterogeneity",
            "Results",
        ),
        ChecklistItem(
            "20d",
            "Results of syntheses - sensitivity analyses",
            "Present results of all sensitivity analyses conducted to assess robustness of synthesized results",
            "Results",
        ),
        ChecklistItem(
            "21",
            "Reporting biases",
            "Assessments of risk of bias due to missing results (Egger's test, funnel plots) for each synthesis",
            "Results",
        ),
        ChecklistItem(
            "22",
            "Certainty of evidence",
            "Assessments of certainty in body of evidence for each primary clinical outcome (GRADE)",
            "Results",
        ),
        ChecklistItem(
            "23a",
            "Discussion - interpretation",
            "General interpretation of findings in context of other evidence",
            "Discussion",
        ),
        ChecklistItem(
            "23b",
            "Discussion - limitations of evidence",
            "Discuss any limitations of the evidence included in the review",
            "Discussion",
        ),
        ChecklistItem(
            "23c",
            "Discussion - limitations of review",
            "Discuss any limitations of the review processes used",
            "Discussion",
        ),
        ChecklistItem(
            "23d",
            "Discussion - implications",
            "Discuss implications of results for practice, policy, and future research",
            "Discussion",
        ),
        ChecklistItem(
            "24a",
            "Registration and protocol - registration",
            "Provide registration information for review, including register name (e.g. PROSPERO) and registration number",
            "Other information",
        ),
        ChecklistItem(
            "24b",
            "Registration and protocol - protocol",
            "Indicate where the review protocol can be accessed, or state that a protocol was not prepared",
            "Other information",
        ),
        ChecklistItem(
            "24c",
            "Registration and protocol - amendments",
            "Describe and explain any amendments to information provided at registration or in protocol",
            "Other information",
        ),
        ChecklistItem(
            "25",
            "Support",
            "Sources of financial and non-financial support, and role of funders",
            "Other information",
        ),
        ChecklistItem(
            "26",
            "Competing interests",
            "Financial or non-financial competing interests of review authors",
            "Other information",
        ),
        ChecklistItem(
            "27",
            "Availability of data and code",
            "Report which data, code, and other materials are publicly available and where",
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
