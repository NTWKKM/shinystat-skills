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


def get_checklist(name: str) -> ReportingChecklist:
    """Retrieve checklist by name (strobe, consort, tripod)."""
    key = name.lower().strip()
    if "strobe" in key:
        return get_strobe_checklist()
    elif "consort" in key:
        return get_consort_checklist()
    elif "tripod" in key:
        return get_tripod_checklist()
    else:
        raise ValueError(
            f"Unknown reporting checklist: '{name}'. Choose 'strobe', 'consort', or 'tripod'."
        )
