# Medical Journal Table Styles & Reporting Guidelines

Reference guide for publication-ready table formatting, journal-specific typography, and reporting guideline compliance.

---

## 1. Target Journal Typography & Table Rules

### New England Journal of Medicine (NEJM)
- **Rules & Borders**:
  - Double horizontal border at table top (thick outer line, thin inner rule separating headings).
  - Single horizontal bottom border.
  - Zero vertical dividing rules.
- **Alignment & Typography**:
  - Variable labels left-aligned in stub column; categories indented by 2 non-breaking spaces.
  - Numbers aligned on the decimal point or centered in data columns.
  - Parenthetical confidence intervals: `1.45 (95% CI, 1.12 to 1.88)` or `1.45 (95% CI, 1.12–1.88)`.
- **P-Value Format**:
  - Expressed with capital italic $P$: $P = 0.04$, $P = 0.003$, $P < 0.001$.
  - Always includes leading zero before decimal point ($0.05$, not $.05$).
- **Footnote Notation**:
  - Standard sequence of superscript symbols: $*$, $\dagger$, $\ddagger$, $\S$, $\P$, $\#$, $**$, $\dagger\dagger$.

### Journal of the American Medical Association (JAMA)
- **Rules & Borders**:
  - Minimal horizontal rules: one above header, one beneath header, one at table footer.
  - Zero vertical rules.
- **Alignment & Typography**:
  - Header titles capitalized in sentence case or title case according to section.
  - Column spans used for grouped categories (e.g. `No. (%) [n = 250]`).
- **P-Value Format**:
  - Expressed with capital italic $P$: $P = .04$, $P = .008$, $P < .001$.
  - **No leading zero** before the decimal point because p-values cannot exceed 1.
  - Reported as $P > .99$ if greater than $.99$; rounded to 2 decimal places if $P \ge .01$; exact to 3 decimal places if $.001 \le P < .01$; reported as $P < .001$ otherwise.
  - **No asterisks** for statistical significance; exact p-values are stated directly in table cells.
- **Footnote Notation**:
  - Superscript lowercase letters ($^a$, $^b$, $^c$).

### American Psychological Association 7th Edition (APA 7)
- **Rules & Borders**:
  - Three horizontal lines: top border, bottom of header row, bottom border of table.
  - Zero vertical rules.
- **Alignment & Typography**:
  - Table number bold on its own line: **Table 1**.
  - Table title italicized in title case on the next line: *Baseline Demographic and Clinical Characteristics*.
  - Statistical symbols italicized: $N$, $n$, $M$, $SD$, $OR$, $HR$, $p$, $t$, $F$, $z$.
- **P-Value Format**:
  - Lowercase italic $p$ without leading zero: $p = .042$, $p < .001$.
- **Footnote Notation**:
  - Three tiers of notes:
    - *Note.* General table notes and abbreviations.
    - $^a$ Specific footnotes.
    - $*p < .05. \quad **p < .01. \quad ***p < .001.$ Probability notes.

---

## 2. Reporting Standards Checklists

### STROBE (Observational Cohort, Case-Control, Cross-Sectional)
Strengthening the Reporting of Observational Studies in Epidemiology:
- **Item 12 (Statistical Methods)**:
  - 12a: Describe all statistical methods, including those used to control for confounding.
  - 12b: Describe any methods used to examine subgroups and interactions.
  - 12c: Explain how missing data were addressed (e.g. complete-case, MICE, KNN with explicit justification).
  - 12d: *Cohort studies*: describe methods for addressing loss to follow-up. *Case-control studies*: describe matching criteria (caliper, ratio, replacement). *Cross-sectional studies*: if applicable, describe analytical methods taking account of sampling strategy.
  - 12e: Describe any sensitivity analyses (e.g. VanderWeele E-value, unmeasured confounding bounds).
- **Item 13 (Participants)**:
  - Report numbers of individuals at each stage: $N_{\text{screened}} \to N_{\text{eligible}} \to N_{\text{enrolled}} \to N_{\text{analyzed}}$.
  - Give reasons for non-participation at each stage.
- **Item 14 (Descriptive Data)**:
  - Characteristics of study participants (demographic, clinical, social) and information on exposures and potential confounders.
  - Indicate number of participants with missing data for each variable of interest.
- **Item 16 (Main Results)**:
  - Give unadjusted estimates and confounder-adjusted estimates with 95% confidence intervals.
  - Make clear which confounders were adjusted for and why they were included.

### CONSORT 2025 (Randomized Controlled Trials)
Consolidated Standards of Reporting Trials (2025 Statement):
- **Item 16 (Sample Size)**:
  - How sample size was determined (including assumptions and calculations).
- **Item 21b (Analysis Populations)**:
  - Define analysis populations and describe who is included in each analysis.
- **Item 22 (Participant Flow)**:
  - Flow diagram depicting enrollment, allocation, follow-up, and analysis.
- **Item 25 (Baseline Data)**:
  - Baseline demographic and clinical characteristics of each group.
- **Item 26 (Numbers Analyzed and Outcomes)**:
  - Number of participants included in each analysis with estimated effect sizes and precision (95% CI).

### TRIPOD+AI (Clinical Prediction Models & Diagnostic Machine Learning)
Transparent Reporting of a multivariable prediction model for Individual Prognosis Or Diagnosis + Artificial Intelligence (BMJ 2024;385:e078378; supersedes TRIPOD 2015):
- **Item 10 (Sample Size)**:
  - Report the number of participants (events and non-events) and how sample size was determined.
- **Item 11 (Missing Data)**:
  - Details on handling missing data; describe imputation method if used.
- **Item 12a (Data Use & Partitioning)**:
  - Describe how data were used in the analysis (e.g., for development and evaluation of model performance), including whether the data were partitioned, any sample-size sufficiency considerations, and measures to prevent data leakage.
- **Item 12b (Predictor Handling)**:
  - Describe how predictors were handled in the analyses (functional form, rescaling, transformation, standardization, splines, embeddings).
- **Item 12c (Model Specification & Building)**:
  - Specify the type of model and rationale, all model-building steps (including hyperparameter tuning), the method for internal validation (e.g., k-fold cross-validation, bootstrapping), model stability assessment, and handling of repeated-record or clustered data.
- **Item 12d (Heterogeneity)**:
  - Describe if and how any heterogeneity in model parameter estimates and model performance across clusters (e.g., centers, subgroups) was handled.
- **Item 12e (Performance Measures & Plots)**:
  - Specify all measures and plots used to evaluate model performance, including rationale for selected metrics: discrimination (AUC with DeLong CIs for binary outcomes; C-index with censoring-aware bootstrap or Harrell's method for survival models), calibration (intercept, slope, calibration curves), clinical utility (Decision Curve Analysis net benefit), and any model comparisons.
- **Item 16 (Development vs. Evaluation Datasets)**:
  - Describe any differences between development and evaluation datasets in setting, eligibility criteria, outcome, and predictors.
- **Item 23a (Model Performance)**:
  - Report performance estimates with confidence intervals (discrimination, calibration, and net benefit) across development and validation sets, including results in key clinical subgroups.
- **Item 23b (Heterogeneity Across Clusters)**:
  - Report results evaluating heterogeneity in model performance across clusters (e.g. centers, hospitals, regions), if examined.

---

## 3. Automated Methods Narrative Templates

When synthesizing statistical methods text for clinical manuscripts, state procedures only when confirmed by the actual analysis record:

### Multivariable Logistic Regression
> "Continuous variables were summarized as mean (SD) or median (IQR) [when verified: based on normality evaluated via the Shapiro-Wilk test], and categorical variables as frequencies and percentages. [When performed: Baseline group differences were assessed using Standardized Mean Differences (SMDs), with $|\text{SMD}| < 0.10$ indicating adequate balance.] Multivariable logistic regression was fitted to estimate adjusted odds ratios (aORs) and 95% confidence intervals for the primary outcome. Missing data were handled using [complete-case analysis under MCAR / multiple imputation by chained equations (MICE, with $m$ determined by fraction of missing information) under MAR]. [When specified: Sensitivity to unmeasured confounding was quantified using the VanderWeele E-value.] [When confirmed: All tests were two-tailed with statistical significance set at \$\\alpha = 0.05\$.]"  

### Survival Analysis (Cox Proportional Hazards)
> "[When performed: Time-to-event outcomes were analyzed using the Kaplan-Meier method, and group differences were evaluated using the log-rank test.] Multivariable Cox proportional hazards regression was performed to estimate adjusted hazard ratios (aHRs) and 95% confidence intervals. [When verified: The proportional hazards assumption was assessed using Schoenfeld residual tests.] [When Firth estimation was used: Firth's penalized likelihood estimation with profile likelihood confidence intervals was employed to resolve monotone likelihood and small-sample bias.]"
