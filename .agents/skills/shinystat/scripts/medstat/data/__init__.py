"""
src/medstat/data: Clinically Sound Data Cleaning, Missing Data Imputation & Audited Retention Flow.
"""

from __future__ import annotations

from medstat.data.clean import (
    DataCleaningError,
    DataValidationError,
    LittlesMCARResult,
    MissingnessAudit,
    MissingPattern,
    OutlierResult,
    VariableMissingAudit,
    apply_missing_values_to_df,
    audit_missingness,
    check_missing_data_impact,
    clean_dataframe,
    clean_numeric,
    clean_numeric_vector,
    detect_missing_in_variable,
    detect_outliers,
    detect_zero_variance,
    get_cleaning_summary,
    get_missing_summary_df,
    handle_outliers,
    littles_mcar_test,
)
from medstat.data.missing import (
    MICEImputer,
    MICEResult,
    MissingDataError,
    MissingStrategyRequiredError,
    PooledEstimate,
    PooledRegressionResults,
    create_imputation_diagnostics,
    get_imputation_summary,
    handle_missing_for_analysis,
    impute_indicator,
    impute_knn,
    pool_estimates,
    pool_regression_results,
    prepare_data_for_analysis,
)
from medstat.data.quality import (
    ColumnRule,
    CrossVariableRule,
    DataQualityReport,
    DataQualitySchema,
    check_data_quality,
)
from medstat.data.retention import (
    ExclusionReason,
    FlowDesign,
    FlowStage,
    SampleFlowTracker,
)

__all__ = [
    # clean.py
    "clean_numeric",
    "clean_numeric_vector",
    "clean_dataframe",
    "get_cleaning_summary",
    "detect_outliers",
    "handle_outliers",
    "detect_zero_variance",
    "apply_missing_values_to_df",
    "detect_missing_in_variable",
    "get_missing_summary_df",
    "check_missing_data_impact",
    "audit_missingness",
    "littles_mcar_test",
    "MissingnessAudit",
    "VariableMissingAudit",
    "MissingPattern",
    "OutlierResult",
    "LittlesMCARResult",
    "DataCleaningError",
    "DataValidationError",
    # missing.py
    "MissingDataError",
    "MissingStrategyRequiredError",
    "prepare_data_for_analysis",
    "handle_missing_for_analysis",
    "MICEImputer",
    "MICEResult",
    "PooledEstimate",
    "PooledRegressionResults",
    "pool_estimates",
    "pool_regression_results",
    "impute_knn",
    "impute_indicator",
    "get_imputation_summary",
    "create_imputation_diagnostics",
    # retention.py
    "SampleFlowTracker",
    "FlowStage",
    "ExclusionReason",
    "FlowDesign",
    # quality.py
    "DataQualityReport",
    "check_data_quality",
    "DataQualitySchema",
    "ColumnRule",
    "CrossVariableRule",
]
