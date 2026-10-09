"""
Publication Reporting, Table 1, and Narrative Generation Module.
"""

from medstat.reporting.integrity import (
    IntegrityReport,
    verify_report_integrity,
)
from medstat.reporting.ir import (
    Block,
    CalloutBlock,
    FigureBlock,
    HeadingBlock,
    ParagraphBlock,
    ReportDocument,
    TableBlock,
)
from medstat.reporting.narrative import (
    generate_methods_narrative,
    validate_calibration_metrics,
)
from medstat.reporting.renderers import (
    render_docx,
    render_html,
    render_markdown,
    render_pdf,
    render_pptx,
)
from medstat.reporting.table1 import generate_table_one
from medstat.reporting.tables import (
    Estimate,
    EstimateTable,
    ModelMeta,
    PublicationRenderer,
    format_journal_p_value,
    render_apa_table,
    render_ascii_table,
    render_jama_table,
    render_nejm_table,
)

__all__ = [
    "generate_table_one",
    "Estimate",
    "ModelMeta",
    "EstimateTable",
    "PublicationRenderer",
    "format_journal_p_value",
    "render_nejm_table",
    "render_jama_table",
    "render_apa_table",
    "render_ascii_table",
    "generate_methods_narrative",
    "validate_calibration_metrics",
    "ReportDocument",
    "Block",
    "HeadingBlock",
    "ParagraphBlock",
    "TableBlock",
    "FigureBlock",
    "CalloutBlock",
    "render_markdown",
    "render_html",
    "render_docx",
    "render_pptx",
    "render_pdf",
    "verify_report_integrity",
    "IntegrityReport",
]
