"""
src/medstat/reporting/renderers/__init__.py: Exposes multi-format document renderers.
"""

from medstat.reporting.renderers.docx import render_docx
from medstat.reporting.renderers.html import render_html
from medstat.reporting.renderers.markdown import render_markdown
from medstat.reporting.renderers.pdf import render_pdf
from medstat.reporting.renderers.pptx import render_pptx

__all__ = [
    "render_markdown",
    "render_html",
    "render_docx",
    "render_pptx",
    "render_pdf",
]
