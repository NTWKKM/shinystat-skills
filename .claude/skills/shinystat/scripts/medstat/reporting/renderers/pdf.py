"""
src/medstat/reporting/renderers/pdf.py: PDF document renderer.
Uses Playwright headless Chromium as primary engine, with LibreOffice (soffice) as fallback.
"""

from __future__ import annotations

import shutil
import subprocess
import tempfile
from pathlib import Path

from medstat.reporting.ir import ReportDocument
from medstat.reporting.renderers.html import render_html


def render_pdf(
    doc: ReportDocument,
    out_path: str | Path,
) -> str:
    """
    Renders ReportDocument into a publication-grade PDF file.

    Primary engine: Headless Chromium via Playwright (page.pdf)
    Fallback engine: LibreOffice (soffice --headless --convert-to pdf)
    """
    out_p = Path(out_path).resolve()
    out_p.parent.mkdir(parents=True, exist_ok=True)

    # 1. Try Playwright
    try:
        from playwright.sync_api import sync_playwright  # type: ignore

        with tempfile.NamedTemporaryFile(suffix=".html", delete=False) as tmp:
            tmp_html = Path(tmp.name)

        try:
            render_html(doc, tmp_html)
            with sync_playwright() as p:
                browser = p.chromium.launch(headless=True)
                page = browser.new_page()
                page.goto(f"file://{tmp_html.resolve()}", wait_until="networkidle")
                page.pdf(
                    path=str(out_p),
                    format="A4",
                    print_background=True,
                    margin={
                        "top": "20mm",
                        "bottom": "20mm",
                        "left": "20mm",
                        "right": "20mm",
                    },
                )
                browser.close()
            return str(out_p)
        finally:
            if tmp_html.exists():
                tmp_html.unlink()
    except Exception as pw_err:
        # 2. Try soffice fallback if Playwright fails
        soffice_path = shutil.which("soffice")
        if soffice_path:
            with tempfile.NamedTemporaryFile(suffix=".docx", delete=False) as tmp_docx:
                docx_path = Path(tmp_docx.name)
            try:
                from medstat.reporting.renderers.docx import render_docx

                render_docx(doc, docx_path)
                cmd = [
                    soffice_path,
                    "--headless",
                    "--convert-to",
                    "pdf",
                    str(docx_path),
                    "--outdir",
                    str(out_p.parent),
                ]
                subprocess.run(cmd, capture_output=True, text=True, check=True)
                generated_pdf = out_p.parent / f"{docx_path.stem}.pdf"
                if generated_pdf.exists():
                    shutil.move(str(generated_pdf), str(out_p))
                    return str(out_p)
            except Exception as soffice_err:
                raise RuntimeError(
                    f"PDF rendering failed with Playwright ({pw_err}) and LibreOffice ({soffice_err})."
                ) from soffice_err
            finally:
                if docx_path.exists():
                    docx_path.unlink()

        raise RuntimeError(
            f"PDF rendering failed: Playwright Chromium error ({pw_err}), and LibreOffice 'soffice' was not found on system PATH. "
            f"Ensure Playwright browser is installed via 'uv run playwright install chromium'."
        ) from pw_err
