"""
src/medstat/reporting/renderers/pptx.py: Native PowerPoint (.pptx) presentation renderer.
Generates 16:9 widescreen presentations with executive summary, 1 figure per slide layout,
and formatted tables.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import PP_ALIGN
from pptx.util import Inches, Pt

from medstat.reporting.ir import (
    CalloutBlock,
    FigureBlock,
    HeadingBlock,
    ParagraphBlock,
    ReportDocument,
    TableBlock,
)

# Color Palette (Clinical Executive Slate & Blue)
COLOR_PRIMARY = RGBColor(15, 23, 42)  # Slate 900
COLOR_SECONDARY = RGBColor(30, 41, 59)  # Slate 800
COLOR_ACCENT = RGBColor(2, 132, 199)  # Sky 600
COLOR_MUTED = RGBColor(100, 116, 139)  # Slate 500
COLOR_BG_LIGHT = RGBColor(248, 250, 252)  # Slate 50
COLOR_WHITE = RGBColor(255, 255, 255)
COLOR_CARD_BORDER = RGBColor(226, 232, 240)


def _add_title_slide(prs: Presentation, doc: ReportDocument) -> None:
    """Creates an executive 16:9 widescreen title slide."""
    blank_layout = prs.slide_layouts[6]
    slide = prs.slides.add_slide(blank_layout)

    # Accent top border bar
    top_bar = slide.shapes.add_shape(
        MSO_SHAPE.RECTANGLE, Inches(0), Inches(0), Inches(13.333), Inches(0.15)
    )
    top_bar.fill.solid()
    top_bar.fill.fore_color.rgb = COLOR_ACCENT
    top_bar.line.color.rgb = COLOR_ACCENT

    # Title box
    title_box = slide.shapes.add_textbox(
        Inches(1.2), Inches(2.2), Inches(10.9), Inches(2.2)
    )
    tf = title_box.text_frame
    tf.word_wrap = True

    p = tf.paragraphs[0]
    p.text = doc.title
    p.font.name = "Arial"
    p.font.size = Pt(36)
    p.font.bold = True
    p.font.color.rgb = COLOR_PRIMARY
    p.space_after = Pt(14)

    # Subtitle / Affiliation
    if doc.institution:
        p2 = tf.add_paragraph()
        p2.text = doc.institution.upper()
        p2.font.name = "Arial"
        p2.font.size = Pt(13)
        p2.font.bold = True
        p2.font.color.rgb = COLOR_ACCENT
        p2.space_after = Pt(20)

    # Authors and Date metadata
    meta_parts = []
    if doc.authors:
        meta_parts.append(f"Authors: {', '.join(doc.authors)}")
    if doc.date:
        meta_parts.append(f"Date: {doc.date}")

    if meta_parts:
        p3 = tf.add_paragraph()
        p3.text = "   |   ".join(meta_parts)
        p3.font.name = "Arial"
        p3.font.size = Pt(12)
        p3.font.color.rgb = COLOR_MUTED


def _add_header_to_slide(
    slide, title_text: str, category_text: str = "CLINICAL RESEARCH"
) -> None:
    """Adds a standard header area to a content slide."""
    header_box = slide.shapes.add_textbox(
        Inches(0.8), Inches(0.5), Inches(11.7), Inches(1.0)
    )
    tf = header_box.text_frame
    tf.word_wrap = True

    # Category breadcrumb
    p_cat = tf.paragraphs[0]
    p_cat.text = category_text.upper()
    p_cat.font.name = "Arial"
    p_cat.font.size = Pt(9.5)
    p_cat.font.bold = True
    p_cat.font.color.rgb = COLOR_ACCENT
    p_cat.space_after = Pt(2)

    # Slide Title
    p_title = tf.add_paragraph()
    p_title.text = title_text
    p_title.font.name = "Arial"
    p_title.font.size = Pt(22)
    p_title.font.bold = True
    p_title.font.color.rgb = COLOR_PRIMARY


def _add_figure_slide(
    prs: Presentation, block: FigureBlock, section_title: str
) -> None:
    """Creates a dedicated 16:9 slide for a Figure with 1-figure-per-slide layout."""
    blank_layout = prs.slide_layouts[6]
    slide = prs.slides.add_slide(blank_layout)

    # Slide Title from figure caption or section
    title = section_title if section_title else "Figure Visualization"
    _add_header_to_slide(slide, title, category_text="FIGURE & VISUALIZATION")

    fig_path = Path(block.figure.png_path)
    if not fig_path.exists():
        raise FileNotFoundError(f"Figure file not found: {fig_path}")

    # Split layout: Figure on left (7.6 in), Callout / Description card on right (3.8 in)
    has_text_card = bool(block.figure.caption or block.figure.alt_text)

    if has_text_card:
        # Add Picture
        slide.shapes.add_picture(
            str(fig_path), Inches(0.8), Inches(1.6), width=Inches(7.6)
        )

        # Add Side Details Card
        card = slide.shapes.add_shape(
            MSO_SHAPE.ROUNDED_RECTANGLE,
            Inches(8.7),
            Inches(1.6),
            Inches(3.8),
            Inches(5.0),
        )
        card.fill.solid()
        card.fill.fore_color.rgb = COLOR_BG_LIGHT
        card.line.color.rgb = COLOR_CARD_BORDER
        card.line.width = Pt(1)

        card_tf = card.text_frame
        card_tf.word_wrap = True
        card_tf.margin_left = Inches(0.3)
        card_tf.margin_right = Inches(0.3)
        card_tf.margin_top = Inches(0.3)
        card_tf.margin_bottom = Inches(0.3)

        p1 = card_tf.paragraphs[0]
        p1.text = "Key Takeaways"
        p1.font.name = "Arial"
        p1.font.size = Pt(13)
        p1.font.bold = True
        p1.font.color.rgb = COLOR_PRIMARY
        p1.space_after = Pt(8)

        p2 = card_tf.add_paragraph()
        p2.text = block.figure.caption
        p2.font.name = "Arial"
        p2.font.size = Pt(10.5)
        p2.font.color.rgb = COLOR_SECONDARY
        p2.space_after = Pt(12)

        if block.figure.alt_text and block.figure.alt_text != block.figure.caption:
            p3 = card_tf.add_paragraph()
            p3.text = f"Context: {block.figure.alt_text}"
            p3.font.name = "Arial"
            p3.font.size = Pt(9.5)
            p3.font.italic = True
            p3.font.color.rgb = COLOR_MUTED
    else:
        # Centered Figure
        slide.shapes.add_picture(
            str(fig_path), Inches(2.2), Inches(1.6), width=Inches(8.9)
        )


def _add_table_slide(prs: Presentation, block: TableBlock, section_title: str) -> None:
    """Creates dedicated slide(s) for TableBlock, continuing onto multiple slides if > 12 rows."""
    df = block.df
    chunk_size = 12
    total_rows = len(df)
    n_chunks = (
        max(1, (total_rows + chunk_size - 1) // chunk_size) if total_rows > 0 else 1
    )
    base_title = section_title if section_title else block.caption

    for chunk_idx in range(n_chunks):
        start_row = chunk_idx * chunk_size
        end_row = min(start_row + chunk_size, total_rows)
        chunk_df = df.iloc[start_row:end_row] if total_rows > 0 else df

        blank_layout = prs.slide_layouts[6]
        slide = prs.slides.add_slide(blank_layout)

        title = base_title
        if n_chunks > 1:
            title = f"{base_title} (Part {chunk_idx + 1}/{n_chunks})"
        _add_header_to_slide(slide, title, category_text="STATISTICAL TABLE")

        n_rows, n_cols = len(chunk_df) + 1, len(chunk_df.columns)

        table_left = Inches(0.8)
        table_top = Inches(1.6)
        table_width = Inches(11.7)
        table_height = Inches(min(4.8, 0.35 * n_rows + 0.4))

        table_shape = slide.shapes.add_table(
            n_rows, n_cols, table_left, table_top, table_width, table_height
        )
        tbl = table_shape.table

        # Format Header Row
        for c_idx, col_name in enumerate(chunk_df.columns):
            cell = tbl.cell(0, c_idx)
            cell.text = str(col_name)
            cell.fill.solid()
            cell.fill.fore_color.rgb = COLOR_PRIMARY
            p = cell.text_frame.paragraphs[0]
            p.alignment = PP_ALIGN.CENTER
            p.font.name = "Arial"
            p.font.size = Pt(10)
            p.font.bold = True
            p.font.color.rgb = COLOR_WHITE

        # Format Data Rows
        for r_idx, (_, row) in enumerate(chunk_df.iterrows()):
            bg_color = COLOR_BG_LIGHT if r_idx % 2 == 1 else COLOR_WHITE
            for c_idx, col_name in enumerate(chunk_df.columns):
                cell = tbl.cell(r_idx + 1, c_idx)
                val = str(row[col_name]) if pd.notna(row[col_name]) else ""
                cell.text = val
                cell.fill.solid()
                cell.fill.fore_color.rgb = bg_color
                p = cell.text_frame.paragraphs[0]
                p.alignment = PP_ALIGN.LEFT if c_idx == 0 else PP_ALIGN.CENTER
                p.font.name = "Arial"
                p.font.size = Pt(9.5)
                p.font.color.rgb = COLOR_SECONDARY

        # Footnote
        notes = []
        if block.footnote:
            notes.append(block.footnote)
        if n_chunks > 1:
            notes.append(f"Rows {start_row + 1}–{end_row} of {total_rows}")

        if notes:
            fn_box = slide.shapes.add_textbox(
                Inches(0.8), Inches(6.6), Inches(11.7), Inches(0.6)
            )
            fn_tf = fn_box.text_frame
            fn_tf.word_wrap = True
            fn_p = fn_tf.paragraphs[0]
            fn_p.text = "   |   ".join(notes)
            fn_p.font.name = "Arial"
            fn_p.font.size = Pt(9)
            fn_p.font.italic = True
            fn_p.font.color.rgb = COLOR_MUTED


def _add_content_slide(
    prs: Presentation, title: str, paragraphs: list[str], callouts: list[str]
) -> None:
    """Creates a text and narrative slide with bullet points and callout boxes."""
    chunk_size = 5
    para_chunks = (
        [paragraphs[i : i + chunk_size] for i in range(0, len(paragraphs), chunk_size)]
        if paragraphs
        else [[]]
    )

    for chunk_idx, chunk in enumerate(para_chunks):
        blank_layout = prs.slide_layouts[6]
        slide = prs.slides.add_slide(blank_layout)

        slide_title = f"{title} (cont.)" if chunk_idx > 0 else title
        _add_header_to_slide(slide, slide_title, category_text="FINDINGS & METHODS")

        # Main text box
        tb = slide.shapes.add_textbox(
            Inches(0.8), Inches(1.6), Inches(11.7), Inches(3.8)
        )
        tf = tb.text_frame
        tf.word_wrap = True

        for idx, p_text in enumerate(chunk):
            p = tf.paragraphs[0] if idx == 0 else tf.add_paragraph()
            p.text = f"•  {p_text}"
            p.font.name = "Arial"
            p.font.size = Pt(13)
            p.font.color.rgb = COLOR_SECONDARY
            p.space_after = Pt(10)

        # Render all callouts on the final slide of this section
        if chunk_idx == len(para_chunks) - 1 and callouts:
            c_top = Inches(5.4)
            card = slide.shapes.add_shape(
                MSO_SHAPE.ROUNDED_RECTANGLE,
                Inches(0.8),
                c_top,
                Inches(11.7),
                Inches(1.5),
            )
            card.fill.solid()
            card.fill.fore_color.rgb = COLOR_BG_LIGHT
            card.line.color.rgb = COLOR_ACCENT
            card.line.width = Pt(1.5)

            ctf = card.text_frame
            ctf.word_wrap = True
            for c_idx, c_text in enumerate(callouts):
                cp = ctf.paragraphs[0] if c_idx == 0 else ctf.add_paragraph()
                cp.text = f"NOTE: {c_text}"
                cp.font.name = "Arial"
                cp.font.size = Pt(11)
                cp.font.bold = True
                cp.font.color.rgb = COLOR_PRIMARY
                if c_idx < len(callouts) - 1:
                    cp.space_after = Pt(4)


def render_pptx(
    doc: ReportDocument,
    out_path: str | Path,
) -> str:
    """
    Renders ReportDocument into a 16:9 widescreen PowerPoint presentation (.pptx).

    Structure:
    1. Title Slide
    2. Sequenced Content / Table / Figure slides per section
    """
    out_p = Path(out_path).resolve()
    out_p.parent.mkdir(parents=True, exist_ok=True)

    prs = Presentation()
    # 16:9 Widescreen standard: 13.333 in x 7.5 in
    prs.slide_width = Inches(13.333)
    prs.slide_height = Inches(7.5)

    # Slide 1: Title Slide
    _add_title_slide(prs, doc)

    # Iterate through blocks, grouping narrative paragraphs by heading
    current_heading = "Summary & Overview"
    pending_paragraphs: list[str] = []
    pending_callouts: list[str] = []

    def flush_pending_content():
        nonlocal pending_paragraphs, pending_callouts
        if pending_paragraphs or pending_callouts:
            _add_content_slide(
                prs, current_heading, pending_paragraphs, pending_callouts
            )
            pending_paragraphs = []
            pending_callouts = []

    for block in doc.blocks:
        if isinstance(block, HeadingBlock):
            flush_pending_content()
            current_heading = block.text
        elif isinstance(block, ParagraphBlock):
            pending_paragraphs.append(block.text)
        elif isinstance(block, CalloutBlock):
            pending_callouts.append(block.text)
        elif isinstance(block, TableBlock):
            flush_pending_content()
            _add_table_slide(prs, block, current_heading)
        elif isinstance(block, FigureBlock):
            flush_pending_content()
            _add_figure_slide(prs, block, current_heading)

    flush_pending_content()

    prs.save(str(out_p))
    return str(out_p)
