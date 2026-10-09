"""
src/medstat/reporting/renderers/markdown.py: Markdown (.md) document renderer.
"""

from __future__ import annotations

import shutil
from pathlib import Path

from medstat.reporting.ir import (
    CalloutBlock,
    FigureBlock,
    HeadingBlock,
    ParagraphBlock,
    ReportDocument,
    TableBlock,
)


def render_markdown(
    doc: ReportDocument,
    out_path: str | Path,
    assets_dir_name: str = "figures",
) -> str:
    """
    Renders ReportDocument into clean GitHub Flavored Markdown with linked images.
    """
    out_p = Path(out_path).resolve()
    out_p.parent.mkdir(parents=True, exist_ok=True)
    assets_dir = out_p.parent / assets_dir_name
    assets_dir.mkdir(parents=True, exist_ok=True)

    lines: list[str] = [f"# {doc.title}\n"]

    if doc.authors or doc.date:
        meta_items = []
        if doc.authors:
            meta_items.append(f"**Authors:** {', '.join(doc.authors)}")
        if doc.date:
            meta_items.append(f"**Date:** {doc.date}")
        if doc.institution:
            meta_items.append(f"**Affiliation:** {doc.institution}")
        lines.append(" | ".join(meta_items) + "\n\n---\n")

    for block in doc.blocks:
        if isinstance(block, HeadingBlock):
            prefix = "#" * max(1, min(6, block.level + 1))  # Document title is H1
            lines.append(f"\n{prefix} {block.text}\n")
        elif isinstance(block, ParagraphBlock):
            lines.append(f"{block.text}\n")
        elif isinstance(block, CalloutBlock):
            lines.append(f"\n> [!NOTE]\n> {block.text}\n")
        elif isinstance(block, TableBlock):
            lines.append(f"\n**{block.caption}**\n")
            try:
                table_md = block.df.to_markdown(index=False)
            except Exception:
                table_md = block.df.to_string(index=False)
            lines.append(f"{table_md}\n")
            if block.footnote:
                lines.append(f"*{block.footnote}*\n")
        elif isinstance(block, FigureBlock):
            # Copy figure into assets directory
            fig_src = Path(block.figure.png_path)
            fig_dest = assets_dir / fig_src.name
            if fig_src.resolve() != fig_dest.resolve():
                shutil.copy2(fig_src, fig_dest)
            rel_path = f"{assets_dir_name}/{fig_src.name}"

            lines.append(f"\n![{block.figure.alt_text}]({rel_path})\n")
            lines.append(f"*{block.figure.caption}*\n")

    md_content = "\n".join(lines)
    out_p.write_text(md_content, encoding="utf-8")
    return str(out_p)
