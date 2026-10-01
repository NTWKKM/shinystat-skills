"""
Universal Clinical Data Ingestion Module.

Handles multi-format clinical files (.csv, .tsv, .xlsx, .parquet, .txt),
normalizes column naming, and provides typo-tolerant validation with difflib suggestions.
"""

from __future__ import annotations

import difflib
from pathlib import Path

import click
import pandas as pd


def load_clinical_data(
    path: str | Path,
    sheet_name: str | int = 0,
    encoding: str | None = None,
) -> pd.DataFrame:
    """
    Universally load clinical datasets across file formats.

    Supported formats:
    - Comma-separated (.csv, .txt)
    - Tab-separated (.tsv)
    - Microsoft Excel (.xlsx)
    - Columnar Parquet (.parquet)
    """
    p = Path(path)
    if not p.exists():
        raise click.ClickException(f"Dataset file not found: {path}")

    ext = p.suffix.lower()
    try:
        if ext == ".xlsx":
            df = pd.read_excel(p, sheet_name=sheet_name)
        elif ext == ".xls":
            raise click.ClickException(
                "Legacy Excel format (.xls) is not supported. "
                "Please convert the file to modern Excel (.xlsx) or CSV."
            )
        elif ext == ".parquet":
            df = pd.read_parquet(p)
        elif ext in (".csv", ".tsv", ".txt"):
            sep = "\t" if ext == ".tsv" else ","
            encodings = (
                [encoding] if encoding else ["utf-8", "utf-8-sig", "cp1252", "latin1"]
            )
            df = None
            last_decode_err: Exception | None = None
            for enc in encodings:
                try:
                    df = pd.read_csv(p, sep=sep, encoding=enc)
                    break
                except UnicodeDecodeError as err:
                    last_decode_err = err
                    continue
            if df is None:
                if last_decode_err is not None:
                    raise last_decode_err
                raise click.ClickException(
                    f"Failed to decode '{path}' with supported encodings."
                )
        else:
            raise click.ClickException(
                f"Unsupported file format '{ext}'. "
                "Supported formats are: .csv, .tsv, .txt, .xlsx, .parquet."
            )
    except click.ClickException:
        raise
    except Exception as e:
        raise click.ClickException(f"Failed to load dataset '{path}': {e}")

    # Standardize column headers: strip leading/trailing whitespace
    df.columns = [str(c).strip() for c in df.columns]
    return df


def validate_columns(
    df: pd.DataFrame,
    requested: list[str],
    subcommand: str,
) -> None:
    """
    Defensive column validator with typo-tolerant difflib suggestions.
    """
    missing = [c for c in requested if c and c not in df.columns]
    if missing:
        suggestions: dict[str, list[str]] = {}
        for m in missing:
            matches = difflib.get_close_matches(
                m, [str(c) for c in df.columns], n=3, cutoff=0.5
            )
            if matches:
                suggestions[m] = matches

        sug_text = ""
        if suggestions:
            sug_lines = [
                f"  - Did you mean '{', '.join(v)}' instead of '{k}'?"
                for k, v in suggestions.items()
            ]
            sug_text = "\nSuggestions:\n" + "\n".join(sug_lines)

        avail_cols = list(df.columns)
        col_list_str = ", ".join(f"'{c}'" for c in avail_cols[:20])
        if len(avail_cols) > 20:
            col_list_str += f" ... ({len(avail_cols) - 20} more)"

        raise click.ClickException(
            f"[{subcommand}] Required column(s) {missing} not found in dataset.\n"
            f"Available columns: [{col_list_str}]{sug_text}"
        )
