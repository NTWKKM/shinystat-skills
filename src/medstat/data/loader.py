"""
Universal Clinical Data Ingestion Module.

Handles multi-format clinical files (.csv, .tsv, .xlsx, .parquet, .txt),
normalizes column naming, and provides typo-tolerant validation with difflib suggestions.
"""

from __future__ import annotations

import difflib
from pathlib import Path
from typing import Any

import click
import pandas as pd

_READ_ENCODINGS = ["utf-8", "utf-8-sig", "cp1252", "latin1"]


def _normalize_columns(raw_cols: list[Any] | pd.Index) -> list[str]:
    """Strip whitespace and verify header uniqueness."""
    normalized_cols = [str(c).strip() for c in raw_cols]
    seen = set()
    duplicates = []
    for c in normalized_cols:
        if c in seen and c not in duplicates:
            duplicates.append(c)
        seen.add(c)
    if duplicates:
        raise click.ClickException(
            f"Dataset contains duplicate column names after normalization: {duplicates}. "
            "Please ensure each column has a unique header in the source file."
        )
    return normalized_cols


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
            df = pd.read_excel(
                p, sheet_name=sheet_name, keep_default_na=False, na_values=[""]
            )
        elif ext == ".xls":
            raise click.ClickException(
                "Legacy Excel format (.xls) is not supported. "
                "Please convert the file to modern Excel (.xlsx) or CSV."
            )
        elif ext == ".parquet":
            df = pd.read_parquet(p)
        elif ext in (".csv", ".tsv", ".txt"):
            sep = "\t" if ext == ".tsv" else ","
            encodings = [encoding] if encoding else _READ_ENCODINGS
            df = None
            last_decode_err: Exception | None = None
            for enc in encodings:
                try:
                    df = pd.read_csv(
                        p, sep=sep, encoding=enc, keep_default_na=False, na_values=[""]
                    )
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

    df.columns = _normalize_columns(df.columns)
    return df


def get_clinical_columns(
    path: str | Path,
    sheet_name: str | int = 0,
) -> list[str]:
    """
    Read column names from dataset headers or schema metadata without loading all records.
    Preserves whitespace normalization and duplicate column verification.
    """
    p = Path(path)
    if not p.exists():
        raise click.ClickException(f"Dataset file not found: {path}")

    ext = p.suffix.lower()
    try:
        if ext == ".parquet":
            try:
                import pyarrow.parquet as pq

                schema = pq.read_schema(p)
                raw_cols = list(schema.names)
                if schema.pandas_metadata:
                    index_cols = set()
                    for idx_col in schema.pandas_metadata.get("index_columns", []):
                        if isinstance(idx_col, str):
                            index_cols.add(idx_col)
                        elif isinstance(idx_col, dict) and idx_col.get("name"):
                            index_cols.add(idx_col["name"])
                    if index_cols:
                        raw_cols = [c for c in raw_cols if c not in index_cols]
            except ImportError:
                df_empty = pd.read_parquet(p)
                raw_cols = list(df_empty.columns)
        elif ext == ".xlsx":
            df_head = pd.read_excel(p, sheet_name=sheet_name, nrows=0)
            raw_cols = list(df_head.columns)
        elif ext == ".xls":
            raise click.ClickException(
                "Legacy Excel format (.xls) is not supported. "
                "Please convert the file to modern Excel (.xlsx) or CSV."
            )
        elif ext in (".csv", ".tsv", ".txt"):
            sep = "\t" if ext == ".tsv" else ","
            raw_cols = None
            last_decode_err = None
            for enc in _READ_ENCODINGS:
                try:
                    df_head = pd.read_csv(p, sep=sep, encoding=enc, nrows=0)
                    raw_cols = list(df_head.columns)
                    break
                except UnicodeDecodeError as err:
                    last_decode_err = err
                    continue
            if raw_cols is None:
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
        raise click.ClickException(f"Failed to read columns from '{path}': {e}")

    return _normalize_columns(raw_cols)


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
