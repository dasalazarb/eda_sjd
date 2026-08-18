"""Tests for shared pipeline utilities."""

from __future__ import annotations

import warnings
from pathlib import Path

import pandas as pd
import pytest

from src.common import upsert_eda_sheets_xlsx


def test_upsert_eda_sheets_appends_consolidated_rows(tmp_path: Path) -> None:
    """Rows are appended to a consolidated sheet in a valid workbook."""
    workbook = tmp_path / "eda_unificado.xlsx"
    pd.DataFrame({"value": [1]}).to_excel(
        workbook, sheet_name="data_summary", index=False
    )

    upsert_eda_sheets_xlsx(
        workbook,
        {"data_summary": pd.DataFrame({"value": [2]})},
    )

    assert pd.read_excel(workbook, sheet_name="data_summary").to_dict("list") == {
        "value": [1, 2]
    }


def test_upsert_eda_sheets_preserves_and_replaces_corrupt_workbook(
    tmp_path: Path,
) -> None:
    """A corrupt workbook is backed up and replaced with readable output."""
    workbook = tmp_path / "eda_unificado.xlsx"
    corrupt_content = b"not an Excel zip archive"
    workbook.write_bytes(corrupt_content)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = upsert_eda_sheets_xlsx(
            workbook,
            {"data_summary": pd.DataFrame({"value": [1]})},
        )

    assert result == workbook
    assert pd.read_excel(workbook, sheet_name="data_summary").to_dict("list") == {
        "value": [1]
    }
    assert (tmp_path / "eda_unificado.xlsx.corrupt").read_bytes() == corrupt_content
    assert any("Preserved the corrupt workbook" in str(item.message) for item in caught)


def test_upsert_eda_sheets_keeps_original_if_write_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A failed update cannot replace the last valid workbook."""
    workbook = tmp_path / "eda_unificado.xlsx"
    pd.DataFrame({"value": [1]}).to_excel(
        workbook, sheet_name="data_summary", index=False
    )
    original_content = workbook.read_bytes()

    def fail_to_excel(*args, **kwargs) -> None:
        raise RuntimeError("simulated write failure")

    monkeypatch.setattr(pd.DataFrame, "to_excel", fail_to_excel)

    try:
        upsert_eda_sheets_xlsx(
            workbook,
            {"data_summary": pd.DataFrame({"value": [2]})},
        )
    except RuntimeError as exc:
        assert str(exc) == "simulated write failure"
    else:
        raise AssertionError("Expected the simulated write failure")

    assert workbook.read_bytes() == original_content
    assert not list(tmp_path.glob("*.tmp.xlsx"))
