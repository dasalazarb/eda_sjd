"""Tests for PRO completeness rules in ``23_eval_cohorts_sample.py``."""

import importlib.util
from pathlib import Path

import pandas as pd
import pytest


SCRIPT_PATH = Path(__file__).parents[1] / "src" / "23_eval_cohorts_sample.py"
SPEC = importlib.util.spec_from_file_location("eval_cohorts_sample", SCRIPT_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_complete_pro_visit_mask_accepts_complete_esspri() -> None:
    """A visit with all three ESSPRI items qualifies."""
    df = pd.DataFrame(
        {
            MODULE.COL_ESSPRI_DRY: [1.0],
            MODULE.COL_ESSPRI_FAT: [2.0],
            MODULE.COL_ESSPRI_PAIN: [3.0],
        }
    )

    assert MODULE.complete_pro_visit_mask(df).tolist() == [True]


def test_complete_pro_visit_mask_falls_back_to_complete_sf36() -> None:
    """Incomplete ESSPRI falls through to a complete SF-36 assessment."""
    data = {column: [1.0] for column in MODULE.COLS_SF36}
    data.update(
        {
            MODULE.COL_ESSPRI_DRY: [1.0],
            MODULE.COL_ESSPRI_FAT: [2.0],
            MODULE.COL_ESSPRI_PAIN: [None],
        }
    )
    df = pd.DataFrame(data)

    assert MODULE.complete_pro_visit_mask(df).tolist() == [True]


@pytest.mark.parametrize(
    "columns",
    [
        [f"{MODULE.PROFAD_PREFIX}{item}" for item in range(1, 20)],
        [f"{MODULE.MDAFS_MAF_PREFIX}{item}" for item in range(1, 5)],
    ],
    ids=["profad", "mdafs_maf"],
)
def test_complete_pro_visit_mask_accepts_complete_fatigue_group(
    columns: list[str],
) -> None:
    """Complete PROFAD and MDAFS/MAF assessments each qualify a visit."""
    df = pd.DataFrame({column: [1.0] for column in columns})

    assert MODULE.complete_pro_visit_mask(df).tolist() == [True]


def test_complete_pro_visit_mask_rejects_all_incomplete_groups() -> None:
    """Partial questionnaires and an incomplete 19-item PROFAD do not qualify."""
    data = {
        MODULE.COL_ESSPRI_DRY: [1.0],
        MODULE.COL_ESSPRI_FAT: [2.0],
        MODULE.COL_ESSPRI_PAIN: [None],
    }
    data.update(
        {
            f"{MODULE.PROFAD_PREFIX}{item}": [1.0 if item < 19 else None]
            for item in range(1, 20)
        }
    )
    df = pd.DataFrame(data)

    assert MODULE.complete_pro_visit_mask(df).tolist() == [False]
