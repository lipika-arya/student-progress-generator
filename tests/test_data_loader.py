import io
from pathlib import Path

import pandas as pd
import pytest

from src.data_loader import DataLoadError, compute_summary, load_student_data, validate

SAMPLE_DIR = Path(__file__).resolve().parent.parent / "data" / "sample"


def make_df(**overrides):
    base = {
        "student_id": ["S1", "S2"],
        "student_name": ["Ann", "Ben"],
        "parent_name": ["Pat", "Sam"],
        "parent_email": ["pat@example.com", "sam@example.com"],
        "subject": ["Math", "Math"],
        "current_score": ["70", "35"],
        "previous_score": ["60", "45"],
        "attendance_percentage": ["95", "80"],
    }
    base.update(overrides)
    return pd.DataFrame(base)


def test_sample_file_loads_cleanly():
    with open(SAMPLE_DIR / "sample_students.csv", "rb") as f:
        result = load_student_data(f, "sample_students.csv")
    assert result.excluded.empty and result.duplicates.empty and result.warnings.empty
    summary = compute_summary(result.data)
    assert summary["total_students"] == 10
    assert summary["students_needing_attention"] == 2  # S003 and S008 score below 40


def test_error_sample_file_reports_each_problem():
    with open(SAMPLE_DIR / "sample_students_with_errors.csv", "rb") as f:
        result = load_student_data(f, "sample_students_with_errors.csv")
    reasons = " ".join(result.excluded["reason"])
    assert "current_score must be 0–100" in reasons          # 105
    assert "attendance_percentage must be 0–100" in reasons  # 120
    assert "missing student_name" in reasons
    assert "current_score is not a number" in reasons        # abc
    assert len(result.duplicates) == 1
    issues = " ".join(result.warnings["issue"])
    assert "missing parent_name" in issues
    assert "missing previous_score" in issues
    assert "not-an-email" not in issues  # that row was excluded, so no extra warning


def test_score_change_and_summary():
    result = validate(make_df())
    assert result.data["score_change"].tolist() == [10, -10]
    assert result.data["trend"].tolist() == ["Improved", "Declined"]
    summary = compute_summary(result.data)
    assert summary == {
        "total_students": 2,
        "average_current_score": 52.5,
        "average_improvement": 0.0,
        "students_needing_attention": 1,
    }


def test_messy_headers_are_normalized():
    df = make_df()
    df.columns = [f" {c.replace('_', ' ').title()} " for c in df.columns]
    assert len(validate(df).data) == 2


def test_missing_column_raises_clear_error():
    with pytest.raises(DataLoadError, match="attendance_percentage"):
        validate(make_df().drop(columns=["attendance_percentage"]))


def test_invalid_email_is_a_warning_not_an_error():
    result = validate(make_df(parent_email=["pat@example.com", "nope"]))
    assert len(result.data) == 2
    assert "looks invalid" in result.warnings.iloc[0]["issue"]


def test_missing_previous_score_gives_unknown_trend():
    result = validate(make_df(previous_score=["60", ""]))
    assert result.data.loc[1, "trend"] == "Unknown"


def test_all_rows_invalid_raises():
    with pytest.raises(DataLoadError, match="No valid rows"):
        validate(make_df(current_score=["200", "-5"]))


def test_unsupported_file_type():
    with pytest.raises(DataLoadError, match="Unsupported"):
        load_student_data(io.BytesIO(b""), "scores.pdf")
