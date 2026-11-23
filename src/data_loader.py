"""Feature 1 — load and validate student data from CSV/Excel uploads.

Expected format: one row per student per subject.

Problems are sorted into two levels:
- Errors: the row can't be used and is excluded (e.g. missing student_id, score of 105).
- Warnings: the row is kept but something needs a look (e.g. missing parent email).
"""
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

REQUIRED_COLUMNS = [
    "student_id",
    "student_name",
    "parent_name",
    "parent_email",
    "subject",
    "current_score",
    "previous_score",
    "attendance_percentage",
]
SCORE_COLUMNS = ["current_score", "previous_score", "attendance_percentage"]
TEXT_COLUMNS = ["student_id", "student_name", "parent_name", "parent_email", "subject"]

# Blank values here make a row unusable.
MUST_HAVE_VALUES = ["student_id", "student_name", "subject", "current_score"]
# Blank values here are reported but the row is kept.
NICE_TO_HAVE_VALUES = ["parent_name", "parent_email", "previous_score", "attendance_percentage"]

ATTENTION_THRESHOLD = 40  # current_score below this needs attention
SUPPORTED_EXTENSIONS = {".csv", ".xlsx", ".xls"}
EMAIL_PATTERN = r"^[^@\s]+@[^@\s]+\.[^@\s]+$"


class DataLoadError(Exception):
    """The file can't be used at all (wrong type, unreadable, empty, or missing columns)."""


@dataclass
class LoadResult:
    data: pd.DataFrame  # clean rows, with score_change and trend added
    excluded: pd.DataFrame = field(default_factory=pd.DataFrame)  # rows removed, with reason
    warnings: pd.DataFrame = field(default_factory=pd.DataFrame)  # row, student_id, issue
    duplicates: pd.DataFrame = field(default_factory=pd.DataFrame)  # duplicate rows removed
    missing_counts: pd.Series = field(default_factory=pd.Series)  # blanks per column
    total_rows: int = 0


def read_file(file, filename: str) -> pd.DataFrame:
    """Read a CSV or Excel file from a path or file-like object (e.g. a Streamlit upload)."""
    ext = Path(filename).suffix.lower()
    if ext not in SUPPORTED_EXTENSIONS:
        raise DataLoadError(f"Unsupported file type '{ext}'. Please upload a .csv or .xlsx file.")
    try:
        # Read everything as text first so we can tell "blank" apart from "not a number".
        if ext == ".csv":
            return pd.read_csv(file, dtype=str)
        return pd.read_excel(file, dtype=str)
    except Exception as exc:  # pandas raises many different error types
        raise DataLoadError(f"Could not read '{filename}'. Is it a valid file? ({exc})") from exc


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Turn headers like ' Student ID ' or 'Current-Score' into 'student_id', 'current_score'."""
    df = df.copy()
    df.columns = (
        df.columns.astype(str).str.strip().str.lower().str.replace(r"[\s\-]+", "_", regex=True)
    )
    return df


def validate(raw: pd.DataFrame) -> LoadResult:
    df = normalize_columns(raw)

    if df.empty:
        raise DataLoadError("The file has no data rows.")

    missing_cols = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing_cols:
        raise DataLoadError(
            f"Missing required column(s): {', '.join(missing_cols)}. "
            f"Columns found: {', '.join(df.columns)}."
        )

    df = df[REQUIRED_COLUMNS].copy()  # drop any extra columns
    df.insert(0, "row", df.index + 2)  # spreadsheet row number (row 1 is the header)

    for col in TEXT_COLUMNS + SCORE_COLUMNS:
        df[col] = df[col].astype("string").str.strip().replace("", pd.NA)

    missing_counts = df[REQUIRED_COLUMNS].isna().sum()
    missing_counts = missing_counts[missing_counts > 0]

    errors = pd.Series("", index=df.index)
    warnings = []

    for col in MUST_HAVE_VALUES:
        errors[df[col].isna()] += f"missing {col}; "

    for col in SCORE_COLUMNS:
        blank = df[col].isna()
        numbers = pd.to_numeric(df[col], errors="coerce")
        errors[~blank & numbers.isna()] += f"{col} is not a number; "
        errors[numbers.notna() & ((numbers < 0) | (numbers > 100))] += f"{col} must be 0–100; "
        df[col] = numbers

    for col in NICE_TO_HAVE_VALUES:
        for _, r in df[df[col].isna()].iterrows():
            warnings.append((r["row"], r["student_id"], f"missing {col}"))

    bad_email = df["parent_email"].notna() & ~df["parent_email"].str.match(EMAIL_PATTERN)
    for _, r in df[bad_email.fillna(False)].iterrows():
        warnings.append((r["row"], r["student_id"], f"parent_email looks invalid: '{r['parent_email']}'"))

    is_error = errors != ""
    excluded = df[is_error].assign(reason=errors[is_error].str.rstrip("; "))
    df = df[~is_error].copy()

    # Duplicate = same student and subject appearing more than once. Keep the first.
    dup_mask = df.duplicated(subset=["student_id", "subject"], keep="first")
    duplicates = df[dup_mask]
    df = df[~dup_mask].copy()

    # Same student_id used with different names is probably a typo.
    names_per_id = df.groupby("student_id")["student_name"].nunique()
    for sid in names_per_id[names_per_id > 1].index:
        names = ", ".join(df.loc[df["student_id"] == sid, "student_name"].unique())
        warnings.append((None, sid, f"student_id used with different names: {names}"))

    if df.empty:
        raise DataLoadError("No valid rows remain after validation. Check the excluded rows below.")

    df["score_change"] = df["current_score"] - df["previous_score"]
    df["trend"] = pd.cut(
        df["score_change"], bins=[-float("inf"), -0.5, 0.5, float("inf")],
        labels=["Declined", "No change", "Improved"],
    ).astype("string").fillna("Unknown")
    df["needs_attention"] = df["current_score"] < ATTENTION_THRESHOLD

    warnings_df = pd.DataFrame(warnings, columns=["row", "student_id", "issue"])
    # Don't warn about rows that were excluded anyway.
    warnings_df = warnings_df[~warnings_df["row"].isin(excluded["row"])].reset_index(drop=True)
    return LoadResult(
        data=df.reset_index(drop=True),
        excluded=excluded.reset_index(drop=True),
        warnings=warnings_df,
        duplicates=duplicates.reset_index(drop=True),
        missing_counts=missing_counts,
        total_rows=len(raw),
    )


def compute_summary(df: pd.DataFrame) -> dict:
    """Headline numbers for the dashboard."""
    return {
        "total_students": int(df["student_id"].nunique()),
        "average_current_score": round(float(df["current_score"].mean()), 1),
        "average_improvement": round(float(df["score_change"].mean()), 1)
        if df["score_change"].notna().any() else None,
        "students_needing_attention": int(df.loc[df["needs_attention"], "student_id"].nunique()),
    }


def load_student_data(file, filename: str) -> LoadResult:
    """Read and validate an uploaded file in one call."""
    return validate(read_file(file, filename))
