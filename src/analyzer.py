"""Turn validated rows into one fact sheet per student.

All analysis happens here in plain Python, so the AI only *describes* numbers we
computed — it never calculates or invents them.
"""
from dataclasses import asdict, dataclass, field

import pandas as pd

from src.data_loader import ATTENTION_THRESHOLD


@dataclass
class SubjectResult:
    subject: str
    current_score: float
    previous_score: float | None
    change: float | None
    trend: str  # Improved / Declined / No change / Unknown


@dataclass
class StudentProfile:
    student_id: str
    student_name: str
    parent_name: str | None
    parent_email: str | None
    attendance_percentage: float | None
    subjects: list[SubjectResult]
    average_current_score: float
    average_previous_score: float | None  # only over subjects that have a previous score
    strongest_subjects: list[str]
    attention_subjects: list[str]  # current score below ATTENTION_THRESHOLD
    improved_subjects: list[str]
    declined_subjects: list[str]
    missing_information: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)

    def allowed_numbers(self) -> set[float]:
        """Every number the note may legitimately mention (used for fact-checking)."""
        nums = {self.average_current_score, float(ATTENTION_THRESHOLD), 100.0}
        if self.average_previous_score is not None:
            nums.add(self.average_previous_score)
        if self.attendance_percentage is not None:
            nums.add(self.attendance_percentage)
        for s in self.subjects:
            nums.add(s.current_score)
            if s.previous_score is not None:
                nums.add(s.previous_score)
            if s.change is not None:
                nums.add(abs(s.change))
        return nums


def _clean(value):
    """Convert pandas missing values to None and whole floats to ints-as-floats."""
    if value is None or pd.isna(value):
        return None
    return float(value) if isinstance(value, (int, float)) else str(value)


def _first_present(series: pd.Series):
    present = series.dropna()
    return _clean(present.iloc[0]) if not present.empty else None


def build_profile(rows: pd.DataFrame) -> StudentProfile:
    """Build one profile from all rows belonging to a single student."""
    missing: list[str] = []

    subjects = []
    for _, r in rows.sort_values("subject").iterrows():
        prev = _clean(r["previous_score"])
        change = _clean(r["score_change"])
        subjects.append(SubjectResult(
            subject=str(r["subject"]),
            current_score=float(r["current_score"]),
            previous_score=prev,
            change=change,
            trend=str(r["trend"]),
        ))
        if prev is None:
            missing.append(f"previous score for {r['subject']}")

    attendance_values = rows["attendance_percentage"].dropna().unique()
    attendance = float(attendance_values[0]) if len(attendance_values) == 1 else None
    if len(attendance_values) == 0:
        missing.append("attendance percentage")
    elif len(attendance_values) > 1:
        # Rows disagree; don't guess which is right.
        missing.append("a single consistent attendance figure (rows have different values)")

    parent_name = _first_present(rows["parent_name"])
    parent_email = _first_present(rows["parent_email"])
    if parent_name is None:
        missing.append("parent name")

    current = [s.current_score for s in subjects]
    previous = [s.previous_score for s in subjects if s.previous_score is not None]
    best = max(current)

    return StudentProfile(
        student_id=str(rows["student_id"].iloc[0]),
        student_name=str(rows["student_name"].iloc[0]),
        parent_name=parent_name,
        parent_email=parent_email,
        attendance_percentage=attendance,
        subjects=subjects,
        average_current_score=round(sum(current) / len(current), 1),
        average_previous_score=round(sum(previous) / len(previous), 1) if previous else None,
        strongest_subjects=[s.subject for s in subjects if s.current_score == best],
        attention_subjects=[s.subject for s in subjects if s.current_score < ATTENTION_THRESHOLD],
        improved_subjects=[s.subject for s in subjects if s.trend == "Improved"],
        declined_subjects=[s.subject for s in subjects if s.trend == "Declined"],
        missing_information=missing,
    )


def build_profiles(df: pd.DataFrame) -> list[StudentProfile]:
    """One profile per student_id, in student_id order."""
    return [build_profile(group) for _, group in df.groupby("student_id", sort=True)]
