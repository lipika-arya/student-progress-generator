"""Feature 3 — export approved notes to an Excel file.

Builds the workbook in memory so Streamlit can offer it as a download.
Sheet 1 "Approved Notes": one row per approved student, ready to send to parents.
Sheet 2 "Summary": counts, plus students not included and why.
"""
from datetime import datetime
from io import BytesIO

from openpyxl import Workbook
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.utils import get_column_letter

from src.analyzer import StudentProfile

NOTE_COLUMNS = [
    # (header, width)
    ("Student ID", 12),
    ("Student Name", 20),
    ("Parent Name", 20),
    ("Parent Email", 30),
    ("Scores (current vs previous)", 34),
    ("Attendance %", 13),
    ("Progress Note", 90),
    ("Approved On", 19),
    ("Written By", 22),
]
HEADER_FONT = Font(bold=True, color="FFFFFF")
HEADER_FILL = PatternFill("solid", fgColor="305496")
TOP_WRAP = Alignment(vertical="top", wrap_text=True)


class NothingToExportError(Exception):
    pass


def _fmt(n) -> str:
    return "—" if n is None else (str(int(n)) if float(n).is_integer() else str(n))


def scores_summary(profile: StudentProfile) -> str:
    """e.g. 'English: 55 (was 58)\\nMathematics: 38 (was 52)'"""
    return "\n".join(
        f"{s.subject}: {_fmt(s.current_score)} (was {_fmt(s.previous_score)})" for s in profile.subjects
    )


FORMULA_STARTS = ("=", "+", "-", "@")


def _write_row(ws, row_idx: int, values: list) -> None:
    for col_idx, value in enumerate(values, start=1):
        cell = ws.cell(row=row_idx, column=col_idx, value=value)
        cell.alignment = TOP_WRAP
        # Stop Excel treating text like '=HYPERLINK(...)' from an uploaded file as a formula.
        if isinstance(value, str) and value.startswith(FORMULA_STARTS):
            cell.data_type = "s"
            cell.quotePrefix = True


def _write_header(ws, headers: list[tuple[str, int]]) -> None:
    for col_idx, (title, width) in enumerate(headers, start=1):
        cell = ws.cell(row=1, column=col_idx, value=title)
        cell.font, cell.fill = HEADER_FONT, HEADER_FILL
        cell.alignment = Alignment(vertical="center")
        ws.column_dimensions[get_column_letter(col_idx)].width = width
    ws.freeze_panes = "A2"


def _format_date(iso: str | None) -> str:
    return datetime.fromisoformat(iso).strftime("%d %b %Y %H:%M") if iso else ""


def build_workbook(notes: list[dict], profiles: dict[str, StudentProfile]) -> bytes:
    """Return the .xlsx file as bytes. `notes` come from database.get_notes()."""
    relevant = [n for n in notes if n["student_id"] in profiles]
    approved = [n for n in relevant if n["status"] == "Approved"]
    if not approved:
        raise NothingToExportError("No approved notes to export. Approve at least one note first.")

    wb = Workbook()
    ws = wb.active
    ws.title = "Approved Notes"
    _write_header(ws, NOTE_COLUMNS)
    for row_idx, note in enumerate(approved, start=2):
        p = profiles[note["student_id"]]
        _write_row(ws, row_idx, [
            p.student_id, p.student_name, p.parent_name or "", p.parent_email or "",
            scores_summary(p),
            p.attendance_percentage if p.attendance_percentage is not None else "",
            note["note_text"],  # includes any teacher edits
            _format_date(note.get("approved_at")),
            note.get("model") or "",
        ])
    ws.auto_filter.ref = ws.dimensions

    # Summary sheet
    summary = wb.create_sheet("Summary")
    by_id = {n["student_id"]: n for n in relevant}
    status_of = {sid: by_id[sid]["status"] if sid in by_id else "No note yet" for sid in profiles}
    rows = [
        ("Exported on", datetime.now().strftime("%d %b %Y %H:%M")),
        ("Students in uploaded file", len(profiles)),
        ("Approved (in this file)", sum(s == "Approved" for s in status_of.values())),
        ("Pending review", sum(s == "Pending" for s in status_of.values())),
        ("Rejected", sum(s == "Rejected" for s in status_of.values())),
        ("No note generated", sum(s == "No note yet" for s in status_of.values())),
    ]
    _write_header(summary, [("Item", 28), ("Value", 22), ("", 30)])
    for i, (label, value) in enumerate(rows, start=2):
        _write_row(summary, i, [label, value])

    not_included = [(sid, profiles[sid].student_name, st) for sid, st in status_of.items() if st != "Approved"]
    if not_included:
        start = len(rows) + 3
        for col_idx, title in enumerate(["Not included: Student ID", "Student Name", "Status"], start=1):
            cell = summary.cell(row=start, column=col_idx, value=title)
            cell.font, cell.fill = HEADER_FONT, HEADER_FILL
        for i, values in enumerate(not_included, start=start + 1):
            _write_row(summary, i, list(values))

    buffer = BytesIO()
    wb.save(buffer)
    return buffer.getvalue()


def export_filename(now: datetime | None = None) -> str:
    return f"approved_progress_notes_{(now or datetime.now()).strftime('%Y-%m-%d_%H%M')}.xlsx"
