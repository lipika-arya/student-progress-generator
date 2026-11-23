from io import BytesIO
from pathlib import Path

import pytest
from openpyxl import load_workbook

from src import database
from src.analyzer import build_profiles
from src.data_loader import load_student_data
from src.exporter import NothingToExportError, build_workbook, export_filename
from src.note_generator import GeneratedNote

SAMPLE = Path(__file__).resolve().parent.parent / "data" / "sample" / "sample_students.csv"


@pytest.fixture
def setup(tmp_path):
    with open(SAMPLE, "rb") as f:
        profiles = {p.student_id: p for p in build_profiles(load_student_data(f, SAMPLE.name).data)}
    db = tmp_path / "notes.db"
    for p in profiles.values():
        note = GeneratedNote(student_id=p.student_id, text=f"Dear {p.parent_name},\nNote for {p.student_name}.", model="test-model")
        database.save_generated_note(p, note, db_path=db)
    return profiles, db


def read(xlsx: bytes):
    return load_workbook(BytesIO(xlsx))


def test_only_approved_notes_are_exported_with_edits(setup):
    profiles, db = setup
    database.update_note("S003", note_text="Edited by teacher", status="Approved", db_path=db)
    database.update_note("S001", status="Approved", db_path=db)
    database.update_note("S002", status="Rejected", db_path=db)

    wb = read(build_workbook(database.get_notes(db_path=db), profiles))
    ws = wb["Approved Notes"]
    rows = list(ws.iter_rows(min_row=2, values_only=True))
    assert [r[0] for r in rows] == ["S001", "S003"]

    liam = rows[1]
    assert liam[1] == "Liam Johnson" and liam[2] == "Sarah Johnson" and liam[3] == "sarah.johnson@example.com"
    assert liam[4] == "English: 55 (was 58)\nMathematics: 38 (was 52)"
    assert liam[5] == 81
    assert liam[6] == "Edited by teacher"  # teacher's edit, not the original
    assert liam[7]  # approved date filled in


def test_summary_lists_students_not_included(setup):
    profiles, db = setup
    database.update_note("S001", status="Approved", db_path=db)
    database.update_note("S002", status="Rejected", db_path=db)

    summary = read(build_workbook(database.get_notes(db_path=db), profiles))["Summary"]
    values = {r[0]: r[1] for r in summary.iter_rows(min_row=2, max_row=7, values_only=True)}
    assert values["Approved (in this file)"] == 1
    assert values["Rejected"] == 1
    assert values["Pending review"] == 8
    listed = [r for r in summary.iter_rows(min_row=10, values_only=True) if r[0]]  # row 9 is the header
    assert ("S002", "Maya Chen", "Rejected") in listed
    assert len(listed) == 9


def test_notes_for_students_not_in_upload_are_ignored(setup):
    profiles, db = setup
    database.update_note("S001", status="Approved", db_path=db)
    database.update_note("S002", status="Approved", db_path=db)
    only_s002 = {"S002": profiles["S002"]}  # e.g. a different file was uploaded
    rows = list(read(build_workbook(database.get_notes(db_path=db), only_s002))["Approved Notes"].iter_rows(min_row=2, values_only=True))
    assert [r[0] for r in rows] == ["S002"]


def test_nothing_approved_raises(setup):
    profiles, db = setup
    with pytest.raises(NothingToExportError):
        build_workbook(database.get_notes(db_path=db), profiles)


def test_formula_like_text_stays_text(setup):
    profiles, db = setup
    database.update_note("S001", note_text='=HYPERLINK("http://evil.example","click")', status="Approved", db_path=db)
    cell = read(build_workbook(database.get_notes(db_path=db), profiles))["Approved Notes"]["G2"]
    assert cell.data_type == "s" and cell.value.startswith("=HYPERLINK")


def test_regenerating_clears_approval(setup):
    profiles, db = setup
    database.update_note("S001", status="Approved", db_path=db)
    database.save_generated_note(profiles["S001"], GeneratedNote("S001", "New text", "m"), db_path=db)
    note = next(n for n in database.get_notes(db_path=db) if n["student_id"] == "S001")
    assert note["status"] == "Pending" and note["approved_at"] is None


def test_filename_has_date():
    assert export_filename().startswith("approved_progress_notes_") and export_filename().endswith(".xlsx")
