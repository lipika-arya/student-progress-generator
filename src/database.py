"""SQLite storage for generated notes, teacher edits and approval status.

Saved to disk so edits and approvals survive page reloads and app restarts.
One note per student_id; regenerating a student's note replaces it and resets it to Pending.
"""
import json
import sqlite3
from contextlib import closing
from datetime import datetime
from pathlib import Path

from src import config

STATUSES = ("Pending", "Approved", "Rejected")

SCHEMA = """
CREATE TABLE IF NOT EXISTS notes (
    student_id   TEXT PRIMARY KEY,
    student_name TEXT NOT NULL,
    parent_name  TEXT,
    parent_email TEXT,
    note_text    TEXT NOT NULL,
    status       TEXT NOT NULL DEFAULT 'Pending',
    warnings     TEXT NOT NULL DEFAULT '[]',
    model        TEXT,
    generated_at TEXT NOT NULL,
    updated_at   TEXT NOT NULL,
    approved_at  TEXT
)
"""

# Columns added after the first version; added to existing databases on connect.
LATER_COLUMNS = {"approved_at": "TEXT"}


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def connect(db_path: Path | None = None) -> sqlite3.Connection:
    path = Path(db_path or config.DATABASE_PATH)
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    conn.execute(SCHEMA)
    existing = {row["name"] for row in conn.execute("PRAGMA table_info(notes)")}
    for name, col_type in LATER_COLUMNS.items():
        if name not in existing:
            conn.execute(f"ALTER TABLE notes ADD COLUMN {name} {col_type}")
    # Notes approved before approved_at existed: use their last update time.
    conn.execute("UPDATE notes SET approved_at=updated_at WHERE status='Approved' AND approved_at IS NULL")
    conn.commit()
    return conn


def save_generated_note(profile, note, db_path=None) -> None:
    """Insert or replace a freshly generated note (status goes back to Pending)."""
    now = _now()
    with closing(connect(db_path)) as conn, conn:
        conn.execute(
            """INSERT INTO notes (student_id, student_name, parent_name, parent_email, note_text,
                                  status, warnings, model, generated_at, updated_at)
               VALUES (?, ?, ?, ?, ?, 'Pending', ?, ?, ?, ?)
               ON CONFLICT(student_id) DO UPDATE SET
                 student_name=excluded.student_name, parent_name=excluded.parent_name,
                 parent_email=excluded.parent_email, note_text=excluded.note_text,
                 status='Pending', approved_at=NULL, warnings=excluded.warnings, model=excluded.model,
                 generated_at=excluded.generated_at, updated_at=excluded.updated_at""",
            (profile.student_id, profile.student_name, profile.parent_name, profile.parent_email,
             note.text, json.dumps(note.warnings), note.model, now, now),
        )


def update_note(student_id: str, note_text: str | None = None, status: str | None = None, db_path=None) -> None:
    """Save teacher edits and/or change the approval status."""
    if status is not None and status not in STATUSES:
        raise ValueError(f"Invalid status '{status}'. Use one of {STATUSES}.")
    with closing(connect(db_path)) as conn, conn:
        if note_text is not None:
            conn.execute("UPDATE notes SET note_text=?, updated_at=? WHERE student_id=?",
                         (note_text, _now(), student_id))
        if status is not None:
            now = _now()
            conn.execute("UPDATE notes SET status=?, updated_at=?, approved_at=? WHERE student_id=?",
                         (status, now, now if status == "Approved" else None, student_id))


def get_notes(db_path=None) -> list[dict]:
    with closing(connect(db_path)) as conn:
        rows = conn.execute("SELECT * FROM notes ORDER BY student_id").fetchall()
    notes = [dict(r) for r in rows]
    for n in notes:
        n["warnings"] = json.loads(n["warnings"])
    return notes


def clear_notes(db_path=None) -> None:
    with closing(connect(db_path)) as conn, conn:
        conn.execute("DELETE FROM notes")
