import json
from pathlib import Path
from types import SimpleNamespace

import httpx2 as httpx
import openai
import pytest

from src import database
from src.analyzer import build_profiles
from src.data_loader import load_student_data
from src.note_generator import (
    NoteGenerationError, build_fact_sheet, fact_check, generate_note,
)

SAMPLE = Path(__file__).resolve().parent.parent / "data" / "sample" / "sample_students.csv"


@pytest.fixture
def profiles():
    with open(SAMPLE, "rb") as f:
        df = load_student_data(f, SAMPLE.name).data
    return {p.student_id: p for p in build_profiles(df)}


def good_sections(profile):
    return {
        "overall_summary": f"{profile.student_name} has an average of {profile.average_current_score}.",
        "subject_progress": [
            {"subject": s.subject, "comment": f"Scored {s.current_score:g}, previously {s.previous_score:g}."}
            for s in profile.subjects
        ],
        "strengths": "Consistent effort.",
        "areas_for_improvement": "Keep practising.",
        "recommendations": ["Read for 20 minutes a day.", "Review homework together."],
        "closing": "We look forward to more progress.",
    }


class FakeClient:
    def __init__(self, content=None, error=None):
        self.content, self.error, self.calls = content, error, []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kwargs):
        self.calls.append(kwargs)
        if self.error:
            raise self.error
        msg = SimpleNamespace(content=json.dumps(self.content), refusal=None)
        return SimpleNamespace(choices=[SimpleNamespace(message=msg)])


# ---- analyzer ----

def test_one_profile_per_student(profiles):
    assert len(profiles) == 10
    assert all(len(p.subjects) == 2 for p in profiles.values())


def test_profile_facts_match_csv(profiles):
    liam = profiles["S003"]
    scores = {s.subject: (s.current_score, s.previous_score, s.change, s.trend) for s in liam.subjects}
    assert scores == {"English": (55, 58, -3, "Declined"), "Mathematics": (38, 52, -14, "Declined")}
    assert liam.attention_subjects == ["Mathematics"]
    assert liam.strongest_subjects == ["English"]
    assert liam.attendance_percentage == 81
    assert liam.improved_subjects == []


def test_tied_strongest_and_no_change(profiles):
    sofia = profiles["S004"]
    maths = next(s for s in sofia.subjects if s.subject == "Mathematics")
    assert maths.trend == "No change"
    assert sofia.strongest_subjects == ["English"]


# ---- note generation ----

def test_fact_sheet_excludes_contact_details(profiles):
    facts = build_fact_sheet(profiles["S001"])
    assert "parent_email" not in facts and "student_id" not in facts


def test_generate_note_assembles_all_sections(profiles):
    p = profiles["S001"]
    client = FakeClient(content=good_sections(p))
    note = generate_note(p, client=client, model="gpt-4o-mini")
    assert note.text.startswith("Dear Rohan Mehta,")
    for heading in ["Subject progress", "Strengths", "Areas for improvement", "Recommendations"]:
        assert heading in note.text
    assert note.warnings == []
    assert client.calls[0]["response_format"]["type"] == "json_schema"


def test_fact_check_catches_invented_number_and_false_improvement(profiles):
    p = profiles["S003"]  # both subjects declined
    sections = good_sections(p)
    sections["subject_progress"][1]["comment"] = "Mathematics improved to 72."
    warnings = fact_check(p, sections)
    assert any("72" in w for w in warnings)
    assert any("Mathematics improved" in w for w in warnings)


def test_fact_check_allows_needs_improvement(profiles):
    p = profiles["S003"]
    sections = good_sections(p)
    sections["subject_progress"][1]["comment"] = "Mathematics (38) needs improvement."
    assert fact_check(p, sections) == []


def test_fact_check_catches_comparisons_and_missing_subject(profiles):
    p = profiles["S001"]
    sections = good_sections(p)
    sections["strengths"] = "Ahead of his classmates."
    sections["subject_progress"] = sections["subject_progress"][:1]
    warnings = " ".join(fact_check(p, sections))
    assert "classmates" in warnings and "missing from the subject progress" in warnings


def _status_error(cls, status, message="error", body=None):
    request = httpx.Request("POST", "https://api.openai.com/v1/chat/completions")
    return cls(message, response=httpx.Response(status, request=request), body=body)


def test_auth_error_is_fatal(profiles):
    client = FakeClient(error=_status_error(openai.AuthenticationError, 401))
    with pytest.raises(NoteGenerationError) as exc:
        generate_note(profiles["S001"], client=client, model="gpt-4o-mini")
    assert exc.value.fatal


def test_rate_limit_is_not_fatal_but_quota_is(profiles):
    client = FakeClient(error=_status_error(openai.RateLimitError, 429))
    with pytest.raises(NoteGenerationError) as exc:
        generate_note(profiles["S001"], client=client, model="gpt-4o-mini")
    assert not exc.value.fatal and "Rate limit" in str(exc.value)

    # Same shape as the real "no credits" response: the message doesn't mention the code.
    quota_body = {"message": "You have no credits remaining.", "type": "insufficient_quota",
                  "param": None, "code": "credit_balance_exhausted"}
    client = FakeClient(error=_status_error(openai.RateLimitError, 429, "You have no credits remaining.", quota_body))
    with pytest.raises(NoteGenerationError) as exc:
        generate_note(profiles["S001"], client=client, model="gpt-4o-mini")
    assert exc.value.fatal and "no credits" in str(exc.value)


# ---- database ----

def test_database_roundtrip(profiles, tmp_path):
    db = tmp_path / "notes.db"
    p = profiles["S001"]
    note = generate_note(p, client=FakeClient(content=good_sections(p)), model="gpt-4o-mini")
    database.save_generated_note(p, note, db_path=db)
    database.update_note("S001", note_text="Edited text", status="Approved", db_path=db)
    [saved] = database.get_notes(db_path=db)
    assert saved["note_text"] == "Edited text" and saved["status"] == "Approved"

    database.save_generated_note(p, note, db_path=db)  # regenerating resets to Pending
    assert database.get_notes(db_path=db)[0]["status"] == "Pending"

    with pytest.raises(ValueError):
        database.update_note("S001", status="Maybe", db_path=db)


def test_gemini_daily_quota_is_fatal(profiles):
    body = [{"error": {"code": 429, "status": "RESOURCE_EXHAUSTED",
             "message": "Quota exceeded for metric: generate_content_free_tier_requests, quotaId: GenerateRequestsPerDayPerProjectPerModel-FreeTier"}}]
    client = FakeClient(error=_status_error(openai.RateLimitError, 429, "Quota exceeded", body))
    with pytest.raises(NoteGenerationError) as exc:
        generate_note(profiles["S001"], client=client, model="gemini-2.5-flash")
    assert exc.value.fatal


def test_gemini_per_minute_limit_is_retryable(profiles):
    body = [{"error": {"code": 429, "status": "RESOURCE_EXHAUSTED",
             "message": "quotaId: GenerateRequestsPerMinutePerProjectPerModel-FreeTier"}}]
    client = FakeClient(error=_status_error(openai.RateLimitError, 429, "Quota exceeded", body))
    with pytest.raises(NoteGenerationError) as exc:
        generate_note(profiles["S001"], client=client, model="gemini-2.5-flash")
    assert not exc.value.fatal


def test_gemini_bad_key_is_fatal(profiles):
    client = FakeClient(error=_status_error(openai.BadRequestError, 400, "API key not valid. Please pass a valid API key."))
    with pytest.raises(NoteGenerationError) as exc:
        generate_note(profiles["S001"], client=client, model="gemini-2.5-flash")
    assert exc.value.fatal
