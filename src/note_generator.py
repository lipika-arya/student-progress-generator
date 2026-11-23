"""Feature 2 — generate parent progress notes with an AI model (OpenAI or Gemini, via the OpenAI SDK).

Flow: StudentProfile (facts we computed) -> prompt -> structured JSON from the
model -> assembled note text -> automatic fact-check -> teacher review.
"""
import json
import re
from dataclasses import dataclass, field

import openai
from openai import OpenAI

from src import config
from src.analyzer import StudentProfile
from src.data_loader import ATTENTION_THRESHOLD

SYSTEM_PROMPT = f"""You write progress notes from a teacher to a student's parent or guardian.

You will receive a JSON fact sheet about ONE student. It is the only source of truth.

Rules you must follow:
- Use only the facts provided. Never invent scores, subjects, attendance, events or behaviour.
- Quote numbers exactly as given. Do not calculate new numbers (no new averages or percentages).
- Only say a subject improved if its "trend" is "Improved". If the trend is "Declined", say so
  gently. If it is "No change", say it held steady. If it is "Unknown", say there is no earlier
  score to compare with.
- A score below {ATTENTION_THRESHOLD} means the subject needs extra attention. Say this kindly and constructively.
- If anything is listed in "missing_information", acknowledge briefly that it was not available.
- Never compare the student with other students, classmates, or a class average.
- Never make psychological, emotional, medical or family assumptions (e.g. anxiety, ADHD,
  lack of motivation, problems at home). Describe results, not the child's character.
- Don't describe effort, engagement, focus or behaviour; the data only contains scores and attendance.
- Don't say "progress across subjects" or similar if any subject declined. Be specific instead.
- Don't describe actions the school or teacher is taking; you don't know them.
- Tone: warm, supportive, constructive, plain English a busy parent can read in a minute.
- Recommendations must be practical things a parent can do at home, linked to the data.
- Refer to the student by first name.
- The greeting ("Dear ...") and sign-off are added automatically. Do not write them yourself;
  start overall_summary directly with the update.
- Scores are out of 100. Write them as given (e.g. "38 out of 100" or "38"), and attendance as "81%".
"""

NOTE_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": [
        "overall_summary", "subject_progress", "strengths",
        "areas_for_improvement", "recommendations", "closing",
    ],
    "properties": {
        "overall_summary": {"type": "string", "description": "2-3 sentences on overall performance and attendance."},
        "subject_progress": {
            "type": "array",
            "description": "One entry per subject in the fact sheet, using its exact name.",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["subject", "comment"],
                "properties": {
                    "subject": {"type": "string"},
                    "comment": {"type": "string", "description": "1-2 sentences with the current and previous score."},
                },
            },
        },
        "strengths": {"type": "string", "description": "1-2 sentences."},
        "areas_for_improvement": {"type": "string", "description": "1-2 sentences."},
        "recommendations": {
            "type": "array", "items": {"type": "string"},
            "description": "Two or three short, actionable recommendations for parents.",
        },
        "closing": {"type": "string", "description": "One encouraging sentence."},
    },
}

# Words that suggest a comparison with other students or an unsupported assumption.
FORBIDDEN_PHRASES = [
    "class average", "classmates", "other students", "peers", "top of the class", "rank",
    "adhd", "anxiety", "anxious", "depress", "lazy", "unmotivated", "disorder", "diagnos",
]


class NoteGenerationError(Exception):
    """A note could not be generated. If fatal, stop generating for everyone."""

    def __init__(self, message: str, fatal: bool = False):
        super().__init__(message)
        self.fatal = fatal


@dataclass
class GeneratedNote:
    student_id: str
    text: str
    model: str
    warnings: list[str] = field(default_factory=list)  # fact-check results for the teacher


def get_client() -> OpenAI:
    if not config.has_api_key():
        raise NoteGenerationError(
            f"No {config.AI_PROVIDER_LABEL} API key configured. Add {config.AI_KEY_NAME} to .env (local) or to Secrets (Streamlit Cloud).",
            fatal=True,
        )
    # The SDK retries rate limits and temporary server errors with backoff.
    # base_url is None for OpenAI, or Gemini's OpenAI-compatible endpoint.
    return OpenAI(api_key=config.AI_API_KEY, base_url=config.AI_BASE_URL, max_retries=3, timeout=90)


def build_fact_sheet(profile: StudentProfile) -> dict:
    """What we send to the model. Contact details stay local; they aren't needed to write the note."""
    facts = profile.to_dict()
    for key in ("student_id", "parent_email"):
        facts.pop(key)
    facts["attention_threshold"] = ATTENTION_THRESHOLD
    return _whole_numbers_as_int(facts)


def _whole_numbers_as_int(value):
    """Send 38 rather than 38.0, so the model doesn't write "38.0" in the note."""
    if isinstance(value, float) and value.is_integer():
        return int(value)
    if isinstance(value, dict):
        return {k: _whole_numbers_as_int(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_whole_numbers_as_int(v) for v in value]
    return value


def greeting(profile: StudentProfile) -> str:
    if profile.parent_name:
        return f"Dear {profile.parent_name},"
    return f"Dear Parent/Guardian of {profile.student_name},"


def _fmt(n: float) -> str:
    return str(int(n)) if float(n).is_integer() else str(n)


def assemble_note(profile: StudentProfile, sections: dict) -> str:
    lines = [greeting(profile), "", sections["overall_summary"].strip(), "", "Subject progress"]
    for item in sections["subject_progress"]:
        lines.append(f"- {item['subject']}: {item['comment'].strip()}")
    lines += ["", "Strengths", sections["strengths"].strip(),
              "", "Areas for improvement", sections["areas_for_improvement"].strip(),
              "", "Recommendations"]
    lines += [f"{i}. {r.strip()}" for i, r in enumerate(sections["recommendations"], start=1)]
    lines += ["", sections["closing"].strip(), "", "Warm regards,", "Class Teacher"]
    return "\n".join(lines)


def fact_check(profile: StudentProfile, sections: dict) -> list[str]:
    """Flag anything in the note that the data doesn't support. Returns warnings for the teacher."""
    warnings = []
    by_subject = {s.subject: s for s in profile.subjects}

    returned = [item["subject"] for item in sections["subject_progress"]]
    for subj in returned:
        if subj not in by_subject:
            warnings.append(f"Mentions a subject not in the data: '{subj}'.")
    for subj in by_subject:
        if subj not in returned:
            warnings.append(f"Subject '{subj}' is missing from the subject progress section.")

    # Every number in the factual sections must come from the data.
    # (Recommendations are skipped: they may say things like "20 minutes a day".)
    allowed = profile.allowed_numbers()
    factual_text = " ".join(
        [sections["overall_summary"], sections["strengths"], sections["areas_for_improvement"]]
        + [item["comment"] for item in sections["subject_progress"]]
    )
    for num in sorted({float(n) for n in re.findall(r"\d+(?:\.\d+)?", factual_text)}):
        if num not in allowed:
            warnings.append(f"Contains the number {_fmt(num)}, which isn't in the student's data.")

    for item in sections["subject_progress"]:
        s = by_subject.get(item["subject"])
        if s and s.trend != "Improved" and re.search(r"\bimprov(ed|ement|ing)\b", item["comment"], re.I):
            # "needs improvement" is fine; claiming improvement is not.
            if not re.search(r"(need|room|area)s?\s+(for\s+)?improv", item["comment"], re.I):
                warnings.append(f"Says {s.subject} improved, but the data shows: {s.trend}.")

    full_text = json.dumps(sections).lower()
    for phrase in FORBIDDEN_PHRASES:
        if phrase in full_text:
            warnings.append(f"Contains '{phrase}' — check for comparisons or assumptions.")

    if not 2 <= len(sections["recommendations"]) <= 3:
        warnings.append(f"Has {len(sections['recommendations'])} recommendations (expected 2–3).")
    return warnings


QUOTA_CODES = {"insufficient_quota", "credit_balance_exhausted", "billing_hard_limit_reached"}


def _is_quota_error(exc: openai.APIStatusError) -> bool:
    """429 can mean 'slow down' (retry) or 'out of credits' (stop). Tell them apart."""
    body = exc.body if isinstance(exc.body, dict) else {}
    body = body.get("error", body)  # some responses nest the details under "error"
    if not isinstance(body, dict):
        body = {}
    values = {getattr(exc, "type", None), getattr(exc, "code", None), body.get("type"), body.get("code")}
    text = (str(exc) + str(exc.body)).lower().replace(" ", "")
    # OpenAI: insufficient_quota. Gemini free tier: a daily quota ("PerDay") is used up.
    return bool(values & QUOTA_CODES) or "insufficient_quota" in text or "perday" in text


def _request_kwargs(model: str) -> dict:
    # Reasoning models (gpt-5*, o*) don't accept a custom temperature.
    if model.startswith(("gpt-5", "o1", "o3", "o4")):
        return {}
    return {"temperature": 0.3}


def generate_note(profile: StudentProfile, client: OpenAI | None = None, model: str | None = None) -> GeneratedNote:
    client = client or get_client()
    model = model or config.AI_MODEL
    name = config.AI_PROVIDER_LABEL
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": json.dumps(build_fact_sheet(profile), indent=2)},
            ],
            response_format={
                "type": "json_schema",
                "json_schema": {"name": "progress_note", "strict": True, "schema": NOTE_SCHEMA},
            },
            **_request_kwargs(model),
        )
    except openai.AuthenticationError as exc:
        raise NoteGenerationError(f"{name} rejected the API key. Check {config.AI_KEY_NAME} in .env.", fatal=True) from exc
    except openai.PermissionDeniedError as exc:
        raise NoteGenerationError(f"This API key can't use model '{model}'.", fatal=True) from exc
    except openai.NotFoundError as exc:
        raise NoteGenerationError(f"Model '{model}' was not found on {name}. Check the model name in .env.", fatal=True) from exc
    except openai.RateLimitError as exc:
        if _is_quota_error(exc):
            raise NoteGenerationError(
                f"Your {name} account has no credits or free quota left. Check {config.AI_BILLING_URL}.", fatal=True,
            ) from exc
        raise NoteGenerationError("Rate limit reached even after retrying. Wait a minute and try again.") from exc
    except openai.BadRequestError as exc:
        if "api key" in str(exc).lower() or "api_key" in str(exc).lower():
            raise NoteGenerationError(f"{name} rejected the API key. Check {config.AI_KEY_NAME} in .env.", fatal=True) from exc
        raise NoteGenerationError(f"{name} couldn't process the request ({exc.status_code}).") from exc
    except openai.APITimeoutError as exc:
        raise NoteGenerationError(f"The request to {name} timed out.") from exc
    except openai.APIConnectionError as exc:
        raise NoteGenerationError(f"Couldn't connect to {name}. Check your internet connection.") from exc
    except openai.APIStatusError as exc:
        raise NoteGenerationError(f"{name} returned an error ({exc.status_code}). Try again shortly.") from exc

    message = response.choices[0].message
    if getattr(message, "refusal", None):
        raise NoteGenerationError("The model declined to write this note.")
    try:
        sections = json.loads(message.content)
    except (TypeError, json.JSONDecodeError) as exc:
        raise NoteGenerationError("The model returned an unreadable response. Try again.") from exc

    return GeneratedNote(
        student_id=profile.student_id,
        text=assemble_note(profile, sections),
        model=model,
        warnings=fact_check(profile, sections),
    )
