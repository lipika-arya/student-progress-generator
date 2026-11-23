# Student Progress Note Generator

A Streamlit app that helps teachers turn student score data into personalised progress notes for parents.

1. **Upload** a CSV or Excel file of student scores. Columns, ranges, missing values and duplicates are checked.
2. **Generate** one AI-written note per student (OpenAI or Google Gemini). Every note is fact-checked against the data.
3. **Review** each note: edit, approve or reject. Decisions are saved in a local SQLite database.
4. **Export** approved notes to Excel.

## Setup (macOS)

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

Edit `.env`:
- `AI_PROVIDER=gemini` with a free `GEMINI_API_KEY` from https://aistudio.google.com, or
- `AI_PROVIDER=openai` with an `OPENAI_API_KEY`.

Never commit `.env`. It's listed in `.gitignore`.

## Run

```bash
streamlit run app.py
```

Try it with `data/sample/sample_students.csv`. The file `data/sample/sample_students_with_errors.csv` shows the validation messages.

## Input format

One row per student per subject:

| Column | Notes |
|---|---|
| `student_id`, `student_name`, `subject` | Required |
| `current_score`, `previous_score` | 0–100 |
| `attendance_percentage` | 0–100 |
| `parent_name`, `parent_email` | Used for the greeting and export. Emails are never sent to the AI. |

## Tests

```bash
pip install -r requirements-dev.txt
python -m pytest -q
```

## Project layout

```
app.py                 Streamlit UI (upload → generate → review → export)
src/config.py          Settings from .env (AI provider, model, database path)
src/data_loader.py     Reading and validating uploaded files
src/analyzer.py        Per-student facts: trends, strongest subject, subjects needing attention
src/note_generator.py  Prompt, AI call, fact-check, error handling
src/database.py        SQLite storage for notes and approval status
src/exporter.py        Excel export
tests/                 pytest tests (no API calls)
```

## Privacy

Only names, subjects, scores and attendance are sent to the AI provider. Free tiers, such as Gemini's, may use submitted data to improve their products, so use fictional data with them. Use a paid tier or a local model for real student records.
