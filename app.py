"""Streamlit entry point. Run with: streamlit run app.py"""
import time

import pandas as pd
import streamlit as st

from src import config, database
from src.analyzer import build_profiles
from src.exporter import NothingToExportError, build_workbook, export_filename
from src.note_generator import NoteGenerationError, generate_note, get_client
from src.data_loader import (
    ATTENTION_THRESHOLD,
    REQUIRED_COLUMNS,
    DataLoadError,
    compute_summary,
    load_student_data,
)

st.set_page_config(page_title="Student Progress Notes", page_icon="📝", layout="wide")
st.title("📝 Student Progress Note Generator")

with st.sidebar:
    st.header("Setup")
    st.caption(f"AI provider: **{config.AI_PROVIDER_LABEL}** · model `{config.AI_MODEL}`")
    if config.has_api_key():
        st.success(f"{config.AI_PROVIDER_LABEL} API key loaded")
    else:
        st.warning(f"No {config.AI_PROVIDER_LABEL} API key found. Add {config.AI_KEY_NAME} to .env (local) or to Secrets (Streamlit Cloud).")

# ---- Feature 1: upload and validate ----
st.header("1. Upload student data")
with st.expander("Required columns"):
    st.markdown("\n".join(f"- `{c}`" for c in REQUIRED_COLUMNS))
    st.caption("One row per student per subject. Scores and attendance must be 0–100.")

uploaded = st.file_uploader("Choose a CSV or Excel file", type=["csv", "xlsx", "xls"])

if uploaded is None:
    st.info("Upload a file to begin. Try data/sample/sample_students.csv.")
    st.stop()

try:
    result = load_student_data(uploaded, uploaded.name)
except DataLoadError as err:
    st.error(f"❌ {err}")
    st.stop()

df = result.data
st.session_state["student_data"] = df  # used by later features

# Overall status message
problems = len(result.excluded) + len(result.duplicates)
if problems == 0 and result.warnings.empty:
    st.success(f"✅ Loaded all {result.total_rows} rows from {uploaded.name} with no issues.")
elif problems == 0:
    st.success(f"✅ Loaded all {result.total_rows} rows. {len(result.warnings)} warning(s) to review below.")
else:
    st.warning(
        f"⚠️ Loaded {len(df)} of {result.total_rows} rows. "
        f"{len(result.excluded)} excluded for errors, {len(result.duplicates)} duplicate(s) removed."
    )

# Summary metrics
summary = compute_summary(df)
c1, c2, c3, c4 = st.columns(4)
c1.metric("Total students", summary["total_students"])
c2.metric("Average current score", summary["average_current_score"])
avg_imp = summary["average_improvement"]
c3.metric("Average improvement", "—" if avg_imp is None else f"{avg_imp:+.1f}")
c4.metric(f"Needing attention (< {ATTENTION_THRESHOLD})", summary["students_needing_attention"])

# Data table with filters
st.subheader("Student data")
f1, f2 = st.columns([2, 1])
subjects = f1.multiselect("Subject", sorted(df["subject"].unique()), default=sorted(df["subject"].unique()))
only_attention = f2.checkbox("Only students needing attention")

view = df[df["subject"].isin(subjects)]
if only_attention:
    view = view[view["needs_attention"]]

st.dataframe(
    view.drop(columns=["row"]),
    width="stretch",
    hide_index=True,
    column_config={
        "student_id": "ID",
        "student_name": "Student",
        "parent_name": "Parent",
        "parent_email": "Parent email",
        "subject": "Subject",
        "current_score": st.column_config.ProgressColumn("Current", min_value=0, max_value=100, format="%d"),
        "previous_score": st.column_config.NumberColumn("Previous", format="%d"),
        "score_change": st.column_config.NumberColumn("Change", format="%+d"),
        "trend": "Trend",
        "attendance_percentage": st.column_config.ProgressColumn("Attendance %", min_value=0, max_value=100, format="%d%%"),
        "needs_attention": st.column_config.CheckboxColumn("Needs attention"),
    },
)

# Data quality details
if not result.excluded.empty:
    with st.expander(f"❌ Excluded rows ({len(result.excluded)})", expanded=True):
        st.caption("These rows had errors and were not loaded. Fix them in your file and upload again.")
        st.dataframe(result.excluded[["row", "student_id", "student_name", "subject", "reason"]], hide_index=True, width="stretch")

if not result.duplicates.empty:
    with st.expander(f"🔁 Duplicate records removed ({len(result.duplicates)})"):
        st.caption("Same student and subject appeared more than once. The first entry was kept.")
        st.dataframe(result.duplicates[["row", "student_id", "student_name", "subject"]], hide_index=True, width="stretch")

if not result.warnings.empty:
    with st.expander(f"⚠️ Warnings ({len(result.warnings)})"):
        st.caption("These rows were loaded, but some details are missing or look wrong.")
        st.dataframe(result.warnings, hide_index=True, width="stretch")

if not result.missing_counts.empty:
    with st.expander("Missing values by column"):
        st.dataframe(result.missing_counts.rename("blank cells"), width="stretch")

# ---- Feature 2: AI progress notes ----
st.divider()
st.header("2. Generate AI Progress Notes")

profiles = {p.student_id: p for p in build_profiles(df)}
STATUS_ICONS = {"Pending": "🕓", "Approved": "✅", "Rejected": "❌"}


def note_key(student_id: str) -> str:
    return f"note_text_{student_id}"


def generate_and_save(profile, client) -> list[str]:
    note = generate_note(profile, client=client)
    database.save_generated_note(profile, note)
    st.session_state.pop(note_key(profile.student_id), None)  # show the new text in the editor
    return note.warnings


def save_edits(student_id: str):
    database.update_note(student_id, note_text=st.session_state[note_key(student_id)])
    st.session_state["flash"] = ("success", f"Saved edits for {profiles[student_id].student_name}.")


def set_status(student_id: str, status: str):
    # Approving or rejecting also saves whatever is currently in the editor.
    database.update_note(student_id, note_text=st.session_state[note_key(student_id)], status=status)
    st.session_state["flash"] = ("success", f"{profiles[student_id].student_name}: marked {status}.")


def regenerate_one(student_id: str):
    try:
        generate_and_save(profiles[student_id], get_client())
        st.session_state["flash"] = ("success", f"Regenerated the note for {profiles[student_id].student_name}.")
    except NoteGenerationError as err:
        st.session_state["flash"] = ("error", f"{profiles[student_id].student_name}: {err}")


st.caption(
    f"Writes one consolidated note per student ({len(profiles)} students) using {config.AI_PROVIDER_LABEL} (`{config.AI_MODEL}`). "
    f"Only the scores, subjects, attendance and names from your file are sent to {config.AI_PROVIDER_LABEL}. Parent emails stay on this computer. "
    "Nothing is emailed or messaged."
)

existing = {n["student_id"]: n for n in database.get_notes() if n["student_id"] in profiles}
keep_approved = st.checkbox(
    "Keep notes that are already approved (don't regenerate them)", value=True,
    disabled=not any(n["status"] == "Approved" for n in existing.values()),
)

missing = [p for sid, p in profiles.items() if sid not in existing]
col_all, col_missing = st.columns(2)
generate_all = col_all.button(f"✨ Generate notes for all {len(profiles)} students", type="primary", width="stretch")
# After a partial run (e.g. the AI service was busy), fill the gaps without redoing finished notes.
generate_missing = (
    col_missing.button(f"🔁 Generate the {len(missing)} missing note(s)", width="stretch")
    if existing and missing else False
)

targets = None
if generate_all:
    targets = [p for sid, p in profiles.items()
               if not (keep_approved and existing.get(sid, {}).get("status") == "Approved")]
elif generate_missing:
    targets = missing

if targets is not None:
    progress = st.progress(0.0, text="Starting…")
    failures, flagged = [], 0
    try:
        client = get_client()
        min_gap = 60 / config.REQUESTS_PER_MINUTE if config.REQUESTS_PER_MINUTE else 0
        last_start = 0.0
        for i, profile in enumerate(targets, start=1):
            wait = min_gap - (time.monotonic() - last_start)
            if wait > 0:  # free tiers allow only a few requests per minute
                progress.progress((i - 1) / len(targets), text=f"Pausing {wait:.0f}s to stay within the free-tier limit…")
                time.sleep(wait)
            last_start = time.monotonic()
            progress.progress((i - 1) / len(targets), text=f"Writing note {i} of {len(targets)}: {profile.student_name}…")
            try:
                if generate_and_save(profile, client):
                    flagged += 1
            except NoteGenerationError as err:
                if err.fatal:
                    raise
                failures.append(f"{profile.student_name}: {err}")
        progress.progress(1.0, text="Done")
        done = len(targets) - len(failures)
        st.session_state["flash"] = (
            "success" if not failures else "warning",
            f"Generated {done} of {len(targets)} notes."
            + (f" {flagged} need a closer look (see ⚠️ below)." if flagged else "")
            + ("" if not failures else " Failed: " + "; ".join(failures)),
        )
    except NoteGenerationError as err:
        progress.empty()
        st.session_state["flash"] = ("error", f"Generation stopped: {err}")
    st.rerun()

flash = st.session_state.pop("flash", None)
if flash:
    getattr(st, flash[0])(flash[1])

notes = [n for n in database.get_notes() if n["student_id"] in profiles]
if not notes:
    st.info("No notes yet. Click the button above to generate them.")
    st.stop()

counts = {s: sum(n["status"] == s for n in notes) for s in database.STATUSES}
m1, m2, m3, m4 = st.columns(4)
m1.metric("Notes", f"{len(notes)} / {len(profiles)}")
m2.metric("🕓 Pending", counts["Pending"])
m3.metric("✅ Approved", counts["Approved"])
m4.metric("❌ Rejected", counts["Rejected"])

status_filter = st.segmented_control("Show", ["All", *database.STATUSES], default="All")

for note in notes:
    if status_filter not in (None, "All") and note["status"] != status_filter:
        continue
    sid, profile = note["student_id"], profiles[note["student_id"]]
    label = f"{STATUS_ICONS[note['status']]} {profile.student_name} ({sid}) — {note['status']}"
    if note["warnings"]:
        label += f"  ·  ⚠️ {len(note['warnings'])} check(s)"

    with st.expander(label):
        for w in note["warnings"]:
            st.warning(f"Fact-check: {w}")

        left, right = st.columns([3, 2])
        with right:
            st.caption("Data from your file — compare with the note")
            st.dataframe(
                pd.DataFrame([{
                    "Subject": s.subject, "Current": s.current_score,
                    "Previous": s.previous_score, "Change": s.change, "Trend": s.trend,
                } for s in profile.subjects]),
                hide_index=True, width="stretch",
                column_config={
                    "Current": st.column_config.NumberColumn(format="%d"),
                    "Previous": st.column_config.NumberColumn(format="%d"),
                    "Change": st.column_config.NumberColumn(format="%+d"),
                },
            )
            attendance = "not available" if profile.attendance_percentage is None else f"{profile.attendance_percentage:g}%"
            st.caption(f"Attendance: {attendance} · Parent: {profile.parent_name or '—'} ({profile.parent_email or 'no email'})")

        with left:
            if note_key(sid) not in st.session_state:
                st.session_state[note_key(sid)] = note["note_text"]
            st.text_area("Note (editable)", key=note_key(sid), height=420)

        b1, b2, b3, b4 = st.columns(4)
        b1.button("💾 Save edits", key=f"save_{sid}", on_click=save_edits, args=(sid,), width="stretch")
        b2.button("✅ Approve", key=f"approve_{sid}", on_click=set_status, args=(sid, "Approved"),
                  disabled=note["status"] == "Approved", width="stretch")
        b3.button("❌ Reject", key=f"reject_{sid}", on_click=set_status, args=(sid, "Rejected"),
                  disabled=note["status"] == "Rejected", width="stretch")
        b4.button("🔄 Regenerate", key=f"regen_{sid}", on_click=regenerate_one, args=(sid,), width="stretch")
        st.caption(f"Generated {note['generated_at']} with {note['model']} · last updated {note['updated_at']}")

# ---- Feature 3: export approved notes ----
st.divider()
st.header("3. Export Approved Notes")

approved = [n for n in notes if n["status"] == "Approved"]
if not approved:
    st.info("No approved notes yet. Approve notes above, then download them here as an Excel file.")
else:
    not_ready = len(profiles) - len(approved)
    st.write(f"**{len(approved)} of {len(profiles)}** students have an approved note and will be exported.")
    if not_ready:
        st.caption(f"{not_ready} student(s) are pending, rejected or have no note; they're listed on the Summary sheet.")

    st.dataframe(
        pd.DataFrame([{
            "Student": profiles[n["student_id"]].student_name,
            "Parent": profiles[n["student_id"]].parent_name or "—",
            "Parent email": profiles[n["student_id"]].parent_email or "—",
            "Approved on": (n.get("approved_at") or "").replace("T", " "),
        } for n in approved]),
        hide_index=True, width="stretch",
    )

    try:
        st.download_button(
            "⬇️ Download approved notes (Excel)",
            data=build_workbook(notes, profiles),
            file_name=export_filename(),
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            type="primary",
        )
    except NothingToExportError as err:
        st.info(str(err))
