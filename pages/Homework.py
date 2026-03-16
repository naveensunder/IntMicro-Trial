"""
pages/Homework.py — Renders the selected homework's questions.
"""
import streamlit as st
import datetime
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from db import parse_deadline, get_student_submissions, get_homework_configs
from ui import inject_css, page_header, COLORS, banner
from question_engine import (
    get_shuffled_questions, render_question, get_hw_summary, ALL_HW_CONFIGS
)

st.set_page_config(
    page_title="Homework — Microeconomics",
    page_icon="📝",
    layout="centered",
    initial_sidebar_state="expanded",
)

inject_css()

# ── Auth guard ────────────────────────────────────────────────────────────────
if not st.session_state.get("authenticated"):
    st.warning("Please sign in first.")
    if st.button("Go to sign in"):
        st.switch_page("Home.py")
    st.stop()

email = st.session_state["student_email"]
name  = st.session_state["student_name"]
hw_id = st.session_state.get("current_hw")

if not hw_id:
    st.warning("No homework selected.")
    if st.button("← Back to Dashboard"):
        st.switch_page("pages/Dashboard.py")
    st.stop()

# ── Load configs ──────────────────────────────────────────────────────────────
hw_configs = st.session_state.get("hw_configs", [])
if not hw_configs:
    hw_configs = get_homework_configs()
    st.session_state["hw_configs"] = hw_configs

hw_cfg = next((c for c in hw_configs if c.get("HW_ID") == hw_id), {})
if not hw_cfg:
    st.error("Homework configuration not found.")
    st.stop()

title         = hw_cfg.get("Title", hw_id)
deadline_str  = hw_cfg.get("Deadline", "2099-12-31 23:59")
grace_minutes = int(hw_cfg.get("Grace_Minutes", 15))
announce      = hw_cfg.get("Announcement", "")

past_hard, past_soft, dl_dt, dl_grace = parse_deadline(deadline_str, grace_minutes)
grace_active = past_soft and not past_hard

# ── Load submissions ──────────────────────────────────────────────────────────
submissions = st.session_state.get("submissions", {})

# ── Track reloads per question ────────────────────────────────────────────────
for q in get_shuffled_questions(hw_id, email):
    k = f"reloads_{q['q_id']}"
    st.session_state[k] = st.session_state.get(k, 0) + 1

# ── Header ─────────────────────────────────────────────────────────────────────
try:
    dl_display = dl_dt.strftime("%d %b %Y at %H:%M")
except Exception:
    dl_display = deadline_str

page_header(
    "Intermediate Microeconomics",
    title,
    f"{name} · Deadline: {dl_display}"
)

# ── Announcement ─────────────────────────────────────────────────────────────
if announce:
    banner(f"📢 {announce}", "warning")

# ── Deadline banner ───────────────────────────────────────────────────────────
if past_hard:
    banner("🔒 This assignment is now closed. Submissions are no longer accepted.", "error")
elif grace_active:
    banner("⏳ The deadline has passed but you are within the grace period. Submit now.", "warning")
else:
    now = datetime.datetime.now()
    remaining = dl_grace - now
    days  = remaining.days
    hours = remaining.seconds // 3600
    mins  = (remaining.seconds % 3600) // 60
    if days == 0 and hours < 6:
        banner(f"⚠ Deadline approaching — {hours}h {mins}m remaining.", "warning")

# ── Progress indicator ────────────────────────────────────────────────────────
summary = get_hw_summary(hw_id, email, submissions)
pct = int(summary["n_submitted"] / max(summary["n_total"], 1) * 100)

st.markdown(f"""
<div class="progress-wrap" style="margin-bottom:1.5rem;">
  <div class="progress-label">Assignment Progress</div>
  <div class="progress-bar-outer">
    <div class="progress-bar-inner" style="width:{pct}%;"></div>
  </div>
  <div class="progress-text">
    {summary['n_submitted']} of {summary['n_total']} questions submitted
    {f" · Score so far: {summary['total_score']}/{summary['total_max']}" if summary['n_submitted'] > 0 else ""}
  </div>
</div>
""", unsafe_allow_html=True)

# ── Questions ─────────────────────────────────────────────────────────────────
questions = get_shuffled_questions(hw_id, email)

if not questions:
    st.info("No questions found for this assignment.")
else:
    for i, q_cfg in enumerate(questions):
        if i > 0:
            st.markdown("<hr style='border:none;border-top:1px solid #E5E7EB;margin:2rem 0;'>",
                        unsafe_allow_html=True)
        render_question(
            q_cfg, hw_id, email, name,
            past_hard, grace_active,
            submissions
        )

# ── Completion screen ─────────────────────────────────────────────────────────
summary_fresh = get_hw_summary(hw_id, email,
                                st.session_state.get("submissions", {}))
if summary_fresh["all_done"]:
    st.markdown(f"""
    <div style="background:{COLORS['navy']};border-radius:10px;
         padding:2rem;text-align:center;margin-top:2rem;">
      <div style="font-family:'DM Serif Display',serif;color:white;
           font-size:1.4rem;margin-bottom:0.5rem;">Assignment Complete</div>
      <div style="color:#94A3B8;font-size:0.9rem;margin-bottom:1rem;">
        All questions submitted ·
        Final score: <strong style="color:white;">
        {summary_fresh['total_score']} / {summary_fresh['total_max']}
        </strong>
      </div>
      <div style="color:#64748B;font-size:0.82rem;">
        Submitted answers are locked. Your instructor has received your responses.
      </div>
    </div>
    """, unsafe_allow_html=True)

# ── Navigation ────────────────────────────────────────────────────────────────
st.markdown("<br>", unsafe_allow_html=True)
if st.button("← Back to Dashboard"):
    # Refresh submissions before going back
    st.session_state["submissions"] = get_student_submissions(email)
    st.switch_page("pages/Dashboard.py")
