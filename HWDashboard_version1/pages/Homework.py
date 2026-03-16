"""
pages/Homework.py — Renders homework questions.
HWDashboard v3 — stability-first.
Summary table rendered with separate st.markdown calls — no dynamic HTML injection.
"""
import streamlit as st
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from db import parse_deadline, get_student_submissions, get_homework_configs
from ui import inject_css, page_header, COLORS, banner
from question_engine import get_questions, render_question, get_hw_summary, ALL_HW_CONFIGS

st.set_page_config(
    page_title="Homework — Microeconomics",
    page_icon="📝",
    layout="centered",
    initial_sidebar_state="expanded",
)
inject_css()

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
    if st.button("← Dashboard"):
        st.switch_page("pages/Dashboard.py")
    st.stop()

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
instructions  = hw_cfg.get("Instructions", "")

past_hard, past_soft, dl_dt, dl_grace = parse_deadline(deadline_str, grace_minutes)
grace_active = past_soft and not past_hard

submissions = st.session_state.get("submissions", {})

try:
    dl_display = dl_dt.strftime("%d %b %Y at %H:%M")
except Exception:
    dl_display = deadline_str

page_header("Intermediate Microeconomics", title,
            f"{name} · Deadline: {dl_display}")

# ── Announcement ───────────────────────────────────────────────────────────────
if announce:
    banner(f"📢 {announce}", "warning")

# ── Instructions ───────────────────────────────────────────────────────────────
if instructions:
    st.markdown(
        f'<div style="background:#F9FAFB;border:1px solid #E5E7EB;border-radius:8px;'
        f'padding:0.9rem 1.2rem;margin-bottom:1rem;font-size:0.89rem;'
        f'color:{COLORS["neutral_700"]};line-height:1.7;">'
        f'<strong>Instructions:</strong> {instructions}</div>',
        unsafe_allow_html=True
    )

# ── Deadline banner ────────────────────────────────────────────────────────────
if past_hard:
    banner("🔒 This assignment is now closed. Submissions are no longer accepted.", "error")
elif grace_active:
    banner("⏳ The deadline has passed but you are within the grace period. Submit now.", "warning")

# ── Instructor password reminder (only for instructor in preview) ──────────────
if st.session_state.get("preview_mode"):
    st.markdown(
        '<div class="instructor-note">👁 You are in instructor preview mode. '
        'Sign out to return to the instructor dashboard.</div>',
        unsafe_allow_html=True
    )

# ── Question summary table ─────────────────────────────────────────────────────
hw_subs   = submissions.get(hw_id, {}) if isinstance(submissions, dict) else {}
questions = ALL_HW_CONFIGS.get(hw_id, {}).get("questions", [])
summary   = get_hw_summary(hw_id, email, submissions)

total_score = 0; total_max = 0

# Table header — separate call
st.markdown(
    '<div style="background:#FFFFFF;border:1px solid #E5E7EB;border-radius:10px;'
    'padding:1rem 1.3rem;margin-bottom:1.3rem;">'
    '<div style="font-size:0.67rem;font-weight:600;letter-spacing:0.1em;'
    'text-transform:uppercase;color:#6B7280;margin-bottom:0.7rem;">Question Overview</div>'
    '<table class="q-summary">'
    '<tr><th>Question</th><th style="text-align:center;">Score</th>'
    '<th style="text-align:center;">Status</th></tr>',
    unsafe_allow_html=True
)

# Each row — separate call to avoid injection issues
for q in questions:
    q_id   = q["q_id"]
    q_title = q["title"]
    max_sc  = q["marks"]
    sub     = hw_subs.get(q_id, {}) if isinstance(hw_subs, dict) else {}
    submitted = str(sub.get("Status","")) == "submitted"

    if submitted:
        score     = int(sub.get("Score", 0))
        score_str = f"{score} / {max_sc}"
        status    = '<span class="status-done">✓ Submitted</span>'
        total_score += score
    else:
        score_str = f"— / {max_sc}"
        status    = '<span class="status-todo">Not submitted</span>'

    total_max += max_sc

    st.markdown(
        f'<tr>'
        f'<td style="font-size:0.86rem;">{q_title}</td>'
        f'<td style="text-align:center;">{score_str}</td>'
        f'<td>{status}</td>'
        f'</tr>',
        unsafe_allow_html=True
    )

# Total row + close table
total_pct = int(total_score / total_max * 100) if total_max > 0 else 0
st.markdown(
    f'<tr class="total-row">'
    f'<td><strong>Total</strong></td>'
    f'<td style="text-align:center;"><strong>{total_score} / {total_max} ({total_pct}%)</strong></td>'
    f'<td></td>'
    f'</tr>'
    f'</table></div>',
    unsafe_allow_html=True
)

# ── Questions ──────────────────────────────────────────────────────────────────
rendered_questions = get_questions(hw_id)

if not rendered_questions:
    st.info("No questions found for this assignment.")
else:
    for i, q_cfg in enumerate(rendered_questions):
        if i > 0:
            st.markdown(
                "<hr style='border:none;border-top:1px solid #E5E7EB;margin:2rem 0;'>",
                unsafe_allow_html=True
            )
        render_question(
            q_cfg, hw_id, email,
            past_hard, grace_active,
            submissions
        )

# ── Completion ─────────────────────────────────────────────────────────────────
fresh = get_hw_summary(hw_id, email, st.session_state.get("submissions", {}))
if fresh["all_done"]:
    st.markdown(
        f'<div style="background:{COLORS["navy"]};border-radius:10px;'
        f'padding:1.8rem;text-align:center;margin-top:2rem;">'
        f'<div style="font-family:\'DM Serif Display\',serif;color:white;'
        f'font-size:1.3rem;margin-bottom:0.4rem;">Assignment Complete</div>'
        f'<div style="color:#94A3B8;font-size:0.87rem;margin-bottom:0.4rem;">'
        f'Final score: <strong style="color:white;">'
        f'{fresh["total_score"]} / {fresh["total_max"]}</strong></div>'
        f'<div style="color:#64748B;font-size:0.79rem;">'
        f'Your answers are locked. Return here anytime to review your solutions.'
        f'</div></div>',
        unsafe_allow_html=True
    )

# ── Extension info ─────────────────────────────────────────────────────────────
if past_hard and not fresh["all_done"]:
    st.markdown("<br>", unsafe_allow_html=True)
    banner(
        "The deadline has passed. If you have extenuating circumstances, "
        "please email your instructor directly at nsunder@bentley.edu "
        "with your name and the assignment name.",
        "warning"
    )

# ── Navigation ─────────────────────────────────────────────────────────────────
st.markdown("<br>", unsafe_allow_html=True)
if st.button("← Back to Dashboard"):
    st.session_state["submissions"] = get_student_submissions(email)
    st.switch_page("pages/Dashboard.py")
