"""
pages/Homework.py — HWDashboard v5
Progress line at top. Overview rows. Questions with dividers.
"""
import streamlit as st
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from db import parse_deadline, get_student_submissions, get_homework_configs
from ui import inject_css, page_header, COLORS, banner, page_footer, q_divider
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
    if st.button("← Dashboard"): st.switch_page("pages/Dashboard.py")
    st.stop()

hw_configs = st.session_state.get("hw_configs", [])
if not hw_configs:
    hw_configs = get_homework_configs()
    st.session_state["hw_configs"] = hw_configs

hw_cfg = next((c for c in hw_configs if c.get("HW_ID")==hw_id), {})
if not hw_cfg:
    st.error("Homework configuration not found.")
    st.stop()

title         = hw_cfg.get("Title", hw_id)
deadline_str  = hw_cfg.get("Deadline", "2099-12-31 23:59")
grace_minutes = int(hw_cfg.get("Grace_Minutes") or 15)
announce      = hw_cfg.get("Announcement","")
instructions  = hw_cfg.get("Instructions","")

past_hard, past_soft, dl_dt, dl_grace = parse_deadline(deadline_str, grace_minutes)
grace_active = past_soft and not past_hard
submissions  = st.session_state.get("submissions", {})

# Natural language deadline
try:
    dl_display = dl_dt.strftime("%A, %d %B %Y at %I:%M %p")
except Exception:
    dl_display = deadline_str

page_header("Intermediate Microeconomics", title,
            f"{name} · Due: {dl_display}")

if announce:
    banner(f"📢 {announce}", "warning")

if instructions:
    st.markdown(
        f'<div style="background:#F5F5F5;border:1.5px solid #E0E0E0;'
        f'border-radius:8px;padding:1rem 1.3rem;margin-bottom:1.2rem;'
        f'font-size:1rem;color:{COLORS["black"]};line-height:1.7;">'
        f'<strong>Instructions:</strong> {instructions}</div>',
        unsafe_allow_html=True)

if past_hard:
    banner("🔒 This assignment is now closed.", "error")
elif grace_active:
    banner("⏳ The deadline has passed but you are within the grace period. Submit now.", "warning")

if st.session_state.get("preview_mode"):
    banner("👁 Instructor preview mode.", "info")

# ── Progress line ──────────────────────────────────────────────────────────────
summary = get_hw_summary(hw_id, email, submissions)
st.markdown(
    f'<div class="progress-line">'
    f'<strong>{summary["n_submitted"]} of {summary["n_total"]}</strong> questions submitted'
    f'{f" · Score so far: <strong>{summary[\'total_score\']} / {summary[\'total_max\']}</strong>" if summary["n_submitted"] > 0 else ""}'
    f'</div>',
    unsafe_allow_html=True
)

# ── Question overview rows ─────────────────────────────────────────────────────
hw_subs   = submissions.get(hw_id, {}) if isinstance(submissions, dict) else {}
questions = ALL_HW_CONFIGS.get(hw_id, {}).get("questions", [])

st.markdown(
    '<div style="font-size:0.8rem;font-weight:600;letter-spacing:0.1em;'
    'text-transform:uppercase;color:#555555;margin-bottom:0.6rem;">'
    'Question Overview</div>',
    unsafe_allow_html=True)

total_score = 0; total_max = 0
for q in questions:
    q_id = q["q_id"]; q_title = q["title"]; max_sc = q["marks"]
    sub  = hw_subs.get(q_id,{}) if isinstance(hw_subs,dict) else {}
    submitted = str(sub.get("Status",""))=="submitted"
    if submitted:
        score = int(sub.get("Score",0))
        score_str   = f"{score} / {max_sc}"
        status_html = '<span class="ov-done">✓ Submitted</span>'
        total_score += score
    else:
        score_str   = f"— / {max_sc}"
        status_html = '<span class="ov-todo">Not submitted</span>'
    total_max += max_sc
    st.markdown(
        f'<div class="ov-row">'
        f'<div class="ov-title">{q_title}</div>'
        f'<div class="ov-score">{score_str}</div>'
        f'<div class="ov-status">{status_html}</div>'
        f'</div>',
        unsafe_allow_html=True)

total_pct = int(total_score/total_max*100) if total_max > 0 else 0
st.markdown(
    f'<div class="ov-total">'
    f'<span>Total</span>'
    f'<span>{total_score} / {total_max} &nbsp;({total_pct}%)</span>'
    f'</div>',
    unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ── Questions ──────────────────────────────────────────────────────────────────
rendered = get_questions(hw_id)
if not rendered:
    st.info("No questions found for this assignment.")
else:
    for i, q_cfg in enumerate(rendered):
        if i > 0:
            q_divider(q_cfg["title"])
        render_question(q_cfg, hw_id, email, past_hard, grace_active, submissions)

# ── Completion ─────────────────────────────────────────────────────────────────
fresh = get_hw_summary(hw_id, email, st.session_state.get("submissions",{}))
if fresh["all_done"]:
    st.markdown(
        f'<div style="background:{COLORS["navy"]};border-radius:10px;'
        f'padding:2rem;text-align:center;margin-top:2rem;">'
        f'<div style="font-family:\'DM Serif Display\',serif;color:white;'
        f'font-size:1.5rem;margin-bottom:0.4rem;">Assignment Complete</div>'
        f'<div style="color:#94A3B8;font-size:1rem;margin-bottom:0.4rem;">'
        f'Final score: <strong style="color:white;">'
        f'{fresh["total_score"]} / {fresh["total_max"]}</strong></div>'
        f'<div style="color:#64748B;font-size:0.9rem;">'
        f'Your answers are locked. Return here to review solutions anytime.</div>'
        f'</div>',
        unsafe_allow_html=True)

if past_hard and not fresh["all_done"]:
    st.markdown("<br>", unsafe_allow_html=True)
    banner(
        "The deadline has passed. If you have extenuating circumstances, "
        "please email <strong>nsunder@bentley.edu</strong> with your name "
        "and the assignment name.",
        "warning")

st.markdown("<br>", unsafe_allow_html=True)
if st.button("← Back to Dashboard"):
    st.session_state["submissions"] = get_student_submissions(email)
    st.switch_page("pages/Dashboard.py")

page_footer()
