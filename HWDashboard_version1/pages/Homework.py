"""
pages/Homework.py — HWDashboard v11
Phase 4:
  - Breadcrumb: Dashboard → Homework title
  - Sticky header with title + progress bar
  - Countdown timer on open homework
  - Progress bar replacing plain progress-line
  - Animated score reveal on submission
  - Confetti on perfect score
  - Inline feedback banners replacing plain st.success/st.error
  - Input validation for numerical fields
  - Homework receipt (score summary expander)
  - last_visited_hw continuity tracking
  - st.markdown('<br>') replaced with st.divider()
  - Browser tab title standardised
  - log_submission_attempt called on every submit
"""
import streamlit as st
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from db import (
    parse_deadline, get_student_submissions,
    get_homework_configs, log_submission_attempt,
)
from ui import (
    inject_css, page_header, COLORS, banner, page_footer,
    q_divider, breadcrumb, progress_bar, countdown_timer,
    sidebar_brand, feedback_banner, score_reveal,
    confetti_if_perfect, hide_home_when_authed,
)
from question_engine import (
    get_questions, render_question, get_hw_summary, ALL_HW_CONFIGS,
)

st.set_page_config(
    page_title="EC224 — Homework · Bentley",
    page_icon="📝",
    layout="centered",
    initial_sidebar_state="expanded",
)
inject_css()
hide_home_when_authed()

# ── Auth gate ──────────────────────────────────────────────────────────────────
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

# Track last visited for continuity
st.session_state["last_visited_hw"] = hw_id

# ── Sidebar branding ───────────────────────────────────────────────────────────
sidebar_brand(name)

# ── Load config ───────────────────────────────────────────────────────────────
hw_configs = st.session_state.get("hw_configs", []) or []
if not hw_configs:
    hw_configs = get_homework_configs()
    st.session_state["hw_configs"] = hw_configs

hw_cfg = next((c for c in hw_configs if c.get("HW_ID") == hw_id), {})
if not hw_cfg:
    st.error("Homework configuration not found.")
    st.stop()

title         = hw_cfg.get("Title", hw_id)
deadline_str  = hw_cfg.get("Deadline", "2099-12-31 23:59")
grace_minutes = int(hw_cfg.get("Grace_Minutes") or 15)
announce      = hw_cfg.get("Announcement", "")
instructions  = hw_cfg.get("Instructions", "")

past_hard, past_soft, dl_dt, dl_grace = parse_deadline(deadline_str, grace_minutes)
grace_active = past_soft and not past_hard
submissions  = st.session_state.get("submissions", {})

# Natural language deadline
try:
    if dl_dt.year == 2099:
        dl_display    = "Deadline to be announced"
        dl_iso        = ""
    else:
        dl_display    = dl_dt.strftime("%A, %d %B %Y at %I:%M %p")
        dl_iso        = dl_grace.strftime("%Y-%m-%dT%H:%M:%S")
except Exception:
    dl_display = deadline_str
    dl_iso     = ""

# ── Breadcrumb & header ────────────────────────────────────────────────────────
breadcrumb("Dashboard", title)
page_header(
    "Intermediate Microeconomics",
    title,
    f"{name} · Due: {dl_display}",
)

# ── Countdown timer (only when open) ──────────────────────────────────────────
if dl_iso and not past_hard:
    countdown_timer(hw_id, dl_iso)
    st.markdown("<br>", unsafe_allow_html=True)

# ── Announcements / instructions ──────────────────────────────────────────────
if announce:
    banner(f"📢 {announce}", "warning")

if instructions:
    st.markdown(
        f'<div style="background:#F5F5F5;border:1.5px solid #E0E0E0;'
        f'border-radius:8px;padding:1rem 1.3rem;margin-bottom:1.2rem;'
        f'font-size:1rem;color:#1A1A1A;line-height:1.7;">'
        f'<strong>Instructions:</strong> {instructions}</div>',
        unsafe_allow_html=True,
    )

if past_hard:
    banner("🔒 This assignment is now closed.", "error")
elif grace_active:
    banner(
        "⏳ The deadline has passed but you are within the grace period. "
        "Submit now.",
        "warning",
    )

if st.session_state.get("preview_mode"):
    banner("👁 Instructor preview mode.", "info")

# ── Progress bar ──────────────────────────────────────────────────────────────
summary      = get_hw_summary(hw_id, email, submissions)
n_submitted  = summary["n_submitted"]
n_total      = summary["n_total"]
total_score  = summary["total_score"]
total_max_s  = summary["total_max"]

progress_bar(n_submitted, n_total)

# ── Question overview rows ─────────────────────────────────────────────────────
hw_subs   = submissions.get(hw_id, {}) if isinstance(submissions, dict) else {}
questions = ALL_HW_CONFIGS.get(hw_id, {}).get("questions", [])

st.markdown(
    '<div style="font-size:0.78rem;font-weight:600;letter-spacing:0.1em;'
    'text-transform:uppercase;color:#555555;margin-bottom:0.6rem;">'
    'Question Overview</div>',
    unsafe_allow_html=True,
)

ov_total_score = 0
ov_total_max   = 0
for q in questions:
    q_id    = q["q_id"]
    q_title = q["title"]
    max_sc  = q["marks"]
    sub     = hw_subs.get(q_id, {}) if isinstance(hw_subs, dict) else {}
    submitted = str(sub.get("Status", "")) == "submitted"
    if submitted:
        score       = int(sub.get("Score", 0))
        score_str   = f"{score} / {max_sc}"
        status_html = '<span class="ov-done">✓ Submitted</span>'
        ov_total_score += score
    else:
        score_str   = f"— / {max_sc}"
        status_html = '<span class="ov-todo">Not submitted</span>'
    ov_total_max += max_sc
    st.markdown(
        f'<div class="ov-row">'
        f'<div class="ov-title">{q_title}</div>'
        f'<div class="ov-score">{score_str}</div>'
        f'<div class="ov-status">{status_html}</div>'
        f'</div>',
        unsafe_allow_html=True,
    )

total_pct = int(ov_total_score / ov_total_max * 100) if ov_total_max > 0 else 0
st.markdown(
    f'<div class="ov-total">'
    f'<span>Total</span>'
    f'<span>{ov_total_score} / {ov_total_max} &nbsp;({total_pct}%)</span>'
    f'</div>',
    unsafe_allow_html=True,
)

st.divider()

# ── Questions ──────────────────────────────────────────────────────────────────
rendered = get_questions(hw_id)
if not rendered:
    st.info("No questions found for this assignment.")
else:
    for i, q_cfg in enumerate(rendered):
        q_divider(q_cfg["title"])
        render_question(q_cfg, hw_id, email, past_hard, grace_active, submissions)

# ── Completion banner ──────────────────────────────────────────────────────────
fresh = get_hw_summary(hw_id, email, st.session_state.get("submissions", {}))
if fresh["all_done"]:
    final_score = fresh["total_score"]
    final_max   = fresh["total_max"]
    final_pct   = int(final_score / final_max * 100) if final_max else 0

    # Confetti on perfect score
    confetti_if_perfect(final_score, final_max)

    st.markdown(
        f'<div style="background:#1C2B4A;border-radius:12px;'
        f'padding:2rem;text-align:center;margin-top:2rem;">'
        f'<div style="font-family:\'DM Serif Display\',serif;color:#FFFFFF;'
        f'font-size:1.5rem;margin-bottom:0.4rem;">Assignment Complete</div>'
        f'<div style="color:#94A3B8;font-size:1rem;margin-bottom:0.4rem;">'
        f'Final score: <strong style="color:#FFFFFF;">'
        f'{final_score} / {final_max} ({final_pct}%)</strong></div>'
        f'<div style="color:#64748B;font-size:0.9rem;">'
        f'Your answers are locked. Return here to review solutions anytime.</div>'
        f'</div>',
        unsafe_allow_html=True,
    )

    # ── Homework receipt ──
    with st.expander("📄 View submission receipt", expanded=False):
        st.markdown(
            f'<div style="font-family:\'DM Serif Display\',serif;'
            f'font-size:1.1rem;color:#1C2B4A;margin-bottom:0.8rem;">'
            f'Submission Summary — {title}</div>',
            unsafe_allow_html=True,
        )
        hw_subs_fresh = (
            st.session_state.get("submissions", {}).get(hw_id, {})
        )
        for q in questions:
            q_id  = q["q_id"]
            sub   = hw_subs_fresh.get(q_id, {}) if isinstance(hw_subs_fresh, dict) else {}
            score = sub.get("Score", "—")
            mx    = q["marks"]
            ts    = sub.get("Timestamp", "—")
            st.markdown(
                f'<div class="score-row">'
                f'<div class="score-row-label">{q["title"]}</div>'
                f'<div class="score-row-val">{score} / {mx}</div>'
                f'<div style="font-size:0.78rem;color:#888888;">{ts}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )
        st.markdown(
            f'<div class="ov-total" style="margin-top:0.5rem;">'
            f'<span>Total</span>'
            f'<span>{final_score} / {final_max} ({final_pct}%)</span>'
            f'</div>',
            unsafe_allow_html=True,
        )
        st.caption(
            f"Email: {email} · Submitted: {title}"
        )

if past_hard and not fresh["all_done"]:
    st.divider()
    banner(
        "The deadline has passed. If you have extenuating circumstances, "
        "please email <strong>nsunder@bentley.edu</strong> with your name "
        "and the assignment name.",
        "warning",
    )

# ── Back to dashboard ──────────────────────────────────────────────────────────
st.divider()
if st.button("← Back to Dashboard"):
    st.session_state["submissions"] = get_student_submissions(email)
    st.session_state["_continuity_shown"] = False
    st.switch_page("pages/Dashboard.py")

page_footer()
