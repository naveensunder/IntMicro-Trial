"""
pages/Dashboard.py — Student landing page.
HWDashboard v2 — Phase 1
Features: semester score summary, per-hw scores, countdown timers,
student name prominent, last login, no raw HTML leaking.
"""
import streamlit as st
import datetime
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from db import (
    get_homework_configs, get_student_submissions,
    parse_deadline, check_auto_enable
)
from ui import inject_css, page_header, COLORS
from question_engine import (
    get_shuffled_questions, get_hw_summary, ALL_HW_CONFIGS
)

st.set_page_config(
    page_title="Dashboard — Microeconomics",
    page_icon="📘",
    layout="centered",
    initial_sidebar_state="expanded",
)

inject_css()

# ── Auth guard ─────────────────────────────────────────────────────────────────
if not st.session_state.get("authenticated"):
    st.warning("Please sign in first.")
    if st.button("Go to sign in"):
        st.switch_page("Home.py")
    st.stop()

email       = st.session_state["student_email"]
name        = st.session_state["student_name"]
record      = st.session_state.get("student_record", {})
last_login  = record.get("Last_Login", "")

# ── Refresh ────────────────────────────────────────────────────────────────────
check_auto_enable()

col_hd, col_rf = st.columns([5, 1])
with col_rf:
    if st.button("↻", help="Refresh", key="dash_refresh"):
        st.session_state["submissions"] = get_student_submissions(email)
        st.session_state["hw_configs"]  = get_homework_configs()
        st.rerun()

submissions = st.session_state.get("submissions", {})
hw_configs  = st.session_state.get("hw_configs",  [])

# ── Header ─────────────────────────────────────────────────────────────────────
with col_hd:
    page_header(
        "Intermediate Microeconomics",
        f"Welcome, {name}",
        f"Homework Dashboard"
        + (f" · Last sign-in: {last_login}" if last_login else "")
    )

# ── Semester summary ───────────────────────────────────────────────────────────
total_earned = 0
total_max    = 0
total_hws    = 0
done_hws     = 0

for cfg in hw_configs:
    if cfg.get("Enabled", "").upper() != "TRUE":
        continue
    total_hws += 1
    summary = get_hw_summary(cfg["HW_ID"], email, submissions)
    total_earned += summary["total_score"]
    total_max    += summary["total_max"]
    if summary["all_done"]:
        done_hws += 1

if total_max > 0:
    pct = int(total_earned / total_max * 100)
    st.markdown(f"""
    <div class="semester-summary">
      <div>
        <div class="sem-sum-label">Semester Score</div>
        <div class="sem-sum-value">{total_earned} / {total_max}</div>
        <div class="sem-sum-sub">{pct}% · {done_hws} of {total_hws} assignments complete</div>
      </div>
      <div style="width:180px;">
        <div class="progress-bar-outer" style="height:8px;">
          <div class="progress-bar-inner" style="width:{pct}%;"></div>
        </div>
        <div style="font-size:0.75rem;color:#64748B;margin-top:0.3rem;">
          {total_earned} pts earned of {total_max} possible
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)

# ── Assignment list ────────────────────────────────────────────────────────────
st.markdown(f"""
<div style="font-size:0.68rem;font-weight:600;letter-spacing:0.12em;
 text-transform:uppercase;color:{COLORS['neutral_500']};margin-bottom:0.7rem;">
  Assignments
</div>
""", unsafe_allow_html=True)

if not hw_configs:
    st.markdown('<div class="banner-info">No assignments have been posted yet.</div>',
                unsafe_allow_html=True)

sorted_configs = sorted(hw_configs, key=lambda x: x.get("HW_ID", ""))
now = datetime.datetime.now()

for cfg in sorted_configs:
    hw_id     = cfg.get("HW_ID", "")
    title     = cfg.get("Title", hw_id)
    enabled   = cfg.get("Enabled", "FALSE").upper() == "TRUE"
    deadline  = cfg.get("Deadline", "")
    grace_min = int(cfg.get("Grace_Minutes", 15))
    announce  = cfg.get("Announcement", "")
    max_marks = cfg.get("Max_Marks", "—")

    past_hard, past_soft, dl_dt, dl_grace = parse_deadline(deadline, grace_min)
    summary = get_hw_summary(hw_id, email, submissions)

    # ── Badge (text only — no raw HTML in variables) ───────────────────────────
    if not enabled:
        badge_cls  = "badge-locked"
        badge_text = "Not yet available"
    elif past_hard:
        badge_cls  = "badge-closed"
        badge_text = "Closed"
    elif summary["all_done"]:
        badge_cls  = "badge-complete"
        badge_text = "Complete"
    else:
        badge_cls  = "badge-open"
        badge_text = "Open"

    # ── Deadline display ───────────────────────────────────────────────────────
    try:
        dl_str = dl_dt.strftime("%d %b %Y, %H:%M")
    except Exception:
        dl_str = deadline

    # ── Countdown ─────────────────────────────────────────────────────────────
    countdown_html = ""
    if enabled and not past_hard:
        rem = dl_grace - now
        if rem.total_seconds() > 0:
            d = rem.days
            h = (rem.seconds) // 3600
            m = (rem.seconds % 3600) // 60
            urgent = d == 0 and h < 6
            cls = "countdown-urgent" if urgent else ""
            if d > 0:
                time_left = f"{d}d {h}h {m}m remaining"
            elif h > 0:
                time_left = f"{h}h {m}m remaining"
            else:
                time_left = f"{m}m remaining"
            countdown_html = (
                f'<span class="countdown-item {cls}">'
                f'⏳ <span class="countdown-value {cls}">{time_left}</span>'
                f'</span>'
            )

    # ── Score display ──────────────────────────────────────────────────────────
    score_html = ""
    if summary["n_submitted"] > 0:
        if summary["all_done"]:
            score_html = (
                f'<div class="hw-score">'
                f'Score: {summary["total_score"]} / {summary["total_max"]}'
                f'</div>'
            )
        else:
            score_html = (
                f'<div class="hw-score">'
                f'{summary["n_submitted"]} of {summary["n_total"]} questions submitted'
                f' · {summary["total_score"]} pts so far'
                f'</div>'
            )

    # ── Announcement ──────────────────────────────────────────────────────────
    ann_html = ""
    if announce:
        ann_html = (
            f'<div class="hw-meta" style="color:{COLORS["warning"]};margin-top:0.25rem;">'
            f'📢 {announce}'
            f'</div>'
        )

    # ── Card (all values safely interpolated, no bare HTML) ───────────────────
    card_cls = "hw-card-locked" if not enabled else "hw-card"

    st.markdown(f"""
    <div class="{card_cls}">
      <div style="display:flex;justify-content:space-between;align-items:flex-start;gap:1rem;">
        <div style="flex:1;min-width:0;">
          <div class="hw-title">{title}</div>
          <div class="hw-meta">Deadline: {dl_str} &nbsp;·&nbsp; {max_marks} pts</div>
          <div class="countdown-bar">{countdown_html}</div>
          {score_html}
          {ann_html}
        </div>
        <div style="flex-shrink:0;">
          <span class="badge {badge_cls}">{badge_text}</span>
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    if enabled and not past_hard:
        if st.button(f"Open {title} →",
                     key=f"open_{hw_id}"):
            st.session_state["current_hw"] = hw_id
            st.switch_page("pages/Homework.py")

# ── All complete screen ────────────────────────────────────────────────────────
if total_hws > 0 and done_hws == total_hws:
    st.markdown(f"""
    <div style="background:{COLORS['navy']};border-radius:10px;
         padding:1.8rem;text-align:center;margin-top:1.5rem;">
      <div style="font-family:'DM Serif Display',serif;color:white;
           font-size:1.25rem;margin-bottom:0.35rem;">
        All assignments submitted
      </div>
      <div style="color:#94A3B8;font-size:0.86rem;">
        Final semester score: {total_earned} / {total_max} pts
      </div>
    </div>
    """, unsafe_allow_html=True)

# ── Sign out ───────────────────────────────────────────────────────────────────
st.markdown("<br>", unsafe_allow_html=True)
if st.button("Sign out", key="signout"):
    for k in ["authenticated","student_email","student_name","student_record",
              "submissions","hw_configs","current_hw","login_flow"]:
        st.session_state.pop(k, None)
    st.switch_page("Home.py")
