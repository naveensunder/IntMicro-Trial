"""
pages/Dashboard.py — Student landing page.
Shows all homeworks with status, deadlines, and progress.
"""
import streamlit as st
import datetime
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from db import get_homework_configs, get_student_submissions, parse_deadline
from ui import inject_css, page_header, COLORS
from question_engine import get_shuffled_questions, get_hw_summary, ALL_HW_CONFIGS

st.set_page_config(
    page_title="Dashboard — Microeconomics",
    page_icon="📘",
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

# ── Refresh data ──────────────────────────────────────────────────────────────
if st.button("↻ Refresh", key="dash_refresh"):
    st.session_state["submissions"] = get_student_submissions(email)
    st.session_state["hw_configs"]  = get_homework_configs()
    st.rerun()

submissions = st.session_state.get("submissions", {})
hw_configs  = st.session_state.get("hw_configs",  [])

# ── Header ─────────────────────────────────────────────────────────────────────
page_header(
    "Intermediate Microeconomics",
    "Homework Dashboard",
    f"Welcome, {name}"
)

# ── Overall progress ──────────────────────────────────────────────────────────
total_hws   = len([c for c in hw_configs if c.get("Enabled","").upper() == "TRUE"])
done_hws    = 0
for c in hw_configs:
    if c.get("Enabled","").upper() != "TRUE": continue
    summary = get_hw_summary(c["HW_ID"], email, submissions)
    if summary["all_done"]: done_hws += 1

if total_hws > 0:
    pct = int(done_hws / total_hws * 100)
    st.markdown(f"""
    <div class="progress-wrap">
      <div class="progress-label">Semester Progress</div>
      <div class="progress-bar-outer">
        <div class="progress-bar-inner" style="width:{pct}%;"></div>
      </div>
      <div class="progress-text">{done_hws} of {total_hws} assignments fully submitted</div>
    </div>
    """, unsafe_allow_html=True)

# ── Homework list ─────────────────────────────────────────────────────────────
st.markdown(f"""
<div style="font-size:0.72rem;font-weight:600;letter-spacing:0.1em;
 text-transform:uppercase;color:{COLORS['neutral_500']};margin-bottom:0.8rem;">
  Assignments
</div>
""", unsafe_allow_html=True)

if not hw_configs:
    st.markdown('<div class="banner-info">No assignments have been posted yet. Check back soon.</div>',
                unsafe_allow_html=True)

# Sort by HW_ID
sorted_configs = sorted(hw_configs, key=lambda x: x.get("HW_ID", ""))

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

    # Status badge
    if not enabled:
        badge = '<span class="badge badge-locked">Not yet available</span>'
        card_class = "hw-card-locked"
    elif past_hard:
        badge = '<span class="badge badge-closed">Closed</span>'
        card_class = "hw-card"
    elif summary["all_done"]:
        badge = '<span class="badge badge-complete">Complete</span>'
        card_class = "hw-card"
    else:
        badge = '<span class="badge badge-open">Open</span>'
        card_class = "hw-card"

    # Deadline string
    try:
        dl_str = dl_dt.strftime("%d %b %Y, %H:%M")
        remaining = dl_grace - datetime.datetime.now()
        if remaining.total_seconds() > 0 and enabled and not past_hard:
            days = remaining.days
            hours = (remaining.seconds) // 3600
            if days > 0:
                time_left = f" · {days}d {hours}h left"
            else:
                time_left = f" · {hours}h left"
        else:
            time_left = ""
    except Exception:
        dl_str = deadline; time_left = ""

    # Progress within homework
    n_sub = summary["n_submitted"]; n_tot = summary["n_total"]
    hw_progress = f"{n_sub}/{n_tot} questions submitted"
    if summary["all_done"] and n_tot > 0:
        score_str = f" · Score: {summary['total_score']}/{summary['total_max']}"
    else:
        score_str = ""

    st.markdown(f"""
    <div class="{card_class}">
      <div style="display:flex;justify-content:space-between;align-items:flex-start;">
        <div>
          <div class="hw-title">{title}</div>
          <div class="hw-meta">
            Deadline: {dl_str}{time_left}
            &nbsp;·&nbsp; {max_marks} pts total
          </div>
          <div class="hw-meta" style="margin-top:0.15rem;">
            {hw_progress}{score_str}
          </div>
          {'<div class="hw-meta" style="color:#D97706;margin-top:0.3rem;">📢 ' + announce + '</div>' if announce else ''}
        </div>
        <div style="flex-shrink:0;margin-left:1rem;">{badge}</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # Open button
    if enabled and not past_hard:
        if st.button(f"Open {title}", key=f"open_{hw_id}", use_container_width=False):
            st.session_state["current_hw"] = hw_id
            st.switch_page("pages/Homework.py")

# ── Completion screen ─────────────────────────────────────────────────────────
if total_hws > 0 and done_hws == total_hws:
    st.markdown(f"""
    <div style="background:{COLORS['navy']};border-radius:10px;
         padding:1.8rem;text-align:center;margin-top:1.5rem;">
      <div style="font-family:'DM Serif Display',serif;color:white;
           font-size:1.3rem;margin-bottom:0.4rem;">All assignments submitted</div>
      <div style="color:#94A3B8;font-size:0.88rem;">
        You have completed all available homework. Well done.
      </div>
    </div>
    """, unsafe_allow_html=True)

# ── Sign out ──────────────────────────────────────────────────────────────────
st.markdown("<br>", unsafe_allow_html=True)
if st.button("Sign out", key="signout"):
    for key in ["authenticated","student_email","student_name","student_record",
                "submissions","hw_configs","current_hw","login_flow"]:
        st.session_state.pop(key, None)
    st.switch_page("Home.py")
