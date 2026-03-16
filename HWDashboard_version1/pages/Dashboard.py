"""
pages/Dashboard.py — Student landing page.
HWDashboard v4 — readability-first.
New: deadline urgent red, complete badge, last login, 48hr warning, footer.
"""
import streamlit as st
import datetime
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from db import get_homework_configs, get_student_submissions, parse_deadline
from ui import inject_css, page_header, COLORS, banner, page_footer
from question_engine import get_hw_summary

st.set_page_config(
    page_title="Dashboard — Microeconomics",
    page_icon="📘",
    layout="centered",
    initial_sidebar_state="expanded",
)
inject_css()

if not st.session_state.get("authenticated"):
    st.warning("Please sign in first.")
    if st.button("Go to sign in"):
        st.switch_page("Home.py")
    st.stop()

email  = st.session_state["student_email"]
name   = st.session_state["student_name"]
record = st.session_state.get("student_record", {})
last_login = record.get("Last_Login", "")

col_h, col_r = st.columns([5,1])
with col_r:
    if st.button("↻", help="Refresh", key="dash_ref"):
        st.session_state["submissions"] = get_student_submissions(email)
        st.session_state["hw_configs"]  = get_homework_configs()
        st.rerun()

submissions = st.session_state.get("submissions", {})
hw_configs  = st.session_state.get("hw_configs",  [])

with col_h:
    last_login_str = f" · Last sign-in: {last_login}" if last_login else ""
    page_header(
        "Intermediate Microeconomics",
        f"Welcome, {name}",
        f"Homework Dashboard{last_login_str}"
    )

# ── Instructor password reminder (shown if still default — best effort) ────────
if st.session_state.get("show_pw_reminder"):
    st.markdown(
        '<div class="banner banner-warning">'
        '🔑 Reminder: please change your default instructor password in the '
        'Instructor Dashboard → Settings tab.</div>',
        unsafe_allow_html=True
    )

# ── Semester score ─────────────────────────────────────────────────────────────
total_earned = 0; total_max = 0
for cfg in hw_configs:
    if cfg.get("Enabled","").upper() != "TRUE": continue
    s = get_hw_summary(cfg["HW_ID"], email, submissions)
    total_earned += s["total_score"]; total_max += s["total_max"]

if total_max > 0:
    st.markdown(
        f'<div class="sem-box">'
        f'<div class="sem-label">Semester Score</div>'
        f'<div class="sem-score">{total_earned} / {total_max}</div>'
        f'<div class="sem-sub">Points earned so far this semester</div>'
        f'</div>',
        unsafe_allow_html=True
    )

# ── 48-hour warning for any incomplete open homework ──────────────────────────
now = datetime.datetime.now()
for cfg in hw_configs:
    if cfg.get("Enabled","").upper() != "TRUE": continue
    hw_id    = cfg.get("HW_ID","")
    deadline = cfg.get("Deadline","")
    grace    = int(cfg.get("Grace_Minutes") or 15)
    past_hard, past_soft, dl_dt, dl_grace = parse_deadline(deadline, grace)
    if past_hard: continue
    summary = get_hw_summary(hw_id, email, submissions)
    if summary["all_done"]: continue
    rem = dl_grace - now
    if 0 < rem.total_seconds() < 48 * 3600:
        h = int(rem.total_seconds() // 3600)
        m = int((rem.total_seconds() % 3600) // 60)
        banner(
            f"⏰ <strong>{cfg.get('Title', hw_id)}</strong> is due in "
            f"<strong>{h}h {m}m</strong> and you have not yet submitted all questions.",
            "warning"
        )

# ── Assignments ────────────────────────────────────────────────────────────────
st.markdown(
    f'<div style="font-size:0.8rem;font-weight:600;letter-spacing:0.1em;'
    f'text-transform:uppercase;color:{COLORS["grey_text"]};margin-bottom:0.8rem;">'
    f'Assignments</div>',
    unsafe_allow_html=True
)

if not hw_configs:
    banner("No assignments posted yet. Check back soon.", "info")

for cfg in sorted(hw_configs, key=lambda x: x.get("HW_ID","")):
    hw_id     = cfg.get("HW_ID","")
    title     = cfg.get("Title", hw_id)
    enabled   = cfg.get("Enabled","FALSE").upper() == "TRUE"
    deadline  = cfg.get("Deadline","")
    grace_min = int(cfg.get("Grace_Minutes") or 15)
    announce  = cfg.get("Announcement","")
    max_marks = cfg.get("Max_Marks","—")

    past_hard, past_soft, dl_dt, dl_grace = parse_deadline(deadline, grace_min)
    summary = get_hw_summary(hw_id, email, submissions)

    # Deadline string and urgency
    try:
        dl_str = dl_dt.strftime("%d %b %Y, %H:%M")
        rem    = dl_grace - now
        urgent = (not past_hard and rem.total_seconds() < 24*3600
                  and rem.total_seconds() > 0)
    except Exception:
        dl_str = deadline; urgent = False

    # Badge
    if not enabled:
        badge_cls = "badge-locked"; badge_txt = "Not yet available"
    elif past_hard:
        badge_cls = "badge-closed"; badge_txt = "Closed"
    elif summary["all_done"]:
        badge_cls = "badge-complete"; badge_txt = "✓ Complete"
    else:
        badge_cls = "badge-open"; badge_txt = "Open"

    # Deadline display
    dl_html = (
        f'<div class="hw-deadline-urgent">⚠ Due: {dl_str} — less than 24 hours!</div>'
        if urgent else
        f'<div class="hw-meta">Due: {dl_str} &nbsp;·&nbsp; {max_marks} pts</div>'
    )

    # Score
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

    ann_html = ""
    if announce:
        ann_html = (
            f'<div class="hw-meta" style="color:{COLORS["warning"]};margin-top:0.2rem;">'
            f'📢 {announce}</div>'
        )

    card_cls = "hw-card-locked" if not enabled else "hw-card"

    st.markdown(
        f'<div class="{card_cls}">'
        f'<div style="display:flex;justify-content:space-between;'
        f'align-items:flex-start;gap:1rem;">'
        f'<div style="flex:1;min-width:0;">'
        f'<div class="hw-title">{title}</div>'
        f'{dl_html}'
        f'{score_html}'
        f'{ann_html}'
        f'</div>'
        f'<div style="flex-shrink:0;">'
        f'<span class="badge {badge_cls}">{badge_txt}</span>'
        f'</div>'
        f'</div>'
        f'</div>',
        unsafe_allow_html=True
    )

    if enabled and not past_hard:
        if st.button(f"Open {title} →", key=f"open_{hw_id}"):
            st.session_state["current_hw"] = hw_id
            st.switch_page("pages/Homework.py")

st.markdown("<br>", unsafe_allow_html=True)
if st.button("Sign out", key="signout"):
    for k in ["authenticated","student_email","student_name","student_record",
              "submissions","hw_configs","current_hw","login_flow","preview_mode"]:
        st.session_state.pop(k, None)
    st.switch_page("Home.py")

page_footer()
