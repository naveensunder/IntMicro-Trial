"""
pages/Dashboard.py — HWDashboard v11
Phase 4:
  - Sidebar branding with student avatar
  - Breadcrumb navigation
  - Assignment cards with prominent due date, pts, status pills
  - NEW pill on recently opened homeworks
  - Section rules replacing inline label divs
  - Session state fully cleaned up on sign-out
  - Last-visited continuity (resume in-progress hw on load)
  - Score history chart row per homework
  - st.markdown('<br>') replaced with st.divider()
  - Browser tab title standardised
"""
import streamlit as st
import datetime
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import re as _re_dash
from db import (
    get_homework_configs, get_student_submissions,
    parse_deadline, check_auto_enable,
)
from ui import (
    inject_css, page_header, COLORS, banner, page_footer,
    sidebar_brand, breadcrumb, section_rule, progress_bar,
    empty_state, hide_home_when_authed,
)
from question_engine import get_hw_summary

st.set_page_config(
    page_title="EC224 — Dashboard · Bentley",
    page_icon="📘",
    layout="centered",
    initial_sidebar_state="expanded",
)
inject_css()
hide_home_when_authed()


def _week_num(hw_id: str) -> int:
    m = _re_dash.search(r"(\d+)", hw_id)
    return int(m.group(1)) if m else 999


# ── Auth gate ──────────────────────────────────────────────────────────────────
if not st.session_state.get("authenticated"):
    st.warning("Please sign in first.")
    if st.button("Go to sign in"):
        st.switch_page("Home.py")
    st.stop()

email  = st.session_state["student_email"]
name   = st.session_state["student_name"]
record = st.session_state.get("student_record", {})

# ── Sidebar branding ───────────────────────────────────────────────────────────
sidebar_brand(name)

# ── Top bar: refresh + sign out top right ─────────────────────────────────────
col_h, col_r1, col_r2 = st.columns([5, 0.7, 0.9])
with col_r1:
    if st.button("↻", help="Refresh", key="dash_ref"):
        st.session_state["submissions"] = get_student_submissions(email)
        st.session_state["hw_configs"]  = get_homework_configs()
        st.rerun()
with col_r2:
    if st.button("Sign out", key="signout_top"):
        _keys = [
            "authenticated", "student_email", "student_name", "student_record",
            "submissions", "hw_configs", "current_hw", "login_flow",
            "preview_mode", "last_visited_hw", "_continuity_shown",
            "login_attempts", "lockout_until", "reg_success_name",
        ]
        for k in _keys:
            st.session_state.pop(k, None)
        st.switch_page("Home.py")

try:
    check_auto_enable()
except Exception:
    pass

submissions = st.session_state.get("submissions", {})
hw_configs  = st.session_state.get("hw_configs",  []) or []

with col_h:
    last_login = record.get("Last_Login", "")
    last_str   = f" · Last sign-in: {last_login}" if last_login else ""
    page_header(
        "Intermediate Microeconomics",
        f"Welcome, {name}",
        f"Homework Dashboard{last_str}",
    )


# ── Last-visited continuity: resume in-progress homework ──────────────────────
last_hw = st.session_state.get("last_visited_hw")
if last_hw and not st.session_state.get("_continuity_shown"):
    last_cfg = next(
        (c for c in hw_configs if c.get("HW_ID") == last_hw), None
    )
    if last_cfg:
        last_title  = last_cfg.get("Title", last_hw)
        last_enabled = last_cfg.get("Enabled", "FALSE").upper() == "TRUE"
        last_summary = get_hw_summary(last_hw, email, submissions)
        if last_enabled and not last_summary.get("all_done"):
            st.markdown(
                f'<div class="banner banner-info" style="display:flex;'
                f'justify-content:space-between;align-items:center;gap:1rem;">'
                f'<span>📌 Continue where you left off: '
                f'<strong>{last_title}</strong> · '
                f'{last_summary["n_submitted"]} of '
                f'{last_summary["n_total"]} submitted</span>'
                f'</div>',
                unsafe_allow_html=True,
            )
            if st.button(f"Resume {last_title} →", key="resume_hw"):
                st.session_state["current_hw"]        = last_hw
                st.session_state["_continuity_shown"] = True
                st.switch_page("pages/Homework.py")

# ── Semester score ─────────────────────────────────────────────────────────────
total_earned = 0
total_max    = 0
now          = datetime.datetime.now()

for cfg in sorted(hw_configs, key=lambda x: _week_num(x.get("HW_ID", ""))):
    if cfg.get("Enabled", "").upper() != "TRUE":
        continue
    s = get_hw_summary(cfg["HW_ID"], email, submissions)
    total_earned += s["total_score"]
    total_max    += s["total_max"]

if total_max > 0:
    st.markdown(
        f'<div class="sem-box">'
        f'<div class="sem-label">Semester Score</div>'
        f'<div class="sem-score">{total_earned} / {total_max}</div>'
        f'<div class="sem-sub">Points earned so far this semester</div>'
        f'</div>',
        unsafe_allow_html=True,
    )

# ── 48-hour deadline warnings ──────────────────────────────────────────────────
for cfg in hw_configs:
    if cfg.get("Enabled", "").upper() != "TRUE":
        continue
    hw_id    = cfg.get("HW_ID", "")
    deadline = cfg.get("Deadline", "")
    grace    = int(cfg.get("Grace_Minutes") or 15)
    past_hard, _, dl_dt, dl_grace = parse_deadline(deadline, grace)
    if past_hard:
        continue
    summary = get_hw_summary(hw_id, email, submissions)
    if summary["all_done"]:
        continue
    rem = dl_grace - now
    if 0 < rem.total_seconds() < 48 * 3600:
        h = int(rem.total_seconds() // 3600)
        m = int((rem.total_seconds() % 3600) // 60)
        banner(
            f"⏰ <strong>{cfg.get('Title', hw_id)}</strong> is due in "
            f"<strong>{h}h {m}m</strong> — you haven't submitted all questions.",
            "warning",
        )

# ── Placeholders for upcoming weeks not yet in question_engine ─────────────────
PLACEHOLDERS = [
    {"HW_ID": "HW_WEEK3", "Title": "Week 3 — Consumer Preferences",  "Week": 3},
    {"HW_ID": "HW_WEEK4", "Title": "Week 4 — Utility Maximisation",  "Week": 4},
]

# ── Classify assignments ───────────────────────────────────────────────────────
open_hws     = []
upcoming_hws = []
closed_hws   = []

# Sort by deadline (soonest first) within each group, then by week for upcoming
for cfg in sorted(hw_configs, key=lambda x: x.get("Deadline", "9999")):
    hw_id    = cfg.get("HW_ID", "")
    enabled  = cfg.get("Enabled", "FALSE").upper() == "TRUE"
    deadline = cfg.get("Deadline", "")
    grace    = int(cfg.get("Grace_Minutes") or 15)
    past_hard, past_soft, dl_dt, dl_grace = parse_deadline(deadline, grace)

    opening  = cfg.get("Opening_Date", "").strip()
    open_yet = True
    if opening:
        try:
            op_dt    = datetime.datetime.strptime(opening, "%Y-%m-%d %H:%M")
            open_yet = datetime.datetime.now() >= op_dt
        except Exception:
            open_yet = True

    if enabled and not past_hard and open_yet:
        open_hws.append(cfg)
    elif past_hard:
        closed_hws.append(cfg)
    else:
        upcoming_hws.append(cfg)


# ── Card renderer ──────────────────────────────────────────────────────────────
def render_hw_card(cfg, is_open=False, is_placeholder=False):
    hw_id     = cfg.get("HW_ID", "")
    title     = cfg.get("Title", hw_id)
    deadline  = cfg.get("Deadline", "")
    grace_min = int(cfg.get("Grace_Minutes") or 15)
    announce  = cfg.get("Announcement", "")
    max_marks = cfg.get("Max_Marks", "—")

    past_hard, past_soft, dl_dt, dl_grace = parse_deadline(deadline, grace_min)
    summary = get_hw_summary(hw_id, email, submissions)

    # Deadline formatting
    try:
        is_tbd = dl_dt.year == 2099
        if is_tbd:
            dl_str = "Deadline to be announced"
            urgent = False
        else:
            dl_str = dl_dt.strftime("%a %d %b %Y · %I:%M %p")
            rem    = dl_grace - now
            urgent = (
                not past_hard and
                0 < rem.total_seconds() < 24 * 3600
            )
    except Exception:
        dl_str   = deadline
        urgent   = False
        is_tbd   = False

    # ── Locked / placeholder card ──
    if is_placeholder or (not is_open and not past_hard
                          and cfg.get("Enabled", "FALSE").upper() != "TRUE"):
        st.markdown(
            f'<div class="hw-card-locked">'
            f'<div class="hw-card-header">'
            f'<div>'
            f'<div class="hw-title">{title}</div>'
            f'<div class="hw-meta">Coming soon</div>'
            f'</div>'
            f'<span class="pill pill-upcoming">Upcoming</span>'
            f'</div>'
            f'</div>',
            unsafe_allow_html=True,
        )
        return

    # ── Status pill ──
    if is_open and summary["all_done"]:
        pill_html = '<span class="pill pill-complete">✓ Complete</span>'
    elif is_open:
        pill_html = '<span class="pill pill-open">Open</span>'
    elif past_hard:
        pill_html = '<span class="pill pill-closed">Closed</span>'
    else:
        pill_html = '<span class="pill pill-upcoming">Upcoming</span>'

    # NEW badge — opened in last 48h
    new_badge = ""
    opening = cfg.get("Opening_Date", "").strip()
    if opening and is_open:
        try:
            op_dt = datetime.datetime.strptime(opening, "%Y-%m-%d %H:%M")
            if (now - op_dt).total_seconds() < 48 * 3600:
                new_badge = '<span class="badge-new">New</span>'
        except Exception:
            pass

    # Due date — compact inline version for left-side display
    if is_tbd:
        due_html_inline = "Due: TBA"
    elif urgent:
        due_html_inline = f'<span style="color:#DC2626;font-weight:600;">⚠ Due: {dl_str}</span>'
    else:
        due_html_inline = f'Due: {dl_str} &nbsp;·&nbsp; {max_marks} pts'

    # Score / progress line
    score_html = ""
    if summary["n_submitted"] > 0:
        if summary["all_done"]:
            pct = int(summary["total_score"] / summary["total_max"] * 100) if summary["total_max"] else 0
            score_html = (
                f'<div class="hw-score">'
                f'Score: {summary["total_score"]} / {summary["total_max"]} '
                f'({pct}%)</div>'
            )
        else:
            score_html = (
                f'<div class="hw-score">'
                f'{summary["n_submitted"]} of {summary["n_total"]} submitted'
                f' · {summary["total_score"]} pts so far</div>'
            )

    ann_html = ""
    if announce:
        ann_html = (
            f'<div class="hw-meta" style="color:#D97706;margin-top:0.2rem;">'
            f'📢 {announce}</div>'
        )

    card_cls  = "hw-card-open" if is_open else "hw-card"
    title_cls = "hw-title-open" if is_open else "hw-title"

    # Shorten title to week number only for button
    import re as _re
    wm = _re.search(r"(\d+)", hw_id)
    short_num = wm.group(1) if wm else title

    btn_action = ""
    if is_open and not past_hard:
        action_word = "Continue" if summary["n_submitted"] > 0 and not summary["all_done"] else "Open"
        btn_label   = f"{action_word} Week {short_num} →"
        btn_html    = (
            f'<div style="flex-shrink:0;margin-left:1rem;">'
            f'<span class="hw-open-btn-placeholder" data-key="open_{hw_id}">{btn_label}</span>'
            f'</div>'
        )
    else:
        btn_html = ""

    st.markdown(
        f'<div class="{card_cls}">'
        f'<div class="hw-card-header">'
        f'<div style="flex:1;min-width:0;">'
        f'<div class="{title_cls}">{title}{new_badge}</div>'
        f'<div style="display:flex;align-items:center;gap:1rem;flex-wrap:wrap;margin-top:0.25rem;">'
        f'<span style="font-size:0.82rem;color:#555555;">{due_html_inline}</span>'
        f'{score_html}{ann_html}'
        f'</div>'
        f'</div>'
        f'<div style="display:flex;align-items:center;gap:0.8rem;flex-shrink:0;">'
        f'{pill_html}'
        f'</div>'
        f'</div>'
        f'</div>',
        unsafe_allow_html=True,
    )

    if is_open and not past_hard:
        action_word = "Continue" if summary["n_submitted"] > 0 and not summary["all_done"] else "Open"
        btn_label   = f"{action_word} Week {short_num} →"
        col_btn, col_sp = st.columns([1.4, 3.6])
        with col_btn:
            if st.button(btn_label, key=f"open_{hw_id}"):
                st.session_state["current_hw"]      = hw_id
                st.session_state["last_visited_hw"] = hw_id
                st.switch_page("pages/Homework.py")


# ── Open assignments ───────────────────────────────────────────────────────────
section_rule("Open Now")
if open_hws:
    for cfg in open_hws:
        render_hw_card(cfg, is_open=True)
else:
    empty_state(
        "📭",
        "No assignments open right now",
        "Check back soon — your instructor will open upcoming assignments.",
    )

# ── Upcoming ───────────────────────────────────────────────────────────────────
section_rule("Upcoming")
has_upcoming = False
for cfg in sorted(upcoming_hws, key=lambda x: _week_num(x.get("HW_ID", ""))):
    render_hw_card(cfg, is_open=False)
    has_upcoming = True
for p in sorted(PLACEHOLDERS, key=lambda x: x.get("Week", 99)):
    existing_ids = [c.get("HW_ID", "") for c in hw_configs]
    if p["HW_ID"] not in existing_ids:
        render_hw_card(p, is_placeholder=True)
        has_upcoming = True
if not has_upcoming:
    empty_state("📅", "No upcoming assignments", "All caught up!")

# ── Closed ─────────────────────────────────────────────────────────────────────
if closed_hws:
    section_rule("Closed")
    for cfg in closed_hws:
        render_hw_card(cfg, is_open=False)



page_footer()
