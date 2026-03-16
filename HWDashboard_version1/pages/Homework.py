"""
pages/Homework.py — Renders homework questions.
HWDashboard v2 — Phase 1
Features: question summary table, review mode, print to PDF, auto-disable display.
"""
import streamlit as st
import datetime
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from db import (
    parse_deadline, get_student_submissions,
    get_homework_configs, log_audit, submit_extension_request
)
from ui import inject_css, page_header, COLORS, banner
from question_engine import (
    get_shuffled_questions, render_question,
    get_hw_summary, ALL_HW_CONFIGS, get_seed
)

st.set_page_config(
    page_title="Homework — Microeconomics",
    page_icon="📝",
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

email = st.session_state["student_email"]
name  = st.session_state["student_name"]
hw_id = st.session_state.get("current_hw")

if not hw_id:
    st.warning("No homework selected. Please go back to the dashboard.")
    if st.button("← Dashboard"):
        st.switch_page("pages/Dashboard.py")
    st.stop()

# ── Load configs ───────────────────────────────────────────────────────────────
hw_configs = st.session_state.get("hw_configs", [])
if not hw_configs:
    hw_configs = get_homework_configs()
    st.session_state["hw_configs"] = hw_configs

hw_cfg = next((c for c in hw_configs if c.get("HW_ID") == hw_id), {})
if not hw_cfg:
    st.error("Homework not found.")
    st.stop()

title         = hw_cfg.get("Title", hw_id)
deadline_str  = hw_cfg.get("Deadline", "2099-12-31 23:59")
grace_minutes = int(hw_cfg.get("Grace_Minutes", 15))
announce      = hw_cfg.get("Announcement", "")
instructions  = hw_cfg.get("Instructions", "")

past_hard, past_soft, dl_dt, dl_grace = parse_deadline(deadline_str, grace_minutes)
grace_active = past_soft and not past_hard

submissions = st.session_state.get("submissions", {})
questions   = get_shuffled_questions(hw_id, email)
summary     = get_hw_summary(hw_id, email, submissions)

# Determine review mode
review_mode = summary["all_done"]

# ── Track reloads ──────────────────────────────────────────────────────────────
for q in questions:
    k = f"reloads_{q['q_id']}_{hw_id}"
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

# ── Review mode banner ─────────────────────────────────────────────────────────
if review_mode:
    st.markdown("""
    <div class="review-mode-banner">
      📖 <strong>Review mode</strong> — All questions submitted.
      Answers are locked but you can review your responses and solutions below.
      Use this to prepare for exams.
    </div>
    """, unsafe_allow_html=True)

    # Print to PDF button
    st.markdown("""
    <div style="text-align:right;margin-bottom:1rem;">
      <button onclick="window.print()"
              style="background:#1C2B4A;color:white;border:none;border-radius:6px;
                     padding:0.4rem 1rem;font-size:0.84rem;cursor:pointer;
                     font-family:'DM Sans',sans-serif;">
        🖨 Print / Save as PDF
      </button>
    </div>
    """, unsafe_allow_html=True)

# ── Announcements & instructions ───────────────────────────────────────────────
if announce:
    banner(f"📢 {announce}", "warning")

if instructions and not review_mode:
    st.markdown(f"""
    <div class="card" style="margin-bottom:1rem;">
      <div class="card-label">Assignment Instructions</div>
      <div style="font-size:0.9rem;color:{COLORS['neutral_700']};line-height:1.7;">
        {instructions}
      </div>
    </div>
    """, unsafe_allow_html=True)

# ── Deadline banner ────────────────────────────────────────────────────────────
if past_hard and not review_mode:
    banner("🔒 This assignment is now closed.", "error")
elif grace_active:
    banner("⏳ Deadline has passed but you are within the grace period. Submit now.", "warning")
elif not review_mode:
    now = datetime.datetime.now()
    rem = dl_grace - now
    if rem.total_seconds() > 0:
        d = rem.days; h = rem.seconds // 3600; m = (rem.seconds % 3600) // 60
        if d == 0 and h < 6:
            banner(f"⚠ Deadline approaching — {h}h {m}m remaining.", "warning")

# ── Question summary table ─────────────────────────────────────────────────────
hw_subs = submissions.get(hw_id, {})

total_score = 0; total_max = 0
rows_html   = ""
config      = ALL_HW_CONFIGS.get(hw_id, {})

for q in config.get("questions", []):
    q_id   = q["q_id"]
    q_title = q["title"]
    max_sc  = q["marks"]
    sub     = hw_subs.get(q_id, {}) if isinstance(hw_subs, dict) else {}
    submitted = str(sub.get("Status","")) == "submitted"

    if submitted:
        score     = int(sub.get("Score", 0))
        ts        = sub.get("Timestamp", "")
        pct       = int(score / max_sc * 100) if max_sc > 0 else 0
        score_str = f"{score} / {max_sc}"
        pct_str   = f"{pct}%"
        status_str = f'<span class="status-submitted">✓ Submitted</span>'
        total_score += score
    else:
        score_str  = f"— / {max_sc}"
        pct_str    = "—"
        status_str = f'<span class="status-not-started">Not submitted</span>'

    total_max += max_sc
    # Anchor link to question
    rows_html += f"""
    <tr>
      <td>
        <a href="#{q_id}"
           style="color:{COLORS['accent']};text-decoration:none;font-weight:500;
                  font-size:0.87rem;"
           onclick="document.getElementById('{q_id}').scrollIntoView({{behavior:'smooth'}});
                    return false;">
          {q_title}
        </a>
      </td>
      <td style="text-align:center;">{score_str}</td>
      <td style="text-align:center;">{pct_str}</td>
      <td>{status_str}</td>
    </tr>
    """

total_pct = int(total_score / total_max * 100) if total_max > 0 else 0

st.markdown(f"""
<div class="card" style="padding:1rem 1.4rem;">
  <div class="card-label">Question Overview</div>
  <table class="q-summary-table">
    <tr>
      <th>Question</th>
      <th style="text-align:center;">Score</th>
      <th style="text-align:center;">Percent</th>
      <th>Status</th>
    </tr>
    {rows_html}
    <tr class="q-summary-total">
      <td><strong>Total</strong></td>
      <td style="text-align:center;"><strong>{total_score} / {total_max}</strong></td>
      <td style="text-align:center;"><strong>{total_pct}%</strong></td>
      <td></td>
    </tr>
  </table>
</div>
""", unsafe_allow_html=True)

# ── Progress bar ───────────────────────────────────────────────────────────────
n_sub = summary["n_submitted"]; n_tot = summary["n_total"]
pct   = int(n_sub / max(n_tot, 1) * 100)

st.markdown(f"""
<div class="progress-wrap">
  <div class="progress-label">Submission Progress</div>
  <div class="progress-bar-outer">
    <div class="progress-bar-inner" style="width:{pct}%;"></div>
  </div>
  <div class="progress-text">
    {n_sub} of {n_tot} questions submitted
    {f" · Score so far: {summary['total_score']} / {summary['total_max']}" if n_sub > 0 else ""}
  </div>
</div>
""", unsafe_allow_html=True)

# ── Questions ──────────────────────────────────────────────────────────────────
if not questions:
    st.info("No questions found for this assignment.")
else:
    for i, q_cfg in enumerate(questions):
        if i > 0:
            st.markdown(
                "<hr style='border:none;border-top:1px solid #E5E7EB;"
                "margin:2rem 0;'>",
                unsafe_allow_html=True
            )
        render_question(
            q_cfg, hw_id, email, name,
            past_hard, grace_active,
            submissions,
            review_mode=review_mode
        )

# ── Completion screen ──────────────────────────────────────────────────────────
fresh = get_hw_summary(hw_id, email, st.session_state.get("submissions", {}))
if fresh["all_done"] and not review_mode:
    st.markdown(f"""
    <div style="background:{COLORS['navy']};border-radius:10px;
         padding:2rem;text-align:center;margin-top:2rem;">
      <div style="font-family:'DM Serif Display',serif;color:white;
           font-size:1.35rem;margin-bottom:0.4rem;">Assignment Complete</div>
      <div style="color:#94A3B8;font-size:0.88rem;margin-bottom:0.8rem;">
        All questions submitted ·
        Final score:
        <strong style="color:white;">
          {fresh['total_score']} / {fresh['total_max']}
        </strong>
      </div>
      <div style="color:#64748B;font-size:0.8rem;">
        Your answers are locked. Return here anytime to review your solutions.
      </div>
    </div>
    """, unsafe_allow_html=True)

# ── Extension request ──────────────────────────────────────────────────────────
if past_hard and not review_mode:
    st.markdown("<br>", unsafe_allow_html=True)
    with st.expander("Request an extension"):
        st.markdown(
            '<div style="font-size:0.86rem;color:#6B7280;margin-bottom:0.5rem;">'
            'If you have a valid reason for a late submission, submit a request below. '
            'Your instructor will review and respond.</div>',
            unsafe_allow_html=True
        )
        reason = st.text_area("Reason for extension request",
                              placeholder="Please explain your circumstances...",
                              key="ext_reason")
        if st.button("Submit Extension Request", key="submit_ext"):
            if reason.strip():
                ok = submit_extension_request(email, hw_id, reason.strip())
                if ok:
                    log_audit(email, "EXTENSION_REQUEST", f"{hw_id}: {reason[:80]}")
                    st.success("Request submitted. Your instructor will be notified.")
                else:
                    st.error("Failed to submit request. Please email your instructor directly.")
            else:
                st.error("Please provide a reason.")

# ── Navigation ─────────────────────────────────────────────────────────────────
st.markdown("<br>", unsafe_allow_html=True)
if st.button("← Back to Dashboard"):
    st.session_state["submissions"] = get_student_submissions(email)
    st.switch_page("pages/Dashboard.py")
