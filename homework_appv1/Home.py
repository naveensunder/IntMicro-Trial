import streamlit as st
import datetime
import gspread
from google.oauth2.service_account import Credentials
import json

st.set_page_config(
    page_title="Intermediate Microeconomics — Week 2 Assignment",
    page_icon="📘",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# ── Shared CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@400;600;700&family=Source+Sans+3:wght@300;400;500;600&display=swap');

html, body, [class*="css"] {
    font-family: 'Source Sans 3', sans-serif;
}

/* Hide default streamlit chrome */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
.stDeployButton {display: none;}
header {visibility: hidden;}

/* Remove top padding */
.block-container { padding-top: 2rem; padding-bottom: 2rem; max-width: 780px; }

/* ── Page header ── */
.page-header {
    text-align: center;
    padding: 2.5rem 2rem 1.5rem 2rem;
    background: linear-gradient(135deg, #0f2044 0%, #1a3a6b 60%, #1e4d8c 100%);
    border-radius: 16px;
    margin-bottom: 2rem;
    box-shadow: 0 8px 32px rgba(15,32,68,0.18);
}
.page-header h1 {
    font-family: 'Playfair Display', serif;
    color: #ffffff;
    font-size: 1.85rem;
    font-weight: 700;
    margin: 0 0 0.3rem 0;
    letter-spacing: 0.01em;
}
.page-header .subtitle {
    color: #a8c4e8;
    font-size: 0.95rem;
    font-weight: 300;
    letter-spacing: 0.04em;
    text-transform: uppercase;
}
.page-header .course-tag {
    display: inline-block;
    background: rgba(255,255,255,0.12);
    color: #cfe0f5;
    border-radius: 20px;
    padding: 3px 14px;
    font-size: 0.82rem;
    margin-top: 0.6rem;
    letter-spacing: 0.05em;
}

/* ── Info card ── */
.info-card {
    background: #f0f4ff;
    border: 1px solid #c5d3f0;
    border-left: 4px solid #1a3a6b;
    border-radius: 10px;
    padding: 1rem 1.3rem;
    margin-bottom: 1.4rem;
    font-size: 0.92rem;
    color: #2c3e6b;
    line-height: 1.7;
}

/* ── Section label ── */
.section-label {
    font-family: 'Playfair Display', serif;
    font-size: 1.1rem;
    font-weight: 600;
    color: #0f2044;
    margin-bottom: 0.5rem;
    padding-bottom: 0.3rem;
    border-bottom: 2px solid #e0e8f8;
}

/* ── Deadline banner ── */
.deadline-open {
    background: #e6f4ea;
    border: 1px solid #a8d5b0;
    border-radius: 10px;
    padding: 0.8rem 1.2rem;
    color: #1a5c30;
    font-size: 0.9rem;
    margin-bottom: 1.2rem;
    display: flex; align-items: center; gap: 8px;
}
.deadline-closed {
    background: #fdecea;
    border: 1px solid #f5a8a0;
    border-radius: 10px;
    padding: 0.8rem 1.2rem;
    color: #8b1a1a;
    font-size: 0.9rem;
    margin-bottom: 1.2rem;
    display: flex; align-items: center; gap: 8px;
}
.deadline-warning {
    background: #fff8e1;
    border: 1px solid #ffe082;
    border-radius: 10px;
    padding: 0.8rem 1.2rem;
    color: #7a5800;
    font-size: 0.9rem;
    margin-bottom: 1.2rem;
}

/* ── Progress bar ── */
.progress-wrap {
    background: #e8edf8;
    border-radius: 8px;
    padding: 1rem 1.3rem;
    margin-bottom: 1.4rem;
}
.progress-title {
    font-size: 0.85rem;
    color: #4a5980;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    margin-bottom: 0.6rem;
}
.progress-bar-outer {
    background: #cdd5ea;
    border-radius: 20px;
    height: 10px;
    width: 100%;
    overflow: hidden;
}
.progress-bar-inner {
    height: 100%;
    border-radius: 20px;
    background: linear-gradient(90deg, #1a3a6b, #3a72c4);
    transition: width 0.5s ease;
}
.progress-text {
    font-size: 0.82rem;
    color: #5a6a8a;
    margin-top: 0.4rem;
}

/* ── Confirm button ── */
.stButton > button {
    background: linear-gradient(135deg, #1a3a6b, #2a5ca8) !important;
    color: white !important;
    border: none !important;
    border-radius: 8px !important;
    padding: 0.6rem 2rem !important;
    font-family: 'Source Sans 3', sans-serif !important;
    font-weight: 600 !important;
    font-size: 0.95rem !important;
    letter-spacing: 0.03em !important;
    transition: all 0.2s ease !important;
    box-shadow: 0 3px 12px rgba(26,58,107,0.25) !important;
}
.stButton > button:hover {
    background: linear-gradient(135deg, #0f2044, #1a3a6b) !important;
    box-shadow: 0 5px 18px rgba(26,58,107,0.35) !important;
    transform: translateY(-1px) !important;
}

/* ── Text inputs ── */
.stTextInput > div > div > input {
    border: 1.5px solid #c5d3f0 !important;
    border-radius: 8px !important;
    font-family: 'Source Sans 3', sans-serif !important;
    font-size: 0.95rem !important;
    padding: 0.5rem 0.8rem !important;
    transition: border-color 0.2s !important;
}
.stTextInput > div > div > input:focus {
    border-color: #1a3a6b !important;
    box-shadow: 0 0 0 3px rgba(26,58,107,0.1) !important;
}

/* ── Restore banner ── */
.restore-banner {
    background: #e3f2fd;
    border: 1px solid #90caf9;
    border-radius: 10px;
    padding: 0.9rem 1.2rem;
    color: #0d47a1;
    font-size: 0.9rem;
    margin-bottom: 1rem;
}

</style>
""", unsafe_allow_html=True)


# ── Helpers ───────────────────────────────────────────────────────────────────
def get_sheet():
    """Authenticate with Google Sheets using service account from secrets."""
    try:
        creds_dict = dict(st.secrets["gcp_service_account"])
        creds = Credentials.from_service_account_info(
            creds_dict,
            scopes=[
                "https://www.googleapis.com/auth/spreadsheets",
                "https://www.googleapis.com/auth/drive",
            ],
        )
        gc = gspread.authorize(creds)
        sh = gc.open_by_key(st.secrets["SHEET_ID"])
        return sh.sheet1
    except Exception as e:
        st.error(f"Could not connect to Google Sheet: {e}")
        return None


def fetch_previous(ws, school_email):
    """Fetch most recent submission row per question for this student."""
    prev = {}
    try:
        rows = ws.get_all_records()
        for row in rows:
            if str(row.get("School_Email", "")).strip().lower() == school_email.strip().lower():
                qid = str(row.get("Question_ID", ""))
                if qid:
                    prev[qid] = row  # later rows overwrite → keeps most recent
    except Exception:
        pass
    return prev


def is_past_deadline():
    deadline_str = st.secrets.get("DEADLINE", "2099-12-31 23:59")
    try:
        deadline = datetime.datetime.strptime(deadline_str, "%Y-%m-%d %H:%M")
        return datetime.datetime.now() > deadline, deadline
    except Exception:
        return False, None


def deadline_banner(past, deadline_dt):
    now = datetime.datetime.now()
    if past:
        st.markdown(
            f'<div class="deadline-closed">🔒 <strong>Submission closed</strong> — '
            f'Deadline was {deadline_dt.strftime("%d %b %Y at %H:%M")}.</div>',
            unsafe_allow_html=True,
        )
    else:
        remaining = deadline_dt - now
        days = remaining.days
        hours, rem = divmod(remaining.seconds, 3600)
        mins = rem // 60
        if days == 0 and hours < 6:
            st.markdown(
                f'<div class="deadline-warning">⚠️ <strong>Deadline approaching!</strong> '
                f'Closes in {hours}h {mins}m — {deadline_dt.strftime("%d %b %Y at %H:%M")}.</div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f'<div class="deadline-open">✅ <strong>Submission open</strong> — '
                f'Closes {deadline_dt.strftime("%d %b %Y at %H:%M")} '
                f'({days}d {hours}h remaining).</div>',
                unsafe_allow_html=True,
            )


# ── Page ──────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="page-header">
  <h1>Intermediate Microeconomics</h1>
  <div class="subtitle">Graded Assignment — Week 2</div>
  <div class="course-tag">Prof. Naveen Sunder</div>
</div>
""", unsafe_allow_html=True)

past, deadline_dt = is_past_deadline()
if deadline_dt:
    deadline_banner(past, deadline_dt)

st.markdown("""
<div class="info-card">
  📋 <strong>Instructions</strong><br>
  Enter your details below to begin. Your answers are saved automatically when you
  click <em>Submit &amp; Save</em> on each question. You may close this page and
  return later — your progress will be restored from your school email.
  Each question can only be submitted <strong>once</strong>.
</div>
""", unsafe_allow_html=True)

st.markdown('<div class="section-label">👤 Your Details</div>', unsafe_allow_html=True)

with st.form("identity_form"):
    name = st.text_input("Full Name", placeholder="e.g. Jane Smith",
                         value=st.session_state.get("student_name", ""))
    email = st.text_input("School Email", placeholder="e.g. jsmith@university.edu",
                          value=st.session_state.get("student_email", ""))
    submitted = st.form_submit_button("Confirm & Load My Progress →", use_container_width=True)

if submitted:
    name = name.strip()
    email = email.strip().lower()
    if not name:
        st.error("Please enter your full name.")
    elif "@" not in email or "." not in email:
        st.error("That email address doesn't look right. Please check.")
    else:
        with st.spinner("Loading your progress..."):
            ws = get_sheet()
            if ws is not None:
                prev = fetch_previous(ws, email)
                st.session_state["student_name"]  = name
                st.session_state["student_email"] = email
                st.session_state["confirmed"]      = True
                st.session_state["past_deadline"]  = past
                st.session_state["prev_data"]      = prev
                st.session_state["ws"]             = None  # don't cache ws object

                restored = list(prev.keys())
                if restored:
                    st.markdown(
                        f'<div class="restore-banner">📄 Previous progress found for: '
                        f'<strong>{", ".join(restored)}</strong>. '
                        f'Your saved answers will be pre-filled on the question pages.</div>',
                        unsafe_allow_html=True,
                    )
                st.success(f"✓ Welcome, {name}! Navigate to the questions using the sidebar.")
                st.balloons()

# ── Progress summary (if confirmed) ──────────────────────────────────────────
if st.session_state.get("confirmed"):
    prev = st.session_state.get("prev_data", {})
    total_q = 2
    done_q  = sum(1 for qid in ["Q3","Q9"]
                  if str(prev.get(qid, {}).get("Status","")) == "submitted")
    pct = int(done_q / total_q * 100)

    st.markdown(f"""
    <div class="progress-wrap">
      <div class="progress-title">Assignment Progress</div>
      <div class="progress-bar-outer">
        <div class="progress-bar-inner" style="width:{pct}%"></div>
      </div>
      <div class="progress-text">{done_q} of {total_q} questions submitted</div>
    </div>
    """, unsafe_allow_html=True)

    if done_q == total_q:
        st.markdown("""
        <div style="background:#e6f4ea;border:1.5px solid #66bb6a;border-radius:12px;
             padding:1.2rem 1.5rem;text-align:center;margin-top:0.5rem;">
          <div style="font-family:'Playfair Display',serif;font-size:1.2rem;
               color:#1a5c30;font-weight:700;">🎉 Assignment Complete!</div>
          <div style="color:#2d7a45;font-size:0.9rem;margin-top:0.3rem;">
            Both questions have been submitted. You're all done.
          </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(
            "👈 **Use the sidebar** to navigate to **Q3** and **Q9**.",
            unsafe_allow_html=True,
        )
