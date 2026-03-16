"""
Home.py — Login page.
HWDashboard v2 — Phase 1
"""
import streamlit as st
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

from db import (
    init_sheets, get_enrollment_key, get_student, register_student,
    authenticate_student, update_password,
    check_login_attempts, record_failed_attempt, reset_login_attempts,
    get_student_submissions, get_homework_configs, check_auto_enable,
)
from ui import inject_css, COLORS

st.set_page_config(
    page_title="Learning Intermediate Microeconomics",
    page_icon="📘",
    layout="centered",
    initial_sidebar_state="collapsed",
)

inject_css()

st.markdown(f"""
<style>
.block-container {{ max-width: 460px; padding-top: 2.5rem; }}
.login-wrap {{ text-align: center; margin-bottom: 1.8rem; }}
.login-title {{
    font-family: 'DM Serif Display', serif;
    font-size: 1.65rem;
    color: {COLORS['navy']};
    line-height: 1.25;
    margin-bottom: 0.2rem;
}}
.login-eyebrow {{
    font-size: 0.68rem; font-weight: 600; letter-spacing: 0.14em;
    text-transform: uppercase; color: {COLORS['neutral_400']};
    margin-bottom: 0.4rem;
}}
.login-sub {{ font-size: 0.82rem; color: {COLORS['neutral_500']}; }}
.instructor-link {{
    text-align: center; margin-top: 1.8rem;
    font-size: 0.74rem; color: {COLORS['neutral_400']};
}}
.instructor-link a {{
    color: {COLORS['neutral_400']} !important;
    text-decoration: none;
}}
.instructor-link a:hover {{ color: {COLORS['neutral_500']} !important; }}
</style>
""", unsafe_allow_html=True)

# ── Initialise sheets once ─────────────────────────────────────────────────────
@st.cache_resource
def _init():
    try:
        init_sheets()
        check_auto_enable()
    except Exception:
        pass
_init()

# ── Title ──────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="login-wrap">
  <div class="login-eyebrow">Interactive Homework Portal</div>
  <div class="login-title">Learning Intermediate<br>Microeconomics</div>
  <div class="login-sub">Prof. Naveen Sunder</div>
</div>
<div class="mobile-warning">
  📱 For the best experience, please use a laptop or desktop.
</div>
""", unsafe_allow_html=True)

# ── Flow ───────────────────────────────────────────────────────────────────────
if "login_flow" not in st.session_state:
    st.session_state["login_flow"] = "main"

flow = st.session_state["login_flow"]

# ════════════════════════════════════════════════════════════════════════════════
if flow == "main":
    tab_in, tab_reg = st.tabs(["Sign In", "Create Account"])

    # ── Sign in ────────────────────────────────────────────────────────────────
    with tab_in:
        with st.form("login_form"):
            em_in = st.text_input("University email",
                                  placeholder="you@university.edu")
            pw_in = st.text_input("Password", type="password")
            sub   = st.form_submit_button("Sign In",
                                          use_container_width=True)

        if sub:
            if not check_login_attempts():
                pass
            elif not em_in.strip() or not pw_in:
                st.error("Please enter your email and password.")
            else:
                with st.spinner("Signing in..."):
                    ok, result = authenticate_student(
                        em_in.strip().lower(), pw_in)
                if ok:
                    reset_login_attempts()
                    student = result
                    st.session_state["authenticated"]  = True
                    st.session_state["student_email"]  = em_in.strip().lower()
                    st.session_state["student_name"]   = (
                        f"{student.get('First_Name','')} "
                        f"{student.get('Last_Name','')}".strip()
                        or em_in.strip()
                    )
                    st.session_state["student_record"] = student
                    with st.spinner("Loading your progress..."):
                        st.session_state["submissions"] = get_student_submissions(
                            em_in.strip().lower())
                        st.session_state["hw_configs"]  = get_homework_configs()
                    if str(student.get("Force_Reset","")).upper() == "TRUE":
                        st.session_state["login_flow"] = "force_reset"
                        st.rerun()
                    else:
                        st.switch_page("pages/Dashboard.py")
                else:
                    record_failed_attempt()
                    st.error(result)

        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("Forgot password? Reset using enrollment key",
                     use_container_width=True, key="go_reset"):
            st.session_state["login_flow"] = "reset_pw"
            st.rerun()

    # ── Create account ─────────────────────────────────────────────────────────
    with tab_reg:
        st.markdown(
            '<div style="font-size:0.82rem;color:#6B7280;margin-bottom:0.8rem;">'
            'You need the enrollment key from your instructor to register.</div>',
            unsafe_allow_html=True)

        with st.form("reg_form"):
            r_first  = st.text_input("First name")
            r_last   = st.text_input("Last name")
            r_email  = st.text_input("University email",
                                     placeholder="you@university.edu")
            r_key    = st.text_input("Enrollment key")
            r_pw1    = st.text_input("Choose a password", type="password",
                                     help="Minimum 8 characters")
            r_pw2    = st.text_input("Confirm password", type="password")
            r_sub    = st.form_submit_button("Create Account",
                                             use_container_width=True)

        if r_sub:
            errs = []
            if not r_first.strip(): errs.append("First name is required.")
            if not r_last.strip():  errs.append("Last name is required.")
            if "@" not in r_email or "." not in r_email:
                errs.append("Please enter a valid university email.")
            if len(r_pw1) < 8:
                errs.append("Password must be at least 8 characters.")
            if r_pw1 != r_pw2:
                errs.append("Passwords do not match.")

            if errs:
                for e in errs:
                    st.error(e)
            else:
                with st.spinner("Checking enrollment key..."):
                    correct_key = get_enrollment_key()
                if r_key.strip().upper() != correct_key.strip().upper():
                    st.error("Enrollment key is incorrect. "
                             "Please check with your instructor.")
                else:
                    with st.spinner("Creating your account..."):
                        ok, err = register_student(
                            r_email.strip().lower(), r_pw1,
                            r_first.strip(), r_last.strip()
                        )
                    if ok:
                        st.success(
                            "Account created! You can now sign in.")
                        st.markdown("""
                        <div class="banner-warning">
                          📌 <strong>Important:</strong> Note down your password.
                          You can reset it with the enrollment key if needed.
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.error(f"Registration failed: {err}")

# ════════════════════════════════════════════════════════════════════════════════
elif flow == "reset_pw":
    st.markdown(
        f'<div style="font-family:\'DM Serif Display\',serif;font-size:1.1rem;'
        f'color:{COLORS["navy"]};margin-bottom:1rem;">Reset Password</div>',
        unsafe_allow_html=True)

    with st.form("reset_form"):
        rs_email = st.text_input("Your university email")
        rs_key   = st.text_input("Enrollment key")
        rs_pw1   = st.text_input("New password", type="password")
        rs_pw2   = st.text_input("Confirm new password", type="password")
        rs_sub   = st.form_submit_button("Reset Password",
                                         use_container_width=True)

    if rs_sub:
        correct_key = get_enrollment_key()
        if rs_key.strip().upper() != correct_key.strip().upper():
            st.error("Incorrect enrollment key.")
        elif len(rs_pw1) < 8:
            st.error("Password must be at least 8 characters.")
        elif rs_pw1 != rs_pw2:
            st.error("Passwords do not match.")
        elif not get_student(rs_email.strip().lower()):
            st.error("No account found with that email.")
        else:
            if update_password(rs_email.strip().lower(), rs_pw1):
                st.success("Password updated. Please sign in.")
                st.session_state["login_flow"] = "main"
                st.rerun()
            else:
                st.error("Update failed. Please try again.")

    if st.button("← Back to sign in", key="back_from_reset"):
        st.session_state["login_flow"] = "main"
        st.rerun()

# ════════════════════════════════════════════════════════════════════════════════
elif flow == "force_reset":
    st.markdown(
        '<div class="banner-warning">Your instructor has requested a password reset. '
        'Please set a new password to continue.</div>',
        unsafe_allow_html=True)

    with st.form("force_reset_form"):
        fp1 = st.text_input("New password", type="password")
        fp2 = st.text_input("Confirm new password", type="password")
        fs  = st.form_submit_button("Set Password & Continue",
                                    use_container_width=True)
    if fs:
        if len(fp1) < 8:
            st.error("Password must be at least 8 characters.")
        elif fp1 != fp2:
            st.error("Passwords do not match.")
        else:
            if update_password(st.session_state.get("student_email",""), fp1):
                st.session_state["login_flow"] = "main"
                st.switch_page("pages/Dashboard.py")
            else:
                st.error("Failed. Please try again.")

# ── Instructor link ────────────────────────────────────────────────────────────
st.markdown("""
<div class="instructor-link">
  <a href="/Instructor">Instructor access</a>
</div>
""", unsafe_allow_html=True)
