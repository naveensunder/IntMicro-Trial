"""
Home.py — Login page for Learning Intermediate Microeconomics.
"""
import streamlit as st
import datetime
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

from db import (
    init_sheets, get_enrollment_key, get_student, register_student,
    authenticate_student, update_password, check_login_attempts,
    record_failed_attempt, reset_login_attempts, verify_password, hash_password
)
from ui import inject_css, COLORS

st.set_page_config(
    page_title="Learning Intermediate Microeconomics",
    page_icon="📘",
    layout="centered",
    initial_sidebar_state="collapsed",
)

inject_css()

# ── Extra login-page CSS ───────────────────────────────────────────────────────
st.markdown(f"""
<style>
.block-container {{ max-width: 480px; padding-top: 3rem; }}
.login-logo {{
    text-align: center;
    margin-bottom: 2rem;
}}
.login-logo .course-label {{
    font-size: 0.7rem;
    font-weight: 600;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    color: {COLORS['neutral_500']};
    margin-bottom: 0.5rem;
}}
.login-logo .course-title {{
    font-family: 'DM Serif Display', serif;
    font-size: 1.75rem;
    color: {COLORS['navy']};
    line-height: 1.25;
}}
.login-logo .course-sub {{
    font-size: 0.84rem;
    color: {COLORS['neutral_500']};
    margin-top: 0.3rem;
}}
.login-card {{
    background: {COLORS['white']};
    border: 1px solid {COLORS['neutral_200']};
    border-radius: 12px;
    padding: 2rem;
}}
.login-divider {{
    text-align: center;
    margin: 1.5rem 0 0.5rem 0;
    font-size: 0.78rem;
    color: {COLORS['neutral_500']};
}}
.instructor-link {{
    text-align: center;
    margin-top: 2rem;
    font-size: 0.78rem;
    color: {COLORS['neutral_500']};
}}
.step-indicator {{
    display: flex;
    justify-content: center;
    gap: 0.5rem;
    margin-bottom: 1.5rem;
}}
.step-dot {{
    width: 8px; height: 8px;
    border-radius: 50%;
    background: {COLORS['neutral_300']};
}}
.step-dot-active {{
    background: {COLORS['navy']};
}}
</style>
""", unsafe_allow_html=True)

# ── Initialise sheets on first run ────────────────────────────────────────────
@st.cache_resource
def _init():
    try:
        init_sheets()
    except Exception:
        pass
_init()

# ── Logo / title ──────────────────────────────────────────────────────────────
st.markdown("""
<div class="login-logo">
  <div class="course-label">Interactive Homework Portal</div>
  <div class="course-title">Learning Intermediate<br>Microeconomics</div>
  <div class="course-sub">Prof. Naveen Sunder</div>
</div>
<div class="mobile-warning">
  📱 For the best experience, please use a laptop or desktop computer.
</div>
""", unsafe_allow_html=True)

# ── Session flow state ────────────────────────────────────────────────────────
if "login_flow" not in st.session_state:
    st.session_state["login_flow"] = "main"  # main | register_step2 | reset_pw

flow = st.session_state["login_flow"]

# ════════════════════════════════════════════════════════════════════════════════
#  MAIN LOGIN / REGISTER CHOICE
# ════════════════════════════════════════════════════════════════════════════════
if flow == "main":
    tab_login, tab_register = st.tabs(["Sign In", "Create Account"])

    # ── Sign In ────────────────────────────────────────────────────────────────
    with tab_login:
        with st.form("login_form"):
            st.markdown('<div style="font-size:0.85rem;font-weight:600;color:#374151;'
                        'margin-bottom:0.8rem;">Welcome back</div>', unsafe_allow_html=True)
            email_in = st.text_input("Email address", placeholder="you@university.edu")
            pw_in    = st.text_input("Password", type="password")
            submit   = st.form_submit_button("Sign In", use_container_width=True)

        if submit:
            if not check_login_attempts():
                pass
            elif not email_in or not pw_in:
                st.error("Please enter your email and password.")
            else:
                with st.spinner("Signing in..."):
                    ok, result = authenticate_student(email_in.strip().lower(), pw_in)
                if ok:
                    reset_login_attempts()
                    student = result
                    st.session_state["authenticated"]   = True
                    st.session_state["student_email"]   = email_in.strip().lower()
                    st.session_state["student_name"]    = (
                        f"{student.get('First_Name','')} {student.get('Last_Name','')}".strip()
                        or email_in
                    )
                    st.session_state["student_record"]  = student
                    # Load submissions
                    from db import get_student_submissions
                    st.session_state["submissions"] = get_student_submissions(
                        email_in.strip().lower()
                    )
                    # Load hw configs
                    from db import get_homework_configs
                    st.session_state["hw_configs"] = get_homework_configs()

                    if str(student.get("Force_Reset", "")).upper() == "TRUE":
                        st.session_state["login_flow"] = "force_reset"
                        st.rerun()
                    else:
                        st.switch_page("pages/Dashboard.py")
                else:
                    record_failed_attempt()
                    st.error(result)

        st.markdown('<div class="login-divider">Forgot your password?</div>', unsafe_allow_html=True)
        if st.button("Reset password using enrollment key", use_container_width=True,
                     key="go_reset"):
            st.session_state["login_flow"] = "reset_pw"
            st.rerun()

    # ── Create Account ─────────────────────────────────────────────────────────
    with tab_register:
        st.markdown('<div style="font-size:0.85rem;color:#6B7280;margin-bottom:1rem;">'
                    'You will need the enrollment key provided by your instructor.</div>',
                    unsafe_allow_html=True)
        with st.form("register_form"):
            first_name  = st.text_input("First name")
            last_name   = st.text_input("Last name")
            reg_email   = st.text_input("University email", placeholder="you@university.edu")
            enroll_key  = st.text_input("Enrollment key", placeholder="Provided by instructor")
            new_pw      = st.text_input("Choose a password", type="password",
                                        help="Minimum 8 characters")
            confirm_pw  = st.text_input("Confirm password", type="password")
            reg_submit  = st.form_submit_button("Create Account", use_container_width=True)

        if reg_submit:
            errors = []
            if not first_name.strip(): errors.append("First name is required.")
            if not last_name.strip():  errors.append("Last name is required.")
            if "@" not in reg_email or "." not in reg_email:
                errors.append("Please enter a valid university email.")
            if len(new_pw) < 8: errors.append("Password must be at least 8 characters.")
            if new_pw != confirm_pw: errors.append("Passwords do not match.")

            if not errors:
                with st.spinner("Checking enrollment key..."):
                    correct_key = get_enrollment_key()
                if enroll_key.strip().upper() != correct_key.strip().upper():
                    st.error("Enrollment key is incorrect. Please check with your instructor.")
                elif errors:
                    for e in errors: st.error(e)
                else:
                    with st.spinner("Creating your account..."):
                        ok, err = register_student(
                            reg_email.strip().lower(), new_pw,
                            first_name.strip(), last_name.strip()
                        )
                    if ok:
                        st.success("Account created! Please note down your password — "
                                   "you will need it each time you sign in.")
                        st.markdown("""
                        <div style="background:#FFFBEB;border:1px solid #FDE68A;border-radius:8px;
                             padding:0.8rem 1rem;font-size:0.88rem;color:#92400E;margin-top:0.5rem;">
                          📌 <strong>Important:</strong> Write down or save your password now.
                          You can reset it using the enrollment key if needed.
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.error(f"Registration failed: {err}")
            else:
                for e in errors: st.error(e)

# ════════════════════════════════════════════════════════════════════════════════
#  PASSWORD RESET
# ════════════════════════════════════════════════════════════════════════════════
elif flow == "reset_pw":
    st.markdown('<div style="font-family:\'DM Serif Display\',serif;font-size:1.1rem;'
                'color:#1C2B4A;margin-bottom:1.2rem;">Reset Password</div>',
                unsafe_allow_html=True)

    with st.form("reset_form"):
        reset_email = st.text_input("Your university email")
        reset_key   = st.text_input("Enrollment key")
        reset_pw1   = st.text_input("New password", type="password")
        reset_pw2   = st.text_input("Confirm new password", type="password")
        reset_sub   = st.form_submit_button("Reset Password", use_container_width=True)

    if reset_sub:
        correct_key = get_enrollment_key()
        if reset_key.strip().upper() != correct_key.strip().upper():
            st.error("Incorrect enrollment key.")
        elif len(reset_pw1) < 8:
            st.error("Password must be at least 8 characters.")
        elif reset_pw1 != reset_pw2:
            st.error("Passwords do not match.")
        elif not get_student(reset_email.strip().lower()):
            st.error("No account found with that email.")
        else:
            ok = update_password(reset_email.strip().lower(), reset_pw1)
            if ok:
                st.success("Password updated. Please sign in.")
                st.session_state["login_flow"] = "main"
                st.rerun()
            else:
                st.error("Update failed. Please try again.")

    if st.button("← Back to sign in"):
        st.session_state["login_flow"] = "main"
        st.rerun()

# ════════════════════════════════════════════════════════════════════════════════
#  FORCE PASSWORD RESET (admin-triggered)
# ════════════════════════════════════════════════════════════════════════════════
elif flow == "force_reset":
    st.markdown('<div class="banner-warning">Your instructor has requested a password reset.'
                ' Please set a new password to continue.</div>', unsafe_allow_html=True)
    with st.form("force_reset_form"):
        fp1 = st.text_input("New password", type="password")
        fp2 = st.text_input("Confirm new password", type="password")
        fs  = st.form_submit_button("Set Password & Continue", use_container_width=True)
    if fs:
        if len(fp1) < 8:
            st.error("Password must be at least 8 characters.")
        elif fp1 != fp2:
            st.error("Passwords do not match.")
        else:
            ok = update_password(st.session_state.get("student_email", ""), fp1)
            if ok:
                st.session_state["login_flow"] = "main"
                st.switch_page("pages/Dashboard.py")
            else:
                st.error("Failed to update password. Please try again.")

# ── Instructor link (subtle, bottom of page) ──────────────────────────────────
st.markdown("""
<div class="instructor-link">
  <a href="/Instructor" style="color:#9CA3AF;text-decoration:none;font-size:0.76rem;">
    Instructor access
  </a>
</div>
""", unsafe_allow_html=True)
