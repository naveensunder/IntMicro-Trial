"""
Home.py — Login page.
HWDashboard v3 — stability-first build.
"""
import streamlit as st
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

from db import (
    init_sheets, get_enrollment_key, get_student,
    register_student, authenticate, update_password,
    check_login_attempts, record_failed_attempt, reset_login_attempts,
    get_student_submissions, get_homework_configs,
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
.block-container {{ max-width: 450px; padding-top: 2.5rem; }}
.login-title {{
    font-family: 'DM Serif Display', serif;
    font-size: 1.6rem; color: {COLORS['navy']};
    line-height: 1.25; margin-bottom: 0.2rem; text-align:center;
}}
.login-eye {{
    font-size: 0.67rem; font-weight: 600; letter-spacing: 0.14em;
    text-transform: uppercase; color: {COLORS['neutral_500']};
    margin-bottom: 0.35rem; text-align:center;
}}
.login-sub {{ font-size: 0.81rem; color: {COLORS['neutral_500']}; text-align:center; margin-bottom:1.6rem; }}
.inst-link {{ text-align:center; margin-top:1.8rem; font-size:0.73rem; }}
.inst-link a {{ color:#9CA3AF; text-decoration:none; }}
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def _init():
    try:
        init_sheets()
    except Exception:
        pass
_init()

st.markdown("""
<div class="mobile-warn">
  📱 For the best experience, please use a laptop or desktop.
</div>
<div class="login-eye">Interactive Homework Portal</div>
<div class="login-title">Learning Intermediate<br>Microeconomics</div>
<div class="login-sub">Prof. Naveen Sunder · Bentley University</div>
""", unsafe_allow_html=True)

if "login_flow" not in st.session_state:
    st.session_state["login_flow"] = "main"

flow = st.session_state["login_flow"]

# ════════════════════════════════════════════════════════════════════════════════
if flow == "main":
    tab_in, tab_reg = st.tabs(["Sign In", "Create Account"])

    with tab_in:
        with st.form("login_form"):
            em = st.text_input("University email", placeholder="you@university.edu")
            pw = st.text_input("Password", type="password")
            go = st.form_submit_button("Sign In", use_container_width=True)

        if go:
            if not check_login_attempts():
                pass
            elif not em.strip() or not pw:
                st.error("Please enter your email and password.")
            else:
                with st.spinner("Signing in..."):
                    ok, result = authenticate(em.strip().lower(), pw)
                if ok:
                    reset_login_attempts()
                    stu = result
                    st.session_state["authenticated"]  = True
                    st.session_state["student_email"]  = em.strip().lower()
                    st.session_state["student_name"]   = (
                        f"{stu.get('First_Name','')} {stu.get('Last_Name','')}".strip()
                        or em.strip()
                    )
                    st.session_state["student_record"] = stu
                    with st.spinner("Loading your progress..."):
                        st.session_state["submissions"] = get_student_submissions(em.strip().lower())
                        st.session_state["hw_configs"]  = get_homework_configs()
                    if str(stu.get("Force_Reset","")).upper() == "TRUE":
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

    with tab_reg:
        st.markdown(
            '<div style="font-size:0.81rem;color:#6B7280;margin-bottom:0.7rem;">'
            'You need the enrollment key from your instructor to register.</div>',
            unsafe_allow_html=True)
        with st.form("reg_form"):
            rf  = st.text_input("First name")
            rl  = st.text_input("Last name")
            re  = st.text_input("University email", placeholder="you@university.edu")
            rk  = st.text_input("Enrollment key")
            rp1 = st.text_input("Choose a password", type="password",
                                 help="Minimum 8 characters")
            rp2 = st.text_input("Confirm password", type="password")
            rs  = st.form_submit_button("Create Account", use_container_width=True)

        if rs:
            errs = []
            if not rf.strip(): errs.append("First name is required.")
            if not rl.strip(): errs.append("Last name is required.")
            if "@" not in re or "." not in re:
                errs.append("Please enter a valid university email.")
            if len(rp1) < 8: errs.append("Password must be at least 8 characters.")
            if rp1 != rp2:   errs.append("Passwords do not match.")
            if errs:
                for e in errs: st.error(e)
            else:
                with st.spinner("Checking enrollment key..."):
                    correct = get_enrollment_key()
                if rk.strip().upper() != correct.strip().upper():
                    st.error("Enrollment key is incorrect. Please check with your instructor.")
                else:
                    with st.spinner("Creating account..."):
                        ok, err = register_student(re.strip().lower(), rp1,
                                                   rf.strip(), rl.strip())
                    if ok:
                        st.success("Account created! You can now sign in.")
                        st.markdown(
                            '<div class="banner banner-warning">'
                            '📌 <strong>Important:</strong> Note down your password. '
                            'You can reset it with the enrollment key if needed.</div>',
                            unsafe_allow_html=True)
                    else:
                        st.error(f"Registration failed: {err}")

# ════════════════════════════════════════════════════════════════════════════════
elif flow == "reset_pw":
    st.markdown(
        f'<div style="font-family:\'DM Serif Display\',serif;font-size:1.1rem;'
        f'color:{COLORS["navy"]};margin-bottom:1rem;">Reset Password</div>',
        unsafe_allow_html=True)
    with st.form("reset_form"):
        rs_e  = st.text_input("Your university email")
        rs_k  = st.text_input("Enrollment key")
        rs_p1 = st.text_input("New password", type="password")
        rs_p2 = st.text_input("Confirm new password", type="password")
        rs_s  = st.form_submit_button("Reset Password", use_container_width=True)
    if rs_s:
        correct = get_enrollment_key()
        if rs_k.strip().upper() != correct.strip().upper():
            st.error("Incorrect enrollment key.")
        elif len(rs_p1) < 8:
            st.error("Password must be at least 8 characters.")
        elif rs_p1 != rs_p2:
            st.error("Passwords do not match.")
        elif not get_student(rs_e.strip().lower()):
            st.error("No account found with that email.")
        else:
            if update_password(rs_e.strip().lower(), rs_p1):
                st.success("Password updated. Please sign in.")
                st.session_state["login_flow"] = "main"
                st.rerun()
            else:
                st.error("Update failed. Please try again.")
    if st.button("← Back to sign in", key="back_reset"):
        st.session_state["login_flow"] = "main"
        st.rerun()

# ════════════════════════════════════════════════════════════════════════════════
elif flow == "force_reset":
    st.markdown(
        '<div class="banner banner-warning">Your instructor has requested a password reset. '
        'Please set a new password to continue.</div>',
        unsafe_allow_html=True)
    with st.form("force_form"):
        fp1 = st.text_input("New password", type="password")
        fp2 = st.text_input("Confirm new password", type="password")
        fs  = st.form_submit_button("Set Password & Continue", use_container_width=True)
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

st.markdown(
    '<div class="inst-link"><a href="/Instructor">Instructor access</a></div>',
    unsafe_allow_html=True)
