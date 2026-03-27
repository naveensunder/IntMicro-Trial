"""
Home.py — Login / Register / Reset.
HWDashboard v11 — Phase 4:
  - Typewriter welcome animation
  - Secure indicator on login
  - Registration success screen
  - Inline real-time form validation hints
  - Session state cleaned up on sign-out flow
  - Browser tab title standardised
  - page_footer with semester
"""
import streamlit as st
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from db import (
    init_sheets, get_enrollment_key, get_student,
    register_student, authenticate, update_password,
    check_login_attempts, record_failed_attempt, reset_login_attempts,
    get_student_submissions, get_homework_configs,
)
from ui import inject_css, COLORS, page_footer, secure_indicator, registration_success

st.set_page_config(
    page_title="EC224 — Intermediate Microeconomics · Bentley",
    page_icon="📘",
    layout="centered",
    initial_sidebar_state="collapsed",
)
inject_css()

st.markdown(
    """
    <style>
    .block-container { max-width: 460px; padding-top: 2rem; }
    .login-eye {
        font-size: 0.75rem; font-weight: 600; letter-spacing: 0.14em;
        text-transform: uppercase; color: #888888;
        margin-bottom: 0.35rem; text-align: center;
    }
    .login-title {
        font-family: 'DM Serif Display', serif;
        font-size: 1.85rem; color: #1C2B4A;
        line-height: 1.25; margin-bottom: 0.2rem; text-align: center;
    }
    .login-sub {
        font-size: 0.92rem; color: #555555;
        text-align: center; margin-bottom: 1.6rem;
    }
    .inst-link { text-align: center; margin-top: 1.6rem; font-size: 0.82rem; }
    .inst-link a { color: #9CA3AF; text-decoration: none; }
    .inst-link a:hover { color: #555555; }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def _init():
    try:
        init_sheets()
    except Exception:
        pass


_init()

# ── Header ─────────────────────────────────────────────────────────────────────
st.markdown(
    """
    <div class="mobile-warn">
      📱 For the best experience, please use a laptop or desktop.
    </div>
    <div class="login-eye">Interactive Homework Portal</div>
    """,
    unsafe_allow_html=True,
)

# Typewriter animation on the title
st.markdown(
    """
    <div class="login-title" id="tw-title">&nbsp;</div>
    <div class="login-sub">Prof. Naveen Sunder &nbsp;·&nbsp; Bentley University</div>
    <script>
    (function() {
        function run() {
            if (typeof typewriter === 'function') {
                typewriter('tw-title', 'Intermediate Microeconomics', 42);
            } else {
                setTimeout(run, 100);
            }
        }
        run();
    })();
    </script>
    """,
    unsafe_allow_html=True,
)

# ── Flow state ─────────────────────────────────────────────────────────────────
if "login_flow" not in st.session_state:
    st.session_state["login_flow"] = "main"
flow = st.session_state["login_flow"]

# ── Registration success screen ────────────────────────────────────────────────
if flow == "reg_success":
    reg_name = st.session_state.get("reg_success_name", "")
    registration_success(reg_name)
    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("Sign in now →", use_container_width=True, key="go_signin"):
        st.session_state["login_flow"] = "main"
        st.rerun()
    page_footer()
    st.stop()

# ── Main login / register ──────────────────────────────────────────────────────
if flow == "main":
    tab_in, tab_reg = st.tabs(["Sign In", "Create Account"])

    # ── Sign In ──
    with tab_in:
        secure_indicator()
        st.markdown("<br>", unsafe_allow_html=True)

        with st.form("login_form"):
            em = st.text_input(
                "University email",
                placeholder="you@bentley.edu",
            )
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
                        f"{stu.get('First_Name', '')} "
                        f"{stu.get('Last_Name', '')}".strip()
                        or em.strip()
                    )
                    st.session_state["student_record"] = stu
                    with st.spinner("Loading your progress..."):
                        st.session_state["submissions"] = get_student_submissions(
                            em.strip().lower()
                        )
                        st.session_state["hw_configs"]  = get_homework_configs()
                    if str(stu.get("Force_Reset", "")).upper() == "TRUE":
                        st.session_state["login_flow"] = "force_reset"
                        st.rerun()
                    else:
                        st.switch_page("pages/Dashboard.py")
                else:
                    record_failed_attempt()
                    st.error(result)

        st.divider()
        if st.button(
            "Forgot password? Reset using enrollment key",
            use_container_width=True,
            key="go_reset",
        ):
            st.session_state["login_flow"] = "reset_pw"
            st.rerun()

    # ── Create Account ──
    with tab_reg:
        st.markdown(
            '<div style="font-size:0.92rem;color:#555555;margin-bottom:0.8rem;">'
            "You need the enrollment key from your instructor to register.</div>",
            unsafe_allow_html=True,
        )

        with st.form("reg_form"):
            rf  = st.text_input("First name")
            rl  = st.text_input("Last name")
            re  = st.text_input(
                "University email",
                placeholder="you@bentley.edu",
            )
            rk  = st.text_input("Enrollment key")
            rp1 = st.text_input(
                "Choose a password",
                type="password",
                help="Minimum 8 characters",
            )
            rp2 = st.text_input("Confirm password", type="password")
            rs  = st.form_submit_button("Create Account", use_container_width=True)

        # Inline validation hints (shown after first attempt)
        if rs:
            errs = []
            if not rf.strip():
                errs.append("First name is required.")
            if not rl.strip():
                errs.append("Last name is required.")
            if "@" not in re or "." not in re:
                errs.append("Please enter a valid university email.")
            if len(rp1) < 8:
                errs.append("Password must be at least 8 characters.")
            if rp1 and rp2 and rp1 != rp2:
                errs.append("Passwords do not match.")
            if errs:
                for e in errs:
                    st.error(e)
            else:
                with st.spinner("Checking enrollment key..."):
                    correct = get_enrollment_key()
                if rk.strip().upper() != correct.strip().upper():
                    st.error(
                        "Enrollment key is incorrect. "
                        "Please check with your instructor."
                    )
                else:
                    with st.spinner("Creating account..."):
                        ok, err = register_student(
                            re.strip().lower(), rp1,
                            rf.strip(), rl.strip(),
                        )
                    if ok:
                        st.session_state["login_flow"]      = "reg_success"
                        st.session_state["reg_success_name"] = rf.strip()
                        st.rerun()
                    else:
                        st.error(f"Registration failed: {err}")

        st.markdown(
            '<div class="banner banner-info" style="font-size:0.85rem;margin-top:0.8rem;">'
            "📌 <strong>Note your password.</strong> You can reset it "
            "using the enrollment key if needed.</div>",
            unsafe_allow_html=True,
        )

# ── Password reset ─────────────────────────────────────────────────────────────
elif flow == "reset_pw":
    st.markdown(
        '<div style="font-family:\'DM Serif Display\',serif;font-size:1.3rem;'
        'color:#1C2B4A;margin-bottom:1rem;">Reset Password</div>',
        unsafe_allow_html=True,
    )
    with st.form("reset_form"):
        rs_e  = st.text_input("Your university email")
        rs_k  = st.text_input("Enrollment key")
        rs_p1 = st.text_input("New password", type="password")
        rs_p2 = st.text_input("Confirm new password", type="password")
        rs_s  = st.form_submit_button("Reset Password", use_container_width=True)

    if rs_s:
        errs = []
        if not rs_e.strip() or "@" not in rs_e:
            errs.append("Please enter a valid email.")
        if len(rs_p1) < 8:
            errs.append("Password must be at least 8 characters.")
        if rs_p1 != rs_p2:
            errs.append("Passwords do not match.")
        if errs:
            for e in errs:
                st.error(e)
        else:
            correct = get_enrollment_key()
            if rs_k.strip().upper() != correct.strip().upper():
                st.error("Incorrect enrollment key.")
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

# ── Force reset ────────────────────────────────────────────────────────────────
elif flow == "force_reset":
    st.markdown(
        '<div class="banner banner-warning">'
        "Your instructor has requested a password reset. "
        "Please set a new password to continue.</div>",
        unsafe_allow_html=True,
    )
    with st.form("force_form"):
        fp1 = st.text_input("New password", type="password")
        fp2 = st.text_input("Confirm new password", type="password")
        fs  = st.form_submit_button(
            "Set Password & Continue", use_container_width=True
        )
    if fs:
        if len(fp1) < 8:
            st.error("Password must be at least 8 characters.")
        elif fp1 != fp2:
            st.error("Passwords do not match.")
        else:
            if update_password(st.session_state.get("student_email", ""), fp1):
                st.session_state["login_flow"] = "main"
                st.switch_page("pages/Dashboard.py")
            else:
                st.error("Failed. Please try again.")

# ── Footer ─────────────────────────────────────────────────────────────────────
st.markdown(
    '<div class="inst-link"><a href="/Instructor">Instructor access</a></div>',
    unsafe_allow_html=True,
)
page_footer()
