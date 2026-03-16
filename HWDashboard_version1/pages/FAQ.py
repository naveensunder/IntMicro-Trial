"""
pages/FAQ.py
HWDashboard v3
"""
import streamlit as st
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from ui import inject_css, page_header

st.set_page_config(page_title="FAQ — Microeconomics", page_icon="❓",
                   layout="centered", initial_sidebar_state="expanded")
inject_css()

if not st.session_state.get("authenticated"):
    st.warning("Please sign in first.")
    if st.button("Go to sign in"): st.switch_page("Home.py")
    st.stop()

page_header("Intermediate Microeconomics", "Frequently Asked Questions")

FAQS = [
    ("Getting Started", [
        ("How do I create an account?",
         "Click 'Create Account' on the login page. You need the enrollment key from your instructor, your university email, and a password of at least 8 characters."),
        ("I forgot my password. What do I do?",
         "Click 'Forgot password?' on the login page. You need the enrollment key and your university email to set a new password."),
        ("What is the enrollment key?",
         "A code your instructor shares at the start of semester. Required to register or reset your password. Contact your instructor if you have lost it."),
        ("Can I use my phone?",
         "The app works on mobile but is designed for laptop or desktop. On a small screen some layouts may be hard to read."),
    ]),
    ("Completing Assignments", [
        ("Why are my question numbers different from my classmates?",
         "Each student receives unique parameter values (numbers) for each question, generated from your email address. The method to solve is identical for everyone. This ensures academic integrity."),
        ("The Submit button is not appearing.",
         "The Submit button only appears once all answer boxes have a non-zero value. Make sure every input field has been filled in."),
        ("I submitted a wrong answer by mistake. Can I redo it?",
         "No — each question is locked after submission. If you believe there is a technical error, contact your instructor at nsunder@bentley.edu."),
        ("I closed the tab and my answers are gone.",
         "Only submitted answers are permanently saved. Answers you typed but did not submit are lost when you close the tab. Always click Submit before leaving."),
        ("How precise do my answers need to be?",
         "Answers are rounded to 2 decimal places before grading. For example, 6.67 and 6.6667 will both be accepted if the correct answer rounds to 6.67."),
    ]),
    ("Grades and Scores", [
        ("How is my score calculated?",
         "Each question part is graded automatically after submission. You see the correct answer, your answer, and your score immediately. Answers are compared after rounding to 2 decimal places."),
        ("Where can I see my total score?",
         "Your semester score is shown at the top of the Dashboard page. It updates as you submit questions."),
        ("I got a wrong answer but think my working was correct.",
         "Review the step-by-step solution shown after submission. If you believe there is an error, contact your instructor at nsunder@bentley.edu with your name and the specific question."),
        ("Can I see my answers after the deadline?",
         "Yes. Submitted homework always shows your answers and the correct solutions. Go to the Dashboard and open the homework to review."),
    ]),
    ("Technical Issues", [
        ("The page is loading very slowly.",
         "The app connects to Google Sheets to save your data. This can occasionally be slow. Wait a few seconds and try refreshing. Avoid clicking buttons multiple times."),
        ("I see an error message instead of the questions.",
         "Try refreshing the page first. If it persists, sign out and sign back in. If still not working, contact your instructor with a screenshot."),
        ("I cannot sign in even though my password is correct.",
         "After 5 failed login attempts the system locks you out for 5 minutes. Wait 5 minutes and try again, or use the password reset option."),
    ]),
    ("Deadlines", [
        ("What happens if I miss the deadline?",
         "After the deadline the assignment closes. There is a short grace period. If you have extenuating circumstances, email your instructor at nsunder@bentley.edu before the deadline if possible."),
        ("My submission shows as late. Is that a problem?",
         "Late submissions are flagged automatically. Your instructor will see this. Contact them directly if you had a valid reason."),
    ]),
]

for section, items in FAQS:
    st.markdown(
        f'<div style="font-size:0.68rem;font-weight:600;letter-spacing:0.1em;'
        f'text-transform:uppercase;color:#6B7280;margin:1.5rem 0 0.6rem 0;'
        f'padding-bottom:0.3rem;border-bottom:1px solid #E5E7EB;">{section}</div>',
        unsafe_allow_html=True
    )
    for q, a in items:
        with st.expander(q):
            st.markdown(f'<div class="faq-a">{a}</div>', unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)
st.markdown(
    '<div style="background:#F9FAFB;border:1px solid #E5E7EB;border-radius:7px;'
    'padding:0.9rem 1.1rem;font-size:0.84rem;color:#6B7280;">'
    'Still have a question? Email <a href="mailto:nsunder@bentley.edu" '
    'style="color:#2563EB;">nsunder@bentley.edu</a></div>',
    unsafe_allow_html=True
)
