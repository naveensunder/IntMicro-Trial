"""
pages/FAQ.py — Student FAQ page.
HWDashboard v2 — Phase 1
"""
import streamlit as st
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from ui import inject_css, page_header

st.set_page_config(
    page_title="FAQ — Microeconomics",
    page_icon="❓",
    layout="centered",
    initial_sidebar_state="expanded",
)

inject_css()

if not st.session_state.get("authenticated"):
    st.warning("Please sign in first.")
    if st.button("Go to sign in"):
        st.switch_page("Home.py")
    st.stop()

page_header("Intermediate Microeconomics", "Frequently Asked Questions", "")

FAQS = [
    {
        "category": "Getting Started",
        "items": [
            {
                "q": "How do I create an account?",
                "a": "Click 'Create Account' on the login page. You will need the enrollment key provided by your instructor, your university email address, and a password of at least 8 characters."
            },
            {
                "q": "I forgot my password. What do I do?",
                "a": "Click 'Forgot password? Reset using enrollment key' on the login page. You will need the enrollment key (same one used to register) and your university email. You can then set a new password."
            },
            {
                "q": "What is the enrollment key?",
                "a": "The enrollment key is a code provided by your instructor at the start of the semester. It is required to create an account or reset your password. If you have lost it, contact your instructor."
            },
            {
                "q": "Can I use my phone to complete assignments?",
                "a": "The app works on mobile but is designed for laptop or desktop use. On a small screen, some layouts may be difficult to read. We strongly recommend using a laptop."
            },
        ]
    },
    {
        "category": "Completing Assignments",
        "items": [
            {
                "q": "Why are my question parameters different from my classmates?",
                "a": "Each student receives a unique set of numbers for each question, generated from your email address. This is intentional — it ensures academic integrity. The method to solve the question is identical for everyone."
            },
            {
                "q": "The Submit button is not appearing. What should I do?",
                "a": "The Submit button only appears once all answer boxes have been filled in. Make sure every input field has a non-zero value entered. If a correct answer is zero, try entering it explicitly."
            },
            {
                "q": "I submitted a wrong answer by mistake. Can I redo it?",
                "a": "No — each question can only be submitted once and is locked after submission. This is by design. If you believe there is a technical error, use the 'Flag this question' button or contact your instructor."
            },
            {
                "q": "I started the assignment, closed the page, and now my answers are gone. What happened?",
                "a": "Your submitted answers are permanently saved. Any answers you typed but did not submit are not saved (there is no auto-save for unsubmitted answers). You will see a warning if you try to navigate away with unsaved inputs."
            },
            {
                "q": "I closed the tab mid-assignment. When I came back, my progress was gone.",
                "a": "Only submitted answers are stored in the database. Typed-but-unsubmitted answers are held in your browser session and lost if you close the tab. Always click Submit before leaving."
            },
            {
                "q": "What does 'Worked example unlocks in X minutes' mean?",
                "a": "To encourage genuine effort, the worked example for each question is hidden for the first 20 minutes you spend on that question. After 20 minutes, a 'Show worked example' button will appear. The timer resets if you close the page."
            },
            {
                "q": "Can I complete the questions in any order?",
                "a": "Yes. You can submit each question independently in any order. The question order on your screen may differ from your classmates — this is also intentional."
            },
        ]
    },
    {
        "category": "Grades and Scores",
        "items": [
            {
                "q": "How is my score calculated?",
                "a": "Each question part is graded automatically and immediately after submission. You will see the correct answer, your answer, and your score straight away. Answers must be exact — there is no tolerance margin."
            },
            {
                "q": "Where can I see my total score for the semester?",
                "a": "Your semester score is displayed at the top of the Dashboard page. It updates automatically as you submit questions."
            },
            {
                "q": "I got the answer wrong but I think my working was correct. What should I do?",
                "a": "Check the step-by-step solution shown after submission. If you believe there is an error in the question or grading, click the 'Flag this question' button on the question page. Your instructor will be notified and will review."
            },
            {
                "q": "Can I see my answers after the deadline?",
                "a": "Yes. Submitted homework is always available in read-only review mode. Go to the Dashboard, click on the homework, and you will see all your answers and the correct solutions. This is useful for exam preparation."
            },
            {
                "q": "My submission receipt shows a wrong score. What should I do?",
                "a": "Screenshot the confirmation banner immediately and contact your instructor. The timestamp on the banner is your proof of submission."
            },
        ]
    },
    {
        "category": "Technical Issues",
        "items": [
            {
                "q": "The page is loading very slowly.",
                "a": "The app connects to Google Sheets to save and retrieve your data. Occasionally this can be slow depending on your internet connection. Wait a few seconds and try refreshing. Avoid clicking buttons multiple times."
            },
            {
                "q": "I see an error message instead of the questions.",
                "a": "Try refreshing the page first. If the error persists, sign out and sign back in. If it still does not work, contact your instructor with a screenshot of the error."
            },
            {
                "q": "The 'Submit' button disappeared after I clicked it. Did it work?",
                "a": "If you see a green confirmation banner with a timestamp, your submission was successful. If you see a yellow warning banner, the save may have failed — screenshot it and contact your instructor."
            },
            {
                "q": "I cannot sign in even though my password is correct.",
                "a": "After 5 failed login attempts, the system locks you out for 5 minutes as a security measure. Wait 5 minutes and try again. If the problem continues, use the password reset option."
            },
        ]
    },
    {
        "category": "Deadlines and Extensions",
        "items": [
            {
                "q": "What happens if I miss the deadline?",
                "a": "After the deadline, the assignment closes and you cannot submit. There is a short grace period (shown on the homework page) after the official deadline. If you have extenuating circumstances, use the 'Request an extension' option which appears on the homework page after the deadline."
            },
            {
                "q": "How do I request an extension?",
                "a": "Go to the homework page after the deadline has passed. At the bottom of the page you will see a 'Request an extension' section. Enter your reason and submit. Your instructor will review the request and respond."
            },
            {
                "q": "I submitted after the deadline. Will my instructor know?",
                "a": "Yes. All submissions are automatically flagged with a late submission indicator that is visible to your instructor."
            },
        ]
    },
]

for section in FAQS:
    st.markdown(f"""
    <div style="font-size:0.72rem;font-weight:600;letter-spacing:0.1em;
     text-transform:uppercase;color:#6B7280;margin:1.6rem 0 0.7rem 0;
     padding-bottom:0.35rem;border-bottom:1px solid #E5E7EB;">
      {section['category']}
    </div>
    """, unsafe_allow_html=True)

    for item in section["items"]:
        with st.expander(item["q"]):
            st.markdown(
                f'<div class="faq-a">{item["a"]}</div>',
                unsafe_allow_html=True
            )

st.markdown("<br>", unsafe_allow_html=True)
st.markdown(f"""
<div style="background:#F9FAFB;border:1px solid #E5E7EB;border-radius:8px;
     padding:1rem 1.3rem;font-size:0.86rem;color:#6B7280;">
  Still have a question? Contact your instructor at
  <a href="mailto:nsunder@bentley.edu" style="color:#2563EB;">
    nsunder@bentley.edu
  </a>
</div>
""", unsafe_allow_html=True)
