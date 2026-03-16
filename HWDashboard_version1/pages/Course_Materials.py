"""
pages/Course_Materials.py
HWDashboard v3

HOW TO ADD/UPDATE MATERIALS:
- Edit the MATERIALS dict below
- Add a dict with: title, description, url, updated, file_type
- For Dropbox: right-click file → Share → Copy link (change dl=0 to dl=1 to force download)
- For Google Drive: Share → Anyone with link → Copy link
- To update a file: replace it in Dropbox/Drive with the same name (link stays the same),
  then update the "updated" date here and push to GitHub
- Push to GitHub — Streamlit auto-updates within ~1 minute
"""
import streamlit as st
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from ui import inject_css, page_header

st.set_page_config(page_title="Course Materials — Microeconomics", page_icon="📁",
                   layout="centered", initial_sidebar_state="expanded")
inject_css()

if not st.session_state.get("authenticated"):
    st.warning("Please sign in first.")
    if st.button("Go to sign in"): st.switch_page("Home.py")
    st.stop()

page_header("Intermediate Microeconomics", "Course Materials",
            "Lecture slides, readings, and resources")

# ══════════════════════════════════════════════════════════════════════════════
#  ADD YOUR MATERIALS HERE
# ══════════════════════════════════════════════════════════════════════════════
MATERIALS = {
    "Lecture Slides": [
        {
            "title":       "Week 1 — Introduction to Microeconomics",
            "description": "Course overview, key concepts, and methodology.",
            "url":         "https://www.dropbox.com/your-link-here",
            "updated":     "10 Jan 2026",
            "file_type":   "PPTX",
        },
        {
            "title":       "Week 2 — Budget Constraints",
            "description": "Budget sets, intercepts, slopes, and comparative statics.",
            "url":         "https://www.dropbox.com/your-link-here",
            "updated":     "17 Jan 2026",
            "file_type":   "PPTX",
        },
    ],
    "Readings": [
        {
            "title":       "Varian — Chapter 2: Budget Constraint",
            "description": "Core textbook reading for Week 2.",
            "url":         "https://www.dropbox.com/your-link-here",
            "updated":     "10 Jan 2026",
            "file_type":   "PDF",
        },
        {
            "title":       "Varian — Chapter 3: Preferences",
            "description": "Core textbook reading for Week 3.",
            "url":         "https://www.dropbox.com/your-link-here",
            "updated":     "10 Jan 2026",
            "file_type":   "PDF",
        },
    ],
    "Other Resources": [
        {
            "title":       "Course Syllabus",
            "description": "Assessment schedule, grading policy, and reading list.",
            "url":         "https://www.dropbox.com/your-link-here",
            "updated":     "10 Jan 2026",
            "file_type":   "PDF",
        },
        {
            "title":       "Desmos Graphing Calculator",
            "description": "Useful for plotting budget lines and indifference curves.",
            "url":         "https://www.desmos.com/calculator",
            "updated":     "",
            "file_type":   "Link",
        },
    ],
}
# ══════════════════════════════════════════════════════════════════════════════

TYPE_COLORS = {
    "PDF": "#DC2626", "PPTX": "#D97706",
    "XLSX": "#16A34A", "Link": "#2563EB", "DOCX": "#1C2B4A",
}

for section, items in MATERIALS.items():
    st.markdown(
        f'<div style="font-size:0.68rem;font-weight:600;letter-spacing:0.1em;'
        f'text-transform:uppercase;color:#6B7280;'
        f'margin:1.4rem 0 0.55rem 0;padding-bottom:0.28rem;'
        f'border-bottom:1px solid #E5E7EB;">{section}</div>',
        unsafe_allow_html=True
    )
    for item in items:
        ft      = item.get("file_type","")
        fc      = TYPE_COLORS.get(ft, "#6B7280")
        updated = item.get("updated","")
        upd_html = (f'<span style="font-size:0.71rem;color:#9CA3AF;">Updated: {updated}</span>'
                    if updated else "")
        desc    = item.get("description","")
        desc_html = f'<div class="mat-meta">{desc}</div>' if desc else ""

        st.markdown(
            f'<div class="mat-row">'
            f'<div>'
            f'<div class="mat-title">{item["title"]}</div>'
            f'{desc_html}'
            f'{upd_html}'
            f'</div>'
            f'<div style="display:flex;align-items:center;gap:0.7rem;'
            f'flex-shrink:0;margin-left:1rem;">'
            f'<span style="background:{fc}15;color:{fc};border:1px solid {fc}40;'
            f'border-radius:4px;padding:2px 7px;font-size:0.69rem;font-weight:600;">'
            f'{ft}</span>'
            f'<a href="{item["url"]}" target="_blank" '
            f'style="background:#1C2B4A;color:white;border-radius:5px;'
            f'padding:4px 11px;font-size:0.77rem;font-weight:500;'
            f'text-decoration:none;font-family:\'DM Sans\',sans-serif;">Open ↗</a>'
            f'</div>'
            f'</div>',
            unsafe_allow_html=True
        )

st.markdown("<br>", unsafe_allow_html=True)
st.markdown(
    '<div style="background:#F9FAFB;border:1px solid #E5E7EB;border-radius:7px;'
    'padding:0.85rem 1.1rem;font-size:0.81rem;color:#6B7280;">'
    '🔗 All links open in a new tab. If a link is broken, '
    'email <a href="mailto:nsunder@bentley.edu" style="color:#2563EB;">'
    'nsunder@bentley.edu</a></div>',
    unsafe_allow_html=True
)
