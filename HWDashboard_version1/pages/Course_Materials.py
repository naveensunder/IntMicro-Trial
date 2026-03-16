"""
pages/Course_Materials.py — Course materials page with Dropbox/Drive links.
HWDashboard v2 — Phase 1
"""
import streamlit as st
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from ui import inject_css, page_header

st.set_page_config(
    page_title="Course Materials — Microeconomics",
    page_icon="📁",
    layout="centered",
    initial_sidebar_state="expanded",
)

inject_css()

if not st.session_state.get("authenticated"):
    st.warning("Please sign in first.")
    if st.button("Go to sign in"):
        st.switch_page("Home.py")
    st.stop()

page_header(
    "Intermediate Microeconomics",
    "Course Materials",
    "Lecture slides, readings, and resources"
)

# ══════════════════════════════════════════════════════════════════════════════
#  HOW TO ADD MATERIALS — READ THIS SECTION
#
#  Each material is a dict with these fields:
#    title       — display name shown to students
#    description — short description (optional, can be "")
#    url         — the shareable link from Dropbox or Google Drive
#    updated     — date last updated, shown to students e.g. "14 Mar 2026"
#    file_type   — shown as a small label e.g. "PDF", "PPTX", "Link"
#
#  HOW TO GET A DROPBOX LINK:
#    1. Upload the file to Dropbox
#    2. Right-click the file → Share → Copy link
#    3. Change "dl=0" at the end of the URL to "dl=1" to force download
#       (or leave as "dl=0" to open in browser)
#    4. Paste the URL in the "url" field below
#
#  HOW TO GET A GOOGLE DRIVE LINK:
#    1. Upload the file to Google Drive
#    2. Right-click → Share → Change to "Anyone with the link" → Copy link
#    3. Paste the URL in the "url" field below
#
#  HOW TO UPDATE A FILE:
#    Option A (recommended) — Replace the file in Dropbox/Drive with the same
#    name. The link stays the same. Just update the "updated" date below.
#
#    Option B — Upload a new file, get a new link, replace the "url" below.
#
#  TO ADD A NEW MATERIAL:
#    Copy one of the dict blocks below and add it to the appropriate section.
#    Then redeploy (push to GitHub — Streamlit will auto-update).
#
#  TO REMOVE A MATERIAL:
#    Delete its dict block below and redeploy.
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
        {
            "title":       "Week 3 — Consumer Preferences",
            "description": "Utility functions, indifference curves, and MRS.",
            "url":         "https://www.dropbox.com/your-link-here",
            "updated":     "24 Jan 2026",
            "file_type":   "PPTX",
        },
    ],
    "Readings & Textbook Chapters": [
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
    "Problem Sets & Solutions": [
        {
            "title":       "Week 2 — Practice Problems",
            "description": "Additional practice questions on budget constraints.",
            "url":         "https://www.dropbox.com/your-link-here",
            "updated":     "17 Jan 2026",
            "file_type":   "PDF",
        },
        {
            "title":       "Week 2 — Solutions",
            "description": "Full worked solutions. Released after the homework deadline.",
            "url":         "https://www.dropbox.com/your-link-here",
            "updated":     "25 Jan 2026",
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

# ── Render ─────────────────────────────────────────────────────────────────────
FILE_TYPE_COLORS = {
    "PDF":  "#DC2626",
    "PPTX": "#D97706",
    "XLSX": "#16A34A",
    "Link": "#2563EB",
    "DOCX": "#1C2B4A",
}

for section_title, items in MATERIALS.items():
    st.markdown(f"""
    <div style="font-size:0.72rem;font-weight:600;letter-spacing:0.1em;
     text-transform:uppercase;color:#6B7280;
     margin:1.5rem 0 0.6rem 0;padding-bottom:0.3rem;
     border-bottom:1px solid #E5E7EB;">
      {section_title}
    </div>
    """, unsafe_allow_html=True)

    for item in items:
        ft         = item.get("file_type", "")
        ft_color   = FILE_TYPE_COLORS.get(ft, "#6B7280")
        updated    = item.get("updated", "")
        updated_html = (
            f'<span style="font-size:0.72rem;color:#9CA3AF;">Updated: {updated}</span>'
            if updated else ""
        )
        desc = item.get("description", "")
        desc_html = (
            f'<div class="material-meta">{desc}</div>' if desc else ""
        )

        st.markdown(f"""
        <div class="material-row">
          <div>
            <div class="material-title">{item['title']}</div>
            {desc_html}
            {updated_html}
          </div>
          <div style="display:flex;align-items:center;gap:0.8rem;flex-shrink:0;margin-left:1rem;">
            <span style="background:{ft_color}15;color:{ft_color};
                  border:1px solid {ft_color}40;border-radius:4px;
                  padding:2px 8px;font-size:0.7rem;font-weight:600;">
              {ft}
            </span>
            <a href="{item['url']}" target="_blank"
               style="background:#1C2B4A;color:white;border-radius:5px;
                      padding:4px 12px;font-size:0.78rem;font-weight:500;
                      text-decoration:none;font-family:'DM Sans',sans-serif;">
              Open ↗
            </a>
          </div>
        </div>
        """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)
st.markdown("""
<div style="background:#F9FAFB;border:1px solid #E5E7EB;border-radius:8px;
     padding:0.9rem 1.2rem;font-size:0.82rem;color:#6B7280;">
  🔗 All links open in a new tab. If a link appears broken, please let your instructor know.
</div>
""", unsafe_allow_html=True)
