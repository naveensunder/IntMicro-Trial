"""
ui.py — Design system.
HWDashboard v5
"""

COLORS = {
    "navy":       "#1C2B4A",
    "navy_mid":   "#2C3E6B",
    "accent":     "#2563EB",
    "green":      "#2D6A4F",   # sophisticated green for open assignments
    "green_bg":   "#F0F7F4",
    "green_bd":   "#95D5B2",
    "success":    "#16A34A",
    "warning":    "#D97706",
    "error":      "#DC2626",
    "grey_light": "#F5F5F5",
    "grey_mid":   "#E0E0E0",
    "grey_text":  "#555555",
    "black":      "#1A1A1A",
    "white":      "#FFFFFF",
    "sol_bg":     "#F0F4F8",
    "ans_bg":     "#FAF9F6",   # warm grey for answer section
    "teal":       "#2A7F7F",   # answer section left border
}

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600&family=DM+Serif+Display&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    font-size: 17px;
    color: #1A1A1A;
}
#MainMenu, footer, header, .stDeployButton { visibility: hidden; display: none; }
.block-container { padding-top: 2rem; padding-bottom: 4rem; max-width: 820px; }

[data-testid="stSidebar"] { background: #1C2B4A !important; }
[data-testid="stSidebar"] * { color: #CBD5E1 !important; font-size: 15px !important; }

/* ── Page header ── */
.page-header {
    padding: 2rem 2.2rem; background: #1C2B4A;
    border-radius: 10px; margin-bottom: 2rem;
}
.page-header-eye {
    font-size: 0.75rem; font-weight: 600; letter-spacing: 0.14em;
    text-transform: uppercase; color: #94A3B8; margin-bottom: 0.4rem;
}
.page-header-title {
    font-family: 'DM Serif Display', serif;
    font-size: 2rem; color: #FFFFFF; line-height: 1.2; margin-bottom: 0.25rem;
}
.page-header-sub { font-size: 1rem; color: #94A3B8; }

/* ── Homework cards ── */
.hw-card {
    background: #FFFFFF; border: 1.5px solid #E0E0E0;
    border-radius: 10px; padding: 1.3rem 1.7rem; margin-bottom: 0.9rem;
}
.hw-card-open {
    background: #F0F7F4; border: 2px solid #95D5B2;
    border-left: 5px solid #2D6A4F;
    border-radius: 10px; padding: 1.3rem 1.7rem; margin-bottom: 0.9rem;
}
.hw-card-locked {
    background: #F5F5F5; border: 1.5px solid #E0E0E0;
    border-radius: 10px; padding: 1.3rem 1.7rem; margin-bottom: 0.9rem; opacity: 0.6;
}
.hw-title        { font-size: 1.1rem; font-weight: 600; color: #1A1A1A; margin-bottom: 0.2rem; }
.hw-title-open   { font-size: 1.1rem; font-weight: 600; color: #2D6A4F; margin-bottom: 0.2rem; }
.hw-meta         { font-size: 0.9rem; color: #555555; margin-top: 0.15rem; }
.hw-score        { font-size: 0.92rem; color: #2563EB; font-weight: 600; margin-top: 0.2rem; }
.hw-deadline-urgent { font-size: 0.92rem; color: #DC2626; font-weight: 600; margin-top: 0.15rem; }

/* ── Badges ── */
.badge { display: inline-block; padding: 3px 11px; border-radius: 20px; font-size: 0.78rem; font-weight: 600; }
.badge-open     { background: #F0F7F4; color: #2D6A4F; border: 1.5px solid #95D5B2; }
.badge-closed   { background: #FEF2F2; color: #DC2626; border: 1.5px solid #FCA5A5; }
.badge-locked   { background: #F5F5F5; color: #555555; border: 1.5px solid #E0E0E0; }
.badge-complete { background: #EFF4FF; color: #2563EB; border: 1.5px solid #93C5FD; }

/* ── Question banner (collapsible header) ── */
.q-banner-open {
    background: #1C2B4A; border-radius: 8px;
    padding: 1rem 1.5rem; margin-bottom: 1rem;
    display: flex; justify-content: space-between; align-items: center;
}
.q-banner-submitted {
    background: #2D6A4F; border-radius: 8px;
    padding: 1rem 1.5rem; margin-bottom: 1rem;
    display: flex; justify-content: space-between; align-items: center;
}
.q-banner-title {
    font-family: 'DM Serif Display', serif;
    font-size: 1.4rem; font-weight: 700; color: #FFFFFF;
}
.q-banner-score { font-size: 1rem; color: #A7F3D0; font-weight: 600; }

/* ── Question body ── */
.q-body {
    background: #FFFFFF;
    border-left: 5px solid #1C2B4A;
    border-radius: 0 8px 8px 0;
    padding: 1.5rem 1.8rem;
    margin-bottom: 0;
    font-size: 1rem; line-height: 1.85;
}
.q-pts   { font-size: 0.88rem; color: #555555; margin-bottom: 1rem; }
.q-text  { font-size: 1rem; line-height: 1.85; color: #1A1A1A; margin-bottom: 1rem; }
.q-part  { font-size: 1rem; line-height: 1.85; color: #1A1A1A; padding: 0.15rem 0 0.15rem 1.3rem; }
.q-part-label { font-weight: 600; color: #1C2B4A; }
.q-ungraded   { font-size: 0.83rem; color: #888888; font-style: italic; }

/* ── Divider between question sections ── */
.section-divider {
    border: none; border-top: 1px solid #E0E0E0;
    margin: 1.2rem 0;
}

/* ── Answer section ── */
.ans-section {
    background: #FAF9F6;
    border-left: 5px solid #2A7F7F;
    border-radius: 0 8px 8px 0;
    padding: 1.3rem 1.8rem;
    margin-bottom: 0;
}
.ans-label {
    font-size: 0.8rem; font-weight: 700; letter-spacing: 0.12em;
    text-transform: uppercase; color: #2A7F7F; margin-bottom: 0.8rem;
}

/* ── Solution section ── */
.sol-section {
    background: #F0F4F8;
    border-left: 5px solid #16A34A;
    border-radius: 0 8px 8px 0;
    padding: 1.3rem 1.8rem;
}
.sol-title {
    font-family: 'DM Serif Display', serif;
    font-size: 1.2rem; color: #1C2B4A; margin-bottom: 1rem;
}
.sol-label {
    font-size: 0.8rem; font-weight: 700; letter-spacing: 0.12em;
    text-transform: uppercase; color: #555555; margin-bottom: 0.5rem; margin-top: 1rem;
}
.sol-steps {
    background: #FFFFFF; border-left: 4px solid #2563EB;
    border-radius: 0 6px 6px 0; padding: 1rem 1.2rem;
    font-size: 1rem; line-height: 1.9; color: #1A1A1A;
}
.sol-mistakes {
    background: #FFFFFF; border-left: 4px solid #F97316;
    border-radius: 0 6px 6px 0; padding: 1rem 1.2rem;
    font-size: 1rem; line-height: 1.9; color: #1A1A1A;
}
.sol-revise {
    background: #FFFFFF; border-left: 4px solid #16A34A;
    border-radius: 0 6px 6px 0; padding: 1rem 1.2rem;
    font-size: 1rem; line-height: 1.9; color: #1A1A1A;
}

/* ── Submitted answers box ── */
.submitted-ans {
    background: #F5F5F5;
    border-left: 5px solid #555555;
    border-radius: 0 8px 8px 0;
    padding: 1.2rem 1.8rem;
    font-size: 1rem; line-height: 1.85;
}
.submitted-ans-label {
    font-family: 'DM Serif Display', serif;
    font-size: 1.1rem; color: #555555; margin-bottom: 0.7rem;
}

/* ── Score rows ── */
.score-row {
    display: flex; align-items: center; justify-content: space-between;
    padding: 0.7rem 0; border-bottom: 1px solid #E0E0E0;
    font-size: 1rem; color: #1A1A1A;
}
.score-row:last-child { border-bottom: none; }
.score-row-label { flex: 2; color: #555555; }
.score-row-val   { flex: 1; text-align: right; font-weight: 500; }
.score-row-total { font-weight: 700; color: #1C2B4A; font-size: 1.05rem; }
.chip-ok    { background: #ECFDF5; color: #16A34A; border-radius: 4px; padding: 3px 9px; font-size: 0.88rem; font-weight: 600; }
.chip-wrong { background: #FEF2F2; color: #DC2626; border-radius: 4px; padding: 3px 9px; font-size: 0.88rem; font-weight: 600; }

/* ── TF solution ── */
.tf-item { padding: 1rem 0; border-bottom: 1px solid #E0E0E0; font-size: 1rem; line-height: 1.8; }
.tf-item:last-child { border-bottom: none; }
.tf-stmt   { font-weight: 500; color: #1A1A1A; margin-bottom: 0.4rem; }
.tf-result { font-size: 0.95rem; margin-bottom: 0.4rem; }
.tf-expl   { color: #555555; font-size: 0.95rem; line-height: 1.75; }

/* ── Parameters ── */
.param-row {
    background: #F5F5F5; border-radius: 7px;
    padding: 0.65rem 1rem; font-size: 0.95rem;
    color: #555555; margin-bottom: 1rem;
}
.param-val { font-weight: 600; color: #1A1A1A; }

/* ── Between-question divider ── */
.q-divider {
    display: flex; align-items: center; gap: 1rem;
    margin: 2.5rem 0 2rem 0;
}
.q-divider-line { flex: 1; height: 3px; background: #1C2B4A; border-radius: 2px; }
.q-divider-label {
    font-family: 'DM Serif Display', serif; font-size: 1rem;
    white-space: nowrap; padding: 0.3rem 1rem;
    background: #1C2B4A; color: #FFFFFF; border-radius: 20px;
}

/* ── Banners ── */
.banner { border-radius: 8px; padding: 0.95rem 1.2rem; font-size: 1rem; margin: 0.8rem 0; line-height: 1.6; }
.banner-success { background: #ECFDF5; border: 1.5px solid #86EFAC; color: #166534; }
.banner-warning { background: #FFFBEB; border: 1.5px solid #FDE68A; color: #92400E; }
.banner-error   { background: #FEF2F2; border: 1.5px solid #FCA5A5; color: #991B1B; }
.banner-info    { background: #EFF4FF; border: 1.5px solid #93C5FD; color: #1E40AF; }
.banner-locked  { background: #F5F3FF; border: 1.5px solid #C4B5FD; color: #5B21B6; }
.banner-restore { background: #F5F5F5; border: 1.5px solid #E0E0E0; color: #555555; }

/* ── Saved timestamp ── */
.saved-ts { font-size: 0.88rem; color: #888888; margin-top: 0.3rem; font-style: italic; }

/* ── Semester summary ── */
.sem-box { background: #1C2B4A; border-radius: 10px; padding: 1.3rem 1.8rem; margin-bottom: 1.8rem; }
.sem-label { font-size: 0.78rem; font-weight: 600; letter-spacing: 0.1em; text-transform: uppercase; color: #94A3B8; margin-bottom: 0.25rem; }
.sem-score { font-family: 'DM Serif Display', serif; font-size: 2rem; color: #FFFFFF; line-height: 1; }
.sem-sub   { font-size: 0.92rem; color: #64748B; margin-top: 0.2rem; }

/* ── Question overview rows ── */
.ov-row {
    display: flex; align-items: center; justify-content: space-between;
    padding: 0.8rem 1rem; background: #FFFFFF;
    border: 1.5px solid #E0E0E0; border-radius: 8px; margin-bottom: 0.4rem; font-size: 1rem;
}
.ov-title  { flex: 3; font-weight: 500; color: #1A1A1A; }
.ov-score  { flex: 1; text-align: center; color: #555555; }
.ov-status { flex: 1; text-align: right; }
.ov-done   { color: #16A34A; font-weight: 600; }
.ov-todo   { color: #888888; }
.ov-total  {
    display: flex; align-items: center; justify-content: space-between;
    padding: 0.8rem 1rem; background: #1C2B4A;
    border-radius: 8px; margin-top: 0.3rem; font-size: 1rem; color: #FFFFFF; font-weight: 600;
}

/* ── Progress line ── */
.progress-line {
    font-size: 0.95rem; color: #555555;
    background: #F5F5F5; border-radius: 7px;
    padding: 0.6rem 1rem; margin-bottom: 1.2rem;
}
.progress-line strong { color: #1C2B4A; }

/* ── Inputs ── */
.stTextInput > div > div > input,
.stNumberInput > div > div > input,
.stTextArea > div > div > textarea {
    border: 1.5px solid #E0E0E0 !important; border-radius: 7px !important;
    font-family: 'DM Sans', sans-serif !important; font-size: 1rem !important;
    padding: 0.55rem 0.8rem !important; color: #1A1A1A !important;
}
.stTextInput > div > div > input:focus,
.stNumberInput > div > div > input:focus {
    border-color: #2A7F7F !important;
    box-shadow: 0 0 0 3px rgba(42,127,127,0.1) !important;
}
.stRadio > div { gap: 1rem !important; }
.stRadio label { font-size: 1rem !important; }

/* ── Buttons ── */
.stButton > button {
    background: #1C2B4A !important; color: #FFFFFF !important;
    border: none !important; border-radius: 7px !important;
    font-family: 'DM Sans', sans-serif !important;
    font-weight: 500 !important; font-size: 1rem !important;
    padding: 0.55rem 1.5rem !important;
}
.stButton > button:hover { background: #2C3E6B !important; }

/* ── Expander — bolder show/hide label ── */
.streamlit-expanderHeader {
    font-size: 1.05rem !important;
    font-weight: 700 !important;
    color: #1C2B4A !important;
}

/* ── Mobile ── */
.mobile-warn {
    display: none; background: #FFFBEB; border: 1.5px solid #FDE68A;
    border-radius: 8px; padding: 0.9rem 1.1rem; font-size: 1rem;
    color: #92400E; margin-bottom: 1rem;
}
@media (max-width: 768px) { .mobile-warn { display: block; } }

/* ── Footer ── */
.page-footer {
    margin-top: 3rem; padding-top: 1rem;
    border-top: 1px solid #E0E0E0;
    font-size: 0.88rem; color: #888888; text-align: center;
}
.page-footer a { color: #2563EB; text-decoration: none; }

/* ── FAQ / Materials ── */
.faq-a { font-size: 1rem; color: #1A1A1A; line-height: 1.75; }
.mat-row {
    display: flex; align-items: center; justify-content: space-between;
    padding: 0.9rem 1.1rem; background: #FFFFFF;
    border: 1.5px solid #E0E0E0; border-radius: 8px; margin-bottom: 0.45rem;
}
.mat-title { font-size: 1rem; font-weight: 500; color: #1A1A1A; }
.mat-meta  { font-size: 0.88rem; color: #888888; margin-top: 0.1rem; }

/* ── Instructor ── */
.dash-metric { background:#FFFFFF; border:1.5px solid #E0E0E0; border-radius:9px; padding:1.1rem 1.3rem; text-align:center; }
.dash-metric-num { font-family:'DM Serif Display',serif; font-size:2rem; color:#1C2B4A; line-height:1; }
.dash-metric-lbl { font-size:0.8rem; color:#555555; margin-top:0.3rem; font-weight:600; text-transform:uppercase; letter-spacing:0.07em; }
.sec { font-family:'DM Serif Display',serif; font-size:1.1rem; color:#1C2B4A; margin:1.6rem 0 0.7rem 0; padding-bottom:0.35rem; border-bottom:2px solid #E0E0E0; }
</style>
"""


def inject_css():
    import streamlit as st
    st.markdown(CSS, unsafe_allow_html=True)


def page_header(eyebrow: str, title: str, sub: str = ""):
    import streamlit as st
    sub_html = f'<div class="page-header-sub">{sub}</div>' if sub else ""
    st.markdown(
        f'<div class="page-header">'
        f'<div class="page-header-eye">{eyebrow}</div>'
        f'<div class="page-header-title">{title}</div>'
        f'{sub_html}</div>',
        unsafe_allow_html=True
    )


def banner(text: str, kind: str = "info"):
    import streamlit as st
    st.markdown(f'<div class="banner banner-{kind}">{text}</div>',
                unsafe_allow_html=True)


def page_footer():
    import streamlit as st
    st.markdown(
        '<div class="page-footer">Questions? See the '
        '<a href="/FAQ">FAQ</a> or email '
        '<a href="mailto:nsunder@bentley.edu">nsunder@bentley.edu</a>'
        '</div>',
        unsafe_allow_html=True
    )


def q_divider(label: str):
    import streamlit as st
    st.markdown(
        f'<div class="q-divider">'
        f'<div class="q-divider-line"></div>'
        f'<div class="q-divider-label">{label}</div>'
        f'<div class="q-divider-line"></div>'
        f'</div>',
        unsafe_allow_html=True
    )
