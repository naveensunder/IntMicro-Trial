"""
ui.py — Shared design system, CSS, and UI component helpers.
"""

# ── Design tokens ──────────────────────────────────────────────────────────────
COLORS = {
    "navy":        "#1C2B4A",
    "navy_mid":    "#2C3E6B",
    "navy_light":  "#4A6090",
    "accent":      "#2563EB",
    "accent_soft": "#EFF4FF",
    "success":     "#16A34A",
    "success_bg":  "#F0FDF4",
    "success_bd":  "#BBF7D0",
    "warning":     "#D97706",
    "warning_bg":  "#FFFBEB",
    "warning_bd":  "#FDE68A",
    "error":       "#DC2626",
    "error_bg":    "#FEF2F2",
    "error_bd":    "#FECACA",
    "neutral_50":  "#F9FAFB",
    "neutral_100": "#F3F4F6",
    "neutral_200": "#E5E7EB",
    "neutral_300": "#D1D5DB",
    "neutral_500": "#6B7280",
    "neutral_700": "#374151",
    "neutral_900": "#111827",
    "white":       "#FFFFFF",
}

GLOBAL_CSS = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600&family=DM+Serif+Display&display=swap');

/* ── Reset & base ── */
html, body, [class*="css"] {{
    font-family: 'DM Sans', sans-serif;
    color: {COLORS['neutral_900']};
}}
#MainMenu {{visibility: hidden;}}
footer {{visibility: hidden;}}
.stDeployButton {{display: none;}}
header {{visibility: hidden;}}
.block-container {{
    padding-top: 2rem;
    padding-bottom: 3rem;
    max-width: 900px;
}}

/* ── Responsive max-width ── */
@media (min-width: 1400px) {{
    .block-container {{ max-width: 1100px; }}
}}

/* ── Sidebar ── */
[data-testid="stSidebar"] {{
    background: {COLORS['navy']} !important;
    border-right: none !important;
}}
[data-testid="stSidebar"] * {{
    color: #CBD5E1 !important;
}}
[data-testid="stSidebar"] a:hover {{
    color: {COLORS['white']} !important;
    background: rgba(255,255,255,0.08) !important;
    border-radius: 6px;
}}
[data-testid="stSidebarNav"] li {{
    padding: 2px 0;
}}

/* ── Page header ── */
.page-header {{
    padding: 2.2rem 2.5rem;
    background: {COLORS['navy']};
    border-radius: 12px;
    margin-bottom: 2rem;
}}
.page-header-eyebrow {{
    font-size: 0.72rem;
    font-weight: 500;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: #94A3B8;
    margin-bottom: 0.4rem;
}}
.page-header-title {{
    font-family: 'DM Serif Display', serif;
    font-size: 1.9rem;
    color: {COLORS['white']};
    line-height: 1.2;
    margin-bottom: 0.3rem;
}}
.page-header-sub {{
    font-size: 0.88rem;
    color: #94A3B8;
    font-weight: 300;
}}

/* ── Card ── */
.card {{
    background: {COLORS['white']};
    border: 1px solid {COLORS['neutral_200']};
    border-radius: 10px;
    padding: 1.4rem 1.8rem;
    margin-bottom: 1.2rem;
}}
.card-title {{
    font-size: 0.7rem;
    font-weight: 600;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: {COLORS['neutral_500']};
    margin-bottom: 0.8rem;
}}

/* ── Homework card ── */
.hw-card {{
    background: {COLORS['white']};
    border: 1px solid {COLORS['neutral_200']};
    border-radius: 10px;
    padding: 1.2rem 1.6rem;
    margin-bottom: 0.8rem;
    transition: border-color 0.15s ease, box-shadow 0.15s ease;
}}
.hw-card:hover {{
    border-color: {COLORS['accent']};
    box-shadow: 0 2px 12px rgba(37,99,235,0.08);
}}
.hw-card-locked {{
    background: {COLORS['neutral_50']};
    border: 1px solid {COLORS['neutral_200']};
    border-radius: 10px;
    padding: 1.2rem 1.6rem;
    margin-bottom: 0.8rem;
    opacity: 0.6;
}}
.hw-title {{
    font-size: 1rem;
    font-weight: 600;
    color: {COLORS['neutral_900']};
    margin-bottom: 0.2rem;
}}
.hw-meta {{
    font-size: 0.82rem;
    color: {COLORS['neutral_500']};
}}

/* ── Status badges ── */
.badge {{
    display: inline-block;
    padding: 2px 10px;
    border-radius: 20px;
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 0.04em;
}}
.badge-open {{
    background: {COLORS['success_bg']};
    color: {COLORS['success']};
    border: 1px solid {COLORS['success_bd']};
}}
.badge-closed {{
    background: {COLORS['error_bg']};
    color: {COLORS['error']};
    border: 1px solid {COLORS['error_bd']};
}}
.badge-locked {{
    background: {COLORS['neutral_100']};
    color: {COLORS['neutral_500']};
    border: 1px solid {COLORS['neutral_300']};
}}
.badge-complete {{
    background: {COLORS['accent_soft']};
    color: {COLORS['accent']};
    border: 1px solid #BFDBFE;
}}
.badge-graded {{
    background: #FEF3C7;
    color: #92400E;
    border: 1px solid #FDE68A;
}}

/* ── Question block ── */
.q-header {{
    background: {COLORS['navy']};
    border-radius: 10px 10px 0 0;
    padding: 1rem 1.6rem;
    display: flex;
    justify-content: space-between;
    align-items: center;
}}
.q-header-title {{
    font-family: 'DM Serif Display', serif;
    color: {COLORS['white']};
    font-size: 1.1rem;
}}
.q-header-sub {{
    color: #94A3B8;
    font-size: 0.8rem;
    margin-top: 0.15rem;
}}
.q-body {{
    background: {COLORS['neutral_50']};
    border: 1px solid {COLORS['neutral_200']};
    border-top: none;
    border-radius: 0 0 10px 10px;
    padding: 1.2rem 1.6rem;
    margin-bottom: 1.2rem;
    font-size: 0.94rem;
    line-height: 1.75;
}}
.part-row {{
    display: flex;
    align-items: flex-start;
    gap: 10px;
    background: {COLORS['white']};
    border: 1px solid {COLORS['neutral_200']};
    border-radius: 8px;
    padding: 0.7rem 1rem;
    margin: 0.45rem 0;
}}
.part-badge {{
    background: {COLORS['navy']};
    color: {COLORS['white']};
    border-radius: 5px;
    padding: 2px 9px;
    font-size: 0.72rem;
    font-weight: 600;
    white-space: nowrap;
    flex-shrink: 0;
    margin-top: 3px;
}}
.part-text {{
    flex: 1;
    font-size: 0.9rem;
    line-height: 1.6;
    color: {COLORS['neutral_700']};
}}
.part-ungraded {{
    background: {COLORS['neutral_100']};
    color: {COLORS['neutral_500']};
    border-radius: 5px;
    padding: 2px 8px;
    font-size: 0.7rem;
    white-space: nowrap;
    flex-shrink: 0;
}}

/* ── Answer input area ── */
.answer-area {{
    background: {COLORS['white']};
    border: 1px solid {COLORS['neutral_200']};
    border-radius: 10px;
    padding: 1.2rem 1.6rem;
    margin-bottom: 1rem;
}}
.answer-label {{
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: {COLORS['neutral_500']};
    margin-bottom: 0.8rem;
}}

/* ── Timer bar ── */
.timer-bar {{
    display: flex;
    gap: 1rem;
    background: {COLORS['neutral_50']};
    border: 1px solid {COLORS['neutral_200']};
    border-radius: 8px;
    padding: 0.6rem 1rem;
    margin-bottom: 1rem;
    font-size: 0.82rem;
    color: {COLORS['neutral_500']};
}}
.timer-item {{
    display: flex;
    align-items: center;
    gap: 6px;
}}
.timer-value {{
    font-weight: 600;
    color: {COLORS['neutral_900']};
    font-variant-numeric: tabular-nums;
}}

/* ── Solution card ── */
.sol-card {{
    background: {COLORS['white']};
    border: 1px solid {COLORS['neutral_200']};
    border-radius: 10px;
    overflow: hidden;
    margin-top: 1rem;
}}
.sol-header {{
    background: {COLORS['navy']};
    color: {COLORS['white']};
    padding: 0.7rem 1.4rem;
    font-size: 0.9rem;
    font-weight: 600;
    letter-spacing: 0.02em;
}}
.sol-section {{
    padding: 1rem 1.4rem;
    border-bottom: 1px solid {COLORS['neutral_100']};
}}
.sol-section:last-child {{
    border-bottom: none;
}}
.sol-section-title {{
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: {COLORS['neutral_500']};
    margin-bottom: 0.6rem;
}}
.sol-steps-box {{
    background: #EFF6FF;
    border-left: 3px solid {COLORS['accent']};
    border-radius: 0 6px 6px 0;
    padding: 0.9rem 1.1rem;
    font-size: 0.9rem;
    line-height: 1.8;
}}
.sol-mistakes-box {{
    background: #FFF7ED;
    border-left: 3px solid #F97316;
    border-radius: 0 6px 6px 0;
    padding: 0.9rem 1.1rem;
    font-size: 0.9rem;
    line-height: 1.8;
}}

/* ── Score table ── */
.score-table {{
    width: 100%;
    border-collapse: collapse;
    font-size: 0.88rem;
    margin-top: 0.8rem;
}}
.score-table th {{
    background: {COLORS['neutral_100']};
    padding: 7px 10px;
    text-align: left;
    font-weight: 600;
    color: {COLORS['neutral_700']};
    border-bottom: 1px solid {COLORS['neutral_200']};
}}
.score-table td {{
    padding: 7px 10px;
    border-bottom: 1px solid {COLORS['neutral_100']};
    color: {COLORS['neutral_700']};
}}
.score-table tr:last-child td {{
    background: {COLORS['neutral_50']};
    font-weight: 600;
    border-bottom: none;
}}
.chip-ok {{
    background: {COLORS['success_bg']};
    color: {COLORS['success']};
    border-radius: 4px;
    padding: 2px 8px;
    font-size: 0.78rem;
    font-weight: 600;
}}
.chip-wrong {{
    background: {COLORS['error_bg']};
    color: {COLORS['error']};
    border-radius: 4px;
    padding: 2px 8px;
    font-size: 0.78rem;
    font-weight: 600;
}}
.chip-partial {{
    background: {COLORS['warning_bg']};
    color: {COLORS['warning']};
    border-radius: 4px;
    padding: 2px 8px;
    font-size: 0.78rem;
    font-weight: 600;
}}

/* ── Banners ── */
.banner-success {{
    background: {COLORS['success_bg']};
    border: 1px solid {COLORS['success_bd']};
    border-radius: 8px;
    padding: 0.9rem 1.2rem;
    font-size: 0.9rem;
    color: #166534;
    margin: 0.8rem 0;
}}
.banner-warning {{
    background: {COLORS['warning_bg']};
    border: 1px solid {COLORS['warning_bd']};
    border-radius: 8px;
    padding: 0.9rem 1.2rem;
    font-size: 0.9rem;
    color: #92400E;
    margin: 0.8rem 0;
}}
.banner-error {{
    background: {COLORS['error_bg']};
    border: 1px solid {COLORS['error_bd']};
    border-radius: 8px;
    padding: 0.9rem 1.2rem;
    font-size: 0.9rem;
    color: #991B1B;
    margin: 0.8rem 0;
}}
.banner-info {{
    background: {COLORS['accent_soft']};
    border: 1px solid #BFDBFE;
    border-radius: 8px;
    padding: 0.9rem 1.2rem;
    font-size: 0.9rem;
    color: #1E40AF;
    margin: 0.8rem 0;
}}
.banner-restore {{
    background: {COLORS['neutral_50']};
    border: 1px solid {COLORS['neutral_300']};
    border-radius: 8px;
    padding: 0.8rem 1.2rem;
    font-size: 0.86rem;
    color: {COLORS['neutral_700']};
    margin: 0.8rem 0;
}}
.banner-locked {{
    background: #F5F3FF;
    border: 1px solid #DDD6FE;
    border-radius: 8px;
    padding: 0.8rem 1.2rem;
    font-size: 0.86rem;
    color: #5B21B6;
    margin: 0.8rem 0;
}}

/* ── Param chips ── */
.param-chip {{
    display: inline-block;
    background: {COLORS['neutral_100']};
    border: 1px solid {COLORS['neutral_200']};
    border-radius: 5px;
    padding: 3px 10px;
    font-size: 0.82rem;
    color: {COLORS['neutral_700']};
    margin: 2px 3px 2px 0;
    font-variant-numeric: tabular-nums;
}}

/* ── Progress ── */
.progress-wrap {{
    background: {COLORS['neutral_50']};
    border: 1px solid {COLORS['neutral_200']};
    border-radius: 8px;
    padding: 0.9rem 1.2rem;
    margin-bottom: 1.2rem;
}}
.progress-label {{
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: {COLORS['neutral_500']};
    margin-bottom: 0.5rem;
}}
.progress-bar-outer {{
    background: {COLORS['neutral_200']};
    border-radius: 20px;
    height: 6px;
    overflow: hidden;
}}
.progress-bar-inner {{
    height: 100%;
    border-radius: 20px;
    background: {COLORS['accent']};
}}
.progress-text {{
    font-size: 0.8rem;
    color: {COLORS['neutral_500']};
    margin-top: 0.35rem;
}}

/* ── Inputs ── */
.stTextInput > div > div > input,
.stNumberInput > div > div > input,
.stSelectbox > div > div,
.stTextArea > div > div > textarea {{
    border: 1px solid {COLORS['neutral_300']} !important;
    border-radius: 7px !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 0.92rem !important;
    color: {COLORS['neutral_900']} !important;
    background: {COLORS['white']} !important;
}}
.stTextInput > div > div > input:focus,
.stNumberInput > div > div > input:focus,
.stTextArea > div > div > textarea:focus {{
    border-color: {COLORS['accent']} !important;
    box-shadow: 0 0 0 3px rgba(37,99,235,0.08) !important;
    outline: none !important;
}}

/* ── Buttons ── */
.stButton > button {{
    background: {COLORS['navy']} !important;
    color: {COLORS['white']} !important;
    border: none !important;
    border-radius: 7px !important;
    font-family: 'DM Sans', sans-serif !important;
    font-weight: 500 !important;
    font-size: 0.9rem !important;
    padding: 0.5rem 1.4rem !important;
    transition: background 0.15s ease !important;
    letter-spacing: 0.01em !important;
}}
.stButton > button:hover {{
    background: {COLORS['navy_mid']} !important;
}}

/* ── Divider ── */
.section-divider {{
    border: none;
    border-top: 1px solid {COLORS['neutral_200']};
    margin: 1.5rem 0;
}}

/* ── Instructor dashboard ── */
.dash-metric {{
    background: {COLORS['white']};
    border: 1px solid {COLORS['neutral_200']};
    border-radius: 10px;
    padding: 1.1rem 1.4rem;
    text-align: center;
}}
.dash-metric-num {{
    font-family: 'DM Serif Display', serif;
    font-size: 2rem;
    color: {COLORS['navy']};
    line-height: 1;
}}
.dash-metric-label {{
    font-size: 0.78rem;
    color: {COLORS['neutral_500']};
    margin-top: 0.3rem;
    font-weight: 500;
}}

/* ── Mobile warning ── */
.mobile-warning {{
    display: none;
    background: {COLORS['warning_bg']};
    border: 1px solid {COLORS['warning_bd']};
    border-radius: 8px;
    padding: 0.9rem 1.2rem;
    font-size: 0.88rem;
    color: #92400E;
    margin-bottom: 1rem;
}}
@media (max-width: 768px) {{
    .mobile-warning {{ display: block; }}
}}

/* ── Flag button ── */
.flag-section {{
    border-top: 1px solid {COLORS['neutral_100']};
    padding-top: 0.8rem;
    margin-top: 0.8rem;
}}
</style>
"""


def inject_css():
    import streamlit as st
    st.markdown(GLOBAL_CSS, unsafe_allow_html=True)


def page_header(eyebrow: str, title: str, subtitle: str = ""):
    import streamlit as st
    sub_html = f'<div class="page-header-sub">{subtitle}</div>' if subtitle else ""
    st.markdown(f"""
    <div class="page-header">
      <div class="page-header-eyebrow">{eyebrow}</div>
      <div class="page-header-title">{title}</div>
      {sub_html}
    </div>
    """, unsafe_allow_html=True)


def banner(text: str, kind: str = "info"):
    import streamlit as st
    st.markdown(f'<div class="banner-{kind}">{text}</div>', unsafe_allow_html=True)


def section_divider():
    import streamlit as st
    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)
