"""
ui.py — Shared design system, CSS, and UI helpers.
HWDashboard v2 — Phase 1
"""

COLORS = {
    "navy":         "#1C2B4A",
    "navy_mid":     "#2C3E6B",
    "navy_light":   "#4A6090",
    "accent":       "#2563EB",
    "accent_soft":  "#EFF4FF",
    "success":      "#16A34A",
    "success_bg":   "#F0FDF4",
    "success_bd":   "#BBF7D0",
    "warning":      "#D97706",
    "warning_bg":   "#FFFBEB",
    "warning_bd":   "#FDE68A",
    "error":        "#DC2626",
    "error_bg":     "#FEF2F2",
    "error_bd":     "#FECACA",
    "neutral_50":   "#F9FAFB",
    "neutral_100":  "#F3F4F6",
    "neutral_200":  "#E5E7EB",
    "neutral_300":  "#D1D5DB",
    "neutral_400":  "#9CA3AF",
    "neutral_500":  "#6B7280",
    "neutral_700":  "#374151",
    "neutral_900":  "#111827",
    "white":        "#FFFFFF",
}

GLOBAL_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:ital,opsz,wght@0,9..40,300;0,9..40,400;0,9..40,500;0,9..40,600;1,9..40,300;1,9..40,400&family=DM+Serif+Display&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    color: #111827;
}
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
.stDeployButton {display: none;}
header {visibility: hidden;}
.block-container {
    padding-top: 1.8rem;
    padding-bottom: 3rem;
    max-width: 860px;
}
@media (min-width: 1400px) {
    .block-container { max-width: 1000px; }
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: #1C2B4A !important;
    border-right: none !important;
}
[data-testid="stSidebar"] * { color: #CBD5E1 !important; }
[data-testid="stSidebar"] a:hover {
    color: #FFFFFF !important;
    background: rgba(255,255,255,0.08) !important;
    border-radius: 6px;
}

/* ── Page header ── */
.page-header {
    padding: 2rem 2.2rem;
    background: #1C2B4A;
    border-radius: 10px;
    margin-bottom: 1.8rem;
}
.page-header-eyebrow {
    font-size: 0.68rem;
    font-weight: 600;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    color: #94A3B8;
    margin-bottom: 0.35rem;
}
.page-header-title {
    font-family: 'DM Serif Display', serif;
    font-size: 1.75rem;
    color: #FFFFFF;
    line-height: 1.2;
    margin-bottom: 0.25rem;
}
.page-header-sub {
    font-size: 0.85rem;
    color: #94A3B8;
    font-weight: 300;
}

/* ── Cards ── */
.card {
    background: #FFFFFF;
    border: 1px solid #E5E7EB;
    border-radius: 10px;
    padding: 1.3rem 1.7rem;
    margin-bottom: 1rem;
}
.card-label {
    font-size: 0.68rem;
    font-weight: 600;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: #6B7280;
    margin-bottom: 0.7rem;
}

/* ── Homework cards on dashboard ── */
.hw-card {
    background: #FFFFFF;
    border: 1px solid #E5E7EB;
    border-radius: 10px;
    padding: 1.1rem 1.5rem;
    margin-bottom: 0.7rem;
}
.hw-card-locked {
    background: #F9FAFB;
    border: 1px solid #E5E7EB;
    border-radius: 10px;
    padding: 1.1rem 1.5rem;
    margin-bottom: 0.7rem;
    opacity: 0.55;
}
.hw-title { font-size: 0.97rem; font-weight: 600; color: #111827; margin-bottom: 0.15rem; }
.hw-meta  { font-size: 0.8rem;  color: #6B7280; }
.hw-score { font-size: 0.82rem; color: #2563EB; font-weight: 500; margin-top: 0.1rem; }

/* ── Badges ── */
.badge {
    display: inline-block;
    padding: 2px 9px;
    border-radius: 20px;
    font-size: 0.7rem;
    font-weight: 600;
    letter-spacing: 0.03em;
}
.badge-open     { background: #F0FDF4; color: #16A34A; border: 1px solid #BBF7D0; }
.badge-closed   { background: #FEF2F2; color: #DC2626; border: 1px solid #FECACA; }
.badge-locked   { background: #F3F4F6; color: #6B7280; border: 1px solid #D1D5DB; }
.badge-complete { background: #EFF4FF; color: #2563EB; border: 1px solid #BFDBFE; }
.badge-graded   { background: #FEF3C7; color: #92400E; border: 1px solid #FDE68A; }
.badge-pending  { background: #FFF7ED; color: #C2410C; border: 1px solid #FED7AA; }

/* ── Question header ── */
.q-header {
    background: #1C2B4A;
    border-radius: 10px 10px 0 0;
    padding: 0.95rem 1.5rem;
    display: flex;
    justify-content: space-between;
    align-items: center;
}
.q-header-title { font-family: 'DM Serif Display', serif; color: #FFFFFF; font-size: 1.05rem; }
.q-header-sub   { color: #94A3B8; font-size: 0.77rem; margin-top: 0.12rem; }

.q-body {
    background: #F9FAFB;
    border: 1px solid #E5E7EB;
    border-top: none;
    border-radius: 0 0 10px 10px;
    padding: 1.1rem 1.5rem;
    margin-bottom: 1.1rem;
    font-size: 0.92rem;
    line-height: 1.75;
}
.part-row {
    display: flex;
    align-items: flex-start;
    gap: 10px;
    background: #FFFFFF;
    border: 1px solid #E5E7EB;
    border-radius: 7px;
    padding: 0.65rem 0.95rem;
    margin: 0.4rem 0;
}
.part-badge    { background: #1C2B4A; color: #FFFFFF; border-radius: 4px; padding: 2px 8px; font-size: 0.7rem; font-weight: 600; white-space: nowrap; flex-shrink: 0; margin-top: 3px; }
.part-text     { flex: 1; font-size: 0.88rem; line-height: 1.6; color: #374151; }
.part-ungraded { background: #F3F4F6; color: #6B7280; border-radius: 4px; padding: 2px 7px; font-size: 0.68rem; white-space: nowrap; flex-shrink: 0; }

/* ── Answer area ── */
.answer-area {
    background: #FFFFFF;
    border: 1px solid #E5E7EB;
    border-radius: 10px;
    padding: 1.1rem 1.5rem;
    margin-bottom: 0.9rem;
}
.answer-label {
    font-size: 0.68rem;
    font-weight: 600;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: #6B7280;
    margin-bottom: 0.7rem;
}

/* ── Timer bar ── */
.timer-bar {
    display: flex;
    gap: 1.5rem;
    background: #F9FAFB;
    border: 1px solid #E5E7EB;
    border-radius: 7px;
    padding: 0.55rem 1rem;
    margin-bottom: 0.9rem;
    font-size: 0.8rem;
    color: #6B7280;
    flex-wrap: wrap;
}
.timer-item  { display: flex; align-items: center; gap: 5px; }
.timer-value { font-weight: 600; color: #111827; font-variant-numeric: tabular-nums; }
.timer-warn  { color: #DC2626; }

/* ── Question summary table ── */
.q-summary-table {
    width: 100%;
    border-collapse: collapse;
    font-size: 0.87rem;
    margin-bottom: 1.5rem;
}
.q-summary-table th {
    background: #1C2B4A;
    color: #FFFFFF;
    padding: 8px 12px;
    text-align: left;
    font-weight: 500;
    font-size: 0.75rem;
    letter-spacing: 0.04em;
}
.q-summary-table td {
    padding: 8px 12px;
    border-bottom: 1px solid #F3F4F6;
    color: #374151;
}
.q-summary-table tr:hover td { background: #F9FAFB; }
.q-summary-table tr:last-child td { border-bottom: none; }
.q-summary-total td {
    background: #F3F4F6;
    font-weight: 600;
    border-bottom: none;
}
.status-submitted { color: #16A34A; font-weight: 500; }
.status-not-started { color: #6B7280; }

/* ── Solution card ── */
.sol-card {
    background: #FFFFFF;
    border: 1px solid #E5E7EB;
    border-radius: 10px;
    overflow: hidden;
    margin-top: 0.9rem;
}
.sol-header {
    background: #1C2B4A;
    color: #FFFFFF;
    padding: 0.65rem 1.3rem;
    font-size: 0.87rem;
    font-weight: 600;
    letter-spacing: 0.02em;
}
.sol-section {
    padding: 0.9rem 1.3rem;
    border-bottom: 1px solid #F3F4F6;
}
.sol-section:last-child { border-bottom: none; }
.sol-section-label {
    font-size: 0.68rem;
    font-weight: 600;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: #6B7280;
    margin-bottom: 0.5rem;
}
.sol-steps-box {
    background: #EFF6FF;
    border-left: 3px solid #2563EB;
    border-radius: 0 6px 6px 0;
    padding: 0.85rem 1rem;
    font-size: 0.88rem;
    line-height: 1.8;
}
.sol-mistakes-box {
    background: #FFF7ED;
    border-left: 3px solid #F97316;
    border-radius: 0 6px 6px 0;
    padding: 0.85rem 1rem;
    font-size: 0.88rem;
    line-height: 1.8;
}
.sol-revise-box {
    background: #F0FDF4;
    border-left: 3px solid #16A34A;
    border-radius: 0 6px 6px 0;
    padding: 0.85rem 1rem;
    font-size: 0.88rem;
    line-height: 1.8;
}

/* ── Score table ── */
.score-table {
    width: 100%;
    border-collapse: collapse;
    font-size: 0.86rem;
    margin-top: 0.7rem;
}
.score-table th {
    background: #F3F4F6;
    padding: 6px 10px;
    text-align: left;
    font-weight: 600;
    color: #374151;
    border-bottom: 1px solid #E5E7EB;
}
.score-table td {
    padding: 6px 10px;
    border-bottom: 1px solid #F3F4F6;
    color: #374151;
}
.score-table tr:last-child td {
    background: #F9FAFB;
    font-weight: 600;
    border-bottom: none;
}
.chip-ok      { background: #F0FDF4; color: #16A34A; border-radius: 4px; padding: 2px 7px; font-size: 0.76rem; font-weight: 600; }
.chip-wrong   { background: #FEF2F2; color: #DC2626; border-radius: 4px; padding: 2px 7px; font-size: 0.76rem; font-weight: 600; }

/* ── Banners ── */
.banner-success { background: #F0FDF4; border: 1px solid #BBF7D0; border-radius: 8px; padding: 0.85rem 1.1rem; font-size: 0.88rem; color: #166534; margin: 0.7rem 0; }
.banner-warning { background: #FFFBEB; border: 1px solid #FDE68A; border-radius: 8px; padding: 0.85rem 1.1rem; font-size: 0.88rem; color: #92400E; margin: 0.7rem 0; }
.banner-error   { background: #FEF2F2; border: 1px solid #FECACA; border-radius: 8px; padding: 0.85rem 1.1rem; font-size: 0.88rem; color: #991B1B; margin: 0.7rem 0; }
.banner-info    { background: #EFF4FF; border: 1px solid #BFDBFE; border-radius: 8px; padding: 0.85rem 1.1rem; font-size: 0.88rem; color: #1E40AF; margin: 0.7rem 0; }
.banner-restore { background: #F9FAFB; border: 1px solid #D1D5DB; border-radius: 8px; padding: 0.75rem 1.1rem; font-size: 0.84rem; color: #374151; margin: 0.7rem 0; }
.banner-locked  { background: #F5F3FF; border: 1px solid #DDD6FE; border-radius: 8px; padding: 0.75rem 1.1rem; font-size: 0.84rem; color: #5B21B6; margin: 0.7rem 0; }
.banner-review  { background: #EFF4FF; border: 1px solid #BFDBFE; border-radius: 8px; padding: 0.85rem 1.1rem; font-size: 0.88rem; color: #1E40AF; margin: 0.7rem 0; }

/* ── Param chips ── */
.param-chip {
    display: inline-block;
    background: #F3F4F6;
    border: 1px solid #E5E7EB;
    border-radius: 5px;
    padding: 3px 9px;
    font-size: 0.8rem;
    color: #374151;
    margin: 2px 3px 2px 0;
    font-variant-numeric: tabular-nums;
}

/* ── Progress ── */
.progress-wrap { background: #F9FAFB; border: 1px solid #E5E7EB; border-radius: 8px; padding: 0.85rem 1.1rem; margin-bottom: 1.1rem; }
.progress-label { font-size: 0.68rem; font-weight: 600; letter-spacing: 0.1em; text-transform: uppercase; color: #6B7280; margin-bottom: 0.45rem; }
.progress-bar-outer { background: #E5E7EB; border-radius: 20px; height: 5px; overflow: hidden; }
.progress-bar-inner { height: 100%; border-radius: 20px; background: #2563EB; }
.progress-text { font-size: 0.78rem; color: #6B7280; margin-top: 0.3rem; }

/* ── Semester summary ── */
.semester-summary {
    background: #1C2B4A;
    border-radius: 10px;
    padding: 1.2rem 1.6rem;
    margin-bottom: 1.5rem;
    display: flex;
    justify-content: space-between;
    align-items: center;
    flex-wrap: wrap;
    gap: 1rem;
}
.sem-sum-label { font-size: 0.68rem; font-weight: 600; letter-spacing: 0.1em; text-transform: uppercase; color: #94A3B8; margin-bottom: 0.2rem; }
.sem-sum-value { font-family: 'DM Serif Display', serif; font-size: 1.8rem; color: #FFFFFF; line-height: 1; }
.sem-sum-sub   { font-size: 0.78rem; color: #64748B; margin-top: 0.15rem; }

/* ── Inputs ── */
.stTextInput > div > div > input,
.stNumberInput > div > div > input,
.stTextArea > div > div > textarea {
    border: 1px solid #D1D5DB !important;
    border-radius: 6px !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 0.9rem !important;
    color: #111827 !important;
    background: #FFFFFF !important;
}
.stTextInput > div > div > input:focus,
.stNumberInput > div > div > input:focus,
.stTextArea > div > div > textarea:focus {
    border-color: #2563EB !important;
    box-shadow: 0 0 0 3px rgba(37,99,235,0.08) !important;
    outline: none !important;
}

/* ── Buttons ── */
.stButton > button {
    background: #1C2B4A !important;
    color: #FFFFFF !important;
    border: none !important;
    border-radius: 6px !important;
    font-family: 'DM Sans', sans-serif !important;
    font-weight: 500 !important;
    font-size: 0.88rem !important;
    padding: 0.45rem 1.3rem !important;
    transition: background 0.15s ease !important;
    letter-spacing: 0.01em !important;
}
.stButton > button:hover { background: #2C3E6B !important; }

/* ── Flag section ── */
.flag-section { border-top: 1px solid #F3F4F6; padding-top: 0.75rem; margin-top: 0.75rem; }

/* ── Mobile warning ── */
.mobile-warning { display: none; background: #FFFBEB; border: 1px solid #FDE68A; border-radius: 8px; padding: 0.85rem 1.1rem; font-size: 0.86rem; color: #92400E; margin-bottom: 1rem; }
@media (max-width: 768px) { .mobile-warning { display: block; } .block-container { padding-left: 0.8rem !important; padding-right: 0.8rem !important; } }

/* ── Unsaved warning ── */
.unsaved-warning { background: #FFFBEB; border: 1px solid #FDE68A; border-radius: 7px; padding: 0.6rem 1rem; font-size: 0.82rem; color: #92400E; margin-bottom: 0.6rem; }

/* ── Review mode banner ── */
.review-mode-banner {
    background: #EFF4FF;
    border: 1.5px solid #BFDBFE;
    border-radius: 8px;
    padding: 0.9rem 1.2rem;
    font-size: 0.88rem;
    color: #1E40AF;
    margin-bottom: 1.2rem;
    text-align: center;
}

/* ── FAQ ── */
.faq-q { font-weight: 600; color: #1C2B4A; font-size: 0.92rem; margin-bottom: 0.3rem; }
.faq-a { color: #374151; font-size: 0.88rem; line-height: 1.7; }

/* ── Course materials ── */
.material-row {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 0.7rem 1rem;
    background: #FFFFFF;
    border: 1px solid #E5E7EB;
    border-radius: 7px;
    margin-bottom: 0.4rem;
}
.material-title { font-size: 0.9rem; font-weight: 500; color: #111827; }
.material-meta  { font-size: 0.76rem; color: #6B7280; margin-top: 0.1rem; }

/* ── Countdown on dashboard ── */
.countdown-bar {
    display: flex;
    gap: 1.2rem;
    flex-wrap: wrap;
    margin-top: 0.35rem;
}
.countdown-item { font-size: 0.78rem; color: #6B7280; }
.countdown-value { font-weight: 600; color: #2563EB; }
.countdown-urgent { color: #DC2626; }

/* ── Print styles ── */
@media print {
    [data-testid="stSidebar"] { display: none !important; }
    .stButton { display: none !important; }
    .block-container { max-width: 100% !important; padding: 0 !important; }
    .page-header { background: #1C2B4A !important; -webkit-print-color-adjust: exact; }
    .sol-header  { background: #1C2B4A !important; -webkit-print-color-adjust: exact; }
}
</style>
"""


def inject_css():
    import streamlit as st
    st.markdown(GLOBAL_CSS, unsafe_allow_html=True)


def page_header(eyebrow: str, title: str, subtitle: str = ""):
    import streamlit as st
    sub = f'<div class="page-header-sub">{subtitle}</div>' if subtitle else ""
    st.markdown(f"""
    <div class="page-header">
      <div class="page-header-eyebrow">{eyebrow}</div>
      <div class="page-header-title">{title}</div>
      {sub}
    </div>
    """, unsafe_allow_html=True)


def banner(text: str, kind: str = "info"):
    import streamlit as st
    st.markdown(f'<div class="banner-{kind}">{text}</div>', unsafe_allow_html=True)
