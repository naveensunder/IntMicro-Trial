"""
ui.py — Shared design system.
HWDashboard v3 — stability-first build.
Key rule: NO dynamic HTML assembly. Every st.markdown() block is
a complete, static string. No f-string variable interpolation of HTML fragments.
"""

COLORS = {
    "navy":        "#1C2B4A",
    "navy_mid":    "#2C3E6B",
    "accent":      "#2563EB",
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

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600&family=DM+Serif+Display&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    color: #111827;
}
#MainMenu, footer, header, .stDeployButton { visibility: hidden; display: none; }
.block-container { padding-top: 1.8rem; padding-bottom: 3rem; max-width: 860px; }

[data-testid="stSidebar"] { background: #1C2B4A !important; }
[data-testid="stSidebar"] * { color: #CBD5E1 !important; }

.page-header {
    padding: 1.8rem 2rem; background: #1C2B4A;
    border-radius: 10px; margin-bottom: 1.6rem;
}
.page-header-eye {
    font-size: 0.67rem; font-weight: 600; letter-spacing: 0.14em;
    text-transform: uppercase; color: #94A3B8; margin-bottom: 0.3rem;
}
.page-header-title {
    font-family: 'DM Serif Display', serif;
    font-size: 1.65rem; color: #FFFFFF; line-height: 1.2; margin-bottom: 0.2rem;
}
.page-header-sub { font-size: 0.83rem; color: #94A3B8; }

.hw-card {
    background: #FFFFFF; border: 1px solid #E5E7EB;
    border-radius: 10px; padding: 1.1rem 1.5rem; margin-bottom: 0.7rem;
}
.hw-card-locked {
    background: #F9FAFB; border: 1px solid #E5E7EB;
    border-radius: 10px; padding: 1.1rem 1.5rem; margin-bottom: 0.7rem; opacity: 0.55;
}
.hw-title { font-size: 0.96rem; font-weight: 600; color: #111827; margin-bottom: 0.12rem; }
.hw-meta  { font-size: 0.79rem; color: #6B7280; }
.hw-score { font-size: 0.82rem; color: #2563EB; font-weight: 500; margin-top: 0.1rem; }

.badge { display: inline-block; padding: 2px 9px; border-radius: 20px; font-size: 0.69rem; font-weight: 600; }
.badge-open     { background: #F0FDF4; color: #16A34A; border: 1px solid #BBF7D0; }
.badge-closed   { background: #FEF2F2; color: #DC2626; border: 1px solid #FECACA; }
.badge-locked   { background: #F3F4F6; color: #6B7280; border: 1px solid #D1D5DB; }
.badge-complete { background: #EFF4FF; color: #2563EB; border: 1px solid #BFDBFE; }

.q-header {
    background: #1C2B4A; border-radius: 10px 10px 0 0;
    padding: 0.9rem 1.4rem; margin-top: 1rem;
}
.q-header-title { font-family: 'DM Serif Display', serif; color: #FFFFFF; font-size: 1.05rem; }
.q-header-pts   { color: #94A3B8; font-size: 0.77rem; margin-top: 0.1rem; }

.q-body {
    background: #F9FAFB; border: 1px solid #E5E7EB; border-top: none;
    border-radius: 0 0 10px 10px; padding: 1rem 1.4rem;
    margin-bottom: 1rem; font-size: 0.91rem; line-height: 1.75;
}
.part-row {
    display: flex; align-items: flex-start; gap: 9px;
    background: #FFFFFF; border: 1px solid #E5E7EB;
    border-radius: 6px; padding: 0.6rem 0.9rem; margin: 0.35rem 0;
}
.part-badge {
    background: #1C2B4A; color: #FFFFFF; border-radius: 4px;
    padding: 2px 7px; font-size: 0.69rem; font-weight: 600;
    white-space: nowrap; flex-shrink: 0; margin-top: 2px;
}
.part-text     { flex: 1; font-size: 0.87rem; line-height: 1.6; color: #374151; }
.part-ungraded { background: #F3F4F6; color: #6B7280; border-radius: 4px; padding: 2px 7px; font-size: 0.67rem; white-space: nowrap; flex-shrink: 0; }

.answer-area {
    background: #FFFFFF; border: 1px solid #E5E7EB;
    border-radius: 10px; padding: 1rem 1.4rem; margin-bottom: 0.8rem;
}
.answer-label {
    font-size: 0.67rem; font-weight: 600; letter-spacing: 0.1em;
    text-transform: uppercase; color: #6B7280; margin-bottom: 0.65rem;
}

.param-chip {
    display: inline-block; background: #F3F4F6; border: 1px solid #E5E7EB;
    border-radius: 4px; padding: 2px 8px; font-size: 0.79rem;
    color: #374151; margin: 2px 3px 2px 0;
}

.sol-card { background: #FFFFFF; border: 1px solid #E5E7EB; border-radius: 10px; overflow: hidden; margin-top: 0.8rem; }
.sol-header { background: #1C2B4A; color: #FFFFFF; padding: 0.6rem 1.2rem; font-size: 0.86rem; font-weight: 600; }
.sol-section { padding: 0.85rem 1.2rem; border-bottom: 1px solid #F3F4F6; }
.sol-section:last-child { border-bottom: none; }
.sol-label { font-size: 0.67rem; font-weight: 600; letter-spacing: 0.1em; text-transform: uppercase; color: #6B7280; margin-bottom: 0.45rem; }
.sol-steps  { background: #EFF6FF; border-left: 3px solid #2563EB; border-radius: 0 5px 5px 0; padding: 0.8rem 0.95rem; font-size: 0.87rem; line-height: 1.8; }
.sol-mistakes { background: #FFF7ED; border-left: 3px solid #F97316; border-radius: 0 5px 5px 0; padding: 0.8rem 0.95rem; font-size: 0.87rem; line-height: 1.8; }
.sol-revise   { background: #F0FDF4; border-left: 3px solid #16A34A; border-radius: 0 5px 5px 0; padding: 0.8rem 0.95rem; font-size: 0.87rem; line-height: 1.8; }

.score-table { width: 100%; border-collapse: collapse; font-size: 0.85rem; margin-top: 0.65rem; }
.score-table th { background: #F3F4F6; padding: 6px 9px; text-align: left; font-weight: 600; color: #374151; border-bottom: 1px solid #E5E7EB; }
.score-table td { padding: 6px 9px; border-bottom: 1px solid #F3F4F6; color: #374151; }
.score-table .total-row td { background: #F9FAFB; font-weight: 600; border-bottom: none; }
.chip-ok    { background: #F0FDF4; color: #16A34A; border-radius: 4px; padding: 2px 6px; font-size: 0.75rem; font-weight: 600; }
.chip-wrong { background: #FEF2F2; color: #DC2626; border-radius: 4px; padding: 2px 6px; font-size: 0.75rem; font-weight: 600; }

.q-summary { width: 100%; border-collapse: collapse; font-size: 0.86rem; margin-bottom: 1.3rem; }
.q-summary th { background: #1C2B4A; color: #FFFFFF; padding: 7px 11px; text-align: left; font-size: 0.74rem; font-weight: 500; letter-spacing: 0.03em; }
.q-summary td { padding: 7px 11px; border-bottom: 1px solid #F3F4F6; color: #374151; }
.q-summary .total-row td { background: #F3F4F6; font-weight: 600; border-bottom: none; }
.status-done { color: #16A34A; font-weight: 500; }
.status-todo { color: #6B7280; }

.banner { border-radius: 7px; padding: 0.8rem 1rem; font-size: 0.87rem; margin: 0.65rem 0; }
.banner-success { background: #F0FDF4; border: 1px solid #BBF7D0; color: #166534; }
.banner-warning { background: #FFFBEB; border: 1px solid #FDE68A; color: #92400E; }
.banner-error   { background: #FEF2F2; border: 1px solid #FECACA; color: #991B1B; }
.banner-info    { background: #EFF4FF; border: 1px solid #BFDBFE; color: #1E40AF; }
.banner-locked  { background: #F5F3FF; border: 1px solid #DDD6FE; color: #5B21B6; }
.banner-restore { background: #F9FAFB; border: 1px solid #D1D5DB; color: #374151; }

.saved-ts { font-size: 0.78rem; color: #6B7280; margin-top: 0.3rem; font-style: italic; }

.stTextInput > div > div > input,
.stNumberInput > div > div > input,
.stTextArea > div > div > textarea {
    border: 1px solid #D1D5DB !important; border-radius: 6px !important;
    font-family: 'DM Sans', sans-serif !important; font-size: 0.9rem !important;
}
.stTextInput > div > div > input:focus,
.stNumberInput > div > div > input:focus {
    border-color: #2563EB !important;
    box-shadow: 0 0 0 3px rgba(37,99,235,0.08) !important;
}
.stButton > button {
    background: #1C2B4A !important; color: #FFFFFF !important;
    border: none !important; border-radius: 6px !important;
    font-family: 'DM Sans', sans-serif !important;
    font-weight: 500 !important; font-size: 0.88rem !important;
    padding: 0.45rem 1.3rem !important;
}
.stButton > button:hover { background: #2C3E6B !important; }

.mobile-warn {
    display: none; background: #FFFBEB; border: 1px solid #FDE68A;
    border-radius: 7px; padding: 0.8rem 1rem; font-size: 0.85rem;
    color: #92400E; margin-bottom: 0.8rem;
}
@media (max-width: 768px) { .mobile-warn { display: block; } }

.faq-a { color: #374151; font-size: 0.87rem; line-height: 1.7; }
.mat-row {
    display: flex; align-items: center; justify-content: space-between;
    padding: 0.65rem 0.95rem; background: #FFFFFF;
    border: 1px solid #E5E7EB; border-radius: 7px; margin-bottom: 0.35rem;
}
.mat-title { font-size: 0.89rem; font-weight: 500; color: #111827; }
.mat-meta  { font-size: 0.75rem; color: #6B7280; margin-top: 0.08rem; }

.sem-box {
    background: #1C2B4A; border-radius: 10px;
    padding: 1.1rem 1.5rem; margin-bottom: 1.4rem;
}
.sem-label { font-size: 0.67rem; font-weight: 600; letter-spacing: 0.1em; text-transform: uppercase; color: #94A3B8; margin-bottom: 0.2rem; }
.sem-score { font-family: 'DM Serif Display', serif; font-size: 1.7rem; color: #FFFFFF; line-height: 1; }
.sem-sub   { font-size: 0.78rem; color: #64748B; margin-top: 0.15rem; }

.instructor-note {
    background: #FFFBEB; border: 1px solid #FDE68A; border-radius: 7px;
    padding: 0.75rem 1rem; font-size: 0.84rem; color: #92400E; margin-bottom: 1rem;
}
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
        f'{sub_html}'
        f'</div>',
        unsafe_allow_html=True
    )


def banner(text: str, kind: str = "info"):
    import streamlit as st
    st.markdown(f'<div class="banner banner-{kind}">{text}</div>',
                unsafe_allow_html=True)
