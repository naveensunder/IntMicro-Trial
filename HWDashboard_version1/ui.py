"""
ui.py — Design system.
HWDashboard v11 — Phase 1: Full LMS-grade UI foundation.
"""
import streamlit as st

COLORS = {
    "navy":       "#1C2B4A",
    "navy_mid":   "#2C3E6B",
    "accent":     "#2563EB",
    "green":      "#2D6A4F",
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
    "ans_bg":     "#FAF9F6",
    "teal":       "#2A7F7F",
}

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600&family=DM+Serif+Display&display=swap');

/* ── Reset & base ── */
*, *::before, *::after { box-sizing: border-box; }

html {
    scroll-behavior: smooth;
}

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    font-size: 17px;
    color: #1A1A1A;
}

/* ── Dark mode ── */
@media (prefers-color-scheme: dark) {
    .dark-mode-enabled html,
    .dark-mode-enabled body,
    .dark-mode-enabled [class*="css"] {
        background-color: #0F172A !important;
        color: #E2E8F0 !important;
    }
}

body.dark-mode {
    background-color: #0F172A !important;
    color: #E2E8F0 !important;
}

body.dark-mode .hw-card,
body.dark-mode .ov-row,
body.dark-mode .mat-row {
    background: #1E293B !important;
    border-color: #334155 !important;
    color: #E2E8F0 !important;
}

body.dark-mode .q-body,
body.dark-mode .ans-section,
body.dark-mode .sol-section,
body.dark-mode .submitted-ans {
    background: #1E293B !important;
    color: #E2E8F0 !important;
}

body.dark-mode .stApp {
    background-color: #0F172A !important;
}

/* ── Hide Streamlit chrome ── */
#MainMenu, footer, header, .stDeployButton { visibility: hidden; display: none; }

/* ── Layout ── */
.block-container {
    padding-top: 1.5rem;
    padding-bottom: 4rem;
    max-width: 860px;
    margin: 0 auto;
}

/* ── Page transitions ── */
.block-container {
    animation: fadeSlideIn 0.28s ease-out both;
}

@keyframes fadeSlideIn {
    from { opacity: 0; transform: translateY(10px); }
    to   { opacity: 1; transform: translateY(0); }
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: #1C2B4A !important;
}
[data-testid="stSidebar"] * {
    color: #CBD5E1 !important;
    font-size: 15px !important;
}

/* ── Sidebar branding ── */
.sidebar-brand {
    padding: 1.2rem 1rem 1rem 1rem;
    border-bottom: 1px solid rgba(255,255,255,0.1);
    margin-bottom: 0.8rem;
}
.sidebar-course {
    font-family: 'DM Serif Display', serif;
    font-size: 1.1rem;
    color: #FFFFFF !important;
    line-height: 1.2;
    margin-bottom: 0.15rem;
}
.sidebar-semester {
    font-size: 0.75rem;
    color: #94A3B8 !important;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.1em;
}
.sidebar-student {
    margin-top: 0.7rem;
    display: flex;
    align-items: center;
    gap: 0.6rem;
}
.sidebar-avatar {
    width: 32px; height: 32px;
    border-radius: 50%;
    background: #2563EB;
    display: flex; align-items: center; justify-content: center;
    font-size: 0.85rem; font-weight: 700; color: #FFFFFF !important;
    flex-shrink: 0;
}
.sidebar-name {
    font-size: 0.88rem;
    color: #CBD5E1 !important;
    font-weight: 500;
}

/* ── Breadcrumbs ── */
.breadcrumb {
    display: flex;
    align-items: center;
    gap: 0.4rem;
    font-size: 0.82rem;
    color: #888888;
    margin-bottom: 1.2rem;
    flex-wrap: wrap;
}
.breadcrumb-item { color: #888888; }
.breadcrumb-item.active { color: #1C2B4A; font-weight: 600; }
.breadcrumb-sep { color: #CCCCCC; font-size: 0.75rem; }

/* ── Page header ── */
.page-header {
    padding: 1.8rem 2rem;
    background: #1C2B4A;
    border-radius: 12px;
    margin-bottom: 1.8rem;
    position: relative;
    overflow: hidden;
}
.page-header::after {
    content: '';
    position: absolute;
    top: -40px; right: -40px;
    width: 180px; height: 180px;
    border-radius: 50%;
    background: rgba(37,99,235,0.15);
    pointer-events: none;
}
.page-header-eye {
    font-size: 0.72rem; font-weight: 600; letter-spacing: 0.14em;
    text-transform: uppercase; color: #94A3B8; margin-bottom: 0.35rem;
}
.page-header-title {
    font-family: 'DM Serif Display', serif;
    font-size: 1.9rem; color: #FFFFFF; line-height: 1.2; margin-bottom: 0.2rem;
}
.page-header-sub { font-size: 0.95rem; color: #94A3B8; }

/* ── Section headers ── */
.sec-header {
    display: flex;
    align-items: center;
    gap: 0.6rem;
    font-family: 'DM Serif Display', serif;
    font-size: 1.05rem;
    color: #1C2B4A;
    margin: 1.8rem 0 0.8rem 0;
    padding-bottom: 0.4rem;
    border-bottom: 2px solid #E0E0E0;
}
.sec-header-icon {
    width: 22px; height: 22px;
    background: #1C2B4A;
    border-radius: 5px;
    display: inline-flex;
    align-items: center;
    justify-content: center;
    font-size: 0.7rem;
    color: #FFFFFF;
    flex-shrink: 0;
}

/* ── Section dividers with labels ── */
.section-rule {
    display: flex;
    align-items: center;
    gap: 1rem;
    margin: 1.8rem 0 1.2rem 0;
    color: #888888;
    font-size: 0.78rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.1em;
}
.section-rule::before,
.section-rule::after {
    content: '';
    flex: 1;
    height: 1px;
    background: #E0E0E0;
}

/* ── Skeleton loaders ── */
@keyframes shimmer {
    0%   { background-position: -800px 0; }
    100% { background-position: 800px 0; }
}
.skeleton {
    background: linear-gradient(90deg, #F0F0F0 25%, #E8E8E8 50%, #F0F0F0 75%);
    background-size: 800px 100%;
    animation: shimmer 1.4s ease-in-out infinite;
    border-radius: 6px;
}
.skeleton-text  { height: 14px; margin-bottom: 8px; }
.skeleton-title { height: 22px; margin-bottom: 12px; width: 60%; }
.skeleton-card  { height: 90px; margin-bottom: 12px; border-radius: 10px; }
.skeleton-btn   { height: 38px; width: 120px; border-radius: 7px; }

/* ── Homework cards ── */
.hw-card {
    background: #FFFFFF;
    border: 1.5px solid #E0E0E0;
    border-radius: 12px;
    padding: 1.2rem 1.5rem;
    margin-bottom: 0.75rem;
    transition: box-shadow 0.18s ease, transform 0.18s ease, border-color 0.18s ease;
}
.hw-card:hover {
    box-shadow: 0 4px 16px rgba(28,43,74,0.10);
    transform: translateY(-1px);
    border-color: #C5D0E0;
}
.hw-card-open {
    background: #F0F7F4;
    border: 2px solid #95D5B2;
    border-left: 5px solid #2D6A4F;
    border-radius: 12px;
    padding: 1.2rem 1.5rem;
    margin-bottom: 0.75rem;
    transition: box-shadow 0.18s ease, transform 0.18s ease;
}
.hw-card-open:hover {
    box-shadow: 0 4px 20px rgba(45,106,79,0.12);
    transform: translateY(-1px);
}
.hw-card-locked {
    background: #F5F5F5;
    border: 1.5px solid #E0E0E0;
    border-radius: 12px;
    padding: 1.2rem 1.5rem;
    margin-bottom: 0.75rem;
    opacity: 0.6;
}
.hw-card-header {
    display: flex;
    align-items: flex-start;
    justify-content: space-between;
    gap: 1rem;
    margin-bottom: 0.5rem;
}
.hw-title      { font-size: 1.05rem; font-weight: 600; color: #1A1A1A; margin-bottom: 0.15rem; }
.hw-title-open { font-size: 1.05rem; font-weight: 600; color: #2D6A4F; margin-bottom: 0.15rem; }
.hw-meta       { font-size: 0.87rem; color: #555555; margin-top: 0.1rem; }
.hw-score      { font-size: 0.9rem; color: #2563EB; font-weight: 600; margin-top: 0.15rem; }
.hw-deadline-urgent { font-size: 0.9rem; color: #DC2626; font-weight: 600; margin-top: 0.1rem; }
.hw-due-date {
    font-size: 0.82rem;
    font-weight: 700;
    color: #1C2B4A;
    text-align: right;
    white-space: nowrap;
    flex-shrink: 0;
}
.hw-due-date-label {
    font-size: 0.7rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: #888888;
    margin-bottom: 0.1rem;
}
.hw-pts {
    font-size: 0.8rem;
    color: #888888;
    margin-top: 0.1rem;
    text-align: right;
}

/* ── NEW badge on assignments ── */
.badge-new {
    display: inline-block;
    padding: 1px 7px;
    border-radius: 20px;
    font-size: 0.68rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    background: #FEF3C7;
    color: #92400E;
    border: 1px solid #FDE68A;
    margin-left: 0.4rem;
    vertical-align: middle;
}

/* ── Status pills (replacing emoji badges) ── */
.pill {
    display: inline-flex;
    align-items: center;
    gap: 0.3rem;
    padding: 3px 10px;
    border-radius: 20px;
    font-size: 0.76rem;
    font-weight: 600;
    letter-spacing: 0.02em;
}
.pill::before {
    content: '';
    display: inline-block;
    width: 6px; height: 6px;
    border-radius: 50%;
}
.pill-open     { background: #F0F7F4; color: #2D6A4F; border: 1.5px solid #95D5B2; }
.pill-open::before     { background: #2D6A4F; }
.pill-closed   { background: #FEF2F2; color: #DC2626; border: 1.5px solid #FCA5A5; }
.pill-closed::before   { background: #DC2626; }
.pill-upcoming { background: #FFFBEB; color: #92400E; border: 1.5px solid #FDE68A; }
.pill-upcoming::before { background: #D97706; }
.pill-complete { background: #EFF4FF; color: #2563EB; border: 1.5px solid #93C5FD; }
.pill-complete::before { background: #2563EB; }
.pill-graded   { background: #F5F3FF; color: #5B21B6; border: 1.5px solid #C4B5FD; }
.pill-graded::before   { background: #7C3AED; }

/* ── Legacy badge aliases (backwards compat) ── */
.badge { display: inline-block; padding: 3px 11px; border-radius: 20px; font-size: 0.78rem; font-weight: 600; }
.badge-open     { background: #F0F7F4; color: #2D6A4F; border: 1.5px solid #95D5B2; }
.badge-closed   { background: #FEF2F2; color: #DC2626; border: 1.5px solid #FCA5A5; }
.badge-locked   { background: #F5F5F5; color: #555555; border: 1.5px solid #E0E0E0; }
.badge-complete { background: #EFF4FF; color: #2563EB; border: 1.5px solid #93C5FD; }

/* ── Countdown timer ── */
.countdown-wrap {
    display: inline-flex;
    align-items: center;
    gap: 0.5rem;
    background: #FFF7ED;
    border: 1.5px solid #FED7AA;
    border-radius: 8px;
    padding: 0.35rem 0.8rem;
    font-size: 0.85rem;
    font-weight: 600;
    color: #C2410C;
}
.countdown-wrap.urgent {
    background: #FEF2F2;
    border-color: #FCA5A5;
    color: #DC2626;
    animation: pulse 1.5s ease-in-out infinite;
}
@keyframes pulse {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.7; }
}
.countdown-icon { font-size: 0.95rem; }

/* ── Sticky homework header ── */
.hw-sticky-header {
    position: sticky;
    top: 0;
    z-index: 100;
    background: #FFFFFF;
    border-bottom: 2px solid #E0E0E0;
    padding: 0.7rem 0;
    margin-bottom: 1rem;
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 1rem;
}
.hw-sticky-title {
    font-family: 'DM Serif Display', serif;
    font-size: 1.1rem;
    color: #1C2B4A;
    font-weight: 400;
}

/* ── Progress bar ── */
.progress-container {
    margin-bottom: 1.2rem;
}
.progress-bar-wrap {
    background: #E0E0E0;
    border-radius: 99px;
    height: 8px;
    overflow: hidden;
    margin-top: 0.35rem;
}
.progress-bar-fill {
    height: 100%;
    border-radius: 99px;
    background: linear-gradient(90deg, #2D6A4F, #52B788);
    transition: width 0.5s ease;
}
.progress-label {
    display: flex;
    justify-content: space-between;
    font-size: 0.8rem;
    color: #555555;
    font-weight: 500;
}
.progress-label strong { color: #1C2B4A; }

/* ── Progress line (legacy) ── */
.progress-line {
    font-size: 0.95rem; color: #555555;
    background: #F5F5F5; border-radius: 7px;
    padding: 0.6rem 1rem; margin-bottom: 1.2rem;
}
.progress-line strong { color: #1C2B4A; }

/* ── Question cards ── */
.q-card {
    background: #FFFFFF;
    border: 1.5px solid #E0E0E0;
    border-radius: 12px;
    overflow: hidden;
    margin-bottom: 1.5rem;
    transition: box-shadow 0.2s ease;
}
.q-card:hover {
    box-shadow: 0 6px 24px rgba(28,43,74,0.09);
}
.q-card-submitted {
    background: #FFFFFF;
    border: 1.5px solid #95D5B2;
    border-radius: 12px;
    overflow: hidden;
    margin-bottom: 1.5rem;
}

/* ── Question banner ── */
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
    font-size: 1.35rem; font-weight: 700; color: #FFFFFF;
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
.section-divider { border: none; border-top: 1px solid #E0E0E0; margin: 1.2rem 0; }

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

/* ── Inline feedback banners (slide in) ── */
.feedback-banner {
    border-radius: 10px;
    padding: 1rem 1.3rem;
    font-size: 1rem;
    margin: 0.8rem 0;
    line-height: 1.5;
    display: flex;
    align-items: center;
    gap: 0.8rem;
    animation: slideDown 0.3s ease-out both;
}
@keyframes slideDown {
    from { opacity: 0; transform: translateY(-8px); }
    to   { opacity: 1; transform: translateY(0); }
}
.feedback-icon { font-size: 1.3rem; flex-shrink: 0; }
.feedback-title { font-weight: 600; margin-bottom: 0.1rem; }
.feedback-sub   { font-size: 0.88rem; opacity: 0.85; }
.feedback-success { background: #ECFDF5; border: 1.5px solid #86EFAC; color: #166534; }
.feedback-error   { background: #FEF2F2; border: 1.5px solid #FCA5A5; color: #991B1B; }
.feedback-partial { background: #FFFBEB; border: 1.5px solid #FDE68A; color: #92400E; }

/* ── Score reveal (JS animated) ── */
.score-reveal-wrap {
    text-align: center;
    padding: 1.5rem;
}
.score-reveal-num {
    font-family: 'DM Serif Display', serif;
    font-size: 3.5rem;
    color: #1C2B4A;
    line-height: 1;
}
.score-reveal-denom {
    font-size: 1.1rem;
    color: #888888;
    margin-top: 0.2rem;
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

/* ── Modal dialog ── */
.modal-overlay {
    position: fixed;
    top: 0; left: 0; right: 0; bottom: 0;
    background: rgba(0,0,0,0.5);
    z-index: 999;
    display: flex;
    align-items: center;
    justify-content: center;
    animation: fadeIn 0.15s ease;
}
@keyframes fadeIn { from { opacity: 0; } to { opacity: 1; } }
.modal-box {
    background: #FFFFFF;
    border-radius: 14px;
    padding: 2rem;
    max-width: 420px;
    width: 90%;
    box-shadow: 0 20px 60px rgba(0,0,0,0.25);
    animation: scaleIn 0.18s ease;
}
@keyframes scaleIn {
    from { transform: scale(0.94); opacity: 0; }
    to   { transform: scale(1);    opacity: 1; }
}
.modal-title {
    font-family: 'DM Serif Display', serif;
    font-size: 1.25rem;
    color: #1C2B4A;
    margin-bottom: 0.6rem;
}
.modal-body { font-size: 0.95rem; color: #555555; line-height: 1.6; margin-bottom: 1.4rem; }
.modal-actions { display: flex; gap: 0.7rem; justify-content: flex-end; }

/* ── Empty states ── */
.empty-state {
    text-align: center;
    padding: 3rem 1rem;
    color: #888888;
}
.empty-state-icon {
    font-size: 2.8rem;
    margin-bottom: 0.8rem;
    opacity: 0.6;
}
.empty-state-title {
    font-family: 'DM Serif Display', serif;
    font-size: 1.2rem;
    color: #555555;
    margin-bottom: 0.4rem;
}
.empty-state-sub {
    font-size: 0.9rem;
    color: #888888;
    line-height: 1.6;
}

/* ── Error state ── */
.error-state {
    background: #FEF2F2;
    border: 1.5px solid #FCA5A5;
    border-radius: 12px;
    padding: 2rem;
    text-align: center;
    color: #991B1B;
}
.error-state-title {
    font-family: 'DM Serif Display', serif;
    font-size: 1.2rem;
    margin-bottom: 0.5rem;
}
.error-state-sub { font-size: 0.9rem; opacity: 0.85; line-height: 1.6; }

/* ── Registration success ── */
.reg-success {
    background: linear-gradient(135deg, #F0F7F4 0%, #EFF4FF 100%);
    border: 2px solid #95D5B2;
    border-radius: 14px;
    padding: 2.5rem 2rem;
    text-align: center;
    animation: fadeSlideIn 0.4s ease-out both;
}
.reg-success-icon { font-size: 3rem; margin-bottom: 0.8rem; }
.reg-success-title {
    font-family: 'DM Serif Display', serif;
    font-size: 1.5rem;
    color: #1C2B4A;
    margin-bottom: 0.4rem;
}
.reg-success-sub { font-size: 0.95rem; color: #555555; line-height: 1.6; }

/* ── Saved timestamp ── */
.saved-ts { font-size: 0.82rem; color: #888888; margin-top: 0.25rem; font-style: italic; }

/* ── Last-updated chip ── */
.last-updated {
    display: inline-flex;
    align-items: center;
    gap: 0.3rem;
    font-size: 0.78rem;
    color: #888888;
    background: #F5F5F5;
    border: 1px solid #E0E0E0;
    border-radius: 20px;
    padding: 2px 10px;
}
.last-updated-dot {
    width: 6px; height: 6px;
    border-radius: 50%;
    background: #16A34A;
    flex-shrink: 0;
}

/* ── Semester summary ── */
.sem-box { background: #1C2B4A; border-radius: 10px; padding: 1.3rem 1.8rem; margin-bottom: 1.8rem; }
.sem-label { font-size: 0.78rem; font-weight: 600; letter-spacing: 0.1em; text-transform: uppercase; color: #94A3B8; margin-bottom: 0.25rem; }
.sem-score { font-family: 'DM Serif Display', serif; font-size: 2rem; color: #FFFFFF; line-height: 1; }
.sem-sub   { font-size: 0.92rem; color: #64748B; margin-top: 0.2rem; }

/* ── Score history chart row ── */
.score-hist-row {
    display: flex; align-items: center; gap: 0.8rem;
    padding: 0.6rem 0; border-bottom: 1px solid #F0F0F0; font-size: 0.92rem;
}
.score-hist-row:last-child { border-bottom: none; }
.score-hist-bar-wrap {
    flex: 1; background: #F0F0F0; border-radius: 99px; height: 6px; overflow: hidden;
}
.score-hist-bar {
    height: 100%; border-radius: 99px;
    background: linear-gradient(90deg, #2563EB, #60A5FA);
}

/* ── Question overview rows ── */
.ov-row {
    display: flex; align-items: center; justify-content: space-between;
    padding: 0.8rem 1rem; background: #FFFFFF;
    border: 1.5px solid #E0E0E0; border-radius: 8px; margin-bottom: 0.4rem;
    font-size: 1rem;
    transition: background 0.15s ease;
}
.ov-row:hover { background: #FAFAFA; }
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

/* ── Inputs ── */
.stTextInput > div > div > input,
.stNumberInput > div > div > input,
.stTextArea > div > div > textarea {
    border: 1.5px solid #E0E0E0 !important;
    border-radius: 7px !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 1rem !important;
    padding: 0.55rem 0.8rem !important;
    color: #1A1A1A !important;
    transition: border-color 0.15s ease, box-shadow 0.15s ease !important;
}
.stTextInput > div > div > input:focus,
.stNumberInput > div > div > input:focus,
.stTextArea > div > div > textarea:focus {
    border-color: #2563EB !important;
    box-shadow: 0 0 0 3px rgba(37,99,235,0.12) !important;
    outline: none !important;
}
.stRadio > div { gap: 1rem !important; }
.stRadio label { font-size: 1rem !important; }

/* ── Input validation states ── */
.input-error .stTextInput > div > div > input,
.input-error .stNumberInput > div > div > input {
    border-color: #DC2626 !important;
    box-shadow: 0 0 0 3px rgba(220,38,38,0.1) !important;
}
.input-ok .stTextInput > div > div > input {
    border-color: #16A34A !important;
    box-shadow: 0 0 0 3px rgba(22,163,74,0.1) !important;
}
.field-hint {
    font-size: 0.78rem;
    margin-top: 0.2rem;
    padding-left: 0.2rem;
}
.field-hint-error { color: #DC2626; }
.field-hint-ok    { color: #16A34A; }

/* ── Buttons ── */
.stButton > button {
    background: #1C2B4A !important;
    color: #FFFFFF !important;
    border: none !important;
    border-radius: 7px !important;
    font-family: 'DM Sans', sans-serif !important;
    font-weight: 500 !important;
    font-size: 1rem !important;
    padding: 0.55rem 1.5rem !important;
    transition: background 0.15s ease, transform 0.1s ease, box-shadow 0.15s ease !important;
}
.stButton > button:hover {
    background: #2C3E6B !important;
    box-shadow: 0 4px 12px rgba(28,43,74,0.25) !important;
    transform: translateY(-1px) !important;
}
.stButton > button:active {
    transform: translateY(0) !important;
    box-shadow: none !important;
}

/* ── Expander ── */
.streamlit-expanderHeader {
    font-size: 1.05rem !important;
    font-weight: 700 !important;
    color: #1C2B4A !important;
}

/* ── Instructor metric cards ── */
.dash-metric {
    background: #FFFFFF;
    border: 1.5px solid #E0E0E0;
    border-radius: 12px;
    padding: 1.1rem 1.3rem;
    text-align: center;
    transition: box-shadow 0.18s ease, transform 0.18s ease;
}
.dash-metric:hover {
    box-shadow: 0 4px 16px rgba(28,43,74,0.08);
    transform: translateY(-1px);
}
.dash-metric-num {
    font-family: 'DM Serif Display', serif;
    font-size: 2rem; color: #1C2B4A; line-height: 1;
}
.dash-metric-lbl {
    font-size: 0.78rem; color: #555555; margin-top: 0.3rem;
    font-weight: 600; text-transform: uppercase; letter-spacing: 0.07em;
}

/* ── Instructor section header (legacy .sec) ── */
.sec {
    font-family: 'DM Serif Display', serif; font-size: 1.1rem;
    color: #1C2B4A; margin: 1.6rem 0 0.7rem 0;
    padding-bottom: 0.35rem; border-bottom: 2px solid #E0E0E0;
}

/* ── Mobile ── */
.mobile-warn {
    display: none; background: #FFFBEB; border: 1.5px solid #FDE68A;
    border-radius: 8px; padding: 0.9rem 1.1rem; font-size: 1rem;
    color: #92400E; margin-bottom: 1rem;
}
@media (max-width: 768px) {
    .mobile-warn { display: block; }
    .block-container { padding-left: 1rem !important; padding-right: 1rem !important; }
    .page-header { padding: 1.3rem 1.2rem; }
    .page-header-title { font-size: 1.5rem; }
    .hw-card, .hw-card-open, .hw-card-locked { padding: 1rem 1.1rem; }
    .q-body, .ans-section, .sol-section { padding: 1rem 1.2rem; }
}

/* ── Footer ── */
.page-footer {
    margin-top: 3rem;
    padding: 1.2rem 0 0.5rem 0;
    border-top: 1px solid #E0E0E0;
    font-size: 0.82rem;
    color: #888888;
    text-align: center;
    line-height: 1.8;
}
.page-footer a { color: #2563EB; text-decoration: none; }
.page-footer a:hover { text-decoration: underline; }
.page-footer-course {
    font-weight: 600;
    color: #555555;
    margin-bottom: 0.1rem;
}

/* ── Security indicator ── */
.secure-indicator {
    display: inline-flex;
    align-items: center;
    gap: 0.35rem;
    font-size: 0.78rem;
    color: #16A34A;
    font-weight: 500;
}

/* ── Typewriter cursor ── */
.typewriter-cursor {
    display: inline-block;
    width: 2px;
    height: 1em;
    background: currentColor;
    margin-left: 2px;
    vertical-align: text-bottom;
    animation: blink 0.8s step-end infinite;
}
@keyframes blink {
    0%, 100% { opacity: 1; }
    50%       { opacity: 0; }
}

/* ── FAQ / Materials ── */
.faq-a { font-size: 1rem; color: #1A1A1A; line-height: 1.75; }
.mat-row {
    display: flex; align-items: center; justify-content: space-between;
    padding: 0.9rem 1.1rem; background: #FFFFFF;
    border: 1.5px solid #E0E0E0; border-radius: 8px; margin-bottom: 0.45rem;
    transition: box-shadow 0.15s ease;
}
.mat-row:hover { box-shadow: 0 2px 10px rgba(0,0,0,0.07); }
.mat-title { font-size: 1rem; font-weight: 500; color: #1A1A1A; }
.mat-meta  { font-size: 0.88rem; color: #888888; margin-top: 0.1rem; }
</style>

<!-- Lucide icons CDN -->
<script src="https://unpkg.com/lucide@latest/dist/umd/lucide.js"></script>
"""

# ── JS helpers injected once per page ─────────────────────────────────────────
JS_HELPERS = """
<script>
// Dark mode toggle
function toggleDarkMode() {
    document.body.classList.toggle('dark-mode');
    const isDark = document.body.classList.contains('dark-mode');
    localStorage.setItem('ec224_dark', isDark ? '1' : '0');
}
// Restore dark mode preference
(function() {
    if (localStorage.getItem('ec224_dark') === '1') {
        document.body.classList.add('dark-mode');
    }
})();

// Animated score counter
function animateScore(elementId, target, duration) {
    const el = document.getElementById(elementId);
    if (!el) return;
    let start = 0;
    const step = target / (duration / 16);
    const timer = setInterval(function() {
        start += step;
        if (start >= target) { start = target; clearInterval(timer); }
        el.textContent = (Number.isInteger(target) ? Math.round(start) : start.toFixed(1));
    }, 16);
}

// Countdown timer
function startCountdown(elementId, deadlineISO) {
    const el = document.getElementById(elementId);
    if (!el) return;
    function update() {
        const diff = new Date(deadlineISO) - new Date();
        if (diff <= 0) { el.textContent = 'Closed'; el.closest('.countdown-wrap').classList.add('urgent'); return; }
        const h = Math.floor(diff / 3600000);
        const m = Math.floor((diff % 3600000) / 60000);
        const s = Math.floor((diff % 60000) / 1000);
        if (h < 2) el.closest('.countdown-wrap').classList.add('urgent');
        el.textContent = h > 0 ? h + 'h ' + m + 'm' : m + 'm ' + s + 's';
    }
    update();
    setInterval(update, 1000);
}

// Confetti
function fireConfetti() {
    if (typeof confetti === 'undefined') return;
    confetti({ particleCount: 120, spread: 80, origin: { y: 0.6 },
               colors: ['#1C2B4A','#2563EB','#2D6A4F','#52B788','#FDE68A'] });
}

// Typewriter
function typewriter(elementId, text, speed) {
    const el = document.getElementById(elementId);
    if (!el) return;
    let i = 0;
    el.textContent = '';
    const cursor = document.createElement('span');
    cursor.className = 'typewriter-cursor';
    el.appendChild(cursor);
    const timer = setInterval(function() {
        if (i < text.length) {
            el.insertBefore(document.createTextNode(text[i]), cursor);
            i++;
        } else {
            clearInterval(timer);
            setTimeout(function() { cursor.style.display = 'none'; }, 2000);
        }
    }, speed || 45);
}

// Lucide icons init
document.addEventListener('DOMContentLoaded', function() {
    if (typeof lucide !== 'undefined') lucide.createIcons();
});
</script>
<script src="https://cdn.jsdelivr.net/npm/canvas-confetti@1.9.2/dist/confetti.browser.min.js"></script>
"""


def inject_css():
    # PWA manifest, favicon, accessibility meta, service worker registration
    st.markdown(
        """
        <link rel="manifest" href="/app/static/manifest.json">
        <link rel="icon" type="image/png" href="/app/static/bentleylogo.png">
        <link rel="apple-touch-icon" href="/app/static/bentleylogo.png">
        <meta name="theme-color" content="#1C2B4A">
        <meta name="apple-mobile-web-app-capable" content="yes">
        <meta name="apple-mobile-web-app-status-bar-style" content="black-translucent">
        <meta name="apple-mobile-web-app-title" content="EC224">
        <meta name="description" content="Interactive Homework Portal — EC224 Intermediate Microeconomics, Bentley University">
        <meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">
        <script>
        if ('serviceWorker' in navigator) {
            window.addEventListener('load', function() {
                navigator.serviceWorker.register('/app/static/service_worker.js')
                    .catch(function() {});
            });
        }
        </script>
        """,
        unsafe_allow_html=True,
    )
    st.markdown(CSS + JS_HELPERS, unsafe_allow_html=True)


def page_header(eyebrow: str, title: str, sub: str = ""):
    sub_html = f'<div class="page-header-sub">{sub}</div>' if sub else ""
    st.markdown(
        f'<div class="page-header">'
        f'<div class="page-header-eye">{eyebrow}</div>'
        f'<div class="page-header-title">{title}</div>'
        f'{sub_html}</div>',
        unsafe_allow_html=True
    )


def section_header(label: str, icon: str = ""):
    """Replaces all inline <div class='sec'> usages."""
    icon_html = (
        f'<span class="sec-header-icon">{icon}</span>' if icon else ""
    )
    st.markdown(
        f'<div class="sec-header">{icon_html}{label}</div>',
        unsafe_allow_html=True
    )


def section_rule(label: str = ""):
    st.markdown(
        f'<div class="section-rule">{label}</div>',
        unsafe_allow_html=True
    )


def banner(text: str, kind: str = "info"):
    st.markdown(
        f'<div class="banner banner-{kind}">{text}</div>',
        unsafe_allow_html=True
    )


def feedback_banner(title: str, sub: str = "", kind: str = "success"):
    icons = {"success": "✓", "error": "✗", "partial": "~"}
    icon = icons.get(kind, "ℹ")
    sub_html = f'<div class="feedback-sub">{sub}</div>' if sub else ""
    st.markdown(
        f'<div class="feedback-banner feedback-{kind}">'
        f'<div class="feedback-icon">{icon}</div>'
        f'<div><div class="feedback-title">{title}</div>{sub_html}</div>'
        f'</div>',
        unsafe_allow_html=True
    )


def status_pill(label: str, kind: str):
    """kind: open | closed | upcoming | complete | graded"""
    st.markdown(
        f'<span class="pill pill-{kind}">{label}</span>',
        unsafe_allow_html=True
    )


def progress_bar(done: int, total: int):
    pct = int((done / total * 100)) if total else 0
    st.markdown(
        f'<div class="progress-container">'
        f'<div class="progress-label">'
        f'<span><strong>{done}</strong> of {total} questions submitted</span>'
        f'<span><strong>{pct}%</strong></span>'
        f'</div>'
        f'<div class="progress-bar-wrap">'
        f'<div class="progress-bar-fill" style="width:{pct}%"></div>'
        f'</div>'
        f'</div>',
        unsafe_allow_html=True
    )


def skeleton_card(n: int = 3):
    cards = "".join(
        ['<div class="skeleton skeleton-card"></div>'] * n
    )
    st.markdown(
        f'<div class="skeleton skeleton-title"></div>{cards}',
        unsafe_allow_html=True
    )


def breadcrumb(*items):
    """breadcrumb('Dashboard', 'Homework 2') — last item is active."""
    parts = []
    for i, item in enumerate(items):
        active = ' active' if i == len(items) - 1 else ''
        parts.append(f'<span class="breadcrumb-item{active}">{item}</span>')
        if i < len(items) - 1:
            parts.append('<span class="breadcrumb-sep">›</span>')
    st.markdown(
        f'<div class="breadcrumb">{"".join(parts)}</div>',
        unsafe_allow_html=True
    )


def sidebar_brand(student_name: str = ""):
    initial = student_name[0].upper() if student_name else "?"
    student_html = (
        f'<div class="sidebar-student">'
        f'<div class="sidebar-avatar">{initial}</div>'
        f'<div class="sidebar-name">{student_name}</div>'
        f'</div>'
    ) if student_name else ""
    st.sidebar.markdown(
        f'<div class="sidebar-brand">'
        f'<div class="sidebar-course">EC224</div>'
        f'<div class="sidebar-semester">Intermediate Microeconomics</div>'
        f'{student_html}'
        f'</div>',
        unsafe_allow_html=True
    )


def countdown_timer(hw_id: str, deadline_iso: str):
    """Renders a live countdown. deadline_iso: '2025-04-15T23:59:00'"""
    timer_id = f"cd_{hw_id}"
    st.markdown(
        f'<div class="countdown-wrap">'
        f'<span class="countdown-icon">⏱</span>'
        f'<span id="{timer_id}">...</span>'
        f'</div>'
        f'<script>startCountdown("{timer_id}", "{deadline_iso}");</script>',
        unsafe_allow_html=True
    )


def last_updated_chip(ts: str):
    st.markdown(
        f'<span class="last-updated">'
        f'<span class="last-updated-dot"></span>'
        f'Data refreshed {ts}'
        f'</span>',
        unsafe_allow_html=True
    )


def empty_state(icon: str, title: str, sub: str = ""):
    sub_html = f'<div class="empty-state-sub">{sub}</div>' if sub else ""
    st.markdown(
        f'<div class="empty-state">'
        f'<div class="empty-state-icon">{icon}</div>'
        f'<div class="empty-state-title">{title}</div>'
        f'{sub_html}'
        f'</div>',
        unsafe_allow_html=True
    )


def error_state(title: str, sub: str = ""):
    sub_html = f'<div class="error-state-sub">{sub}</div>' if sub else ""
    st.markdown(
        f'<div class="error-state">'
        f'<div class="error-state-title">⚠ {title}</div>'
        f'{sub_html}'
        f'</div>',
        unsafe_allow_html=True
    )


def registration_success(name: str):
    st.markdown(
        f'<div class="reg-success">'
        f'<div class="reg-success-icon">🎉</div>'
        f'<div class="reg-success-title">You\'re enrolled, {name}!</div>'
        f'<div class="reg-success-sub">'
        f'Welcome to EC224 · Intermediate Microeconomics.<br>'
        f'Head to your Dashboard to see your assignments.'
        f'</div>'
        f'</div>',
        unsafe_allow_html=True
    )


def secure_indicator():
    st.markdown(
        '<div class="secure-indicator">🔒 Your data is encrypted and secure</div>',
        unsafe_allow_html=True
    )


def page_footer(semester: str = "Spring 2025"):
    st.markdown(
        f'<div class="page-footer">'
        f'<div class="page-footer-course">EC224 · Intermediate Microeconomics</div>'
        f'Bentley University · Prof. Naveen Sunder · {semester}<br>'
        f'Questions? See the <a href="/FAQ">FAQ</a> or email '
        f'<a href="mailto:nsunder@bentley.edu">nsunder@bentley.edu</a>'
        f'</div>',
        unsafe_allow_html=True
    )


def dark_mode_toggle():
    st.markdown(
        '<button onclick="toggleDarkMode()" '
        'style="background:none;border:1px solid #E0E0E0;border-radius:6px;'
        'padding:4px 10px;font-size:0.82rem;cursor:pointer;color:#555555;">'
        '🌙 Dark mode'
        '</button>',
        unsafe_allow_html=True
    )


def q_divider(label: str):
    st.markdown(
        f'<div class="q-divider">'
        f'<div class="q-divider-line"></div>'
        f'<div class="q-divider-label">{label}</div>'
        f'<div class="q-divider-line"></div>'
        f'</div>',
        unsafe_allow_html=True
    )


def score_reveal(element_id: str, score: float, max_score: float):
    """Animated score counter on submission."""
    st.markdown(
        f'<div class="score-reveal-wrap">'
        f'<div class="score-reveal-num" id="{element_id}">0</div>'
        f'<div class="score-reveal-denom">out of {max_score:.0f} pts</div>'
        f'</div>'
        f'<script>animateScore("{element_id}", {score}, 800);</script>',
        unsafe_allow_html=True
    )


def confetti_if_perfect(score: float, max_score: float):
    if max_score > 0 and score >= max_score:
        st.markdown(
            '<script>fireConfetti();</script>',
            unsafe_allow_html=True
        )
