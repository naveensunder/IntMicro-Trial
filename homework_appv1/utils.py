"""
utils.py — shared helpers for the homework app.
Imported by both question pages.
"""

import streamlit as st
import datetime
import gspread
from google.oauth2.service_account import Credentials
import hashlib
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import io
import base64


# ── Google Sheets ──────────────────────────────────────────────────────────────
HEADER = [
    "Timestamp", "Student_Name", "School_Email",
    "Question_ID", "Status", "Is_Late",
    "Attempt_Reloads",
    "Param_Seed",
    # Q3 fields
    "Q3b_X_Int", "Q3b_X_Score",
    "Q3b_Y_Int", "Q3b_Y_Score",
    "Q3c_Slope",  "Q3c_Slope_Score",
    # Q9 fields
    "Q9a_Tom_X", "Q9a_Tom_Y", "Q9a_Tom_Score",
    "Q9b_Jerry_X", "Q9b_Jerry_Y", "Q9b_Jerry_Score",
    # Totals
    "Total_Score", "Max_Score",
]


def get_sheet():
    try:
        creds_dict = dict(st.secrets["gcp_service_account"])
        creds = Credentials.from_service_account_info(
            creds_dict,
            scopes=[
                "https://www.googleapis.com/auth/spreadsheets",
                "https://www.googleapis.com/auth/drive",
            ],
        )
        gc = gspread.authorize(creds)
        sh = gc.open_by_key(st.secrets["SHEET_ID"])
        return sh.sheet1
    except Exception as e:
        st.error(f"Sheet connection error: {e}")
        return None


def write_row(row_data):
    ws = get_sheet()
    if ws is None:
        return False, "Could not connect to sheet"
    try:
        # Ensure header exists
        try:
            existing = ws.cell(1, 1).value
        except Exception:
            existing = None
        if not existing:
            ws.update("A1", [HEADER])
        ws.append_row(row_data)
        return True, ""
    except Exception as e:
        return False, str(e)[:150]


def fetch_previous(school_email):
    ws = get_sheet()
    prev = {}
    if ws is None:
        return prev
    try:
        rows = ws.get_all_records()
        for row in rows:
            if str(row.get("School_Email", "")).strip().lower() == school_email.strip().lower():
                qid = str(row.get("Question_ID", ""))
                if qid:
                    prev[qid] = row
    except Exception:
        pass
    return prev


# ── Deadline ───────────────────────────────────────────────────────────────────
def check_deadline():
    deadline_str = st.secrets.get("DEADLINE", "2099-12-31 23:59")
    try:
        deadline = datetime.datetime.strptime(deadline_str, "%Y-%m-%d %H:%M")
        return datetime.datetime.now() > deadline, deadline
    except Exception:
        return False, None


def is_late_submission():
    past, _ = check_deadline()
    return past


# ── Parameter randomisation (seeded by email) ─────────────────────────────────
def get_seed(email: str) -> int:
    """Deterministic seed from email — same every time student returns."""
    return int(hashlib.md5(email.lower().encode()).hexdigest(), 16) % 100_000


def q3_params(email: str):
    """
    Returns (I, Px, Py) for Q3 — unique per student, always recoverable.
    Chosen so intercepts are clean integers.
    """
    rng = np.random.default_rng(get_seed(email))
    Px_choices = [2, 3, 4, 5]
    Py_choices = [2, 3, 4, 5, 6]
    I_choices  = [60, 80, 100, 120, 150, 200]

    Px = int(rng.choice(Px_choices))
    Py = int(rng.choice([p for p in Py_choices if p != Px]))
    # Pick I divisible by both Px and Py for clean intercepts
    valid_I = [i for i in I_choices if i % Px == 0 and i % Py == 0]
    if not valid_I:
        valid_I = [120]
    I = int(rng.choice(valid_I))
    return I, Px, Py


def q9_params(email: str):
    """
    Returns (I, Px, Py, tom_a, tom_b) for Q9.
    Tom: U = tom_a*X + tom_b*Y (perfect substitutes, corner solution)
    Jerry: U = min(X, Y) (perfect complements, kink)
    """
    rng = np.random.default_rng(get_seed(email) + 1)
    I_choices  = [60, 80, 90, 120]
    Px_choices = [2, 3, 4, 5]
    Py_choices = [3, 4, 6, 8]
    I  = int(rng.choice(I_choices))
    Px = int(rng.choice(Px_choices))
    Py = int(rng.choice([p for p in Py_choices if p != Px]))
    # Tom: make sure corner is at X (tom_a/Px > tom_b/Py)
    tom_a = int(rng.choice([2, 3]))
    tom_b = 1
    # Ensure tom_a/Px > tom_b/Py so corner is at X axis
    while not (tom_a / Px > tom_b / Py):
        Px = int(rng.choice(Px_choices))
        Py = int(rng.choice([p for p in Py_choices if p != Px]))
    return I, Px, Py, tom_a, tom_b


def q3_answers(I, Px, Py):
    return {
        "xint":  float(I / Px),
        "yint":  float(I / Py),
        "slope": float(-Px / Py),
    }


def q9_answers(I, Px, Py, tom_a, tom_b):
    # Tom corner: all on X
    tom_x = float(I / Px)
    tom_y = 0.0
    # Jerry kink: X=Y, Px*X + Py*X = I
    jerry_x = float(I / (Px + Py))
    jerry_y = jerry_x
    return {
        "tom_x": tom_x, "tom_y": tom_y,
        "jerry_x": jerry_x, "jerry_y": jerry_y,
    }


# ── Matplotlib figures ────────────────────────────────────────────────────────
def fig_to_b64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=130)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")


def make_q3_graph(I, Px, Py):
    plt.style.use("default")
    fig, ax = plt.subplots(figsize=(6, 5))
    fig.patch.set_facecolor("#f8faff")
    ax.set_facecolor("#f8faff")
    Xmax = I / Px; Ymax = I / Py
    Xv = np.linspace(0, Xmax, 300)
    Yv = (I - Px * Xv) / Py
    ax.plot(Xv, Yv, color="#1a3a6b", lw=2.8, label=f"${Px}X + {Py}Y = {I}$")
    ax.fill_between(Xv, Yv, alpha=0.10, color="#1a3a6b", label="Budget set")
    ax.plot(Xmax, 0, "o", color="#1a3a6b", ms=9, zorder=5)
    ax.plot(0, Ymax, "o", color="#1a3a6b", ms=9, zorder=5)
    ax.annotate(
        f"({int(Xmax)}, 0)",
        xy=(Xmax, 0), xytext=(Xmax - Xmax*0.2, Ymax*0.08),
        fontsize=9, color="#1a3a6b",
        arrowprops=dict(arrowstyle="->", color="#1a3a6b", lw=1.2),
    )
    ax.annotate(
        f"(0, {int(Ymax)})",
        xy=(0, Ymax), xytext=(Xmax*0.08, Ymax - Ymax*0.12),
        fontsize=9, color="#1a3a6b",
        arrowprops=dict(arrowstyle="->", color="#1a3a6b", lw=1.2),
    )
    ax.text(
        Xmax * 0.45, Ymax * 0.52,
        f"Slope = $-{Px}/{Py}$ = {round(-Px/Py, 4)}",
        fontsize=9, color="#1a3a6b",
        bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="#1a3a6b", alpha=0.9),
    )
    ax.set_xlabel("Quantity of $X$", fontsize=11)
    ax.set_ylabel("Quantity of $Y$", fontsize=11)
    ax.set_title("Q3(d) — Budget Line  *(practice — not graded)*", fontsize=10)
    ax.set_xlim(0, Xmax * 1.18); ax.set_ylim(0, Ymax * 1.22)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    ax.legend(loc="upper right", fontsize=9, framealpha=0.9)
    plt.tight_layout()
    return fig


def make_q9_graph(I, Px, Py, tom_a, tom_b):
    plt.style.use("default")
    fig, axs = plt.subplots(1, 2, figsize=(13, 5.5))
    fig.patch.set_facecolor("#f8faff")
    fig.suptitle(
        f"Q9 — Tom (Corner) vs Jerry (Kink)   [I={I}, $P_x$={Px}, $P_y$={Py}]",
        fontsize=12, fontweight="bold", color="#0f2044",
    )
    Xt = I / Px; Yt = 0.0
    Xj = I / (Px + Py); Yj = Xj
    Xv = np.linspace(0, I / Px + 2, 400)
    Yv = (I - Px * Xv) / Py

    # Tom
    ax = axs[0]; ax.set_facecolor("#f8faff")
    ax.plot(Xv, np.where(Yv >= 0, Yv, np.nan), color="#1a3a6b", lw=2.5, label="Budget line")
    UT = tom_a * Xt + tom_b * Yt
    for Ul, alp in [(UT * 0.65, 0.20), (UT * 0.82, 0.35), (UT, 0.90)]:
        ICs = (Ul - tom_a * Xv) / tom_b
        ax.plot(Xv, np.where(ICs >= 0, ICs, np.nan), color="#c62828", lw=1.6, alpha=alp)
    ax.plot(Xt, Yt, "o", color="#c62828", ms=11, zorder=6,
            label=f"Optimum ({int(Xt)}, {int(Yt)})")
    ax.annotate(
        f"Corner: ({int(Xt)}, {int(Yt)})",
        xy=(Xt, Yt), xytext=(Xt - Xt * 0.35, Yt + (I/Py) * 0.12),
        fontsize=9, color="#c62828",
        arrowprops=dict(arrowstyle="->", color="#c62828"),
    )
    ax.set_title(f"Tom: $U_T = {tom_a}X + {tom_b}Y$  (Perfect Substitutes)", fontsize=10)
    ax.set_xlabel("$X$"); ax.set_ylabel("$Y$")
    ax.set_xlim(0, I/Px * 1.2); ax.set_ylim(0, I/Py * 1.3)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    ax.legend(fontsize=9, framealpha=0.9)

    # Jerry
    ax = axs[1]; ax.set_facecolor("#f8faff")
    ax.plot(Xv, np.where(Yv >= 0, Yv, np.nan), color="#1a3a6b", lw=2.5, label="Budget line")
    for Ul, alp in [(Xj * 0.65, 0.20), (Xj * 0.82, 0.35), (Xj, 0.90)]:
        xlim = I / Px * 1.2; ylim = I / Py * 1.3
        ax.plot([Ul, xlim], [Ul, Ul], color="#c62828", lw=1.6, alpha=alp)
        ax.plot([Ul, Ul], [Ul, ylim], color="#c62828", lw=1.6, alpha=alp)
    diag = np.linspace(0, min(I/Px, I/Py) * 1.1, 100)
    ax.plot(diag, diag, "k--", lw=1.2, alpha=0.35, label="Kink ray $X=Y$")
    ax.plot(Xj, Yj, "o", color="#c62828", ms=11, zorder=6,
            label=f"Optimum ({round(Xj, 2)}, {round(Yj, 2)})")
    ax.annotate(
        f"Kink: ({round(Xj,2)}, {round(Yj,2)})",
        xy=(Xj, Yj), xytext=(Xj + Xj * 0.25, Yj - (I/Py) * 0.15),
        fontsize=9, color="#c62828",
        arrowprops=dict(arrowstyle="->", color="#c62828"),
    )
    ax.set_title("Jerry: $U_J = \\min(X,Y)$  (Perfect Complements)", fontsize=10)
    ax.set_xlabel("$X$"); ax.set_ylabel("$Y$")
    ax.set_xlim(0, I/Px * 1.2); ax.set_ylim(0, I/Py * 1.3)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    ax.legend(fontsize=9, framealpha=0.9)

    plt.tight_layout()
    return fig


# ── Shared CSS injected on every question page ────────────────────────────────
SHARED_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@400;600;700&family=Source+Sans+3:wght@300;400;500;600&display=swap');
html, body, [class*="css"] { font-family: 'Source Sans 3', sans-serif; }
#MainMenu {visibility: hidden;} footer {visibility: hidden;}
.stDeployButton {display: none;} header {visibility: hidden;}
.block-container { padding-top: 1.5rem; padding-bottom: 2rem; max-width: 820px; }

.q-header {
    background: linear-gradient(135deg, #0f2044 0%, #1a3a6b 60%, #1e4d8c 100%);
    border-radius: 14px; padding: 1.4rem 1.8rem; margin-bottom: 1.5rem;
    box-shadow: 0 6px 24px rgba(15,32,68,0.16);
    display: flex; justify-content: space-between; align-items: center;
}
.q-header-title {
    font-family: 'Playfair Display', serif; color: #fff;
    font-size: 1.3rem; font-weight: 700;
}
.q-header-sub { color: #a8c4e8; font-size: 0.85rem; margin-top: 0.2rem; }
.badge-graded {
    background: #f9a825; color: #2c1800;
    border-radius: 20px; padding: 4px 14px;
    font-size: 0.8rem; font-weight: 700; white-space: nowrap;
}

.q-body {
    background: #f0f4ff; border: 1px solid #c5d3f0;
    border-radius: 12px; padding: 1.2rem 1.6rem; margin-bottom: 1rem;
    font-size: 0.95rem; line-height: 1.8; color: #1a2340;
}
.part-row {
    display: flex; align-items: flex-start; gap: 12px;
    background: #fff; border: 1px solid #dde4f5; border-radius: 8px;
    padding: 0.75rem 1rem; margin: 0.5rem 0;
}
.part-badge {
    background: #1a3a6b; color: white; border-radius: 6px;
    padding: 3px 10px; font-size: 0.78rem; font-weight: 700;
    white-space: nowrap; flex-shrink: 0; margin-top: 2px;
}
.part-text { flex: 1; font-size: 0.92rem; line-height: 1.6; }
.chip-ungraded {
    background: #78909c; color: white; border-radius: 10px;
    padding: 2px 10px; font-size: 0.75rem; white-space: nowrap; flex-shrink: 0;
}

.answer-box {
    background: #fff; border: 1.5px solid #c5d3f0; border-radius: 10px;
    padding: 1rem 1.4rem; margin-bottom: 1rem;
}
.answer-label {
    font-family: 'Playfair Display', serif; font-weight: 600;
    color: #0f2044; font-size: 1rem; margin-bottom: 0.6rem;
}

.sol-card {
    background: #fff; border: 1px solid #dde4f5; border-radius: 12px;
    overflow: hidden; margin-top: 1rem;
}
.sol-header {
    background: #1a3a6b; color: #fff;
    padding: 0.7rem 1.2rem; font-family: 'Playfair Display', serif;
    font-size: 1rem; font-weight: 700;
}
.sol-steps {
    background: #e3f2fd; border-left: 4px solid #1565c0;
    border-radius: 8px; padding: 1rem 1.2rem; margin: 0.8rem 0;
    font-size: 0.9rem; line-height: 1.8;
}
.sol-mistakes {
    background: #fce4ec; border-left: 4px solid #c62828;
    border-radius: 8px; padding: 1rem 1.2rem; margin: 0.8rem 0;
    font-size: 0.9rem; line-height: 1.8;
}
.sol-graph {
    background: #e0f2f1; border-left: 4px solid #00695c;
    border-radius: 8px; padding: 1rem 1.2rem; margin: 0.8rem 0;
}
.sol-steps h4, .sol-mistakes h4, .sol-graph h4 {
    margin: 0 0 0.5rem 0; font-size: 0.92rem; font-weight: 700;
}

.score-table {
    width: 100%; border-collapse: collapse; font-size: 0.88rem; margin-top: 0.6rem;
}
.score-table th {
    background: #bbdefb; padding: 6px 10px;
    text-align: left; font-weight: 600; color: #0d47a1;
}
.score-table td { padding: 6px 10px; border-bottom: 1px solid #e8edf8; }
.score-table tr:last-child td { background: #e3f2fd; font-weight: 700; }
.chip-ok {
    background: #d4edda; color: #155724; border-radius: 10px;
    padding: 2px 10px; font-size: 0.82rem; font-weight: 700;
}
.chip-wrong {
    background: #f8d7da; color: #721c24; border-radius: 10px;
    padding: 2px 10px; font-size: 0.82rem; font-weight: 700;
}

.confirm-banner {
    background: #d4edda; border: 1.5px solid #28a745; border-radius: 10px;
    padding: 1rem 1.3rem; margin: 0.8rem 0; font-size: 0.9rem; color: #155724;
}
.warn-banner {
    background: #fff3cd; border: 1.5px solid #ffc107; border-radius: 10px;
    padding: 1rem 1.3rem; margin: 0.8rem 0; font-size: 0.9rem; color: #7a5800;
}
.restore-banner {
    background: #e3f2fd; border: 1.5px solid #90caf9; border-radius: 10px;
    padding: 0.8rem 1.2rem; margin: 0.8rem 0; font-size: 0.88rem; color: #0d47a1;
}
.locked-banner {
    background: #f3e5f5; border: 1.5px solid #ab47bc; border-radius: 10px;
    padding: 0.8rem 1.2rem; margin: 0.8rem 0; font-size: 0.88rem; color: #4a148c;
}
.deadline-closed {
    background: #fdecea; border: 1px solid #f5a8a0; border-radius: 10px;
    padding: 0.8rem 1.2rem; color: #8b1a1a; font-size: 0.9rem; margin-bottom: 1rem;
}
.param-chip {
    display: inline-block; background: #e8edf8; border: 1px solid #c5d3f0;
    border-radius: 8px; padding: 4px 12px; font-size: 0.85rem;
    color: #2c3e6b; margin: 2px 4px 2px 0; font-family: monospace;
}
.stButton > button {
    background: linear-gradient(135deg, #1a3a6b, #2a5ca8) !important;
    color: white !important; border: none !important;
    border-radius: 8px !important; font-family: 'Source Sans 3', sans-serif !important;
    font-weight: 600 !important; font-size: 0.95rem !important;
    transition: all 0.2s ease !important;
    box-shadow: 0 3px 12px rgba(26,58,107,0.25) !important;
}
.stButton > button:hover {
    background: linear-gradient(135deg, #0f2044, #1a3a6b) !important;
    transform: translateY(-1px) !important;
}
.stNumberInput > div > div > input {
    border: 1.5px solid #c5d3f0 !important; border-radius: 8px !important;
    font-family: 'Source Sans 3', sans-serif !important;
}
.stNumberInput > div > div > input:focus {
    border-color: #1a3a6b !important;
    box-shadow: 0 0 0 3px rgba(26,58,107,0.1) !important;
}
</style>
"""


def guard_page():
    """Check student is confirmed; redirect message if not."""
    if not st.session_state.get("confirmed"):
        st.markdown(SHARED_CSS, unsafe_allow_html=True)
        st.warning("Please go to the **Home** page first and enter your details.")
        st.stop()


def increment_reload(q_key: str):
    """Track how many times this question page has been loaded."""
    k = f"reloads_{q_key}"
    st.session_state[k] = st.session_state.get(k, 0) + 1
    return st.session_state[k]
