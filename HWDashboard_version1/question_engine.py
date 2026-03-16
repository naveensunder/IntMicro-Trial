"""
question_engine.py — Modular question renderer.
HWDashboard v2 — Phase 1
Fixes: _render_truefalse submissions lookup bug.
New: topics-to-revise, worked example unlock timer, dual timers.
"""

import streamlit as st
import datetime
import numpy as np
import hashlib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import io, base64
from db import write_submission, log_audit


# ── Seed helper ────────────────────────────────────────────────────────────────
def get_seed(email: str) -> int:
    return int(hashlib.md5(email.lower().encode()).hexdigest(), 16) % 100_000


# ── Figure helper ──────────────────────────────────────────────────────────────
def fig_to_b64(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=130)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")


# ════════════════════════════════════════════════════════════════════════════════
#  PARAMETER GENERATORS
# ════════════════════════════════════════════════════════════════════════════════

def _q3_params(email: str):
    rng     = np.random.default_rng(get_seed(email))
    Px_opts = [2, 3, 4, 5]
    Py_opts = [2, 3, 4, 5, 6]
    I_opts  = [60, 80, 100, 120, 150, 200]
    Px      = int(rng.choice(Px_opts))
    Py      = int(rng.choice([p for p in Py_opts if p != Px]))
    valid_I = [i for i in I_opts if i % Px == 0 and i % Py == 0]
    I       = int(rng.choice(valid_I if valid_I else [120]))
    return I, Px, Py


def _q9_params(email: str):
    rng     = np.random.default_rng(get_seed(email) + 1)
    I_opts  = [60, 80, 90, 120]
    Px_opts = [2, 3, 4, 5]
    Py_opts = [3, 4, 6, 8]
    I       = int(rng.choice(I_opts))
    Px      = int(rng.choice(Px_opts))
    Py      = int(rng.choice([p for p in Py_opts if p != Px]))
    tom_a   = int(rng.choice([2, 3]))
    tom_b   = 1
    attempts = 0
    while not (tom_a / Px > tom_b / Py) and attempts < 50:
        Px = int(rng.choice(Px_opts))
        Py = int(rng.choice([p for p in Py_opts if p != Px]))
        attempts += 1
    return I, Px, Py, tom_a, tom_b


# ════════════════════════════════════════════════════════════════════════════════
#  TRUE/FALSE STATEMENTS
# ════════════════════════════════════════════════════════════════════════════════

TF_STATEMENTS = [
    {
        "text": "A rise in income causes the budget line to pivot around one intercept.",
        "correct": False,
        "explanation": "A rise in income causes a <strong>parallel outward shift</strong> — both intercepts move proportionally. A pivot occurs only when a <strong>price</strong> changes.",
        "topic": "Budget constraints and income changes"
    },
    {
        "text": "For a Cobb-Douglas utility function $U = X^{0.4}Y^{0.6}$, the consumer always spends 40% of income on $X$ regardless of prices.",
        "correct": True,
        "explanation": "Correct. For Cobb-Douglas $U = X^\\alpha Y^\\beta$, the optimal spending shares are always $\\alpha$ on $X$ and $\\beta$ on $Y$, independent of prices.",
        "topic": "Cobb-Douglas utility and optimal spending shares"
    },
    {
        "text": "Perfect complements have downward-sloping indifference curves.",
        "correct": False,
        "explanation": "Perfect complements have <strong>L-shaped (kinked)</strong> indifference curves, not downward-sloping smooth ones.",
        "topic": "Perfect complements and indifference curve shapes"
    },
    {
        "text": "For a consumer facing perfect substitutes, the optimal solution is always a corner solution.",
        "correct": True,
        "explanation": "Correct — unless MRS exactly equals the price ratio (a knife-edge case), the consumer spends all income on whichever good delivers more utility per dollar.",
        "topic": "Perfect substitutes and corner solutions"
    },
]


# ════════════════════════════════════════════════════════════════════════════════
#  HOMEWORK CONFIGS
# ════════════════════════════════════════════════════════════════════════════════

HW_WEEK2_CONFIG = {
    "hw_id": "HW_WEEK2",
    "questions": [
        {"q_id": "Q3",  "type": "numerical",  "title": "Q3 — Budget Constraint",              "marks": 6},
        {"q_id": "Q9",  "type": "numerical",  "title": "Q9 — Tom & Jerry: Corner vs Kink",    "marks": 8},
        {"q_id": "QTF", "type": "truefalse",  "title": "Q10 — True or False",                 "marks": 4},
    ]
}

ALL_HW_CONFIGS = {
    "HW_WEEK2": HW_WEEK2_CONFIG,
}


def get_shuffled_questions(hw_id: str, email: str) -> list:
    config    = ALL_HW_CONFIGS.get(hw_id, {})
    questions = config.get("questions", [])
    if not questions:
        return []
    rng     = np.random.default_rng(get_seed(email) + 99)
    indices = list(range(len(questions)))
    rng.shuffle(indices)
    return [questions[i] for i in indices]


def get_hw_summary(hw_id: str, email: str, submissions: dict) -> dict:
    hw_subs     = submissions.get(hw_id, {})
    total_score = 0
    total_max   = 0
    all_done    = True
    config      = ALL_HW_CONFIGS.get(hw_id, {})
    for q in config.get("questions", []):
        q_id  = q["q_id"]
        max_sc = q["marks"]
        sub   = hw_subs.get(q_id, {})
        if str(sub.get("Status","")) == "submitted":
            try:
                total_score += int(sub.get("Score", 0))
            except Exception:
                pass
            total_max += max_sc
        else:
            all_done  = False
            total_max += max_sc
    n_submitted = sum(
        1 for q in config.get("questions", [])
        if str(hw_subs.get(q["q_id"], {}).get("Status","")) == "submitted"
    )
    return {
        "total_score":  total_score,
        "total_max":    total_max,
        "all_done":     all_done,
        "n_submitted":  n_submitted,
        "n_total":      len(config.get("questions", [])),
    }


# ════════════════════════════════════════════════════════════════════════════════
#  SHARED UI COMPONENTS
# ════════════════════════════════════════════════════════════════════════════════

def _render_timers(q_id: str, deadline_str: str, grace_minutes: int) -> float:
    """Returns elapsed seconds on this question."""
    key = f"q_start_{q_id}"
    if key not in st.session_state:
        st.session_state[key] = datetime.datetime.now()

    elapsed   = datetime.datetime.now() - st.session_state[key]
    elapsed_s = elapsed.total_seconds()
    el_m      = int(elapsed_s // 60)
    el_s      = int(elapsed_s % 60)

    # Deadline countdown
    try:
        dl       = datetime.datetime.strptime(deadline_str.strip(), "%Y-%m-%d %H:%M")
        dl_grace = dl + datetime.timedelta(minutes=grace_minutes)
        rem      = dl_grace - datetime.datetime.now()
        if rem.total_seconds() > 0:
            tot_s  = int(rem.total_seconds())
            d      = tot_s // 86400
            h      = (tot_s % 86400) // 3600
            m      = (tot_s % 3600) // 60
            urgent = d == 0 and h < 6
            if d > 0:
                dl_str = f"{d}d {h}h {m}m"
            elif h > 0:
                dl_str = f"{h}h {m}m"
            else:
                dl_str = f"{m}m"
            dl_cls = "timer-warn" if urgent else ""
        else:
            dl_str = "Closed"
            dl_cls = "timer-warn"
    except Exception:
        dl_str = "—"
        dl_cls = ""

    st.markdown(f"""
    <div class="timer-bar">
      <div class="timer-item">
        <span>⏱ Time on question:</span>
        <span class="timer-value">{el_m:02d}:{el_s:02d}</span>
      </div>
      <div class="timer-item" style="margin-left:auto;">
        <span>📅 Time to deadline:</span>
        <span class="timer-value {dl_cls}">{dl_str}</span>
      </div>
    </div>
    """, unsafe_allow_html=True)

    return elapsed_s


def _worked_example_section(q_id: str, elapsed_s: float,
                             example_html: str, unlock_after: int = 1200):
    """Worked example — locked for first 20 minutes."""
    if elapsed_s >= unlock_after:
        with st.expander("📖 Show worked example (similar problem)"):
            st.markdown(example_html, unsafe_allow_html=True)
    else:
        mins_left = max(1, int((unlock_after - elapsed_s) // 60) + 1)
        st.markdown(f"""
        <div style="background:#F9FAFB;border:1px solid #E5E7EB;border-radius:7px;
             padding:0.7rem 1rem;margin-bottom:0.7rem;font-size:0.84rem;color:#6B7280;">
          📖 Worked example unlocks in <strong>{mins_left} minute{'s' if mins_left != 1 else ''}</strong>.
          Attempt the question first.
        </div>
        """, unsafe_allow_html=True)


def _flag_section(q_id: str, hw_id: str, email: str):
    flag_key = f"flagged_{hw_id}_{q_id}"
    if not st.session_state.get(flag_key):
        if st.button("🚩 Flag this question for review",
                     key=f"flag_{q_id}_{hw_id}"):
            log_audit(email, "FLAG_QUESTION", f"{hw_id}/{q_id}")
            st.session_state[flag_key] = True
            st.success("Flagged. Your instructor will be notified.")
    else:
        st.caption("✓ This question has been flagged.")


def _unsaved_warning(q_id: str, has_input: bool, submitted: bool):
    """Show unsaved warning if student has typed but not submitted."""
    if has_input and not submitted:
        st.markdown(
            '<div class="unsaved-warning">⚠ You have unsaved answers — '
            'do not navigate away without submitting.</div>',
            unsafe_allow_html=True
        )


# ════════════════════════════════════════════════════════════════════════════════
#  MASTER DISPATCHER
# ════════════════════════════════════════════════════════════════════════════════

def render_question(q_config: dict, hw_id: str, email: str, name: str,
                    past_deadline: bool, grace_active: bool,
                    submissions: dict, review_mode: bool = False):
    q_type = q_config.get("type", "")
    if q_type == "numerical":
        q_id = q_config.get("q_id", "")
        if q_id == "Q3":
            _render_q3(q_config, hw_id, email, name,
                       past_deadline, grace_active, submissions, review_mode)
        elif q_id == "Q9":
            _render_q9(q_config, hw_id, email, name,
                       past_deadline, grace_active, submissions, review_mode)
    elif q_type == "truefalse":
        _render_truefalse(q_config, hw_id, email, name,
                          past_deadline, grace_active, submissions, review_mode)


# ════════════════════════════════════════════════════════════════════════════════
#  MATPLOTLIB GRAPHS
# ════════════════════════════════════════════════════════════════════════════════

def _make_budget_graph(I, Px, Py):
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    fig.patch.set_facecolor("#FAFAFA")
    ax.set_facecolor("#FAFAFA")
    Xmax = I / Px; Ymax = I / Py
    Xv   = np.linspace(0, Xmax, 300)
    Yv   = (I - Px * Xv) / Py
    ax.plot(Xv, Yv, color="#1C2B4A", lw=2.5,
            label=f"${Px}X + {Py}Y = {I}$")
    ax.fill_between(Xv, Yv, alpha=0.07, color="#1C2B4A")
    ax.plot(Xmax, 0, "o", color="#1C2B4A", ms=7, zorder=5)
    ax.plot(0, Ymax, "o", color="#1C2B4A", ms=7, zorder=5)
    ax.annotate(f"({int(Xmax)}, 0)", xy=(Xmax, 0),
                xytext=(Xmax - Xmax*0.25, Ymax*0.09),
                fontsize=9, color="#1C2B4A",
                arrowprops=dict(arrowstyle="->", color="#1C2B4A", lw=1.0))
    ax.annotate(f"(0, {int(Ymax)})", xy=(0, Ymax),
                xytext=(Xmax*0.08, Ymax - Ymax*0.14),
                fontsize=9, color="#1C2B4A",
                arrowprops=dict(arrowstyle="->", color="#1C2B4A", lw=1.0))
    ax.text(Xmax*0.42, Ymax*0.5,
            f"Slope = $-{Px}/{Py}$ = {round(-Px/Py, 4)}",
            fontsize=9, color="#1C2B4A",
            bbox=dict(boxstyle="round,pad=0.3", fc="white",
                      ec="#D1D5DB", alpha=0.95))
    ax.set_xlabel("Quantity of $X$", fontsize=10)
    ax.set_ylabel("Quantity of $Y$", fontsize=10)
    ax.set_title("Budget Line — Reference Diagram", fontsize=10, color="#374151")
    ax.set_xlim(0, Xmax * 1.2); ax.set_ylim(0, Ymax * 1.25)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    ax.tick_params(labelsize=9)
    plt.tight_layout()
    return fig


def _make_q9_graph(I, Px, Py, tom_a, tom_b):
    fig, axs = plt.subplots(1, 2, figsize=(12, 5))
    fig.patch.set_facecolor("#FAFAFA")
    fig.suptitle(
        f"Reference: Tom vs Jerry  [I={I}, $P_x$={Px}, $P_y$={Py}]",
        fontsize=10, color="#1C2B4A", fontweight="600"
    )
    Xt = I/Px; Yt = 0.0; Xj = I/(Px+Py); Yj = Xj
    Xv = np.linspace(0, I/Px + 2, 400); Yv = (I - Px*Xv)/Py

    for ax in axs:
        ax.set_facecolor("#FAFAFA")
        ax.plot(Xv, np.where(Yv >= 0, Yv, np.nan),
                color="#1C2B4A", lw=2.2, label="Budget line")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(labelsize=9)
        ax.set_xlabel("$X$", fontsize=10)
        ax.set_ylabel("$Y$", fontsize=10)

    ax = axs[0]
    UT = tom_a * Xt + tom_b * Yt
    for Ul, alp in [(UT*0.6, 0.18), (UT*0.8, 0.32), (UT, 0.85)]:
        ICs = (Ul - tom_a*Xv)/tom_b
        ax.plot(Xv, np.where(ICs >= 0, ICs, np.nan),
                color="#DC2626", lw=1.5, alpha=alp)
    ax.plot(Xt, Yt, "o", color="#DC2626", ms=9, zorder=6,
            label=f"Optimum ({int(Xt)}, {int(Yt)})")
    ax.annotate(f"({int(Xt)}, {int(Yt)})", xy=(Xt, Yt),
                xytext=(Xt - Xt*0.38, Yt + (I/Py)*0.12),
                fontsize=9, color="#DC2626",
                arrowprops=dict(arrowstyle="->", color="#DC2626", lw=1.0))
    ax.set_title(f"Tom: $U = {tom_a}X + {tom_b}Y$ (Perfect Substitutes)",
                 fontsize=10, color="#374151")
    ax.set_xlim(0, I/Px*1.2); ax.set_ylim(0, I/Py*1.3)
    ax.legend(fontsize=9)

    ax = axs[1]
    for Ul, alp in [(Xj*0.6, 0.18), (Xj*0.8, 0.32), (Xj, 0.85)]:
        xlim = I/Px*1.2; ylim = I/Py*1.3
        ax.plot([Ul, xlim], [Ul, Ul], color="#DC2626", lw=1.5, alpha=alp)
        ax.plot([Ul, Ul], [Ul, ylim], color="#DC2626", lw=1.5, alpha=alp)
    diag = np.linspace(0, min(I/Px, I/Py)*1.1, 100)
    ax.plot(diag, diag, "--", color="#9CA3AF", lw=1.1, alpha=0.6,
            label="Ray $X=Y$")
    ax.plot(Xj, Yj, "o", color="#DC2626", ms=9, zorder=6,
            label=f"Optimum ({round(Xj,2)}, {round(Yj,2)})")
    ax.annotate(f"({round(Xj,2)}, {round(Yj,2)})", xy=(Xj, Yj),
                xytext=(Xj + Xj*0.25, Yj - (I/Py)*0.14),
                fontsize=9, color="#DC2626",
                arrowprops=dict(arrowstyle="->", color="#DC2626", lw=1.0))
    ax.set_title("Jerry: $U = \\min(X,Y)$ (Perfect Complements)",
                 fontsize=10, color="#374151")
    ax.set_xlim(0, I/Px*1.2); ax.set_ylim(0, I/Py*1.3)
    ax.legend(fontsize=9)

    plt.tight_layout()
    return fig


# ════════════════════════════════════════════════════════════════════════════════
#  Q3 — BUDGET CONSTRAINT
# ════════════════════════════════════════════════════════════════════════════════

def _render_q3(q_config, hw_id, email, name, past_deadline,
               grace_active, submissions, review_mode):
    I, Px, Py = _q3_params(email)
    ANS = {
        "xint":  float(I / Px),
        "yint":  float(I / Py),
        "slope": float(-Px / Py),
    }

    # Get previous submission safely
    prev = {}
    hw_subs = submissions.get(hw_id, {})
    if isinstance(hw_subs, dict):
        prev = hw_subs.get("Q3", {})
    if not isinstance(prev, dict):
        prev = {}

    already_submitted = str(prev.get("Status", "")) == "submitted"
    disabled = already_submitted or (past_deadline and not grace_active) or review_mode

    # ── Question block ─────────────────────────────────────────────────────────
    st.markdown(f"""
    <div id="Q3"></div>
    <div class="q-header">
      <div>
        <div class="q-header-title">Q3 — Budget Constraint</div>
        <div class="q-header-sub">6 points</div>
      </div>
      <span class="badge badge-graded">GRADED</span>
    </div>
    <div class="q-body">
      <p>A consumer has income <strong>I&nbsp;=&nbsp;${I}</strong>,
         <strong>P<sub>x</sub>&nbsp;=&nbsp;${Px}</strong>,
         <strong>P<sub>y</sub>&nbsp;=&nbsp;${Py}</strong>.</p>
      <div class="part-row">
        <span class="part-badge">(a)</span>
        <span class="part-text">Write the equation of the budget constraint.</span>
        <span class="part-ungraded">Ungraded</span>
      </div>
      <div class="part-row">
        <span class="part-badge">(b) 4 pts</span>
        <span class="part-text">Find the <strong>X-intercept</strong> and <strong>Y-intercept</strong>.</span>
      </div>
      <div class="part-row">
        <span class="part-badge">(c) 2 pts</span>
        <span class="part-text">What is the <strong>slope</strong> of the budget line?</span>
      </div>
      <div class="part-row">
        <span class="part-badge">(d)</span>
        <span class="part-text">Draw the budget line on a labelled diagram.</span>
        <span class="part-ungraded">See reference graph after submitting</span>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # Parameters
    st.markdown(f"""
    <div style="margin-bottom:0.7rem;">
      <span style="font-size:0.68rem;font-weight:600;letter-spacing:0.1em;
       text-transform:uppercase;color:#6B7280;">Your parameters</span><br>
      <span class="param-chip">I = ${I}</span>
      <span class="param-chip">P<sub>x</sub> = ${Px}</span>
      <span class="param-chip">P<sub>y</sub> = ${Py}</span>
    </div>
    """, unsafe_allow_html=True)

    # Timers (not in review mode)
    elapsed_s = 0.0
    if not review_mode:
        hw_cfg        = _get_hw_cfg(hw_id)
        deadline_str  = hw_cfg.get("Deadline", "2099-12-31 23:59")
        grace_minutes = int(hw_cfg.get("Grace_Minutes", 15))
        elapsed_s     = _render_timers("Q3", deadline_str, grace_minutes)

    # Status banners
    if review_mode:
        st.markdown('<div class="banner-review">📖 Review mode — answers are read-only.</div>',
                    unsafe_allow_html=True)
    elif already_submitted:
        ts_prev   = prev.get("Timestamp", "")
        sc_prev   = prev.get("Score", "?")
        late_note = " · Late submission" if prev.get("Is_Late") == "Yes" else ""
        st.markdown(
            f'<div class="banner-locked">🔒 Submitted — Score: <strong>{sc_prev} / 6</strong>'
            f' · {ts_prev}{late_note}</div>',
            unsafe_allow_html=True
        )
    elif prev:
        st.markdown('<div class="banner-restore">Draft restored from previous session.</div>',
                    unsafe_allow_html=True)
    elif past_deadline and not grace_active:
        st.markdown('<div class="banner-error">🔒 Deadline has passed.</div>',
                    unsafe_allow_html=True)

    # Worked example
    if not review_mode:
        _worked_example_section("Q3", elapsed_s, _q3_worked_example(I, Px, Py))

    # ── Recover saved values ──────────────────────────────────────────────────
    def _pv(key, default=0.0):
        # Try raw_answer dict first
        raw = prev.get("Raw_Answer", "")
        if raw:
            try:
                d = eval(str(raw))
                if key in d:
                    return float(d[key])
            except Exception:
                pass
        v = prev.get(key)
        if v not in (None, ""):
            try:
                return float(v)
            except Exception:
                pass
        return default

    default_x = _pv("xint", 0.0)
    default_y = _pv("yint", 0.0)
    default_s = _pv("slope", 0.0)

    # ── Answer inputs ─────────────────────────────────────────────────────────
    st.markdown('<div class="answer-area"><div class="answer-label">Your Answers</div>',
                unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    with c1:
        xint_ans = st.number_input(
            "(b) X-intercept", value=float(default_x),
            step=0.1, format="%.4f",
            disabled=disabled, key=f"q3_x_{hw_id}"
        )
    with c2:
        yint_ans = st.number_input(
            "(b) Y-intercept", value=float(default_y),
            step=0.1, format="%.4f",
            disabled=disabled, key=f"q3_y_{hw_id}"
        )
    with c3:
        slope_ans = st.number_input(
            "(c) Slope", value=float(default_s),
            step=0.01, format="%.4f",
            disabled=disabled, key=f"q3_s_{hw_id}"
        )
    st.markdown('</div>', unsafe_allow_html=True)

    # Validation
    if not disabled:
        if xint_ans < 0 or yint_ans < 0:
            st.markdown(
                '<div class="banner-warning">⚠ Intercepts should be positive.</div>',
                unsafe_allow_html=True
            )

    # Unsaved warning
    has_input = not (xint_ans == 0.0 and yint_ans == 0.0 and slope_ans == 0.0)
    if not review_mode:
        _unsaved_warning("Q3", has_input, already_submitted)

    # ── Submit ────────────────────────────────────────────────────────────────
    all_filled = has_input
    if not already_submitted and not (past_deadline and not grace_active) and not review_mode:
        if all_filled:
            if st.button("Submit Q3", key=f"sub_q3_{hw_id}",
                         use_container_width=True):
                _submit_q3(hw_id, email, xint_ans, yint_ans, slope_ans,
                           ANS, past_deadline)
                st.rerun()
        else:
            st.caption("Fill in all three boxes above to enable submission.")

    # ── Solution (shown if submitted or review mode) ───────────────────────────
    if already_submitted or review_mode:
        _show_q3_solution(prev, ANS, I, Px, Py)

    # Flag
    if not disabled and not review_mode:
        st.markdown('<div class="flag-section">', unsafe_allow_html=True)
        _flag_section("Q3", hw_id, email)
        st.markdown('</div>', unsafe_allow_html=True)


def _submit_q3(hw_id, email, xint_ans, yint_ans, slope_ans,
               ANS, past_deadline):
    x_ok = (xint_ans == ANS["xint"])
    y_ok = (yint_ans == ANS["yint"])
    s_ok = (slope_ans == ANS["slope"])
    sc   = 2*int(x_ok) + 2*int(y_ok) + 2*int(s_ok)
    ts   = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    late = "Yes" if past_deadline else "No"
    raw  = str({"xint": xint_ans, "yint": yint_ans, "slope": slope_ans})

    row = [
        ts, email, hw_id, "Q3", "numerical", "submitted", late,
        st.session_state.get(f"reloads_Q3_{hw_id}", 0),
        f"xint={ANS['xint']},yint={ANS['yint']},slope={ANS['slope']}",
        raw, sc, 6,
        f"xint={ANS['xint']},yint={ANS['yint']},slope={ANS['slope']}", "1"
    ]
    ok, err = write_submission(row)

    # Update session state
    st.session_state.setdefault("submissions", {})
    st.session_state["submissions"].setdefault(hw_id, {})
    st.session_state["submissions"][hw_id]["Q3"] = {
        "Status": "submitted", "Timestamp": ts, "Score": sc,
        "Max_Score": 6, "Is_Late": late,
        "Raw_Answer": raw,
    }

    if ok:
        st.markdown(f"""
        <div class="banner-success">
          ✓ Q3 submitted — <strong>Score: {sc} / 6</strong>
          <span style="font-size:0.8rem;opacity:0.75;margin-left:0.5rem;">{ts}</span>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(
            f'<div class="banner-warning">⚠ Sheet write failed ({err}). '
            f'Screenshot this. Score: {sc}/6 at {ts}</div>',
            unsafe_allow_html=True
        )


def _show_q3_solution(prev, ANS, I, Px, Py):
    # Parse saved answers
    xv = sv = yv = 0.0
    raw = prev.get("Raw_Answer", "")
    if raw:
        try:
            d  = eval(str(raw))
            xv = float(d.get("xint", 0))
            yv = float(d.get("yint", 0))
            sv = float(d.get("slope", 0))
        except Exception:
            pass

    x_ok = (xv == ANS["xint"])
    y_ok = (yv == ANS["yint"])
    s_ok = (sv == ANS["slope"])
    sc   = 2*int(x_ok) + 2*int(y_ok) + 2*int(s_ok)

    def chip(ok, pts=2):
        return (f'<span class="chip-ok">+{pts}</span>'
                if ok else f'<span class="chip-wrong">0</span>')

    # Revise topics
    revise = []
    if not x_ok or not y_ok:
        revise.append("Budget constraint intercepts — review how to set Y=0 and X=0")
    if not s_ok:
        revise.append("Budget line slope — review why slope = −P<sub>x</sub>/P<sub>y</sub>")

    fig3 = _make_budget_graph(I, Px, Py)
    b64  = fig_to_b64(fig3)
    plt.close(fig3)

    revise_html = ""
    if revise:
        items = "".join(f"<li>{r}</li>" for r in revise)
        revise_html = f"""
        <div class="sol-section">
          <div class="sol-section-label">Topics to revise</div>
          <div class="sol-revise-box">
            <ul style="margin:0;padding-left:1.2rem;font-size:0.88rem;line-height:1.8;">
              {items}
            </ul>
          </div>
        </div>
        """

    st.markdown(f"""
    <div class="sol-card">
      <div class="sol-header">Solution — Q3</div>
      <div class="sol-section">
        <div class="sol-section-label">Step-by-step</div>
        <div class="sol-steps-box">
          <p><strong>(a)</strong> Budget constraint: ${Px}X + {Py}Y = {I}$</p>
          <p><strong>(b)</strong>
            Set $Y=0$: $X_{{max}} = {I}/{Px} = {ANS['xint']:.4g}$
            &nbsp;·&nbsp;
            Set $X=0$: $Y_{{max}} = {I}/{Py} = {ANS['yint']:.4g}$
          </p>
          <p><strong>(c)</strong>
            Slope $= -P_x/P_y = -{Px}/{Py} = {ANS['slope']:.4g}$
          </p>
        </div>
        <table class="score-table" style="margin-top:0.8rem;">
          <tr><th>Part</th><th>Correct answer</th><th>Your answer</th><th>Score</th></tr>
          <tr>
            <td>(b) X-intercept</td>
            <td>{ANS['xint']:.4g}</td>
            <td>{xv:.4g}</td>
            <td>{chip(x_ok)}</td>
          </tr>
          <tr>
            <td>(b) Y-intercept</td>
            <td>{ANS['yint']:.4g}</td>
            <td>{yv:.4g}</td>
            <td>{chip(y_ok)}</td>
          </tr>
          <tr>
            <td>(c) Slope</td>
            <td>{ANS['slope']:.4g}</td>
            <td>{sv:.4g}</td>
            <td>{chip(s_ok)}</td>
          </tr>
          <tr>
            <td colspan="3"><strong>Total</strong></td>
            <td><strong>{sc} / 6</strong></td>
          </tr>
        </table>
      </div>
      <div class="sol-section">
        <div class="sol-section-label">Common mistakes</div>
        <div class="sol-mistakes-box">
          <ul style="margin:0;padding-left:1.2rem;font-size:0.88rem;line-height:1.8;">
            <li>Writing slope as $-P_y/P_x$ instead of $-P_x/P_y$</li>
            <li>Confusing a parallel shift (income change) with a pivot (price change)</li>
          </ul>
        </div>
      </div>
      {revise_html}
      <div class="sol-section">
        <div class="sol-section-label">Reference diagram (part d)</div>
        <div style="text-align:center;padding:0.5rem 0;">
          <img src="data:image/png;base64,{b64}"
               style="max-width:400px;width:100%;border-radius:6px;
                      border:1px solid #E5E7EB;">
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)


def _q3_worked_example(I, Px, Py):
    ex_I = 100; ex_Px = 4; ex_Py = 5
    return f"""
    <div style="font-size:0.88rem;line-height:1.8;color:#374151;">
      <p><strong>Similar problem:</strong>
         I = ${ex_I}, P<sub>x</sub> = ${ex_Px}, P<sub>y</sub> = ${ex_Py}</p>
      <p><strong>Step 1</strong> — Write budget constraint:
         {ex_Px}X + {ex_Py}Y = {ex_I}</p>
      <p><strong>Step 2</strong> — X-intercept: set Y = 0 →
         X = {ex_I}/{ex_Px} = <strong>{ex_I//ex_Px}</strong></p>
      <p><strong>Step 3</strong> — Y-intercept: set X = 0 →
         Y = {ex_I}/{ex_Py} = <strong>{ex_I//ex_Py}</strong></p>
      <p><strong>Step 4</strong> — Slope:
         −P<sub>x</sub>/P<sub>y</sub> = −{ex_Px}/{ex_Py} =
         <strong>{round(-ex_Px/ex_Py, 4)}</strong></p>
      <p style="font-size:0.8rem;color:#6B7280;margin-top:0.5rem;">
        Your question has different values — apply the same method.</p>
    </div>
    """


# ════════════════════════════════════════════════════════════════════════════════
#  Q9 — TOM & JERRY
# ════════════════════════════════════════════════════════════════════════════════

def _render_q9(q_config, hw_id, email, name, past_deadline,
               grace_active, submissions, review_mode):
    I, Px, Py, tom_a, tom_b = _q9_params(email)
    ANS = {
        "tom_x":   float(I / Px),
        "tom_y":   0.0,
        "jerry_x": float(I / (Px + Py)),
        "jerry_y": float(I / (Px + Py)),
    }

    prev = {}
    hw_subs = submissions.get(hw_id, {})
    if isinstance(hw_subs, dict):
        prev = hw_subs.get("Q9", {})
    if not isinstance(prev, dict):
        prev = {}

    already_submitted = str(prev.get("Status", "")) == "submitted"
    disabled = already_submitted or (past_deadline and not grace_active) or review_mode

    st.markdown(f"""
    <div id="Q9"></div>
    <div class="q-header">
      <div>
        <div class="q-header-title">Q9 — Tom &amp; Jerry: Corner vs Kink</div>
        <div class="q-header-sub">8 points</div>
      </div>
      <span class="badge badge-graded">GRADED</span>
    </div>
    <div class="q-body">
      <p>Tom and Jerry both have <strong>I&nbsp;=&nbsp;${I}</strong>,
         <strong>P<sub>x</sub>&nbsp;=&nbsp;${Px}</strong>,
         <strong>P<sub>y</sub>&nbsp;=&nbsp;${Py}</strong>.</p>
      <ul style="margin:0.35rem 0 0.35rem 1.1rem;font-size:0.9rem;">
        <li><strong>Tom:</strong>
            $U_T = {tom_a}X + {tom_b}Y$ &nbsp;(perfect substitutes)</li>
        <li><strong>Jerry:</strong>
            $U_J = \\min(X, Y)$ &nbsp;(perfect complements)</li>
      </ul>
      <div class="part-row">
        <span class="part-badge">(a) 4 pts</span>
        <span class="part-text">Find <strong>Tom's</strong> optimal bundle $(X^*, Y^*)$.</span>
      </div>
      <div class="part-row">
        <span class="part-badge">(b) 4 pts</span>
        <span class="part-text">Find <strong>Jerry's</strong> optimal bundle $(X^*, Y^*)$.</span>
      </div>
      <div class="part-row">
        <span class="part-badge">(c)</span>
        <span class="part-text">Explain why their optimal bundles differ so dramatically.</span>
        <span class="part-ungraded">Ungraded</span>
      </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown(f"""
    <div style="margin-bottom:0.7rem;">
      <span style="font-size:0.68rem;font-weight:600;letter-spacing:0.1em;
       text-transform:uppercase;color:#6B7280;">Your parameters</span><br>
      <span class="param-chip">I = ${I}</span>
      <span class="param-chip">P<sub>x</sub> = ${Px}</span>
      <span class="param-chip">P<sub>y</sub> = ${Py}</span>
      <span class="param-chip">Tom: U = {tom_a}X + {tom_b}Y</span>
    </div>
    """, unsafe_allow_html=True)

    elapsed_s = 0.0
    if not review_mode:
        hw_cfg       = _get_hw_cfg(hw_id)
        deadline_str = hw_cfg.get("Deadline", "2099-12-31 23:59")
        grace_min    = int(hw_cfg.get("Grace_Minutes", 15))
        elapsed_s    = _render_timers("Q9", deadline_str, grace_min)

    if review_mode:
        st.markdown('<div class="banner-review">📖 Review mode — answers are read-only.</div>',
                    unsafe_allow_html=True)
    elif already_submitted:
        ts_prev   = prev.get("Timestamp", "")
        sc_prev   = prev.get("Score", "?")
        late_note = " · Late submission" if prev.get("Is_Late") == "Yes" else ""
        st.markdown(
            f'<div class="banner-locked">🔒 Submitted — Score: <strong>{sc_prev} / 8</strong>'
            f' · {ts_prev}{late_note}</div>',
            unsafe_allow_html=True
        )
    elif prev:
        st.markdown('<div class="banner-restore">Draft restored from previous session.</div>',
                    unsafe_allow_html=True)

    if not review_mode:
        _worked_example_section("Q9", elapsed_s,
                                _q9_worked_example(I, Px, Py, tom_a, tom_b))

    def _pv9(key, default=0.0):
        raw = prev.get("Raw_Answer", "")
        if raw:
            try:
                d = eval(str(raw))
                if key in d:
                    return float(d[key])
            except Exception:
                pass
        v = prev.get(key)
        if v not in (None, ""):
            try:
                return float(v)
            except Exception:
                pass
        return default

    default_tx = _pv9("tom_x"); default_ty = _pv9("tom_y")
    default_jx = _pv9("jerry_x"); default_jy = _pv9("jerry_y")

    st.markdown('<div class="answer-area"><div class="answer-label">Your Answers</div>',
                unsafe_allow_html=True)
    st.markdown("**(a) Tom's optimal bundle:**")
    c1, c2 = st.columns(2)
    with c1:
        tom_x = st.number_input(
            "Tom X*", value=float(default_tx), step=0.01,
            format="%.4f", disabled=disabled, key=f"q9_tx_{hw_id}"
        )
    with c2:
        tom_y = st.number_input(
            "Tom Y*", value=float(default_ty), step=0.01,
            format="%.4f", disabled=disabled, key=f"q9_ty_{hw_id}"
        )
    st.markdown("**(b) Jerry's optimal bundle:**")
    c3, c4 = st.columns(2)
    with c3:
        jerry_x = st.number_input(
            "Jerry X*", value=float(default_jx), step=0.01,
            format="%.4f", disabled=disabled, key=f"q9_jx_{hw_id}"
        )
    with c4:
        jerry_y = st.number_input(
            "Jerry Y*", value=float(default_jy), step=0.01,
            format="%.4f", disabled=disabled, key=f"q9_jy_{hw_id}"
        )
    st.markdown('</div>', unsafe_allow_html=True)

    if not disabled:
        if (abs(jerry_x - jerry_y) > 0.01
                and (jerry_x != 0.0 or jerry_y != 0.0)):
            st.markdown(
                '<div class="banner-info">💡 For Jerry (perfect complements): '
                'the optimum always satisfies X* = Y*.</div>',
                unsafe_allow_html=True
            )

    has_input = any(v != 0.0 for v in [tom_x, tom_y, jerry_x, jerry_y])
    if not review_mode:
        _unsaved_warning("Q9", has_input, already_submitted)

    all_filled = has_input
    if not already_submitted and not (past_deadline and not grace_active) and not review_mode:
        if all_filled:
            if st.button("Submit Q9", key=f"sub_q9_{hw_id}",
                         use_container_width=True):
                _submit_q9(hw_id, email, tom_x, tom_y, jerry_x, jerry_y,
                           ANS, past_deadline)
                st.rerun()
        else:
            st.caption("Fill in all four boxes above to enable submission.")

    if already_submitted or review_mode:
        _show_q9_solution(prev, ANS, I, Px, Py, tom_a, tom_b)

    if not disabled and not review_mode:
        st.markdown('<div class="flag-section">', unsafe_allow_html=True)
        _flag_section("Q9", hw_id, email)
        st.markdown('</div>', unsafe_allow_html=True)


def _submit_q9(hw_id, email, tom_x, tom_y, jerry_x, jerry_y,
               ANS, past_deadline):
    tok  = (tom_x == ANS["tom_x"] and tom_y == ANS["tom_y"])
    jok  = (jerry_x == ANS["jerry_x"] and jerry_y == ANS["jerry_y"])
    sc   = 4*int(tok) + 4*int(jok)
    ts   = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    late = "Yes" if past_deadline else "No"
    raw  = str({"tom_x": tom_x, "tom_y": tom_y,
                "jerry_x": jerry_x, "jerry_y": jerry_y})

    row = [
        ts, email, hw_id, "Q9", "numerical", "submitted", late,
        st.session_state.get(f"reloads_Q9_{hw_id}", 0),
        f"tom=({ANS['tom_x']},{ANS['tom_y']}),jerry=({ANS['jerry_x']:.4f},{ANS['jerry_y']:.4f})",
        raw, sc, 8,
        f"tom=({ANS['tom_x']},{ANS['tom_y']}),jerry=({ANS['jerry_x']:.4f},{ANS['jerry_y']:.4f})",
        "1"
    ]
    ok, err = write_submission(row)

    st.session_state.setdefault("submissions", {})
    st.session_state["submissions"].setdefault(hw_id, {})
    st.session_state["submissions"][hw_id]["Q9"] = {
        "Status": "submitted", "Timestamp": ts, "Score": sc,
        "Max_Score": 8, "Is_Late": late, "Raw_Answer": raw,
    }

    if ok:
        st.markdown(f"""
        <div class="banner-success">
          ✓ Q9 submitted — <strong>Score: {sc} / 8</strong>
          <span style="font-size:0.8rem;opacity:0.75;margin-left:0.5rem;">{ts}</span>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(
            f'<div class="banner-warning">⚠ Sheet write failed ({err}). '
            f'Screenshot this. Score: {sc}/8 at {ts}</div>',
            unsafe_allow_html=True
        )


def _show_q9_solution(prev, ANS, I, Px, Py, tom_a, tom_b):
    tx = ty = jx = jy = 0.0
    raw = prev.get("Raw_Answer", "")
    if raw:
        try:
            d  = eval(str(raw))
            tx = float(d.get("tom_x", 0))
            ty = float(d.get("tom_y", 0))
            jx = float(d.get("jerry_x", 0))
            jy = float(d.get("jerry_y", 0))
        except Exception:
            pass

    tok = (tx == ANS["tom_x"] and ty == ANS["tom_y"])
    jok = (jx == ANS["jerry_x"] and jy == ANS["jerry_y"])
    sc  = 4*int(tok) + 4*int(jok)

    def chip(ok, pts=4):
        return (f'<span class="chip-ok">+{pts}</span>'
                if ok else f'<span class="chip-wrong">0</span>')

    revise = []
    if not tok:
        revise.append("Perfect substitutes — compare MU<sub>x</sub>/P<sub>x</sub> vs MU<sub>y</sub>/P<sub>y</sub> for corner solutions")
    if not jok:
        revise.append("Perfect complements — kink condition X = Y and substitution into budget constraint")

    fig9 = _make_q9_graph(I, Px, Py, tom_a, tom_b)
    b64  = fig_to_b64(fig9)
    plt.close(fig9)

    revise_html = ""
    if revise:
        items = "".join(f"<li>{r}</li>" for r in revise)
        revise_html = f"""
        <div class="sol-section">
          <div class="sol-section-label">Topics to revise</div>
          <div class="sol-revise-box">
            <ul style="margin:0;padding-left:1.2rem;font-size:0.88rem;line-height:1.8;">
              {items}
            </ul>
          </div>
        </div>
        """

    st.markdown(f"""
    <div class="sol-card">
      <div class="sol-header">Solution — Q9</div>
      <div class="sol-section">
        <div class="sol-section-label">Step-by-step</div>
        <div class="sol-steps-box">
          <p><strong>(a) Tom</strong> — compare bang-for-the-buck:<br>
            $MU_x/P_x = {tom_a}/{Px} = {round(tom_a/Px,4)}$ &nbsp;vs&nbsp;
            $MU_y/P_y = {tom_b}/{Py} = {round(tom_b/Py,4)}$<br>
            X dominates → spend all income on X:<br>
            $X^* = {I}/{Px} = {ANS['tom_x']:.4g}$, $Y^* = 0$
          </p>
          <p><strong>(b) Jerry</strong> — kink condition $X = Y$, substitute into budget:<br>
            ${Px}X + {Py}X = {I}$ &nbsp;→&nbsp; ${Px+Py}X = {I}$
            &nbsp;→&nbsp; $X^* = Y^* = {ANS['jerry_x']:.4f}$
          </p>
        </div>
        <table class="score-table" style="margin-top:0.8rem;">
          <tr><th>Part</th><th>Correct $(X^*, Y^*)$</th><th>Your answer</th><th>Score</th></tr>
          <tr>
            <td>(a) Tom</td>
            <td>({ANS['tom_x']:.4g}, {ANS['tom_y']:.4g})</td>
            <td>({tx:.4g}, {ty:.4g})</td>
            <td>{chip(tok)}</td>
          </tr>
          <tr>
            <td>(b) Jerry</td>
            <td>({ANS['jerry_x']:.4f}, {ANS['jerry_y']:.4f})</td>
            <td>({jx:.4f}, {jy:.4f})</td>
            <td>{chip(jok)}</td>
          </tr>
          <tr>
            <td colspan="3"><strong>Total</strong></td>
            <td><strong>{sc} / 8</strong></td>
          </tr>
        </table>
      </div>
      <div class="sol-section">
        <div class="sol-section-label">Common mistakes</div>
        <div class="sol-mistakes-box">
          <ul style="margin:0;padding-left:1.2rem;font-size:0.88rem;line-height:1.8;">
            <li><strong>Tom:</strong> Do not use MRS = Px/Py for linear preferences.
                Always compare MUx/Px vs MUy/Py.</li>
            <li><strong>Jerry:</strong> X = Y is the condition — substitute into
                the budget constraint to get numeric values.</li>
          </ul>
        </div>
      </div>
      {revise_html}
      <div class="sol-section">
        <div class="sol-section-label">Reference diagrams</div>
        <div style="text-align:center;padding:0.5rem 0;">
          <img src="data:image/png;base64,{b64}"
               style="max-width:660px;width:100%;border-radius:6px;
                      border:1px solid #E5E7EB;">
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)


def _q9_worked_example(I, Px, Py, tom_a, tom_b):
    ex_I = 80; ex_Px = 4; ex_Py = 8; ex_a = 3
    return f"""
    <div style="font-size:0.88rem;line-height:1.8;color:#374151;">
      <p><strong>Similar problem:</strong>
         I=${ex_I}, P<sub>x</sub>=${ex_Px}, P<sub>y</sub>=${ex_Py},
         Tom: U={ex_a}X+Y, Jerry: U=min(X,Y)</p>
      <p><strong>Tom</strong> — MU<sub>x</sub>/P<sub>x</sub> = {ex_a}/{ex_Px} = {round(ex_a/ex_Px,3)},
         MU<sub>y</sub>/P<sub>y</sub> = 1/{ex_Py} = {round(1/ex_Py,3)}<br>
         X dominates → X* = {ex_I}/{ex_Px} = <strong>{ex_I//ex_Px}</strong>,
         Y* = <strong>0</strong></p>
      <p><strong>Jerry</strong> — set X=Y, substitute:
         {ex_Px}X + {ex_Py}X = {ex_I} →
         {ex_Px+ex_Py}X = {ex_I} →
         X* = Y* = <strong>{round(ex_I/(ex_Px+ex_Py),4)}</strong></p>
      <p style="font-size:0.8rem;color:#6B7280;margin-top:0.4rem;">
        Your question has different values — apply the same method.</p>
    </div>
    """


# ════════════════════════════════════════════════════════════════════════════════
#  QTF — TRUE/FALSE  (bug fixed: submissions lookup)
# ════════════════════════════════════════════════════════════════════════════════

def _render_truefalse(q_config, hw_id, email, name, past_deadline,
                      grace_active, submissions, review_mode):
    q_id = q_config.get("q_id", "QTF")

    # ── FIXED: correct submissions lookup ─────────────────────────────────────
    prev = {}
    hw_subs = submissions.get(hw_id, {})
    if isinstance(hw_subs, dict):
        prev = hw_subs.get(q_id, {})
    if not isinstance(prev, dict):
        prev = {}
    # ─────────────────────────────────────────────────────────────────────────

    already_submitted = str(prev.get("Status", "")) == "submitted"
    disabled = already_submitted or (past_deadline and not grace_active) or review_mode

    # Shuffle statements per student
    rng      = np.random.default_rng(get_seed(email) + 42)
    indices  = list(range(len(TF_STATEMENTS)))
    rng.shuffle(indices)
    shuffled = [TF_STATEMENTS[i] for i in indices]
    correct_combo = tuple(s["correct"] for s in shuffled)

    st.markdown(f"""
    <div id="{q_id}"></div>
    <div class="q-header">
      <div>
        <div class="q-header-title">Q10 — True or False</div>
        <div class="q-header-sub">4 points — 1 point per statement</div>
      </div>
      <span class="badge badge-graded">GRADED</span>
    </div>
    <div class="q-body">
      <p>Indicate whether each statement is <strong>True</strong>
         or <strong>False</strong>.</p>
    </div>
    """, unsafe_allow_html=True)

    elapsed_s = 0.0
    if not review_mode:
        hw_cfg       = _get_hw_cfg(hw_id)
        deadline_str = hw_cfg.get("Deadline", "2099-12-31 23:59")
        grace_min    = int(hw_cfg.get("Grace_Minutes", 15))
        elapsed_s    = _render_timers(q_id, deadline_str, grace_min)

    if review_mode:
        st.markdown('<div class="banner-review">📖 Review mode — answers are read-only.</div>',
                    unsafe_allow_html=True)
    elif already_submitted:
        ts_prev   = prev.get("Timestamp", "")
        sc_prev   = prev.get("Score", "?")
        late_note = " · Late submission" if prev.get("Is_Late") == "Yes" else ""
        st.markdown(
            f'<div class="banner-locked">🔒 Submitted — Score: <strong>{sc_prev} / 4</strong>'
            f' · {ts_prev}{late_note}</div>',
            unsafe_allow_html=True
        )
    elif prev:
        st.markdown('<div class="banner-restore">Draft restored from previous session.</div>',
                    unsafe_allow_html=True)

    if not review_mode:
        _worked_example_section(q_id, elapsed_s, _tf_worked_example())

    # Restore previous answers
    prev_answers = {}
    raw = prev.get("Raw_Answer", "")
    if raw:
        try:
            prev_answers = eval(str(raw))
            if not isinstance(prev_answers, dict):
                prev_answers = {}
        except Exception:
            prev_answers = {}

    st.markdown('<div class="answer-area"><div class="answer-label">Your Answers</div>',
                unsafe_allow_html=True)

    student_answers = []
    for i, stmt in enumerate(shuffled):
        prev_val = prev_answers.get(f"s{i}", None)
        default_idx = 0 if (prev_val is True or prev_val is None) else 1

        st.markdown(f"""
        <div class="part-row" style="margin-bottom:0.25rem;">
          <span class="part-badge">({i+1})</span>
          <span class="part-text">{stmt['text']}</span>
        </div>
        """, unsafe_allow_html=True)

        choice = st.radio(
            f"Statement {i+1}",
            options=["True", "False"],
            index=default_idx,
            disabled=disabled,
            horizontal=True,
            key=f"tf_{q_id}_{i}_{hw_id}",
            label_visibility="collapsed",
        )
        student_answers.append(choice == "True")

    st.markdown('</div>', unsafe_allow_html=True)

    has_input = True  # radios always have a value
    if not review_mode:
        _unsaved_warning(q_id, has_input, already_submitted)

    if not already_submitted and not (past_deadline and not grace_active) and not review_mode:
        if st.button("Submit Q10", key=f"sub_tf_{hw_id}",
                     use_container_width=True):
            _submit_tf(hw_id, q_id, email, student_answers,
                       correct_combo, shuffled, past_deadline)
            st.rerun()

    if already_submitted or review_mode:
        _show_tf_solution(prev, shuffled, correct_combo)

    if not disabled and not review_mode:
        st.markdown('<div class="flag-section">', unsafe_allow_html=True)
        _flag_section(q_id, hw_id, email)
        st.markdown('</div>', unsafe_allow_html=True)


def _submit_tf(hw_id, q_id, email, student_answers,
               correct_combo, shuffled, past_deadline):
    sc   = sum(1 for s, c in zip(student_answers, correct_combo) if s == c)
    ts   = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    late = "Yes" if past_deadline else "No"
    raw  = str({f"s{i}": v for i, v in enumerate(student_answers)})
    corr = str({f"s{i}": v for i, v in enumerate(correct_combo)})

    row = [
        ts, email, hw_id, q_id, "truefalse", "submitted", late,
        st.session_state.get(f"reloads_{q_id}_{hw_id}", 0),
        corr, raw, sc, 4, corr, "1"
    ]
    ok, err = write_submission(row)

    st.session_state.setdefault("submissions", {})
    st.session_state["submissions"].setdefault(hw_id, {})
    st.session_state["submissions"][hw_id][q_id] = {
        "Status": "submitted", "Timestamp": ts, "Score": sc,
        "Max_Score": 4, "Is_Late": late, "Raw_Answer": raw,
    }

    if ok:
        st.markdown(f"""
        <div class="banner-success">
          ✓ Q10 submitted — <strong>Score: {sc} / 4</strong>
          <span style="font-size:0.8rem;opacity:0.75;margin-left:0.5rem;">{ts}</span>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(
            f'<div class="banner-warning">⚠ Sheet write failed ({err}). '
            f'Screenshot this. Score: {sc}/4 at {ts}</div>',
            unsafe_allow_html=True
        )


def _show_tf_solution(prev, shuffled, correct_combo):
    prev_answers = {}
    raw = prev.get("Raw_Answer", "")
    if raw:
        try:
            prev_answers = eval(str(raw))
            if not isinstance(prev_answers, dict):
                prev_answers = {}
        except Exception:
            prev_answers = {}

    student_answers = [prev_answers.get(f"s{i}", None)
                       for i in range(len(shuffled))]
    sc = sum(1 for s, c in zip(student_answers, correct_combo)
             if s is not None and s == c)

    revise = []
    rows_html = ""
    for i, (stmt, correct, student) in enumerate(
            zip(shuffled, correct_combo, student_answers)):
        correct_str = "True" if correct else "False"
        student_str = ("True" if student else "False") if student is not None else "—"
        is_ok = (student == correct) if student is not None else False
        chip  = ('<span class="chip-ok">✓</span>'
                 if is_ok else '<span class="chip-wrong">✗</span>')
        if not is_ok:
            revise.append(stmt.get("topic", ""))
        rows_html += f"""
        <tr>
          <td style="max-width:320px;font-size:0.85rem;">{stmt['text']}</td>
          <td style="text-align:center;"><strong>{correct_str}</strong></td>
          <td style="text-align:center;">{student_str}</td>
          <td style="text-align:center;">{chip}</td>
        </tr>
        <tr>
          <td colspan="4" style="font-size:0.81rem;color:#6B7280;
              padding:2px 10px 10px 10px;font-style:italic;">
            {stmt['explanation']}
          </td>
        </tr>
        """

    revise_html = ""
    revise = [r for r in revise if r]
    if revise:
        items = "".join(f"<li>{r}</li>" for r in revise)
        revise_html = f"""
        <div class="sol-section">
          <div class="sol-section-label">Topics to revise</div>
          <div class="sol-revise-box">
            <ul style="margin:0;padding-left:1.2rem;font-size:0.88rem;line-height:1.8;">
              {items}
            </ul>
          </div>
        </div>
        """

    st.markdown(f"""
    <div class="sol-card">
      <div class="sol-header">Solution — Q10</div>
      <div class="sol-section">
        <table class="score-table">
          <tr>
            <th>Statement</th>
            <th style="text-align:center;">Correct</th>
            <th style="text-align:center;">Your answer</th>
            <th style="text-align:center;"></th>
          </tr>
          {rows_html}
          <tr>
            <td colspan="3"><strong>Total</strong></td>
            <td><strong>{sc} / 4</strong></td>
          </tr>
        </table>
      </div>
      {revise_html}
    </div>
    """, unsafe_allow_html=True)


def _tf_worked_example():
    return """
    <div style="font-size:0.88rem;line-height:1.8;color:#374151;">
      <p><strong>Example statement:</strong>
         "A consumer with Cobb-Douglas preferences always spends a fixed
         fraction of income on each good."</p>
      <p><strong>Answer: True</strong> — For U = X<sup>α</sup>Y<sup>β</sup>,
         optimal spending on X is always α·I and on Y is always β·I,
         regardless of prices.</p>
      <p><strong>How to approach T/F questions:</strong> Think of the general rule,
         then test against extreme cases. If you can find one counterexample,
         the statement is False.</p>
    </div>
    """


# ── Config helper ──────────────────────────────────────────────────────────────
def _get_hw_cfg(hw_id: str) -> dict:
    configs = st.session_state.get("hw_configs", [])
    for c in configs:
        if c.get("HW_ID") == hw_id:
            return c
    return {"Deadline": "2099-12-31 23:59", "Grace_Minutes": "15"}
