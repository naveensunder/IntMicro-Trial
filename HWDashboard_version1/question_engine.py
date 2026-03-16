"""
question_engine.py — Question rendering.
HWDashboard v3 — stability-first build.

Core rules:
- NO dynamic HTML assembly (no revise_html, rows_html variables injected into f-strings)
- Each section rendered as its own st.markdown() call with static strings
- NO timers, no JS, no shuffle, no worked examples
- Answers rounded to 2dp before comparison
- "Last saved" timestamp shown after submission
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


# ── Seed ───────────────────────────────────────────────────────────────────────
def get_seed(email: str) -> int:
    return int(hashlib.md5(email.lower().encode()).hexdigest(), 16) % 100_000


def r2(v):
    """Round to 2 decimal places for comparison."""
    try:
        return round(float(v), 2)
    except Exception:
        return v


# ── Figure helper ──────────────────────────────────────────────────────────────
def fig_to_b64(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=120)
    buf.seek(0)
    b64 = base64.b64encode(buf.read()).decode("utf-8")
    plt.close(fig)
    return b64


# ════════════════════════════════════════════════════════════════════════════════
#  PARAMETERS
# ════════════════════════════════════════════════════════════════════════════════

def _q1_params(email: str):
    rng     = np.random.default_rng(get_seed(email))
    Px_opts = [2, 3, 4, 5]
    Py_opts = [2, 3, 4, 5, 6]
    I_opts  = [60, 80, 100, 120, 150, 200]
    Px      = int(rng.choice(Px_opts))
    Py      = int(rng.choice([p for p in Py_opts if p != Px]))
    valid_I = [i for i in I_opts if i % Px == 0 and i % Py == 0]
    I       = int(rng.choice(valid_I if valid_I else [120]))
    return I, Px, Py


def _q2_params(email: str):
    rng     = np.random.default_rng(get_seed(email) + 1)
    I_opts  = [60, 80, 90, 120]
    Px_opts = [2, 3, 4, 5]
    Py_opts = [3, 4, 6, 8]
    I       = int(rng.choice(I_opts))
    Px      = int(rng.choice(Px_opts))
    Py      = int(rng.choice([p for p in Py_opts if p != Px]))
    tom_a   = int(rng.choice([2, 3]))
    # Ensure Tom's optimum is clearly X (MUx/Px > MUy/Py)
    attempts = 0
    while not (tom_a / Px > 1 / Py) and attempts < 50:
        Px = int(rng.choice(Px_opts))
        Py = int(rng.choice([p for p in Py_opts if p != Px]))
        attempts += 1
    return I, Px, Py, tom_a


# ════════════════════════════════════════════════════════════════════════════════
#  TRUE/FALSE STATEMENTS
# ════════════════════════════════════════════════════════════════════════════════

TF_STATEMENTS = [
    {
        "text":        "A rise in income causes the budget line to pivot around one intercept.",
        "correct":     False,
        "explanation": "A rise in income causes a parallel outward shift — both intercepts move proportionally. A pivot occurs only when a price changes.",
        "topic":       "Income changes and budget line shifts",
    },
    {
        "text":        "For a Cobb-Douglas utility function U = X^0.4 Y^0.6, the consumer always spends 40% of income on X regardless of prices.",
        "correct":     True,
        "explanation": "Correct. For Cobb-Douglas U = X^a Y^b, optimal spending shares are always a on X and b on Y, independent of prices.",
        "topic":       "Cobb-Douglas utility and spending shares",
    },
    {
        "text":        "Perfect complements have downward-sloping indifference curves.",
        "correct":     False,
        "explanation": "Perfect complements have L-shaped (kinked) indifference curves, not downward-sloping smooth ones.",
        "topic":       "Perfect complements and indifference curve shapes",
    },
    {
        "text":        "For a consumer facing perfect substitutes, the optimal solution is always a corner solution.",
        "correct":     True,
        "explanation": "Correct — unless MRS exactly equals the price ratio, the consumer spends all income on whichever good gives more utility per dollar.",
        "topic":       "Perfect substitutes and corner solutions",
    },
]


# ════════════════════════════════════════════════════════════════════════════════
#  HOMEWORK CONFIGS
# ════════════════════════════════════════════════════════════════════════════════

ALL_HW_CONFIGS = {
    "HW_WEEK2": {
        "hw_id": "HW_WEEK2",
        "questions": [
            {"q_id": "Q1", "type": "numerical",  "title": "Question 1 — Budget Constraint",        "marks": 6},
            {"q_id": "Q2", "type": "numerical",  "title": "Question 2 — Tom & Jerry",              "marks": 8},
            {"q_id": "Q3", "type": "truefalse",  "title": "Question 3 — True or False",            "marks": 4},
        ]
    }
}


def get_questions(hw_id: str) -> list:
    return ALL_HW_CONFIGS.get(hw_id, {}).get("questions", [])


def get_hw_summary(hw_id: str, email: str, submissions: dict) -> dict:
    hw_subs     = submissions.get(hw_id, {}) if isinstance(submissions, dict) else {}
    config      = ALL_HW_CONFIGS.get(hw_id, {})
    questions   = config.get("questions", [])
    total_score = 0
    total_max   = 0
    n_submitted = 0
    for q in questions:
        q_id  = q["q_id"]
        max_s = q["marks"]
        sub   = hw_subs.get(q_id, {}) if isinstance(hw_subs, dict) else {}
        total_max += max_s
        if str(sub.get("Status","")) == "submitted":
            n_submitted += 1
            try:
                total_score += int(sub.get("Score", 0))
            except Exception:
                pass
    return {
        "total_score": total_score,
        "total_max":   total_max,
        "n_submitted": n_submitted,
        "n_total":     len(questions),
        "all_done":    n_submitted == len(questions),
    }


# ════════════════════════════════════════════════════════════════════════════════
#  MASTER DISPATCHER
# ════════════════════════════════════════════════════════════════════════════════

def render_question(q_config: dict, hw_id: str, email: str,
                    past_deadline: bool, grace_active: bool,
                    submissions: dict):
    q_id   = q_config.get("q_id","")
    q_type = q_config.get("type","")
    if q_type == "numerical":
        if q_id == "Q1":
            _render_q1(q_config, hw_id, email, past_deadline, grace_active, submissions)
        elif q_id == "Q2":
            _render_q2(q_config, hw_id, email, past_deadline, grace_active, submissions)
    elif q_type == "truefalse":
        _render_q3_tf(q_config, hw_id, email, past_deadline, grace_active, submissions)


# ════════════════════════════════════════════════════════════════════════════════
#  QUESTION 1 — BUDGET CONSTRAINT
# ════════════════════════════════════════════════════════════════════════════════

def _render_q1(q_config, hw_id, email, past_deadline, grace_active, submissions):
    I, Px, Py = _q1_params(email)
    ANS_xint  = r2(I / Px)
    ANS_yint  = r2(I / Py)
    ANS_slope = r2(-Px / Py)

    hw_subs = submissions.get(hw_id, {}) if isinstance(submissions, dict) else {}
    prev    = hw_subs.get("Q1", {}) if isinstance(hw_subs, dict) else {}
    if not isinstance(prev, dict):
        prev = {}

    already_submitted = str(prev.get("Status","")) == "submitted"
    disabled = already_submitted or (past_deadline and not grace_active)

    # Header
    st.markdown(
        '<div class="q-header">'
        '<div class="q-header-title">Question 1 — Budget Constraint</div>'
        '<div class="q-header-pts">6 points</div>'
        '</div>',
        unsafe_allow_html=True
    )

    # Body
    st.markdown(
        f'<div class="q-body">'
        f'<p>A consumer has income <strong>I = ${I}</strong>, '
        f'<strong>P<sub>x</sub> = ${Px}</strong>, '
        f'<strong>P<sub>y</sub> = ${Py}</strong>.</p>'
        f'<div class="part-row">'
        f'<span class="part-badge">(a)</span>'
        f'<span class="part-text">Write the equation of the budget constraint.</span>'
        f'<span class="part-ungraded">Ungraded</span>'
        f'</div>'
        f'<div class="part-row">'
        f'<span class="part-badge">(b) 4 pts</span>'
        f'<span class="part-text">Find the <strong>X-intercept</strong> and <strong>Y-intercept</strong>.</span>'
        f'</div>'
        f'<div class="part-row">'
        f'<span class="part-badge">(c) 2 pts</span>'
        f'<span class="part-text">What is the <strong>slope</strong> of the budget line?</span>'
        f'</div>'
        f'<div class="part-row">'
        f'<span class="part-badge">(d)</span>'
        f'<span class="part-text">Draw the budget line on a labelled diagram.</span>'
        f'<span class="part-ungraded">Ungraded — reference graph shown after submitting</span>'
        f'</div>'
        f'</div>',
        unsafe_allow_html=True
    )

    # Parameters
    st.markdown(
        f'<div style="margin-bottom:0.7rem;">'
        f'<span style="font-size:0.67rem;font-weight:600;letter-spacing:0.1em;'
        f'text-transform:uppercase;color:#6B7280;">Your parameters &nbsp;</span>'
        f'<span class="param-chip">I = ${I}</span>'
        f'<span class="param-chip">Px = ${Px}</span>'
        f'<span class="param-chip">Py = ${Py}</span>'
        f'</div>',
        unsafe_allow_html=True
    )

    # Status
    if already_submitted:
        sc   = prev.get("Score","?")
        ts   = prev.get("Timestamp","")
        late = " · Late submission" if prev.get("Is_Late","")=="Yes" else ""
        st.markdown(
            f'<div class="banner banner-locked">'
            f'🔒 Submitted — Score: <strong>{sc} / 6</strong> · {ts}{late}'
            f'</div>',
            unsafe_allow_html=True
        )
    elif past_deadline and not grace_active:
        st.markdown(
            '<div class="banner banner-error">🔒 Deadline has passed.</div>',
            unsafe_allow_html=True
        )
    elif prev:
        st.markdown(
            '<div class="banner banner-restore">Draft restored from previous session.</div>',
            unsafe_allow_html=True
        )

    # Recover saved values
    def _pv(key, default=0.0):
        raw = prev.get("Raw_Answer","")
        if raw:
            try:
                d = eval(str(raw))
                if isinstance(d, dict) and key in d:
                    return float(d[key])
            except Exception:
                pass
        return default

    default_x = _pv("xint"); default_y = _pv("yint"); default_s = _pv("slope")

    # Inputs
    st.markdown(
        '<div class="answer-area"><div class="answer-label">Your Answers</div>',
        unsafe_allow_html=True
    )
    c1, c2, c3 = st.columns(3)
    with c1:
        xint_ans = st.number_input("(b) X-intercept", value=float(default_x),
                                   step=0.01, format="%.2f",
                                   disabled=disabled, key=f"q1x_{hw_id}")
    with c2:
        yint_ans = st.number_input("(b) Y-intercept", value=float(default_y),
                                   step=0.01, format="%.2f",
                                   disabled=disabled, key=f"q1y_{hw_id}")
    with c3:
        slope_ans = st.number_input("(c) Slope", value=float(default_s),
                                    step=0.01, format="%.2f",
                                    disabled=disabled, key=f"q1s_{hw_id}")
    st.markdown('</div>', unsafe_allow_html=True)

    # Submit
    if not already_submitted and not (past_deadline and not grace_active):
        has_input = not (xint_ans == 0.0 and yint_ans == 0.0 and slope_ans == 0.0)
        if has_input:
            if st.button("Submit Question 1", key=f"sub_q1_{hw_id}",
                         use_container_width=True):
                _submit_q1(hw_id, email, xint_ans, yint_ans, slope_ans,
                           ANS_xint, ANS_yint, ANS_slope, past_deadline, submissions)
                st.rerun()
        else:
            st.caption("Enter your answers above to enable submission.")

    # Solution
    if already_submitted:
        _show_q1_solution(prev, ANS_xint, ANS_yint, ANS_slope, I, Px, Py)


def _submit_q1(hw_id, email, xint_ans, yint_ans, slope_ans,
               ANS_x, ANS_y, ANS_s, past_deadline, submissions):
    x_ok = r2(xint_ans)  == ANS_x
    y_ok = r2(yint_ans)  == ANS_y
    s_ok = r2(slope_ans) == ANS_s
    sc   = 2*int(x_ok) + 2*int(y_ok) + 2*int(s_ok)
    ts   = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    late = "Yes" if past_deadline else "No"
    raw  = str({"xint": r2(xint_ans), "yint": r2(yint_ans), "slope": r2(slope_ans)})
    corr = str({"xint": ANS_x, "yint": ANS_y, "slope": ANS_s})

    ok, err = write_submission([
        ts, email, hw_id, "Q1", "numerical", "submitted", late,
        raw, sc, 6, corr
    ])

    st.session_state.setdefault("submissions",{}).setdefault(hw_id,{})["Q1"] = {
        "Status": "submitted", "Timestamp": ts, "Score": sc,
        "Max_Score": 6, "Is_Late": late, "Raw_Answer": raw,
    }

    if ok:
        st.markdown(
            f'<div class="banner banner-success">'
            f'✓ Question 1 submitted — <strong>Score: {sc} / 6</strong></div>',
            unsafe_allow_html=True
        )
        st.markdown(f'<div class="saved-ts">Saved at {ts}</div>',
                    unsafe_allow_html=True)
    else:
        st.markdown(
            f'<div class="banner banner-warning">'
            f'⚠ Sheet write failed ({err}). Screenshot this page. '
            f'Score: {sc}/6 at {ts}</div>',
            unsafe_allow_html=True
        )


def _show_q1_solution(prev, ANS_x, ANS_y, ANS_s, I, Px, Py):
    raw = prev.get("Raw_Answer","")
    xv = yv = sv = 0.0
    if raw:
        try:
            d  = eval(str(raw))
            xv = float(d.get("xint", 0))
            yv = float(d.get("yint", 0))
            sv = float(d.get("slope", 0))
        except Exception:
            pass

    x_ok = r2(xv) == ANS_x
    y_ok = r2(yv) == ANS_y
    s_ok = r2(sv) == ANS_s
    sc   = 2*int(x_ok) + 2*int(y_ok) + 2*int(s_ok)

    # Build graph
    fig, ax = plt.subplots(figsize=(5.2, 4.2))
    fig.patch.set_facecolor("#FAFAFA"); ax.set_facecolor("#FAFAFA")
    Xmax = I/Px; Ymax = I/Py
    Xv   = np.linspace(0, Xmax, 300)
    Yv   = (I - Px*Xv)/Py
    ax.plot(Xv, Yv, color="#1C2B4A", lw=2.5)
    ax.fill_between(Xv, Yv, alpha=0.07, color="#1C2B4A")
    ax.plot(Xmax, 0, "o", color="#1C2B4A", ms=7, zorder=5)
    ax.plot(0, Ymax, "o", color="#1C2B4A", ms=7, zorder=5)
    ax.annotate(f"({Xmax:.2f}, 0)", xy=(Xmax,0),
                xytext=(Xmax-Xmax*0.28, Ymax*0.08),
                fontsize=9, color="#1C2B4A",
                arrowprops=dict(arrowstyle="->", color="#1C2B4A", lw=1.0))
    ax.annotate(f"(0, {Ymax:.2f})", xy=(0,Ymax),
                xytext=(Xmax*0.07, Ymax-Ymax*0.14),
                fontsize=9, color="#1C2B4A",
                arrowprops=dict(arrowstyle="->", color="#1C2B4A", lw=1.0))
    ax.set_xlabel("X", fontsize=10); ax.set_ylabel("Y", fontsize=10)
    ax.set_title("Budget Line", fontsize=10, color="#374151")
    ax.set_xlim(0, Xmax*1.2); ax.set_ylim(0, Ymax*1.25)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    ax.tick_params(labelsize=9)
    plt.tight_layout()
    b64 = fig_to_b64(fig)

    # Render solution card — all static, no variable HTML injection
    st.markdown('<div class="sol-card"><div class="sol-header">Solution — Question 1</div>',
                unsafe_allow_html=True)

    # Step by step
    st.markdown(
        f'<div class="sol-section">'
        f'<div class="sol-label">Step-by-step</div>'
        f'<div class="sol-steps">'
        f'<p><strong>(a)</strong> Budget constraint: ${Px}X + {Py}Y = {I}$</p>'
        f'<p><strong>(b)</strong> X-intercept: set Y=0 → X = {I}/{Px} = <strong>{ANS_x}</strong><br>'
        f'Y-intercept: set X=0 → Y = {I}/{Py} = <strong>{ANS_y}</strong></p>'
        f'<p><strong>(c)</strong> Slope = −Px/Py = −{Px}/{Py} = <strong>{ANS_s}</strong></p>'
        f'</div></div>',
        unsafe_allow_html=True
    )

    # Score table — rendered separately, not injected into another f-string
    x_chip = '<span class="chip-ok">+2</span>' if x_ok else '<span class="chip-wrong">0</span>'
    y_chip = '<span class="chip-ok">+2</span>' if y_ok else '<span class="chip-wrong">0</span>'
    s_chip = '<span class="chip-ok">+2</span>' if s_ok else '<span class="chip-wrong">0</span>'

    st.markdown(
        f'<div class="sol-section">'
        f'<div class="sol-label">Your score</div>'
        f'<table class="score-table">'
        f'<tr><th>Part</th><th>Correct</th><th>Your answer</th><th>Score</th></tr>'
        f'<tr><td>(b) X-intercept</td><td>{ANS_x}</td><td>{r2(xv)}</td><td>{x_chip}</td></tr>'
        f'<tr><td>(b) Y-intercept</td><td>{ANS_y}</td><td>{r2(yv)}</td><td>{y_chip}</td></tr>'
        f'<tr><td>(c) Slope</td><td>{ANS_s}</td><td>{r2(sv)}</td><td>{s_chip}</td></tr>'
        f'<tr class="total-row"><td colspan="3"><strong>Total</strong></td>'
        f'<td><strong>{sc} / 6</strong></td></tr>'
        f'</table></div>',
        unsafe_allow_html=True
    )

    # Mistakes
    st.markdown(
        '<div class="sol-section"><div class="sol-label">Common mistakes</div>'
        '<div class="sol-mistakes"><ul style="margin:0;padding-left:1.1rem;'
        'font-size:0.87rem;line-height:1.8;">'
        '<li>Writing slope as −Py/Px instead of −Px/Py</li>'
        '<li>Forgetting that a change in income shifts the line parallel '
        '(not a pivot)</li>'
        '</ul></div></div>',
        unsafe_allow_html=True
    )

    # Topics to revise — only if got something wrong, rendered as separate call
    revise_items = []
    if not x_ok or not y_ok:
        revise_items.append("Budget constraint intercepts — setting Y=0 and X=0")
    if not s_ok:
        revise_items.append("Budget line slope — why slope = −Px/Py")

    if revise_items:
        items_html = "".join(f"<li>{r}</li>" for r in revise_items)
        st.markdown(
            f'<div class="sol-section"><div class="sol-label">Topics to revise</div>'
            f'<div class="sol-revise"><ul style="margin:0;padding-left:1.1rem;'
            f'font-size:0.87rem;line-height:1.8;">{items_html}</ul></div></div>',
            unsafe_allow_html=True
        )

    # Diagram
    st.markdown(
        f'<div class="sol-section"><div class="sol-label">Reference diagram (part d)</div>'
        f'<div style="text-align:center;padding:0.4rem 0;">'
        f'<img src="data:image/png;base64,{b64}" '
        f'style="max-width:380px;width:100%;border-radius:6px;'
        f'border:1px solid #E5E7EB;"></div></div>',
        unsafe_allow_html=True
    )

    st.markdown('</div>', unsafe_allow_html=True)  # close sol-card


# ════════════════════════════════════════════════════════════════════════════════
#  QUESTION 2 — TOM & JERRY
# ════════════════════════════════════════════════════════════════════════════════

def _render_q2(q_config, hw_id, email, past_deadline, grace_active, submissions):
    I, Px, Py, tom_a = _q2_params(email)
    ANS_tx = r2(I / Px)
    ANS_ty = 0.0
    ANS_jx = r2(I / (Px + Py))
    ANS_jy = r2(I / (Px + Py))

    hw_subs = submissions.get(hw_id, {}) if isinstance(submissions, dict) else {}
    prev    = hw_subs.get("Q2", {}) if isinstance(hw_subs, dict) else {}
    if not isinstance(prev, dict):
        prev = {}

    already_submitted = str(prev.get("Status","")) == "submitted"
    disabled = already_submitted or (past_deadline and not grace_active)

    st.markdown(
        '<div class="q-header">'
        '<div class="q-header-title">Question 2 — Tom &amp; Jerry</div>'
        '<div class="q-header-pts">8 points</div>'
        '</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        f'<div class="q-body">'
        f'<p>Tom and Jerry both have <strong>I = ${I}</strong>, '
        f'<strong>Px = ${Px}</strong>, <strong>Py = ${Py}</strong>.</p>'
        f'<ul style="margin:0.3rem 0 0.3rem 1.1rem;font-size:0.9rem;">'
        f'<li><strong>Tom:</strong> U = {tom_a}X + Y &nbsp;(perfect substitutes)</li>'
        f'<li><strong>Jerry:</strong> U = min(X, Y) &nbsp;(perfect complements)</li>'
        f'</ul>'
        f'<div class="part-row">'
        f'<span class="part-badge">(a) 4 pts</span>'
        f'<span class="part-text">Find <strong>Tom\'s</strong> optimal bundle (X*, Y*).</span>'
        f'</div>'
        f'<div class="part-row">'
        f'<span class="part-badge">(b) 4 pts</span>'
        f'<span class="part-text">Find <strong>Jerry\'s</strong> optimal bundle (X*, Y*).</span>'
        f'</div>'
        f'<div class="part-row">'
        f'<span class="part-badge">(c)</span>'
        f'<span class="part-text">Explain why their bundles differ so dramatically.</span>'
        f'<span class="part-ungraded">Ungraded</span>'
        f'</div>'
        f'</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        f'<div style="margin-bottom:0.7rem;">'
        f'<span style="font-size:0.67rem;font-weight:600;letter-spacing:0.1em;'
        f'text-transform:uppercase;color:#6B7280;">Your parameters &nbsp;</span>'
        f'<span class="param-chip">I = ${I}</span>'
        f'<span class="param-chip">Px = ${Px}</span>'
        f'<span class="param-chip">Py = ${Py}</span>'
        f'<span class="param-chip">Tom: U = {tom_a}X + Y</span>'
        f'</div>',
        unsafe_allow_html=True
    )

    if already_submitted:
        sc   = prev.get("Score","?")
        ts   = prev.get("Timestamp","")
        late = " · Late submission" if prev.get("Is_Late","")=="Yes" else ""
        st.markdown(
            f'<div class="banner banner-locked">'
            f'🔒 Submitted — Score: <strong>{sc} / 8</strong> · {ts}{late}'
            f'</div>',
            unsafe_allow_html=True
        )
    elif past_deadline and not grace_active:
        st.markdown(
            '<div class="banner banner-error">🔒 Deadline has passed.</div>',
            unsafe_allow_html=True
        )
    elif prev:
        st.markdown(
            '<div class="banner banner-restore">Draft restored from previous session.</div>',
            unsafe_allow_html=True
        )

    def _pv(key, default=0.0):
        raw = prev.get("Raw_Answer","")
        if raw:
            try:
                d = eval(str(raw))
                if isinstance(d,dict) and key in d:
                    return float(d[key])
            except Exception:
                pass
        return default

    st.markdown(
        '<div class="answer-area"><div class="answer-label">Your Answers</div>',
        unsafe_allow_html=True
    )
    st.markdown("**(a) Tom's optimal bundle:**")
    c1, c2 = st.columns(2)
    with c1:
        tom_x = st.number_input("Tom X*", value=float(_pv("tom_x")),
                                step=0.01, format="%.2f",
                                disabled=disabled, key=f"q2tx_{hw_id}")
    with c2:
        tom_y = st.number_input("Tom Y*", value=float(_pv("tom_y")),
                                step=0.01, format="%.2f",
                                disabled=disabled, key=f"q2ty_{hw_id}")
    st.markdown("**(b) Jerry's optimal bundle:**")
    c3, c4 = st.columns(2)
    with c3:
        jerry_x = st.number_input("Jerry X*", value=float(_pv("jerry_x")),
                                  step=0.01, format="%.2f",
                                  disabled=disabled, key=f"q2jx_{hw_id}")
    with c4:
        jerry_y = st.number_input("Jerry Y*", value=float(_pv("jerry_y")),
                                  step=0.01, format="%.2f",
                                  disabled=disabled, key=f"q2jy_{hw_id}")
    st.markdown('</div>', unsafe_allow_html=True)

    if not disabled:
        jx_r = r2(jerry_x); jy_r = r2(jerry_y)
        if abs(jx_r - jy_r) > 0.05 and (jerry_x != 0.0 or jerry_y != 0.0):
            st.markdown(
                '<div class="banner banner-info">💡 For Jerry (perfect complements):'
                ' the optimal bundle always satisfies X* = Y*.</div>',
                unsafe_allow_html=True
            )

    has_input = any(v != 0.0 for v in [tom_x, tom_y, jerry_x, jerry_y])
    if not already_submitted and not (past_deadline and not grace_active):
        if has_input:
            if st.button("Submit Question 2", key=f"sub_q2_{hw_id}",
                         use_container_width=True):
                _submit_q2(hw_id, email, tom_x, tom_y, jerry_x, jerry_y,
                           ANS_tx, ANS_ty, ANS_jx, ANS_jy, past_deadline)
                st.rerun()
        else:
            st.caption("Enter your answers above to enable submission.")

    if already_submitted:
        _show_q2_solution(prev, ANS_tx, ANS_ty, ANS_jx, ANS_jy,
                          I, Px, Py, tom_a)


def _submit_q2(hw_id, email, tom_x, tom_y, jerry_x, jerry_y,
               ANS_tx, ANS_ty, ANS_jx, ANS_jy, past_deadline):
    tok = (r2(tom_x)==ANS_tx and r2(tom_y)==ANS_ty)
    jok = (r2(jerry_x)==ANS_jx and r2(jerry_y)==ANS_jy)
    sc  = 4*int(tok) + 4*int(jok)
    ts  = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    late = "Yes" if past_deadline else "No"
    raw  = str({"tom_x":r2(tom_x),"tom_y":r2(tom_y),
                "jerry_x":r2(jerry_x),"jerry_y":r2(jerry_y)})
    corr = str({"tom":f"({ANS_tx},{ANS_ty})",
                "jerry":f"({ANS_jx},{ANS_jy})"})

    ok, err = write_submission([
        ts, email, hw_id, "Q2", "numerical", "submitted", late,
        raw, sc, 8, corr
    ])

    st.session_state.setdefault("submissions",{}).setdefault(hw_id,{})["Q2"] = {
        "Status":"submitted","Timestamp":ts,"Score":sc,
        "Max_Score":8,"Is_Late":late,"Raw_Answer":raw,
    }

    if ok:
        st.markdown(
            f'<div class="banner banner-success">'
            f'✓ Question 2 submitted — <strong>Score: {sc} / 8</strong></div>',
            unsafe_allow_html=True
        )
        st.markdown(f'<div class="saved-ts">Saved at {ts}</div>',
                    unsafe_allow_html=True)
    else:
        st.markdown(
            f'<div class="banner banner-warning">'
            f'⚠ Sheet write failed ({err}). Screenshot this. '
            f'Score: {sc}/8 at {ts}</div>',
            unsafe_allow_html=True
        )


def _show_q2_solution(prev, ANS_tx, ANS_ty, ANS_jx, ANS_jy, I, Px, Py, tom_a):
    raw = prev.get("Raw_Answer","")
    tx=ty=jx=jy=0.0
    if raw:
        try:
            d  = eval(str(raw))
            tx = float(d.get("tom_x",0)); ty = float(d.get("tom_y",0))
            jx = float(d.get("jerry_x",0)); jy = float(d.get("jerry_y",0))
        except Exception:
            pass

    tok = (r2(tx)==ANS_tx and r2(ty)==ANS_ty)
    jok = (r2(jx)==ANS_jx and r2(jy)==ANS_jy)
    sc  = 4*int(tok) + 4*int(jok)

    # Graph
    fig, axs = plt.subplots(1, 2, figsize=(11, 4.5))
    fig.patch.set_facecolor("#FAFAFA")
    fig.suptitle(f"Tom vs Jerry  [I={I}, Px={Px}, Py={Py}]",
                 fontsize=10, color="#1C2B4A", fontweight="600")
    Xv = np.linspace(0, I/Px+1, 400)
    Yv = (I - Px*Xv)/Py

    for ax in axs:
        ax.set_facecolor("#FAFAFA")
        ax.plot(Xv, np.where(Yv>=0,Yv,np.nan), color="#1C2B4A", lw=2.2)
        ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
        ax.tick_params(labelsize=9)
        ax.set_xlabel("X",fontsize=10); ax.set_ylabel("Y",fontsize=10)

    Xt = I/Px; Yt = 0.0; Xj = I/(Px+Py); Yj=Xj
    ax = axs[0]
    UT = tom_a*Xt + Yt
    for Ul,alp in [(UT*0.6,0.18),(UT*0.8,0.32),(UT,0.85)]:
        ICs = (Ul - tom_a*Xv)/1.0
        ax.plot(Xv, np.where(ICs>=0,ICs,np.nan), color="#DC2626", lw=1.5, alpha=alp)
    ax.plot(Xt,Yt,"o",color="#DC2626",ms=9,zorder=6)
    ax.set_title(f"Tom: U={tom_a}X+Y (Perfect Substitutes)",fontsize=9,color="#374151")
    ax.set_xlim(0,I/Px*1.2); ax.set_ylim(0,I/Py*1.3)

    ax = axs[1]
    for Ul,alp in [(Xj*0.6,0.18),(Xj*0.8,0.32),(Xj,0.85)]:
        ax.plot([Ul,I/Px*1.2],[Ul,Ul],color="#DC2626",lw=1.5,alpha=alp)
        ax.plot([Ul,Ul],[Ul,I/Py*1.3],color="#DC2626",lw=1.5,alpha=alp)
    diag = np.linspace(0,min(I/Px,I/Py)*1.1,100)
    ax.plot(diag,diag,"--",color="#9CA3AF",lw=1.1,alpha=0.6)
    ax.plot(Xj,Yj,"o",color="#DC2626",ms=9,zorder=6)
    ax.set_title("Jerry: U=min(X,Y) (Perfect Complements)",fontsize=9,color="#374151")
    ax.set_xlim(0,I/Px*1.2); ax.set_ylim(0,I/Py*1.3)
    plt.tight_layout()
    b64 = fig_to_b64(fig)

    t_chip = '<span class="chip-ok">+4</span>' if tok else '<span class="chip-wrong">0</span>'
    j_chip = '<span class="chip-ok">+4</span>' if jok else '<span class="chip-wrong">0</span>'

    st.markdown('<div class="sol-card"><div class="sol-header">Solution — Question 2</div>',
                unsafe_allow_html=True)

    st.markdown(
        f'<div class="sol-section"><div class="sol-label">Step-by-step</div>'
        f'<div class="sol-steps">'
        f'<p><strong>(a) Tom</strong> — compare MUx/Px vs MUy/Py:<br>'
        f'MUx/Px = {tom_a}/{Px} = {round(tom_a/Px,2)} &nbsp;vs&nbsp; '
        f'MUy/Py = 1/{Py} = {round(1/Py,2)}<br>'
        f'X gives more utility per dollar → spend all income on X:<br>'
        f'X* = {I}/{Px} = <strong>{ANS_tx}</strong>, Y* = <strong>0</strong></p>'
        f'<p><strong>(b) Jerry</strong> — kink condition X=Y, substitute into budget:<br>'
        f'{Px}X + {Py}X = {I} → {Px+Py}X = {I} → '
        f'X* = Y* = <strong>{ANS_jx}</strong></p>'
        f'</div></div>',
        unsafe_allow_html=True
    )

    st.markdown(
        f'<div class="sol-section"><div class="sol-label">Your score</div>'
        f'<table class="score-table">'
        f'<tr><th>Part</th><th>Correct (X*, Y*)</th><th>Your answer</th><th>Score</th></tr>'
        f'<tr><td>(a) Tom</td><td>({ANS_tx}, {ANS_ty})</td>'
        f'<td>({r2(tx)}, {r2(ty)})</td><td>{t_chip}</td></tr>'
        f'<tr><td>(b) Jerry</td><td>({ANS_jx}, {ANS_jy})</td>'
        f'<td>({r2(jx)}, {r2(jy)})</td><td>{j_chip}</td></tr>'
        f'<tr class="total-row"><td colspan="3"><strong>Total</strong></td>'
        f'<td><strong>{sc} / 8</strong></td></tr>'
        f'</table></div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="sol-section"><div class="sol-label">Common mistakes</div>'
        '<div class="sol-mistakes"><ul style="margin:0;padding-left:1.1rem;'
        'font-size:0.87rem;line-height:1.8;">'
        '<li><strong>Tom:</strong> Do not use MRS=Px/Py for linear preferences. '
        'Always compare MUx/Px vs MUy/Py directly.</li>'
        '<li><strong>Jerry:</strong> X=Y is the condition at the kink — substitute '
        'into the budget constraint to find the numeric value.</li>'
        '</ul></div></div>',
        unsafe_allow_html=True
    )

    revise_items = []
    if not tok:
        revise_items.append("Perfect substitutes — bang-per-buck comparison and corner solutions")
    if not jok:
        revise_items.append("Perfect complements — kink condition and budget substitution")

    if revise_items:
        items_html = "".join(f"<li>{r}</li>" for r in revise_items)
        st.markdown(
            f'<div class="sol-section"><div class="sol-label">Topics to revise</div>'
            f'<div class="sol-revise"><ul style="margin:0;padding-left:1.1rem;'
            f'font-size:0.87rem;line-height:1.8;">{items_html}</ul></div></div>',
            unsafe_allow_html=True
        )

    st.markdown(
        f'<div class="sol-section"><div class="sol-label">Reference diagrams</div>'
        f'<div style="text-align:center;padding:0.4rem 0;">'
        f'<img src="data:image/png;base64,{b64}" '
        f'style="max-width:640px;width:100%;border-radius:6px;'
        f'border:1px solid #E5E7EB;"></div></div>',
        unsafe_allow_html=True
    )

    st.markdown('</div>', unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════════════════
#  QUESTION 3 — TRUE/FALSE
# ════════════════════════════════════════════════════════════════════════════════

def _render_q3_tf(q_config, hw_id, email, past_deadline, grace_active, submissions):
    # Fixed order for stability — no shuffle
    statements = TF_STATEMENTS
    correct    = [s["correct"] for s in statements]

    hw_subs = submissions.get(hw_id, {}) if isinstance(submissions, dict) else {}
    prev    = hw_subs.get("Q3", {}) if isinstance(hw_subs, dict) else {}
    if not isinstance(prev, dict):
        prev = {}

    already_submitted = str(prev.get("Status","")) == "submitted"
    disabled = already_submitted or (past_deadline and not grace_active)

    st.markdown(
        '<div class="q-header">'
        '<div class="q-header-title">Question 3 — True or False</div>'
        '<div class="q-header-pts">4 points — 1 point per statement</div>'
        '</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="q-body">'
        '<p>Indicate whether each statement is <strong>True</strong> or <strong>False</strong>.</p>'
        '</div>',
        unsafe_allow_html=True
    )

    if already_submitted:
        sc   = prev.get("Score","?")
        ts   = prev.get("Timestamp","")
        late = " · Late submission" if prev.get("Is_Late","")=="Yes" else ""
        st.markdown(
            f'<div class="banner banner-locked">'
            f'🔒 Submitted — Score: <strong>{sc} / 4</strong> · {ts}{late}'
            f'</div>',
            unsafe_allow_html=True
        )
    elif past_deadline and not grace_active:
        st.markdown(
            '<div class="banner banner-error">🔒 Deadline has passed.</div>',
            unsafe_allow_html=True
        )
    elif prev:
        st.markdown(
            '<div class="banner banner-restore">Draft restored.</div>',
            unsafe_allow_html=True
        )

    # Restore previous answers
    prev_answers = {}
    raw = prev.get("Raw_Answer","")
    if raw:
        try:
            pa = eval(str(raw))
            if isinstance(pa, dict):
                prev_answers = pa
        except Exception:
            pass

    st.markdown(
        '<div class="answer-area"><div class="answer-label">Your Answers</div>',
        unsafe_allow_html=True
    )

    student_answers = []
    for i, stmt in enumerate(statements):
        prev_val = prev_answers.get(f"s{i}", None)
        default_idx = 0 if (prev_val is True or prev_val is None) else 1

        st.markdown(
            f'<div class="part-row" style="margin-bottom:0.2rem;">'
            f'<span class="part-badge">({i+1})</span>'
            f'<span class="part-text">{stmt["text"]}</span>'
            f'</div>',
            unsafe_allow_html=True
        )

        choice = st.radio(
            f"Statement {i+1}",
            options=["True","False"],
            index=default_idx,
            disabled=disabled,
            horizontal=True,
            key=f"tf_{i}_{hw_id}",
            label_visibility="collapsed",
        )
        student_answers.append(choice == "True")

    st.markdown('</div>', unsafe_allow_html=True)

    if not already_submitted and not (past_deadline and not grace_active):
        if st.button("Submit Question 3", key=f"sub_q3tf_{hw_id}",
                     use_container_width=True):
            _submit_q3_tf(hw_id, email, student_answers, correct,
                          statements, past_deadline)
            st.rerun()

    if already_submitted:
        _show_q3_tf_solution(prev, statements, correct)


def _submit_q3_tf(hw_id, email, student_answers, correct,
                  statements, past_deadline):
    sc   = sum(1 for s,c in zip(student_answers, correct) if s==c)
    ts   = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    late = "Yes" if past_deadline else "No"
    raw  = str({f"s{i}": v for i,v in enumerate(student_answers)})
    corr = str({f"s{i}": v for i,v in enumerate(correct)})

    ok, err = write_submission([
        ts, email, hw_id, "Q3", "truefalse", "submitted", late,
        raw, sc, 4, corr
    ])

    st.session_state.setdefault("submissions",{}).setdefault(hw_id,{})["Q3"] = {
        "Status":"submitted","Timestamp":ts,"Score":sc,
        "Max_Score":4,"Is_Late":late,"Raw_Answer":raw,
    }

    if ok:
        st.markdown(
            f'<div class="banner banner-success">'
            f'✓ Question 3 submitted — <strong>Score: {sc} / 4</strong></div>',
            unsafe_allow_html=True
        )
        st.markdown(f'<div class="saved-ts">Saved at {ts}</div>',
                    unsafe_allow_html=True)
    else:
        st.markdown(
            f'<div class="banner banner-warning">'
            f'⚠ Sheet write failed ({err}). Screenshot this. '
            f'Score: {sc}/4 at {ts}</div>',
            unsafe_allow_html=True
        )


def _show_q3_tf_solution(prev, statements, correct):
    prev_answers = {}
    raw = prev.get("Raw_Answer","")
    if raw:
        try:
            pa = eval(str(raw))
            if isinstance(pa, dict):
                prev_answers = pa
        except Exception:
            pass

    student_answers = [prev_answers.get(f"s{i}", None)
                       for i in range(len(statements))]
    sc = sum(1 for s,c in zip(student_answers, correct)
             if s is not None and s==c)

    st.markdown('<div class="sol-card"><div class="sol-header">Solution — Question 3</div>',
                unsafe_allow_html=True)

    # Render each statement row as a SEPARATE st.markdown call
    # This is the fix for the raw HTML issue — no dynamic rows_html variable
    st.markdown(
        '<div class="sol-section"><div class="sol-label">Results</div>'
        '<table class="score-table">'
        '<tr><th>Statement</th><th style="text-align:center;">Correct</th>'
        '<th style="text-align:center;">Your answer</th>'
        '<th style="text-align:center;"></th></tr>',
        unsafe_allow_html=True
    )

    revise_items = []
    for i, (stmt, corr_val, student_val) in enumerate(
            zip(statements, correct, student_answers)):
        corr_str    = "True"  if corr_val   else "False"
        student_str = ("True" if student_val else "False") if student_val is not None else "—"
        is_ok       = (student_val == corr_val) if student_val is not None else False
        chip        = '<span class="chip-ok">✓</span>' if is_ok else '<span class="chip-wrong">✗</span>'

        if not is_ok and stmt.get("topic"):
            revise_items.append(stmt["topic"])

        # Each row is its own markdown call — no concatenation into a parent f-string
        st.markdown(
            f'<tr>'
            f'<td style="font-size:0.85rem;max-width:300px;">{stmt["text"]}</td>'
            f'<td style="text-align:center;"><strong>{corr_str}</strong></td>'
            f'<td style="text-align:center;">{student_str}</td>'
            f'<td style="text-align:center;">{chip}</td>'
            f'</tr>',
            unsafe_allow_html=True
        )
        # Explanation row
        st.markdown(
            f'<tr><td colspan="4" style="font-size:0.81rem;color:#6B7280;'
            f'padding:2px 9px 9px 9px;font-style:italic;">'
            f'{stmt["explanation"]}</td></tr>',
            unsafe_allow_html=True
        )

    st.markdown(
        f'<tr class="total-row"><td colspan="3"><strong>Total</strong></td>'
        f'<td><strong>{sc} / 4</strong></td></tr>'
        f'</table></div>',
        unsafe_allow_html=True
    )

    if revise_items:
        items_html = "".join(f"<li>{r}</li>" for r in revise_items)
        st.markdown(
            f'<div class="sol-section"><div class="sol-label">Topics to revise</div>'
            f'<div class="sol-revise"><ul style="margin:0;padding-left:1.1rem;'
            f'font-size:0.87rem;line-height:1.8;">{items_html}</ul></div></div>',
            unsafe_allow_html=True
        )

    st.markdown('</div>', unsafe_allow_html=True)
