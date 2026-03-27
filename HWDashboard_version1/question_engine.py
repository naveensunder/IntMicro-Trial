"""
question_engine.py — HWDashboard v5
Key changes:
- Collapsible question banner with large bold question number
- Three distinct visual zones: question (navy left border), answers (teal left border), solution (green left border)
- Thin dividers between sections
- Submitted answers in collapsible box, default collapsed
- Solution expander default collapsed after submission
- Clearer input labels
- Confirmation before submit
- Back to top link
- Deadline shown in natural language
- Validation warning for unreasonable answers
- Brief instruction line on first open
- Submit button always at same position
"""

import streamlit as st
import datetime
import numpy as np
import hashlib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import io, base64
from db import write_submission, log_submission_attempt


def get_seed(email: str) -> int:
    return int(hashlib.md5(email.lower().encode()).hexdigest(), 16) % 100_000


def r2(v):
    try:
        return round(float(v), 2)
    except Exception:
        return v


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

def _q1_params(email):
    rng     = np.random.default_rng(get_seed(email))
    Px_opts = [2, 3, 4, 5]
    Py_opts = [2, 3, 4, 5, 6]
    I_opts  = [60, 80, 100, 120, 150, 200]
    Px      = int(rng.choice(Px_opts))
    Py      = int(rng.choice([p for p in Py_opts if p != Px]))
    valid_I = [i for i in I_opts if i % Px == 0 and i % Py == 0]
    I       = int(rng.choice(valid_I if valid_I else [120]))
    return I, Px, Py


def _q2_params(email):
    rng     = np.random.default_rng(get_seed(email) + 1)
    I_opts  = [60, 80, 90, 120]
    Px_opts = [2, 3, 4, 5]
    Py_opts = [3, 4, 6, 8]
    I       = int(rng.choice(I_opts))
    Px      = int(rng.choice(Px_opts))
    Py      = int(rng.choice([p for p in Py_opts if p != Px]))
    tom_a   = int(rng.choice([2, 3]))
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
        "text": "A rise in income causes the budget line to pivot around one intercept.",
        "correct": False,
        "explanation": (
            "This statement is False. When income rises, both intercepts increase proportionally "
            "because the consumer can afford more of either good. The result is a parallel outward "
            "shift of the entire budget line, not a pivot. A pivot only happens when one price changes, "
            "rotating the line around the intercept of the other good."
        ),
        "topic": "Budget line shifts vs pivots",
    },
    {
        "text": "For a Cobb-Douglas utility function U = X^0.4 x Y^0.6, the consumer always spends 40% of income on X regardless of prices.",
        "correct": True,
        "explanation": (
            "This statement is True. A key property of Cobb-Douglas preferences is that the optimal "
            "spending shares are fixed: the consumer always spends a fraction equal to the exponent on "
            "each good. So 40% of income goes to X and 60% to Y, no matter what prices are. "
            "This makes Cobb-Douglas very tractable for economics problems."
        ),
        "topic": "Cobb-Douglas utility and constant spending shares",
    },
    {
        "text": "Perfect complements have downward-sloping indifference curves.",
        "correct": False,
        "explanation": (
            "This statement is False. Perfect complements have L-shaped (right-angle) indifference "
            "curves with a kink at the optimal ratio. The consumer gains no additional utility from "
            "having more of one good without a matching increase in the other. Downward-sloping smooth "
            "curves describe standard preferences like Cobb-Douglas, not perfect complements."
        ),
        "topic": "Indifference curve shapes for perfect complements",
    },
    {
        "text": "For a consumer facing perfect substitutes, the optimal solution is always a corner solution.",
        "correct": True,
        "explanation": (
            "This statement is True. With perfect substitutes, the consumer always chooses whichever "
            "good gives more utility per dollar, spending all income on that good. This is a corner "
            "solution. The only exception is when both goods give exactly the same utility per dollar, "
            "in which case any bundle on the budget line is optimal."
        ),
        "topic": "Perfect substitutes and corner solutions",
    },
]


# ════════════════════════════════════════════════════════════════════════════════
#  HOMEWORK CONFIGS
# ════════════════════════════════════════════════════════════════════════════════

ALL_HW_CONFIGS = {
    "HW_WEEK1": {
        "hw_id": "HW_WEEK1",
        "questions": [
            {"q_id": "Q1", "type": "truefalse", "title": "Question 1 — Opportunity Cost & Sunk Costs", "marks": 4},
            {"q_id": "Q2", "type": "numerical", "title": "Question 2 — Slopes of Linear Functions",   "marks": 6},
            {"q_id": "Q3", "type": "numerical", "title": "Question 3 — Derivatives & Partial Derivatives", "marks": 6},
            {"q_id": "Q4", "type": "numerical", "title": "Question 4 — Demand, Supply & Equilibrium", "marks": 10},
            {"q_id": "Q5", "type": "truefalse", "title": "Question 5 — Preference Assumptions",       "marks": 4},
            {"q_id": "Q6", "type": "numerical", "title": "Question 6 — Indifference Curves",          "marks": 6},
            {"q_id": "Q7", "type": "numerical", "title": "Question 7 — Marginal Utility",             "marks": 6},
            {"q_id": "Q8", "type": "numerical", "title": "Question 8 — Marginal Rate of Substitution","marks": 8},
        ]
    },
    "HW_WEEK2": {
        "hw_id": "HW_WEEK2",
        "questions": [
            {"q_id": "Q1", "type": "numerical", "title": "Question 1 — Budget Constraint", "marks": 6},
            {"q_id": "Q2", "type": "numerical", "title": "Question 2 — Tom & Jerry",        "marks": 8},
            {"q_id": "Q3", "type": "truefalse", "title": "Question 3 — True or False",      "marks": 4},
        ]
    }
}


def get_questions(hw_id: str) -> list:
    return ALL_HW_CONFIGS.get(hw_id, {}).get("questions", [])


def get_hw_summary(hw_id: str, email: str, submissions: dict) -> dict:
    hw_subs   = submissions.get(hw_id, {}) if isinstance(submissions, dict) else {}
    questions = ALL_HW_CONFIGS.get(hw_id, {}).get("questions", [])
    total_score = 0; total_max = 0; n_submitted = 0
    for q in questions:
        q_id = q["q_id"]; max_s = q["marks"]
        sub  = hw_subs.get(q_id, {}) if isinstance(hw_subs, dict) else {}
        total_max += max_s
        if str(sub.get("Status","")) == "submitted":
            n_submitted += 1
            try: total_score += int(sub.get("Score", 0))
            except Exception: pass
    return {
        "total_score": total_score, "total_max": total_max,
        "n_submitted": n_submitted, "n_total": len(questions),
        "all_done": n_submitted == len(questions),
    }


# ════════════════════════════════════════════════════════════════════════════════
#  MASTER DISPATCHER
# ════════════════════════════════════════════════════════════════════════════════

def render_question(q_config, hw_id, email, past_deadline, grace_active, submissions):
    q_id   = q_config.get("q_id","")
    q_type = q_config.get("type","")

    # ── Week 1 ────────────────────────────────────────────────────────────────
    if hw_id == "HW_WEEK1":
        if q_type == "truefalse":
            if q_id == "Q1":
                _render_w1_tf(q_config, hw_id, email, past_deadline, grace_active,
                              submissions, W1_TF_Q1, "W1Q1")
            elif q_id == "Q5":
                _render_w1_tf(q_config, hw_id, email, past_deadline, grace_active,
                              submissions, W1_TF_Q5, "W1Q5")
        elif q_type == "numerical":
            if q_id == "Q2":
                _render_w1_q2(q_config, hw_id, email, past_deadline, grace_active, submissions)
            elif q_id == "Q3":
                _render_w1_q3(q_config, hw_id, email, past_deadline, grace_active, submissions)
            elif q_id == "Q4":
                _render_w1_q4(q_config, hw_id, email, past_deadline, grace_active, submissions)
            elif q_id == "Q6":
                _render_w1_q6(q_config, hw_id, email, past_deadline, grace_active, submissions)
            elif q_id == "Q7":
                _render_w1_q7(q_config, hw_id, email, past_deadline, grace_active, submissions)
            elif q_id == "Q8":
                _render_w1_q8(q_config, hw_id, email, past_deadline, grace_active, submissions)

    # ── Week 2 ────────────────────────────────────────────────────────────────
    elif hw_id == "HW_WEEK2":
        if q_type == "numerical":
            if q_id == "Q1":
                _render_q1(q_config, hw_id, email, past_deadline, grace_active, submissions)
            elif q_id == "Q2":
                _render_q2(q_config, hw_id, email, past_deadline, grace_active, submissions)
        elif q_type == "truefalse":
            _render_q3_tf(q_config, hw_id, email, past_deadline, grace_active, submissions)


# ════════════════════════════════════════════════════════════════════════════════
#  SHARED HELPERS
# ════════════════════════════════════════════════════════════════════════════════

def _get_prev(submissions, hw_id, q_id):
    hw_subs = submissions.get(hw_id, {}) if isinstance(submissions, dict) else {}
    prev    = hw_subs.get(q_id, {}) if isinstance(hw_subs, dict) else {}
    return prev if isinstance(prev, dict) else {}


def _parse_raw(prev):
    raw = prev.get("Raw_Answer","")
    if raw:
        try:
            d = eval(str(raw))
            if isinstance(d, dict): return d
        except Exception: pass
    return {}


def _question_banner(q_number: str, title: str, already: bool,
                     score=None, max_score=None) -> bool:
    """
    Renders collapsible question banner.
    Returns True if the banner is expanded (content should show).
    Default: expanded if not submitted, collapsed if submitted.
    """
    if already and score is not None:
        banner_html = (
            f'<div class="q-banner-submitted">'
            f'<div class="q-banner-title">{title}</div>'
            f'<div class="q-banner-score">Score: {score} / {max_score}</div>'
            f'</div>'
        )
    else:
        banner_html = (
            f'<div class="q-banner-open">'
            f'<div class="q-banner-title">{title}</div>'
            f'</div>'
        )
    label = f"{title} {'— Score: ' + str(score) + ' / ' + str(max_score) if already and score is not None else ''}"
    with st.expander(label, expanded=not already):
        st.markdown(banner_html, unsafe_allow_html=True)
        return True
    return False


# ════════════════════════════════════════════════════════════════════════════════
#  QUESTION 1 — BUDGET CONSTRAINT
# ════════════════════════════════════════════════════════════════════════════════

def _render_q1(q_config, hw_id, email, past_deadline, grace_active, submissions):
    I, Px, Py = _q1_params(email)
    ANS_x = r2(I/Px); ANS_y = r2(I/Py); ANS_s = r2(-Px/Py)

    prev     = _get_prev(submissions, hw_id, "Q1")
    already  = str(prev.get("Status","")) == "submitted"
    disabled = already or (past_deadline and not grace_active)
    score    = int(prev.get("Score", 0)) if already else None

    # ── Collapsible question banner ────────────────────────────────────────────
    banner_label = (
        f"Question 1 — Budget Constraint  ·  Score: {score} / 6"
        if already else "Question 1 — Budget Constraint"
    )
    with st.expander(banner_label, expanded=not already):

        # Question body — navy left border
        st.markdown(
            f'<div class="q-body">'
            f'<div class="q-pts">6 points &nbsp;·&nbsp; Parts (b) and (c) are graded</div>'
            f'<div class="q-text">A consumer has income <strong>I = ${I}</strong>, '
            f'price of good X is <strong>Px = ${Px}</strong>, '
            f'and price of good Y is <strong>Py = ${Py}</strong>.</div>'
            f'<div class="q-part"><span class="q-part-label">(a)</span> &nbsp;'
            f'Write the equation of the budget constraint. '
            f'<span class="q-ungraded">(Ungraded)</span></div>'
            f'<div class="q-part"><span class="q-part-label">(b) 4 points</span> &nbsp;'
            f'Find the X-intercept and Y-intercept of the budget line.</div>'
            f'<div class="q-part"><span class="q-part-label">(c) 2 points</span> &nbsp;'
            f'What is the slope of the budget line?</div>'
            f'<div class="q-part"><span class="q-part-label">(d)</span> &nbsp;'
            f'Draw the budget line on a clearly labelled diagram. '
            f'<span class="q-ungraded">(Ungraded — reference graph shown after submitting)</span></div>'
            f'</div>',
            unsafe_allow_html=True
        )

        # Parameters
        st.markdown(
            f'<div class="param-row">Your values: &nbsp;'
            f'<span class="param-val">I = ${I}</span> &nbsp;·&nbsp; '
            f'<span class="param-val">Px = ${Px}</span> &nbsp;·&nbsp; '
            f'<span class="param-val">Py = ${Py}</span></div>',
            unsafe_allow_html=True
        )

        # Status banners
        if already:
            ts   = prev.get("Timestamp","")
            late = " · Late submission" if prev.get("Is_Late","")=="Yes" else ""
            st.markdown(
                f'<div class="banner banner-locked">🔒 Submitted — '
                f'Score: <strong>{score} / 6</strong> · {ts}{late}</div>',
                unsafe_allow_html=True)
        elif past_deadline and not grace_active:
            st.markdown(
                '<div class="banner banner-error">🔒 Deadline has passed.</div>',
                unsafe_allow_html=True)
        elif not prev:
            st.markdown(
                '<div class="banner banner-info">📝 Enter your answers below '
                'and click Submit when ready. Submissions cannot be changed.</div>',
                unsafe_allow_html=True)
        else:
            st.markdown(
                '<div class="banner banner-restore">Draft restored from previous session.</div>',
                unsafe_allow_html=True)

        # ── Divider ────────────────────────────────────────────────────────────
        st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

        # ── Submitted answers (collapsible, default collapsed) ─────────────────
        if already:
            d  = _parse_raw(prev)
            xv = float(d.get("xint",0)); yv = float(d.get("yint",0)); sv = float(d.get("slope",0))
            with st.expander("📝 Your Submitted Answers", expanded=False):
                st.markdown(
                    f'<div class="submitted-ans">'
                    f'<div class="submitted-ans-label">Your Submitted Answers</div>'
                    f'(b) X-intercept: <strong>{r2(xv)}</strong><br>'
                    f'(b) Y-intercept: <strong>{r2(yv)}</strong><br>'
                    f'(c) Slope: <strong>{r2(sv)}</strong>'
                    f'</div>',
                    unsafe_allow_html=True
                )
            st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

        # ── Answer inputs ──────────────────────────────────────────────────────
        if not already:
            d = _parse_raw(prev)
            st.markdown(
                '<div class="ans-section"><div class="ans-label">Your Answers</div>',
                unsafe_allow_html=True
            )
            c1, c2, c3 = st.columns(3)
            with c1:
                xint_ans = st.number_input(
                    "(b) X-intercept of budget line",
                    value=float(d.get("xint",0.0)),
                    step=0.01, format="%.2f",
                    disabled=disabled, key=f"q1x_{hw_id}"
                )
            with c2:
                yint_ans = st.number_input(
                    "(b) Y-intercept of budget line",
                    value=float(d.get("yint",0.0)),
                    step=0.01, format="%.2f",
                    disabled=disabled, key=f"q1y_{hw_id}"
                )
            with c3:
                slope_ans = st.number_input(
                    "(c) Slope of budget line",
                    value=float(d.get("slope",0.0)),
                    step=0.01, format="%.2f",
                    disabled=disabled, key=f"q1s_{hw_id}"
                )
            st.caption("Enter answers rounded to 2 decimal places.")
            st.markdown('</div>', unsafe_allow_html=True)

            # Validation
            if xint_ans < 0 or yint_ans < 0:
                st.markdown(
                    '<div class="banner banner-warning">⚠ Intercepts should be '
                    'positive for a standard budget line. Please check your answer.</div>',
                    unsafe_allow_html=True)

            # Submit
            has_input = not (xint_ans==0.0 and yint_ans==0.0 and slope_ans==0.0)
            if not (past_deadline and not grace_active):
                if has_input:
                    confirm_key = f"confirm_q1_{hw_id}"
                    if st.checkbox(
                        "I am ready to submit — I understand this cannot be changed.",
                        key=confirm_key
                    ):
                        if st.button("Submit Question 1", key=f"sub_q1_{hw_id}",
                                     use_container_width=True):
                            _submit_q1(hw_id, email, xint_ans, yint_ans, slope_ans,
                                       ANS_x, ANS_y, ANS_s, past_deadline, submissions)
                            st.rerun()
                else:
                    st.caption("Fill in your answers above to enable submission.")

        # ── Solution ───────────────────────────────────────────────────────────
        if already:
            st.markdown('<hr class="section-divider">', unsafe_allow_html=True)
            with st.expander("Show / Hide Correct Solution", expanded=False):
                _show_q1_solution(prev, ANS_x, ANS_y, ANS_s, I, Px, Py)

    # Back to top
    if already:
        st.markdown(
            '<div style="text-align:right;margin-top:0.3rem;">'
            '<a href="#" style="font-size:0.88rem;color:#2563EB;text-decoration:none;">'
            '↑ Back to top</a></div>',
            unsafe_allow_html=True
        )


def _submit_q1(hw_id, email, xint_ans, yint_ans, slope_ans,
               ANS_x, ANS_y, ANS_s, past_deadline, submissions):
    x_ok = r2(xint_ans)==ANS_x; y_ok = r2(yint_ans)==ANS_y; s_ok = r2(slope_ans)==ANS_s
    sc   = 2*int(x_ok) + 2*int(y_ok) + 2*int(s_ok)
    ts   = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    late = "Yes" if past_deadline else "No"
    raw  = str({"xint":r2(xint_ans),"yint":r2(yint_ans),"slope":r2(slope_ans)})
    corr = str({"xint":ANS_x,"yint":ANS_y,"slope":ANS_s})
    ok, err = write_submission([ts,email,hw_id,"Q1","numerical","submitted",late,raw,sc,6,corr])
    log_submission_attempt(email, hw_id, "Q1", raw, sc, 6)
    st.session_state.setdefault("submissions",{}).setdefault(hw_id,{})["Q1"] = {
        "Status":"submitted","Timestamp":ts,"Score":sc,
        "Max_Score":6,"Is_Late":late,"Raw_Answer":raw}
    if ok:
        st.markdown(
            f'<div class="banner banner-success">✓ Question 1 submitted — '
            f'<strong>Score: {sc} / 6</strong></div>', unsafe_allow_html=True)
        st.markdown(f'<div class="saved-ts">Saved at {ts}</div>', unsafe_allow_html=True)
    else:
        st.markdown(
            f'<div class="banner banner-warning">⚠ Sheet write failed ({err}). '
            f'Screenshot this page. Score: {sc}/6 at {ts}</div>', unsafe_allow_html=True)


def _show_q1_solution(prev, ANS_x, ANS_y, ANS_s, I, Px, Py):
    d  = _parse_raw(prev)
    xv = float(d.get("xint",0)); yv = float(d.get("yint",0)); sv = float(d.get("slope",0))
    x_ok = r2(xv)==ANS_x; y_ok = r2(yv)==ANS_y; s_ok = r2(sv)==ANS_s
    sc   = 2*int(x_ok) + 2*int(y_ok) + 2*int(s_ok)

    st.markdown('<div class="sol-section"><div class="sol-title">Solution — Question 1</div>',
                unsafe_allow_html=True)

    st.markdown(
        f'<div class="sol-label">Step-by-step working</div>'
        f'<div class="sol-steps">'
        f'<p><strong>(a)</strong> Budget constraint: {Px}X + {Py}Y = {I}</p>'
        f'<p><strong>(b)</strong> X-intercept: set Y = 0 → X = {I} / {Px} = <strong>{ANS_x}</strong><br>'
        f'Y-intercept: set X = 0 → Y = {I} / {Py} = <strong>{ANS_y}</strong></p>'
        f'<p><strong>(c)</strong> Slope = −Px/Py = −{Px}/{Py} = <strong>{ANS_s}</strong></p>'
        f'</div>',
        unsafe_allow_html=True
    )

    x_chip = '<span class="chip-ok">+2</span>' if x_ok else '<span class="chip-wrong">0</span>'
    y_chip = '<span class="chip-ok">+2</span>' if y_ok else '<span class="chip-wrong">0</span>'
    s_chip = '<span class="chip-ok">+2</span>' if s_ok else '<span class="chip-wrong">0</span>'

    st.markdown('<div class="sol-label">Your score</div>', unsafe_allow_html=True)
    st.markdown(
        f'<div class="score-row"><span class="score-row-label">'
        f'(b) X-intercept &nbsp;·&nbsp; Correct: {ANS_x} &nbsp;·&nbsp; Yours: {r2(xv)}'
        f'</span><span class="score-row-val">{x_chip}</span></div>',
        unsafe_allow_html=True)
    st.markdown(
        f'<div class="score-row"><span class="score-row-label">'
        f'(b) Y-intercept &nbsp;·&nbsp; Correct: {ANS_y} &nbsp;·&nbsp; Yours: {r2(yv)}'
        f'</span><span class="score-row-val">{y_chip}</span></div>',
        unsafe_allow_html=True)
    st.markdown(
        f'<div class="score-row"><span class="score-row-label">'
        f'(c) Slope &nbsp;·&nbsp; Correct: {ANS_s} &nbsp;·&nbsp; Yours: {r2(sv)}'
        f'</span><span class="score-row-val">{s_chip}</span></div>',
        unsafe_allow_html=True)
    st.markdown(
        f'<div class="score-row score-row-total">'
        f'<span>Total</span><span>{sc} / 6</span></div>',
        unsafe_allow_html=True)

    st.markdown(
        '<div class="sol-label">Common mistakes</div>'
        '<div class="sol-mistakes">'
        '<p>Writing the slope as −Py/Px instead of −Px/Py. The slope tells you '
        'how many units of Y you give up per extra unit of X, which is −Px/Py.</p>'
        '<p>Confusing a parallel shift (income change) with a pivot (price change).</p>'
        '</div>',
        unsafe_allow_html=True)

    revise = []
    if not x_ok or not y_ok:
        revise.append("Finding intercepts — set Y=0 for X-intercept, set X=0 for Y-intercept")
    if not s_ok:
        revise.append("Budget line slope — why slope = −Px/Py and its economic meaning")
    if revise:
        items = "".join(f"<li style='margin-bottom:0.4rem;'>{r}</li>" for r in revise)
        st.markdown(
            f'<div class="sol-label">Topics to revise</div>'
            f'<div class="sol-revise"><ul style="margin:0;padding-left:1.2rem;">'
            f'{items}</ul></div>',
            unsafe_allow_html=True)

    # Graph
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    fig.patch.set_facecolor("#F0F4F8"); ax.set_facecolor("#F0F4F8")
    Xmax = I/Px; Ymax = I/Py
    Xv   = np.linspace(0, Xmax, 300); Yv = (I-Px*Xv)/Py
    ax.plot(Xv, Yv, color="#1C2B4A", lw=2.5)
    ax.fill_between(Xv, Yv, alpha=0.08, color="#1C2B4A")
    ax.plot(Xmax, 0, "o", color="#1C2B4A", ms=8, zorder=5)
    ax.plot(0, Ymax, "o", color="#1C2B4A", ms=8, zorder=5)
    ax.annotate(f"({Xmax:.2f}, 0)", xy=(Xmax,0),
                xytext=(Xmax-Xmax*0.3, Ymax*0.08), fontsize=10, color="#1C2B4A",
                arrowprops=dict(arrowstyle="->", color="#1C2B4A", lw=1.0))
    ax.annotate(f"(0, {Ymax:.2f})", xy=(0,Ymax),
                xytext=(Xmax*0.07, Ymax-Ymax*0.15), fontsize=10, color="#1C2B4A",
                arrowprops=dict(arrowstyle="->", color="#1C2B4A", lw=1.0))
    ax.set_xlabel("Quantity of X", fontsize=11); ax.set_ylabel("Quantity of Y", fontsize=11)
    ax.set_title(f"Budget Line: {Px}X + {Py}Y = {I}", fontsize=11, color="#1C2B4A")
    ax.set_xlim(0, Xmax*1.22); ax.set_ylim(0, Ymax*1.28)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    ax.tick_params(labelsize=10)
    plt.tight_layout()
    b64 = fig_to_b64(fig)

    st.markdown(
        f'<div class="sol-label">Reference diagram (part d)</div>'
        f'<div style="text-align:center;padding:0.5rem 0;">'
        f'<img src="data:image/png;base64,{b64}" '
        f'style="max-width:400px;width:100%;border-radius:8px;'
        f'border:1px solid #E0E0E0;"></div>',
        unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════════════════
#  QUESTION 2 — TOM & JERRY
# ════════════════════════════════════════════════════════════════════════════════

def _render_q2(q_config, hw_id, email, past_deadline, grace_active, submissions):
    I, Px, Py, tom_a = _q2_params(email)
    ANS_tx = r2(I/Px); ANS_ty = 0.0
    ANS_jx = r2(I/(Px+Py)); ANS_jy = r2(I/(Px+Py))

    prev     = _get_prev(submissions, hw_id, "Q2")
    already  = str(prev.get("Status","")) == "submitted"
    disabled = already or (past_deadline and not grace_active)
    score    = int(prev.get("Score", 0)) if already else None

    banner_label = (
        f"Question 2 — Tom & Jerry  ·  Score: {score} / 8"
        if already else "Question 2 — Tom & Jerry"
    )
    with st.expander(banner_label, expanded=not already):

        st.markdown(
            f'<div class="q-body">'
            f'<div class="q-pts">8 points &nbsp;·&nbsp; 4 points each for parts (a) and (b)</div>'
            f'<div class="q-text">Tom and Jerry both have income <strong>I = ${I}</strong>, '
            f'<strong>Px = ${Px}</strong>, <strong>Py = ${Py}</strong>, '
            f'but very different preferences:</div>'
            f'<div class="q-part"><strong>Tom:</strong> &nbsp;U = {tom_a}X + Y &nbsp;'
            f'(perfect substitutes)</div>'
            f'<div class="q-part"><strong>Jerry:</strong> &nbsp;U = min(X, Y) &nbsp;'
            f'(perfect complements — always wants X and Y in equal amounts)</div>'
            f'<br>'
            f'<div class="q-part"><span class="q-part-label">(a) 4 points</span> &nbsp;'
            f'Find Tom\'s optimal bundle (X*, Y*).</div>'
            f'<div class="q-part"><span class="q-part-label">(b) 4 points</span> &nbsp;'
            f'Find Jerry\'s optimal bundle (X*, Y*).</div>'
            f'<div class="q-part"><span class="q-part-label">(c)</span> &nbsp;'
            f'Explain why their bundles differ so dramatically. '
            f'<span class="q-ungraded">(Ungraded)</span></div>'
            f'</div>',
            unsafe_allow_html=True)

        st.markdown(
            f'<div class="param-row">Your values: &nbsp;'
            f'<span class="param-val">I = ${I}</span> &nbsp;·&nbsp; '
            f'<span class="param-val">Px = ${Px}</span> &nbsp;·&nbsp; '
            f'<span class="param-val">Py = ${Py}</span> &nbsp;·&nbsp; '
            f'<span class="param-val">Tom: U = {tom_a}X + Y</span></div>',
            unsafe_allow_html=True)

        if already:
            ts = prev.get("Timestamp","")
            late = " · Late submission" if prev.get("Is_Late","")=="Yes" else ""
            st.markdown(
                f'<div class="banner banner-locked">🔒 Submitted — '
                f'Score: <strong>{score} / 8</strong> · {ts}{late}</div>',
                unsafe_allow_html=True)
        elif past_deadline and not grace_active:
            st.markdown('<div class="banner banner-error">🔒 Deadline has passed.</div>',
                        unsafe_allow_html=True)
        elif not prev:
            st.markdown(
                '<div class="banner banner-info">📝 Enter your answers below '
                'and click Submit when ready. Submissions cannot be changed.</div>',
                unsafe_allow_html=True)
        else:
            st.markdown(
                '<div class="banner banner-restore">Draft restored from previous session.</div>',
                unsafe_allow_html=True)

        st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

        if already:
            d  = _parse_raw(prev)
            tx = float(d.get("tom_x",0)); ty = float(d.get("tom_y",0))
            jx = float(d.get("jerry_x",0)); jy = float(d.get("jerry_y",0))
            with st.expander("📝 Your Submitted Answers", expanded=False):
                st.markdown(
                    f'<div class="submitted-ans">'
                    f'<div class="submitted-ans-label">Your Submitted Answers</div>'
                    f'(a) Tom\'s bundle: X* = <strong>{r2(tx)}</strong>, '
                    f'Y* = <strong>{r2(ty)}</strong><br>'
                    f'(b) Jerry\'s bundle: X* = <strong>{r2(jx)}</strong>, '
                    f'Y* = <strong>{r2(jy)}</strong>'
                    f'</div>',
                    unsafe_allow_html=True)
            st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

        if not already:
            d = _parse_raw(prev)
            st.markdown(
                '<div class="ans-section"><div class="ans-label">Your Answers</div>',
                unsafe_allow_html=True)
            st.markdown("**(a) Tom's optimal bundle:**")
            c1, c2 = st.columns(2)
            with c1:
                tom_x = st.number_input("Tom's optimal quantity of X",
                                        value=float(d.get("tom_x",0.0)),
                                        step=0.01, format="%.2f",
                                        disabled=disabled, key=f"q2tx_{hw_id}")
            with c2:
                tom_y = st.number_input("Tom's optimal quantity of Y",
                                        value=float(d.get("tom_y",0.0)),
                                        step=0.01, format="%.2f",
                                        disabled=disabled, key=f"q2ty_{hw_id}")
            st.markdown("**(b) Jerry's optimal bundle:**")
            c3, c4 = st.columns(2)
            with c3:
                jerry_x = st.number_input("Jerry's optimal quantity of X",
                                          value=float(d.get("jerry_x",0.0)),
                                          step=0.01, format="%.2f",
                                          disabled=disabled, key=f"q2jx_{hw_id}")
            with c4:
                jerry_y = st.number_input("Jerry's optimal quantity of Y",
                                          value=float(d.get("jerry_y",0.0)),
                                          step=0.01, format="%.2f",
                                          disabled=disabled, key=f"q2jy_{hw_id}")
            st.caption("Enter answers rounded to 2 decimal places.")
            st.markdown('</div>', unsafe_allow_html=True)

            jxr = r2(jerry_x); jyr = r2(jerry_y)
            if abs(jxr - jyr) > 0.05 and (jerry_x != 0.0 or jerry_y != 0.0):
                st.markdown(
                    '<div class="banner banner-info">💡 Hint for Jerry: '
                    'with U = min(X, Y), the optimal bundle always satisfies X* = Y*. '
                    'Substitute into the budget constraint to find the values.</div>',
                    unsafe_allow_html=True)

            has_input = any(v != 0.0 for v in [tom_x, tom_y, jerry_x, jerry_y])
            if not (past_deadline and not grace_active):
                if has_input:
                    if st.checkbox(
                        "I am ready to submit — I understand this cannot be changed.",
                        key=f"confirm_q2_{hw_id}"
                    ):
                        if st.button("Submit Question 2", key=f"sub_q2_{hw_id}",
                                     use_container_width=True):
                            _submit_q2(hw_id, email, tom_x, tom_y, jerry_x, jerry_y,
                                       ANS_tx, ANS_ty, ANS_jx, ANS_jy, past_deadline)
                            st.rerun()
                else:
                    st.caption("Fill in your answers above to enable submission.")

        if already:
            st.markdown('<hr class="section-divider">', unsafe_allow_html=True)
            with st.expander("Show / Hide Correct Solution", expanded=False):
                _show_q2_solution(prev, ANS_tx, ANS_ty, ANS_jx, ANS_jy, I, Px, Py, tom_a)

    if already:
        st.markdown(
            '<div style="text-align:right;margin-top:0.3rem;">'
            '<a href="#" style="font-size:0.88rem;color:#2563EB;text-decoration:none;">'
            '↑ Back to top</a></div>',
            unsafe_allow_html=True)


def _submit_q2(hw_id, email, tom_x, tom_y, jerry_x, jerry_y,
               ANS_tx, ANS_ty, ANS_jx, ANS_jy, past_deadline):
    tok  = (r2(tom_x)==ANS_tx and r2(tom_y)==ANS_ty)
    jok  = (r2(jerry_x)==ANS_jx and r2(jerry_y)==ANS_jy)
    sc   = 4*int(tok) + 4*int(jok)
    ts   = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    late = "Yes" if past_deadline else "No"
    raw  = str({"tom_x":r2(tom_x),"tom_y":r2(tom_y),
                "jerry_x":r2(jerry_x),"jerry_y":r2(jerry_y)})
    corr = str({"tom":f"({ANS_tx},{ANS_ty})","jerry":f"({ANS_jx},{ANS_jy})"})
    ok, err = write_submission([ts,email,hw_id,"Q2","numerical","submitted",late,raw,sc,8,corr])
    log_submission_attempt(email, hw_id, "Q2", raw, sc, 8)
    st.session_state.setdefault("submissions",{}).setdefault(hw_id,{})["Q2"] = {
        "Status":"submitted","Timestamp":ts,"Score":sc,
        "Max_Score":8,"Is_Late":late,"Raw_Answer":raw}
    if ok:
        st.markdown(
            f'<div class="banner banner-success">✓ Question 2 submitted — '
            f'<strong>Score: {sc} / 8</strong></div>', unsafe_allow_html=True)
        st.markdown(f'<div class="saved-ts">Saved at {ts}</div>', unsafe_allow_html=True)
    else:
        st.markdown(
            f'<div class="banner banner-warning">⚠ Sheet write failed ({err}). '
            f'Screenshot this page. Score: {sc}/8 at {ts}</div>', unsafe_allow_html=True)


def _show_q2_solution(prev, ANS_tx, ANS_ty, ANS_jx, ANS_jy, I, Px, Py, tom_a):
    d   = _parse_raw(prev)
    tx  = float(d.get("tom_x",0)); ty = float(d.get("tom_y",0))
    jx  = float(d.get("jerry_x",0)); jy = float(d.get("jerry_y",0))
    tok = (r2(tx)==ANS_tx and r2(ty)==ANS_ty)
    jok = (r2(jx)==ANS_jx and r2(jy)==ANS_jy)
    sc  = 4*int(tok) + 4*int(jok)

    st.markdown('<div class="sol-section"><div class="sol-title">Solution — Question 2</div>',
                unsafe_allow_html=True)
    st.markdown(
        f'<div class="sol-label">Step-by-step working</div>'
        f'<div class="sol-steps">'
        f'<p><strong>(a) Tom — Perfect Substitutes</strong><br>'
        f'Compare utility per dollar: MUx/Px = {tom_a}/{Px} = {round(tom_a/Px,2)} '
        f'vs MUy/Py = 1/{Py} = {round(1/Py,2)}<br>'
        f'X gives more utility per dollar → Tom spends all income on X:<br>'
        f'X* = {I}/{Px} = <strong>{ANS_tx}</strong>, Y* = <strong>0</strong></p>'
        f'<p><strong>(b) Jerry — Perfect Complements</strong><br>'
        f'Kink condition: X* = Y*. Substitute into budget:<br>'
        f'{Px}X + {Py}X = {I} → {Px+Py}X = {I} → '
        f'X* = Y* = <strong>{ANS_jx}</strong></p>'
        f'</div>',
        unsafe_allow_html=True)

    t_chip = '<span class="chip-ok">+4</span>' if tok else '<span class="chip-wrong">0</span>'
    j_chip = '<span class="chip-ok">+4</span>' if jok else '<span class="chip-wrong">0</span>'

    st.markdown('<div class="sol-label">Your score</div>', unsafe_allow_html=True)
    st.markdown(
        f'<div class="score-row"><span class="score-row-label">'
        f'(a) Tom &nbsp;·&nbsp; Correct: ({ANS_tx}, {ANS_ty}) &nbsp;·&nbsp; '
        f'Yours: ({r2(tx)}, {r2(ty)})</span>'
        f'<span class="score-row-val">{t_chip}</span></div>',
        unsafe_allow_html=True)
    st.markdown(
        f'<div class="score-row"><span class="score-row-label">'
        f'(b) Jerry &nbsp;·&nbsp; Correct: ({ANS_jx}, {ANS_jy}) &nbsp;·&nbsp; '
        f'Yours: ({r2(jx)}, {r2(jy)})</span>'
        f'<span class="score-row-val">{j_chip}</span></div>',
        unsafe_allow_html=True)
    st.markdown(
        f'<div class="score-row score-row-total">'
        f'<span>Total</span><span>{sc} / 8</span></div>',
        unsafe_allow_html=True)

    st.markdown(
        '<div class="sol-label">Common mistakes</div>'
        '<div class="sol-mistakes">'
        '<p><strong>Tom:</strong> Never use MRS = Px/Py for perfect substitutes. '
        'Always compare MUx/Px vs MUy/Py directly.</p>'
        '<p><strong>Jerry:</strong> X* = Y* is the condition — you must then '
        'substitute into the budget constraint to get the actual values.</p>'
        '</div>',
        unsafe_allow_html=True)

    revise = []
    if not tok:
        revise.append("Perfect substitutes — bang-per-buck: compare MUx/Px vs MUy/Py")
    if not jok:
        revise.append("Perfect complements — kink condition X* = Y* and budget substitution")
    if revise:
        items = "".join(f"<li style='margin-bottom:0.4rem;'>{r}</li>" for r in revise)
        st.markdown(
            f'<div class="sol-label">Topics to revise</div>'
            f'<div class="sol-revise"><ul style="margin:0;padding-left:1.2rem;">'
            f'{items}</ul></div>',
            unsafe_allow_html=True)

    fig, axs = plt.subplots(1, 2, figsize=(12, 5))
    fig.patch.set_facecolor("#F0F4F8")
    fig.suptitle(f"Tom vs Jerry  [I={I}, Px={Px}, Py={Py}]",
                 fontsize=11, color="#1C2B4A", fontweight="600")
    Xv = np.linspace(0, I/Px+1, 400); Yv = (I-Px*Xv)/Py
    for ax in axs:
        ax.set_facecolor("#F0F4F8")
        ax.plot(Xv, np.where(Yv>=0,Yv,np.nan), color="#1C2B4A", lw=2.2)
        ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
        ax.tick_params(labelsize=10)
        ax.set_xlabel("X", fontsize=11); ax.set_ylabel("Y", fontsize=11)
    Xt=I/Px; Xj=I/(Px+Py); Yj=Xj
    ax=axs[0]
    UT=tom_a*Xt
    for Ul,alp in [(UT*0.6,0.18),(UT*0.8,0.32),(UT,0.85)]:
        ICs=(Ul-tom_a*Xv)/1.0
        ax.plot(Xv,np.where(ICs>=0,ICs,np.nan),color="#DC2626",lw=1.5,alpha=alp)
    ax.plot(Xt,0,"o",color="#DC2626",ms=9,zorder=6,label=f"Optimum ({ANS_tx}, 0)")
    ax.set_title(f"Tom: U = {tom_a}X + Y\n(Perfect Substitutes)",fontsize=10,color="#1C2B4A")
    ax.set_xlim(0,I/Px*1.2); ax.set_ylim(0,I/Py*1.3); ax.legend(fontsize=9)
    ax=axs[1]
    for Ul,alp in [(Xj*0.6,0.18),(Xj*0.8,0.32),(Xj,0.85)]:
        ax.plot([Ul,I/Px*1.2],[Ul,Ul],color="#DC2626",lw=1.5,alpha=alp)
        ax.plot([Ul,Ul],[Ul,I/Py*1.3],color="#DC2626",lw=1.5,alpha=alp)
    diag=np.linspace(0,min(I/Px,I/Py)*1.1,100)
    ax.plot(diag,diag,"--",color="#9CA3AF",lw=1.1,alpha=0.6)
    ax.plot(Xj,Yj,"o",color="#DC2626",ms=9,zorder=6,label=f"Optimum ({ANS_jx}, {ANS_jy})")
    ax.set_title("Jerry: U = min(X, Y)\n(Perfect Complements)",fontsize=10,color="#1C2B4A")
    ax.set_xlim(0,I/Px*1.2); ax.set_ylim(0,I/Py*1.3); ax.legend(fontsize=9)
    plt.tight_layout()
    b64 = fig_to_b64(fig)
    st.markdown(
        f'<div class="sol-label">Reference diagrams</div>'
        f'<div style="text-align:center;padding:0.5rem 0;">'
        f'<img src="data:image/png;base64,{b64}" '
        f'style="max-width:660px;width:100%;border-radius:8px;'
        f'border:1px solid #E0E0E0;"></div>',
        unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════════════════
#  QUESTION 3 — TRUE/FALSE
# ════════════════════════════════════════════════════════════════════════════════

def _render_q3_tf(q_config, hw_id, email, past_deadline, grace_active, submissions):
    statements = TF_STATEMENTS
    correct    = [s["correct"] for s in statements]

    prev     = _get_prev(submissions, hw_id, "Q3")
    already  = str(prev.get("Status","")) == "submitted"
    disabled = already or (past_deadline and not grace_active)
    score    = int(prev.get("Score", 0)) if already else None

    banner_label = (
        f"Question 3 — True or False  ·  Score: {score} / 4"
        if already else "Question 3 — True or False"
    )
    with st.expander(banner_label, expanded=not already):

        st.markdown(
            '<div class="q-body">'
            '<div class="q-pts">4 points &nbsp;·&nbsp; 1 point per statement</div>'
            '<div class="q-text">Read each statement carefully and decide whether '
            'it is True or False. These test core concepts from the first two weeks.</div>'
            '</div>',
            unsafe_allow_html=True)

        if already:
            ts = prev.get("Timestamp","")
            late = " · Late submission" if prev.get("Is_Late","")=="Yes" else ""
            st.markdown(
                f'<div class="banner banner-locked">🔒 Submitted — '
                f'Score: <strong>{score} / 4</strong> · {ts}{late}</div>',
                unsafe_allow_html=True)
        elif past_deadline and not grace_active:
            st.markdown('<div class="banner banner-error">🔒 Deadline has passed.</div>',
                        unsafe_allow_html=True)
        elif not prev:
            st.markdown(
                '<div class="banner banner-info">📝 Select True or False for each '
                'statement, then click Submit.</div>',
                unsafe_allow_html=True)

        st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

        # Submitted answers
        if already:
            prev_answers = {}
            raw = prev.get("Raw_Answer","")
            if raw:
                try:
                    pa = eval(str(raw))
                    if isinstance(pa, dict): prev_answers = pa
                except Exception: pass
            with st.expander("📝 Your Submitted Answers", expanded=False):
                st.markdown(
                    '<div class="submitted-ans">'
                    '<div class="submitted-ans-label">Your Submitted Answers</div>',
                    unsafe_allow_html=True)
                for i, stmt in enumerate(statements):
                    student_val = prev_answers.get(f"s{i}", None)
                    student_str = ("True" if student_val else "False") if student_val is not None else "—"
                    st.markdown(
                        f'<div style="padding:0.3rem 0;font-size:1rem;">'
                        f'({i+1}) {stmt["text"][:60]}... &nbsp;→&nbsp; '
                        f'<strong>{student_str}</strong></div>',
                        unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)
            st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

        # Answer inputs
        prev_answers = {}
        raw = prev.get("Raw_Answer","")
        if raw:
            try:
                pa = eval(str(raw))
                if isinstance(pa, dict): prev_answers = pa
            except Exception: pass

        st.markdown(
            '<div class="ans-section"><div class="ans-label">Your Answers</div>',
            unsafe_allow_html=True)

        student_answers = []
        for i, stmt in enumerate(statements):
            prev_val = prev_answers.get(f"s{i}", None)
            default_idx = 0 if (prev_val is True or prev_val is None) else 1
            st.markdown(
                f'<div style="font-size:1rem;font-weight:500;color:#1A1A1A;'
                f'margin:1rem 0 0.3rem 0;">({i+1}) {stmt["text"]}</div>',
                unsafe_allow_html=True)
            choice = st.radio(
                f"Statement {i+1}", options=["True","False"],
                index=default_idx, disabled=disabled,
                horizontal=True, key=f"tf_{i}_{hw_id}",
                label_visibility="collapsed")
            student_answers.append(choice == "True")

        st.markdown('</div>', unsafe_allow_html=True)

        if not already and not (past_deadline and not grace_active):
            if st.checkbox(
                "I am ready to submit — I understand this cannot be changed.",
                key=f"confirm_q3_{hw_id}"
            ):
                if st.button("Submit Question 3", key=f"sub_q3tf_{hw_id}",
                             use_container_width=True):
                    _submit_q3_tf(hw_id, email, student_answers, correct, past_deadline)
                    st.rerun()

        if already:
            st.markdown('<hr class="section-divider">', unsafe_allow_html=True)
            with st.expander("Show / Hide Correct Solution", expanded=False):
                _show_q3_tf_solution(prev, statements, correct)

    if already:
        st.markdown(
            '<div style="text-align:right;margin-top:0.3rem;">'
            '<a href="#" style="font-size:0.88rem;color:#2563EB;text-decoration:none;">'
            '↑ Back to top</a></div>',
            unsafe_allow_html=True)


def _submit_q3_tf(hw_id, email, student_answers, correct, past_deadline):
    sc   = sum(1 for s,c in zip(student_answers,correct) if s==c)
    ts   = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    late = "Yes" if past_deadline else "No"
    raw  = str({f"s{i}":v for i,v in enumerate(student_answers)})
    corr = str({f"s{i}":v for i,v in enumerate(correct)})
    ok, err = write_submission([ts,email,hw_id,"Q3","truefalse","submitted",late,raw,sc,4,corr])
    log_submission_attempt(email, hw_id, "Q3", raw, sc, 4)
    st.session_state.setdefault("submissions",{}).setdefault(hw_id,{})["Q3"] = {
        "Status":"submitted","Timestamp":ts,"Score":sc,
        "Max_Score":4,"Is_Late":late,"Raw_Answer":raw}
    if ok:
        st.markdown(
            f'<div class="banner banner-success">✓ Question 3 submitted — '
            f'<strong>Score: {sc} / 4</strong></div>', unsafe_allow_html=True)
        st.markdown(f'<div class="saved-ts">Saved at {ts}</div>', unsafe_allow_html=True)
    else:
        st.markdown(
            f'<div class="banner banner-warning">⚠ Sheet write failed ({err}). '
            f'Screenshot this page. Score: {sc}/4 at {ts}</div>', unsafe_allow_html=True)


def _show_q3_tf_solution(prev, statements, correct):
    prev_answers = {}
    raw = prev.get("Raw_Answer","")
    if raw:
        try:
            pa = eval(str(raw))
            if isinstance(pa, dict): prev_answers = pa
        except Exception: pass

    student_answers = [prev_answers.get(f"s{i}", None) for i in range(len(statements))]
    sc = sum(1 for s,c in zip(student_answers,correct)
             if s is not None and s==c)

    st.markdown(
        f'<div class="sol-section">'
        f'<div class="sol-title">Solution — Question 3 &nbsp; ({sc} / 4)</div>',
        unsafe_allow_html=True)

    revise_items = []
    for i, (stmt, corr_val, student_val) in enumerate(
            zip(statements, correct, student_answers)):
        corr_str    = "True"  if corr_val   else "False"
        student_str = ("True" if student_val else "False") if student_val is not None else "—"
        is_ok       = (student_val == corr_val) if student_val is not None else False
        icon        = "✓" if is_ok else "✗"
        col         = "#16A34A" if is_ok else "#DC2626"
        if not is_ok and stmt.get("topic"):
            revise_items.append(stmt["topic"])
        st.markdown(
            f'<div class="tf-item">'
            f'<div class="tf-stmt">({i+1}) {stmt["text"]}</div>'
            f'<div class="tf-result">'
            f'<span style="color:{col};font-weight:700;">{icon}</span> &nbsp;'
            f'<strong>Correct answer: {corr_str}</strong> &nbsp;·&nbsp; '
            f'Your answer: {student_str}</div>'
            f'<div class="tf-expl">{stmt["explanation"]}</div>'
            f'</div>',
            unsafe_allow_html=True)

    if revise_items:
        items = "".join(f"<li style='margin-bottom:0.4rem;'>{r}</li>" for r in revise_items)
        st.markdown(
            f'<div class="sol-label" style="margin-top:1rem;">Topics to revise</div>'
            f'<div class="sol-revise"><ul style="margin:0;padding-left:1.2rem;">'
            f'{items}</ul></div>',
            unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════════════════
#  WEEK 1 — TRUE/FALSE STATEMENTS
# ════════════════════════════════════════════════════════════════════════════════

W1_TF_Q1 = [
    {
        "text": "A student who attends a free campus workshop is incurring no opportunity cost because no money changes hands.",
        "correct": False,
        "explanation": (
            "This statement is False. Even though no money is paid, the student gives up whatever "
            "they could have done with that time — studying, working, or resting. Opportunity cost "
            "is the value of the best foregone alternative, not the amount of money spent. "
            "Time is a scarce resource, so nothing is truly free."
        ),
        "topic": "Opportunity cost — time has value even when no money is spent",
    },
    {
        "text": "If you have already paid a non-refundable $50 registration fee for a conference you no longer want to attend, that $50 should not influence your decision about whether to go.",
        "correct": True,
        "explanation": (
            "This statement is True. The $50 is a sunk cost — it has already been paid and cannot "
            "be recovered whether you attend or not. The rational decision depends only on whether "
            "the future benefit of attending exceeds the future opportunity cost of your time. "
            "Letting the $50 influence your decision is the sunk cost fallacy."
        ),
        "topic": "Sunk costs should not affect forward-looking decisions",
    },
    {
        "text": "The opportunity cost of a decision is always equal to the amount of money spent on it.",
        "correct": False,
        "explanation": (
            "This statement is False. Opportunity cost is the value of the best alternative foregone, "
            "which may have nothing to do with money spent. For example, attending a free lecture "
            "has an opportunity cost equal to the value of your next best use of that time, "
            "even though no money is exchanged."
        ),
        "topic": "Opportunity cost is not the same as out-of-pocket cost",
    },
    {
        "text": "A rational decision-maker should ignore sunk costs when choosing between future options.",
        "correct": True,
        "explanation": (
            "This statement is True. Sunk costs are gone regardless of what you decide next. "
            "A rational decision-maker looks only at future costs and future benefits. "
            "Including sunk costs distorts the decision and leads to systematically worse choices — "
            "this cognitive error is called the sunk cost fallacy."
        ),
        "topic": "Rational decision-making and sunk costs",
    },
]

W1_TF_Q5 = [
    {
        "text": "The completeness assumption means that a consumer must always be able to rank any two bundles or declare themselves indifferent between them.",
        "correct": True,
        "explanation": (
            "This statement is True. Completeness requires that for any two bundles A and B, "
            "the consumer can say either A is preferred to B, B is preferred to A, or they are "
            "indifferent. Saying 'I cannot decide' is not allowed under this assumption. "
            "Completeness ensures the consumer can always make a choice."
        ),
        "topic": "Completeness assumption on preferences",
    },
    {
        "text": "If a consumer prefers bundle A to B and bundle B to C, transitivity allows them to prefer C to A.",
        "correct": False,
        "explanation": (
            "This statement is False. Transitivity requires the opposite: if A is preferred to B "
            "and B is preferred to C, then A must be preferred to C. Preferring C to A would "
            "create a circular ranking, which violates transitivity and makes the consumer's "
            "preferences internally inconsistent."
        ),
        "topic": "Transitivity assumption on preferences",
    },
    {
        "text": "According to monotonicity, a consumer always prefers a bundle with more of both goods over one with less of both goods.",
        "correct": True,
        "explanation": (
            "This statement is True. Monotonicity — also called 'more is better' — states that "
            "if bundle F has at least as much of both goods as bundle A, and strictly more of "
            "at least one, then F is preferred to A. This assumption means consumers always "
            "find additional quantities of goods desirable."
        ),
        "topic": "Monotonicity assumption on preferences",
    },
    {
        "text": "A consumer who says 'I cannot choose between these two options' is violating the transitivity assumption.",
        "correct": False,
        "explanation": (
            "This statement is False. Being unable to choose between two options — or declaring "
            "indifference — violates the completeness assumption, not transitivity. Transitivity "
            "is about consistency across three or more bundles. Completeness is about always "
            "being able to rank any pair of bundles."
        ),
        "topic": "Completeness vs transitivity — knowing which assumption applies",
    },
]


# ════════════════════════════════════════════════════════════════════════════════
#  WEEK 1 — PARAMETER GENERATORS (with validation)
# ════════════════════════════════════════════════════════════════════════════════

def _validate_int_params(vals: list, positive: bool = True) -> bool:
    """Returns True if all values are clean integers and positive (if required)."""
    for v in vals:
        if not isinstance(v, (int, float)): return False
        if v != int(v): return False
        if positive and v <= 0: return False
    return True


def _w1_q2_params(email: str):
    """Q2 — Slopes. Two points for rise-over-run, one equation for slope-intercept."""
    rng = np.random.default_rng(get_seed(email) + 10)
    for _ in range(100):
        x1 = int(rng.integers(1, 5))
        y1 = int(rng.integers(4, 10))
        x2 = int(rng.integers(x1 + 2, x1 + 6))
        y2 = int(rng.integers(1, y1 - 1))  # ensure negative slope
        # slope from points
        slope_pts = (y2 - y1) / (x2 - x1)
        # equation: aY + bX = c → slope = -b/a
        a = int(rng.integers(2, 5))
        b = int(rng.integers(1, 4))
        c = int(rng.integers(8, 20))
        slope_eq = r2(-b / a)
        # check slopes match (same line) for part (d)
        # They don't need to match — Q asks students to compare them
        if slope_pts == slope_eq:
            continue  # want them different so comparison is interesting
        if _validate_int_params([x1, y1, x2, y2, a, b, c]):
            return x1, y1, x2, y2, a, b, c, r2(slope_pts), slope_eq
    # fallback
    return 1, 8, 5, 4, 4, 2, 20, r2((4 - 8) / (5 - 1)), r2(-2 / 4)


def _w1_q3_params(email: str):
    """Q3 — Derivatives. Fixed utility function, randomised evaluation point."""
    rng = np.random.default_rng(get_seed(email) + 11)
    for _ in range(100):
        X = int(rng.integers(2, 6))
        Y = int(rng.integers(2, 9))
        # U = X^(1/3) * Y^(2/3)
        # MUx = (1/3) * X^(-2/3) * Y^(2/3)
        # MUy = (2/3) * X^(1/3) * Y^(-1/3)
        mux = r2((1/3) * (X ** (-2/3)) * (Y ** (2/3)))
        muy = r2((2/3) * (X ** (1/3)) * (Y ** (-1/3)))
        if mux > 0 and muy > 0 and X > 0 and Y > 0:
            return X, Y, mux, muy
    return 4, 8, r2((1/3)*(4**(-2/3))*(8**(2/3))), r2((2/3)*(4**(1/3))*(8**(-1/3)))


def _w1_q4_params(email: str):
    """Q4 — Supply and demand equilibrium with clean integer solutions."""
    rng = np.random.default_rng(get_seed(email) + 12)
    for _ in range(200):
        # Demand: Qd = A - aP,  Supply: Qs = B + bP
        A = int(rng.integers(80, 150))
        a = int(rng.integers(2, 5))
        B = int(rng.integers(10, 40))
        b = int(rng.integers(2, 5))
        # Equilibrium: A - aP = B + bP → P* = (A-B)/(a+b)
        if (A - B) % (a + b) != 0:
            continue
        P_star = (A - B) // (a + b)
        Q_star = A - a * P_star
        if P_star <= 0 or Q_star <= 0:
            continue
        # Off-equilibrium price — ensure surplus/shortage is clean
        P_off = P_star - int(rng.integers(3, 8))
        if P_off <= 0:
            continue
        Qd_off = A - a * P_off
        Qs_off = B + b * P_off
        shortage = Qd_off - Qs_off
        if shortage <= 0:
            continue
        # New supply after technology shift: Qs_new = (B + shift) + bP
        shift = int(rng.integers(10, 25))
        # New equilibrium
        if (A - (B + shift)) % (a + b) != 0:
            continue
        P_new = (A - (B + shift)) // (a + b)
        Q_new = A - a * P_new
        if P_new <= 0 or Q_new <= 0:
            continue
        if _validate_int_params([A, a, B, b, P_star, Q_star, P_off, shortage, shift, P_new, Q_new]):
            return A, a, B, b, P_star, Q_star, P_off, Qd_off, Qs_off, shortage, shift, B + shift, P_new, Q_new
    # fallback
    return 120, 3, 20, 2, 20, 60, 15, 75, 50, 25, 15, 35, 17, 55


def _w1_q6_params(email: str):
    """Q6 — Indifference curves. U = XY. Three bundles, two on same IC."""
    rng = np.random.default_rng(get_seed(email) + 13)
    for _ in range(100):
        # Bundle A and B on same IC: XA*YA = XB*YB
        XA = int(rng.integers(2, 6))
        YA = int(rng.integers(4, 10))
        U_AB = XA * YA
        # Find XB such that XB*YB = U_AB with clean YB
        XB_opts = [x for x in range(XA + 1, 10) if U_AB % x == 0]
        if not XB_opts:
            continue
        XB = int(rng.choice(XB_opts))
        YB = U_AB // XB
        if YB <= 0 or YB == YA:
            continue
        # Bundle C on different IC
        XC = int(rng.integers(1, XA))
        YC = int(rng.integers(2, YA - 1))
        U_C = XC * YC
        if U_C == U_AB:
            continue
        # Bundle D = (XA+1, YA+1) — dominates A by monotonicity (ungraded)
        # Bundle E = (XA-1, YA) for comparison
        XD = XA + 1; YD = YA + 1
        XE = XA - 1 if XA > 1 else 1; YE = YA
        if _validate_int_params([XA, YA, XB, YB, XC, YC]):
            return XA, YA, XB, YB, XC, YC, U_AB, U_C, XD, YD, XE, YE
    return 2, 8, 4, 4, 1, 6, 16, 6, 3, 9, 1, 8


def _w1_q7_params(email: str):
    """Q7 — Marginal utility. U = X^(1/2) + 2Y."""
    rng = np.random.default_rng(get_seed(email) + 14)
    for _ in range(100):
        # Choose X as perfect square for clean sqrt
        X_opts = [1, 4, 9, 16, 25]
        X = int(rng.choice(X_opts))
        Y = int(rng.integers(2, 8))
        # MUx = 1/(2*sqrt(X)), MUy = 2
        mux = r2(1 / (2 * (X ** 0.5)))
        muy = 2.0
        if mux > 0 and X > 0 and Y > 0:
            return X, Y, mux, muy
    return 4, 3, 0.25, 2.0


def _w1_q8_params(email: str):
    """Q8 — MRS. U = X^(1/2) * Y^(1/2). Two bundles for comparison."""
    rng = np.random.default_rng(get_seed(email) + 15)
    for _ in range(100):
        X1 = int(rng.integers(2, 5))
        Y1_opts = [y for y in range(6, 14) if y != X1]
        if not Y1_opts:
            continue
        Y1 = int(rng.choice(Y1_opts))
        # Second bundle: more X, less Y
        X2 = X1 * 2
        Y2 = Y1 // 2
        if Y2 <= 0 or X2 == Y2:
            continue
        # MRS = -Y/X for U = sqrt(XY)
        mrs1 = r2(-Y1 / X1)
        mrs2 = r2(-Y2 / X2)
        # Price ratio for part (e)
        Px_opts = [1, 2, 3]
        Py_opts = [2, 4, 6]
        Px = int(rng.choice(Px_opts))
        Py = int(rng.choice(Py_opts))
        price_ratio = r2(Px / Py)
        if abs(mrs1) != price_ratio:  # want them different for interesting comparison
            return X1, Y1, X2, Y2, mrs1, mrs2, Px, Py, price_ratio
    return 2, 8, 4, 4, -4.0, -1.0, 2, 4, 0.5


# ════════════════════════════════════════════════════════════════════════════════
#  WEEK 1 — SHARED TRUE/FALSE RENDERER
# ════════════════════════════════════════════════════════════════════════════════

def _render_w1_tf(q_config, hw_id, email, past_deadline, grace_active,
                  submissions, statements, tf_key):
    """Generic T/F renderer for Week 1 (reused for Q1 and Q5)."""
    q_id    = q_config.get("q_id","")
    q_title = q_config.get("title","")
    marks   = q_config.get("marks", 4)
    correct = [s["correct"] for s in statements]

    prev     = _get_prev(submissions, hw_id, q_id)
    already  = str(prev.get("Status","")) == "submitted"
    disabled = already or (past_deadline and not grace_active)
    score    = int(prev.get("Score", 0)) if already else None

    if already:
        banner_label = f"{q_title}  ·  Score: {score} / {marks}"
    else:
        banner_label = f"{q_title}  ·  Not yet attempted"

    with st.expander(banner_label, expanded=False):

        st.markdown(
            f'<div class="q-body">'
            f'<div class="q-pts">{marks} points &nbsp;·&nbsp; 1 point per statement</div>'
            f'<div class="q-text">Indicate whether each statement is '
            f'<strong>True</strong> or <strong>False</strong>.</div>'
            f'</div>',
            unsafe_allow_html=True)

        if already:
            ts   = prev.get("Timestamp","")
            late = " · Late submission" if prev.get("Is_Late","")=="Yes" else ""
            st.markdown(
                f'<div class="banner banner-locked">🔒 Submitted — '
                f'Score: <strong>{score} / {marks}</strong> · {ts}{late}</div>',
                unsafe_allow_html=True)
        elif past_deadline and not grace_active:
            st.markdown(
                '<div class="banner banner-error">🔒 Deadline has passed.</div>',
                unsafe_allow_html=True)
        elif not prev:
            st.markdown(
                '<div class="banner banner-info">📝 Select True or False for each '
                'statement, then click Submit.</div>',
                unsafe_allow_html=True)

        st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

        # Submitted answers (collapsed)
        prev_answers = {}
        raw = prev.get("Raw_Answer","")
        if raw:
            try:
                pa = eval(str(raw))
                if isinstance(pa, dict): prev_answers = pa
            except Exception: pass

        if already:
            with st.expander("📝 Your Submitted Answers", expanded=False):
                st.markdown(
                    '<div class="submitted-ans">'
                    '<div class="submitted-ans-label">Your Submitted Answers</div>',
                    unsafe_allow_html=True)
                for i, stmt in enumerate(statements):
                    sv = prev_answers.get(f"s{i}", None)
                    ss = ("True" if sv else "False") if sv is not None else "—"
                    st.markdown(
                        f'<div style="padding:0.3rem 0;font-size:1rem;">'
                        f'({i+1}) {stmt["text"][:70]}... &nbsp;→&nbsp; '
                        f'<strong>{ss}</strong></div>',
                        unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)
            st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

        # Answer inputs
        st.markdown(
            '<div class="ans-section"><div class="ans-label">Your Answers</div>',
            unsafe_allow_html=True)

        student_answers = []
        for i, stmt in enumerate(statements):
            prev_val = prev_answers.get(f"s{i}", None)
            default_idx = 0 if (prev_val is True or prev_val is None) else 1
            st.markdown(
                f'<div style="font-size:1rem;font-weight:500;color:#1A1A1A;'
                f'margin:1rem 0 0.3rem 0;">({i+1}) {stmt["text"]}</div>',
                unsafe_allow_html=True)
            choice = st.radio(
                f"S{i}_{tf_key}", options=["True","False"],
                index=default_idx, disabled=disabled,
                horizontal=True, key=f"tf_{tf_key}_{i}_{hw_id}",
                label_visibility="collapsed")
            student_answers.append(choice == "True")

        st.markdown('</div>', unsafe_allow_html=True)

        if not already and not (past_deadline and not grace_active):
            if st.checkbox(
                "I am ready to submit — I understand this cannot be changed.",
                key=f"confirm_{tf_key}_{hw_id}"
            ):
                if st.button(f"Submit {q_title.split('—')[0].strip()}",
                             key=f"sub_{tf_key}_{hw_id}",
                             use_container_width=True):
                    _submit_w1_tf(hw_id, q_id, email, student_answers,
                                  correct, marks, past_deadline)
                    st.rerun()

        if already:
            st.markdown('<hr class="section-divider">', unsafe_allow_html=True)
            with st.expander("Show / Hide Correct Solution", expanded=False):
                _show_w1_tf_solution(prev, statements, correct, marks)

    if already:
        st.markdown(
            '<div style="text-align:right;margin-top:0.3rem;">'
            '<a href="#" style="font-size:0.88rem;color:#2563EB;text-decoration:none;">'
            '↑ Back to top</a></div>',
            unsafe_allow_html=True)


def _submit_w1_tf(hw_id, q_id, email, student_answers, correct, marks, past_deadline):
    sc   = sum(1 for s,c in zip(student_answers, correct) if s==c)
    ts   = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    late = "Yes" if past_deadline else "No"
    raw  = str({f"s{i}": v for i,v in enumerate(student_answers)})
    corr = str({f"s{i}": v for i,v in enumerate(correct)})
    ok, err = write_submission([ts,email,hw_id,q_id,"truefalse","submitted",late,raw,sc,marks,corr])
    log_submission_attempt(email, hw_id, q_id, raw, sc, marks)
    st.session_state.setdefault("submissions",{}).setdefault(hw_id,{})[q_id] = {
        "Status":"submitted","Timestamp":ts,"Score":sc,
        "Max_Score":marks,"Is_Late":late,"Raw_Answer":raw}
    if ok:
        st.markdown(
            f'<div class="banner banner-success">✓ Submitted — '
            f'<strong>Score: {sc} / {marks}</strong></div>', unsafe_allow_html=True)
        st.markdown(f'<div class="saved-ts">Saved at {ts}</div>', unsafe_allow_html=True)
    else:
        st.markdown(
            f'<div class="banner banner-warning">⚠ Sheet write failed ({err}). '
            f'Screenshot this page. Score: {sc}/{marks} at {ts}</div>', unsafe_allow_html=True)


def _show_w1_tf_solution(prev, statements, correct, marks):
    prev_answers = {}
    raw = prev.get("Raw_Answer","")
    if raw:
        try:
            pa = eval(str(raw))
            if isinstance(pa, dict): prev_answers = pa
        except Exception: pass

    student_answers = [prev_answers.get(f"s{i}", None) for i in range(len(statements))]
    sc = sum(1 for s,c in zip(student_answers, correct)
             if s is not None and s==c)

    st.markdown(
        f'<div class="sol-section">'
        f'<div class="sol-title">Solution &nbsp; ({sc} / {marks})</div>',
        unsafe_allow_html=True)

    revise_items = []
    for i, (stmt, corr_val, student_val) in enumerate(
            zip(statements, correct, student_answers)):
        corr_str    = "True"  if corr_val   else "False"
        student_str = ("True" if student_val else "False") if student_val is not None else "—"
        is_ok       = (student_val == corr_val) if student_val is not None else False
        icon        = "✓" if is_ok else "✗"
        col         = "#16A34A" if is_ok else "#DC2626"
        if not is_ok and stmt.get("topic"):
            revise_items.append(stmt["topic"])
        st.markdown(
            f'<div class="tf-item">'
            f'<div class="tf-stmt">({i+1}) {stmt["text"]}</div>'
            f'<div class="tf-result">'
            f'<span style="color:{col};font-weight:700;">{icon}</span> &nbsp;'
            f'<strong>Correct answer: {corr_str}</strong> &nbsp;·&nbsp; '
            f'Your answer: {student_str}</div>'
            f'<div class="tf-expl">{stmt["explanation"]}</div>'
            f'</div>',
            unsafe_allow_html=True)

    if revise_items:
        items = "".join(f"<li style='margin-bottom:0.4rem;'>{r}</li>" for r in revise_items)
        st.markdown(
            f'<div class="sol-label" style="margin-top:1rem;">Topics to revise</div>'
            f'<div class="sol-revise"><ul style="margin:0;padding-left:1.2rem;">'
            f'{items}</ul></div>',
            unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════════════════
#  WEEK 1 — Q2: SLOPES OF LINEAR FUNCTIONS
# ════════════════════════════════════════════════════════════════════════════════

def _render_w1_q2(q_config, hw_id, email, past_deadline, grace_active, submissions):
    x1, y1, x2, y2, a, b, c, slope_pts, slope_eq = _w1_q2_params(email)
    q_id  = q_config.get("q_id","Q2")
    marks = q_config.get("marks", 6)

    prev     = _get_prev(submissions, hw_id, q_id)
    already  = str(prev.get("Status","")) == "submitted"
    disabled = already or (past_deadline and not grace_active)
    score    = int(prev.get("Score", 0)) if already else None

    if already:
        banner_label = f"Question 2 — Slopes of Linear Functions  ·  Score: {score} / {marks}"
    else:
        banner_label = "Question 2 — Slopes of Linear Functions  ·  Not yet attempted"

    with st.expander(banner_label, expanded=False):

        st.markdown(
            f'<div class="q-body">'
            f'<div class="q-pts">{marks} points &nbsp;·&nbsp; parts (b), (c), (d) are graded</div>'
            f'<div class="q-part"><span class="q-part-label">(a)</span> &nbsp;'
            f'Plot the two points A = ({x1}, {y1}) and B = ({x2}, {y2}) on a clearly '
            f'labelled diagram and draw the line passing through them. '
            f'<span class="q-ungraded">(Ungraded)</span></div>'
            f'<div class="q-part"><span class="q-part-label">(b) 2 points</span> &nbsp;'
            f'Using the rise-over-run formula, calculate the slope of the line passing '
            f'through A = ({x1}, {y1}) and B = ({x2}, {y2}).</div>'
            f'<div class="q-part"><span class="q-part-label">(c) 2 points</span> &nbsp;'
            f'Consider the equation {a}Y + {b}X = {c}. Rearrange it into the form '
            f'Y = mX + b and state the slope.</div>'
            f'<div class="q-part"><span class="q-part-label">(d) 2 points</span> &nbsp;'
            f'Are the slopes in (b) and (c) the same? What does this tell you '
            f'about the two lines?</div>'
            f'</div>',
            unsafe_allow_html=True)

        st.markdown(
            f'<div class="param-row">Your values: &nbsp;'
            f'<span class="param-val">A = ({x1}, {y1})</span> &nbsp;·&nbsp; '
            f'<span class="param-val">B = ({x2}, {y2})</span> &nbsp;·&nbsp; '
            f'<span class="param-val">Equation: {a}Y + {b}X = {c}</span></div>',
            unsafe_allow_html=True)

        if already:
            ts   = prev.get("Timestamp","")
            late = " · Late submission" if prev.get("Is_Late","")=="Yes" else ""
            st.markdown(
                f'<div class="banner banner-locked">🔒 Submitted — '
                f'Score: <strong>{score} / {marks}</strong> · {ts}{late}</div>',
                unsafe_allow_html=True)
        elif past_deadline and not grace_active:
            st.markdown('<div class="banner banner-error">🔒 Deadline has passed.</div>',
                        unsafe_allow_html=True)
        elif not prev:
            st.markdown(
                '<div class="banner banner-info">📝 Enter your answers below '
                'and click Submit when ready.</div>', unsafe_allow_html=True)
        else:
            st.markdown(
                '<div class="banner banner-restore">Draft restored.</div>',
                unsafe_allow_html=True)

        st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

        if already:
            d = _parse_raw(prev)
            with st.expander("📝 Your Submitted Answers", expanded=False):
                st.markdown(
                    f'<div class="submitted-ans">'
                    f'<div class="submitted-ans-label">Your Submitted Answers</div>'
                    f'(b) Slope from points: <strong>{d.get("slope_pts","—")}</strong><br>'
                    f'(c) Slope from equation: <strong>{d.get("slope_eq","—")}</strong><br>'
                    f'(d) Same line? <strong>{"Yes" if d.get("same_line") else "No"}</strong>'
                    f'</div>', unsafe_allow_html=True)
            st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

        if not already:
            d = _parse_raw(prev)
            st.markdown(
                '<div class="ans-section"><div class="ans-label">Your Answers</div>',
                unsafe_allow_html=True)
            col1, col2 = st.columns(2)
            with col1:
                ans_slope_pts = st.number_input(
                    "(b) Slope from points A and B",
                    value=float(d.get("slope_pts", 0.0)),
                    step=0.01, format="%.2f",
                    disabled=disabled, key=f"w1q2_sp_{hw_id}")
            with col2:
                ans_slope_eq = st.number_input(
                    "(c) Slope from equation",
                    value=float(d.get("slope_eq", 0.0)),
                    step=0.01, format="%.2f",
                    disabled=disabled, key=f"w1q2_se_{hw_id}")
            ans_same = st.radio(
                "(d) Are the two slopes the same?",
                options=["Yes — they are the same line",
                         "No — they are different lines"],
                index=0 if d.get("same_line", False) else 1,
                disabled=disabled, key=f"w1q2_sl_{hw_id}")
            st.caption("Enter slopes rounded to 2 decimal places.")
            st.markdown('</div>', unsafe_allow_html=True)

            if not (past_deadline and not grace_active):
                has_input = ans_slope_pts != 0.0 or ans_slope_eq != 0.0
                if has_input:
                    if st.checkbox(
                        "I am ready to submit — I understand this cannot be changed.",
                        key=f"confirm_w1q2_{hw_id}"):
                        if st.button("Submit Question 2", key=f"sub_w1q2_{hw_id}",
                                     use_container_width=True):
                            same_correct = (slope_pts != slope_eq)  # they differ → "No"
                            student_same = "Yes" in ans_same
                            sp_ok = r2(ans_slope_pts) == slope_pts
                            se_ok = r2(ans_slope_eq)  == slope_eq
                            sl_ok = student_same == (not same_correct)
                            sc = 2*int(sp_ok) + 2*int(se_ok) + 2*int(sl_ok)
                            ts   = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                            late = "Yes" if past_deadline else "No"
                            raw  = str({"slope_pts": r2(ans_slope_pts),
                                        "slope_eq":  r2(ans_slope_eq),
                                        "same_line": student_same})
                            corr = str({"slope_pts": slope_pts,
                                        "slope_eq":  slope_eq,
                                        "same_line": not same_correct})
                            ok, err = write_submission([ts,email,hw_id,q_id,"numerical",
                                                        "submitted",late,raw,sc,marks,corr])
                            log_submission_attempt(email, hw_id, q_id, raw, sc, marks)
                            st.session_state.setdefault("submissions",{}).setdefault(hw_id,{})[q_id] = {
                                "Status":"submitted","Timestamp":ts,"Score":sc,
                                "Max_Score":marks,"Is_Late":late,"Raw_Answer":raw}
                            if ok:
                                st.markdown(
                                    f'<div class="banner banner-success">✓ Question 2 submitted — '
                                    f'<strong>Score: {sc} / {marks}</strong></div>',
                                    unsafe_allow_html=True)
                                st.markdown(f'<div class="saved-ts">Saved at {ts}</div>',
                                            unsafe_allow_html=True)
                            else:
                                st.markdown(
                                    f'<div class="banner banner-warning">⚠ Sheet write failed. '
                                    f'Score: {sc}/{marks} at {ts}</div>', unsafe_allow_html=True)
                            st.rerun()
                else:
                    st.caption("Fill in your answers above to enable submission.")

        if already:
            st.markdown('<hr class="section-divider">', unsafe_allow_html=True)
            with st.expander("Show / Hide Correct Solution", expanded=False):
                _show_w1_q2_solution(prev, x1, y1, x2, y2, a, b, c, slope_pts, slope_eq)

    if already:
        st.markdown(
            '<div style="text-align:right;margin-top:0.3rem;">'
            '<a href="#" style="font-size:0.88rem;color:#2563EB;text-decoration:none;">'
            '↑ Back to top</a></div>', unsafe_allow_html=True)


def _show_w1_q2_solution(prev, x1, y1, x2, y2, a, b, c, slope_pts, slope_eq):
    d = _parse_raw(prev)
    ans_sp = float(d.get("slope_pts", 0))
    ans_se = float(d.get("slope_eq",  0))
    ans_sl = d.get("same_line", False)
    same_correct = slope_pts != slope_eq
    sp_ok = r2(ans_sp) == slope_pts
    se_ok = r2(ans_se) == slope_eq
    sl_ok = ans_sl == (not same_correct)
    sc = 2*int(sp_ok) + 2*int(se_ok) + 2*int(sl_ok)

    sp_chip = '<span class="chip-ok">+2</span>' if sp_ok else '<span class="chip-wrong">0</span>'
    se_chip = '<span class="chip-ok">+2</span>' if se_ok else '<span class="chip-wrong">0</span>'
    sl_chip = '<span class="chip-ok">+2</span>' if sl_ok else '<span class="chip-wrong">0</span>'

    intercept = r2(c / a)
    slope_eq_str = r2(-b / a)

    st.markdown('<div class="sol-section"><div class="sol-title">Solution — Question 2</div>',
                unsafe_allow_html=True)
    st.markdown(
        f'<div class="sol-label">Step-by-step working</div>'
        f'<div class="sol-steps">'
        f'<p><strong>(b) Rise-over-run:</strong><br>'
        f'Slope = (y2 - y1) / (x2 - x1) = ({y2} - {y1}) / ({x2} - {x1}) '
        f'= {y2-y1} / {x2-x1} = <strong>{slope_pts}</strong></p>'
        f'<p><strong>(c) From equation:</strong><br>'
        f'{a}Y + {b}X = {c}<br>'
        f'{a}Y = {c} - {b}X<br>'
        f'Y = {intercept} - {abs(slope_eq_str)}X<br>'
        f'Slope = <strong>{slope_eq}</strong></p>'
        f'<p><strong>(d)</strong> The slope from (b) is {slope_pts} and from (c) is {slope_eq}. '
        f'{"They are different, so these are two different lines with different steepness." if same_correct else "They are the same, so these are the same line."}</p>'
        f'</div>',
        unsafe_allow_html=True)

    st.markdown('<div class="sol-label">Your score</div>', unsafe_allow_html=True)
    st.markdown(
        f'<div class="score-row"><span class="score-row-label">'
        f'(b) Slope from points &nbsp;·&nbsp; Correct: {slope_pts} &nbsp;·&nbsp; Yours: {r2(ans_sp)}'
        f'</span><span class="score-row-val">{sp_chip}</span></div>',
        unsafe_allow_html=True)
    st.markdown(
        f'<div class="score-row"><span class="score-row-label">'
        f'(c) Slope from equation &nbsp;·&nbsp; Correct: {slope_eq} &nbsp;·&nbsp; Yours: {r2(ans_se)}'
        f'</span><span class="score-row-val">{se_chip}</span></div>',
        unsafe_allow_html=True)
    st.markdown(
        f'<div class="score-row"><span class="score-row-label">'
        f'(d) Same line? &nbsp;·&nbsp; Correct: {"Yes" if not same_correct else "No"} &nbsp;·&nbsp; Yours: {"Yes" if ans_sl else "No"}'
        f'</span><span class="score-row-val">{sl_chip}</span></div>',
        unsafe_allow_html=True)
    st.markdown(
        f'<div class="score-row score-row-total"><span>Total</span><span>{sc} / 6</span></div>',
        unsafe_allow_html=True)

    st.markdown(
        '<div class="sol-label">Common mistakes</div>'
        '<div class="sol-mistakes">'
        '<p>Forgetting to rearrange the equation into Y = mX + b form before reading off the slope. '
        'Always isolate Y on the left-hand side first.</p>'
        '<p>Reversing the rise and run — always compute (y2 - y1) / (x2 - x1), not the other way around.</p>'
        '</div>', unsafe_allow_html=True)

    revise = []
    if not sp_ok: revise.append("Rise-over-run formula: slope = (y2-y1) / (x2-x1)")
    if not se_ok: revise.append("Rearranging a linear equation into Y = mX + b to identify the slope")
    if not sl_ok: revise.append("Comparing slopes to determine whether two lines are the same or different")
    if revise:
        items = "".join(f"<li style='margin-bottom:0.4rem;'>{r}</li>" for r in revise)
        st.markdown(
            f'<div class="sol-label">Topics to revise</div>'
            f'<div class="sol-revise"><ul style="margin:0;padding-left:1.2rem;">'
            f'{items}</ul></div>', unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════════════════
#  WEEK 1 — Q3: DERIVATIVES & PARTIAL DERIVATIVES
# ════════════════════════════════════════════════════════════════════════════════

def _render_w1_q3(q_config, hw_id, email, past_deadline, grace_active, submissions):
    X, Y, mux, muy = _w1_q3_params(email)
    q_id  = q_config.get("q_id","Q3")
    marks = q_config.get("marks", 6)

    prev     = _get_prev(submissions, hw_id, q_id)
    already  = str(prev.get("Status","")) == "submitted"
    disabled = already or (past_deadline and not grace_active)
    score    = int(prev.get("Score", 0)) if already else None

    if already:
        banner_label = f"Question 3 — Derivatives & Partial Derivatives  ·  Score: {score} / {marks}"
    else:
        banner_label = "Question 3 — Derivatives & Partial Derivatives  ·  Not yet attempted"

    with st.expander(banner_label, expanded=False):

        st.markdown(
            f'<div class="q-body">'
            f'<div class="q-pts">{marks} points &nbsp;·&nbsp; parts (b) and (c) are graded</div>'
            f'<div class="q-part"><span class="q-part-label">(a)</span> &nbsp;'
            f'State the power rule in your own words. '
            f'<span class="q-ungraded">(Ungraded)</span></div>'
            f'<div class="q-part"><span class="q-part-label">(b) 3 points</span> &nbsp;'
            f'Find the derivative dy/dx for each of the following:<br>'
            f'&nbsp;&nbsp;(i) y = x^3 &nbsp;&nbsp; '
            f'(ii) y = x^(1/2) &nbsp;&nbsp; '
            f'(iii) y = 3x^4 + 2x^(1/2) + 5</div>'
            f'<div class="q-part"><span class="q-part-label">(c) 3 points</span> &nbsp;'
            f'A consumer has utility function U(X, Y) = X^(1/3) x Y^(2/3). '
            f'Compute MUx and MUy at the bundle (X, Y) = ({X}, {Y}). '
            f'Are both positive? Which preference assumption does this reflect?</div>'
            f'</div>',
            unsafe_allow_html=True)

        st.markdown(
            f'<div class="param-row">Your values for part (c): &nbsp;'
            f'<span class="param-val">X = {X}</span> &nbsp;·&nbsp; '
            f'<span class="param-val">Y = {Y}</span></div>',
            unsafe_allow_html=True)

        if already:
            ts   = prev.get("Timestamp","")
            late = " · Late submission" if prev.get("Is_Late","")=="Yes" else ""
            st.markdown(
                f'<div class="banner banner-locked">🔒 Submitted — '
                f'Score: <strong>{score} / {marks}</strong> · {ts}{late}</div>',
                unsafe_allow_html=True)
        elif past_deadline and not grace_active:
            st.markdown('<div class="banner banner-error">🔒 Deadline has passed.</div>',
                        unsafe_allow_html=True)
        elif not prev:
            st.markdown(
                '<div class="banner banner-info">📝 Enter your answers below '
                'and click Submit when ready.</div>', unsafe_allow_html=True)

        st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

        if already:
            d = _parse_raw(prev)
            with st.expander("📝 Your Submitted Answers", expanded=False):
                st.markdown(
                    f'<div class="submitted-ans">'
                    f'<div class="submitted-ans-label">Your Submitted Answers</div>'
                    f'(b)(i) dy/dx for x^3: <strong>{d.get("b1","—")}</strong> '
                    f'(entered as power of x, e.g. 3 means 3x^2)<br>'
                    f'(b)(ii) dy/dx for x^(1/2): coefficient <strong>{d.get("b2c","—")}</strong>, '
                    f'power <strong>{d.get("b2p","—")}</strong><br>'
                    f'(b)(iii) dy/dx for 3x^4+2x^(1/2)+5: <strong>{d.get("b3","—")}</strong><br>'
                    f'(c) MUx = <strong>{d.get("mux","—")}</strong>, '
                    f'MUy = <strong>{d.get("muy","—")}</strong>'
                    f'</div>', unsafe_allow_html=True)
            st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

        if not already:
            d = _parse_raw(prev)
            st.markdown(
                '<div class="ans-section"><div class="ans-label">Your Answers</div>',
                unsafe_allow_html=True)

            st.markdown("**(b) Derivatives** — apply the power rule to each:")
            st.markdown("**(i) y = x^3** — enter the coefficient of the derivative "
                        "(e.g. if dy/dx = 3x^2, enter 3):")
            ans_b1 = st.number_input("Coefficient of derivative of x^3",
                                     value=float(d.get("b1",0.0)),
                                     step=1.0, format="%.0f",
                                     disabled=disabled, key=f"w1q3_b1_{hw_id}")

            st.markdown("**(ii) y = x^(1/2)** — enter the coefficient and new power:")
            col1, col2 = st.columns(2)
            with col1:
                ans_b2c = st.number_input("Coefficient (e.g. 0.5)",
                                          value=float(d.get("b2c",0.0)),
                                          step=0.01, format="%.2f",
                                          disabled=disabled, key=f"w1q3_b2c_{hw_id}")
            with col2:
                ans_b2p = st.number_input("New power (e.g. -0.5)",
                                          value=float(d.get("b2p",0.0)),
                                          step=0.01, format="%.2f",
                                          disabled=disabled, key=f"w1q3_b2p_{hw_id}")

            st.markdown("**(iii) y = 3x^4 + 2x^(1/2) + 5** — enter the coefficient "
                        "of the x^3 term in the derivative:")
            ans_b3 = st.number_input("Coefficient of x^3 term in derivative of 3x^4+2x^(1/2)+5",
                                     value=float(d.get("b3",0.0)),
                                     step=1.0, format="%.0f",
                                     disabled=disabled, key=f"w1q3_b3_{hw_id}")

            st.markdown(f"**(c) Marginal utilities** at (X={X}, Y={Y}):")
            col3, col4 = st.columns(2)
            with col3:
                ans_mux = st.number_input("MUx at this bundle",
                                          value=float(d.get("mux",0.0)),
                                          step=0.01, format="%.4f",
                                          disabled=disabled, key=f"w1q3_mux_{hw_id}")
            with col4:
                ans_muy = st.number_input("MUy at this bundle",
                                          value=float(d.get("muy",0.0)),
                                          step=0.01, format="%.4f",
                                          disabled=disabled, key=f"w1q3_muy_{hw_id}")
            st.caption("Enter answers rounded to 4 decimal places for parts (c).")
            st.markdown('</div>', unsafe_allow_html=True)

            if not (past_deadline and not grace_active):
                has_input = any(v != 0.0 for v in [ans_b1, ans_b2c, ans_b2p, ans_b3, ans_mux, ans_muy])
                if has_input:
                    if st.checkbox(
                        "I am ready to submit — I understand this cannot be changed.",
                        key=f"confirm_w1q3_{hw_id}"):
                        if st.button("Submit Question 3", key=f"sub_w1q3_{hw_id}",
                                     use_container_width=True):
                            b1_ok  = r2(ans_b1) == 3.0
                            b2c_ok = r2(ans_b2c) == 0.5
                            b2p_ok = r2(ans_b2p) == -0.5
                            b3_ok  = r2(ans_b3) == 12.0
                            mux_ok = round(ans_mux, 4) == round(mux, 4)
                            muy_ok = round(ans_muy, 4) == round(muy, 4)
                            # scoring: b part = 1pt each (3 sub-parts = 3pts), c = 1.5+1.5=3pts
                            sc = int(b1_ok) + int(b2c_ok and b2p_ok) + int(b3_ok) + \
                                 (2 if (mux_ok and muy_ok) else (1 if (mux_ok or muy_ok) else 0))
                            sc = min(sc, marks)
                            ts   = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                            late = "Yes" if past_deadline else "No"
                            raw  = str({"b1":r2(ans_b1),"b2c":r2(ans_b2c),"b2p":r2(ans_b2p),
                                        "b3":r2(ans_b3),"mux":round(ans_mux,4),"muy":round(ans_muy,4)})
                            corr = str({"b1":3.0,"b2c":0.5,"b2p":-0.5,"b3":12.0,
                                        "mux":round(mux,4),"muy":round(muy,4)})
                            ok, err = write_submission([ts,email,hw_id,q_id,"numerical",
                                                        "submitted",late,raw,sc,marks,corr])
                            log_submission_attempt(email, hw_id, q_id, raw, sc, marks)
                            st.session_state.setdefault("submissions",{}).setdefault(hw_id,{})[q_id] = {
                                "Status":"submitted","Timestamp":ts,"Score":sc,
                                "Max_Score":marks,"Is_Late":late,"Raw_Answer":raw}
                            if ok:
                                st.markdown(
                                    f'<div class="banner banner-success">✓ Question 3 submitted — '
                                    f'<strong>Score: {sc} / {marks}</strong></div>',
                                    unsafe_allow_html=True)
                                st.markdown(f'<div class="saved-ts">Saved at {ts}</div>',
                                            unsafe_allow_html=True)
                            else:
                                st.markdown(
                                    f'<div class="banner banner-warning">⚠ Sheet write failed. '
                                    f'Score: {sc}/{marks} at {ts}</div>', unsafe_allow_html=True)
                            st.rerun()
                else:
                    st.caption("Fill in your answers above to enable submission.")

        if already:
            st.markdown('<hr class="section-divider">', unsafe_allow_html=True)
            with st.expander("Show / Hide Correct Solution", expanded=False):
                _show_w1_q3_solution(prev, X, Y, mux, muy, marks)

    if already:
        st.markdown(
            '<div style="text-align:right;margin-top:0.3rem;">'
            '<a href="#" style="font-size:0.88rem;color:#2563EB;text-decoration:none;">'
            '↑ Back to top</a></div>', unsafe_allow_html=True)


def _show_w1_q3_solution(prev, X, Y, mux, muy, marks):
    d = _parse_raw(prev)
    ans_b1  = float(d.get("b1",  0))
    ans_b2c = float(d.get("b2c", 0))
    ans_b2p = float(d.get("b2p", 0))
    ans_b3  = float(d.get("b3",  0))
    ans_mux = float(d.get("mux", 0))
    ans_muy = float(d.get("muy", 0))

    b1_ok  = r2(ans_b1)  == 3.0
    b2c_ok = r2(ans_b2c) == 0.5
    b2p_ok = r2(ans_b2p) == -0.5
    b3_ok  = r2(ans_b3)  == 12.0
    mux_ok = round(ans_mux, 4) == round(mux, 4)
    muy_ok = round(ans_muy, 4) == round(muy, 4)
    sc = int(b1_ok) + int(b2c_ok and b2p_ok) + int(b3_ok) + \
         (2 if (mux_ok and muy_ok) else (1 if (mux_ok or muy_ok) else 0))
    sc = min(sc, marks)

    b1_chip  = '<span class="chip-ok">+1</span>' if b1_ok  else '<span class="chip-wrong">0</span>'
    b2_chip  = '<span class="chip-ok">+1</span>' if (b2c_ok and b2p_ok) else '<span class="chip-wrong">0</span>'
    b3_chip  = '<span class="chip-ok">+1</span>' if b3_ok  else '<span class="chip-wrong">0</span>'
    mu_chip  = ('<span class="chip-ok">+2</span>' if (mux_ok and muy_ok)
                else ('<span class="chip-ok">+1</span>' if (mux_ok or muy_ok)
                      else '<span class="chip-wrong">0</span>'))

    st.markdown('<div class="sol-section"><div class="sol-title">Solution — Question 3</div>',
                unsafe_allow_html=True)
    st.markdown(
        f'<div class="sol-label">Step-by-step working</div>'
        f'<div class="sol-steps">'
        f'<p><strong>(b)(i)</strong> y = x^3: dy/dx = 3x^2 (bring down exponent, reduce by 1). '
        f'Coefficient = <strong>3</strong></p>'
        f'<p><strong>(b)(ii)</strong> y = x^(1/2): dy/dx = (1/2)x^(1/2 - 1) = (1/2)x^(-1/2). '
        f'Coefficient = <strong>0.5</strong>, new power = <strong>-0.5</strong></p>'
        f'<p><strong>(b)(iii)</strong> y = 3x^4 + 2x^(1/2) + 5:<br>'
        f'dy/dx = 3(4)x^3 + 2(1/2)x^(-1/2) + 0 = 12x^3 + x^(-1/2). '
        f'Coefficient of x^3 term = <strong>12</strong></p>'
        f'<p><strong>(c)</strong> U = X^(1/3) x Y^(2/3):<br>'
        f'MUx = (1/3) x X^(-2/3) x Y^(2/3) = (1/3) x {X}^(-2/3) x {Y}^(2/3) = <strong>{round(mux,4)}</strong><br>'
        f'MUy = (2/3) x X^(1/3) x Y^(-1/3) = (2/3) x {X}^(1/3) x {Y}^(-1/3) = <strong>{round(muy,4)}</strong><br>'
        f'Both are positive, reflecting the <strong>monotonicity</strong> assumption: '
        f'more of either good always increases utility.</p>'
        f'</div>',
        unsafe_allow_html=True)

    st.markdown('<div class="sol-label">Your score</div>', unsafe_allow_html=True)
    st.markdown(
        f'<div class="score-row"><span class="score-row-label">'
        f'(b)(i) Coefficient of derivative of x^3 &nbsp;·&nbsp; Correct: 3 &nbsp;·&nbsp; Yours: {r2(ans_b1)}'
        f'</span><span class="score-row-val">{b1_chip}</span></div>', unsafe_allow_html=True)
    st.markdown(
        f'<div class="score-row"><span class="score-row-label">'
        f'(b)(ii) Coefficient 0.5, power -0.5 &nbsp;·&nbsp; Yours: coeff={r2(ans_b2c)}, power={r2(ans_b2p)}'
        f'</span><span class="score-row-val">{b2_chip}</span></div>', unsafe_allow_html=True)
    st.markdown(
        f'<div class="score-row"><span class="score-row-label">'
        f'(b)(iii) Coefficient of x^3 term &nbsp;·&nbsp; Correct: 12 &nbsp;·&nbsp; Yours: {r2(ans_b3)}'
        f'</span><span class="score-row-val">{b3_chip}</span></div>', unsafe_allow_html=True)
    st.markdown(
        f'<div class="score-row"><span class="score-row-label">'
        f'(c) MUx = {round(mux,4)}, MUy = {round(muy,4)} &nbsp;·&nbsp; '
        f'Yours: MUx={round(ans_mux,4)}, MUy={round(ans_muy,4)}'
        f'</span><span class="score-row-val">{mu_chip}</span></div>', unsafe_allow_html=True)
    st.markdown(
        f'<div class="score-row score-row-total"><span>Total</span><span>{sc} / {marks}</span></div>',
        unsafe_allow_html=True)

    st.markdown(
        '<div class="sol-label">Common mistakes</div>'
        '<div class="sol-mistakes">'
        '<p>Forgetting to rewrite 1/x^n as x^(-n) before applying the power rule. '
        'Always convert to a negative exponent first.</p>'
        '<p>When computing partial derivatives, treating the other variable as a constant '
        'but accidentally differentiating it too. When computing MUx, Y is held fixed — '
        'it does not change.</p>'
        '</div>', unsafe_allow_html=True)

    revise = []
    if not b1_ok or not (b2c_ok and b2p_ok) or not b3_ok:
        revise.append("Power rule: d(x^n)/dx = n x x^(n-1)")
    if not mux_ok or not muy_ok:
        revise.append("Partial derivatives: hold all other variables fixed when differentiating")
    if revise:
        items = "".join(f"<li style='margin-bottom:0.4rem;'>{r}</li>" for r in revise)
        st.markdown(
            f'<div class="sol-label">Topics to revise</div>'
            f'<div class="sol-revise"><ul style="margin:0;padding-left:1.2rem;">'
            f'{items}</ul></div>', unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════════════════
#  WEEK 1 — Q4: DEMAND, SUPPLY & EQUILIBRIUM
# ════════════════════════════════════════════════════════════════════════════════

def _render_w1_q4(q_config, hw_id, email, past_deadline, grace_active, submissions):
    A, a, B, b, P_star, Q_star, P_off, Qd_off, Qs_off, shortage, shift, B_new, P_new, Q_new = _w1_q4_params(email)
    q_id  = q_config.get("q_id","Q4")
    marks = q_config.get("marks", 10)

    prev     = _get_prev(submissions, hw_id, q_id)
    already  = str(prev.get("Status","")) == "submitted"
    disabled = already or (past_deadline and not grace_active)
    score    = int(prev.get("Score", 0)) if already else None

    if already:
        banner_label = f"Question 4 — Demand, Supply & Equilibrium  ·  Score: {score} / {marks}"
    else:
        banner_label = "Question 4 — Demand, Supply & Equilibrium  ·  Not yet attempted"

    with st.expander(banner_label, expanded=False):

        st.markdown(
            f'<div class="q-body">'
            f'<div class="q-pts">{marks} points &nbsp;·&nbsp; parts (b), (c), (d) are graded</div>'
            f'<div class="q-text">A market has the following demand and supply equations:<br>'
            f'<strong>Demand:</strong> Qd = {A} - {a}P &nbsp;&nbsp;&nbsp; '
            f'<strong>Supply:</strong> Qs = {B} + {b}P</div>'
            f'<div class="q-part"><span class="q-part-label">(a)</span> &nbsp;'
            f'Draw a clearly labelled supply and demand diagram for this market. '
            f'<span class="q-ungraded">(Ungraded)</span></div>'
            f'<div class="q-part"><span class="q-part-label">(b) 4 points</span> &nbsp;'
            f'Find the equilibrium price P* and equilibrium quantity Q*.</div>'
            f'<div class="q-part"><span class="q-part-label">(c) 2 points</span> &nbsp;'
            f'At a price of P = {P_off}, calculate whether there is a surplus or shortage '
            f'and by how much.</div>'
            f'<div class="q-part"><span class="q-part-label">(d) 4 points</span> &nbsp;'
            f'A new technology reduces production costs, shifting supply to '
            f'Qs = {B_new} + {b}P. Find the new equilibrium price and quantity. '
            f'Did price rise or fall? Does this make intuitive sense?</div>'
            f'</div>',
            unsafe_allow_html=True)

        st.markdown(
            f'<div class="param-row">Your market: &nbsp;'
            f'<span class="param-val">Qd = {A} - {a}P</span> &nbsp;·&nbsp; '
            f'<span class="param-val">Qs = {B} + {b}P</span> &nbsp;·&nbsp; '
            f'<span class="param-val">Off-equilibrium price: {P_off}</span></div>',
            unsafe_allow_html=True)

        if already:
            ts   = prev.get("Timestamp","")
            late = " · Late submission" if prev.get("Is_Late","")=="Yes" else ""
            st.markdown(
                f'<div class="banner banner-locked">🔒 Submitted — '
                f'Score: <strong>{score} / {marks}</strong> · {ts}{late}</div>',
                unsafe_allow_html=True)
        elif past_deadline and not grace_active:
            st.markdown('<div class="banner banner-error">🔒 Deadline has passed.</div>',
                        unsafe_allow_html=True)
        elif not prev:
            st.markdown(
                '<div class="banner banner-info">📝 Enter your answers below '
                'and click Submit when ready.</div>', unsafe_allow_html=True)

        st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

        if already:
            d = _parse_raw(prev)
            with st.expander("📝 Your Submitted Answers", expanded=False):
                st.markdown(
                    f'<div class="submitted-ans">'
                    f'<div class="submitted-ans-label">Your Submitted Answers</div>'
                    f'(b) P* = <strong>{d.get("pstar","—")}</strong>, '
                    f'Q* = <strong>{d.get("qstar","—")}</strong><br>'
                    f'(c) Shortage/Surplus amount = <strong>{d.get("gap","—")}</strong><br>'
                    f'(d) New P* = <strong>{d.get("pnew","—")}</strong>, '
                    f'New Q* = <strong>{d.get("qnew","—")}</strong>'
                    f'</div>', unsafe_allow_html=True)
            st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

        if not already:
            d = _parse_raw(prev)
            st.markdown(
                '<div class="ans-section"><div class="ans-label">Your Answers</div>',
                unsafe_allow_html=True)
            st.markdown("**(b) Equilibrium:**")
            col1, col2 = st.columns(2)
            with col1:
                ans_pstar = st.number_input("Equilibrium price P*",
                                            value=float(d.get("pstar",0.0)),
                                            step=1.0, format="%.0f",
                                            disabled=disabled, key=f"w1q4_ps_{hw_id}")
            with col2:
                ans_qstar = st.number_input("Equilibrium quantity Q*",
                                            value=float(d.get("qstar",0.0)),
                                            step=1.0, format="%.0f",
                                            disabled=disabled, key=f"w1q4_qs_{hw_id}")

            st.markdown(f"**(c) At P = {P_off}:**")
            ans_gap = st.number_input(
                f"Size of shortage (positive) or surplus (negative) at P = {P_off}",
                value=float(d.get("gap",0.0)),
                step=1.0, format="%.0f",
                disabled=disabled, key=f"w1q4_gap_{hw_id}")
            st.caption("Enter a positive number for a shortage, negative for a surplus.")

            st.markdown("**(d) New equilibrium after supply shift:**")
            col3, col4 = st.columns(2)
            with col3:
                ans_pnew = st.number_input("New equilibrium price",
                                           value=float(d.get("pnew",0.0)),
                                           step=1.0, format="%.0f",
                                           disabled=disabled, key=f"w1q4_pn_{hw_id}")
            with col4:
                ans_qnew = st.number_input("New equilibrium quantity",
                                           value=float(d.get("qnew",0.0)),
                                           step=1.0, format="%.0f",
                                           disabled=disabled, key=f"w1q4_qn_{hw_id}")
            st.markdown('</div>', unsafe_allow_html=True)

            if not (past_deadline and not grace_active):
                has_input = any(v != 0.0 for v in [ans_pstar, ans_qstar, ans_gap, ans_pnew, ans_qnew])
                if has_input:
                    if st.checkbox(
                        "I am ready to submit — I understand this cannot be changed.",
                        key=f"confirm_w1q4_{hw_id}"):
                        if st.button("Submit Question 4", key=f"sub_w1q4_{hw_id}",
                                     use_container_width=True):
                            ps_ok   = r2(ans_pstar) == float(P_star)
                            qs_ok   = r2(ans_qstar) == float(Q_star)
                            gap_ok  = r2(ans_gap)   == float(shortage)
                            pn_ok   = r2(ans_pnew)  == float(P_new)
                            qn_ok   = r2(ans_qnew)  == float(Q_new)
                            sc = 2*int(ps_ok) + 2*int(qs_ok) + 2*int(gap_ok) + \
                                 2*int(pn_ok) + 2*int(qn_ok)
                            sc = min(sc, marks)
                            ts   = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                            late = "Yes" if past_deadline else "No"
                            raw  = str({"pstar":r2(ans_pstar),"qstar":r2(ans_qstar),
                                        "gap":r2(ans_gap),"pnew":r2(ans_pnew),"qnew":r2(ans_qnew)})
                            corr = str({"pstar":P_star,"qstar":Q_star,"gap":shortage,
                                        "pnew":P_new,"qnew":Q_new})
                            ok, err = write_submission([ts,email,hw_id,q_id,"numerical",
                                                        "submitted",late,raw,sc,marks,corr])
                            log_submission_attempt(email, hw_id, q_id, raw, sc, marks)
                            st.session_state.setdefault("submissions",{}).setdefault(hw_id,{})[q_id] = {
                                "Status":"submitted","Timestamp":ts,"Score":sc,
                                "Max_Score":marks,"Is_Late":late,"Raw_Answer":raw}
                            if ok:
                                st.markdown(
                                    f'<div class="banner banner-success">✓ Question 4 submitted — '
                                    f'<strong>Score: {sc} / {marks}</strong></div>',
                                    unsafe_allow_html=True)
                                st.markdown(f'<div class="saved-ts">Saved at {ts}</div>',
                                            unsafe_allow_html=True)
                            else:
                                st.markdown(
                                    f'<div class="banner banner-warning">⚠ Sheet write failed. '
                                    f'Score: {sc}/{marks} at {ts}</div>', unsafe_allow_html=True)
                            st.rerun()
                else:
                    st.caption("Fill in your answers above to enable submission.")

        if already:
            st.markdown('<hr class="section-divider">', unsafe_allow_html=True)
            with st.expander("Show / Hide Correct Solution", expanded=False):
                _show_w1_q4_solution(prev, A, a, B, b, P_star, Q_star,
                                     P_off, Qd_off, Qs_off, shortage,
                                     shift, B_new, P_new, Q_new, marks)

    if already:
        st.markdown(
            '<div style="text-align:right;margin-top:0.3rem;">'
            '<a href="#" style="font-size:0.88rem;color:#2563EB;text-decoration:none;">'
            '↑ Back to top</a></div>', unsafe_allow_html=True)


def _show_w1_q4_solution(prev, A, a, B, b, P_star, Q_star,
                          P_off, Qd_off, Qs_off, shortage,
                          shift, B_new, P_new, Q_new, marks):
    d = _parse_raw(prev)
    ans_ps  = float(d.get("pstar", 0))
    ans_qs  = float(d.get("qstar", 0))
    ans_gap = float(d.get("gap",   0))
    ans_pn  = float(d.get("pnew",  0))
    ans_qn  = float(d.get("qnew",  0))

    ps_ok  = r2(ans_ps)  == float(P_star)
    qs_ok  = r2(ans_qs)  == float(Q_star)
    gap_ok = r2(ans_gap) == float(shortage)
    pn_ok  = r2(ans_pn)  == float(P_new)
    qn_ok  = r2(ans_qn)  == float(Q_new)
    sc = min(2*int(ps_ok)+2*int(qs_ok)+2*int(gap_ok)+2*int(pn_ok)+2*int(qn_ok), marks)

    def chip2(ok): return '<span class="chip-ok">+2</span>' if ok else '<span class="chip-wrong">0</span>'

    st.markdown('<div class="sol-section"><div class="sol-title">Solution — Question 4</div>',
                unsafe_allow_html=True)
    st.markdown(
        f'<div class="sol-label">Step-by-step working</div>'
        f'<div class="sol-steps">'
        f'<p><strong>(b) Equilibrium:</strong> Set Qd = Qs:<br>'
        f'{A} - {a}P = {B} + {b}P<br>'
        f'{A} - {B} = {a}P + {b}P<br>'
        f'{A-B} = {a+b}P<br>'
        f'P* = {A-B} / {a+b} = <strong>{P_star}</strong><br>'
        f'Q* = {A} - {a}({P_star}) = <strong>{Q_star}</strong></p>'
        f'<p><strong>(c) At P = {P_off}:</strong><br>'
        f'Qd = {A} - {a}({P_off}) = {Qd_off}<br>'
        f'Qs = {B} + {b}({P_off}) = {Qs_off}<br>'
        f'Since Qd ({Qd_off}) > Qs ({Qs_off}), there is a <strong>shortage</strong> of '
        f'<strong>{shortage}</strong> units. '
        f'P = {P_off} is below equilibrium so buyers want more than sellers supply.</p>'
        f'<p><strong>(d) New supply:</strong> Qs = {B_new} + {b}P. Set Qd = Qs:<br>'
        f'{A} - {a}P = {B_new} + {b}P<br>'
        f'{A-B_new} = {a+b}P<br>'
        f'New P* = {A-B_new} / {a+b} = <strong>{P_new}</strong><br>'
        f'New Q* = {A} - {a}({P_new}) = <strong>{Q_new}</strong><br>'
        f'Price <strong>fell</strong> from {P_star} to {P_new}. '
        f'This makes intuitive sense: lower production costs increase supply, '
        f'pushing the price down and quantity up.</p>'
        f'</div>',
        unsafe_allow_html=True)

    st.markdown('<div class="sol-label">Your score</div>', unsafe_allow_html=True)
    st.markdown(
        f'<div class="score-row"><span class="score-row-label">'
        f'(b) P* &nbsp;·&nbsp; Correct: {P_star} &nbsp;·&nbsp; Yours: {r2(ans_ps)}'
        f'</span><span class="score-row-val">{chip2(ps_ok)}</span></div>', unsafe_allow_html=True)
    st.markdown(
        f'<div class="score-row"><span class="score-row-label">'
        f'(b) Q* &nbsp;·&nbsp; Correct: {Q_star} &nbsp;·&nbsp; Yours: {r2(ans_qs)}'
        f'</span><span class="score-row-val">{chip2(qs_ok)}</span></div>', unsafe_allow_html=True)
    st.markdown(
        f'<div class="score-row"><span class="score-row-label">'
        f'(c) Shortage &nbsp;·&nbsp; Correct: {shortage} &nbsp;·&nbsp; Yours: {r2(ans_gap)}'
        f'</span><span class="score-row-val">{chip2(gap_ok)}</span></div>', unsafe_allow_html=True)
    st.markdown(
        f'<div class="score-row"><span class="score-row-label">'
        f'(d) New P* &nbsp;·&nbsp; Correct: {P_new} &nbsp;·&nbsp; Yours: {r2(ans_pn)}'
        f'</span><span class="score-row-val">{chip2(pn_ok)}</span></div>', unsafe_allow_html=True)
    st.markdown(
        f'<div class="score-row"><span class="score-row-label">'
        f'(d) New Q* &nbsp;·&nbsp; Correct: {Q_new} &nbsp;·&nbsp; Yours: {r2(ans_qn)}'
        f'</span><span class="score-row-val">{chip2(qn_ok)}</span></div>', unsafe_allow_html=True)
    st.markdown(
        f'<div class="score-row score-row-total"><span>Total</span><span>{sc} / {marks}</span></div>',
        unsafe_allow_html=True)

    st.markdown(
        '<div class="sol-label">Common mistakes</div>'
        '<div class="sol-mistakes">'
        '<p>Setting up the equilibrium equation correctly but making an arithmetic error '
        'when solving for P. Always verify by substituting P* back into both equations.</p>'
        '<p>Confusing a shortage with a surplus. When price is below equilibrium, '
        'Qd > Qs — that is a shortage. When price is above equilibrium, Qs > Qd — that is a surplus.</p>'
        '</div>', unsafe_allow_html=True)

    revise = []
    if not ps_ok or not qs_ok: revise.append("Setting Qd = Qs and solving for equilibrium P* and Q*")
    if not gap_ok: revise.append("Calculating shortage (Qd - Qs) or surplus (Qs - Qd) at off-equilibrium prices")
    if not pn_ok or not qn_ok: revise.append("Finding new equilibrium after a supply curve shift")
    if revise:
        items = "".join(f"<li style='margin-bottom:0.4rem;'>{r}</li>" for r in revise)
        st.markdown(
            f'<div class="sol-label">Topics to revise</div>'
            f'<div class="sol-revise"><ul style="margin:0;padding-left:1.2rem;">'
            f'{items}</ul></div>', unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════════════════
#  WEEK 1 — Q6: INDIFFERENCE CURVES
# ════════════════════════════════════════════════════════════════════════════════

def _render_w1_q6(q_config, hw_id, email, past_deadline, grace_active, submissions):
    XA, YA, XB, YB, XC, YC, U_AB, U_C, XD, YD, XE, YE = _w1_q6_params(email)
    q_id  = q_config.get("q_id","Q6")
    marks = q_config.get("marks", 6)

    prev     = _get_prev(submissions, hw_id, q_id)
    already  = str(prev.get("Status","")) == "submitted"
    disabled = already or (past_deadline and not grace_active)
    score    = int(prev.get("Score", 0)) if already else None

    if already:
        banner_label = f"Question 6 — Indifference Curves  ·  Score: {score} / {marks}"
    else:
        banner_label = "Question 6 — Indifference Curves  ·  Not yet attempted"

    with st.expander(banner_label, expanded=False):

        st.markdown(
            f'<div class="q-body">'
            f'<div class="q-pts">{marks} points &nbsp;·&nbsp; parts (b), (c), (d) are graded</div>'
            f'<div class="q-text">A consumer has utility function U(X, Y) = X x Y.</div>'
            f'<div class="q-part"><span class="q-part-label">(a)</span> &nbsp;'
            f'Draw a clearly labelled indifference map showing at least two indifference curves. '
            f'<span class="q-ungraded">(Ungraded)</span></div>'
            f'<div class="q-part"><span class="q-part-label">(b) 2 points</span> &nbsp;'
            f'Compute the utility level for each bundle: '
            f'A = ({XA}, {YA}), B = ({XB}, {YB}), C = ({XC}, {YC}).</div>'
            f'<div class="q-part"><span class="q-part-label">(c) 2 points</span> &nbsp;'
            f'Which bundles lie on the same indifference curve? '
            f'What does this mean about how the consumer ranks them?</div>'
            f'<div class="q-part"><span class="q-part-label">(d) 2 points</span> &nbsp;'
            f'Bundle D = ({XD}, {YD}) and bundle E = ({XE}, {YE}). '
            f'Without computing utility, use the monotonicity assumption to determine '
            f'which bundle is preferred. Explain in one sentence.</div>'
            f'</div>',
            unsafe_allow_html=True)

        st.markdown(
            f'<div class="param-row">Your bundles: &nbsp;'
            f'<span class="param-val">A = ({XA}, {YA})</span> &nbsp;·&nbsp; '
            f'<span class="param-val">B = ({XB}, {YB})</span> &nbsp;·&nbsp; '
            f'<span class="param-val">C = ({XC}, {YC})</span></div>',
            unsafe_allow_html=True)

        if already:
            ts   = prev.get("Timestamp","")
            late = " · Late submission" if prev.get("Is_Late","")=="Yes" else ""
            st.markdown(
                f'<div class="banner banner-locked">🔒 Submitted — '
                f'Score: <strong>{score} / {marks}</strong> · {ts}{late}</div>',
                unsafe_allow_html=True)
        elif past_deadline and not grace_active:
            st.markdown('<div class="banner banner-error">🔒 Deadline has passed.</div>',
                        unsafe_allow_html=True)
        elif not prev:
            st.markdown(
                '<div class="banner banner-info">📝 Enter your answers below '
                'and click Submit when ready.</div>', unsafe_allow_html=True)

        st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

        if already:
            d = _parse_raw(prev)
            with st.expander("📝 Your Submitted Answers", expanded=False):
                st.markdown(
                    f'<div class="submitted-ans">'
                    f'<div class="submitted-ans-label">Your Submitted Answers</div>'
                    f'(b) U(A) = <strong>{d.get("ua","—")}</strong>, '
                    f'U(B) = <strong>{d.get("ub","—")}</strong>, '
                    f'U(C) = <strong>{d.get("uc","—")}</strong><br>'
                    f'(c) Same IC: <strong>{d.get("same_ic","—")}</strong><br>'
                    f'(d) Preferred bundle: <strong>{d.get("preferred","—")}</strong>'
                    f'</div>', unsafe_allow_html=True)
            st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

        if not already:
            d = _parse_raw(prev)
            st.markdown(
                '<div class="ans-section"><div class="ans-label">Your Answers</div>',
                unsafe_allow_html=True)
            st.markdown("**(b) Utility levels:**")
            col1, col2, col3 = st.columns(3)
            with col1:
                ans_ua = st.number_input(f"U(A) at ({XA},{YA})",
                                         value=float(d.get("ua",0.0)),
                                         step=1.0, format="%.0f",
                                         disabled=disabled, key=f"w1q6_ua_{hw_id}")
            with col2:
                ans_ub = st.number_input(f"U(B) at ({XB},{YB})",
                                         value=float(d.get("ub",0.0)),
                                         step=1.0, format="%.0f",
                                         disabled=disabled, key=f"w1q6_ub_{hw_id}")
            with col3:
                ans_uc = st.number_input(f"U(C) at ({XC},{YC})",
                                         value=float(d.get("uc",0.0)),
                                         step=1.0, format="%.0f",
                                         disabled=disabled, key=f"w1q6_uc_{hw_id}")

            st.markdown("**(c) Which two bundles are on the same indifference curve?**")
            ans_same = st.radio(
                "Same IC",
                options=["A and B", "A and C", "B and C"],
                index=0,
                disabled=disabled, horizontal=True,
                key=f"w1q6_same_{hw_id}",
                label_visibility="collapsed")

            st.markdown(f"**(d) Which bundle is preferred — D = ({XD},{YD}) or E = ({XE},{YE})?**")
            ans_pref = st.radio(
                "Preferred",
                options=[f"D = ({XD}, {YD})", f"E = ({XE}, {YE})", "Cannot tell from monotonicity alone"],
                index=0,
                disabled=disabled,
                key=f"w1q6_pref_{hw_id}",
                label_visibility="collapsed")
            st.markdown('</div>', unsafe_allow_html=True)

            if not (past_deadline and not grace_active):
                has_input = ans_ua != 0.0 or ans_ub != 0.0 or ans_uc != 0.0
                if has_input:
                    if st.checkbox(
                        "I am ready to submit — I understand this cannot be changed.",
                        key=f"confirm_w1q6_{hw_id}"):
                        if st.button("Submit Question 6", key=f"sub_w1q6_{hw_id}",
                                     use_container_width=True):
                            ua_ok   = r2(ans_ua) == float(U_AB)
                            ub_ok   = r2(ans_ub) == float(U_AB)
                            uc_ok   = r2(ans_uc) == float(U_C)
                            same_ok = "A and B" in ans_same
                            # D has more of both than E if XD>XE and YD>YE
                            d_dominates = (XD > XE and YD > YE)
                            pref_ok = (f"D = ({XD}, {YD})" in ans_pref) == d_dominates or \
                                      ("Cannot tell" in ans_pref) == (not d_dominates and not (XE>XD and YE>YD))
                            sc = 1*int(ua_ok) + 1*int(ub_ok) + 1*int(uc_ok) + \
                                 1*int(same_ok) + 2*int(pref_ok)
                            sc = min(sc, marks)
                            ts   = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                            late = "Yes" if past_deadline else "No"
                            raw  = str({"ua":r2(ans_ua),"ub":r2(ans_ub),"uc":r2(ans_uc),
                                        "same_ic":ans_same,"preferred":ans_pref})
                            corr = str({"ua":U_AB,"ub":U_AB,"uc":U_C,
                                        "same_ic":"A and B","preferred":f"D = ({XD}, {YD})" if d_dominates else "Cannot tell"})
                            ok, err = write_submission([ts,email,hw_id,q_id,"numerical",
                                                        "submitted",late,raw,sc,marks,corr])
                            log_submission_attempt(email, hw_id, q_id, raw, sc, marks)
                            st.session_state.setdefault("submissions",{}).setdefault(hw_id,{})[q_id] = {
                                "Status":"submitted","Timestamp":ts,"Score":sc,
                                "Max_Score":marks,"Is_Late":late,"Raw_Answer":raw}
                            if ok:
                                st.markdown(
                                    f'<div class="banner banner-success">✓ Question 6 submitted — '
                                    f'<strong>Score: {sc} / {marks}</strong></div>',
                                    unsafe_allow_html=True)
                                st.markdown(f'<div class="saved-ts">Saved at {ts}</div>',
                                            unsafe_allow_html=True)
                            else:
                                st.markdown(
                                    f'<div class="banner banner-warning">⚠ Sheet write failed. '
                                    f'Score: {sc}/{marks} at {ts}</div>', unsafe_allow_html=True)
                            st.rerun()
                else:
                    st.caption("Fill in your answers above to enable submission.")

        if already:
            st.markdown('<hr class="section-divider">', unsafe_allow_html=True)
            with st.expander("Show / Hide Correct Solution", expanded=False):
                _show_w1_q6_solution(prev, XA, YA, XB, YB, XC, YC,
                                     U_AB, U_C, XD, YD, XE, YE, marks)

    if already:
        st.markdown(
            '<div style="text-align:right;margin-top:0.3rem;">'
            '<a href="#" style="font-size:0.88rem;color:#2563EB;text-decoration:none;">'
            '↑ Back to top</a></div>', unsafe_allow_html=True)


def _show_w1_q6_solution(prev, XA, YA, XB, YB, XC, YC,
                          U_AB, U_C, XD, YD, XE, YE, marks):
    d = _parse_raw(prev)
    ans_ua   = float(d.get("ua", 0))
    ans_ub   = float(d.get("ub", 0))
    ans_uc   = float(d.get("uc", 0))
    ans_same = str(d.get("same_ic",""))
    ans_pref = str(d.get("preferred",""))

    ua_ok   = r2(ans_ua) == float(U_AB)
    ub_ok   = r2(ans_ub) == float(U_AB)
    uc_ok   = r2(ans_uc) == float(U_C)
    same_ok = "A and B" in ans_same
    d_dominates = (XD > XE and YD > YE)
    pref_ok = (f"D = ({XD}, {YD})" in ans_pref) == d_dominates or \
              ("Cannot tell" in ans_pref) == (not d_dominates and not (XE>XD and YE>YD))
    sc = min(int(ua_ok)+int(ub_ok)+int(uc_ok)+int(same_ok)+2*int(pref_ok), marks)

    def chip1(ok): return '<span class="chip-ok">+1</span>' if ok else '<span class="chip-wrong">0</span>'

    correct_pref = f"D = ({XD}, {YD})" if d_dominates else "Cannot determine from monotonicity alone"

    st.markdown('<div class="sol-section"><div class="sol-title">Solution — Question 6</div>',
                unsafe_allow_html=True)
    st.markdown(
        f'<div class="sol-label">Step-by-step working</div>'
        f'<div class="sol-steps">'
        f'<p><strong>(b) Utility levels (U = X x Y):</strong><br>'
        f'U(A) = {XA} x {YA} = <strong>{U_AB}</strong><br>'
        f'U(B) = {XB} x {YB} = <strong>{U_AB}</strong><br>'
        f'U(C) = {XC} x {YC} = <strong>{U_C}</strong></p>'
        f'<p><strong>(c)</strong> Bundles A and B both give U = {U_AB}, so they lie on the '
        f'same indifference curve. The consumer is <strong>indifferent</strong> between them — '
        f'neither is preferred to the other.</p>'
        f'<p><strong>(d)</strong> D = ({XD}, {YD}) has {"more" if d_dominates else "the same or less"} '
        f'of both goods compared to E = ({XE}, {YE}). '
        f'{"By monotonicity, D is preferred to E." if d_dominates else "Since neither bundle has more of both goods, monotonicity alone cannot determine which is preferred."}'
        f' Correct answer: <strong>{correct_pref}</strong></p>'
        f'</div>',
        unsafe_allow_html=True)

    st.markdown('<div class="sol-label">Your score</div>', unsafe_allow_html=True)
    st.markdown(
        f'<div class="score-row"><span class="score-row-label">'
        f'(b) U(A) &nbsp;·&nbsp; Correct: {U_AB} &nbsp;·&nbsp; Yours: {r2(ans_ua)}'
        f'</span><span class="score-row-val">{chip1(ua_ok)}</span></div>', unsafe_allow_html=True)
    st.markdown(
        f'<div class="score-row"><span class="score-row-label">'
        f'(b) U(B) &nbsp;·&nbsp; Correct: {U_AB} &nbsp;·&nbsp; Yours: {r2(ans_ub)}'
        f'</span><span class="score-row-val">{chip1(ub_ok)}</span></div>', unsafe_allow_html=True)
    st.markdown(
        f'<div class="score-row"><span class="score-row-label">'
        f'(b) U(C) &nbsp;·&nbsp; Correct: {U_C} &nbsp;·&nbsp; Yours: {r2(ans_uc)}'
        f'</span><span class="score-row-val">{chip1(uc_ok)}</span></div>', unsafe_allow_html=True)
    st.markdown(
        f'<div class="score-row"><span class="score-row-label">'
        f'(c) Same IC &nbsp;·&nbsp; Correct: A and B &nbsp;·&nbsp; Yours: {ans_same}'
        f'</span><span class="score-row-val">{chip1(same_ok)}</span></div>', unsafe_allow_html=True)
    st.markdown(
        f'<div class="score-row"><span class="score-row-label">'
        f'(d) Preferred &nbsp;·&nbsp; Correct: {correct_pref} &nbsp;·&nbsp; Yours: {ans_pref}'
        f'</span><span class="score-row-val">{"<span class=\"chip-ok\">+2</span>" if pref_ok else "<span class=\"chip-wrong\">0</span>"}</span></div>',
        unsafe_allow_html=True)
    st.markdown(
        f'<div class="score-row score-row-total"><span>Total</span><span>{sc} / {marks}</span></div>',
        unsafe_allow_html=True)

    st.markdown(
        '<div class="sol-label">Common mistakes</div>'
        '<div class="sol-mistakes">'
        '<p>Assuming that having more of one good automatically means a bundle is preferred. '
        'Monotonicity only guarantees preference when a bundle has more of both goods simultaneously.</p>'
        '<p>Confusing indifference with equality — two bundles on the same IC are equally preferred, '
        'not "the same." The consumer would be happy with either.</p>'
        '</div>', unsafe_allow_html=True)

    revise = []
    if not ua_ok or not ub_ok or not uc_ok:
        revise.append("Computing utility: substitute X and Y values into U(X,Y)")
    if not same_ok:
        revise.append("Indifference curves: bundles with equal utility lie on the same IC")
    if not pref_ok:
        revise.append("Monotonicity: more of BOTH goods guarantees preference — more of just one is not enough")
    if revise:
        items = "".join(f"<li style='margin-bottom:0.4rem;'>{r}</li>" for r in revise)
        st.markdown(
            f'<div class="sol-label">Topics to revise</div>'
            f'<div class="sol-revise"><ul style="margin:0;padding-left:1.2rem;">'
            f'{items}</ul></div>', unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════════════════
#  WEEK 1 — Q7: MARGINAL UTILITY
# ════════════════════════════════════════════════════════════════════════════════

def _render_w1_q7(q_config, hw_id, email, past_deadline, grace_active, submissions):
    X, Y, mux, muy = _w1_q7_params(email)
    q_id  = q_config.get("q_id","Q7")
    marks = q_config.get("marks", 6)

    prev     = _get_prev(submissions, hw_id, q_id)
    already  = str(prev.get("Status","")) == "submitted"
    disabled = already or (past_deadline and not grace_active)
    score    = int(prev.get("Score", 0)) if already else None

    if already:
        banner_label = f"Question 7 — Marginal Utility  ·  Score: {score} / {marks}"
    else:
        banner_label = "Question 7 — Marginal Utility  ·  Not yet attempted"

    with st.expander(banner_label, expanded=False):

        st.markdown(
            f'<div class="q-body">'
            f'<div class="q-pts">{marks} points</div>'
            f'<div class="q-text">A consumer has utility function U(X, Y) = X^(1/2) + 2Y.</div>'
            f'<div class="q-part"><span class="q-part-label">(a) 2 points</span> &nbsp;'
            f'Compute the general formulas for MUx and MUy.</div>'
            f'<div class="q-part"><span class="q-part-label">(b) 2 points</span> &nbsp;'
            f'Evaluate MUx and MUy at the bundle (X, Y) = ({X}, {Y}). '
            f'Which good provides more utility at the margin?</div>'
            f'<div class="q-part"><span class="q-part-label">(c) 2 points</span> &nbsp;'
            f'As the consumer consumes more X (holding Y fixed), what happens to MUx? '
            f'What economic principle does this illustrate?</div>'
            f'</div>',
            unsafe_allow_html=True)

        st.markdown(
            f'<div class="param-row">Your bundle for part (b): &nbsp;'
            f'<span class="param-val">X = {X}</span> &nbsp;·&nbsp; '
            f'<span class="param-val">Y = {Y}</span></div>',
            unsafe_allow_html=True)

        if already:
            ts   = prev.get("Timestamp","")
            late = " · Late submission" if prev.get("Is_Late","")=="Yes" else ""
            st.markdown(
                f'<div class="banner banner-locked">🔒 Submitted — '
                f'Score: <strong>{score} / {marks}</strong> · {ts}{late}</div>',
                unsafe_allow_html=True)
        elif past_deadline and not grace_active:
            st.markdown('<div class="banner banner-error">🔒 Deadline has passed.</div>',
                        unsafe_allow_html=True)
        elif not prev:
            st.markdown(
                '<div class="banner banner-info">📝 Enter your answers below '
                'and click Submit when ready.</div>', unsafe_allow_html=True)

        st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

        if already:
            d = _parse_raw(prev)
            with st.expander("📝 Your Submitted Answers", expanded=False):
                st.markdown(
                    f'<div class="submitted-ans">'
                    f'<div class="submitted-ans-label">Your Submitted Answers</div>'
                    f'(b) MUx at bundle = <strong>{d.get("mux","—")}</strong>, '
                    f'MUy at bundle = <strong>{d.get("muy","—")}</strong><br>'
                    f'(c) MUx as X increases: <strong>{d.get("mux_direction","—")}</strong>'
                    f'</div>', unsafe_allow_html=True)
            st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

        if not already:
            d = _parse_raw(prev)
            st.markdown(
                '<div class="ans-section"><div class="ans-label">Your Answers</div>',
                unsafe_allow_html=True)
            st.markdown("**(a)** The general formulas are: MUx = (1/2) x X^(-1/2) and MUy = 2. "
                        "These are fixed — confirm you understand them before entering (b).")
            st.markdown(f"**(b)** Evaluate at (X={X}, Y={Y}):")
            col1, col2 = st.columns(2)
            with col1:
                ans_mux = st.number_input(f"MUx at X={X}",
                                          value=float(d.get("mux",0.0)),
                                          step=0.01, format="%.4f",
                                          disabled=disabled, key=f"w1q7_mux_{hw_id}")
            with col2:
                ans_muy = st.number_input(f"MUy at Y={Y} (constant)",
                                          value=float(d.get("muy",0.0)),
                                          step=0.01, format="%.4f",
                                          disabled=disabled, key=f"w1q7_muy_{hw_id}")

            st.markdown("**(c)** As X increases (holding Y fixed), MUx:")
            ans_dir = st.radio(
                "MUx direction",
                options=["Increases", "Decreases", "Stays the same"],
                index=1,
                disabled=disabled, horizontal=True,
                key=f"w1q7_dir_{hw_id}",
                label_visibility="collapsed")
            st.markdown('</div>', unsafe_allow_html=True)

            if not (past_deadline and not grace_active):
                has_input = ans_mux != 0.0 or ans_muy != 0.0
                if has_input:
                    if st.checkbox(
                        "I am ready to submit — I understand this cannot be changed.",
                        key=f"confirm_w1q7_{hw_id}"):
                        if st.button("Submit Question 7", key=f"sub_w1q7_{hw_id}",
                                     use_container_width=True):
                            mux_ok = round(ans_mux, 4) == round(mux, 4)
                            muy_ok = round(ans_muy, 4) == round(muy, 4)
                            dir_ok = "Decreases" in ans_dir
                            sc = 2*int(mux_ok and muy_ok) + 2*int(dir_ok) + \
                                 (2 if (mux_ok and muy_ok) else 0)
                            # simpler: 2 for b, 2 for c, 2 for a (auto-given)
                            sc = 2*int(mux_ok) + 2*int(muy_ok) + 2*int(dir_ok)
                            sc = min(sc, marks)
                            ts   = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                            late = "Yes" if past_deadline else "No"
                            raw  = str({"mux":round(ans_mux,4),"muy":round(ans_muy,4),
                                        "mux_direction":ans_dir})
                            corr = str({"mux":round(mux,4),"muy":round(muy,4),
                                        "mux_direction":"Decreases"})
                            ok, err = write_submission([ts,email,hw_id,q_id,"numerical",
                                                        "submitted",late,raw,sc,marks,corr])
                            log_submission_attempt(email, hw_id, q_id, raw, sc, marks)
                            st.session_state.setdefault("submissions",{}).setdefault(hw_id,{})[q_id] = {
                                "Status":"submitted","Timestamp":ts,"Score":sc,
                                "Max_Score":marks,"Is_Late":late,"Raw_Answer":raw}
                            if ok:
                                st.markdown(
                                    f'<div class="banner banner-success">✓ Question 7 submitted — '
                                    f'<strong>Score: {sc} / {marks}</strong></div>',
                                    unsafe_allow_html=True)
                                st.markdown(f'<div class="saved-ts">Saved at {ts}</div>',
                                            unsafe_allow_html=True)
                            else:
                                st.markdown(
                                    f'<div class="banner banner-warning">⚠ Sheet write failed. '
                                    f'Score: {sc}/{marks} at {ts}</div>', unsafe_allow_html=True)
                            st.rerun()
                else:
                    st.caption("Fill in your answers above to enable submission.")

        if already:
            st.markdown('<hr class="section-divider">', unsafe_allow_html=True)
            with st.expander("Show / Hide Correct Solution", expanded=False):
                _show_w1_q7_solution(prev, X, Y, mux, muy, marks)

    if already:
        st.markdown(
            '<div style="text-align:right;margin-top:0.3rem;">'
            '<a href="#" style="font-size:0.88rem;color:#2563EB;text-decoration:none;">'
            '↑ Back to top</a></div>', unsafe_allow_html=True)


def _show_w1_q7_solution(prev, X, Y, mux, muy, marks):
    d = _parse_raw(prev)
    ans_mux = float(d.get("mux", 0))
    ans_muy = float(d.get("muy", 0))
    ans_dir = str(d.get("mux_direction",""))

    mux_ok = round(ans_mux,4) == round(mux,4)
    muy_ok = round(ans_muy,4) == round(muy,4)
    dir_ok = "Decreases" in ans_dir
    sc = min(2*int(mux_ok)+2*int(muy_ok)+2*int(dir_ok), marks)

    def chip2(ok): return '<span class="chip-ok">+2</span>' if ok else '<span class="chip-wrong">0</span>'

    st.markdown('<div class="sol-section"><div class="sol-title">Solution — Question 7</div>',
                unsafe_allow_html=True)
    st.markdown(
        f'<div class="sol-label">Step-by-step working</div>'
        f'<div class="sol-steps">'
        f'<p><strong>(a) General formulas:</strong><br>'
        f'U = X^(1/2) + 2Y<br>'
        f'MUx = dU/dX = (1/2)X^(-1/2) = 1 / (2 x sqrt(X))<br>'
        f'MUy = dU/dY = 2 (constant — does not depend on X or Y)</p>'
        f'<p><strong>(b) At ({X}, {Y}):</strong><br>'
        f'MUx = 1 / (2 x sqrt({X})) = 1 / (2 x {X**0.5}) = <strong>{round(mux,4)}</strong><br>'
        f'MUy = <strong>{round(muy,4)}</strong><br>'
        f'{"MUy > MUx, so good Y provides more utility at the margin at this bundle." if muy > mux else "MUx > MUy, so good X provides more utility at the margin at this bundle."}</p>'
        f'<p><strong>(c)</strong> As X increases, MUx = 1/(2 x sqrt(X)) <strong>decreases</strong> '
        f'because sqrt(X) grows larger, making the fraction smaller. '
        f'This illustrates the <strong>law of diminishing marginal utility</strong>: '
        f'each additional unit of X adds less and less to total utility.</p>'
        f'</div>',
        unsafe_allow_html=True)

    st.markdown('<div class="sol-label">Your score</div>', unsafe_allow_html=True)
    st.markdown(
        f'<div class="score-row"><span class="score-row-label">'
        f'(b) MUx &nbsp;·&nbsp; Correct: {round(mux,4)} &nbsp;·&nbsp; Yours: {round(ans_mux,4)}'
        f'</span><span class="score-row-val">{chip2(mux_ok)}</span></div>', unsafe_allow_html=True)
    st.markdown(
        f'<div class="score-row"><span class="score-row-label">'
        f'(b) MUy &nbsp;·&nbsp; Correct: {round(muy,4)} &nbsp;·&nbsp; Yours: {round(ans_muy,4)}'
        f'</span><span class="score-row-val">{chip2(muy_ok)}</span></div>', unsafe_allow_html=True)
    st.markdown(
        f'<div class="score-row"><span class="score-row-label">'
        f'(c) MUx as X increases &nbsp;·&nbsp; Correct: Decreases &nbsp;·&nbsp; Yours: {ans_dir}'
        f'</span><span class="score-row-val">{chip2(dir_ok)}</span></div>', unsafe_allow_html=True)
    st.markdown(
        f'<div class="score-row score-row-total"><span>Total</span><span>{sc} / {marks}</span></div>',
        unsafe_allow_html=True)

    st.markdown(
        '<div class="sol-label">Common mistakes</div>'
        '<div class="sol-mistakes">'
        '<p>Forgetting that MUy = 2 is a constant — it does not depend on Y. '
        'When you differentiate 2Y with respect to Y, you simply get 2.</p>'
        '<p>Saying MUx increases as X increases. Since MUx = 1/(2 x sqrt(X)), '
        'a larger X makes the denominator bigger, so MUx gets smaller — it decreases.</p>'
        '</div>', unsafe_allow_html=True)

    revise = []
    if not mux_ok or not muy_ok:
        revise.append("Computing marginal utility by differentiating the utility function")
    if not dir_ok:
        revise.append("Diminishing marginal utility: MU decreases as consumption of a good increases")
    if revise:
        items = "".join(f"<li style='margin-bottom:0.4rem;'>{r}</li>" for r in revise)
        st.markdown(
            f'<div class="sol-label">Topics to revise</div>'
            f'<div class="sol-revise"><ul style="margin:0;padding-left:1.2rem;">'
            f'{items}</ul></div>', unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════════════════
#  WEEK 1 — Q8: MARGINAL RATE OF SUBSTITUTION
# ════════════════════════════════════════════════════════════════════════════════

def _render_w1_q8(q_config, hw_id, email, past_deadline, grace_active, submissions):
    X1, Y1, X2, Y2, mrs1, mrs2, Px, Py, price_ratio = _w1_q8_params(email)
    q_id  = q_config.get("q_id","Q8")
    marks = q_config.get("marks", 8)

    prev     = _get_prev(submissions, hw_id, q_id)
    already  = str(prev.get("Status","")) == "submitted"
    disabled = already or (past_deadline and not grace_active)
    score    = int(prev.get("Score", 0)) if already else None

    if already:
        banner_label = f"Question 8 — Marginal Rate of Substitution  ·  Score: {score} / {marks}"
    else:
        banner_label = "Question 8 — Marginal Rate of Substitution  ·  Not yet attempted"

    with st.expander(banner_label, expanded=False):

        st.markdown(
            f'<div class="q-body">'
            f'<div class="q-pts">{marks} points</div>'
            f'<div class="q-text">A consumer has utility function U(X, Y) = X^(1/2) x Y^(1/2).</div>'
            f'<div class="q-part"><span class="q-part-label">(a)</span> &nbsp;'
            f'Write down the general formula MRS = -MUx / MUy. '
            f'<span class="q-ungraded">(Ungraded)</span></div>'
            f'<div class="q-part"><span class="q-part-label">(b) 2 points</span> &nbsp;'
            f'Compute MUx and MUy for this utility function.</div>'
            f'<div class="q-part"><span class="q-part-label">(c) 2 points</span> &nbsp;'
            f'Derive the MRS as a function of X and Y.</div>'
            f'<div class="q-part"><span class="q-part-label">(d) 2 points</span> &nbsp;'
            f'Evaluate the MRS at bundle ({X1}, {Y1}) and at bundle ({X2}, {Y2}). '
            f'Is the MRS diminishing as X increases?</div>'
            f'<div class="q-part"><span class="q-part-label">(e) 2 points</span> &nbsp;'
            f'At bundle ({X1}, {Y1}), the consumer faces prices Px = {Px} and Py = {Py}. '
            f'The price ratio Px/Py = {price_ratio}. '
            f'Compare this to |MRS| at that bundle. '
            f'Should the consumer buy more X or more Y? Explain in one sentence.</div>'
            f'</div>',
            unsafe_allow_html=True)

        st.markdown(
            f'<div class="param-row">Your bundles: &nbsp;'
            f'<span class="param-val">Bundle 1: ({X1}, {Y1})</span> &nbsp;·&nbsp; '
            f'<span class="param-val">Bundle 2: ({X2}, {Y2})</span> &nbsp;·&nbsp; '
            f'<span class="param-val">Px = {Px}, Py = {Py}</span></div>',
            unsafe_allow_html=True)

        if already:
            ts   = prev.get("Timestamp","")
            late = " · Late submission" if prev.get("Is_Late","")=="Yes" else ""
            st.markdown(
                f'<div class="banner banner-locked">🔒 Submitted — '
                f'Score: <strong>{score} / {marks}</strong> · {ts}{late}</div>',
                unsafe_allow_html=True)
        elif past_deadline and not grace_active:
            st.markdown('<div class="banner banner-error">🔒 Deadline has passed.</div>',
                        unsafe_allow_html=True)
        elif not prev:
            st.markdown(
                '<div class="banner banner-info">📝 Enter your answers below '
                'and click Submit when ready.</div>', unsafe_allow_html=True)

        st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

        if already:
            d = _parse_raw(prev)
            with st.expander("📝 Your Submitted Answers", expanded=False):
                st.markdown(
                    f'<div class="submitted-ans">'
                    f'<div class="submitted-ans-label">Your Submitted Answers</div>'
                    f'(d) MRS at ({X1},{Y1}) = <strong>{d.get("mrs1","—")}</strong>, '
                    f'MRS at ({X2},{Y2}) = <strong>{d.get("mrs2","—")}</strong><br>'
                    f'(e) Buy more: <strong>{d.get("buy_more","—")}</strong>'
                    f'</div>', unsafe_allow_html=True)
            st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

        if not already:
            d = _parse_raw(prev)
            st.markdown(
                '<div class="ans-section"><div class="ans-label">Your Answers</div>',
                unsafe_allow_html=True)
            st.markdown("**(b) and (c)** are conceptual. For U = X^(1/2) x Y^(1/2): "
                        "MUx = (1/2)X^(-1/2)Y^(1/2), MUy = (1/2)X^(1/2)Y^(-1/2), "
                        "MRS = -Y/X.")
            st.markdown("**(d) Evaluate MRS at each bundle:**")
            col1, col2 = st.columns(2)
            with col1:
                ans_mrs1 = st.number_input(
                    f"MRS at ({X1}, {Y1})",
                    value=float(d.get("mrs1",0.0)),
                    step=0.01, format="%.2f",
                    disabled=disabled, key=f"w1q8_m1_{hw_id}")
            with col2:
                ans_mrs2 = st.number_input(
                    f"MRS at ({X2}, {Y2})",
                    value=float(d.get("mrs2",0.0)),
                    step=0.01, format="%.2f",
                    disabled=disabled, key=f"w1q8_m2_{hw_id}")

            st.markdown(f"**(e)** |MRS| at ({X1},{Y1}) vs price ratio {price_ratio}: "
                        f"should the consumer buy more X or Y?")
            # if |MRS| > Px/Py → buy more X
            abs_mrs1 = abs(mrs1)
            correct_buy = "X" if abs_mrs1 > price_ratio else "Y"
            ans_buy = st.radio(
                "Buy more",
                options=["More X", "More Y", "Already at optimum"],
                index=0 if correct_buy=="X" else 1,
                disabled=disabled, horizontal=True,
                key=f"w1q8_buy_{hw_id}",
                label_visibility="collapsed")
            st.markdown('</div>', unsafe_allow_html=True)

            if not (past_deadline and not grace_active):
                has_input = ans_mrs1 != 0.0 or ans_mrs2 != 0.0
                if has_input:
                    if st.checkbox(
                        "I am ready to submit — I understand this cannot be changed.",
                        key=f"confirm_w1q8_{hw_id}"):
                        if st.button("Submit Question 8", key=f"sub_w1q8_{hw_id}",
                                     use_container_width=True):
                            m1_ok  = r2(ans_mrs1) == mrs1
                            m2_ok  = r2(ans_mrs2) == mrs2
                            buy_ok = (correct_buy=="X" and "More X" in ans_buy) or \
                                     (correct_buy=="Y" and "More Y" in ans_buy)
                            sc = 2*int(m1_ok) + 2*int(m2_ok) + 2*int(buy_ok) + 2
                            # 2 pts for b/c (awarded automatically as conceptual)
                            sc = min(sc, marks)
                            ts   = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                            late = "Yes" if past_deadline else "No"
                            raw  = str({"mrs1":r2(ans_mrs1),"mrs2":r2(ans_mrs2),
                                        "buy_more":ans_buy})
                            corr = str({"mrs1":mrs1,"mrs2":mrs2,
                                        "buy_more":f"More {correct_buy}"})
                            ok, err = write_submission([ts,email,hw_id,q_id,"numerical",
                                                        "submitted",late,raw,sc,marks,corr])
                            log_submission_attempt(email, hw_id, q_id, raw, sc, marks)
                            st.session_state.setdefault("submissions",{}).setdefault(hw_id,{})[q_id] = {
                                "Status":"submitted","Timestamp":ts,"Score":sc,
                                "Max_Score":marks,"Is_Late":late,"Raw_Answer":raw}
                            if ok:
                                st.markdown(
                                    f'<div class="banner banner-success">✓ Question 8 submitted — '
                                    f'<strong>Score: {sc} / {marks}</strong></div>',
                                    unsafe_allow_html=True)
                                st.markdown(f'<div class="saved-ts">Saved at {ts}</div>',
                                            unsafe_allow_html=True)
                            else:
                                st.markdown(
                                    f'<div class="banner banner-warning">⚠ Sheet write failed. '
                                    f'Score: {sc}/{marks} at {ts}</div>', unsafe_allow_html=True)
                            st.rerun()
                else:
                    st.caption("Fill in your answers above to enable submission.")

        if already:
            st.markdown('<hr class="section-divider">', unsafe_allow_html=True)
            with st.expander("Show / Hide Correct Solution", expanded=False):
                _show_w1_q8_solution(prev, X1, Y1, X2, Y2, mrs1, mrs2,
                                     Px, Py, price_ratio, marks)

    if already:
        st.markdown(
            '<div style="text-align:right;margin-top:0.3rem;">'
            '<a href="#" style="font-size:0.88rem;color:#2563EB;text-decoration:none;">'
            '↑ Back to top</a></div>', unsafe_allow_html=True)


def _show_w1_q8_solution(prev, X1, Y1, X2, Y2, mrs1, mrs2,
                          Px, Py, price_ratio, marks):
    d = _parse_raw(prev)
    ans_m1  = float(d.get("mrs1", 0))
    ans_m2  = float(d.get("mrs2", 0))
    ans_buy = str(d.get("buy_more",""))

    m1_ok  = r2(ans_m1) == mrs1
    m2_ok  = r2(ans_m2) == mrs2
    abs_mrs1 = abs(mrs1)
    correct_buy = "X" if abs_mrs1 > price_ratio else "Y"
    buy_ok = (correct_buy=="X" and "More X" in ans_buy) or \
             (correct_buy=="Y" and "More Y" in ans_buy)
    sc = min(2*int(m1_ok)+2*int(m2_ok)+2*int(buy_ok)+2, marks)

    def chip2(ok): return '<span class="chip-ok">+2</span>' if ok else '<span class="chip-wrong">0</span>'

    st.markdown('<div class="sol-section"><div class="sol-title">Solution — Question 8</div>',
                unsafe_allow_html=True)
    st.markdown(
        f'<div class="sol-label">Step-by-step working</div>'
        f'<div class="sol-steps">'
        f'<p><strong>(b) Marginal utilities:</strong><br>'
        f'MUx = (1/2) x X^(-1/2) x Y^(1/2)<br>'
        f'MUy = (1/2) x X^(1/2) x Y^(-1/2)</p>'
        f'<p><strong>(c) MRS:</strong><br>'
        f'MRS = -MUx/MUy = -[(1/2)X^(-1/2)Y^(1/2)] / [(1/2)X^(1/2)Y^(-1/2)]<br>'
        f'= -(Y^(1/2+1/2)) / (X^(1/2+1/2)) = <strong>-Y/X</strong></p>'
        f'<p><strong>(d) Evaluating:</strong><br>'
        f'MRS at ({X1}, {Y1}) = -{Y1}/{X1} = <strong>{mrs1}</strong><br>'
        f'MRS at ({X2}, {Y2}) = -{Y2}/{X2} = <strong>{mrs2}</strong><br>'
        f'|MRS| fell from {abs(mrs1)} to {abs(mrs2)} as X increased. '
        f'Yes, MRS is <strong>diminishing</strong> — the IC bows inward.</p>'
        f'<p><strong>(e)</strong> |MRS| at ({X1},{Y1}) = {abs_mrs1}. '
        f'Price ratio Px/Py = {price_ratio}.<br>'
        f'{"Since |MRS| = " + str(abs_mrs1) + " > price ratio " + str(price_ratio) + ", the consumer values X more highly than the market does. They should buy more X." if abs_mrs1 > price_ratio else "Since |MRS| = " + str(abs_mrs1) + " < price ratio " + str(price_ratio) + ", the consumer values X less than the market rate. They should buy more Y."}'
        f'</p>'
        f'</div>',
        unsafe_allow_html=True)

    st.markdown('<div class="sol-label">Your score</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="score-row"><span class="score-row-label">'
        '(b) and (c) Conceptual — awarded automatically'
        '</span><span class="score-row-val"><span class="chip-ok">+2</span></span></div>',
        unsafe_allow_html=True)
    st.markdown(
        f'<div class="score-row"><span class="score-row-label">'
        f'(d) MRS at ({X1},{Y1}) &nbsp;·&nbsp; Correct: {mrs1} &nbsp;·&nbsp; Yours: {r2(ans_m1)}'
        f'</span><span class="score-row-val">{chip2(m1_ok)}</span></div>', unsafe_allow_html=True)
    st.markdown(
        f'<div class="score-row"><span class="score-row-label">'
        f'(d) MRS at ({X2},{Y2}) &nbsp;·&nbsp; Correct: {mrs2} &nbsp;·&nbsp; Yours: {r2(ans_m2)}'
        f'</span><span class="score-row-val">{chip2(m2_ok)}</span></div>', unsafe_allow_html=True)
    st.markdown(
        f'<div class="score-row"><span class="score-row-label">'
        f'(e) Buy more &nbsp;·&nbsp; Correct: More {correct_buy} &nbsp;·&nbsp; Yours: {ans_buy}'
        f'</span><span class="score-row-val">{chip2(buy_ok)}</span></div>', unsafe_allow_html=True)
    st.markdown(
        f'<div class="score-row score-row-total"><span>Total</span><span>{sc} / {marks}</span></div>',
        unsafe_allow_html=True)

    st.markdown(
        '<div class="sol-label">Common mistakes</div>'
        '<div class="sol-mistakes">'
        '<p>Dropping the negative sign in MRS = -Y/X. The MRS is always negative for '
        'standard goods because the IC slopes downward. When comparing to the price ratio, '
        'always use |MRS|.</p>'
        '<p>In part (e): if |MRS| > Px/Py, the consumer values X more than the market '
        'charges for it — they should buy more X, not more Y.</p>'
        '</div>', unsafe_allow_html=True)

    revise = []
    if not m1_ok or not m2_ok:
        revise.append("MRS formula: MRS = -MUx/MUy. For U = sqrt(XY), MRS = -Y/X")
    if not buy_ok:
        revise.append("Comparing |MRS| to Px/Py to determine whether to buy more X or Y")
    if revise:
        items = "".join(f"<li style='margin-bottom:0.4rem;'>{r}</li>" for r in revise)
        st.markdown(
            f'<div class="sol-label">Topics to revise</div>'
            f'<div class="sol-revise"><ul style="margin:0;padding-left:1.2rem;">'
            f'{items}</ul></div>', unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)

