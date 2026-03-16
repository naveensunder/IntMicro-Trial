"""
question_engine.py — Question rendering.
HWDashboard v4 — readability-first redesign.
Key changes:
- Single unified card per question (no nested boxes)
- Plain sub-part text, no mini boxes
- TF solution in plain text format (no table)
- Score breakdown as styled rows (no table)
- Collapse button on solutions
- Bold dividers between questions
- Larger fonts throughout
- Math rendered with unicode/plain text to avoid dollar sign display bugs
"""

import streamlit as st
import datetime
import numpy as np
import hashlib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import io, base64
from db import write_submission


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
        "text": "For a Cobb-Douglas utility function U = X^0.4 × Y^0.6, the consumer always spends 40% of income on X regardless of prices.",
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
            "This statement is False. Perfect complements — like left and right shoes — have "
            "L-shaped (right-angle) indifference curves with a kink at the optimal ratio. "
            "The consumer gains no additional utility from having more of one good without "
            "a matching increase in the other. Downward-sloping smooth curves describe standard "
            "preferences like Cobb-Douglas, not perfect complements."
        ),
        "topic": "Indifference curve shapes for perfect complements",
    },
    {
        "text": "For a consumer facing perfect substitutes, the optimal solution is always a corner solution.",
        "correct": True,
        "explanation": (
            "This statement is True. With perfect substitutes, the consumer treats the two goods as "
            "identical in utility terms (up to a ratio), so they always choose whichever gives more "
            "utility per dollar. This leads to spending all income on one good — a corner solution. "
            "The only exception is the knife-edge case where both goods give exactly the same "
            "utility per dollar, in which case any point on the budget line is optimal."
        ),
        "topic": "Perfect substitutes and corner solutions",
    },
]


# ════════════════════════════════════════════════════════════════════════════════
#  HOMEWORK CONFIGS
# ════════════════════════════════════════════════════════════════════════════════

ALL_HW_CONFIGS = {
    "HW_WEEK2": {
        "hw_id": "HW_WEEK2",
        "questions": [
            {"q_id": "Q1", "type": "numerical", "title": "Question 1 — Budget Constraint",  "marks": 6},
            {"q_id": "Q2", "type": "numerical", "title": "Question 2 — Tom & Jerry",         "marks": 8},
            {"q_id": "Q3", "type": "truefalse", "title": "Question 3 — True or False",       "marks": 4},
        ]
    }
}


def get_questions(hw_id: str) -> list:
    return ALL_HW_CONFIGS.get(hw_id, {}).get("questions", [])


def get_hw_summary(hw_id: str, email: str, submissions: dict) -> dict:
    hw_subs  = submissions.get(hw_id, {}) if isinstance(submissions, dict) else {}
    questions = ALL_HW_CONFIGS.get(hw_id, {}).get("questions", [])
    total_score = 0; total_max = 0; n_submitted = 0
    for q in questions:
        q_id  = q["q_id"]; max_s = q["marks"]
        sub   = hw_subs.get(q_id, {}) if isinstance(hw_subs, dict) else {}
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
    if q_type == "numerical":
        if q_id == "Q1":
            _render_q1(q_config, hw_id, email, past_deadline, grace_active, submissions)
        elif q_id == "Q2":
            _render_q2(q_config, hw_id, email, past_deadline, grace_active, submissions)
    elif q_type == "truefalse":
        _render_q3_tf(q_config, hw_id, email, past_deadline, grace_active, submissions)


# ════════════════════════════════════════════════════════════════════════════════
#  HELPERS
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
            if isinstance(d, dict):
                return d
        except Exception:
            pass
    return {}


def _status_banner(prev, max_score, past_deadline, grace_active):
    already = str(prev.get("Status","")) == "submitted"
    if already:
        sc   = prev.get("Score","?"); ts = prev.get("Timestamp","")
        late = " · Late submission" if prev.get("Is_Late","")=="Yes" else ""
        st.markdown(
            f'<div class="banner banner-locked">'
            f'🔒 Submitted — Score: <strong>{sc} / {max_score}</strong> · {ts}{late}'
            f'</div>', unsafe_allow_html=True)
    elif past_deadline and not grace_active:
        st.markdown(
            '<div class="banner banner-error">🔒 Deadline has passed.</div>',
            unsafe_allow_html=True)
    elif prev:
        st.markdown(
            '<div class="banner banner-restore">Draft restored from previous session.</div>',
            unsafe_allow_html=True)
    return already


# ════════════════════════════════════════════════════════════════════════════════
#  QUESTION 1 — BUDGET CONSTRAINT
# ════════════════════════════════════════════════════════════════════════════════

def _render_q1(q_config, hw_id, email, past_deadline, grace_active, submissions):
    I, Px, Py = _q1_params(email)
    ANS_x = r2(I / Px); ANS_y = r2(I / Py); ANS_s = r2(-Px / Py)

    prev     = _get_prev(submissions, hw_id, "Q1")
    already  = str(prev.get("Status","")) == "submitted"
    disabled = already or (past_deadline and not grace_active)

    # ── Single unified question card ───────────────────────────────────────────
    st.markdown(
        f'<div class="q-card">'
        f'<div class="q-title">Question 1 — Budget Constraint</div>'
        f'<div class="q-pts">6 points total &nbsp;·&nbsp; Parts (b) and (c) are graded</div>'
        f'<div class="q-text">'
        f'A consumer has income <strong>I = ${I}</strong>, '
        f'price of good X is <strong>Px = ${Px}</strong>, '
        f'and price of good Y is <strong>Py = ${Py}</strong>.'
        f'</div>'
        f'<div class="q-part"><span class="q-part-label">(a)</span> &nbsp;'
        f'Write the equation of the budget constraint. '
        f'<span class="q-ungraded">(Ungraded — for your own working)</span></div>'
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
        f'<div class="param-row">'
        f'Your values: &nbsp;'
        f'<span class="param-val">I = ${I}</span> &nbsp;·&nbsp; '
        f'<span class="param-val">Px = ${Px}</span> &nbsp;·&nbsp; '
        f'<span class="param-val">Py = ${Py}</span>'
        f'</div>',
        unsafe_allow_html=True
    )

    _status_banner(prev, 6, past_deadline, grace_active)

    # Inputs
    d = _parse_raw(prev)
    st.markdown('<div class="answer-label">Your Answers</div>', unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    with c1:
        xint_ans = st.number_input("(b) X-intercept", value=float(d.get("xint",0.0)),
                                   step=0.01, format="%.2f",
                                   disabled=disabled, key=f"q1x_{hw_id}")
    with c2:
        yint_ans = st.number_input("(b) Y-intercept", value=float(d.get("yint",0.0)),
                                   step=0.01, format="%.2f",
                                   disabled=disabled, key=f"q1y_{hw_id}")
    with c3:
        slope_ans = st.number_input("(c) Slope", value=float(d.get("slope",0.0)),
                                    step=0.01, format="%.2f",
                                    disabled=disabled, key=f"q1s_{hw_id}")

    st.caption("Enter answers rounded to 2 decimal places.")

    if not already and not (past_deadline and not grace_active):
        has_input = not (xint_ans==0.0 and yint_ans==0.0 and slope_ans==0.0)
        if has_input:
            if st.button("Submit Question 1", key=f"sub_q1_{hw_id}", use_container_width=True):
                _submit_q1(hw_id, email, xint_ans, yint_ans, slope_ans,
                           ANS_x, ANS_y, ANS_s, past_deadline, submissions)
                st.rerun()
        else:
            st.caption("Fill in your answers above to enable submission.")

    if already:
        with st.expander("Show / hide solution", expanded=True):
            _show_q1_solution(prev, ANS_x, ANS_y, ANS_s, I, Px, Py)


def _submit_q1(hw_id, email, xint_ans, yint_ans, slope_ans,
               ANS_x, ANS_y, ANS_s, past_deadline, submissions):
    x_ok = r2(xint_ans)==ANS_x; y_ok = r2(yint_ans)==ANS_y; s_ok = r2(slope_ans)==ANS_s
    sc   = 2*int(x_ok) + 2*int(y_ok) + 2*int(s_ok)
    ts   = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    late = "Yes" if past_deadline else "No"
    raw  = str({"xint":r2(xint_ans),"yint":r2(yint_ans),"slope":r2(slope_ans)})
    corr = str({"xint":ANS_x,"yint":ANS_y,"slope":ANS_s})
    ok, err = write_submission([ts,email,hw_id,"Q1","numerical","submitted",late,raw,sc,6,corr])
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
            f'Screenshot this. Score: {sc}/6 at {ts}</div>', unsafe_allow_html=True)


def _show_q1_solution(prev, ANS_x, ANS_y, ANS_s, I, Px, Py):
    d  = _parse_raw(prev)
    xv = float(d.get("xint",0)); yv = float(d.get("yint",0)); sv = float(d.get("slope",0))
    x_ok = r2(xv)==ANS_x; y_ok = r2(yv)==ANS_y; s_ok = r2(sv)==ANS_s
    sc   = 2*int(x_ok) + 2*int(y_ok) + 2*int(s_ok)

    st.markdown('<div class="sol-block"><div class="sol-title">Solution — Question 1</div>',
                unsafe_allow_html=True)

    # Steps — plain text, no dollar signs
    st.markdown(
        f'<div class="sol-section-label">Step-by-step working</div>'
        f'<div class="sol-steps">'
        f'<p><strong>(a) Budget constraint equation:</strong> '
        f'{Px}X + {Py}Y = {I}</p>'
        f'<p><strong>(b) X-intercept:</strong> Set Y = 0 → '
        f'{Px}X = {I} → X = {I} ÷ {Px} = <strong>{ANS_x}</strong><br>'
        f'<strong>Y-intercept:</strong> Set X = 0 → '
        f'{Py}Y = {I} → Y = {I} ÷ {Py} = <strong>{ANS_y}</strong></p>'
        f'<p><strong>(c) Slope:</strong> '
        f'−Px/Py = −{Px}/{Py} = <strong>{ANS_s}</strong></p>'
        f'</div>',
        unsafe_allow_html=True
    )

    # Score rows
    x_chip = '<span class="chip-ok">+2</span>' if x_ok else '<span class="chip-wrong">0</span>'
    y_chip = '<span class="chip-ok">+2</span>' if y_ok else '<span class="chip-wrong">0</span>'
    s_chip = '<span class="chip-ok">+2</span>' if s_ok else '<span class="chip-wrong">0</span>'

    st.markdown('<div class="sol-section-label">Your score</div>', unsafe_allow_html=True)
    st.markdown(
        f'<div class="score-row">'
        f'<span class="score-row-label">(b) X-intercept — correct answer: {ANS_x}, your answer: {r2(xv)}</span>'
        f'<span class="score-row-val">{x_chip}</span>'
        f'</div>',
        unsafe_allow_html=True
    )
    st.markdown(
        f'<div class="score-row">'
        f'<span class="score-row-label">(b) Y-intercept — correct answer: {ANS_y}, your answer: {r2(yv)}</span>'
        f'<span class="score-row-val">{y_chip}</span>'
        f'</div>',
        unsafe_allow_html=True
    )
    st.markdown(
        f'<div class="score-row">'
        f'<span class="score-row-label">(c) Slope — correct answer: {ANS_s}, your answer: {r2(sv)}</span>'
        f'<span class="score-row-val">{s_chip}</span>'
        f'</div>',
        unsafe_allow_html=True
    )
    st.markdown(
        f'<div class="score-row score-row-total">'
        f'<span>Total</span>'
        f'<span>{sc} / 6</span>'
        f'</div>',
        unsafe_allow_html=True
    )

    # Mistakes
    st.markdown(
        '<div class="sol-section-label">Common mistakes</div>'
        '<div class="sol-mistakes">'
        '<p>Writing the slope as −Py/Px instead of −Px/Py. Remember: the slope '
        'tells you how many units of Y you give up for one more unit of X, '
        'which equals the price ratio −Px/Py.</p>'
        '<p>Confusing a parallel shift (what happens when income changes) with '
        'a pivot (what happens when a price changes).</p>'
        '</div>',
        unsafe_allow_html=True
    )

    # Topics to revise
    revise = []
    if not x_ok or not y_ok:
        revise.append("Finding budget line intercepts — practice setting Y=0 to get X-intercept and X=0 to get Y-intercept")
    if not s_ok:
        revise.append("Budget line slope — review why slope = −Px/Py and its economic interpretation")

    if revise:
        items = "".join(f"<li style='margin-bottom:0.4rem;'>{r}</li>" for r in revise)
        st.markdown(
            f'<div class="sol-section-label">Topics to revise</div>'
            f'<div class="sol-revise"><ul style="margin:0;padding-left:1.2rem;">'
            f'{items}</ul></div>',
            unsafe_allow_html=True
        )

    # Graph
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    fig.patch.set_facecolor("#F0F4F8"); ax.set_facecolor("#F0F4F8")
    Xmax = I/Px; Ymax = I/Py
    Xv   = np.linspace(0, Xmax, 300); Yv = (I - Px*Xv)/Py
    ax.plot(Xv, Yv, color="#1C2B4A", lw=2.5)
    ax.fill_between(Xv, Yv, alpha=0.08, color="#1C2B4A")
    ax.plot(Xmax, 0, "o", color="#1C2B4A", ms=8, zorder=5)
    ax.plot(0, Ymax, "o", color="#1C2B4A", ms=8, zorder=5)
    ax.annotate(f"({Xmax:.2f}, 0)", xy=(Xmax,0),
                xytext=(Xmax-Xmax*0.3, Ymax*0.08),
                fontsize=10, color="#1C2B4A",
                arrowprops=dict(arrowstyle="->", color="#1C2B4A", lw=1.0))
    ax.annotate(f"(0, {Ymax:.2f})", xy=(0,Ymax),
                xytext=(Xmax*0.07, Ymax-Ymax*0.15),
                fontsize=10, color="#1C2B4A",
                arrowprops=dict(arrowstyle="->", color="#1C2B4A", lw=1.0))
    ax.set_xlabel("Quantity of X", fontsize=11); ax.set_ylabel("Quantity of Y", fontsize=11)
    ax.set_title(f"Budget Line: {Px}X + {Py}Y = {I}", fontsize=11, color="#1C2B4A")
    ax.set_xlim(0, Xmax*1.22); ax.set_ylim(0, Ymax*1.28)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    ax.tick_params(labelsize=10)
    plt.tight_layout()
    b64 = fig_to_b64(fig)

    st.markdown(
        f'<div class="sol-section-label">Reference diagram (part d)</div>'
        f'<div style="text-align:center;padding:0.5rem 0;">'
        f'<img src="data:image/png;base64,{b64}" '
        f'style="max-width:400px;width:100%;border-radius:8px;'
        f'border:1px solid #E0E0E0;"></div>',
        unsafe_allow_html=True
    )

    st.markdown('</div>', unsafe_allow_html=True)  # close sol-block


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

    st.markdown(
        f'<div class="q-card">'
        f'<div class="q-title">Question 2 — Tom &amp; Jerry</div>'
        f'<div class="q-pts">8 points total &nbsp;·&nbsp; 4 points each for parts (a) and (b)</div>'
        f'<div class="q-text">'
        f'Tom and Jerry are two consumers who both face the same budget: '
        f'income <strong>I = ${I}</strong>, price of X is <strong>Px = ${Px}</strong>, '
        f'price of Y is <strong>Py = ${Py}</strong>. '
        f'They have very different preferences:'
        f'</div>'
        f'<div class="q-part"><strong>Tom:</strong> &nbsp; U = {tom_a}X + Y &nbsp; '
        f'(perfect substitutes — Tom treats X and Y as interchangeable)</div>'
        f'<div class="q-part"><strong>Jerry:</strong> &nbsp; U = min(X, Y) &nbsp; '
        f'(perfect complements — Jerry always wants X and Y in equal proportions)</div>'
        f'<br>'
        f'<div class="q-part"><span class="q-part-label">(a) 4 points</span> &nbsp;'
        f'Find Tom\'s optimal bundle (X*, Y*).</div>'
        f'<div class="q-part"><span class="q-part-label">(b) 4 points</span> &nbsp;'
        f'Find Jerry\'s optimal bundle (X*, Y*).</div>'
        f'<div class="q-part"><span class="q-part-label">(c)</span> &nbsp;'
        f'In your own words, explain why their optimal bundles differ so dramatically. '
        f'<span class="q-ungraded">(Ungraded — good exam practice)</span></div>'
        f'</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        f'<div class="param-row">'
        f'Your values: &nbsp;'
        f'<span class="param-val">I = ${I}</span> &nbsp;·&nbsp; '
        f'<span class="param-val">Px = ${Px}</span> &nbsp;·&nbsp; '
        f'<span class="param-val">Py = ${Py}</span> &nbsp;·&nbsp; '
        f'<span class="param-val">Tom: U = {tom_a}X + Y</span>'
        f'</div>',
        unsafe_allow_html=True
    )

    _status_banner(prev, 8, past_deadline, grace_active)

    d = _parse_raw(prev)
    st.markdown('<div class="answer-label">Your Answers</div>', unsafe_allow_html=True)
    st.markdown("**(a) Tom's optimal bundle:**")
    c1, c2 = st.columns(2)
    with c1:
        tom_x = st.number_input("Tom X*", value=float(d.get("tom_x",0.0)),
                                step=0.01, format="%.2f",
                                disabled=disabled, key=f"q2tx_{hw_id}")
    with c2:
        tom_y = st.number_input("Tom Y*", value=float(d.get("tom_y",0.0)),
                                step=0.01, format="%.2f",
                                disabled=disabled, key=f"q2ty_{hw_id}")
    st.markdown("**(b) Jerry's optimal bundle:**")
    c3, c4 = st.columns(2)
    with c3:
        jerry_x = st.number_input("Jerry X*", value=float(d.get("jerry_x",0.0)),
                                  step=0.01, format="%.2f",
                                  disabled=disabled, key=f"q2jx_{hw_id}")
    with c4:
        jerry_y = st.number_input("Jerry Y*", value=float(d.get("jerry_y",0.0)),
                                  step=0.01, format="%.2f",
                                  disabled=disabled, key=f"q2jy_{hw_id}")

    st.caption("Enter answers rounded to 2 decimal places.")

    if not disabled:
        jxr = r2(jerry_x); jyr = r2(jerry_y)
        if abs(jxr - jyr) > 0.05 and (jerry_x != 0.0 or jerry_y != 0.0):
            st.markdown(
                '<div class="banner banner-info">💡 Hint for Jerry: '
                'with U = min(X, Y), the optimal bundle always satisfies X* = Y*. '
                'Substitute this into the budget constraint to find the values.</div>',
                unsafe_allow_html=True)

    has_input = any(v != 0.0 for v in [tom_x, tom_y, jerry_x, jerry_y])
    if not already and not (past_deadline and not grace_active):
        if has_input:
            if st.button("Submit Question 2", key=f"sub_q2_{hw_id}", use_container_width=True):
                _submit_q2(hw_id, email, tom_x, tom_y, jerry_x, jerry_y,
                           ANS_tx, ANS_ty, ANS_jx, ANS_jy, past_deadline)
                st.rerun()
        else:
            st.caption("Fill in your answers above to enable submission.")

    if already:
        with st.expander("Show / hide solution", expanded=True):
            _show_q2_solution(prev, ANS_tx, ANS_ty, ANS_jx, ANS_jy, I, Px, Py, tom_a)


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
            f'Screenshot this. Score: {sc}/8 at {ts}</div>', unsafe_allow_html=True)


def _show_q2_solution(prev, ANS_tx, ANS_ty, ANS_jx, ANS_jy, I, Px, Py, tom_a):
    d   = _parse_raw(prev)
    tx  = float(d.get("tom_x",0)); ty = float(d.get("tom_y",0))
    jx  = float(d.get("jerry_x",0)); jy = float(d.get("jerry_y",0))
    tok = (r2(tx)==ANS_tx and r2(ty)==ANS_ty)
    jok = (r2(jx)==ANS_jx and r2(jy)==ANS_jy)
    sc  = 4*int(tok) + 4*int(jok)

    st.markdown('<div class="sol-block"><div class="sol-title">Solution — Question 2</div>',
                unsafe_allow_html=True)

    st.markdown(
        f'<div class="sol-section-label">Step-by-step working</div>'
        f'<div class="sol-steps">'
        f'<p><strong>(a) Tom — Perfect Substitutes</strong><br>'
        f'Compare utility per dollar for each good:<br>'
        f'MUx / Px = {tom_a} / {Px} = {round(tom_a/Px,2)} &nbsp;&nbsp; vs &nbsp;&nbsp; '
        f'MUy / Py = 1 / {Py} = {round(1/Py,2)}<br>'
        f'Since X gives more utility per dollar, Tom spends all income on X:<br>'
        f'X* = {I} / {Px} = <strong>{ANS_tx}</strong>, &nbsp; Y* = <strong>0</strong></p>'
        f'<p><strong>(b) Jerry — Perfect Complements</strong><br>'
        f'At the optimal bundle, X* = Y* (the kink condition).<br>'
        f'Substitute into budget constraint: {Px}X + {Py}X = {I}<br>'
        f'{Px+Py}X = {I} → X* = {I} / {Px+Py} = <strong>{ANS_jx}</strong>, '
        f'Y* = <strong>{ANS_jy}</strong></p>'
        f'</div>',
        unsafe_allow_html=True
    )

    t_chip = '<span class="chip-ok">+4</span>' if tok else '<span class="chip-wrong">0</span>'
    j_chip = '<span class="chip-ok">+4</span>' if jok else '<span class="chip-wrong">0</span>'

    st.markdown('<div class="sol-section-label">Your score</div>', unsafe_allow_html=True)
    st.markdown(
        f'<div class="score-row">'
        f'<span class="score-row-label">(a) Tom — correct: ({ANS_tx}, {ANS_ty}), '
        f'your answer: ({r2(tx)}, {r2(ty)})</span>'
        f'<span class="score-row-val">{t_chip}</span>'
        f'</div>',
        unsafe_allow_html=True
    )
    st.markdown(
        f'<div class="score-row">'
        f'<span class="score-row-label">(b) Jerry — correct: ({ANS_jx}, {ANS_jy}), '
        f'your answer: ({r2(jx)}, {r2(jy)})</span>'
        f'<span class="score-row-val">{j_chip}</span>'
        f'</div>',
        unsafe_allow_html=True
    )
    st.markdown(
        f'<div class="score-row score-row-total">'
        f'<span>Total</span><span>{sc} / 8</span>'
        f'</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="sol-section-label">Common mistakes</div>'
        '<div class="sol-mistakes">'
        '<p><strong>For Tom:</strong> Never use MRS = Px/Py for perfect substitutes. '
        'The correct method is to compare MUx/Px with MUy/Py directly. '
        'Whichever good gives more utility per dollar gets all the spending.</p>'
        '<p><strong>For Jerry:</strong> The kink condition is X* = Y*. '
        'Many students stop there — you must substitute back into the budget '
        'constraint to find the actual numeric values.</p>'
        '</div>',
        unsafe_allow_html=True
    )

    revise = []
    if not tok:
        revise.append("Perfect substitutes — bang-per-buck comparison: always compare MUx/Px vs MUy/Py")
    if not jok:
        revise.append("Perfect complements — kink condition X* = Y* and substitution into budget constraint")
    if revise:
        items = "".join(f"<li style='margin-bottom:0.4rem;'>{r}</li>" for r in revise)
        st.markdown(
            f'<div class="sol-section-label">Topics to revise</div>'
            f'<div class="sol-revise"><ul style="margin:0;padding-left:1.2rem;">'
            f'{items}</ul></div>',
            unsafe_allow_html=True
        )

    # Graph
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
    Xt=I/Px; Yt=0.0; Xj=I/(Px+Py); Yj=Xj
    ax=axs[0]
    UT=tom_a*Xt+Yt
    for Ul,alp in [(UT*0.6,0.18),(UT*0.8,0.32),(UT,0.85)]:
        ICs=(Ul-tom_a*Xv)/1.0
        ax.plot(Xv,np.where(ICs>=0,ICs,np.nan),color="#DC2626",lw=1.5,alpha=alp)
    ax.plot(Xt,Yt,"o",color="#DC2626",ms=9,zorder=6,label=f"Optimum ({ANS_tx}, {ANS_ty})")
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
        f'<div class="sol-section-label">Reference diagrams</div>'
        f'<div style="text-align:center;padding:0.5rem 0;">'
        f'<img src="data:image/png;base64,{b64}" '
        f'style="max-width:660px;width:100%;border-radius:8px;'
        f'border:1px solid #E0E0E0;"></div>',
        unsafe_allow_html=True
    )
    st.markdown('</div>', unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════════════════
#  QUESTION 3 — TRUE/FALSE (plain text solution, no table)
# ════════════════════════════════════════════════════════════════════════════════

def _render_q3_tf(q_config, hw_id, email, past_deadline, grace_active, submissions):
    statements = TF_STATEMENTS
    correct    = [s["correct"] for s in statements]

    prev     = _get_prev(submissions, hw_id, "Q3")
    already  = str(prev.get("Status","")) == "submitted"
    disabled = already or (past_deadline and not grace_active)

    st.markdown(
        '<div class="q-card">'
        '<div class="q-title">Question 3 — True or False</div>'
        '<div class="q-pts">4 points total &nbsp;·&nbsp; 1 point per statement</div>'
        '<div class="q-text">'
        'Read each statement carefully and decide whether it is True or False. '
        'These statements test core concepts from the first two weeks of the course.'
        '</div>'
        '</div>',
        unsafe_allow_html=True
    )

    _status_banner(prev, 4, past_deadline, grace_active)

    prev_answers = {}
    raw = prev.get("Raw_Answer","")
    if raw:
        try:
            pa = eval(str(raw))
            if isinstance(pa, dict): prev_answers = pa
        except Exception: pass

    st.markdown('<div class="answer-label">Your Answers</div>', unsafe_allow_html=True)

    student_answers = []
    for i, stmt in enumerate(statements):
        prev_val = prev_answers.get(f"s{i}", None)
        default_idx = 0 if (prev_val is True or prev_val is None) else 1
        st.markdown(
            f'<div style="font-size:1rem;font-weight:500;color:#1A1A1A;'
            f'margin:1rem 0 0.3rem 0;">({i+1}) {stmt["text"]}</div>',
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

    if not already and not (past_deadline and not grace_active):
        if st.button("Submit Question 3", key=f"sub_q3tf_{hw_id}", use_container_width=True):
            _submit_q3_tf(hw_id, email, student_answers, correct, past_deadline)
            st.rerun()

    if already:
        with st.expander("Show / hide solution", expanded=True):
            _show_q3_tf_solution(prev, statements, correct)


def _submit_q3_tf(hw_id, email, student_answers, correct, past_deadline):
    sc   = sum(1 for s,c in zip(student_answers,correct) if s==c)
    ts   = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    late = "Yes" if past_deadline else "No"
    raw  = str({f"s{i}":v for i,v in enumerate(student_answers)})
    corr = str({f"s{i}":v for i,v in enumerate(correct)})
    ok, err = write_submission([ts,email,hw_id,"Q3","truefalse","submitted",late,raw,sc,4,corr])
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
            f'Screenshot this. Score: {sc}/4 at {ts}</div>', unsafe_allow_html=True)


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
        f'<div class="sol-block">'
        f'<div class="sol-title">Solution — Question 3 &nbsp; ({sc} / 4)</div>',
        unsafe_allow_html=True
    )

    revise_items = []
    for i, (stmt, corr_val, student_val) in enumerate(
            zip(statements, correct, student_answers)):
        corr_str    = "True"  if corr_val   else "False"
        student_str = ("True" if student_val else "False") if student_val is not None else "—"
        is_ok       = (student_val == corr_val) if student_val is not None else False
        result_icon = "✓" if is_ok else "✗"
        result_col  = "#16A34A" if is_ok else "#DC2626"

        if not is_ok and stmt.get("topic"):
            revise_items.append(stmt["topic"])

        # Each statement rendered as its own separate st.markdown call
        st.markdown(
            f'<div class="tf-item">'
            f'<div class="tf-stmt">({i+1}) {stmt["text"]}</div>'
            f'<div class="tf-result">'
            f'<span style="color:{result_col};font-weight:700;">{result_icon}</span> &nbsp;'
            f'<strong>Correct answer: {corr_str}</strong> &nbsp;·&nbsp; '
            f'Your answer: {student_str}'
            f'</div>'
            f'<div class="tf-expl">{stmt["explanation"]}</div>'
            f'</div>',
            unsafe_allow_html=True
        )

    if revise_items:
        items = "".join(f"<li style='margin-bottom:0.4rem;'>{r}</li>" for r in revise_items)
        st.markdown(
            f'<div class="sol-section-label" style="margin-top:1rem;">Topics to revise</div>'
            f'<div class="sol-revise"><ul style="margin:0;padding-left:1.2rem;">'
            f'{items}</ul></div>',
            unsafe_allow_html=True
        )

    st.markdown('</div>', unsafe_allow_html=True)
