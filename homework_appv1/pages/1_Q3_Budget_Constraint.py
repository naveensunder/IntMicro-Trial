import streamlit as st
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import datetime
from utils import (
    guard_page, SHARED_CSS, get_seed, q3_params, q3_answers,
    make_q3_graph, fig_to_b64, write_row, check_deadline, increment_reload,
)

st.set_page_config(
    page_title="Q3 — Budget Constraint",
    page_icon="📗",
    layout="centered",
)

# ── Auth guard ────────────────────────────────────────────────────────────────
guard_page()
st.markdown(SHARED_CSS, unsafe_allow_html=True)

# ── State & params ────────────────────────────────────────────────────────────
email    = st.session_state["student_email"]
name     = st.session_state["student_name"]
prev_all = st.session_state.get("prev_data", {})
prev3    = prev_all.get("Q3", {})
past_dl, deadline_dt = check_deadline()
reloads  = increment_reload("Q3")
seed     = get_seed(email)
I, Px, Py = q3_params(email)
ANS      = q3_answers(I, Px, Py)
TOL      = 0.5
already_submitted = str(prev3.get("Status", "")) == "submitted"

# ── Header ─────────────────────────────────────────────────────────────────────
st.markdown(f"""
<div class="q-header">
  <div>
    <div class="q-header-title">Q3 — Budget Constraint</div>
    <div class="q-header-sub">👤 {name} &nbsp;|&nbsp; 📧 {email}</div>
  </div>
  <div class="badge-graded">🎯 GRADED &nbsp;|&nbsp; 6 pts</div>
</div>
""", unsafe_allow_html=True)

# ── Deadline closed ───────────────────────────────────────────────────────────
if past_dl:
    st.markdown(
        f'<div class="deadline-closed">🔒 Submission closed — deadline was '
        f'{deadline_dt.strftime("%d %b %Y at %H:%M")}. '
        f'Your answers are read-only.</div>',
        unsafe_allow_html=True,
    )

# ── Your unique parameters ────────────────────────────────────────────────────
st.markdown(f"""
<div style="margin-bottom:1rem;">
  <span style="font-size:0.82rem;color:#5a6a8a;font-weight:600;
   text-transform:uppercase;letter-spacing:0.05em;">Your Parameters</span><br>
  <span class="param-chip">I = ${I}</span>
  <span class="param-chip">P<sub>x</sub> = ${Px}</span>
  <span class="param-chip">P<sub>y</sub> = ${Py}</span>
  <span style="font-size:0.78rem;color:#8a9ab0;margin-left:6px;">
    (unique to your email — tolerance ±{TOL})
  </span>
</div>
""", unsafe_allow_html=True)

# ── Question block ─────────────────────────────────────────────────────────────
st.markdown(f"""
<div class="q-body">
  <p>A consumer has income <strong>I = ${I}</strong>,
     <strong>P<sub>x</sub> = ${Px}</strong>,
     <strong>P<sub>y</sub> = ${Py}</strong>.</p>
  <div class="part-row">
    <span class="part-badge">(a) ungraded</span>
    <span class="part-text">Write the equation of the budget constraint.</span>
    <span class="chip-ungraded">Conceptual</span>
  </div>
  <div class="part-row">
    <span class="part-badge">(b) 4 pts</span>
    <span class="part-text">
      Find the <strong>X-intercept</strong> and <strong>Y-intercept</strong>
      of the budget line.
    </span>
  </div>
  <div class="part-row">
    <span class="part-badge">(c) 2 pts</span>
    <span class="part-text">What is the <strong>slope</strong> of the budget line?</span>
  </div>
  <div class="part-row">
    <span class="part-badge">(d) ungraded</span>
    <span class="part-text">Draw the budget line on a labelled diagram.</span>
    <span class="chip-ungraded">Practice</span>
  </div>
</div>
""", unsafe_allow_html=True)

# ── Restore / lock banners ────────────────────────────────────────────────────
if already_submitted:
    ts_prev = prev3.get("Timestamp", "")
    sc_prev = prev3.get("Total_Score", "?")
    mx_prev = prev3.get("Max_Score", 6)
    late_flag = prev3.get("Is_Late", "")
    late_note = " *(late submission)*" if late_flag == "Yes" else ""
    st.markdown(
        f'<div class="locked-banner">🔒 <strong>Q3 already submitted</strong> — '
        f'Score: <strong>{sc_prev} / {mx_prev}</strong> &nbsp;|&nbsp; '
        f'{ts_prev}{late_note}<br>'
        f'Answers are locked and shown below for reference.</div>',
        unsafe_allow_html=True,
    )
elif prev3:
    st.markdown(
        '<div class="restore-banner">📄 Previous draft restored — '
        'your saved answers are pre-filled below.</div>',
        unsafe_allow_html=True,
    )

# ── Answer inputs ─────────────────────────────────────────────────────────────
st.markdown('<div class="answer-box">', unsafe_allow_html=True)
st.markdown('<div class="answer-label">✏️ Enter your answers</div>', unsafe_allow_html=True)
st.caption("Submit button appears once all three boxes are filled. Answers within ±0.5 are accepted.")

disabled = already_submitted or past_dl

def _prev_val(key, default=0.0):
    v = prev3.get(key, None)
    if v not in (None, ""):
        try: return float(v)
        except: pass
    return default

col1, col2, col3 = st.columns(3)
with col1:
    x_int_ans = st.number_input(
        "(b) X-intercept", value=_prev_val("Q3b_X_Int"),
        step=0.1, format="%.2f", disabled=disabled, key="q3_xint",
    )
with col2:
    y_int_ans = st.number_input(
        "(b) Y-intercept", value=_prev_val("Q3b_Y_Int"),
        step=0.1, format="%.2f", disabled=disabled, key="q3_yint",
    )
with col3:
    slope_ans = st.number_input(
        "(c) Slope", value=_prev_val("Q3c_Slope"),
        step=0.01, format="%.4f", disabled=disabled, key="q3_slope",
    )
st.markdown('</div>', unsafe_allow_html=True)

# ── Validation warning ────────────────────────────────────────────────────────
if not already_submitted and not past_dl:
    if x_int_ans < 0 or y_int_ans < 0:
        st.warning("⚠️ Intercepts should be positive for a standard budget line. Please check your answer.")

# ── Submit button ─────────────────────────────────────────────────────────────
all_filled = (x_int_ans != 0.0 or y_int_ans != 0.0 or slope_ans != 0.0)

if not already_submitted and not past_dl:
    if all_filled:
        if st.button("💾 Submit & Save Q3", use_container_width=True):
            x_ok = abs(x_int_ans - ANS["xint"])  < TOL
            y_ok = abs(y_int_ans - ANS["yint"])  < TOL
            s_ok = abs(slope_ans - ANS["slope"]) < TOL
            sc   = 2*int(x_ok) + 2*int(y_ok) + 2*int(s_ok)
            ts   = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            late = "Yes" if past_dl else "No"

            row = [
                ts, name, email,
                "Q3", "submitted", late,
                reloads, seed,
                x_int_ans, 2 if x_ok else 0,
                y_int_ans, 2 if y_ok else 0,
                slope_ans, 2 if s_ok else 0,
                "", "", "", "", "", "",
                sc, 6,
            ]
            ok, err = write_row(row)

            # Update session state so progress bar refreshes on home
            prev3_new = {
                "Status": "submitted", "Timestamp": ts,
                "Total_Score": sc, "Max_Score": 6, "Is_Late": late,
                "Q3b_X_Int": x_int_ans, "Q3b_Y_Int": y_int_ans, "Q3c_Slope": slope_ans,
            }
            st.session_state["prev_data"]["Q3"] = prev3_new

            def chip(c, pts=2):
                return f'<span class="chip-ok">✓ +{pts} pts</span>' if c else f'<span class="chip-wrong">✗ +0 pts</span>'

            if ok:
                st.markdown(f"""
                <div class="confirm-banner">
                  ✅ <strong>Q3 saved successfully</strong><br>
                  <span style="font-size:0.88em;">
                    🕐 {ts} &nbsp;|&nbsp; 👤 {name} &nbsp;|&nbsp; 📧 {email}
                  </span><br><br>
                  <strong>Score: {sc} / 6</strong>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="warn-banner">
                  ⚠️ <strong>Sheet write failed</strong> ({err}). Please screenshot this page.<br>
                  🕐 {ts} &nbsp;|&nbsp; Score: <strong>{sc} / 6</strong>
                </div>
                """, unsafe_allow_html=True)

            # Solution
            fig3 = make_q3_graph(I, Px, Py)
            b64  = fig_to_b64(fig3)
            import matplotlib.pyplot as plt; plt.close(fig3)

            st.markdown(f"""
            <div class="sol-card">
              <div class="sol-header">📖 Q3 — Solution</div>
              <div style="padding:1rem 1.4rem;">
                <div class="sol-steps">
                  <h4>📐 Step-by-Step</h4>
                  <p><strong>(a)</strong> Budget constraint: ${Px}X + {Py}Y = {I}$</p>
                  <p><strong>(b)</strong>
                    Set $Y=0$: $X_{{\\max}} = {I}/{Px} = {ANS["xint"]:.4g}$
                    &nbsp; | &nbsp;
                    Set $X=0$: $Y_{{\\max}} = {I}/{Py} = {ANS["yint"]:.4g}$
                  </p>
                  <p><strong>(c)</strong>
                    Slope $= -P_x/P_y = -{Px}/{Py} = {ANS["slope"]:.4g}$
                  </p>
                  <table class="score-table">
                    <tr><th>Part</th><th>Correct</th><th>Your Answer</th><th>Score</th></tr>
                    <tr>
                      <td>(b) X-intercept</td>
                      <td>{ANS["xint"]:.4g}</td>
                      <td>{x_int_ans:.4g}</td>
                      <td>{chip(x_ok)}</td>
                    </tr>
                    <tr>
                      <td>(b) Y-intercept</td>
                      <td>{ANS["yint"]:.4g}</td>
                      <td>{y_int_ans:.4g}</td>
                      <td>{chip(y_ok)}</td>
                    </tr>
                    <tr>
                      <td>(c) Slope</td>
                      <td>{ANS["slope"]:.4g}</td>
                      <td>{slope_ans:.4g}</td>
                      <td>{chip(s_ok)}</td>
                    </tr>
                    <tr><td colspan="3"><strong>Total</strong></td><td><strong>{sc} / 6</strong></td></tr>
                  </table>
                </div>
                <div class="sol-mistakes">
                  <h4>⚠️ Common Mistakes</h4>
                  <ul>
                    <li>Writing slope as $-P_y/P_x$ instead of $-P_x/P_y$.</li>
                    <li>Confusing a <strong>shift</strong> (income change)
                        with a <strong>pivot</strong> (price change).</li>
                  </ul>
                </div>
                <div class="sol-graph">
                  <h4>📊 (d) Reference Graph &nbsp;<span class="chip-ungraded">Not graded</span></h4>
                  <div style="text-align:center;margin-top:0.8rem;">
                    <img src="data:image/png;base64,{b64}"
                         style="max-width:440px;border-radius:8px;
                                box-shadow:0 2px 10px rgba(0,0,0,0.12);">
                  </div>
                </div>
              </div>
            </div>
            """, unsafe_allow_html=True)

            st.rerun()
    else:
        st.info("Fill in all three answer boxes above to unlock the Submit button.")

# ── Show locked solution if already submitted ─────────────────────────────────
elif already_submitted:
    x_ok = abs(_prev_val("Q3b_X_Int") - ANS["xint"])  < TOL
    y_ok = abs(_prev_val("Q3b_Y_Int") - ANS["yint"])  < TOL
    s_ok = abs(_prev_val("Q3c_Slope") - ANS["slope"]) < TOL
    sc   = 2*int(x_ok) + 2*int(y_ok) + 2*int(s_ok)

    def chip(c, pts=2):
        return f'<span class="chip-ok">✓ +{pts} pts</span>' if c else f'<span class="chip-wrong">✗ +0 pts</span>'

    fig3 = make_q3_graph(I, Px, Py)
    b64  = fig_to_b64(fig3)
    import matplotlib.pyplot as plt; plt.close(fig3)

    st.markdown(f"""
    <div class="sol-card">
      <div class="sol-header">📖 Q3 — Your Submitted Solution</div>
      <div style="padding:1rem 1.4rem;">
        <div class="sol-steps">
          <h4>📐 Step-by-Step</h4>
          <p><strong>(a)</strong> Budget constraint: ${Px}X + {Py}Y = {I}$</p>
          <p><strong>(b)</strong>
            $X_{{\\max}} = {ANS["xint"]:.4g}$ &nbsp; | &nbsp;
            $Y_{{\\max}} = {ANS["yint"]:.4g}$
          </p>
          <p><strong>(c)</strong> Slope $= {ANS["slope"]:.4g}$</p>
          <table class="score-table">
            <tr><th>Part</th><th>Correct</th><th>Your Answer</th><th>Score</th></tr>
            <tr>
              <td>(b) X-intercept</td><td>{ANS["xint"]:.4g}</td>
              <td>{_prev_val("Q3b_X_Int"):.4g}</td><td>{chip(x_ok)}</td>
            </tr>
            <tr>
              <td>(b) Y-intercept</td><td>{ANS["yint"]:.4g}</td>
              <td>{_prev_val("Q3b_Y_Int"):.4g}</td><td>{chip(y_ok)}</td>
            </tr>
            <tr>
              <td>(c) Slope</td><td>{ANS["slope"]:.4g}</td>
              <td>{_prev_val("Q3c_Slope"):.4g}</td><td>{chip(s_ok)}</td>
            </tr>
            <tr><td colspan="3"><strong>Total</strong></td><td><strong>{sc} / 6</strong></td></tr>
          </table>
        </div>
        <div class="sol-mistakes">
          <h4>⚠️ Common Mistakes</h4>
          <ul>
            <li>Writing slope as $-P_y/P_x$ instead of $-P_x/P_y$.</li>
            <li>Confusing a shift (income change) with a pivot (price change).</li>
          </ul>
        </div>
        <div class="sol-graph">
          <h4>📊 Reference Graph</h4>
          <div style="text-align:center;margin-top:0.8rem;">
            <img src="data:image/png;base64,{b64}"
                 style="max-width:440px;border-radius:8px;
                        box-shadow:0 2px 10px rgba(0,0,0,0.12);">
          </div>
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)
