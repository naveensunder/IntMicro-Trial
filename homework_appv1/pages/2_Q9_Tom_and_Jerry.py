import streamlit as st
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import datetime
from utils import (
    guard_page, SHARED_CSS, get_seed, q9_params, q9_answers,
    make_q9_graph, fig_to_b64, write_row, check_deadline, increment_reload,
)

st.set_page_config(
    page_title="Q9 — Tom & Jerry",
    page_icon="📙",
    layout="centered",
    initial_sidebar_state="expanded",
)

guard_page()
st.markdown(SHARED_CSS, unsafe_allow_html=True)

# ── State & params ────────────────────────────────────────────────────────────
email    = st.session_state["student_email"]
name     = st.session_state["student_name"]
prev_all = st.session_state.get("prev_data", {})
prev9    = prev_all.get("Q9", {})
past_dl, deadline_dt = check_deadline()
reloads  = increment_reload("Q9")
seed     = get_seed(email)
I, Px, Py, tom_a, tom_b = q9_params(email)
ANS      = q9_answers(I, Px, Py, tom_a, tom_b)
TOL      = 0.15
already_submitted = str(prev9.get("Status", "")) == "submitted"

# ── Header ─────────────────────────────────────────────────────────────────────
st.markdown(f"""
<div class="q-header">
  <div>
    <div class="q-header-title">Q9 — Tom &amp; Jerry: Corner vs Kink</div>
    <div class="q-header-sub">👤 {name} &nbsp;|&nbsp; 📧 {email}</div>
  </div>
  <div class="badge-graded">🎯 GRADED &nbsp;|&nbsp; 8 pts</div>
</div>
""", unsafe_allow_html=True)

if past_dl:
    st.markdown(
        f'<div class="deadline-closed">🔒 Submission closed — deadline was '
        f'{deadline_dt.strftime("%d %b %Y at %H:%M")}.</div>',
        unsafe_allow_html=True,
    )

# ── Parameters ────────────────────────────────────────────────────────────────
st.markdown(f"""
<div style="margin-bottom:1rem;">
  <span style="font-size:0.82rem;color:#5a6a8a;font-weight:600;
   text-transform:uppercase;letter-spacing:0.05em;">Your Parameters</span><br>
  <span class="param-chip">I = ${I}</span>
  <span class="param-chip">P<sub>x</sub> = ${Px}</span>
  <span class="param-chip">P<sub>y</sub> = ${Py}</span>
  <span class="param-chip">Tom: U = {tom_a}X + {tom_b}Y</span>
  <span style="font-size:0.78rem;color:#8a9ab0;margin-left:6px;">
    (unique to your email — tolerance ±{TOL})
  </span>
</div>
""", unsafe_allow_html=True)

# ── Question block ─────────────────────────────────────────────────────────────
st.markdown(f"""
<div class="q-body">
  <p>Tom and Jerry both have <strong>I = ${I}</strong>,
     <strong>P<sub>x</sub> = ${Px}</strong>,
     <strong>P<sub>y</sub> = ${Py}</strong>.</p>
  <ul>
    <li><strong>Tom:</strong> $U_T(X,Y) = {tom_a}X + {tom_b}Y$ &nbsp;(perfect substitutes)</li>
    <li><strong>Jerry:</strong> $U_J(X,Y) = \\min(X, Y)$ &nbsp;(perfect complements)</li>
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
    <span class="part-badge">(c) ungraded</span>
    <span class="part-text">
      Explain why their optimal bundles differ so dramatically.
    </span>
    <span class="chip-ungraded">Conceptual</span>
  </div>
</div>
""", unsafe_allow_html=True)

# ── Restore / lock banners ────────────────────────────────────────────────────
if already_submitted:
    ts_prev  = prev9.get("Timestamp", "")
    sc_prev  = prev9.get("Total_Score", "?")
    mx_prev  = prev9.get("Max_Score", 8)
    late_flag = prev9.get("Is_Late", "")
    late_note = " *(late submission)*" if late_flag == "Yes" else ""
    st.markdown(
        f'<div class="locked-banner">🔒 <strong>Q9 already submitted</strong> — '
        f'Score: <strong>{sc_prev} / {mx_prev}</strong> &nbsp;|&nbsp; '
        f'{ts_prev}{late_note}<br>'
        f'Answers are locked and shown below for reference.</div>',
        unsafe_allow_html=True,
    )
elif prev9:
    st.markdown(
        '<div class="restore-banner">📄 Previous draft restored — '
        'your saved answers are pre-filled below.</div>',
        unsafe_allow_html=True,
    )

# ── Answer inputs ─────────────────────────────────────────────────────────────
disabled = already_submitted or past_dl

def _prev_val(key, default=0.0):
    v = prev9.get(key, None)
    if v not in (None, ""):
        try: return float(v)
        except: pass
    return default

st.markdown('<div class="answer-box">', unsafe_allow_html=True)
st.markdown('<div class="answer-label">✏️ Enter your answers</div>', unsafe_allow_html=True)
st.caption(f"Enter X* and Y* for each bundle. Answers within ±{TOL} are accepted.")

st.markdown("**(a) Tom's optimal bundle:**")
col1, col2 = st.columns(2)
with col1:
    tom_x_ans = st.number_input(
        "Tom X*", value=_prev_val("Q9a_Tom_X"),
        step=0.01, format="%.4f", disabled=disabled, key="q9_tx",
    )
with col2:
    tom_y_ans = st.number_input(
        "Tom Y*", value=_prev_val("Q9a_Tom_Y"),
        step=0.01, format="%.4f", disabled=disabled, key="q9_ty",
    )

st.markdown("**(b) Jerry's optimal bundle:**")
col3, col4 = st.columns(2)
with col3:
    jerry_x_ans = st.number_input(
        "Jerry X*", value=_prev_val("Q9b_Jerry_X"),
        step=0.01, format="%.4f", disabled=disabled, key="q9_jx",
    )
with col4:
    jerry_y_ans = st.number_input(
        "Jerry Y*", value=_prev_val("Q9b_Jerry_Y"),
        step=0.01, format="%.4f", disabled=disabled, key="q9_jy",
    )
st.markdown('</div>', unsafe_allow_html=True)

# ── Validation hint ───────────────────────────────────────────────────────────
if not already_submitted and not past_dl:
    if tom_y_ans < 0 or jerry_x_ans < 0 or jerry_y_ans < 0:
        st.warning("⚠️ Quantities cannot be negative. Please check your answers.")
    if abs(jerry_x_ans - jerry_y_ans) > 0.5 and (jerry_x_ans != 0.0 or jerry_y_ans != 0.0):
        st.info("💡 Hint for Jerry: with U = min(X,Y), the optimum always satisfies X* = Y*.")

# ── Submit ────────────────────────────────────────────────────────────────────
all_filled = any(v != 0.0 for v in [tom_x_ans, tom_y_ans, jerry_x_ans, jerry_y_ans])

if not already_submitted and not past_dl:
    if all_filled:
        if st.button("💾 Submit & Save Q9", use_container_width=True):
            tok = (abs(tom_x_ans   - ANS["tom_x"])   < TOL and
                   abs(tom_y_ans   - ANS["tom_y"])   < TOL)
            jok = (abs(jerry_x_ans - ANS["jerry_x"]) < TOL and
                   abs(jerry_y_ans - ANS["jerry_y"]) < TOL)
            sc  = 4*int(tok) + 4*int(jok)
            ts  = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            late = "Yes" if past_dl else "No"

            row = [
                ts, name, email,
                "Q9", "submitted", late,
                reloads, seed,
                "", "", "", "", "", "",
                tom_x_ans, tom_y_ans, 4 if tok else 0,
                jerry_x_ans, jerry_y_ans, 4 if jok else 0,
                sc, 8,
            ]
            ok, err = write_row(row)

            prev9_new = {
                "Status": "submitted", "Timestamp": ts,
                "Total_Score": sc, "Max_Score": 8, "Is_Late": late,
                "Q9a_Tom_X": tom_x_ans, "Q9a_Tom_Y": tom_y_ans,
                "Q9b_Jerry_X": jerry_x_ans, "Q9b_Jerry_Y": jerry_y_ans,
            }
            st.session_state["prev_data"]["Q9"] = prev9_new

            def chip2(c, pts=4):
                return f'<span class="chip-ok">✓ +{pts} pts</span>' if c else f'<span class="chip-wrong">✗ +0 pts</span>'

            if ok:
                st.markdown(f"""
                <div class="confirm-banner">
                  ✅ <strong>Q9 saved successfully</strong><br>
                  <span style="font-size:0.88em;">
                    🕐 {ts} &nbsp;|&nbsp; 👤 {name} &nbsp;|&nbsp; 📧 {email}
                  </span><br><br>
                  <strong>Score: {sc} / 8</strong>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="warn-banner">
                  ⚠️ <strong>Sheet write failed</strong> ({err}). Screenshot this page.<br>
                  🕐 {ts} &nbsp;|&nbsp; Score: <strong>{sc} / 8</strong>
                </div>
                """, unsafe_allow_html=True)

            fig9 = make_q9_graph(I, Px, Py, tom_a, tom_b)
            b64  = fig_to_b64(fig9)
            import matplotlib.pyplot as plt; plt.close(fig9)

            st.markdown(f"""
            <div class="sol-card">
              <div class="sol-header">📖 Q9 — Solution</div>
              <div style="padding:1rem 1.4rem;">
                <div class="sol-steps">
                  <h4>📐 Step-by-Step</h4>
                  <p><strong>(a) Tom</strong> — compare bang-for-the-buck:<br>
                    $MU_x/P_x = {tom_a}/{Px} = {round(tom_a/Px,4)}$ &nbsp;vs&nbsp;
                    $MU_y/P_y = {tom_b}/{Py} = {round(tom_b/Py,4)}$<br>
                    $X$ gives more utility per dollar → spend all on $X$:<br>
                    $X^* = {I}/{Px} = \\mathbf{{{ANS["tom_x"]:.4g}}}$,
                    $Y^* = \\mathbf{{{ANS["tom_y"]:.4g}}}$
                  </p>
                  <p><strong>(b) Jerry</strong> — kink condition $X = Y$, into budget:<br>
                    ${Px}X + {Py}X = {I}$ &nbsp;→&nbsp;
                    ${Px+Py}X = {I}$ &nbsp;→&nbsp;
                    $X^* = Y^* = \\mathbf{{{round(ANS["jerry_x"],4)}}}$
                  </p>
                  <table class="score-table">
                    <tr><th>Part</th><th>Correct $(X^*, Y^*)$</th><th>Your Answer</th><th>Score</th></tr>
                    <tr>
                      <td>(a) Tom</td>
                      <td>({ANS["tom_x"]:.4g}, {ANS["tom_y"]:.4g})</td>
                      <td>({tom_x_ans:.4g}, {tom_y_ans:.4g})</td>
                      <td>{chip2(tok)}</td>
                    </tr>
                    <tr>
                      <td>(b) Jerry</td>
                      <td>({ANS["jerry_x"]:.4g}, {ANS["jerry_y"]:.4g})</td>
                      <td>({jerry_x_ans:.4g}, {jerry_y_ans:.4g})</td>
                      <td>{chip2(jok)}</td>
                    </tr>
                    <tr>
                      <td colspan="3"><strong>Total</strong></td>
                      <td><strong>{sc} / 8</strong></td>
                    </tr>
                  </table>
                </div>
                <div class="sol-mistakes">
                  <h4>⚠️ Common Mistakes</h4>
                  <ul>
                    <li><strong>Tom:</strong> Do not use MRS = Px/Py for linear preferences.
                        Always compare MUx/Px vs MUy/Py.</li>
                    <li><strong>Jerry:</strong> Writing X = Y alone is not enough —
                        substitute into the budget constraint for numeric values.</li>
                  </ul>
                </div>
                <div class="sol-graph">
                  <h4>📊 Reference Graphs &nbsp;<span class="chip-ungraded">Not graded</span></h4>
                  <div style="text-align:center;margin-top:0.8rem;">
                    <img src="data:image/png;base64,{b64}"
                         style="max-width:680px;border-radius:8px;
                                box-shadow:0 2px 10px rgba(0,0,0,0.12);">
                  </div>
                </div>
              </div>
            </div>
            """, unsafe_allow_html=True)

            st.rerun()
    else:
        st.info("Fill in all four answer boxes above to unlock the Submit button.")

# ── Locked: show solution ─────────────────────────────────────────────────────
elif already_submitted:
    tx  = _prev_val("Q9a_Tom_X"); ty  = _prev_val("Q9a_Tom_Y")
    jx  = _prev_val("Q9b_Jerry_X"); jy = _prev_val("Q9b_Jerry_Y")
    tok = abs(tx - ANS["tom_x"]) < TOL and abs(ty - ANS["tom_y"]) < TOL
    jok = abs(jx - ANS["jerry_x"]) < TOL and abs(jy - ANS["jerry_y"]) < TOL
    sc  = 4*int(tok) + 4*int(jok)

    def chip2(c, pts=4):
        return f'<span class="chip-ok">✓ +{pts} pts</span>' if c else f'<span class="chip-wrong">✗ +0 pts</span>'

    fig9 = make_q9_graph(I, Px, Py, tom_a, tom_b)
    b64  = fig_to_b64(fig9)
    import matplotlib.pyplot as plt; plt.close(fig9)

    st.markdown(f"""
    <div class="sol-card">
      <div class="sol-header">📖 Q9 — Your Submitted Solution</div>
      <div style="padding:1rem 1.4rem;">
        <div class="sol-steps">
          <h4>📐 Step-by-Step</h4>
          <p><strong>(a) Tom</strong>:
            $X^* = {ANS["tom_x"]:.4g}$, $Y^* = {ANS["tom_y"]:.4g}$
          </p>
          <p><strong>(b) Jerry</strong>:
            $X^* = Y^* = {round(ANS["jerry_x"],4)}$
          </p>
          <table class="score-table">
            <tr><th>Part</th><th>Correct</th><th>Your Answer</th><th>Score</th></tr>
            <tr>
              <td>(a) Tom</td>
              <td>({ANS["tom_x"]:.4g}, {ANS["tom_y"]:.4g})</td>
              <td>({tx:.4g}, {ty:.4g})</td>
              <td>{chip2(tok)}</td>
            </tr>
            <tr>
              <td>(b) Jerry</td>
              <td>({ANS["jerry_x"]:.4g}, {ANS["jerry_y"]:.4g})</td>
              <td>({jx:.4g}, {jy:.4g})</td>
              <td>{chip2(jok)}</td>
            </tr>
            <tr>
              <td colspan="3"><strong>Total</strong></td>
              <td><strong>{sc} / 8</strong></td>
            </tr>
          </table>
        </div>
        <div class="sol-mistakes">
          <h4>⚠️ Common Mistakes</h4>
          <ul>
            <li><strong>Tom:</strong> Compare MUx/Px vs MUy/Py — never use MRS for linear preferences.</li>
            <li><strong>Jerry:</strong> X = Y is the condition; substitute into budget for numbers.</li>
          </ul>
        </div>
        <div class="sol-graph">
          <h4>📊 Reference Graphs</h4>
          <div style="text-align:center;margin-top:0.8rem;">
            <img src="data:image/png;base64,{b64}"
                 style="max-width:680px;border-radius:8px;
                        box-shadow:0 2px 10px rgba(0,0,0,0.12);">
          </div>
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)
