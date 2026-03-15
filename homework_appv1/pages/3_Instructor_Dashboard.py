import streamlit as st
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from utils import get_sheet, SHARED_CSS

st.set_page_config(
    page_title="Instructor Dashboard",
    page_icon="🔐",
    layout="wide",
)

st.markdown(SHARED_CSS, unsafe_allow_html=True)

# ── Extra dashboard CSS ───────────────────────────────────────────────────────
st.markdown("""
<style>
.dash-header {
    background: linear-gradient(135deg, #1a2340 0%, #2c3e70 100%);
    border-radius: 14px; padding: 1.4rem 2rem; margin-bottom: 1.5rem;
    box-shadow: 0 6px 24px rgba(15,32,68,0.20);
}
.dash-header h1 {
    font-family: 'Playfair Display', serif; color: #fff;
    font-size: 1.5rem; font-weight: 700; margin: 0 0 0.2rem 0;
}
.dash-header .sub { color: #a8c4e8; font-size: 0.88rem; }
.metric-card {
    background: #fff; border: 1px solid #dde4f5; border-radius: 12px;
    padding: 1.1rem 1.4rem; text-align: center;
    box-shadow: 0 2px 10px rgba(26,58,107,0.07);
}
.metric-num {
    font-family: 'Playfair Display', serif; font-size: 2rem;
    font-weight: 700; color: #1a3a6b;
}
.metric-label { font-size: 0.82rem; color: #6a7a9a; margin-top: 0.2rem; }
.flag-row { background: #fff3e0; border-left: 3px solid #f57c00; }
.late-row { background: #fce4ec; }
.section-head {
    font-family: 'Playfair Display', serif; font-size: 1.05rem;
    font-weight: 600; color: #0f2044; margin: 1.4rem 0 0.5rem 0;
    padding-bottom: 0.3rem; border-bottom: 2px solid #e0e8f8;
}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="dash-header">
  <h1>🔐 Instructor Dashboard</h1>
  <div class="sub">Live submission data — Week 2 Graded Assignment</div>
</div>
""", unsafe_allow_html=True)

# ── Password gate ─────────────────────────────────────────────────────────────
if not st.session_state.get("instructor_auth"):
    pwd = st.text_input("Instructor password", type="password",
                        placeholder="Enter password to access dashboard")
    if st.button("Unlock Dashboard"):
        correct = st.secrets.get("INSTRUCTOR_PASSWORD", "changeme123")
        if pwd == correct:
            st.session_state["instructor_auth"] = True
            st.rerun()
        else:
            st.error("Incorrect password.")
    st.stop()

# ── Load data ─────────────────────────────────────────────────────────────────
with st.spinner("Loading submission data..."):
    ws = get_sheet()
    if ws is None:
        st.error("Could not connect to Google Sheet.")
        st.stop()
    try:
        rows = ws.get_all_records()
        df   = pd.DataFrame(rows)
    except Exception as e:
        st.error(f"Error reading sheet: {e}")
        st.stop()

if df.empty:
    st.info("No submissions yet.")
    st.stop()

if st.button("🔄 Refresh Data"):
    st.rerun()

# ── Normalise columns ─────────────────────────────────────────────────────────
for col in ["Total_Score", "Max_Score", "Attempt_Reloads"]:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

df_q3 = df[df["Question_ID"] == "Q3"].copy()
df_q9 = df[df["Question_ID"] == "Q9"].copy()

# Deduplicate: keep most recent per student per question
for d in [df_q3, df_q9]:
    if "Timestamp" in d.columns:
        d.sort_values("Timestamp", inplace=True)
df_q3_latest = df_q3.drop_duplicates(subset=["School_Email"], keep="last") if not df_q3.empty else df_q3
df_q9_latest = df_q9.drop_duplicates(subset=["School_Email"], keep="last") if not df_q9.empty else df_q9

all_emails   = set(df["School_Email"].str.lower().unique())
done_q3      = set(df_q3_latest[df_q3_latest["Status"]=="submitted"]["School_Email"].str.lower())
done_q9      = set(df_q9_latest[df_q9_latest["Status"]=="submitted"]["School_Email"].str.lower())
done_both    = done_q3 & done_q9
started_only = all_emails - done_both

# ── Top metrics ───────────────────────────────────────────────────────────────
c1, c2, c3, c4, c5 = st.columns(5)
for col, num, lbl in [
    (c1, len(all_emails),  "Students\nStarted"),
    (c2, len(done_q3),     "Q3\nSubmitted"),
    (c3, len(done_q9),     "Q9\nSubmitted"),
    (c4, len(done_both),   "Both\nComplete"),
    (c5, len(all_emails) - len(done_both), "Not Yet\nComplete"),
]:
    col.markdown(f"""
    <div class="metric-card">
      <div class="metric-num">{num}</div>
      <div class="metric-label">{lbl}</div>
    </div>
    """, unsafe_allow_html=True)

# ── Score distributions ───────────────────────────────────────────────────────
st.markdown('<div class="section-head">📊 Score Distributions</div>', unsafe_allow_html=True)

fig_cols = st.columns(2)
for idx, (label, df_sub, max_sc) in enumerate([
    ("Q3 (Budget Constraint)", df_q3_latest, 6),
    ("Q9 (Tom & Jerry)",       df_q9_latest, 8),
]):
    with fig_cols[idx]:
        if df_sub.empty or "Total_Score" not in df_sub.columns:
            st.info(f"No {label} data yet.")
            continue
        scores = df_sub[df_sub["Status"]=="submitted"]["Total_Score"].dropna()
        if scores.empty:
            st.info(f"No submitted scores for {label}.")
            continue
        fig, ax = plt.subplots(figsize=(5, 3.5))
        fig.patch.set_facecolor("#f8faff")
        ax.set_facecolor("#f8faff")
        bins = list(range(0, max_sc + 2))
        ax.hist(scores, bins=bins, color="#1a3a6b", edgecolor="white",
                rwidth=0.75, align="left")
        ax.set_xlabel("Score", fontsize=10)
        ax.set_ylabel("Number of students", fontsize=10)
        ax.set_title(f"{label}\nMean: {scores.mean():.1f} / {max_sc}  |  Median: {scores.median():.1f}",
                     fontsize=9.5)
        ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
        plt.tight_layout()
        st.pyplot(fig); plt.close(fig)
        st.caption(f"n={len(scores)} submitted | mean={scores.mean():.2f} | "
                   f"min={int(scores.min())} | max={int(scores.max())}")

# ── Plagiarism / copying flags ─────────────────────────────────────────────────
st.markdown('<div class="section-head">🚩 Potential Copying Flags</div>', unsafe_allow_html=True)
st.caption("Students with *identical* submitted answers but *different* question parameters — strong copying signal.")

flags = []
for df_sub, qid in [(df_q3_latest, "Q3"), (df_q9_latest, "Q9")]:
    if df_sub.empty: continue
    sub = df_sub[df_sub["Status"] == "submitted"].copy()
    if qid == "Q3":
        ans_cols = ["Q3b_X_Int", "Q3b_Y_Int", "Q3c_Slope"]
    else:
        ans_cols = ["Q9a_Tom_X", "Q9a_Tom_Y", "Q9b_Jerry_X", "Q9b_Jerry_Y"]
    ans_cols = [c for c in ans_cols if c in sub.columns]
    if not ans_cols: continue
    sub["_ans_key"] = sub[ans_cols].astype(str).agg("|".join, axis=1)
    dup_keys = sub[sub.duplicated("_ans_key", keep=False)]["_ans_key"].unique()
    for key in dup_keys:
        group = sub[sub["_ans_key"] == key]
        seeds = group["Param_Seed"].astype(str).unique() if "Param_Seed" in group.columns else []
        if len(seeds) > 1:  # same answers, different seeds → flag
            for _, row in group.iterrows():
                flags.append({
                    "Question": qid,
                    "Name":  row.get("Student_Name", ""),
                    "Email": row.get("School_Email", ""),
                    "Seed":  row.get("Param_Seed", ""),
                    "Answers": key[:60],
                })

if flags:
    flag_df = pd.DataFrame(flags)
    st.warning(f"⚠️ {len(flag_df)} flagged entries across {flag_df['Question'].nunique()} question(s).")
    st.dataframe(flag_df, use_container_width=True)
else:
    st.success("✅ No copying flags detected.")

# ── Late submissions ──────────────────────────────────────────────────────────
st.markdown('<div class="section-head">🕐 Late Submissions</div>', unsafe_allow_html=True)
if "Is_Late" in df.columns:
    late_df = df[(df["Is_Late"] == "Yes") & (df["Status"] == "submitted")][
        ["Student_Name", "School_Email", "Question_ID", "Timestamp", "Total_Score", "Max_Score"]
    ]
    if late_df.empty:
        st.success("✅ No late submissions.")
    else:
        st.warning(f"{len(late_df)} late submission(s).")
        st.dataframe(late_df, use_container_width=True)
else:
    st.info("Late flag column not found in sheet.")

# ── Students not yet complete ─────────────────────────────────────────────────
st.markdown('<div class="section-head">📋 Students Not Yet Complete</div>', unsafe_allow_html=True)
if started_only:
    not_done = []
    for em in sorted(started_only):
        row = df[df["School_Email"].str.lower() == em].iloc[-1]
        not_done.append({
            "Name":  row.get("Student_Name", ""),
            "Email": em,
            "Q3":    "✅" if em in done_q3 else "❌",
            "Q9":    "✅" if em in done_q9 else "❌",
            "Last Activity": row.get("Timestamp", ""),
        })
    st.dataframe(pd.DataFrame(not_done), use_container_width=True)
else:
    st.success("✅ All students who started have submitted both questions.")

# ── Full raw data ─────────────────────────────────────────────────────────────
st.markdown('<div class="section-head">📥 Full Submission Log</div>', unsafe_allow_html=True)
with st.expander("Show all rows (raw)"):
    st.dataframe(df, use_container_width=True)
    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button("⬇ Download CSV", csv, "submissions.csv", "text/csv")
