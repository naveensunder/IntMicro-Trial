"""
pages/Instructor.py — Instructor dashboard with full management controls.
Password-protected, invisible to students.
"""
import streamlit as st
import datetime
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from db import (
    verify_instructor, update_instructor_password,
    get_all_students, register_student, delete_student, update_password,
    get_homework_configs, update_homework_config, add_homework_config,
    get_all_submissions, get_enrollment_key, log_audit,
    get_tab, TAB_CONFIG, _update_config_value
)
from ui import inject_css, page_header, COLORS

st.set_page_config(
    page_title="Instructor Dashboard",
    page_icon="🔐",
    layout="wide",
    initial_sidebar_state="expanded",
)

inject_css()

# ── Extra dashboard CSS ───────────────────────────────────────────────────────
st.markdown(f"""
<style>
.block-container {{ max-width: 1200px; }}
.dash-metric {{
    background: {COLORS['white']};
    border: 1px solid {COLORS['neutral_200']};
    border-radius: 10px;
    padding: 1.1rem 1.4rem;
    text-align: center;
}}
.dash-metric-num {{
    font-family: 'DM Serif Display', serif;
    font-size: 2rem; color: {COLORS['navy']}; line-height: 1;
}}
.dash-metric-label {{
    font-size: 0.75rem; color: {COLORS['neutral_500']};
    margin-top: 0.3rem; font-weight: 500;
    text-transform: uppercase; letter-spacing: 0.06em;
}}
.section-head {{
    font-family: 'DM Serif Display', serif;
    font-size: 1.1rem; color: {COLORS['navy']};
    margin: 1.8rem 0 0.8rem 0;
    padding-bottom: 0.4rem;
    border-bottom: 1px solid {COLORS['neutral_200']};
}}
</style>
""", unsafe_allow_html=True)

# ── Auth gate ─────────────────────────────────────────────────────────────────
if not st.session_state.get("instructor_auth"):
    st.markdown(f"""
    <div style="max-width:420px;margin:3rem auto;">
      <div style="font-family:'DM Serif Display',serif;font-size:1.4rem;
           color:{COLORS['navy']};margin-bottom:0.5rem;text-align:center;">
        Instructor Access
      </div>
      <div style="font-size:0.84rem;color:{COLORS['neutral_500']};
           text-align:center;margin-bottom:1.5rem;">
        Learning Intermediate Microeconomics
      </div>
    </div>
    """, unsafe_allow_html=True)

    col = st.columns([1,2,1])[1]
    with col:
        pwd = st.text_input("Instructor password", type="password")
        if st.button("Sign In", use_container_width=True):
            if verify_instructor(pwd):
                st.session_state["instructor_auth"] = True
                log_audit("instructor", "LOGIN", "")
                st.rerun()
            else:
                st.error("Incorrect password.")
    st.stop()

# ── Dashboard ─────────────────────────────────────────────────────────────────
page_header("Learning Intermediate Microeconomics",
            "Instructor Dashboard",
            f"Session started · {datetime.datetime.now().strftime('%d %b %Y, %H:%M')}")

# Tabs
tabs = st.tabs([
    "📊 Overview",
    "📚 Homework Manager",
    "👥 Student Manager",
    "🚩 Flags & Integrity",
    "⚙️ Settings",
])

# ════════════════════════════════════════════════════════════════════════════════
#  TAB 1: OVERVIEW
# ════════════════════════════════════════════════════════════════════════════════
with tabs[0]:
    if st.button("↻ Refresh data", key="refresh_overview"):
        st.rerun()

    with st.spinner("Loading..."):
        students    = get_all_students()
        submissions = get_all_submissions()
        hw_configs  = get_homework_configs()

    df_sub = pd.DataFrame(submissions) if submissions else pd.DataFrame()

    # Top metrics
    n_students = len(students)
    n_subs     = len(df_sub)
    n_complete = 0
    if not df_sub.empty and "Status" in df_sub.columns:
        n_complete = len(df_sub[df_sub["Status"] == "submitted"])

    c1,c2,c3,c4 = st.columns(4)
    for col, num, lbl in [
        (c1, n_students,  "Enrolled\nStudents"),
        (c2, n_complete,  "Submissions\nReceived"),
        (c3, len(hw_configs), "Homework\nAssignments"),
        (c4, len([h for h in hw_configs if h.get("Enabled","").upper()=="TRUE"]), "Currently\nOpen"),
    ]:
        col.markdown(f"""
        <div class="dash-metric">
          <div class="dash-metric-num">{num}</div>
          <div class="dash-metric-label">{lbl}</div>
        </div>
        """, unsafe_allow_html=True)

    # Enrollment key
    st.markdown('<div class="section-head">Enrollment Key</div>', unsafe_allow_html=True)
    enroll_key = get_enrollment_key()
    st.markdown(f"""
    <div style="background:{COLORS['neutral_50']};border:1px solid {COLORS['neutral_200']};
         border-radius:8px;padding:1rem 1.4rem;display:flex;align-items:center;gap:1rem;">
      <div>
        <div style="font-size:0.7rem;font-weight:600;text-transform:uppercase;
             letter-spacing:0.08em;color:{COLORS['neutral_500']};margin-bottom:0.3rem;">
          Current Enrollment Key (valid 6 months)
        </div>
        <div style="font-family:monospace;font-size:1.4rem;font-weight:700;
             color:{COLORS['navy']};letter-spacing:0.15em;">{enroll_key}</div>
      </div>
      <div style="font-size:0.82rem;color:{COLORS['neutral_500']};">
        Share this with your students so they can create an account.
        It rotates automatically every 6 months.
      </div>
    </div>
    """, unsafe_allow_html=True)

    # Score distributions
    if not df_sub.empty:
        st.markdown('<div class="section-head">Score Distributions</div>', unsafe_allow_html=True)
        hw_ids = df_sub["Homework_ID"].unique() if "Homework_ID" in df_sub.columns else []
        if len(hw_ids) > 0:
            fig_cols = st.columns(min(len(hw_ids), 3))
            for idx, hw_id in enumerate(hw_ids[:3]):
                hw_df = df_sub[(df_sub["Homework_ID"]==hw_id) & (df_sub["Status"]=="submitted")]
                hw_df = hw_df.drop_duplicates(subset=["Email","Question_ID"], keep="last")
                if hw_df.empty: continue
                scores = pd.to_numeric(hw_df["Score"], errors="coerce").dropna()
                if scores.empty: continue
                with fig_cols[idx]:
                    fig, ax = plt.subplots(figsize=(4.5, 3))
                    fig.patch.set_facecolor("#FAFAFA"); ax.set_facecolor("#FAFAFA")
                    ax.hist(scores, bins=10, color=COLORS["navy"], edgecolor="white", rwidth=0.8)
                    ax.set_xlabel("Score", fontsize=9); ax.set_ylabel("Count", fontsize=9)
                    ax.set_title(f"{hw_id}\nMean: {scores.mean():.1f}", fontsize=9)
                    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
                    ax.tick_params(labelsize=8)
                    plt.tight_layout()
                    st.pyplot(fig); plt.close(fig)

    # Late submissions
    if not df_sub.empty and "Is_Late" in df_sub.columns:
        late = df_sub[(df_sub["Is_Late"]=="Yes") & (df_sub["Status"]=="submitted")]
        if not late.empty:
            st.markdown('<div class="section-head">Late Submissions</div>', unsafe_allow_html=True)
            st.dataframe(late[["Timestamp","Email","Homework_ID","Question_ID","Score","Max_Score"]],
                         use_container_width=True)

    # Full log
    st.markdown('<div class="section-head">Full Submission Log</div>', unsafe_allow_html=True)
    with st.expander("Show all rows"):
        if not df_sub.empty:
            st.dataframe(df_sub, use_container_width=True)
            csv = df_sub.to_csv(index=False).encode("utf-8")
            st.download_button("⬇ Download CSV", csv, "submissions.csv", "text/csv")
        else:
            st.info("No submissions yet.")

# ════════════════════════════════════════════════════════════════════════════════
#  TAB 2: HOMEWORK MANAGER
# ════════════════════════════════════════════════════════════════════════════════
with tabs[1]:
    st.markdown('<div class="section-head">Manage Assignments</div>', unsafe_allow_html=True)
    st.caption("Changes take effect immediately — no redeployment needed.")

    hw_configs = get_homework_configs()

    if not hw_configs:
        st.info("No homework assignments configured yet.")
    else:
        for cfg in sorted(hw_configs, key=lambda x: x.get("HW_ID","")):
            hw_id   = cfg.get("HW_ID","")
            title   = cfg.get("Title","")
            enabled = cfg.get("Enabled","FALSE").upper() == "TRUE"
            dl      = cfg.get("Deadline","")
            grace   = cfg.get("Grace_Minutes","15")
            ann     = cfg.get("Announcement","")

            with st.expander(f"**{title}** ({hw_id})", expanded=False):
                col_en, col_dl, col_gr = st.columns([1,2,1])

                with col_en:
                    new_enabled = st.toggle("Enabled", value=enabled, key=f"en_{hw_id}")
                    if new_enabled != enabled:
                        update_homework_config(hw_id, "Enabled", str(new_enabled).upper())
                        log_audit("instructor", "HW_ENABLE_TOGGLE",
                                  f"{hw_id} -> {new_enabled}")
                        st.success("Saved.")

                with col_dl:
                    try:
                        dl_dt = datetime.datetime.strptime(dl.strip(), "%Y-%m-%d %H:%M")
                        dl_date_val = dl_dt.date()
                        dl_time_val = dl_dt.time()
                    except Exception:
                        dl_date_val = datetime.date.today()
                        dl_time_val = datetime.time(23, 59)

                    new_date = st.date_input("Deadline date", value=dl_date_val, key=f"dl_d_{hw_id}")
                    new_time = st.time_input("Deadline time", value=dl_time_val, key=f"dl_t_{hw_id}")
                    new_dl_str = f"{new_date.strftime('%Y-%m-%d')} {new_time.strftime('%H:%M')}"

                    if new_dl_str != dl:
                        if st.button("Update deadline", key=f"save_dl_{hw_id}"):
                            update_homework_config(hw_id, "Deadline", new_dl_str)
                            log_audit("instructor", "HW_DEADLINE_CHANGE",
                                      f"{hw_id} -> {new_dl_str}")
                            st.success(f"Deadline updated to {new_dl_str}")
                            st.rerun()

                with col_gr:
                    new_grace = st.number_input("Grace period (min)", value=int(grace),
                                                min_value=0, max_value=120, key=f"gr_{hw_id}")
                    if str(new_grace) != str(grace):
                        if st.button("Save grace", key=f"save_gr_{hw_id}"):
                            update_homework_config(hw_id, "Grace_Minutes", str(new_grace))
                            log_audit("instructor", "HW_GRACE_CHANGE",
                                      f"{hw_id} -> {new_grace}min")
                            st.success("Saved.")

                new_ann = st.text_input("Announcement (shown to students)", value=ann,
                                        key=f"ann_{hw_id}")
                if new_ann != ann:
                    if st.button("Save announcement", key=f"save_ann_{hw_id}"):
                        update_homework_config(hw_id, "Announcement", new_ann)
                        log_audit("instructor", "HW_ANNOUNCEMENT", f"{hw_id}: {new_ann}")
                        st.success("Announcement updated.")

    st.markdown('<div class="section-head">Add New Assignment</div>', unsafe_allow_html=True)
    with st.expander("Add homework"):
        n_id    = st.text_input("Homework ID (e.g. HW_WEEK3)", key="new_hw_id")
        n_title = st.text_input("Title", key="new_hw_title")
        n_en    = st.toggle("Enable immediately", key="new_hw_en")
        n_dl    = st.date_input("Deadline date", key="new_hw_dl")
        n_tm    = st.time_input("Deadline time", key="new_hw_tm")
        n_gr    = st.number_input("Grace minutes", value=15, key="new_hw_gr")
        n_mx    = st.number_input("Max marks", value=14, key="new_hw_mx")
        if st.button("Add Assignment", key="add_hw_btn"):
            if n_id and n_title:
                dl_str = f"{n_dl.strftime('%Y-%m-%d')} {n_tm.strftime('%H:%M')}"
                ok = add_homework_config(n_id, n_title, str(n_en).upper(),
                                         dl_str, n_gr, "", 1, n_mx)
                if ok:
                    log_audit("instructor", "HW_ADDED", f"{n_id}: {n_title}")
                    st.success(f"Added {n_id}.")
                    st.rerun()
                else:
                    st.error("Failed to add. Check sheet connection.")
            else:
                st.error("ID and title are required.")

# ════════════════════════════════════════════════════════════════════════════════
#  TAB 3: STUDENT MANAGER
# ════════════════════════════════════════════════════════════════════════════════
with tabs[2]:
    st.markdown('<div class="section-head">Enrolled Students</div>', unsafe_allow_html=True)

    students = get_all_students()

    if st.button("↻ Refresh students", key="refresh_students"):
        st.rerun()

    if students:
        df_students = pd.DataFrame(students)
        # Show safe columns only
        show_cols = [c for c in ["Email","First_Name","Last_Name","Registered_At","Last_Login"]
                     if c in df_students.columns]
        st.dataframe(df_students[show_cols], use_container_width=True)
        st.caption(f"{len(students)} students enrolled")

        # Per-student actions
        st.markdown('<div class="section-head">Student Actions</div>', unsafe_allow_html=True)
        action_email = st.selectbox("Select student",
                                    [s.get("Email","") for s in students],
                                    key="action_student")
        col_a, col_b, col_c = st.columns(3)

        with col_a:
            if st.button("🔑 Force password reset", key="force_reset_btn"):
                from db import get_tab, TAB_REGISTRY
                ws = get_tab(TAB_REGISTRY)
                rows = ws.get_all_values()
                for i, row in enumerate(rows[1:], start=2):
                    if len(row) >= 1 and row[0].strip().lower() == action_email.strip().lower():
                        ws.update_cell(i, 7, "TRUE")
                        break
                log_audit("instructor", "FORCE_RESET", action_email)
                st.success("Student will be prompted to reset password on next login.")

        with col_b:
            new_temp_pw = st.text_input("Set temporary password", key="temp_pw_input")
            if st.button("Set password", key="set_pw_btn"):
                if new_temp_pw and len(new_temp_pw) >= 8:
                    ok = update_password(action_email, new_temp_pw)
                    if ok:
                        log_audit("instructor", "PASSWORD_RESET", action_email)
                        st.success(f"Password set to: {new_temp_pw}")
                    else:
                        st.error("Failed.")
                else:
                    st.error("Min 8 characters.")

        with col_c:
            if st.button("🗑 Remove student", key="del_student_btn"):
                confirm = st.checkbox("Confirm removal", key="del_confirm")
                if confirm:
                    ok = delete_student(action_email)
                    if ok:
                        log_audit("instructor", "STUDENT_DELETED", action_email)
                        st.success("Student removed.")
                        st.rerun()
    else:
        st.info("No students enrolled yet.")

    # Bulk import
    st.markdown('<div class="section-head">Bulk Enroll Students</div>', unsafe_allow_html=True)
    st.caption("Paste one email per line. Students will be registered with password 'TempPass123' "
               "and prompted to reset on first login.")
    bulk_emails_text = st.text_area("Email list", height=150, key="bulk_emails",
                                    placeholder="student1@uni.edu\nstudent2@uni.edu")
    if st.button("Bulk Enroll", key="bulk_enroll_btn"):
        if bulk_emails_text.strip():
            email_list = [e.strip() for e in bulk_emails_text.strip().splitlines() if e.strip()]
            with st.spinner(f"Enrolling {len(email_list)} students..."):
                from db import bulk_register_students
                added, skipped, errors = bulk_register_students(email_list)
            if added:
                log_audit("instructor", "BULK_ENROLL", f"{len(added)} added")
                st.success(f"Added: {len(added)} students.")
            if skipped:
                st.warning(f"Skipped (already exist): {', '.join(skipped)}")
            if errors:
                st.error(f"Errors: {'; '.join(errors)}")
            # Force reset on all newly added
            for em in added:
                try:
                    from db import get_tab, TAB_REGISTRY
                    ws = get_tab(TAB_REGISTRY)
                    rows = ws.get_all_values()
                    for i, row in enumerate(rows[1:], start=2):
                        if len(row) >= 1 and row[0].strip().lower() == em.strip().lower():
                            ws.update_cell(i, 7, "TRUE")
                            break
                except Exception:
                    pass
        else:
            st.error("Please enter at least one email.")

    # CSV download of student list
    if students:
        df_export = pd.DataFrame(students)
        safe_cols = [c for c in ["Email","First_Name","Last_Name","Registered_At","Last_Login"]
                     if c in df_export.columns]
        csv = df_export[safe_cols].to_csv(index=False).encode("utf-8")
        st.download_button("⬇ Download student list CSV", csv,
                           "students.csv", "text/csv")

# ════════════════════════════════════════════════════════════════════════════════
#  TAB 4: FLAGS & INTEGRITY
# ════════════════════════════════════════════════════════════════════════════════
with tabs[3]:
    st.markdown('<div class="section-head">Plagiarism / Copying Flags</div>', unsafe_allow_html=True)
    st.caption("Students with identical submitted answers but different question parameters.")

    subs = get_all_submissions()
    if not subs:
        st.info("No submissions to analyse.")
    else:
        df_all = pd.DataFrame(subs)
        flags  = []

        if "Homework_ID" in df_all.columns and "Question_ID" in df_all.columns:
            for (hw_id, q_id), grp in df_all.groupby(["Homework_ID","Question_ID"]):
                submitted = grp[grp.get("Status","") == "submitted"] if "Status" in grp.columns else grp
                submitted = submitted.drop_duplicates(subset=["Email"], keep="last")
                if "Raw_Answer" not in submitted.columns or "Param_Seed" not in submitted.columns:
                    continue
                submitted["_ans"] = submitted["Raw_Answer"].astype(str)
                dup_mask = submitted.duplicated("_ans", keep=False)
                dups = submitted[dup_mask]
                if dups.empty: continue
                for key, group in dups.groupby("_ans"):
                    seeds = group["Param_Seed"].astype(str).unique()
                    if len(seeds) > 1:
                        for _, row in group.iterrows():
                            flags.append({
                                "Question": f"{hw_id}/{q_id}",
                                "Email": row.get("Email",""),
                                "Seed": row.get("Param_Seed",""),
                                "Answer": str(key)[:80],
                            })

        if flags:
            st.warning(f"⚠ {len(flags)} flagged entries detected.")
            st.dataframe(pd.DataFrame(flags), use_container_width=True)
        else:
            st.success("✅ No copying flags detected.")

    # Student-flagged questions
    st.markdown('<div class="section-head">Student-Flagged Questions</div>', unsafe_allow_html=True)
    st.caption("Questions students reported as potentially erroneous.")
    try:
        ws   = get_tab(TAB_CONFIG)
        rows = ws.get_all_records()
        audit_rows = [r for r in rows if r.get("Action","") == "FLAG_QUESTION"]
        if audit_rows:
            st.dataframe(pd.DataFrame(audit_rows), use_container_width=True)
        else:
            st.info("No questions have been flagged by students.")
    except Exception:
        st.info("Could not load flags.")

# ════════════════════════════════════════════════════════════════════════════════
#  TAB 5: SETTINGS
# ════════════════════════════════════════════════════════════════════════════════
with tabs[4]:
    st.markdown('<div class="section-head">Change Instructor Password</div>',
                unsafe_allow_html=True)
    with st.form("change_pw_form"):
        cur_pw  = st.text_input("Current password", type="password")
        new_pw1 = st.text_input("New password", type="password")
        new_pw2 = st.text_input("Confirm new password", type="password")
        ch_sub  = st.form_submit_button("Update Password")
    if ch_sub:
        if not verify_instructor(cur_pw):
            st.error("Current password is incorrect.")
        elif len(new_pw1) < 8:
            st.error("New password must be at least 8 characters.")
        elif new_pw1 != new_pw2:
            st.error("Passwords do not match.")
        else:
            ok = update_instructor_password(new_pw1)
            if ok:
                log_audit("instructor", "PASSWORD_CHANGED", "")
                st.success("Password updated.")
            else:
                st.error("Update failed.")

    st.markdown('<div class="section-head">Audit Log</div>', unsafe_allow_html=True)
    try:
        ws        = get_tab(TAB_CONFIG)
        all_rows  = ws.get_all_values()
        audit_start = next(
            (i for i,r in enumerate(all_rows) if len(r)>0 and r[0]=="Timestamp" and i>50),
            None
        )
        if audit_start is not None:
            audit_data = all_rows[audit_start+1:]
            if audit_data:
                df_audit = pd.DataFrame(
                    audit_data,
                    columns=["Timestamp","Actor","Action","Detail"]
                )
                st.dataframe(df_audit.iloc[::-1].reset_index(drop=True),
                             use_container_width=True)
            else:
                st.info("No audit entries yet.")
    except Exception as e:
        st.info(f"Could not load audit log: {e}")

    st.markdown('<div class="section-head">End-of-Semester Reset</div>', unsafe_allow_html=True)
    st.markdown(f"""
    <div class="banner-warning">
      ⚠ <strong>Warning:</strong> This section is for end-of-semester maintenance only.
      See the README for the full reset procedure.
    </div>
    """, unsafe_allow_html=True)

    with st.expander("Reset options (use with caution)"):
        st.markdown("**Step 1** — Download all data first (Overview tab → Download CSV)")
        st.markdown("**Step 2** — Then use the options below")
        if st.button("🗑 Clear all student accounts (Registry tab)", key="clear_students"):
            confirm_clear = st.checkbox("I have downloaded all data and confirm this action",
                                        key="confirm_clear_students")
            if confirm_clear:
                try:
                    ws = get_tab("Registry")
                    ws.clear()
                    ws.update("A1", [["Email","Password_Hash","First_Name","Last_Name",
                                      "Registered_At","Last_Login","Force_Reset"]])
                    log_audit("instructor", "REGISTRY_CLEARED", "End of semester reset")
                    st.success("Registry cleared.")
                except Exception as e:
                    st.error(f"Failed: {e}")

# ── Sign out ──────────────────────────────────────────────────────────────────
st.markdown("<br>", unsafe_allow_html=True)
if st.button("Sign out", key="instructor_signout"):
    log_audit("instructor", "LOGOUT", "")
    st.session_state.pop("instructor_auth", None)
    st.switch_page("Home.py")
