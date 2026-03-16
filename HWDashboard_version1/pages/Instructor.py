"""
pages/Instructor.py — Instructor dashboard.
HWDashboard v2 — Phase 1
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
    get_tab, TAB_REGISTRY, TAB_CONFIG,
    get_extension_requests, grant_individual_access,
    bulk_register_students, verify_password, hash_password
)
from ui import inject_css, page_header, COLORS

st.set_page_config(
    page_title="Instructor Dashboard",
    page_icon="🔐",
    layout="wide",
    initial_sidebar_state="expanded",
)

inject_css()

st.markdown(f"""
<style>
.block-container {{ max-width: 1200px; }}
.dash-metric {{
    background: #FFFFFF; border: 1px solid #E5E7EB;
    border-radius: 10px; padding: 1rem 1.3rem; text-align: center;
}}
.dash-metric-num {{
    font-family: 'DM Serif Display', serif;
    font-size: 1.9rem; color: {COLORS['navy']}; line-height: 1;
}}
.dash-metric-label {{
    font-size: 0.7rem; color: {COLORS['neutral_500']};
    margin-top: 0.3rem; font-weight: 600;
    text-transform: uppercase; letter-spacing: 0.07em;
}}
.section-head {{
    font-family: 'DM Serif Display', serif;
    font-size: 1.05rem; color: {COLORS['navy']};
    margin: 1.6rem 0 0.7rem 0;
    padding-bottom: 0.35rem;
    border-bottom: 1px solid {COLORS['neutral_200']};
}}
.enroll-key-box {{
    background: {COLORS['neutral_50']};
    border: 1px solid {COLORS['neutral_200']};
    border-radius: 8px;
    padding: 1rem 1.4rem;
    display: flex; align-items: center; gap: 1.2rem;
    margin-bottom: 1rem; flex-wrap: wrap;
}}
</style>
""", unsafe_allow_html=True)

# ── Auth gate ──────────────────────────────────────────────────────────────────
if not st.session_state.get("instructor_auth"):
    st.markdown(f"""
    <div style="max-width:400px;margin:3rem auto;text-align:center;">
      <div style="font-family:'DM Serif Display',serif;font-size:1.4rem;
           color:{COLORS['navy']};margin-bottom:0.4rem;">Instructor Access</div>
      <div style="font-size:0.82rem;color:{COLORS['neutral_500']};
           margin-bottom:1.5rem;">Learning Intermediate Microeconomics</div>
    </div>
    """, unsafe_allow_html=True)

    col = st.columns([1,2,1])[1]
    with col:
        pwd = st.text_input("Password", type="password", key="inst_pw_input")
        if st.button("Sign In", use_container_width=True, key="inst_signin"):
            if verify_instructor(pwd):
                st.session_state["instructor_auth"] = True
                log_audit("instructor", "LOGIN", "")
                st.rerun()
            else:
                st.error("Incorrect password.")
    st.stop()

# ── Dashboard ──────────────────────────────────────────────────────────────────
page_header(
    "Learning Intermediate Microeconomics",
    "Instructor Dashboard",
    datetime.datetime.now().strftime("%d %b %Y, %H:%M")
)

tabs = st.tabs([
    "📊 Overview",
    "📚 Homework Manager",
    "👥 Student Manager",
    "🔔 Extension Requests",
    "🚩 Integrity",
    "⚙️ Settings",
])

# ════════════════════════════════════════════════════════════════════════════════
#  TAB 1 — OVERVIEW
# ════════════════════════════════════════════════════════════════════════════════
with tabs[0]:
    if st.button("↻ Refresh", key="ref_overview"):
        st.rerun()

    with st.spinner("Loading data..."):
        students    = get_all_students()
        submissions = get_all_submissions()
        hw_configs  = get_homework_configs()

    df_sub = pd.DataFrame(submissions) if submissions else pd.DataFrame()

    n_students = len(students)
    n_complete = 0
    if not df_sub.empty and "Status" in df_sub.columns:
        n_complete = len(df_sub[df_sub["Status"] == "submitted"])
    n_hw_open = len([c for c in hw_configs
                     if c.get("Enabled","").upper() == "TRUE"])

    c1,c2,c3,c4 = st.columns(4)
    for col, num, lbl in [
        (c1, n_students,  "Enrolled\nStudents"),
        (c2, n_complete,  "Submissions\nReceived"),
        (c3, len(hw_configs), "Total\nAssignments"),
        (c4, n_hw_open,   "Currently\nOpen"),
    ]:
        col.markdown(f"""
        <div class="dash-metric">
          <div class="dash-metric-num">{num}</div>
          <div class="dash-metric-label">{lbl}</div>
        </div>
        """, unsafe_allow_html=True)

    # Enrollment key with copy button
    st.markdown('<div class="section-head">Enrollment Key</div>',
                unsafe_allow_html=True)
    enroll_key = get_enrollment_key()
    st.markdown(f"""
    <div class="enroll-key-box">
      <div>
        <div style="font-size:0.68rem;font-weight:600;text-transform:uppercase;
             letter-spacing:0.1em;color:#6B7280;margin-bottom:0.3rem;">
          Current Key (valid 6 months)
        </div>
        <div style="font-family:monospace;font-size:1.5rem;font-weight:700;
             color:{COLORS['navy']};letter-spacing:0.18em;">{enroll_key}</div>
      </div>
      <div style="font-size:0.82rem;color:#6B7280;max-width:320px;">
        Share with students to let them create accounts.
        Auto-rotates every 6 months.
      </div>
    </div>
    """, unsafe_allow_html=True)
    if st.button("📋 Copy enrollment key to clipboard", key="copy_key"):
        st.code(enroll_key)
        st.caption("Select the text above and copy it.")

    # Score distributions
    if not df_sub.empty and "Homework_ID" in df_sub.columns:
        st.markdown('<div class="section-head">Score Distributions</div>',
                    unsafe_allow_html=True)
        hw_ids = df_sub["Homework_ID"].unique()
        if len(hw_ids):
            cols = st.columns(min(len(hw_ids), 3))
            for idx, hw_id in enumerate(list(hw_ids)[:3]):
                hw_df = df_sub[
                    (df_sub["Homework_ID"] == hw_id) &
                    (df_sub["Status"] == "submitted")
                ].drop_duplicates(subset=["Email","Question_ID"], keep="last")
                if hw_df.empty: continue
                scores = pd.to_numeric(hw_df["Score"], errors="coerce").dropna()
                if scores.empty: continue
                with cols[idx]:
                    fig, ax = plt.subplots(figsize=(4.5, 3))
                    fig.patch.set_facecolor("#FAFAFA")
                    ax.set_facecolor("#FAFAFA")
                    ax.hist(scores, bins=10, color=COLORS["navy"],
                            edgecolor="white", rwidth=0.8)
                    ax.set_xlabel("Score", fontsize=9)
                    ax.set_ylabel("Count",  fontsize=9)
                    ax.set_title(
                        f"{hw_id}\nMean: {scores.mean():.1f}  Median: {scores.median():.1f}",
                        fontsize=9
                    )
                    ax.spines["top"].set_visible(False)
                    ax.spines["right"].set_visible(False)
                    ax.tick_params(labelsize=8)
                    plt.tight_layout()
                    st.pyplot(fig); plt.close(fig)
                    st.caption(f"n={len(scores)} · min={int(scores.min())} · max={int(scores.max())}")

    # Question-level analytics
    if not df_sub.empty:
        st.markdown('<div class="section-head">Question Analytics — Success Rates</div>',
                    unsafe_allow_html=True)
        if "Question_ID" in df_sub.columns and "Score" in df_sub.columns:
            sub_only = df_sub[df_sub["Status"] == "submitted"].copy()
            sub_only = sub_only.drop_duplicates(
                subset=["Email","Homework_ID","Question_ID"], keep="last")
            sub_only["Score"]     = pd.to_numeric(sub_only["Score"], errors="coerce")
            sub_only["Max_Score"] = pd.to_numeric(sub_only["Max_Score"], errors="coerce")
            if not sub_only.empty:
                grp = sub_only.groupby(["Homework_ID","Question_ID"]).agg(
                    Submissions=("Score","count"),
                    Avg_Score=("Score","mean"),
                    Max_Score=("Max_Score","first"),
                ).reset_index()
                grp["Success_Rate"] = (grp["Avg_Score"] / grp["Max_Score"] * 100).round(1)
                grp = grp.sort_values("Success_Rate")
                st.dataframe(grp, use_container_width=True)

    # Late submissions
    if not df_sub.empty and "Is_Late" in df_sub.columns:
        late = df_sub[
            (df_sub["Is_Late"] == "Yes") & (df_sub["Status"] == "submitted")
        ]
        if not late.empty:
            st.markdown('<div class="section-head">Late Submissions</div>',
                        unsafe_allow_html=True)
            show_cols = [c for c in ["Timestamp","Email","Homework_ID",
                                     "Question_ID","Score","Max_Score"]
                         if c in late.columns]
            st.dataframe(late[show_cols], use_container_width=True)

    # Full log + download
    st.markdown('<div class="section-head">Full Submission Log</div>',
                unsafe_allow_html=True)
    with st.expander("Show all rows"):
        if not df_sub.empty:
            st.dataframe(df_sub, use_container_width=True)
            csv = df_sub.to_csv(index=False).encode("utf-8")
            st.download_button("⬇ Download CSV", csv,
                               "submissions.csv", "text/csv",
                               key="dl_full_csv")

            # Gradebook export
            st.markdown("**Gradebook Export** (one row per student, total scores)")
            if "Email" in df_sub.columns:
                sub_only = df_sub[df_sub["Status"]=="submitted"].copy()
                sub_only = sub_only.drop_duplicates(
                    subset=["Email","Homework_ID","Question_ID"], keep="last")
                sub_only["Score"] = pd.to_numeric(sub_only["Score"], errors="coerce").fillna(0)
                gb = sub_only.groupby("Email")["Score"].sum().reset_index()
                gb.columns = ["Email", "Total_Score"]
                # Add names
                stu_df = pd.DataFrame(students)
                if not stu_df.empty and "Email" in stu_df.columns:
                    stu_safe = stu_df[["Email","First_Name","Last_Name"]].copy()
                    stu_safe["Email"] = stu_safe["Email"].str.lower()
                    gb = gb.merge(stu_safe, on="Email", how="left")
                gb_csv = gb.to_csv(index=False).encode("utf-8")
                st.download_button("⬇ Download Gradebook CSV", gb_csv,
                                   "gradebook.csv", "text/csv",
                                   key="dl_gradebook")
        else:
            st.info("No submissions yet.")

# ════════════════════════════════════════════════════════════════════════════════
#  TAB 2 — HOMEWORK MANAGER
# ════════════════════════════════════════════════════════════════════════════════
with tabs[1]:
    st.markdown('<div class="section-head">Manage Assignments</div>',
                unsafe_allow_html=True)
    st.caption("All changes take effect immediately — no redeployment needed.")

    hw_configs = get_homework_configs()

    if not hw_configs:
        st.info("No assignments configured yet.")
    else:
        for cfg in sorted(hw_configs, key=lambda x: x.get("HW_ID","")):
            hw_id   = cfg.get("HW_ID","")
            title   = cfg.get("Title","")
            enabled = cfg.get("Enabled","FALSE").upper() == "TRUE"
            dl      = cfg.get("Deadline","")
            grace   = cfg.get("Grace_Minutes","15")
            ann     = cfg.get("Announcement","")
            instr   = cfg.get("Instructions","")
            auto_en = cfg.get("Auto_Enable_Date","")

            with st.expander(f"**{title}** ({hw_id})", expanded=False):
                c_en, c_dl, c_gr = st.columns([1,2,1])

                with c_en:
                    new_en = st.toggle("Enabled", value=enabled,
                                       key=f"en_{hw_id}")
                    if new_en != enabled:
                        update_homework_config(hw_id,"Enabled",str(new_en).upper())
                        log_audit("instructor","HW_TOGGLE",f"{hw_id}->{new_en}")
                        st.success("Saved.")

                with c_dl:
                    try:
                        dl_dt    = datetime.datetime.strptime(dl.strip(),"%Y-%m-%d %H:%M")
                        dl_date  = dl_dt.date()
                        dl_time  = dl_dt.time()
                    except Exception:
                        dl_date = datetime.date.today()
                        dl_time = datetime.time(23,59)
                    new_date = st.date_input("Deadline date", value=dl_date,
                                             key=f"dl_d_{hw_id}")
                    new_time = st.time_input("Deadline time", value=dl_time,
                                             key=f"dl_t_{hw_id}")
                    new_dl   = f"{new_date.strftime('%Y-%m-%d')} {new_time.strftime('%H:%M')}"
                    if new_dl != dl:
                        if st.button("Update deadline", key=f"save_dl_{hw_id}"):
                            update_homework_config(hw_id,"Deadline",new_dl)
                            log_audit("instructor","HW_DEADLINE",f"{hw_id}->{new_dl}")
                            st.success(f"Deadline updated to {new_dl}")
                            st.rerun()

                with c_gr:
                    new_gr = st.number_input("Grace (min)", value=int(grace),
                                             min_value=0, max_value=120,
                                             key=f"gr_{hw_id}")
                    if str(new_gr) != str(grace):
                        if st.button("Save grace", key=f"save_gr_{hw_id}"):
                            update_homework_config(hw_id,"Grace_Minutes",str(new_gr))
                            log_audit("instructor","HW_GRACE",f"{hw_id}->{new_gr}")
                            st.success("Saved.")

                new_ann = st.text_input("Student announcement",
                                        value=ann, key=f"ann_{hw_id}")
                if new_ann != ann:
                    if st.button("Save announcement", key=f"save_ann_{hw_id}"):
                        update_homework_config(hw_id,"Announcement",new_ann)
                        log_audit("instructor","HW_ANN",f"{hw_id}")
                        st.success("Saved.")

                new_instr = st.text_area("Assignment instructions",
                                          value=instr, key=f"instr_{hw_id}",
                                          height=80)
                if new_instr != instr:
                    if st.button("Save instructions", key=f"save_instr_{hw_id}"):
                        update_homework_config(hw_id,"Instructions",new_instr)
                        log_audit("instructor","HW_INSTR",f"{hw_id}")
                        st.success("Saved.")

                st.markdown("**Auto-enable date** (leave blank to enable manually):")
                try:
                    ae_dt_def = datetime.datetime.strptime(
                        auto_en.strip(),"%Y-%m-%d %H:%M"
                    ) if auto_en.strip() else None
                except Exception:
                    ae_dt_def = None

                ae_date = st.date_input("Auto-enable date",
                                        value=ae_dt_def.date() if ae_dt_def else None,
                                        key=f"ae_d_{hw_id}")
                ae_time = st.time_input("Auto-enable time",
                                        value=ae_dt_def.time() if ae_dt_def else datetime.time(9,0),
                                        key=f"ae_t_{hw_id}")
                if ae_date:
                    new_ae = f"{ae_date.strftime('%Y-%m-%d')} {ae_time.strftime('%H:%M')}"
                    if new_ae != auto_en:
                        if st.button("Save auto-enable", key=f"save_ae_{hw_id}"):
                            update_homework_config(hw_id,"Auto_Enable_Date",new_ae)
                            log_audit("instructor","HW_AUTO_EN",f"{hw_id}->{new_ae}")
                            st.success(f"Will auto-enable at {new_ae}")

                # Individual access
                st.markdown("**Grant individual access** (for extensions):")
                ia_c1, ia_c2 = st.columns(2)
                with ia_c1:
                    ia_email = st.text_input("Student email",
                                             key=f"ia_email_{hw_id}")
                with ia_c2:
                    ia_dl = st.text_input("Custom deadline (YYYY-MM-DD HH:MM)",
                                          key=f"ia_dl_{hw_id}")
                if st.button("Grant access", key=f"ia_grant_{hw_id}"):
                    if ia_email and ia_dl:
                        ok = grant_individual_access(ia_email.strip().lower(),
                                                     hw_id, ia_dl.strip())
                        if ok:
                            log_audit("instructor","INDIVIDUAL_ACCESS",
                                      f"{ia_email} -> {hw_id} until {ia_dl}")
                            st.success(f"Access granted to {ia_email} until {ia_dl}")
                        else:
                            st.error("Failed to grant access.")
                    else:
                        st.error("Email and deadline required.")

    # Add new homework
    st.markdown('<div class="section-head">Add New Assignment</div>',
                unsafe_allow_html=True)
    with st.expander("Add assignment"):
        n_id    = st.text_input("Homework ID (e.g. HW_WEEK3)", key="n_hw_id")
        n_title = st.text_input("Title", key="n_hw_title")
        n_en    = st.toggle("Enable immediately", key="n_hw_en")
        n_dl    = st.date_input("Deadline date", key="n_hw_dl")
        n_tm    = st.time_input("Deadline time", key="n_hw_tm")
        n_gr    = st.number_input("Grace minutes", value=15, key="n_hw_gr")
        n_mx    = st.number_input("Max marks", value=18, key="n_hw_mx")
        n_instr = st.text_area("Instructions", key="n_hw_instr", height=60)
        if st.button("Add Assignment", key="add_hw"):
            if n_id.strip() and n_title.strip():
                dl_str = f"{n_dl.strftime('%Y-%m-%d')} {n_tm.strftime('%H:%M')}"
                ok = add_homework_config(
                    n_id.strip(), n_title.strip(),
                    str(n_en).upper(), dl_str, n_gr, "", 1, n_mx, n_instr.strip()
                )
                if ok:
                    log_audit("instructor","HW_ADDED",f"{n_id}: {n_title}")
                    st.success(f"Added {n_id}.")
                    st.rerun()
                else:
                    st.error("Failed. Check Sheet connection.")
            else:
                st.error("ID and title are required.")

# ════════════════════════════════════════════════════════════════════════════════
#  TAB 3 — STUDENT MANAGER
# ════════════════════════════════════════════════════════════════════════════════
with tabs[2]:
    st.markdown('<div class="section-head">Enrolled Students</div>',
                unsafe_allow_html=True)
    if st.button("↻ Refresh students", key="ref_students"):
        st.rerun()

    students = get_all_students()
    if students:
        df_stu = pd.DataFrame(students)
        show   = [c for c in ["Email","First_Name","Last_Name",
                               "Registered_At","Last_Login"]
                  if c in df_stu.columns]
        st.dataframe(df_stu[show], use_container_width=True)
        st.caption(f"{len(students)} students enrolled")

        # Actions
        st.markdown('<div class="section-head">Student Actions</div>',
                    unsafe_allow_html=True)
        sel_email = st.selectbox(
            "Select student",
            [s.get("Email","") for s in students],
            key="sel_stu"
        )
        cA, cB, cC = st.columns(3)

        with cA:
            if st.button("Force password reset", key="force_rst"):
                try:
                    ws   = get_tab(TAB_REGISTRY)
                    rows = ws.get_all_values()
                    for i, row in enumerate(rows[1:], start=2):
                        if (len(row) >= 1 and
                                row[0].strip().lower() == sel_email.strip().lower()):
                            ws.update_cell(i, 7, "TRUE")
                            break
                    log_audit("instructor","FORCE_RESET", sel_email)
                    st.success("Student will be prompted to reset on next login.")
                except Exception as e:
                    st.error(str(e))

        with cB:
            tmp_pw = st.text_input("Set temporary password", key="tmp_pw")
            if st.button("Set password", key="set_pw"):
                if tmp_pw and len(tmp_pw) >= 8:
                    if update_password(sel_email, tmp_pw):
                        log_audit("instructor","PW_SET", sel_email)
                        st.success(f"Password set. Tell student: {tmp_pw}")
                    else:
                        st.error("Failed.")
                else:
                    st.error("Min 8 characters.")

        with cC:
            if st.button("Remove student", key="del_stu"):
                confirm = st.checkbox("Confirm removal", key="del_conf")
                if confirm:
                    if delete_student(sel_email):
                        log_audit("instructor","STUDENT_DEL", sel_email)
                        st.success("Removed.")
                        st.rerun()

        # Instructor preview mode
        st.markdown('<div class="section-head">Preview as Student</div>',
                    unsafe_allow_html=True)
        st.caption("View the app exactly as this student sees it — their parameters, their submissions.")
        prev_email = st.selectbox(
            "Preview as",
            [s.get("Email","") for s in students],
            key="prev_email"
        )
        if st.button("Enter preview mode", key="enter_preview"):
            from db import get_student_submissions, get_homework_configs
            stu = next((s for s in students
                        if s.get("Email","").strip().lower() == prev_email.strip().lower()),
                       {})
            st.session_state["authenticated"]  = True
            st.session_state["student_email"]  = prev_email.strip().lower()
            st.session_state["student_name"]   = (
                f"{stu.get('First_Name','')} {stu.get('Last_Name','')}".strip()
                or prev_email
            )
            st.session_state["student_record"] = stu
            st.session_state["submissions"]    = get_student_submissions(
                prev_email.strip().lower())
            st.session_state["hw_configs"]     = get_homework_configs()
            st.session_state["preview_mode"]   = True
            log_audit("instructor","PREVIEW_MODE", prev_email)
            st.switch_page("pages/Dashboard.py")

        # CSV download
        if students:
            df_exp = pd.DataFrame(students)
            safe   = [c for c in ["Email","First_Name","Last_Name",
                                   "Registered_At","Last_Login"]
                      if c in df_exp.columns]
            csv = df_exp[safe].to_csv(index=False).encode("utf-8")
            st.download_button("⬇ Download student list",
                               csv, "students.csv", "text/csv",
                               key="dl_stu")
    else:
        st.info("No students enrolled yet.")

    # Bulk enroll
    st.markdown('<div class="section-head">Bulk Enroll</div>', unsafe_allow_html=True)
    st.caption("One email per line. Registered with TempPass123, forced to reset on first login.")
    bulk_text = st.text_area("Email list", height=120, key="bulk_emails",
                             placeholder="student1@uni.edu\nstudent2@uni.edu")
    if st.button("Bulk Enroll", key="bulk_enroll"):
        if bulk_text.strip():
            email_list = [e.strip() for e in bulk_text.strip().splitlines()
                          if e.strip()]
            with st.spinner(f"Enrolling {len(email_list)} students..."):
                added, skipped, errors = bulk_register_students(email_list)
            if added:
                log_audit("instructor","BULK_ENROLL",f"{len(added)} added")
                st.success(f"Added: {len(added)}")
            if skipped:
                st.warning(f"Skipped (exist): {len(skipped)}")
            if errors:
                st.error(f"Errors: {'; '.join(errors)}")
        else:
            st.error("Please enter at least one email.")

# ════════════════════════════════════════════════════════════════════════════════
#  TAB 4 — EXTENSION REQUESTS
# ════════════════════════════════════════════════════════════════════════════════
with tabs[3]:
    st.markdown('<div class="section-head">Pending Extension Requests</div>',
                unsafe_allow_html=True)
    if st.button("↻ Refresh", key="ref_ext"):
        st.rerun()

    pending = get_extension_requests(status="Pending")
    if not pending:
        st.success("✅ No pending extension requests.")
    else:
        st.warning(f"{len(pending)} pending request(s).")
        for req in pending:
            with st.expander(
                f"{req.get('Email','')} — {req.get('HW_ID','')} "
                f"({req.get('Requested_At','')})"):
                st.markdown(f"**Reason:** {req.get('Reason','')}")
                c1, c2 = st.columns(2)
                with c1:
                    new_dl = st.text_input(
                        "New deadline (YYYY-MM-DD HH:MM)",
                        key=f"ext_dl_{req.get('Email','')}_{req.get('HW_ID','')}"
                    )
                    if st.button("Approve & Grant Access",
                                 key=f"ext_approve_{req.get('Email','')}"):
                        if new_dl:
                            ok = grant_individual_access(
                                req["Email"], req["HW_ID"], new_dl)
                            if ok:
                                log_audit("instructor","EXT_APPROVED",
                                          f"{req['Email']} {req['HW_ID']} until {new_dl}")
                                st.success("Access granted.")
                                st.rerun()
                        else:
                            st.error("Enter a new deadline.")
                with c2:
                    if st.button("Deny",
                                 key=f"ext_deny_{req.get('Email','')}"):
                        log_audit("instructor","EXT_DENIED",
                                  f"{req['Email']} {req['HW_ID']}")
                        st.info("Denied. Note: student is not automatically notified — "
                                "please email them at " + req.get("Email",""))

    st.markdown('<div class="section-head">All Extension Requests</div>',
                unsafe_allow_html=True)
    all_reqs = get_extension_requests()
    if all_reqs:
        st.dataframe(pd.DataFrame(all_reqs), use_container_width=True)
    else:
        st.info("No extension requests yet.")

# ════════════════════════════════════════════════════════════════════════════════
#  TAB 5 — INTEGRITY
# ════════════════════════════════════════════════════════════════════════════════
with tabs[4]:
    st.markdown('<div class="section-head">Copying / Plagiarism Flags</div>',
                unsafe_allow_html=True)
    st.caption("Identical answers across students with different question parameters.")

    subs = get_all_submissions()
    if not subs:
        st.info("No submissions to analyse.")
    else:
        df_all = pd.DataFrame(subs)
        flags  = []
        if "Homework_ID" in df_all.columns:
            for (hw_id, q_id), grp in df_all.groupby(["Homework_ID","Question_ID"]):
                if "Status" not in grp.columns: continue
                sub = grp[grp["Status"] == "submitted"].drop_duplicates(
                    subset=["Email"], keep="last")
                if "Raw_Answer" not in sub.columns: continue
                sub = sub.copy()
                sub["_ans"] = sub["Raw_Answer"].astype(str)
                dup_mask = sub.duplicated("_ans", keep=False)
                for key, group in sub[dup_mask].groupby("_ans"):
                    if "Param_Seed" not in group.columns: continue
                    seeds = group["Param_Seed"].astype(str).unique()
                    if len(seeds) > 1:
                        for _, row in group.iterrows():
                            flags.append({
                                "Question": f"{hw_id}/{q_id}",
                                "Email":    row.get("Email",""),
                                "Seed":     row.get("Param_Seed",""),
                                "Answer":   str(key)[:80],
                            })

        if flags:
            st.warning(f"⚠ {len(flags)} flagged entries.")
            st.dataframe(pd.DataFrame(flags), use_container_width=True)
            csv = pd.DataFrame(flags).to_csv(index=False).encode("utf-8")
            st.download_button("⬇ Download plagiarism report",
                               csv, "plagiarism_flags.csv", "text/csv",
                               key="dl_plag")
        else:
            st.success("✅ No copying flags detected.")

    st.markdown('<div class="section-head">Student-Flagged Questions</div>',
                unsafe_allow_html=True)
    try:
        ws        = get_tab(TAB_CONFIG)
        all_rows  = ws.get_all_records()
        flagged   = [r for r in all_rows
                     if str(r.get("Action","")) == "FLAG_QUESTION"]
        if flagged:
            st.dataframe(pd.DataFrame(flagged), use_container_width=True)
        else:
            st.info("No questions flagged by students.")
    except Exception:
        st.info("Could not load flags.")

# ════════════════════════════════════════════════════════════════════════════════
#  TAB 6 — SETTINGS
# ════════════════════════════════════════════════════════════════════════════════
with tabs[5]:
    st.markdown('<div class="section-head">Change Instructor Password</div>',
                unsafe_allow_html=True)
    with st.form("ch_pw_form"):
        cur  = st.text_input("Current password", type="password")
        new1 = st.text_input("New password",     type="password")
        new2 = st.text_input("Confirm",          type="password")
        csub = st.form_submit_button("Update Password")
    if csub:
        if not verify_instructor(cur):
            st.error("Current password incorrect.")
        elif len(new1) < 8:
            st.error("Min 8 characters.")
        elif new1 != new2:
            st.error("Passwords do not match.")
        else:
            if update_instructor_password(new1):
                log_audit("instructor","PW_CHANGED","")
                st.success("Password updated.")
            else:
                st.error("Update failed.")

    st.markdown('<div class="section-head">Audit Log</div>',
                unsafe_allow_html=True)
    try:
        ws       = get_tab(TAB_CONFIG)
        all_vals = ws.get_all_values()
        a_start  = next((i for i,r in enumerate(all_vals)
                         if len(r) > 0 and r[0] == "Timestamp" and i > 250),
                        None)
        if a_start is not None:
            data = all_vals[a_start+1:]
            if data:
                df_audit = pd.DataFrame(
                    data, columns=["Timestamp","Actor","Action","Detail"])
                st.dataframe(df_audit.iloc[::-1].reset_index(drop=True),
                             use_container_width=True)
            else:
                st.info("No audit entries yet.")
    except Exception as e:
        st.info(f"Could not load audit log: {e}")

    st.markdown('<div class="section-head">End-of-Semester Reset</div>',
                unsafe_allow_html=True)
    st.markdown('<div class="banner-warning">⚠ Read the README before using these options.</div>',
                unsafe_allow_html=True)
    with st.expander("Reset options (use with caution)"):
        st.markdown("**Step 1** — Download all data first (Overview tab).")
        st.markdown("**Step 2** — Clear student registry:")
        if st.button("Clear all student accounts", key="clear_registry"):
            confirm = st.checkbox("I have downloaded all data", key="conf_clear")
            if confirm:
                try:
                    ws = get_tab("Registry")
                    ws.clear()
                    ws.update("A1", [["Email","Password_Hash","First_Name",
                                      "Last_Name","Registered_At",
                                      "Last_Login","Force_Reset","Email_Verified"]])
                    log_audit("instructor","REGISTRY_CLEARED","End of semester")
                    st.success("Registry cleared.")
                except Exception as e:
                    st.error(str(e))

# ── Sign out ───────────────────────────────────────────────────────────────────
st.markdown("<br>", unsafe_allow_html=True)
if st.button("Sign out", key="inst_out"):
    log_audit("instructor","LOGOUT","")
    st.session_state.pop("instructor_auth", None)
    st.switch_page("Home.py")
