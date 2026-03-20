"""
pages/Instructor.py — Instructor dashboard.
HWDashboard v8
"""
import streamlit as st
import datetime
import pandas as pd
import re
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from db import (
    verify_instructor, update_instructor_password,
    get_all_students, delete_student, update_password,
    get_homework_configs, update_hw_config, add_hw_config,
    get_all_submissions, get_enrollment_key, log_audit,
    get_tab, TAB_REGISTRY, bulk_register, verify_pw, hash_pw,
    get_student_submissions, check_auto_enable,
)
from ui import inject_css, page_header, COLORS

st.set_page_config(page_title="Instructor Dashboard", page_icon="🔐",
                   layout="wide", initial_sidebar_state="expanded")
inject_css()

st.markdown(f"""
<style>
.block-container {{ max-width: 1150px; }}
.dash-metric {{
    background:#FFFFFF; border:1.5px solid {COLORS['grey_mid']};
    border-radius:9px; padding:1.1rem 1.3rem; text-align:center;
}}
.dash-metric-num {{
    font-family:'DM Serif Display',serif; font-size:2rem;
    color:{COLORS['navy']}; line-height:1;
}}
.dash-metric-lbl {{
    font-size:0.8rem; color:{COLORS['grey_text']}; margin-top:0.3rem;
    font-weight:600; text-transform:uppercase; letter-spacing:0.07em;
}}
.sec {{
    font-family:'DM Serif Display',serif; font-size:1.1rem;
    color:{COLORS['navy']}; margin:1.6rem 0 0.7rem 0;
    padding-bottom:0.35rem; border-bottom:2px solid {COLORS['grey_mid']};
}}
</style>
""", unsafe_allow_html=True)


def _week_num(hw_id: str) -> int:
    m = re.search(r"(\d+)", hw_id)
    return int(m.group(1)) if m else 999


# ── Auth gate ──────────────────────────────────────────────────────────────────
if not st.session_state.get("instructor_auth"):
    st.markdown(
        f'<div style="max-width:380px;margin:3rem auto;text-align:center;">'
        f'<div style="font-family:\'DM Serif Display\',serif;font-size:1.35rem;'
        f'color:{COLORS["navy"]};margin-bottom:0.35rem;">Instructor Access</div>'
        f'<div style="font-size:0.81rem;color:{COLORS["grey_text"]};'
        f'margin-bottom:1.4rem;">Intermediate Microeconomics</div>'
        f'</div>',
        unsafe_allow_html=True
    )
    col = st.columns([1, 2, 1])[1]
    with col:
        pwd = st.text_input("Password", type="password", key="inst_pw")
        if st.button("Sign In", use_container_width=True, key="inst_in"):
            if verify_instructor(pwd):
                st.session_state["instructor_auth"] = True
                log_audit("instructor", "LOGIN", "")
                st.rerun()
            else:
                st.error("Incorrect password.")
    st.markdown(
        '<div style="max-width:380px;margin:1rem auto;">'
        '<div class="banner banner-warning" style="font-size:0.85rem;">'
        'Please change your instructor password in the Settings tab '
        'if you have not already done so.</div></div>',
        unsafe_allow_html=True
    )
    st.stop()


page_header(
    "Intermediate Microeconomics",
    "Instructor Dashboard",
    datetime.datetime.now().strftime("%d %b %Y, %H:%M")
)

# Check and apply any auto-enable dates that have passed
try:
    check_auto_enable()
except Exception:
    pass

tabs = st.tabs(["📊 Overview", "📚 Homework", "👥 Students", "⚙️ Settings"])


# ════════════════════════════════════════════════════════════════════════════════
with tabs[0]:
    if st.button("↻ Refresh", key="ref_ov"):
        st.rerun()

    with st.spinner("Loading..."):
        students    = get_all_students()
        submissions = get_all_submissions()
        hw_configs  = get_homework_configs()

    df = pd.DataFrame(submissions) if submissions else pd.DataFrame()
    n_sub = (
        len(df[df["Status"] == "submitted"])
        if not df.empty and "Status" in df.columns else 0
    )

    c1, c2, c3, c4 = st.columns(4)
    for col, num, lbl in [
        (c1, len(students),   "Enrolled Students"),
        (c2, n_sub,           "Submissions Received"),
        (c3, len(hw_configs), "Total Assignments"),
        (c4, len([h for h in hw_configs
                  if h.get("Enabled", "").upper() == "TRUE"]),
             "Currently Open"),
    ]:
        col.markdown(
            f'<div class="dash-metric">'
            f'<div class="dash-metric-num">{num}</div>'
            f'<div class="dash-metric-lbl">{lbl}</div>'
            f'</div>',
            unsafe_allow_html=True
        )

    st.markdown('<div class="sec">Enrollment Key</div>', unsafe_allow_html=True)
    key = get_enrollment_key()
    st.markdown(
        f'<div style="background:#F5F5F5;border:1.5px solid #E0E0E0;'
        f'border-radius:8px;padding:0.9rem 1.2rem;margin-bottom:0.8rem;">'
        f'<div style="font-size:0.67rem;font-weight:600;text-transform:uppercase;'
        f'letter-spacing:0.1em;color:#555555;margin-bottom:0.25rem;">Current Key</div>'
        f'<div style="font-family:monospace;font-size:1.45rem;font-weight:700;'
        f'color:{COLORS["navy"]};letter-spacing:0.18em;">{key}</div>'
        f'<div style="font-size:0.8rem;color:#6B7280;margin-top:0.25rem;">'
        f'Share with students. Auto-rotates every 6 months.</div></div>',
        unsafe_allow_html=True
    )
    st.code(key, language=None)
    st.caption("Select and copy the key above.")

    if not df.empty and "Question_ID" in df.columns:
        st.markdown('<div class="sec">Question Analytics</div>',
                    unsafe_allow_html=True)
        sub_df = df[df["Status"] == "submitted"].copy()
        sub_df = sub_df.drop_duplicates(
            subset=["Email", "Homework_ID", "Question_ID"], keep="last")
        sub_df["Score"]     = pd.to_numeric(sub_df["Score"],     errors="coerce")
        sub_df["Max_Score"] = pd.to_numeric(sub_df["Max_Score"], errors="coerce")
        if not sub_df.empty:
            grp = sub_df.groupby(["Homework_ID", "Question_ID"]).agg(
                Submissions=("Score", "count"),
                Avg_Score=("Score", "mean"),
                Max_Score=("Max_Score", "first"),
            ).reset_index()
            grp["Success_Rate_%"] = (
                grp["Avg_Score"] / grp["Max_Score"] * 100
            ).round(1)
            st.dataframe(grp.sort_values("Success_Rate_%"),
                         use_container_width=True)

    st.markdown('<div class="sec">Submission Log</div>', unsafe_allow_html=True)
    with st.expander("Show all rows"):
        if not df.empty:
            st.dataframe(df, use_container_width=True)
            st.download_button("⬇ Download all submissions CSV",
                               df.to_csv(index=False).encode("utf-8"),
                               "submissions.csv", "text/csv", key="dl_all")
            st.markdown("**Gradebook:**")
            sub_only = df[df["Status"] == "submitted"].copy()
            sub_only = sub_only.drop_duplicates(
                subset=["Email", "Homework_ID", "Question_ID"], keep="last")
            sub_only["Score"] = pd.to_numeric(
                sub_only["Score"], errors="coerce").fillna(0)
            gb = sub_only.groupby("Email")["Score"].sum().reset_index()
            gb.columns = ["Email", "Total_Score"]
            stu_df = pd.DataFrame(students)
            if not stu_df.empty and "Email" in stu_df.columns:
                safe = stu_df[["Email", "First_Name", "Last_Name"]].copy()
                safe["Email"] = safe["Email"].str.lower()
                gb = gb.merge(safe, on="Email", how="left")
            st.download_button("⬇ Download Gradebook CSV",
                               gb.to_csv(index=False).encode("utf-8"),
                               "gradebook.csv", "text/csv", key="dl_gb")
        else:
            st.info("No submissions yet.")


# ════════════════════════════════════════════════════════════════════════════════
with tabs[1]:
    from question_engine import ALL_HW_CONFIGS

    st.markdown('<div class="sec">Manage Assignments</div>', unsafe_allow_html=True)
    st.caption("Changes take effect immediately — no redeployment needed.")

    hw_configs     = get_homework_configs()
    registered_ids = {c.get("HW_ID", "") for c in hw_configs}
    engine_ids     = set(ALL_HW_CONFIGS.keys())
    orphaned       = registered_ids - engine_ids

    hw_configs_sorted = sorted(
        hw_configs, key=lambda x: _week_num(x.get("HW_ID", "")))

    if orphaned:
        for oid in sorted(orphaned, key=_week_num):
            st.markdown(
                f'<div class="banner banner-error">⚠ <strong>{oid}</strong> is in '
                f'the Sheet but has no questions in question_engine.py. '
                f'Students see it but cannot answer questions. '
                f'Delete this row from the Config tab.</div>',
                unsafe_allow_html=True)

    if not hw_configs_sorted:
        st.info("No assignments configured yet. Add one below.")
    else:
        try:
            all_subs = get_all_submissions()
            df_subs  = pd.DataFrame(all_subs) if all_subs else pd.DataFrame()
        except Exception:
            df_subs = pd.DataFrame()

        for cfg in hw_configs_sorted:
            hw_id   = cfg.get("HW_ID", "")
            title   = cfg.get("Title", hw_id)
            enabled = cfg.get("Enabled", "FALSE").upper() == "TRUE"
            dl      = cfg.get("Deadline", "")
            grace   = cfg.get("Grace_Minutes", "15")
            ann     = cfg.get("Announcement", "")
            instr   = cfg.get("Instructions", "")
            is_orp  = hw_id in orphaned

            n_hw_subs = 0
            if not df_subs.empty and "Homework_ID" in df_subs.columns and "Status" in df_subs.columns:
                n_hw_subs = len(df_subs[
                    (df_subs["Homework_ID"] == hw_id) &
                    (df_subs["Status"] == "submitted")
                ])

            try:
                dl_dt      = datetime.datetime.strptime(dl.strip(), "%Y-%m-%d %H:%M")
                dl_display = dl_dt.strftime("%d %b %Y at %H:%M")
            except Exception:
                dl_dt      = None
                dl_display = dl

            orp_marker = " 🔴" if is_orp else ""
            exp_label  = (
                f"**{title}** ({hw_id}){orp_marker}"
                f" — {n_hw_subs} submissions · Due {dl_display}"
            )

            with st.expander(exp_label, expanded=False):
                if is_orp:
                    st.markdown(
                        '<div class="banner banner-error">🔴 No matching questions '
                        'in question_engine.py.</div>',
                        unsafe_allow_html=True)

                c_en, c_dl, c_gr = st.columns([1, 2, 1])

                with c_en:
                    new_en = st.toggle("Enabled", value=enabled,
                                       key=f"en_{hw_id}")
                    if new_en != enabled:
                        if not new_en and n_hw_subs > 0:
                            if st.checkbox(
                                f"⚠ {n_hw_subs} submissions exist. Confirm disable?",
                                key=f"conf_dis_{hw_id}"):
                                update_hw_config(hw_id, "Enabled", "FALSE")
                                log_audit("instructor", "HW_DISABLE", hw_id)
                                st.success("Disabled.")
                                st.rerun()
                        else:
                            update_hw_config(hw_id, "Enabled",
                                             str(new_en).upper())
                            log_audit("instructor", "HW_TOGGLE",
                                      f"{hw_id}->{new_en}")
                            st.success("Saved.")

                with c_dl:
                    try:
                        dd = dl_dt.date()
                        dt = dl_dt.time()
                    except Exception:
                        dd = datetime.date.today()
                        dt = datetime.time(23, 59)
                    new_d = st.date_input("Deadline date", value=dd,
                                          key=f"dd_{hw_id}")
                    new_t = st.time_input("Deadline time", value=dt,
                                          key=f"dt_{hw_id}")
                    new_dl_str = (f"{new_d.strftime('%Y-%m-%d')} "
                                  f"{new_t.strftime('%H:%M')}")
                    if new_dl_str != dl:
                        if st.button("Update deadline", key=f"sdl_{hw_id}"):
                            new_dl_dt = datetime.datetime.strptime(
                                new_dl_str, "%Y-%m-%d %H:%M")
                            if new_dl_dt < datetime.datetime.now():
                                st.warning(
                                    f"⚠ New deadline {new_dl_str} is in the "
                                    f"past — homework closes immediately.")
                            update_hw_config(hw_id, "Deadline", new_dl_str)
                            log_audit("instructor", "HW_DL",
                                      f"{hw_id}->{new_dl_str}")
                            st.success(f"Deadline updated to {new_dl_str}")
                            st.rerun()

                with c_gr:
                    new_gr = st.number_input(
                        "Grace (min)", value=int(grace or 15),
                        min_value=0, max_value=120, key=f"gr_{hw_id}")
                    if str(new_gr) != str(grace):
                        if st.button("Save", key=f"sgr_{hw_id}"):
                            update_hw_config(hw_id, "Grace_Minutes",
                                             str(new_gr))
                            st.success("Saved.")

                new_ann = st.text_input(
                    "Announcement", value=ann, key=f"ann_{hw_id}")
                if new_ann != ann:
                    if st.button("Save announcement", key=f"sann_{hw_id}"):
                        update_hw_config(hw_id, "Announcement", new_ann)
                        log_audit("instructor", "HW_ANN", hw_id)
                        st.success("Saved.")

                new_instr = st.text_area(
                    "Instructions", value=instr,
                    key=f"instr_{hw_id}", height=70)
                st.caption(f"{len(new_instr)} / 500 characters")
                if new_instr != instr:
                    if st.button("Save instructions", key=f"sinstr_{hw_id}"):
                        update_hw_config(hw_id, "Instructions", new_instr)
                        log_audit("instructor", "HW_INSTR", hw_id)
                        st.success("Saved.")

    # ── Add New Assignment ─────────────────────────────────────────────────────
    st.markdown('<div class="sec">Add New Assignment</div>',
                unsafe_allow_html=True)

    available = sorted(
        [hid for hid in engine_ids if hid not in registered_ids],
        key=_week_num
    )

    if not available:
        st.info("All homeworks in question_engine.py are already registered.")
    else:
        sel = st.selectbox(
            "Select a homework to add:",
            options=["— select one —"] + available,
            key="sel_hw_add"
        )

        if sel and sel != "— select one —":
            hw_data    = ALL_HW_CONFIGS.get(sel, {})
            auto_marks = sum(q.get("marks", 0)
                             for q in hw_data.get("questions", []))
            wm         = re.search(r"WEEK(\d+)", sel)
            auto_title = f"Week {wm.group(1)}" if wm else sel
            n_qs       = len(hw_data.get("questions", []))

            st.markdown(
                f'<div class="banner banner-info">'
                f'<strong>{sel}</strong> &nbsp;·&nbsp; '
                f'{n_qs} questions &nbsp;·&nbsp; '
                f'<strong>{auto_marks} marks</strong> total &nbsp;·&nbsp; '
                f'Title: <strong>{auto_title}</strong>'
                f'</div>',
                unsafe_allow_html=True
            )

            st.markdown("**Set deadline:**")
            col_d, col_t = st.columns(2)
            with col_d:
                n_dl = st.date_input(
                    "Date",
                    value=datetime.date.today() + datetime.timedelta(days=14),
                    key="n_dl"
                )
            with col_t:
                n_tm = st.time_input(
                    "Time",
                    value=datetime.time(23, 59),
                    key="n_tm"
                )

            n_en = st.toggle(
                "Enable immediately for students",
                value=False, key="n_en"
            )

            with st.expander("Advanced options (title, grace period, auto-enable, instructions)"):
                n_title = st.text_input(
                    "Title", value=auto_title, key="n_title")
                n_gr = st.number_input(
                    "Grace period (minutes)",
                    value=15, min_value=0, max_value=120, key="n_gr")
                st.markdown(
                    "**Auto-enable date** (optional) — leave blank to enable manually:")
                col_ae_d, col_ae_t, col_ae_clr = st.columns([2, 2, 1])
                with col_ae_d:
                    n_ae_date = st.date_input(
                        "Auto-enable date",
                        value=None, key="n_ae_date")
                with col_ae_t:
                    n_ae_time = st.time_input(
                        "Auto-enable time",
                        value=datetime.time(0, 0), key="n_ae_time")
                with col_ae_clr:
                    st.markdown("<br>", unsafe_allow_html=True)
                    clear_ae = st.checkbox("Clear", key="clear_ae")
                n_in = st.text_area(
                    "Instructions (optional)",
                    key="n_in", height=70)
                st.caption(f"{len(n_in)} / 500 characters")

            st.markdown("<br>", unsafe_allow_html=True)

            if st.button("Add Assignment →", key="add_hw",
                         use_container_width=True):
                dl_str = (f"{n_dl.strftime('%Y-%m-%d')} "
                          f"{n_tm.strftime('%H:%M')}")
                dl_dt  = datetime.datetime.strptime(dl_str, "%Y-%m-%d %H:%M")

                f_title   = st.session_state.get("n_title", auto_title) or auto_title
                f_gr      = st.session_state.get("n_gr", 15) or 15
                f_in      = st.session_state.get("n_in", "") or ""
                ae_d      = st.session_state.get("n_ae_date", None)
                ae_t      = st.session_state.get("n_ae_time", datetime.time(0,0))
                clear_ae  = st.session_state.get("clear_ae", False)
                if ae_d and not clear_ae:
                    f_ae = f"{ae_d.strftime('%Y-%m-%d')} {ae_t.strftime('%H:%M')}"
                else:
                    f_ae = ""

                errors = []
                if not f_title.strip():
                    errors.append("Title cannot be empty.")
                if len(f_in) > 500:
                    errors.append(
                        f"Instructions too long ({len(f_in)} characters). "
                        f"Maximum is 500.")
                if dl_dt < datetime.datetime.now() and not n_en:
                    errors.append(
                        f"Deadline {dl_str} is already in the past. "
                        f"Set a future deadline, or enable immediately "
                        f"(it will show as closed).")

                if errors:
                    for e in errors:
                        st.error(e)
                else:
                    if dl_dt < datetime.datetime.now():
                        st.warning(
                            f"⚠ Deadline {dl_str} is in the past — "
                            f"homework will show as closed immediately.")

                    ok = add_hw_config(
                        sel, f_title.strip(),
                        str(n_en).upper(), dl_str,
                        f_gr, "", auto_marks, f_in.strip(),
                        f_ae
                    )

                    if ok:
                        log_audit("instructor", "HW_ADDED",
                                  f"{sel}: {f_title}")
                        ae_note = f" · Auto-enables: {f_ae}" if f_ae else ""
                        st.success(
                            f"✓ {sel} added — {auto_marks} marks · "
                            f"Deadline: {dl_str} · "
                            f"{'Enabled' if n_en else 'Disabled (enable when ready)'}"
                            f"{ae_note}"
                        )
                        st.rerun()
                    else:
                        st.error(
                            "Failed to write to Google Sheet. "
                            "Possible causes: corrupted rows in Config tab, "
                            "missing HW_ID header at row 10, or service account "
                            "does not have Editor access. "
                            "Clean the Config tab and try again."
                        )


# ════════════════════════════════════════════════════════════════════════════════
with tabs[2]:
    st.markdown('<div class="sec">Enrolled Students</div>',
                unsafe_allow_html=True)
    if st.button("↻ Refresh", key="ref_stu"):
        st.rerun()

    students = get_all_students()
    if students:
        df_s = pd.DataFrame(students)
        show = [c for c in ["Email", "First_Name", "Last_Name",
                             "Registered_At", "Last_Login"]
                if c in df_s.columns]
        st.dataframe(df_s[show], use_container_width=True)
        st.caption(f"{len(students)} enrolled")

        st.markdown('<div class="sec">Student Actions</div>',
                    unsafe_allow_html=True)
        sel_s = st.selectbox(
            "Select student",
            [s.get("Email", "") for s in students],
            key="sel_s"
        )
        cA, cB, cC = st.columns(3)

        with cA:
            if st.button("Force password reset", key="frc"):
                try:
                    ws   = get_tab(TAB_REGISTRY)
                    rows = ws.get_all_values()
                    for i, r in enumerate(rows[1:], start=2):
                        if (len(r) >= 1 and
                                r[0].strip().lower() == sel_s.strip().lower()):
                            ws.update_cell(i, 7, "TRUE")
                            break
                    log_audit("instructor", "FORCE_RESET", sel_s)
                    st.success("Student prompted to reset on next login.")
                except Exception as e:
                    st.error(str(e))

        with cB:
            tmp = st.text_input("Set temporary password", key="tmp_pw")
            if st.button("Set password", key="set_pw"):
                if tmp and len(tmp) >= 8:
                    if update_password(sel_s, tmp):
                        log_audit("instructor", "PW_SET", sel_s)
                        st.success(f"Password set. Tell student: {tmp}")
                    else:
                        st.error("Failed.")
                else:
                    st.error("Min 8 characters.")

        with cC:
            if st.button("Remove student", key="del_s"):
                if st.checkbox("Confirm removal", key="del_conf"):
                    if delete_student(sel_s):
                        log_audit("instructor", "STU_DEL", sel_s)
                        st.success("Removed.")
                        st.rerun()

        st.markdown('<div class="sec">Preview as Student</div>',
                    unsafe_allow_html=True)
        st.caption("See the app exactly as this student sees it.")
        prev_em = st.selectbox(
            "Preview as",
            [s.get("Email", "") for s in students],
            key="prev_em"
        )
        if st.button("Enter preview mode", key="enter_prev"):
            stu = next(
                (s for s in students
                 if s.get("Email", "").strip().lower() ==
                 prev_em.strip().lower()), {}
            )
            st.session_state["authenticated"]  = True
            st.session_state["student_email"]  = prev_em.strip().lower()
            st.session_state["student_name"]   = (
                f"{stu.get('First_Name', '')} "
                f"{stu.get('Last_Name', '')}".strip() or prev_em
            )
            st.session_state["student_record"] = stu
            st.session_state["submissions"]    = get_student_submissions(
                prev_em.strip().lower())
            st.session_state["hw_configs"]     = get_homework_configs()
            st.session_state["preview_mode"]   = True
            log_audit("instructor", "PREVIEW", prev_em)
            st.switch_page("pages/Dashboard.py")

        csv = df_s[show].to_csv(index=False).encode("utf-8")
        st.download_button("⬇ Download student list",
                           csv, "students.csv", "text/csv", key="dl_stu")
    else:
        st.info("No students enrolled yet.")

    st.markdown('<div class="sec">Bulk Enroll</div>', unsafe_allow_html=True)
    st.caption("One email per line. Registered with TempPass123, "
               "forced to reset on first login.")
    bulk_txt = st.text_area(
        "Email list", height=100, key="bulk_em",
        placeholder="student1@uni.edu\nstudent2@uni.edu")
    if st.button("Bulk Enroll", key="bulk_go"):
        if bulk_txt.strip():
            email_list = [e.strip() for e in bulk_txt.strip().splitlines()
                          if e.strip()]
            with st.spinner(f"Enrolling {len(email_list)}..."):
                added, skipped, errors = bulk_register(email_list)
            if added:
                log_audit("instructor", "BULK_ENROLL", f"{len(added)} added")
                st.success(f"Added: {len(added)}")
            if skipped:
                st.warning(f"Skipped (already exist): {len(skipped)}")
            if errors:
                st.error(f"Errors: {'; '.join(errors)}")
        else:
            st.error("Please enter at least one email.")


# ════════════════════════════════════════════════════════════════════════════════
with tabs[3]:
    st.markdown('<div class="sec">Change Instructor Password</div>',
                unsafe_allow_html=True)
    with st.form("ch_pw"):
        cur  = st.text_input("Current password", type="password")
        new1 = st.text_input("New password",     type="password")
        new2 = st.text_input("Confirm",          type="password")
        cs   = st.form_submit_button("Update Password")
    if cs:
        if not verify_instructor(cur):
            st.error("Current password incorrect.")
        elif len(new1) < 8:
            st.error("Min 8 characters.")
        elif new1 != new2:
            st.error("Passwords do not match.")
        else:
            update_instructor_password(new1)
            log_audit("instructor", "PW_CHANGED", "")
            st.success("Password updated.")

    st.markdown('<div class="sec">Audit Log</div>', unsafe_allow_html=True)
    try:
        ws      = get_tab("Config")
        vals    = ws.get_all_values()
        a_start = next(
            (i for i, r in enumerate(vals)
             if len(r) > 0 and r[0] == "Timestamp" and i > 20),
            None
        )
        if a_start is not None:
            data = vals[a_start + 1:]
            if data:
                df_a = pd.DataFrame(
                    data, columns=["Timestamp", "Actor", "Action", "Detail"])
                st.dataframe(df_a.iloc[::-1].reset_index(drop=True),
                             use_container_width=True)
            else:
                st.info("No audit entries yet.")
    except Exception as e:
        st.info(f"Could not load audit log: {e}")

    st.markdown('<div class="sec">End-of-Semester Reset</div>',
                unsafe_allow_html=True)
    st.markdown(
        '<div class="banner banner-warning">⚠ Read the README before using. '
        'Download all data first.</div>',
        unsafe_allow_html=True
    )
    with st.expander("Reset options"):
        st.markdown("**Step 1** — Download all data from Overview tab first.")
        if st.button("Clear all student accounts", key="clear_reg"):
            if st.checkbox("I have downloaded all data", key="conf_clr"):
                try:
                    ws = get_tab("Registry")
                    ws.clear()
                    ws.update("A1", [[
                        "Email", "Password_Hash", "First_Name",
                        "Last_Name", "Registered_At",
                        "Last_Login", "Force_Reset"
                    ]])
                    log_audit("instructor", "REGISTRY_CLEARED",
                              "End of semester")
                    st.success("Registry cleared.")
                except Exception as e:
                    st.error(str(e))

st.markdown("<br>", unsafe_allow_html=True)
if st.button("Sign out", key="inst_out"):
    log_audit("instructor", "LOGOUT", "")
    st.session_state.pop("instructor_auth", None)
    st.switch_page("Home.py")
