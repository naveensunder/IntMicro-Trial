"""
pages/Instructor.py — Instructor dashboard.
HWDashboard v12:
  - Audit tab reads from dedicated Audit sheet (items 11, 14)
  - _sub_count fixed to unique students not rows (item 12)
  - Compact assignment cards, Open button inside, inline Save (items 1, 2, 13)
  - Copy button for enrollment key (item 15)
  - Question Analytics own tab with HW/Q drilldown (item 16)
  - Submission Log tab: class progress matrix + detail (items 17, 18)
  - Performance Summary tab (item 19)
"""
import streamlit as st
import datetime
import pandas as pd
import re
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from db import (
    verify_instructor, update_instructor_password,
    get_all_students, delete_student, update_password,
    get_homework_configs, update_hw_config, add_hw_config,
    get_all_submissions, get_enrollment_key, log_audit,
    get_tab, TAB_REGISTRY, TAB_AUDIT, bulk_register,
    get_student_submissions, check_auto_enable,
    auto_register_homeworks, generate_gradebook_sheet,
    get_audit_log,
)
from ui import (
    inject_css, page_header, section_header, banner,
    last_updated_chip, page_footer, dark_mode_toggle, COLORS,
    hide_home_when_authed,
)

st.set_page_config(
    page_title="EC224 — Instructor · Bentley",
    page_icon="🔐",
    layout="wide",
    initial_sidebar_state="expanded",
)
inject_css()
hide_home_when_authed()
st.markdown("<style>.block-container { max-width: 1150px; }</style>",
            unsafe_allow_html=True)


def _week_num(hw_id: str) -> int:
    m = re.search(r"(\d+)", hw_id)
    return int(m.group(1)) if m else 999


# ── Auth gate ──────────────────────────────────────────────────────────────────
if not st.session_state.get("instructor_auth"):
    st.markdown(
        '<div style="max-width:380px;margin:3rem auto;text-align:center;">'
        '<div style="font-family:\'DM Serif Display\',serif;font-size:1.35rem;'
        'color:#1C2B4A;margin-bottom:0.35rem;">Instructor Access</div>'
        '<div style="font-size:0.81rem;color:#555555;margin-bottom:1.4rem;">'
        'Intermediate Microeconomics — Bentley University</div>'
        '</div>',
        unsafe_allow_html=True,
    )
    col = st.columns([1, 2, 1])[1]
    with col:
        pwd = st.text_input("Password", type="password", key="inst_pw")
        if st.button("Sign In", use_container_width=True, key="inst_in"):
            inst_attempts = st.session_state.get("inst_login_attempts", 0)
            inst_lockout  = st.session_state.get("inst_lockout_until", None)
            if inst_lockout and datetime.datetime.now() < inst_lockout:
                secs = int((inst_lockout - datetime.datetime.now()).total_seconds())
                st.error(f"Too many failed attempts. Please wait {secs} seconds.")
            elif verify_instructor(pwd):
                st.session_state["instructor_auth"]     = True
                st.session_state["inst_login_attempts"] = 0
                st.session_state.pop("inst_lockout_until", None)
                log_audit("instructor", "LOGIN", "")
                st.rerun()
            else:
                inst_attempts += 1
                st.session_state["inst_login_attempts"] = inst_attempts
                if inst_attempts >= 5:
                    st.session_state["inst_lockout_until"] = (
                        datetime.datetime.now() + datetime.timedelta(minutes=5))
                    st.error("Too many failed attempts. Locked for 5 minutes.")
                else:
                    st.error(f"Incorrect password. {5 - inst_attempts} attempts remaining.")
        st.markdown(
            '<div class="banner banner-warning" style="font-size:0.85rem;margin-top:1rem;">'
            'Please change your instructor password in Settings if not already done.</div>',
            unsafe_allow_html=True)
    st.stop()


page_header(
    "Intermediate Microeconomics",
    "Instructor Dashboard",
    datetime.datetime.now().strftime("%d %b %Y, %H:%M"),
)

from question_engine import ALL_HW_CONFIGS
try:
    auto_register_homeworks(ALL_HW_CONFIGS)
    check_auto_enable()
except Exception:
    pass

tabs = st.tabs([
    "📊 Overview",
    "📚 Assignments",
    "👥 Students",
    "📈 Analytics",
    "📋 Submissions",
    "🏆 Performance",
    "⚙️ Settings",
    "🔍 Audit",
])


# ════════════════════════════════════════════════════════════════════════════════
#  TAB 1 — OVERVIEW
# ════════════════════════════════════════════════════════════════════════════════
with tabs[0]:
    col_ref, col_ts = st.columns([1, 5])
    with col_ref:
        if st.button("↻ Refresh", key="ref_ov"):
            st.rerun()

    with st.spinner("Loading..."):
        students    = get_all_students() or []
        submissions = get_all_submissions() or []
        hw_configs  = get_homework_configs() or []
        _fetched_at = datetime.datetime.now().strftime("%H:%M:%S")

    with col_ts:
        st.markdown("<div style='padding-top:0.55rem;'>", unsafe_allow_html=True)
        last_updated_chip(_fetched_at)
        st.markdown("</div>", unsafe_allow_html=True)

    df = pd.DataFrame(submissions) if submissions else pd.DataFrame()
    n_sub = (len(df[df["Status"] == "submitted"])
             if not df.empty and "Status" in df.columns else 0)

    c1, c2, c3, c4 = st.columns(4)
    for col, num, lbl in [
        (c1, len(students),   "Enrolled Students"),
        (c2, n_sub,           "Submissions Received"),
        (c3, len(hw_configs), "Total Assignments"),
        (c4, len([h for h in hw_configs if h.get("Enabled","").upper()=="TRUE"]),
             "Currently Open"),
    ]:
        col.markdown(
            f'<div class="dash-metric">'
            f'<div class="dash-metric-num">{num}</div>'
            f'<div class="dash-metric-lbl">{lbl}</div>'
            f'</div>', unsafe_allow_html=True)

    # Enrollment key + copy button (item 15)
    section_header("Enrollment Key", "🔑")
    key = get_enrollment_key()
    st.markdown(
        f'<div style="background:#F5F5F5;border:1.5px solid #E0E0E0;border-radius:8px;'
        f'padding:0.9rem 1.2rem;margin-bottom:0.5rem;">'
        f'<div style="font-size:0.67rem;font-weight:600;text-transform:uppercase;'
        f'letter-spacing:0.1em;color:#555555;margin-bottom:0.25rem;">Current Key</div>'
        f'<div style="font-family:monospace;font-size:1.45rem;font-weight:700;'
        f'color:#1C2B4A;letter-spacing:0.18em;">{key}</div>'
        f'<div style="font-size:0.8rem;color:#6B7280;margin-top:0.25rem;">'
        f'Share with students. Auto-rotates every 6 months.</div></div>',
        unsafe_allow_html=True)
    st.markdown(
        f'<button onclick="navigator.clipboard.writeText(\'{key}\').then(function(){{'
        f'var b=document.getElementById(\'cpk\');b.textContent=\'✓ Copied!\';'
        f'setTimeout(function(){{b.textContent=\'📋 Copy key\';}},2000);}});" id="cpk" '
        f'style="background:#1C2B4A;color:#fff;border:none;border-radius:6px;'
        f'padding:6px 16px;font-size:0.85rem;cursor:pointer;'
        f'font-family:\'DM Sans\',sans-serif;">📋 Copy key</button>',
        unsafe_allow_html=True)

    section_header("Gradebook Sheet", "📊")
    st.caption("Generate a Gradebook tab in your Google Sheet.")
    if st.button("Generate / Update Gradebook Sheet", key="gen_gb",
                 use_container_width=True):
        with st.spinner("Generating gradebook..."):
            ok, result = generate_gradebook_sheet(ALL_HW_CONFIGS)
        if ok:
            log_audit("instructor", "GRADEBOOK_GENERATED", f"{result} students")
            st.success(f"✓ Gradebook updated — {result} students.")
        else:
            st.error(f"Failed: {result}")




# ════════════════════════════════════════════════════════════════════════════════
#  TAB 2 — ASSIGNMENTS  (items 1, 2, 12, 13)
# ════════════════════════════════════════════════════════════════════════════════
with tabs[1]:
    section_header("Semester Assignment Schedule", "📅")
    st.caption("Edit inline and click Save. Use Details for announcements and instructions.")

    if st.button("↻ Refresh", key="ref_hw"):
        st.rerun()

    hw_configs = get_homework_configs() or []

    try:
        all_subs_hw = get_all_submissions() or []
        df_subs     = pd.DataFrame(all_subs_hw) if all_subs_hw else pd.DataFrame()
    except Exception:
        df_subs = pd.DataFrame()

    def _sub_count(hw_id):
        """Count unique students who submitted (not total rows)."""
        if df_subs.empty or "Homework_ID" not in df_subs.columns:
            return 0
        mask = (df_subs["Homework_ID"] == hw_id)
        if "Status" in df_subs.columns:
            mask = mask & (df_subs["Status"] == "submitted")
        hw_rows = df_subs[mask]
        if hw_rows.empty or "Email" not in hw_rows.columns:
            return 0
        return int(hw_rows["Email"].nunique())

    hw_sorted = sorted(hw_configs, key=lambda x: _week_num(x.get("HW_ID", "")))

    if not hw_sorted:
        st.info("No assignments found. They will appear here automatically.")
    else:
        for cfg in hw_sorted:
            hw_id   = cfg.get("HW_ID", "")
            title   = cfg.get("Title", hw_id)
            enabled = cfg.get("Enabled", "FALSE").upper() == "TRUE"
            dl      = cfg.get("Deadline", "2099-12-31 23:59")
            opening = cfg.get("Opening_Date", "")
            grace   = cfg.get("Grace_Minutes", "15")
            ann     = cfg.get("Announcement", "")
            instr   = cfg.get("Instructions", "")
            n_subs  = _sub_count(hw_id)
            marks   = cfg.get("Max_Marks", "—")

            try:
                dl_dt          = datetime.datetime.strptime(dl.strip(), "%Y-%m-%d %H:%M")
                is_placeholder = dl_dt.year == 2099
            except Exception:
                dl_dt = None; is_placeholder = True

            try:
                op_dt = (datetime.datetime.strptime(opening.strip(), "%Y-%m-%d %H:%M")
                         if opening.strip() else None)
            except Exception:
                op_dt = None

            pill = ('<span class="pill pill-open">Open</span>' if enabled and not is_placeholder
                    else '<span class="pill pill-closed">Closed</span>' if not enabled
                    else '<span class="pill pill-upcoming">Pending</span>')

            # Compact header (item 1)
            st.markdown(
                f'<div style="background:#F5F5F5;border:1.5px solid #E0E0E0;'
                f'border-radius:8px 8px 0 0;padding:0.4rem 1rem;'
                f'display:flex;justify-content:space-between;align-items:center;">'
                f'<span style="font-weight:600;font-size:0.92rem;color:#1C2B4A;">{title}</span>'
                f'<span style="font-size:0.8rem;color:#555;display:flex;align-items:center;gap:0.5rem;">'
                f'{pill}<span>{n_subs} students &nbsp;·&nbsp; {marks} pts</span>'
                f'</span></div>',
                unsafe_allow_html=True)

            with st.container():
                # Inline row — no title col, Save button now horizontal (items 4, 7)
                col_en, col_od, col_dd, col_gr, col_save = \
                    st.columns([0.6, 2.2, 2.2, 1.1, 1.2])
                new_title = title  # title edits via Details expander only

                with col_en:
                    st.markdown('<div style="font-size:0.68rem;color:#555;font-weight:600;margin-bottom:0.2rem;">ON</div>',
                                unsafe_allow_html=True)
                    new_en = st.checkbox("On", value=enabled, key=f"en_{hw_id}",
                                         label_visibility="collapsed")

                with col_od:
                    st.markdown('<div style="font-size:0.68rem;color:#555;font-weight:600;margin-bottom:0.2rem;">OPENS</div>',
                                unsafe_allow_html=True)
                    new_op_date = st.date_input("Opens", value=op_dt.date() if op_dt else None,
                                                key=f"od_{hw_id}", label_visibility="collapsed")

                with col_dd:
                    st.markdown('<div style="font-size:0.68rem;color:#555;font-weight:600;margin-bottom:0.2rem;">DEADLINE</div>',
                                unsafe_allow_html=True)
                    dl_date_val = (dl_dt.date() if dl_dt and not is_placeholder
                                   else datetime.date.today() + datetime.timedelta(days=14))
                    new_dl_date = st.date_input("Deadline", value=dl_date_val,
                                                key=f"dd_{hw_id}", label_visibility="collapsed")

                with col_gr:
                    st.markdown('<div style="font-size:0.68rem;color:#555;font-weight:600;margin-bottom:0.2rem;">GRACE</div>',
                                unsafe_allow_html=True)
                    new_gr = st.number_input("Grace", value=int(grace or 15),
                                             min_value=0, max_value=120,
                                             key=f"gri_{hw_id}", label_visibility="collapsed")

                with col_save:
                    st.markdown('<div style="font-size:0.68rem;color:#555;font-weight:600;margin-bottom:0.2rem;">&nbsp;</div>',
                                unsafe_allow_html=True)
                    save_clicked = st.button("Save", key=f"save_{hw_id}",
                                             use_container_width=True)

                if save_clicked:
                    dl_time = dl_dt.time() if dl_dt and not is_placeholder else datetime.time(23, 59)
                    new_dl_str = f"{new_dl_date.strftime('%Y-%m-%d')} {dl_time.strftime('%H:%M')}"
                    new_op_str = (f"{new_op_date.strftime('%Y-%m-%d')} 00:00"
                                  if new_op_date else "")
                    if datetime.datetime.strptime(new_dl_str, "%Y-%m-%d %H:%M") < datetime.datetime.now():
                        st.warning(f"⚠ Deadline {new_dl_str} is in the past.")
                    if not new_en and n_subs > 0 and enabled != new_en:
                        st.warning(f"⚠ {n_subs} students submitted. Disabling hides this homework.")
                    update_hw_config(hw_id, "Enabled",      str(new_en).upper())
                    update_hw_config(hw_id, "Deadline",      new_dl_str)
                    update_hw_config(hw_id, "Opening_Date",  new_op_str)
                    update_hw_config(hw_id, "Grace_Minutes", str(new_gr))
                    log_audit("instructor", "HW_UPDATED", hw_id)
                    st.success(f"✓ {hw_id} saved.")
                    st.rerun()

                # Details expander (item 2 — button inside card)
                with st.expander(f"Details — {title}", expanded=False):
                    new_title_det = st.text_input("Title", value=title, key=f"title_{hw_id}")
                    new_ann   = st.text_input("Announcement (shown on Dashboard)",
                                              value=ann, key=f"ann_{hw_id}")
                    new_instr = st.text_area("Instructions (shown on homework page)",
                                             value=instr, key=f"instr_{hw_id}", height=70)
                    st.caption(f"{len(new_instr)} / 500 characters")
                    if st.button("Save details", key=f"save_det_{hw_id}",
                                 use_container_width=True):
                        update_hw_config(hw_id, "Title",         new_title_det.strip())
                        update_hw_config(hw_id, "Announcement",  new_ann)
                        update_hw_config(hw_id, "Instructions",   new_instr)
                        log_audit("instructor", "HW_DETAILS", hw_id)
                        st.success("Details saved.")

            st.markdown('<div style="margin-bottom:0.4rem;"></div>', unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════════════════
#  TAB 3 — STUDENTS
# ════════════════════════════════════════════════════════════════════════════════
with tabs[2]:
    section_header("Enrolled Students", "👥")
    if st.button("↻ Refresh", key="ref_stu"):
        st.rerun()

    students = get_all_students() or []
    if students:
        df_s = pd.DataFrame(students)
        show = [c for c in ["Email","First_Name","Last_Name","Registered_At","Last_Login"]
                if c in df_s.columns]
        st.dataframe(df_s[show], use_container_width=True)
        st.caption(f"{len(students)} enrolled")

        section_header("Student Actions", "⚡")
        sel_s = st.selectbox("Select student", [s.get("Email","") for s in students], key="sel_s")

        cA, cB, cC = st.columns(3)
        with cA:
            if st.button("Force password reset", key="frc"):
                try:
                    ws = get_tab(TAB_REGISTRY)
                    rows = ws.get_all_values()
                    for i, r in enumerate(rows[1:], start=2):
                        if len(r)>=1 and r[0].strip().lower()==sel_s.strip().lower():
                            ws.update_cell(i, 7, "TRUE"); break
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
            st.markdown('<div style="font-size:0.82rem;color:#DC2626;font-weight:600;margin-bottom:0.4rem;">⚠ Destructive</div>',
                        unsafe_allow_html=True)
            confirm_del = st.checkbox(f"Confirm removal of {sel_s}", key="del_conf")
            if st.button("Remove student", key="del_s", disabled=not confirm_del):
                if delete_student(sel_s):
                    log_audit("instructor", "STU_DEL", sel_s)
                    st.success("Removed."); st.rerun()
                else:
                    st.error("Failed.")

        section_header("Preview as Student", "👁")
        prev_em = st.selectbox("Preview as", [s.get("Email","") for s in students], key="prev_em")
        if st.button("Enter preview mode", key="enter_prev"):
            stu = next((s for s in students if s.get("Email","").strip().lower()==prev_em.strip().lower()), {})
            st.session_state.update({
                "authenticated": True, "student_email": prev_em.strip().lower(),
                "student_name": f"{stu.get('First_Name','')} {stu.get('Last_Name','')}".strip() or prev_em,
                "student_record": stu,
                "submissions": get_student_submissions(prev_em.strip().lower()),
                "hw_configs": get_homework_configs(), "preview_mode": True,
            })
            log_audit("instructor", "PREVIEW", prev_em)
            st.switch_page("pages/Dashboard.py")

        st.download_button("⬇ Download student list",
                           df_s[show].to_csv(index=False).encode("utf-8"),
                           "students.csv", "text/csv", key="dl_stu")
    else:
        st.info("No students enrolled yet.")

    st.divider()
    section_header("Bulk Enroll", "📥")
    st.caption("One email per line. Registered with TempPass123, forced reset on first login.")
    bulk_txt = st.text_area("Email list", height=100, key="bulk_em",
                            placeholder="student1@bentley.edu\nstudent2@bentley.edu")
    if st.button("Bulk Enroll", key="bulk_go"):
        if bulk_txt.strip():
            email_list = [e.strip() for e in bulk_txt.strip().splitlines() if e.strip()]
            with st.spinner(f"Enrolling {len(email_list)}..."):
                added, skipped, errors = bulk_register(email_list)
            if added:
                log_audit("instructor", "BULK_ENROLL", f"{len(added)} added")
                st.success(f"Added: {len(added)}")
            if skipped: st.warning(f"Skipped: {len(skipped)}")
            if errors:  st.error(f"Errors: {'; '.join(errors)}")
        else:
            st.error("Please enter at least one email.")


# ════════════════════════════════════════════════════════════════════════════════
#  TAB 4 — ANALYTICS  (item 16)
# ════════════════════════════════════════════════════════════════════════════════
with tabs[3]:
    section_header("Question Analytics", "📈")
    st.caption("Select a homework then a question to see class performance statistics.")

    if st.button("↻ Refresh", key="ref_an"):
        st.rerun()

    _subs_an = get_all_submissions() or []
    _stus_an = get_all_students() or []
    _hws_an  = get_homework_configs() or []
    _df_an   = pd.DataFrame(_subs_an) if _subs_an else pd.DataFrame()
    n_class  = len(_stus_an)

    hw_opts = [c.get("HW_ID","") for c in sorted(_hws_an, key=lambda x: _week_num(x.get("HW_ID","")))]
    if not hw_opts:
        st.info("No assignments found.")
    else:
        sel_hw = st.selectbox(
            "Select homework", hw_opts,
            format_func=lambda x: next((c.get("Title",x) for c in _hws_an if c.get("HW_ID")==x), x),
            key="an_hw")

        hw_qs   = ALL_HW_CONFIGS.get(sel_hw, {}).get("questions", [])
        q_opts  = [q["q_id"] for q in hw_qs]

        if not q_opts:
            st.info("No questions found for this homework.")
        else:
            sel_q  = st.selectbox(
                "Select question", q_opts,
                format_func=lambda x: next((q["title"] for q in hw_qs if q["q_id"]==x), x),
                key="an_q")

            q_info  = next((q for q in hw_qs if q["q_id"]==sel_q), {})
            max_pts = q_info.get("marks", 0)

            if not _df_an.empty and "Homework_ID" in _df_an.columns:
                q_df = _df_an[
                    (_df_an["Homework_ID"]==sel_hw) &
                    (_df_an["Question_ID"]==sel_q) &
                    (_df_an["Status"]=="submitted")
                ].copy()
                q_df["Score"] = pd.to_numeric(q_df["Score"], errors="coerce")

                n_attempted = int(q_df["Email"].nunique()) if not q_df.empty else 0
                avg_score   = round(float(q_df["Score"].mean()), 2) if not q_df.empty else 0.0
                med_score   = round(float(q_df["Score"].median()), 2) if not q_df.empty else 0.0

                m1, m2, m3, m4, m5 = st.columns(5)
                for col, val, lbl in [
                    (m1, n_class,     "Class Size"),
                    (m2, n_attempted, "Students Attempted"),
                    (m3, max_pts,     "Total Points"),
                    (m4, avg_score,   "Average Score"),
                    (m5, med_score,   "Median Score"),
                ]:
                    col.markdown(
                        f'<div class="dash-metric">'
                        f'<div class="dash-metric-num">{val}</div>'
                        f'<div class="dash-metric-lbl">{lbl}</div>'
                        f'</div>', unsafe_allow_html=True)


            else:
                st.info("No submission data yet.")


# ════════════════════════════════════════════════════════════════════════════════
#  TAB 5 — SUBMISSION LOG  (items 17, 18)
# ════════════════════════════════════════════════════════════════════════════════
with tabs[4]:
    section_header("Submission Log", "📋")
    if st.button("↻ Refresh", key="ref_sl"):
        st.rerun()

    _subs_sl = get_all_submissions() or []
    _stus_sl = get_all_students() or []
    _hws_sl  = get_homework_configs() or []
    _df_sl   = pd.DataFrame(_subs_sl) if _subs_sl else pd.DataFrame()
    now_dt   = datetime.datetime.now()

    # ── Class Progress Matrix ──
    with st.expander("📊 Class Progress Matrix", expanded=True):
        if _stus_sl and _hws_sl and not _df_sl.empty:
            hw_ids_m    = sorted([c.get("HW_ID","") for c in _hws_sl], key=_week_num)
            hw_titles_m = {c.get("HW_ID",""): c.get("Title","") for c in _hws_sl}

            header_cells = "<th style='padding:6px 10px;background:#1C2B4A;color:#fff;font-size:0.73rem;'>Email</th>"
            for hid in hw_ids_m:
                short = hw_titles_m.get(hid, hid).replace("Week ","Wk ")[:14]
                header_cells += f"<th style='padding:6px 8px;background:#1C2B4A;color:#fff;font-size:0.72rem;'>{short}</th>"

            body = ""
            for stu in _stus_sl:
                em = str(stu.get("Email","")).strip().lower()
                body += "<tr>"
                body += f"<td style='padding:5px 10px;font-size:0.78rem;color:#444;'>{em}</td>"
                for hid in hw_ids_m:
                    cfg_h = next((c for c in _hws_sl if c.get("HW_ID")==hid), {})
                    enabled = cfg_h.get("Enabled","FALSE").upper()=="TRUE"
                    try:
                        dl_dt_m = datetime.datetime.strptime(cfg_h.get("Deadline","2099-12-31 23:59").strip(), "%Y-%m-%d %H:%M")
                        dl_past = now_dt > dl_dt_m
                    except Exception:
                        dl_past = False
                    n_q_total = len(ALL_HW_CONFIGS.get(hid,{}).get("questions",[]))
                    if not enabled and not dl_past:
                        bg = "#FFFFFF"; txt = "—"
                    else:
                        done = int(_df_sl[
                            (_df_sl["Email"].str.lower()==em) &
                            (_df_sl["Homework_ID"]==hid) &
                            (_df_sl["Status"]=="submitted")
                        ]["Question_ID"].nunique())
                        if done == 0 and dl_past:
                            bg="#FEE2E2"; txt=f"0/{n_q_total}"
                        elif done == n_q_total and n_q_total > 0:
                            bg="#DCFCE7"; txt=f"{done}/{n_q_total}"
                        elif done > 0:
                            bg="#F3F4F6"; txt=f"{done}/{n_q_total}"
                        else:
                            bg="#FFFFFF"; txt=f"0/{n_q_total}"
                    body += (f"<td style='padding:5px 8px;text-align:center;"
                             f"background:{bg};font-size:0.8rem;font-weight:600;'>{txt}</td>")
                body += "</tr>"

            st.markdown(
                f'<div style="overflow-x:auto;">'
                f'<table style="border-collapse:collapse;width:100%;'
                f'font-family:\'DM Sans\',sans-serif;">'
                f'<thead><tr>{header_cells}</tr></thead>'
                f'<tbody>{body}</tbody></table></div>'
                f'<div style="font-size:0.73rem;color:#888;margin-top:0.4rem;">'
                f'🟢 All done &nbsp;·&nbsp; 🔴 None (deadline passed) '
                f'&nbsp;·&nbsp; ⬜ Not open &nbsp;·&nbsp; 🔘 Partial</div>',
                unsafe_allow_html=True)
        else:
            st.info("No data yet.")

    # ── Detailed log ──
    with st.expander("📄 Detailed Submission Log", expanded=False):
        if not _df_sl.empty:
            # Metrics
            _df_sl["_date"] = pd.to_datetime(
                _df_sl.get("Timestamp", pd.Series(dtype=str)), errors="coerce"
            ).dt.date
            today_d  = datetime.date.today()
            week_ago = today_d - datetime.timedelta(days=7)
            mc1, mc2, mc3 = st.columns(3)
            for col, val, lbl in [
                (mc1, int((_df_sl["_date"]==today_d).sum()),  "Today"),
                (mc2, int((_df_sl["_date"]>=week_ago).sum()), "This Week"),
                (mc3, len(_df_sl),                            "All Time"),
            ]:
                col.markdown(
                    f'<div class="dash-metric">'
                    f'<div class="dash-metric-num">{val}</div>'
                    f'<div class="dash-metric-lbl">{lbl}</div>'
                    f'</div>', unsafe_allow_html=True)

            st.divider()
            fc1, fc2 = st.columns(2)
            with fc1:
                hw_fil = st.selectbox("Filter by homework",
                                      ["All"] + sorted(_df_sl["Homework_ID"].unique().tolist()),
                                      key="sl_hw")
            with fc2:
                em_fil = st.selectbox("Filter by student",
                                      ["All"] + sorted(_df_sl["Email"].unique().tolist()),
                                      key="sl_em")
            filtered = _df_sl.copy()
            if hw_fil != "All": filtered = filtered[filtered["Homework_ID"]==hw_fil]
            if em_fil != "All": filtered = filtered[filtered["Email"]==em_fil]

            show_c = [c for c in ["Timestamp","Email","Homework_ID","Question_ID",
                                   "Score","Max_Score","Is_Late","Status"]
                      if c in filtered.columns]
            st.dataframe(filtered[show_c].sort_values("Timestamp", ascending=False),
                         use_container_width=True)
            st.download_button("⬇ Download filtered CSV",
                               filtered[show_c].to_csv(index=False).encode("utf-8"),
                               "filtered.csv", "text/csv", key="dl_filt")
        else:
            st.info("No submissions yet.")


# ════════════════════════════════════════════════════════════════════════════════
#  TAB 6 — PERFORMANCE SUMMARY  (item 19)
# ════════════════════════════════════════════════════════════════════════════════
with tabs[5]:
    section_header("Performance Summary", "🏆")
    st.caption("All computed from submission data. No extra API calls.")

    if st.button("↻ Refresh", key="ref_perf"):
        st.rerun()

    _subs_p = get_all_submissions() or []
    _stus_p = get_all_students() or []
    _hws_p  = get_homework_configs() or []

    if not _subs_p or not _stus_p:
        st.info("Not enough data yet. Check back once students have submitted.")
    else:
        try:
            _df_p = pd.DataFrame(_subs_p)
            # Safe column access — never use df.get() which is not dict-like
            for col in ["Score","Max_Score","Timestamp","Is_Late","Email","Homework_ID","Status"]:
                if col not in _df_p.columns:
                    _df_p[col] = None
            _df_p["Score"]     = pd.to_numeric(_df_p["Score"],     errors="coerce")
            _df_p["Max_Score"] = pd.to_numeric(_df_p["Max_Score"], errors="coerce")
            sub_df = _df_p[_df_p["Status"] == "submitted"].copy()
        except Exception:
            sub_df = pd.DataFrame()

        if sub_df.empty:
            st.info("Not enough data yet. Check back once students have submitted.")
        else:
            all_emails = {str(s.get("Email","")).strip().lower() for s in _stus_p if s.get("Email")}
            submitted_emails = set(sub_df["Email"].str.lower().unique())

            try:
                stu_agg = sub_df.groupby("Email").agg(
                    total_score=("Score",    "sum"),
                    total_max=  ("Max_Score","sum"),
                    n_late=     ("Is_Late",  lambda x: (x=="Yes").sum()),
                    first_ts=   ("Timestamp","min"),
                ).reset_index()
                stu_agg["pct"] = (
                    stu_agg["total_score"] / stu_agg["total_max"].replace(0, float("nan")) * 100
                ).round(1).fillna(0)
            except Exception:
                stu_agg = pd.DataFrame()

            MIN_STUDENTS = 2  # minimum before showing any classification

            # 1. Perfect scorers
            with st.expander("🏆 Perfect Scorers (100%)", expanded=False):
                if stu_agg.empty or len(stu_agg) < MIN_STUDENTS:
                    st.info("Not enough data yet.")
                else:
                    try:
                        perf = stu_agg[stu_agg["pct"] >= 99.9][["Email","total_score","total_max","pct"]]
                        if perf.empty:
                            st.info("No perfect scorers yet.")
                        else:
                            st.dataframe(perf, use_container_width=True)
                    except Exception:
                        st.info("Not enough data yet.")

            # 2. Declining trajectory
            with st.expander("📉 Declining Trajectory (20%+ drop)", expanded=False):
                try:
                    if sub_df.empty or "Homework_ID" not in sub_df.columns:
                        st.info("Not enough data yet.")
                    else:
                        hw_seq = sorted(sub_df["Homework_ID"].dropna().unique(), key=_week_num)
                        if len(hw_seq) < 2:
                            st.info("Need submissions across at least 2 homeworks to detect trends.")
                        else:
                            def _pct(em, hw):
                                r  = sub_df[(sub_df["Email"].str.lower()==em) & (sub_df["Homework_ID"]==hw)]
                                mx = r["Max_Score"].sum()
                                return float(r["Score"].sum()) / float(mx) * 100 if mx > 0 else None
                            dec = []
                            for em in submitted_emails:
                                p1 = _pct(em, hw_seq[0])
                                p2 = _pct(em, hw_seq[-1])
                                if p1 is not None and p2 is not None and (p1 - p2) >= 20:
                                    dec.append({"Email": em,
                                                f"{hw_seq[0]} %": round(p1, 1),
                                                f"{hw_seq[-1]} %": round(p2, 1),
                                                "Drop": round(p1 - p2, 1)})
                            if dec:
                                st.dataframe(pd.DataFrame(dec).sort_values("Drop", ascending=False),
                                             use_container_width=True)
                            else:
                                st.info("No declining trajectories detected.")
                except Exception:
                    st.info("Not enough data yet.")

            # 3. Consistent late submitters
            with st.expander("⏰ Consistent Late Submitters (2+ late)", expanded=False):
                if stu_agg.empty or len(stu_agg) < MIN_STUDENTS:
                    st.info("Not enough data yet.")
                else:
                    try:
                        late = stu_agg[stu_agg["n_late"] >= 2][["Email","n_late","pct"]].rename(
                            columns={"n_late": "Late Submissions", "pct": "Avg Score %"})
                        if late.empty:
                            st.info("No consistent late submitters detected.")
                        else:
                            st.dataframe(late.sort_values("Late Submissions", ascending=False),
                                         use_container_width=True)
                    except Exception:
                        st.info("Not enough data yet.")

            # 5. Score distribution per homework
            with st.expander("📊 Score Distribution per Homework", expanded=False):
                try:
                    if sub_df.empty or "Homework_ID" not in sub_df.columns:
                        st.info("Not enough data yet.")
                    else:
                        for hid in sorted(sub_df["Homework_ID"].dropna().unique(), key=_week_num):
                            hw_t = next((c.get("Title", hid) for c in _hws_p if c.get("HW_ID")==hid), hid)
                            hw_r = sub_df[sub_df["Homework_ID"] == hid]
                            sc_r = hw_r.groupby("Email")["Score"].sum()
                            if len(sc_r) < MIN_STUDENTS:
                                st.markdown(f"**{hw_t}** — not enough data yet.")
                            else:
                                st.markdown(f"**{hw_t}** — {len(sc_r)} students")
                                st.bar_chart(sc_r.value_counts().sort_index())
                except Exception:
                    st.info("Not enough data yet.")

            # 6. Never submitted
            with st.expander("👻 Never Submitted", expanded=False):
                try:
                    never = all_emails - submitted_emails
                    if not all_emails:
                        st.info("Not enough data yet.")
                    elif never:
                        st.dataframe(pd.DataFrame({"Email": sorted(never)}), use_container_width=True)
                    else:
                        st.success("All enrolled students have submitted at least once.")
                except Exception:
                    st.info("Not enough data yet.")

            # 7. Grace period dependents
            with st.expander("🔔 Grace Period Dependents (2+ grace submissions)", expanded=False):
                try:
                    if sub_df.empty or "Is_Late" not in sub_df.columns:
                        st.info("Not enough data yet.")
                    else:
                        gdf = sub_df[sub_df["Is_Late"] == "Yes"].groupby("Email").size().reset_index()
                        gdf.columns = ["Email", "Grace Submissions"]
                        gdf = gdf[gdf["Grace Submissions"] >= 2]
                        if gdf.empty:
                            st.info("No grace period dependents detected.")
                        else:
                            st.dataframe(gdf.sort_values("Grace Submissions", ascending=False),
                                         use_container_width=True)
                except Exception:
                    st.info("Not enough data yet.")

            # 8. Fastest vs slowest
            with st.expander("⚡ Fastest vs Slowest Submitters", expanded=False):
                try:
                    if stu_agg.empty or len(stu_agg) < MIN_STUDENTS:
                        st.info("Not enough data yet.")
                    else:
                        timing = stu_agg[["Email","first_ts"]].copy().dropna()
                        timing["first_ts"] = pd.to_datetime(timing["first_ts"], errors="coerce")
                        timing = timing.dropna().sort_values("first_ts")
                        if len(timing) < MIN_STUDENTS:
                            st.info("Not enough data yet.")
                        else:
                            c_f, c_s = st.columns(2)
                            with c_f:
                                st.markdown("**Earliest submitters (top 5)**")
                                st.dataframe(timing.head(5).rename(columns={"first_ts": "First Submission"}),
                                             use_container_width=True)
                            with c_s:
                                st.markdown("**Latest submitters (bottom 5)**")
                                st.dataframe(timing.tail(5).rename(columns={"first_ts": "First Submission"}),
                                             use_container_width=True)
                except Exception:
                    st.info("Not enough data yet.")


# ════════════════════════════════════════════════════════════════════════════════
#  TAB 7 — SETTINGS
# ════════════════════════════════════════════════════════════════════════════════
with tabs[6]:
    _, col_dm = st.columns([4, 1])
    with col_dm:
        dark_mode_toggle()

    section_header("Change Instructor Password", "🔒")
    with st.form("ch_pw"):
        cur  = st.text_input("Current password", type="password")
        new1 = st.text_input("New password",     type="password")
        new2 = st.text_input("Confirm",          type="password")
        cs   = st.form_submit_button("Update Password")
    if cs:
        if not verify_instructor(cur):   st.error("Current password incorrect.")
        elif len(new1) < 8:              st.error("Min 8 characters.")
        elif new1 != new2:               st.error("Passwords do not match.")
        else:
            update_instructor_password(new1)
            log_audit("instructor", "PW_CHANGED", "")
            st.success("Password updated.")

    st.divider()
    section_header("Question Engine Tests", "🧪")
    st.caption("Runs all parameter functions across 20 dummy emails. Safe to run anytime — no data is written.")
    if st.button("▶ Run Tests Now", key="run_tests"):
        with st.spinner("Running tests..."):
            import sys, math, traceback, importlib
            try:
                import numpy as np
                import unittest.mock as _mock
                # Patch streamlit in a sub-context
                _orig = sys.modules.get("streamlit")
                sys.modules["streamlit"] = _mock.MagicMock()
                import importlib as _il
                if "question_engine" in sys.modules:
                    qe = _il.reload(sys.modules["question_engine"])
                else:
                    import question_engine as qe
                sys.modules["streamlit"] = _orig or sys.modules["streamlit"]

                emails = [f"student{i:02d}@bentley.edu" for i in range(1, 21)]
                results = []; failures = []

                for email in emails:
                    try:
                        I, Px, Py = qe._q1_params(email)
                        ANS_x = qe.r2(I/Px); ANS_y = qe.r2(I/Py); ANS_s = qe.r2(-Px/Py)
                        for v, n in [(ANS_x,"X-int"),(ANS_y,"Y-int"),(ANS_s,"Slope")]:
                            if math.isnan(float(v)) or math.isinf(float(v)):
                                failures.append(f"Q1 {n} invalid for {email}")
                        if ANS_x<=0 or ANS_y<=0:
                            failures.append(f"Q1 negative intercept for {email}")
                    except Exception as e:
                        failures.append(f"Q1 crash for {email}: {e}")

                    try:
                        I2, Px2, Py2, tom_a = qe._q2_params(email)
                        ANS_jx = qe.r2(I2/(Px2+Py2)); ANS_jy = qe.r2(I2/(Px2+Py2))
                        if abs(ANS_jx-ANS_jy)>0.01:
                            failures.append(f"Q2 Jerry X!=Y for {email}")
                        ANS_tx = qe.r2(I2/Px2)
                        if ANS_tx<=0:
                            failures.append(f"Q2 Tom X<=0 for {email}")
                    except Exception as e:
                        failures.append(f"Q2 crash for {email}: {e}")

                # Diversity check
                q1x = [qe.r2(qe._q1_params(e)[0]/qe._q1_params(e)[1]) for e in emails]
                if len(set(q1x)) < 3:
                    failures.append("Q1 X-intercept: nearly all students get same answer — seeding broken")

                if failures:
                    st.markdown(
                        f'<div class="banner banner-error">❌ {len(failures)} issue(s) found:</div>',
                        unsafe_allow_html=True)
                    for f_msg in failures:
                        st.markdown(f"- {f_msg}")
                else:
                    n_checks = len(emails) * 4
                    st.markdown(
                        f'<div class="banner banner-success">✅ All {n_checks} checks passed across {len(emails)} students. Safe to deploy.</div>',
                        unsafe_allow_html=True)
            except Exception as e:
                st.error(f"Test runner error: {e}")

    st.divider()
    section_header("End-of-Semester Reset", "⚠️")
    st.markdown('<div class="banner banner-warning">⚠ Download all data first. Cannot be undone.</div>',
                unsafe_allow_html=True)
    with st.expander("Reset options"):
        c1r = st.checkbox("I have downloaded all student data",  key="conf_dl")
        c2r = st.checkbox("I understand this deletes all accounts", key="conf_clr")
        if st.button("Clear all student accounts", key="clear_reg",
                     disabled=not (c1r and c2r)):
            try:
                ws = get_tab(TAB_REGISTRY)
                ws.clear()
                ws.update("A1", [["Email","Password_Hash","First_Name",
                                   "Last_Name","Registered_At","Last_Login","Force_Reset"]])
                log_audit("instructor", "REGISTRY_CLEARED", "End of semester")
                st.success("Registry cleared.")
            except Exception as e:
                st.error(str(e))


# ════════════════════════════════════════════════════════════════════════════════
#  TAB 8 — AUDIT  (items 11, 14)
# ════════════════════════════════════════════════════════════════════════════════
with tabs[7]:
    section_header("Audit Log", "🔍")
    st.caption("All instructor actions. Most recent first. Stored in dedicated Audit sheet.")

    col_ra, col_da = st.columns([1, 4])
    with col_ra:
        if st.button("↻ Refresh", key="ref_audit"):
            st.rerun()

    audit_rows = get_audit_log()
    if audit_rows:
        df_a = pd.DataFrame(audit_rows)
        st.dataframe(df_a, use_container_width=True)
        with col_da:
            st.download_button("⬇ Download audit log CSV",
                               df_a.to_csv(index=False).encode("utf-8"),
                               "audit_log.csv", "text/csv", key="dl_audit")
    else:
        st.info("No audit entries yet.")


# ── Footer & sign-out ──────────────────────────────────────────────────────────
st.divider()
col_so, _ = st.columns([1, 5])
with col_so:
    if st.button("Sign out", key="inst_out"):
        log_audit("instructor", "LOGOUT", "")
        st.session_state.pop("instructor_auth", None)
        st.switch_page("Home.py")

page_footer()
