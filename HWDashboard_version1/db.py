"""
db.py — Google Sheets operations.
HWDashboard v3 — stability-first build.
"""

import streamlit as st
import gspread
from google.oauth2.service_account import Credentials
import hashlib
import datetime
import secrets
import string

TAB_REGISTRY    = "Registry"
TAB_SUBMISSIONS = "Submissions"
TAB_CONFIG      = "Config"

SHEET_ID = "1QQHk9bf9kC-35im2mswvUwSTymnb4rCv1yYLgbfu9xA"

REGISTRY_HEADER    = ["Email","Password_Hash","First_Name","Last_Name",
                       "Registered_At","Last_Login","Force_Reset"]
SUBMISSIONS_HEADER = ["Timestamp","Email","Homework_ID","Question_ID",
                       "Question_Type","Status","Is_Late","Raw_Answer",
                       "Score","Max_Score","Correct_Answer"]
HW_CONFIG_HEADER   = ["HW_ID","Title","Enabled","Deadline",
                       "Grace_Minutes","Announcement","Max_Marks","Instructions",
                       "Auto_Enable_Date"]
AUDIT_HEADER       = ["Timestamp","Actor","Action","Detail"]


# ── Connection ─────────────────────────────────────────────────────────────────
@st.cache_resource(ttl=300)
def _get_gc():
    creds = Credentials.from_service_account_info(
        dict(st.secrets["gcp_service_account"]),
        scopes=[
            "https://www.googleapis.com/auth/spreadsheets",
            "https://www.googleapis.com/auth/drive",
        ],
    )
    return gspread.authorize(creds)


def get_spreadsheet():
    return _get_gc().open_by_key(SHEET_ID)


def get_tab(name: str):
    sh = get_spreadsheet()
    try:
        return sh.worksheet(name)
    except gspread.WorksheetNotFound:
        return sh.add_worksheet(title=name, rows=2000, cols=25)


# ── Init ───────────────────────────────────────────────────────────────────────
def init_sheets():
    sh       = get_spreadsheet()
    existing = [ws.title for ws in sh.worksheets()]

    def ensure(name, header):
        if name not in existing:
            ws = sh.add_worksheet(title=name, rows=2000, cols=len(header)+2)
            ws.update("A1", [header])
        else:
            ws = sh.worksheet(name)
            if not ws.cell(1,1).value:
                ws.update("A1", [header])

    ensure(TAB_REGISTRY,    REGISTRY_HEADER)
    ensure(TAB_SUBMISSIONS, SUBMISSIONS_HEADER)

    if TAB_CONFIG not in existing:
        ws = sh.add_worksheet(title=TAB_CONFIG, rows=500, cols=15)
        ws.update("A1",  [["Key","Value"]])
        ws.update("A2",  [
            ["enrollment_key",         _gen_key()],
            ["enrollment_key_created", datetime.datetime.now().strftime("%Y-%m-%d")],
            ["instructor_password_hash", hash_pw("Microeconomics")],
        ])
        ws.update("A10", [HW_CONFIG_HEADER])
        ws.update("A11", [[
            "HW_WEEK2",
            "Week 2 — Budget Constraints & Optimal Bundles",
            "TRUE",
            "2027-04-30 23:59",
            "15", "", "18",
            "This homework covers budget constraints, optimal bundles, "
            "and utility maximisation with different preference types."
        ]])
        ws.update("A30", [AUDIT_HEADER])
    else:
        # Repair if empty
        ws   = sh.worksheet(TAB_CONFIG)
        rows = ws.get_all_values()
        has_key = any(len(r)>0 and r[0]=="enrollment_key" for r in rows)
        if not has_key:
            ws.update("A1", [["Key","Value"]])
            ws.update("A2", [
                ["enrollment_key",         _gen_key()],
                ["enrollment_key_created", datetime.datetime.now().strftime("%Y-%m-%d")],
                ["instructor_password_hash", hash_pw("Microeconomics")],
            ])
        has_hw = any(len(r)>0 and r[0]=="HW_ID" for r in rows)
        if not has_hw:
            ws.update("A10", [HW_CONFIG_HEADER])
            ws.update("A11", [[
                "HW_WEEK2",
                "Week 2 — Budget Constraints & Optimal Bundles",
                "TRUE", "2027-04-30 23:59", "15", "", "18",
                "This homework covers budget constraints and utility maximisation."
            ]])
        has_audit = any(len(r)>0 and r[0]=="Timestamp" and i>20
                        for i,r in enumerate(rows))
        if not has_audit:
            ws.update("A30", [AUDIT_HEADER])


# ── Passwords ──────────────────────────────────────────────────────────────────
def hash_pw(pw: str) -> str:
    return hashlib.sha256(pw.strip().encode()).hexdigest()

def verify_pw(pw: str, hashed: str) -> bool:
    return hash_pw(pw) == hashed

def _gen_key(n=10) -> str:
    return "".join(secrets.choice(string.ascii_uppercase+string.digits)
                   for _ in range(n))


# ── Enrollment key ─────────────────────────────────────────────────────────────
def get_enrollment_key() -> str:
    try:
        ws   = get_tab(TAB_CONFIG)
        rows = ws.get_all_values()
        key  = next((r[1] for r in rows if len(r)>=2 and r[0]=="enrollment_key"), None)
        return key or "ERROR"
    except Exception:
        return "ERROR"


def _set_config(key: str, value: str):
    try:
        ws   = get_tab(TAB_CONFIG)
        rows = ws.get_all_values()
        for i,r in enumerate(rows):
            if len(r)>=1 and r[0]==key:
                ws.update_cell(i+1, 2, value)
                return
        ws.append_row([key, value])
    except Exception:
        pass


# ── Registry ───────────────────────────────────────────────────────────────────
def get_all_students() -> list:
    try:
        return get_tab(TAB_REGISTRY).get_all_records()
    except Exception:
        return []


def get_student(email: str) -> dict:
    email = email.strip().lower()
    for s in get_all_students():
        if str(s.get("Email","")).strip().lower() == email:
            return s
    return {}


def register_student(email: str, password: str,
                     first: str, last: str) -> tuple:
    try:
        if get_student(email):
            return False, "An account with this email already exists."
        ws = get_tab(TAB_REGISTRY)
        if not ws.cell(1,1).value:
            ws.update("A1", [REGISTRY_HEADER])
        ts  = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        ws.append_row([email.strip().lower(), hash_pw(password),
                       first.strip(), last.strip(), ts, ts, "FALSE"])
        return True, ""
    except Exception as e:
        return False, str(e)[:120]


def authenticate(email: str, password: str) -> tuple:
    s = get_student(email)
    if not s:
        return False, "No account found with this email."
    if not verify_pw(password, str(s.get("Password_Hash",""))):
        return False, "Incorrect password."
    try:
        ws   = get_tab(TAB_REGISTRY)
        rows = ws.get_all_values()
        for i,r in enumerate(rows[1:], start=2):
            if len(r)>=1 and r[0].strip().lower()==email.strip().lower():
                ws.update_cell(i, 6,
                    datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
                break
    except Exception:
        pass
    return True, s


def update_password(email: str, new_pw: str) -> bool:
    try:
        ws   = get_tab(TAB_REGISTRY)
        rows = ws.get_all_values()
        for i,r in enumerate(rows[1:], start=2):
            if len(r)>=1 and r[0].strip().lower()==email.strip().lower():
                ws.update_cell(i, 2, hash_pw(new_pw))
                ws.update_cell(i, 7, "FALSE")
                return True
    except Exception:
        pass
    return False


def delete_student(email: str) -> bool:
    try:
        ws   = get_tab(TAB_REGISTRY)
        rows = ws.get_all_values()
        for i,r in enumerate(rows[1:], start=2):
            if len(r)>=1 and r[0].strip().lower()==email.strip().lower():
                ws.delete_rows(i)
                return True
    except Exception:
        pass
    return False


def bulk_register(emails: list) -> tuple:
    added=[]; skipped=[]; errors=[]
    for email in emails:
        email = email.strip().lower()
        if not email or "@" not in email:
            continue
        if get_student(email):
            skipped.append(email)
            continue
        ok, err = register_student(email, "TempPass123", "", "")
        if ok:
            try:
                ws   = get_tab(TAB_REGISTRY)
                rows = ws.get_all_values()
                for i,r in enumerate(rows[1:], start=2):
                    if len(r)>=1 and r[0].strip().lower()==email:
                        ws.update_cell(i, 7, "TRUE")
                        break
            except Exception:
                pass
            added.append(email)
        else:
            errors.append(f"{email}: {err}")
    return added, skipped, errors


# ── Homework config ────────────────────────────────────────────────────────────
def get_homework_configs() -> list:
    try:
        ws   = get_tab(TAB_CONFIG)
        rows = ws.get_all_values()
        start = next((i for i,r in enumerate(rows)
                      if len(r)>0 and r[0]=="HW_ID"), None)
        if start is None:
            return []
        header  = rows[start]
        configs = []
        for r in rows[start+1:]:
            if not r or not r[0] or r[0] in ["Key","Timestamp","Email"]:
                continue
            cfg = {header[j]: r[j] if j<len(r) else ""
                   for j in range(len(header))}
            if cfg.get("HW_ID"):
                configs.append(cfg)
        return configs
    except Exception:
        return []


def update_hw_config(hw_id: str, field: str, value: str) -> bool:
    try:
        ws   = get_tab(TAB_CONFIG)
        rows = ws.get_all_values()
        start = next((i for i,r in enumerate(rows)
                      if len(r)>0 and r[0]=="HW_ID"), None)
        if start is None:
            return False
        header = rows[start]
        if field not in header:
            return False
        col = header.index(field)+1
        for i,r in enumerate(rows[start+1:], start=start+2):
            if len(r)>0 and r[0]==hw_id:
                ws.update_cell(i, col, value)
                return True
    except Exception:
        pass
    return False


def add_hw_config(hw_id, title, enabled, deadline,
                  grace, announcement, max_marks, instructions,
                  auto_enable_date="") -> bool:
    """
    Add a new homework config row using append_row() — the most reliable
    gspread method. No position calculation, no insert_row() fragility,
    no range string needed. Works identically across all gspread versions.
    """
    try:
        ws = get_tab(TAB_CONFIG)
        ws.append_row(
            [hw_id, title, str(enabled), deadline,
             str(grace), announcement, str(max_marks),
             instructions, auto_enable_date],
            value_input_option="RAW",
            insert_data_option="INSERT_ROWS",
            table_range="A10"
        )
        return True
    except Exception:
        return False


# ── Submissions ────────────────────────────────────────────────────────────────
def write_submission(row: list) -> tuple:
    try:
        ws = get_tab(TAB_SUBMISSIONS)
        if not ws.cell(1,1).value:
            ws.update("A1", [SUBMISSIONS_HEADER])
        ws.append_row(row)
        return True, ""
    except Exception as e:
        return False, str(e)[:120]


def get_student_submissions(email: str) -> dict:
    result = {}
    try:
        rows = get_tab(TAB_SUBMISSIONS).get_all_records()
        for r in rows:
            if str(r.get("Email","")).strip().lower()==email.strip().lower():
                hw = str(r.get("Homework_ID",""))
                q  = str(r.get("Question_ID",""))
                if hw and q:
                    result.setdefault(hw,{})[q] = r
    except Exception:
        pass
    return result


def get_all_submissions() -> list:
    try:
        return get_tab(TAB_SUBMISSIONS).get_all_records()
    except Exception:
        return []


# ── Instructor auth ────────────────────────────────────────────────────────────
def verify_instructor(pw: str) -> bool:
    try:
        rows = get_tab(TAB_CONFIG).get_all_values()
        for r in rows:
            if len(r)>=2 and r[0]=="instructor_password_hash":
                return verify_pw(pw, r[1])
    except Exception:
        pass
    return False


def update_instructor_password(new_pw: str) -> bool:
    _set_config("instructor_password_hash", hash_pw(new_pw))
    return True


# ── Auto-enable check ─────────────────────────────────────────────────────────
def check_auto_enable():
    """Enable homeworks whose Auto_Enable_Date has passed. Run on app load."""
    try:
        configs = get_homework_configs()
        now     = datetime.datetime.now()
        for cfg in configs:
            auto_date = cfg.get("Auto_Enable_Date", "").strip()
            enabled   = cfg.get("Enabled", "").upper() == "TRUE"
            if auto_date and not enabled:
                try:
                    ae_dt = datetime.datetime.strptime(auto_date, "%Y-%m-%d %H:%M")
                    if now >= ae_dt:
                        update_hw_config(cfg["HW_ID"], "Enabled", "TRUE")
                        log_audit("system", "AUTO_ENABLED", cfg["HW_ID"])
                except Exception:
                    pass
    except Exception:
        pass


# ── Audit ──────────────────────────────────────────────────────────────────────
def log_audit(actor: str, action: str, detail: str=""):
    try:
        ws   = get_tab(TAB_CONFIG)
        rows = ws.get_all_values()
        start = next((i for i,r in enumerate(rows)
                      if len(r)>0 and r[0]=="Timestamp" and i>20), None)
        if start is None:
            return
        ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        ws.append_row([ts, actor, action, detail])
    except Exception:
        pass


# ── Deadline ───────────────────────────────────────────────────────────────────
def parse_deadline(deadline_str: str, grace_minutes: int=15):
    try:
        dl    = datetime.datetime.strptime(deadline_str.strip(), "%Y-%m-%d %H:%M")
        dlg   = dl + datetime.timedelta(minutes=grace_minutes)
        now   = datetime.datetime.now()
        return now>dlg, now>dl, dl, dlg
    except Exception:
        far = datetime.datetime(2099,12,31,23,59)
        return False, False, far, far


# ── Login throttle ─────────────────────────────────────────────────────────────
def check_login_attempts() -> bool:
    lockout = st.session_state.get("lockout_until")
    if lockout and datetime.datetime.now() < lockout:
        secs = int((lockout - datetime.datetime.now()).total_seconds())
        st.error(f"Too many failed attempts. Please wait {secs} seconds.")
        return False
    return True


def record_failed_attempt():
    n = st.session_state.get("login_attempts", 0) + 1
    st.session_state["login_attempts"] = n
    if n >= 5:
        st.session_state["lockout_until"] = (
            datetime.datetime.now() + datetime.timedelta(minutes=5))


def reset_login_attempts():
    st.session_state["login_attempts"] = 0
    st.session_state.pop("lockout_until", None)
