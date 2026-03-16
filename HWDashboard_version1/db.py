"""
db.py — All Google Sheets operations, auth logic, session helpers.
HWDashboard v2 — Phase 1
"""

import streamlit as st
import gspread
from google.oauth2.service_account import Credentials
import hashlib
import datetime
import secrets
import string

# ── Sheet tab names ────────────────────────────────────────────────────────────
TAB_REGISTRY    = "Registry"
TAB_SUBMISSIONS = "Submissions"
TAB_CONFIG      = "Config"

SHEET_ID = "1QQHk9bf9kC-35im2mswvUwSTymnb4rCv1yYLgbfu9xA"

# ── Headers ────────────────────────────────────────────────────────────────────
REGISTRY_HEADER = [
    "Email", "Password_Hash", "First_Name", "Last_Name",
    "Registered_At", "Last_Login", "Force_Reset", "Email_Verified"
]

SUBMISSIONS_HEADER = [
    "Timestamp", "Email", "Homework_ID", "Question_ID",
    "Question_Type", "Status", "Is_Late", "Reloads",
    "Param_Seed", "Raw_Answer", "Score", "Max_Score",
    "Correct_Answer", "Version"
]

HW_CONFIG_HEADER = [
    "HW_ID", "Title", "Enabled", "Deadline",
    "Grace_Minutes", "Announcement", "Version",
    "Max_Marks", "Instructions", "Auto_Enable_Date"
]

AUDIT_HEADER = ["Timestamp", "Actor", "Action", "Detail"]

EXTENSION_HEADER = [
    "Email", "HW_ID", "Requested_At", "Reason",
    "Status", "Custom_Deadline", "Decided_At"
]

INDIVIDUAL_ACCESS_HEADER = [
    "Email", "HW_ID", "Custom_Deadline", "Granted_At"
]


# ── Connection ─────────────────────────────────────────────────────────────────
@st.cache_resource(ttl=300)
def _get_gc():
    creds_dict = dict(st.secrets["gcp_service_account"])
    creds = Credentials.from_service_account_info(
        creds_dict,
        scopes=[
            "https://www.googleapis.com/auth/spreadsheets",
            "https://www.googleapis.com/auth/drive",
        ],
    )
    return gspread.authorize(creds)


def get_spreadsheet():
    gc = _get_gc()
    return gc.open_by_key(SHEET_ID)


def get_tab(tab_name: str):
    sh = get_spreadsheet()
    try:
        return sh.worksheet(tab_name)
    except gspread.WorksheetNotFound:
        ws = sh.add_worksheet(title=tab_name, rows=2000, cols=30)
        return ws


# ── Sheet initialisation ───────────────────────────────────────────────────────
def init_sheets():
    sh = get_spreadsheet()
    existing = [ws.title for ws in sh.worksheets()]

    def ensure_tab(name, header):
        if name not in existing:
            ws = sh.add_worksheet(title=name, rows=2000, cols=len(header) + 5)
            ws.update("A1", [header])
            return ws, True
        else:
            ws = sh.worksheet(name)
            try:
                first = ws.cell(1, 1).value
                if not first:
                    ws.update("A1", [header])
            except Exception:
                pass
            return ws, False

    ensure_tab(TAB_REGISTRY, REGISTRY_HEADER)
    ensure_tab(TAB_SUBMISSIONS, SUBMISSIONS_HEADER)

    if TAB_CONFIG not in existing:
        ws = sh.add_worksheet(title=TAB_CONFIG, rows=1000, cols=20)
        # App config
        ws.update("A1", [["Key", "Value"]])
        ws.update("A2", [
            ["enrollment_key", _generate_enrollment_key()],
            ["enrollment_key_created", datetime.datetime.now().strftime("%Y-%m-%d")],
            ["instructor_password_hash", hash_password("Microeconomics")],
        ])
        # HW config header at row 10
        ws.update("A10", [HW_CONFIG_HEADER])
        ws.update("A11", [[
            "HW_WEEK2",
            "Week 2 — Budget Constraints & Optimal Bundles",
            "TRUE",
            "2027-04-30 23:59",
            "15",
            "",
            "1",
            "14",
            "This homework covers budget constraints, optimal consumption bundles, and utility maximisation.",
            ""
        ]])
        # Extension requests header at row 100
        ws.update("A100", [EXTENSION_HEADER])
        # Individual access header at row 200
        ws.update("A200", [INDIVIDUAL_ACCESS_HEADER])
        # Audit log header at row 300
        ws.update("A300", [AUDIT_HEADER])
    else:
        # Ensure existing config tab has all required sections
        ws = sh.worksheet(TAB_CONFIG)
        try:
            rows = ws.get_all_values()
            has_key = any(len(r) > 0 and r[0] == "enrollment_key" for r in rows)
            if not has_key:
                ws.update("A1", [["Key", "Value"]])
                ws.update("A2", [
                    ["enrollment_key", _generate_enrollment_key()],
                    ["enrollment_key_created", datetime.datetime.now().strftime("%Y-%m-%d")],
                    ["instructor_password_hash", hash_password("Microeconomics")],
                ])
            has_hw = any(len(r) > 0 and r[0] == "HW_ID" for r in rows)
            if not has_hw:
                ws.update("A10", [HW_CONFIG_HEADER])
                ws.update("A11", [[
                    "HW_WEEK2",
                    "Week 2 — Budget Constraints & Optimal Bundles",
                    "TRUE",
                    "2027-04-30 23:59",
                    "15", "", "1", "14",
                    "This homework covers budget constraints, optimal consumption bundles, and utility maximisation.",
                    ""
                ]])
            has_ext = any(len(r) > 0 and r[0] == "Email" and i > 50
                         for i, r in enumerate(rows))
            if not has_ext:
                ws.update("A100", [EXTENSION_HEADER])
                ws.update("A200", [INDIVIDUAL_ACCESS_HEADER])
                ws.update("A300", [AUDIT_HEADER])
        except Exception:
            pass


# ── Password utilities ─────────────────────────────────────────────────────────
def hash_password(password: str) -> str:
    return hashlib.sha256(password.strip().encode()).hexdigest()


def verify_password(password: str, hashed: str) -> bool:
    return hash_password(password) == hashed


def _generate_enrollment_key(length: int = 10) -> str:
    alphabet = string.ascii_uppercase + string.digits
    return "".join(secrets.choice(alphabet) for _ in range(length))


# ── Enrollment key ─────────────────────────────────────────────────────────────
def get_enrollment_key() -> str:
    try:
        ws   = get_tab(TAB_CONFIG)
        rows = ws.get_all_values()
        key  = None
        created_str = None
        for row in rows[1:]:
            if len(row) >= 2 and row[0] == "enrollment_key":
                key = row[1]
            if len(row) >= 2 and row[0] == "enrollment_key_created":
                created_str = row[1]
        if key:
            if created_str:
                try:
                    created = datetime.datetime.strptime(created_str, "%Y-%m-%d")
                    if (datetime.datetime.now() - created).days > 180:
                        new_key = _generate_enrollment_key()
                        _update_config_value("enrollment_key", new_key)
                        _update_config_value(
                            "enrollment_key_created",
                            datetime.datetime.now().strftime("%Y-%m-%d")
                        )
                        return new_key
                except Exception:
                    pass
            return key
    except Exception:
        pass
    return "ERROR"


def _update_config_value(key: str, value: str) -> bool:
    try:
        ws   = get_tab(TAB_CONFIG)
        rows = ws.get_all_values()
        for i, row in enumerate(rows):
            if len(row) >= 1 and row[0] == key:
                ws.update_cell(i + 1, 2, value)
                return True
        # Not found — append
        ws.append_row([key, value])
        return True
    except Exception:
        return False


# ── Registry ───────────────────────────────────────────────────────────────────
def get_all_students() -> list:
    try:
        ws = get_tab(TAB_REGISTRY)
        rows = ws.get_all_records()
        return rows
    except Exception:
        return []


def get_student(email: str) -> dict:
    try:
        ws    = get_tab(TAB_REGISTRY)
        rows  = ws.get_all_records()
        email = email.strip().lower()
        for s in rows:
            if str(s.get("Email", "")).strip().lower() == email:
                return s
    except Exception:
        pass
    return {}


def register_student(email: str, password: str,
                     first_name: str, last_name: str) -> tuple:
    try:
        existing = get_student(email)
        if existing:
            return False, "An account with this email already exists."
        ws = get_tab(TAB_REGISTRY)
        first_cell = ws.cell(1, 1).value
        if not first_cell:
            ws.update("A1", [REGISTRY_HEADER])
        ts  = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        row = [
            email.strip().lower(),
            hash_password(password),
            first_name.strip(),
            last_name.strip(),
            ts, ts, "FALSE", "FALSE"
        ]
        ws.append_row(row)
        return True, ""
    except Exception as e:
        return False, str(e)[:120]


def authenticate_student(email: str, password: str) -> tuple:
    student = get_student(email)
    if not student:
        return False, "No account found with this email."
    if not verify_password(password, str(student.get("Password_Hash", ""))):
        return False, "Incorrect password."
    try:
        ws   = get_tab(TAB_REGISTRY)
        rows = ws.get_all_values()
        for i, row in enumerate(rows[1:], start=2):
            if len(row) >= 1 and row[0].strip().lower() == email.strip().lower():
                ws.update_cell(i, 6, datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
                break
    except Exception:
        pass
    return True, student


def update_password(email: str, new_password: str) -> bool:
    try:
        ws   = get_tab(TAB_REGISTRY)
        rows = ws.get_all_values()
        for i, row in enumerate(rows[1:], start=2):
            if len(row) >= 1 and row[0].strip().lower() == email.strip().lower():
                ws.update_cell(i, 2, hash_password(new_password))
                ws.update_cell(i, 7, "FALSE")
                return True
    except Exception:
        pass
    return False


def delete_student(email: str) -> bool:
    try:
        ws   = get_tab(TAB_REGISTRY)
        rows = ws.get_all_values()
        for i, row in enumerate(rows[1:], start=2):
            if len(row) >= 1 and row[0].strip().lower() == email.strip().lower():
                ws.delete_rows(i)
                return True
    except Exception:
        pass
    return False


def bulk_register_students(email_list: list) -> tuple:
    added = []; skipped = []; errors = []
    for email in email_list:
        email = email.strip().lower()
        if not email or "@" not in email:
            continue
        if get_student(email):
            skipped.append(email)
            continue
        ok, err = register_student(email, "TempPass123", "", "")
        if ok:
            # Mark force reset
            try:
                ws   = get_tab(TAB_REGISTRY)
                rows = ws.get_all_values()
                for i, row in enumerate(rows[1:], start=2):
                    if len(row) >= 1 and row[0].strip().lower() == email:
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
        hw_start = None
        for i, row in enumerate(rows):
            if len(row) > 0 and row[0] == "HW_ID":
                hw_start = i
                break
        if hw_start is None:
            return []
        header  = rows[hw_start]
        configs = []
        for row in rows[hw_start + 1:]:
            if not row or not row[0] or row[0].startswith("#"):
                continue
            # Stop at next section header
            if row[0] in ["Email", "Timestamp", "Key"]:
                break
            config = {}
            for j, col in enumerate(header):
                config[col] = row[j] if j < len(row) else ""
            if config.get("HW_ID"):
                configs.append(config)
        return configs
    except Exception:
        return []


def update_homework_config(hw_id: str, field: str, value: str) -> bool:
    try:
        ws   = get_tab(TAB_CONFIG)
        rows = ws.get_all_values()
        hw_start = None
        header   = []
        for i, row in enumerate(rows):
            if len(row) > 0 and row[0] == "HW_ID":
                hw_start = i
                header   = row
                break
        if hw_start is None or field not in header:
            return False
        col_idx = header.index(field) + 1
        for i, row in enumerate(rows[hw_start + 1:], start=hw_start + 2):
            if len(row) > 0 and row[0] == hw_id:
                ws.update_cell(i, col_idx, value)
                return True
    except Exception:
        pass
    return False


def add_homework_config(hw_id, title, enabled, deadline,
                        grace, announcement, version, max_marks,
                        instructions="", auto_enable_date="") -> bool:
    try:
        ws   = get_tab(TAB_CONFIG)
        rows = ws.get_all_values()
        hw_start = None
        for i, row in enumerate(rows):
            if len(row) > 0 and row[0] == "HW_ID":
                hw_start = i
                break
        if hw_start is None:
            return False
        # Find insert position
        insert_row = hw_start + 2
        for i, row in enumerate(rows[hw_start + 1:], start=hw_start + 1):
            if not row or not row[0] or row[0] in ["Email", "Timestamp", "Key"]:
                insert_row = i + 1
                break
            insert_row = i + 2
        ws.insert_row(
            [hw_id, title, str(enabled), deadline, str(grace),
             announcement, str(version), str(max_marks),
             instructions, auto_enable_date],
            insert_row
        )
        return True
    except Exception:
        return False


# ── Auto-enable check ──────────────────────────────────────────────────────────
def check_auto_enable():
    """Enable homeworks whose Auto_Enable_Date has passed."""
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
                        update_homework_config(cfg["HW_ID"], "Enabled", "TRUE")
                        log_audit("system", "AUTO_ENABLED", cfg["HW_ID"])
                except Exception:
                    pass
    except Exception:
        pass


# ── Individual access overrides ────────────────────────────────────────────────
def get_individual_access(email: str, hw_id: str) -> dict:
    """Returns custom access record if exists, else empty dict."""
    try:
        ws   = get_tab(TAB_CONFIG)
        rows = ws.get_all_values()
        ia_start = None
        for i, row in enumerate(rows):
            if len(row) > 0 and row[0] == "Email" and i > 150:
                ia_start = i
                break
        if ia_start is None:
            return {}
        header = rows[ia_start]
        for row in rows[ia_start + 1:]:
            if not row or not row[0]:
                break
            rec = {header[j]: row[j] if j < len(row) else ""
                   for j in range(len(header))}
            if (rec.get("Email","").strip().lower() == email.strip().lower()
                    and rec.get("HW_ID","") == hw_id):
                return rec
    except Exception:
        pass
    return {}


def grant_individual_access(email: str, hw_id: str,
                             custom_deadline: str) -> bool:
    try:
        ws   = get_tab(TAB_CONFIG)
        rows = ws.get_all_values()
        ia_start = None
        for i, row in enumerate(rows):
            if len(row) > 0 and row[0] == "Email" and i > 150:
                ia_start = i
                break
        if ia_start is None:
            ws.update("A200", [INDIVIDUAL_ACCESS_HEADER])
            ia_start = 199
        ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        ws.append_row([email, hw_id, custom_deadline, ts])
        return True
    except Exception:
        return False


# ── Extension requests ─────────────────────────────────────────────────────────
def submit_extension_request(email: str, hw_id: str, reason: str) -> bool:
    try:
        ws   = get_tab(TAB_CONFIG)
        rows = ws.get_all_values()
        ext_start = None
        for i, row in enumerate(rows):
            if len(row) > 0 and row[0] == "Email" and i > 50 and i < 150:
                ext_start = i
                break
        if ext_start is None:
            ws.update("A100", [EXTENSION_HEADER])
        ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        ws.append_row([email, hw_id, ts, reason, "Pending", "", ""])
        return True
    except Exception:
        return False


def get_extension_requests(status: str = None) -> list:
    try:
        ws   = get_tab(TAB_CONFIG)
        rows = ws.get_all_values()
        ext_start = None
        for i, row in enumerate(rows):
            if len(row) > 0 and row[0] == "Email" and i > 50 and i < 150:
                ext_start = i
                break
        if ext_start is None:
            return []
        header  = rows[ext_start]
        results = []
        for row in rows[ext_start + 1:]:
            if not row or not row[0]:
                continue
            rec = {header[j]: row[j] if j < len(row) else ""
                   for j in range(len(header))}
            if status is None or rec.get("Status","") == status:
                results.append(rec)
        return results
    except Exception:
        return []


# ── Submission operations ─────────────────────────────────────────────────────
def write_submission(row_data: list) -> tuple:
    try:
        ws    = get_tab(TAB_SUBMISSIONS)
        first = ws.cell(1, 1).value
        if not first:
            ws.update("A1", [SUBMISSIONS_HEADER])
        ws.append_row(row_data)
        return True, ""
    except Exception as e:
        return False, str(e)[:120]


def get_student_submissions(email: str) -> dict:
    """Returns {hw_id: {question_id: latest_row}} for a student."""
    result = {}
    try:
        ws   = get_tab(TAB_SUBMISSIONS)
        rows = ws.get_all_records()
        for row in rows:
            if str(row.get("Email","")).strip().lower() == email.strip().lower():
                hw_id = str(row.get("Homework_ID",""))
                q_id  = str(row.get("Question_ID",""))
                if hw_id and q_id:
                    if hw_id not in result:
                        result[hw_id] = {}
                    result[hw_id][q_id] = row
    except Exception:
        pass
    return result


def get_all_submissions() -> list:
    try:
        ws = get_tab(TAB_SUBMISSIONS)
        return ws.get_all_records()
    except Exception:
        return []


# ── Instructor auth ────────────────────────────────────────────────────────────
def verify_instructor(password: str) -> bool:
    try:
        ws   = get_tab(TAB_CONFIG)
        rows = ws.get_all_values()
        for row in rows[1:]:
            if len(row) >= 2 and row[0] == "instructor_password_hash":
                return verify_password(password, row[1])
    except Exception:
        pass
    return False


def update_instructor_password(new_password: str) -> bool:
    return _update_config_value("instructor_password_hash", hash_password(new_password))


# ── Audit log ──────────────────────────────────────────────────────────────────
def log_audit(actor: str, action: str, detail: str = ""):
    try:
        ws   = get_tab(TAB_CONFIG)
        rows = ws.get_all_values()
        audit_start = None
        for i, row in enumerate(rows):
            if len(row) > 0 and row[0] == "Timestamp" and i > 250:
                audit_start = i
                break
        if audit_start is None:
            return
        ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        ws.append_row([ts, actor, action, detail])
    except Exception:
        pass


# ── Deadline utilities ─────────────────────────────────────────────────────────
def parse_deadline(deadline_str: str, grace_minutes: int = 15):
    try:
        dl           = datetime.datetime.strptime(deadline_str.strip(), "%Y-%m-%d %H:%M")
        dl_with_grace = dl + datetime.timedelta(minutes=grace_minutes)
        now          = datetime.datetime.now()
        return now > dl_with_grace, now > dl, dl, dl_with_grace
    except Exception:
        far = datetime.datetime(2099, 12, 31, 23, 59)
        return False, False, far, far


# ── Login throttle ─────────────────────────────────────────────────────────────
def check_login_attempts() -> bool:
    attempts      = st.session_state.get("login_attempts", 0)
    lockout_until = st.session_state.get("lockout_until", None)
    if lockout_until:
        if datetime.datetime.now() < lockout_until:
            remaining = int((lockout_until - datetime.datetime.now()).total_seconds())
            st.error(f"Too many failed attempts. Please wait {remaining} seconds.")
            return False
        else:
            st.session_state["login_attempts"] = 0
            st.session_state["lockout_until"]  = None
    return True


def record_failed_attempt():
    attempts = st.session_state.get("login_attempts", 0) + 1
    st.session_state["login_attempts"] = attempts
    if attempts >= 5:
        st.session_state["lockout_until"] = (
            datetime.datetime.now() + datetime.timedelta(minutes=5)
        )


def reset_login_attempts():
    st.session_state["login_attempts"] = 0
    st.session_state["lockout_until"]  = None
