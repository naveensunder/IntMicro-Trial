"""
db.py — All Google Sheets operations, auth logic, session helpers.
Single source of truth for data access across the entire app.
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
    "Registered_At", "Last_Login", "Force_Reset"
]

SUBMISSIONS_HEADER = [
    "Timestamp", "Email", "Homework_ID", "Question_ID",
    "Question_Type", "Status", "Is_Late", "Reloads",
    "Param_Seed", "Raw_Answer", "Score", "Max_Score",
    "Correct_Answer", "Version"
]

CONFIG_HEADER = [
    "Key", "Value"
]

HW_CONFIG_HEADER = [
    "HW_ID", "Title", "Enabled", "Deadline",
    "Grace_Minutes", "Announcement", "Version", "Max_Marks"
]

AUDIT_HEADER = [
    "Timestamp", "Actor", "Action", "Detail"
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
        ws = sh.add_worksheet(title=tab_name, rows=1000, cols=30)
        return ws


# ── Sheet initialisation (run once on first deploy) ────────────────────────────
def init_sheets():
    """Create all tabs and headers if they don't exist."""
    sh = get_spreadsheet()
    existing = [ws.title for ws in sh.worksheets()]

    def ensure_tab(name, header):
        if name not in existing:
            ws = sh.add_worksheet(title=name, rows=1000, cols=len(header) + 5)
            ws.update("A1", [header])
        else:
            ws = sh.worksheet(name)
            try:
                first = ws.cell(1, 1).value
                if not first:
                    ws.update("A1", [header])
            except Exception:
                pass
        return ws

    ensure_tab(TAB_REGISTRY, REGISTRY_HEADER)
    ensure_tab(TAB_SUBMISSIONS, SUBMISSIONS_HEADER)

    # Config tab: two sections — app config + homework config
    if TAB_CONFIG not in existing:
        ws = sh.add_worksheet(title=TAB_CONFIG, rows=500, cols=20)
        # App config section
        ws.update("A1", [["Key", "Value"]])
        ws.update("A2", [
            ["enrollment_key", _generate_enrollment_key()],
            ["enrollment_key_created", datetime.datetime.now().strftime("%Y-%m-%d")],
            ["instructor_password_hash", hash_password("Microeconomics")],
        ])
        # Homework config section (starts at row 10)
        ws.update("A10", [HW_CONFIG_HEADER])
        ws.update("A11", [[
            "HW_WEEK2",
            "Week 2 — Budget Constraints & Optimal Bundles",
            "TRUE",
            "2027-04-30 23:59",
            "15",
            "",
            "1",
            "14"
        ]])
        # Audit log section (starts at row 100)
        ws.update("A100", [AUDIT_HEADER])


# ── Password utilities ─────────────────────────────────────────────────────────
def hash_password(password: str) -> str:
    return hashlib.sha256(password.strip().encode()).hexdigest()


def verify_password(password: str, hashed: str) -> bool:
    return hash_password(password) == hashed


def _generate_enrollment_key(length: int = 10) -> str:
    alphabet = string.ascii_uppercase + string.digits
    return "".join(secrets.choice(alphabet) for _ in range(length))


# ── Enrollment key ────────────────────────────────────────────────────────────
def get_enrollment_key() -> str:
    try:
        ws = get_tab(TAB_CONFIG)
        rows = ws.get_all_values()
        for row in rows[1:]:
            if len(row) >= 2 and row[0] == "enrollment_key":
                key = row[1]
                # Check if key needs rotation (6 months)
                created_row = next(
                    (r for r in rows[1:] if len(r) >= 2 and r[0] == "enrollment_key_created"),
                    None
                )
                if created_row:
                    try:
                        created = datetime.datetime.strptime(created_row[1], "%Y-%m-%d")
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


def _update_config_value(key: str, value: str):
    try:
        ws = get_tab(TAB_CONFIG)
        rows = ws.get_all_values()
        for i, row in enumerate(rows):
            if len(row) >= 1 and row[0] == key:
                ws.update_cell(i + 1, 2, value)
                return True
    except Exception:
        pass
    return False


# ── Registry operations ────────────────────────────────────────────────────────
def get_all_students():
    try:
        ws = get_tab(TAB_REGISTRY)
        rows = ws.get_all_records()
        return rows
    except Exception:
        return []


def get_student(email: str):
    students = get_all_students()
    email = email.strip().lower()
    for s in students:
        if str(s.get("Email", "")).strip().lower() == email:
            return s
    return None


def register_student(email: str, password: str, first_name: str, last_name: str) -> tuple:
    try:
        existing = get_student(email)
        if existing:
            return False, "An account with this email already exists."
        ws = get_tab(TAB_REGISTRY)
        # Ensure header
        first_cell = ws.cell(1, 1).value
        if not first_cell:
            ws.update("A1", [REGISTRY_HEADER])
        ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        row = [
            email.strip().lower(),
            hash_password(password),
            first_name.strip(),
            last_name.strip(),
            ts, ts, "FALSE"
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
    # Update last login
    try:
        ws = get_tab(TAB_REGISTRY)
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
        ws = get_tab(TAB_REGISTRY)
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
        ws = get_tab(TAB_REGISTRY)
        rows = ws.get_all_values()
        for i, row in enumerate(rows[1:], start=2):
            if len(row) >= 1 and row[0].strip().lower() == email.strip().lower():
                ws.delete_rows(i)
                return True
    except Exception:
        pass
    return False


def bulk_register_students(email_list: list) -> tuple:
    """Register multiple students with temporary passwords."""
    added = []; skipped = []; errors = []
    for email in email_list:
        email = email.strip().lower()
        if not email or "@" not in email:
            continue
        existing = get_student(email)
        if existing:
            skipped.append(email)
            continue
        ok, err = register_student(email, "TempPass123", "", "")
        if ok:
            added.append(email)
        else:
            errors.append(f"{email}: {err}")
    return added, skipped, errors


# ── Homework config operations ────────────────────────────────────────────────
def get_homework_configs() -> list:
    try:
        ws = get_tab(TAB_CONFIG)
        rows = ws.get_all_values()
        # Find the HW_CONFIG_HEADER row
        hw_start = None
        for i, row in enumerate(rows):
            if len(row) > 0 and row[0] == "HW_ID":
                hw_start = i
                break
        if hw_start is None:
            return []
        header = rows[hw_start]
        configs = []
        for row in rows[hw_start + 1:]:
            if not row or not row[0]:
                break
            config = {}
            for j, col in enumerate(header):
                config[col] = row[j] if j < len(row) else ""
            configs.append(config)
        return configs
    except Exception:
        return []


def update_homework_config(hw_id: str, field: str, value: str) -> bool:
    try:
        ws = get_tab(TAB_CONFIG)
        rows = ws.get_all_values()
        hw_start = None
        header = []
        for i, row in enumerate(rows):
            if len(row) > 0 and row[0] == "HW_ID":
                hw_start = i
                header = row
                break
        if hw_start is None:
            return False
        col_idx = header.index(field) + 1 if field in header else None
        if col_idx is None:
            return False
        for i, row in enumerate(rows[hw_start + 1:], start=hw_start + 2):
            if len(row) > 0 and row[0] == hw_id:
                ws.update_cell(i, col_idx, value)
                return True
    except Exception:
        pass
    return False


def add_homework_config(hw_id, title, enabled, deadline, grace, announcement, version, max_marks):
    try:
        ws = get_tab(TAB_CONFIG)
        rows = ws.get_all_values()
        # Find first empty row after HW section
        hw_start = None
        for i, row in enumerate(rows):
            if len(row) > 0 and row[0] == "HW_ID":
                hw_start = i
                break
        if hw_start is None:
            return False
        insert_row = hw_start + 2
        for i, row in enumerate(rows[hw_start + 1:], start=hw_start + 1):
            if not row or not row[0]:
                insert_row = i + 1
                break
            insert_row = i + 2
        ws.insert_row(
            [hw_id, title, str(enabled), deadline, str(grace), announcement, str(version), str(max_marks)],
            insert_row
        )
        return True
    except Exception:
        return False


# ── Submission operations ─────────────────────────────────────────────────────
def write_submission(row_data: list) -> tuple:
    try:
        ws = get_tab(TAB_SUBMISSIONS)
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
        ws = get_tab(TAB_SUBMISSIONS)
        rows = ws.get_all_records()
        for row in rows:
            if str(row.get("Email", "")).strip().lower() == email.strip().lower():
                hw_id = str(row.get("Homework_ID", ""))
                q_id  = str(row.get("Question_ID", ""))
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


# ── Instructor auth ───────────────────────────────────────────────────────────
def verify_instructor(password: str) -> bool:
    try:
        ws = get_tab(TAB_CONFIG)
        rows = ws.get_all_values()
        for row in rows[1:]:
            if len(row) >= 2 and row[0] == "instructor_password_hash":
                return verify_password(password, row[1])
    except Exception:
        pass
    return False


def update_instructor_password(new_password: str) -> bool:
    return _update_config_value("instructor_password_hash", hash_password(new_password))


# ── Audit log ─────────────────────────────────────────────────────────────────
def log_audit(actor: str, action: str, detail: str = ""):
    try:
        ws = get_tab(TAB_CONFIG)
        rows = ws.get_all_values()
        # Find audit section
        audit_start = None
        for i, row in enumerate(rows):
            if len(row) > 0 and row[0] == "Timestamp" and i > 50:
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
        dl = datetime.datetime.strptime(deadline_str.strip(), "%Y-%m-%d %H:%M")
        dl_with_grace = dl + datetime.timedelta(minutes=grace_minutes)
        now = datetime.datetime.now()
        past_hard   = now > dl_with_grace
        past_soft   = now > dl
        return past_hard, past_soft, dl, dl_with_grace
    except Exception:
        far_future = datetime.datetime(2099, 12, 31, 23, 59)
        return False, False, far_future, far_future


# ── Login attempt throttle (session-based) ────────────────────────────────────
def check_login_attempts() -> bool:
    """Returns True if allowed to attempt login."""
    attempts = st.session_state.get("login_attempts", 0)
    lockout_until = st.session_state.get("lockout_until", None)
    if lockout_until:
        if datetime.datetime.now() < lockout_until:
            remaining = (lockout_until - datetime.datetime.now()).seconds
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
