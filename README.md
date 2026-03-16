# HWDashboard_version1 — Complete Guide
## Learning Intermediate Microeconomics — Prof. Naveen Sunder

---

## File Structure

```
HWDashboard_version1/
├── Home.py                  ← Login page (entry point)
├── db.py                    ← All database / Google Sheets operations
├── ui.py                    ← Shared CSS and design system
├── question_engine.py       ← All question logic and rendering
├── requirements.txt
├── secrets.toml             ← Your credentials (DO NOT upload to GitHub)
└── pages/
    ├── Dashboard.py         ← Student landing page (homework list)
    ├── Homework.py          ← Homework question renderer
    └── Instructor.py        ← Instructor dashboard (password protected)
```

---

## What This System Does

- Students log in with email + password
- New students register using an enrollment key you provide
- Each student gets a unique version of every question (seeded by email)
- Answers, scores, and timestamps are written to Google Sheets automatically
- Students can leave and return — progress is always restored
- Once a question is submitted, it is locked permanently
- A deadline system with grace period controls all submissions
- You manage everything (deadlines, student accounts, homework enable/disable)
  directly from the Instructor Dashboard — no code changes needed
- Copying detection compares answers across students with different parameters

---

## Part 1 — First-Time Setup

### Step 1 — Google Sheet

1. Go to drive.google.com → New → Google Sheets
2. Name it something like `Microeconomics_HW_Dashboard`
3. The Sheet ID is already set in `secrets.toml`:
   `1QQHk9bf9kC-35im2mswvUwSTymnb4rCv1yYLgbfu9xA`

   **Important:** The service account has already been given access to this Sheet
   (you did this earlier). The Sheet will auto-create three tabs on first run:
   `Registry`, `Submissions`, `Config`.
   You do not need to create them manually.

### Step 2 — GitHub Repository

1. Go to github.com → Sign in → New repository
2. Name it e.g. `microeconomics-hw-dashboard`
3. Set to **Private** (recommended) or Public
4. Upload ALL files from `HWDashboard_version1/` EXCEPT `secrets.toml`
   - Upload: `Home.py`, `db.py`, `ui.py`, `question_engine.py`,
     `requirements.txt`, and the entire `pages/` folder
   - Do NOT upload `secrets.toml` — it contains your private credentials

### Step 3 — Deploy on Streamlit Community Cloud

1. Go to share.streamlit.io → Sign in with GitHub
2. Click **"Create app"**
3. Fill in:
   - **Repository:** select your repo
   - **Branch:** main
   - **Main file path:** `Home.py`
4. Click **"Advanced settings"** → **Secrets**
5. Open your `secrets.toml` file in Notepad
6. Select all text → Copy → Paste into the Secrets box
7. Click **Deploy**
8. Wait 2-3 minutes. Your app is live at a URL like:
   `https://your-app-name.streamlit.app`

### Step 4 — First Login as Instructor

1. Go to your app URL
2. Scroll to the very bottom → click **"Instructor access"** (small grey link)
3. Default password is: `Microeconomics`
4. **Change this immediately** in the Settings tab of the dashboard

### Step 5 — Get the Enrollment Key

1. Sign in as instructor
2. The enrollment key is displayed prominently on the Overview tab
3. Share this key with your students so they can create accounts

---

## Part 2 — Normal Semester Workflow

### Adding a New Homework Assignment

1. Sign in as instructor → Homework Manager tab
2. Scroll to **"Add New Assignment"**
3. Fill in:
   - **Homework ID:** use format `HW_WEEK3`, `HW_WEEK4` etc.
   - **Title:** what students will see
   - **Deadline date and time**
   - **Grace period:** extra minutes after deadline (default 15)
   - **Max marks:** for display on the dashboard
4. Click **Add Assignment**
5. **Important:** The assignment is only a display entry in the dashboard.
   To add actual questions, you also need to add a config block in `question_engine.py`
   (see "Adding Questions" section below).
6. Enable/disable the homework using the toggle in Homework Manager

### Adding Questions to a Homework (requires code edit)

Open `question_engine.py`. Near the bottom, find `ALL_HW_CONFIGS`.
Copy the `HW_WEEK2_CONFIG` block and modify it for your new homework.
Each question needs:
- `q_id`: unique ID (e.g. "Q1")
- `type`: "numerical" or "truefalse"
- `title`: display name
- `marks`: point value

For numerical questions, you also need to add:
- A `_params()` function that generates randomised numbers
- A `_render_qX()` function that draws the question UI
- Follow the exact pattern of `_render_q3()` or `_render_q9()`

For True/False questions, just update the `TF_STATEMENTS` list.

This is the main place you'll need to edit code each week.

### Enabling/Disabling a Homework

Instructor Dashboard → Homework Manager → find the homework → toggle "Enabled"
Takes effect immediately. Students will see locked homeworks as greyed out.

### Changing a Deadline

Instructor Dashboard → Homework Manager → find the homework → change date/time → "Update deadline"
No redeployment needed.

### Posting an Announcement

Instructor Dashboard → Homework Manager → find the homework → fill in "Announcement"
Shown to students on the Dashboard and inside the homework.

### Adding Students

**Option A — Self-registration (recommended):**
Share the enrollment key (visible on Overview tab) with students.
They register themselves at the app URL.

**Option B — Bulk import:**
Instructor Dashboard → Student Manager → paste email list (one per line) → Bulk Enroll
Students are registered with password `TempPass123` and forced to reset on first login.

### Resetting a Student's Password

Instructor Dashboard → Student Manager → select student → Set temporary password
Tell the student their temporary password. They can then change it themselves.

### Viewing Grades

Instructor Dashboard → Overview tab → Full Submission Log → Download CSV
The CSV contains: Timestamp, Email, Homework_ID, Question_ID, Score, Max_Score,
Raw answers, Is_Late flag, Param_Seed (for verification).

---

## Part 3 — Student Flow

1. Student goes to the app URL
2. First time: clicks "Create Account" → enters name, email, enrollment key, password
3. Sees a note to save their password
4. Signs in → sees the homework dashboard
5. Clicks on an open homework → sees questions in randomised order
6. Fills in answers → Submit button appears when all fields are filled
7. After submitting each question: sees score + full solution immediately
8. Can leave and return: previous answers and submission status are restored
9. Once all questions submitted: sees completion screen with final score

---

## Part 4 — End of Semester Reset

At the end of each semester, do the following in order:

### Step 1 — Download all data
Instructor Dashboard → Overview tab → Full Submission Log → Download CSV
Save this file. It is your permanent grade record.

### Step 2 — Download student list
Instructor Dashboard → Student Manager → Download student list CSV
Save this too.

### Step 3 — Clear the Registry (student accounts)
Instructor Dashboard → Settings tab → "End-of-Semester Reset" → Clear all student accounts
This removes all student logins so new students can register fresh next semester.
**Your grades data in the Submissions tab is NOT deleted.**

### Step 4 — Archive the Submissions sheet (optional but recommended)
In Google Sheets, right-click the "Submissions" tab → Duplicate → rename it
`Submissions_Semester1_2024` or similar. Then clear the original Submissions tab.

### Step 5 — Generate a new enrollment key
The key rotates automatically every 6 months.
If you want to force a new key now: in Google Sheets → Config tab →
find the row with `enrollment_key_created` → change the date to 2020-01-01 →
reload the app. A new key will be generated.

### Step 6 — Disable old homeworks and add new ones
Instructor Dashboard → Homework Manager → toggle off old homeworks →
add new ones for the upcoming semester.

### Do NOT:
- Delete the Google Sheet itself (you'll lose config)
- Delete the Config tab (contains homework settings and audit log)
- Change the SHEET_ID in secrets (it's already pointing to the right sheet)

---

## Part 5 — Troubleshooting

**"Could not connect to Google Sheet"**
→ The service account email needs Editor access on the Sheet.
   Go to your Google Sheet → Share → paste `microeconomics@iron-bedrock-490320-u6.iam.gserviceaccount.com` → Editor.
→ Check that Google Sheets API and Google Drive API are enabled in your Google Cloud project.

**"No questions found for this assignment"**
→ The homework ID in the dashboard config must exactly match the key in
   `ALL_HW_CONFIGS` in `question_engine.py`. Check for typos (case-sensitive).

**Student can't register — enrollment key rejected**
→ Check the key on the Instructor Dashboard (Overview tab).
   Keys are case-insensitive but must match exactly.

**Tabs not showing in the sidebar**
→ Make sure all pages are in the `pages/` folder with the correct filenames.
   Streamlit requires the `pages/` folder to be at the same level as `Home.py`.

**Answers not saving to the sheet**
→ Check the Streamlit Logs (on share.streamlit.io, click your app → "Manage app" → logs).
   Look for any authentication or permission errors.

**Score shows 0 even though answer looks correct**
→ This system uses exact matching (zero tolerance). The student's answer must
   match exactly. For decimal answers like Jerry's bundle, they must enter
   the exact decimal (e.g. 6.6667 not 6.67). Consider communicating this
   to students in your instructions.

**App is slow**
→ Streamlit Community Cloud free tier can be slow, especially when reading
   from Google Sheets. Consider upgrading to Streamlit Teams if speed is important.
   The `@st.cache_resource` on the connection helps but Sheet reads are still network calls.

---

## Part 6 — Security Notes

- Passwords are stored as SHA-256 hashes — never in plain text
- The instructor dashboard is hidden from students (not in the sidebar navigation)
- Sessions expire after the browser tab is closed (Streamlit default)
- The enrollment key prevents random people from creating accounts
- The Google Sheet is accessible only via the service account credentials

---

## Version History

- **v1.0** — Initial release. Login system, Week 2 homework (Q3, Q9, Q10 T/F),
  instructor dashboard with full management controls, plagiarism detection,
  deadline system with grace period, worked examples (unlocks after 20 min),
  dual timers, completion screen.
