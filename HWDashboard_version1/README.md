# HWDashboard v2 — Complete Guide
## Learning Intermediate Microeconomics — Prof. Naveen Sunder

---

## File Structure

```
HWDashboard_v2/
├── Home.py                  ← Login page (entry point)
├── db.py                    ← All Google Sheets operations
├── ui.py                    ← Shared CSS and design system
├── question_engine.py       ← All question logic and rendering
├── requirements.txt
├── secrets.toml             ← Your credentials (DO NOT upload to GitHub)
└── pages/
    ├── Dashboard.py         ← Student landing page
    ├── Homework.py          ← Homework question renderer
    ├── FAQ.py               ← Student FAQ page
    ├── Course_Materials.py  ← Course materials with links
    └── Instructor.py        ← Instructor dashboard (password protected)
```

---

## What's New in v2 (Phase 1)

**Bug fixes:**
- Raw HTML no longer leaks onto the Dashboard card
- True/False question TypeError is fixed — the app no longer crashes on Week 2 homework

**Student-facing improvements:**
- Semester score summary at top of Dashboard (total earned / total possible)
- Per-homework score shown on each homework card with countdown timer
- Student name and last login shown prominently
- Question summary table at top of each homework with clickable links
- Full read-only review mode after all questions submitted (useful for exam prep)
- Print to PDF button in review mode
- Topics to revise shown after each solution based on wrong answers
- Mobile warning shown on small screens
- Unsaved answers warning when navigating away
- Extension request form on homework page after deadline
- FAQ page in sidebar
- Course Materials page in sidebar with Dropbox/Drive links

**Instructor improvements:**
- Homework instructions field (shown to students before starting)
- Auto-enable date per homework (no manual toggling needed)
- Individual access grants per student with custom deadline
- Extension request approval workflow
- Plagiarism report download button
- Copy enrollment key button
- Gradebook export (one row per student, total scores)
- Question analytics (success rates per question)
- Preview as student mode

---

## Part 1 — First-Time Setup

### Step 1 — Critical: Delete the old Config tab

If you are upgrading from v1, the Config tab structure has changed.

1. Go to your Google Sheet
2. Right-click the **Config** tab → Delete
3. The new app will recreate it correctly on first load

If this is a brand new deployment, skip this step.

### Step 2 — GitHub

1. Go to github.com → New repository (Private recommended)
2. Upload ALL files EXCEPT `secrets.toml`:
   - `Home.py`, `db.py`, `ui.py`, `question_engine.py`, `requirements.txt`
   - Entire `pages/` folder with all 5 files inside

### Step 3 — Streamlit Deploy

1. share.streamlit.io → Create app → select your repo
2. Main file path: `Home.py`
3. Advanced settings → Secrets → paste entire contents of `secrets.toml`
4. Deploy

### Step 4 — First Login

1. Go to your app URL
2. Scroll to bottom → click **"Instructor access"**
3. Default password: `Microeconomics`
4. **Change it immediately** in Settings tab

### Step 5 — Share Enrollment Key

Instructor Dashboard → Overview tab → copy the enrollment key → share with students.

---

## Part 2 — Adding Course Materials

Course materials are managed in `pages/Course_Materials.py`. No database needed — links are stored directly in the file.

### How to add a material

Open `Course_Materials.py` and find the `MATERIALS` dictionary. Each section (Lecture Slides, Readings, etc.) contains a list of items. Add a new item like this:

```python
{
    "title":       "Week 4 — Demand Theory",
    "description": "Slides covering income and substitution effects.",
    "url":         "https://www.dropbox.com/your-link-here",
    "updated":     "01 Feb 2026",
    "file_type":   "PPTX",
},
```

Then push to GitHub. Streamlit auto-updates within a minute.

### How to get a Dropbox link

1. Upload file to Dropbox
2. Right-click → Share → Copy link
3. Change `dl=0` at the end to `dl=1` (forces download) or leave as `dl=0` (opens in browser)
4. Paste into the `url` field

### How to get a Google Drive link

1. Upload to Google Drive
2. Right-click → Share → "Anyone with the link" → Copy link
3. Paste into the `url` field

### How to update a file

**Option A (recommended):** Replace the file in Dropbox/Drive with the same filename. The link stays the same. Just update the `"updated"` date in the code and push to GitHub.

**Option B:** Upload a new file, get a new link, replace the `"url"` value and push.

### How to add a new section

In `Course_Materials.py`, add a new key to the `MATERIALS` dictionary:

```python
"Exam Prep": [
    {
        "title": "Midterm Practice Problems",
        ...
    }
],
```

---

## Part 3 — Normal Semester Workflow

### Enable/disable a homework
Instructor Dashboard → Homework Manager → toggle "Enabled" on/off.

### Schedule a homework to auto-enable
Homework Manager → open the homework → set Auto-enable date and time.
The homework will enable itself automatically at that moment — no action needed from you.

### Set an announcement
Homework Manager → open the homework → fill in "Student announcement" → Save.
Shown on the Dashboard card and inside the homework page.

### Add instructions to a homework
Homework Manager → open the homework → fill in "Assignment instructions" → Save.
Shown to students at the top of the homework page before they start.

### Change a deadline
Homework Manager → open the homework → change date/time → "Update deadline".

### Grant an individual extension
Two ways:
1. **Student requests it:** After the deadline, students see a "Request an extension" form on the homework page. The request appears in the Extension Requests tab of your dashboard. You approve it with a custom deadline.
2. **You grant it directly:** Homework Manager → open the homework → "Grant individual access" section → enter the student email and new deadline.

### Add students
- **Self-registration:** Share the enrollment key. Students register themselves.
- **Bulk import:** Student Manager tab → paste email list → Bulk Enroll. Students get `TempPass123` and are forced to reset on first login.

### View grades
Overview tab → Full Submission Log → Download CSV (all raw data)
Overview tab → Download Gradebook CSV (one row per student, total scores ready to submit)

### Preview as a student
Student Manager tab → Preview as Student → select email → Enter preview mode.
You see the app exactly as that student sees it. Sign out to return to normal.

---

## Part 4 — Adding New Homework Questions (requires code edit)

Open `question_engine.py`. Find `ALL_HW_CONFIGS` near the top. Add a new config:

```python
HW_WEEK3_CONFIG = {
    "hw_id": "HW_WEEK3",
    "questions": [
        {"q_id": "Q1", "type": "numerical",  "title": "Q1 — Demand Curve", "marks": 6},
        {"q_id": "QTF","type": "truefalse",  "title": "Q2 — True or False", "marks": 4},
    ]
}

ALL_HW_CONFIGS = {
    "HW_WEEK2": HW_WEEK2_CONFIG,
    "HW_WEEK3": HW_WEEK3_CONFIG,  # ← add here
}
```

For each new numerical question you also need:
- A `_params_qX(email)` function generating randomised numbers
- A `_render_qX(...)` function building the UI (copy from `_render_q3` as a template)
- A `_show_qX_solution(...)` function showing the worked solution

For True/False: just update the `TF_STATEMENTS` list at the top of `question_engine.py`.

After editing, push to GitHub and Streamlit will update automatically.

---

## Part 5 — End-of-Semester Reset

Do these steps in order. Do not skip Step 1.

### Step 1 — Download everything
- Overview tab → Download CSV (full submission log — keep this permanently)
- Overview tab → Download Gradebook CSV (for grade submission)
- Student Manager → Download student list CSV

### Step 2 — Archive the Submissions sheet (recommended)
In Google Sheets, right-click the Submissions tab → Duplicate → rename it
`Submissions_Fall2026` or similar. Then clear the original Submissions tab
(keep the header row).

### Step 3 — Clear student accounts
Instructor Dashboard → Settings → End-of-Semester Reset → Clear all student accounts.
This removes all logins so new students can register fresh next semester.
Your grade data in the Submissions sheet is not affected.

### Step 4 — Rotate the enrollment key (optional)
The key rotates automatically every 6 months. To force a new key now:
In Google Sheets → Config tab → find the `enrollment_key_created` row →
change the date to `2020-01-01` → reload the app. A new key is generated.

### Step 5 — Disable old homeworks
Homework Manager → toggle off all old homeworks. Add new ones for the new semester.

### Do NOT
- Delete the Google Sheet itself
- Delete the Config tab
- Change the `SHEET_ID` in secrets

---

## Part 6 — Troubleshooting

**"Instructor access" with password `Microeconomics` is rejected**
→ The Config tab may exist but be empty (a known initialisation issue).
   Delete the Config tab entirely → reload the app → it will recreate with defaults.

**No questions showing on homework page / TypeError**
→ This was a known bug in v1, fixed in v2. If it recurs, check that the `HW_ID`
   in the Homework Manager exactly matches the key in `ALL_HW_CONFIGS`
   in `question_engine.py` (case-sensitive).

**Raw HTML visible on dashboard**
→ Fixed in v2. If it reappears in a future edit, check that no HTML is built
   using f-strings where the variable contains unescaped angle brackets.

**Submission not saving**
→ Check Streamlit logs (Manage app → Logs). Look for Sheets API errors.
   Ensure the service account still has Editor access to the Sheet.

**App is slow**
→ All data reads go to Google Sheets over the network. The `@st.cache_resource`
   on the connection helps but cannot eliminate latency entirely. If speed
   becomes a problem for 100+ students, consider upgrading to Streamlit Teams.

**Student can't register — enrollment key rejected**
→ Keys are case-insensitive. Check the key shown on the Instructor Overview tab.
   If the Config tab was recently recreated, the key will have changed.

---

## Part 7 — Security Notes

- Passwords are SHA-256 hashed — never stored in plain text
- The instructor dashboard link is not shown in the sidebar navigation
- 5 failed login attempts triggers a 5-minute lockout
- The enrollment key prevents unauthorised account creation
- `secrets.toml` must never be uploaded to GitHub

---

## Version History

- **v1.0** — Initial release
- **v2.0 Phase 1** — Bug fixes, Dashboard improvements, FAQ, Course Materials,
  question summary table, review mode, print to PDF, topics to revise,
  extension workflow, individual access, gradebook export, preview mode,
  auto-enable dates, plagiarism report download
