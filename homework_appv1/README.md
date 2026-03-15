# Week 2 Homework App — Deployment Guide

## File Structure
```
homework_app/
├── Home.py                        ← Student login page
├── utils.py                       ← Shared code (grading, sheets, params)
├── requirements.txt
├── secrets_TEMPLATE.toml          ← Copy & fill in → .streamlit/secrets.toml
└── pages/
    ├── 1_Q3_Budget_Constraint.py
    ├── 2_Q9_Tom_and_Jerry.py
    └── 3_Instructor_Dashboard.py
```

---

## Step 1 — Create your Google Sheet

1. Go to drive.google.com → New → Google Sheets
2. Name it e.g. `Week2_Graded_Responses`
3. Copy the Sheet ID from the URL:
   `https://docs.google.com/spreadsheets/d/<<SHEET_ID>>/edit`
4. Save this ID — you'll need it in Step 3

---

## Step 2 — Set up a Google Service Account (one-time, ~10 min)

1. Go to console.cloud.google.com
2. Create a new project (or use an existing one)
3. Enable two APIs:
   - "Google Sheets API"
   - "Google Drive API"
4. Go to IAM & Admin → Service Accounts → Create Service Account
5. Give it any name (e.g. `homework-app`)
6. Click the account → Keys tab → Add Key → JSON
7. Download the JSON file — this is your credentials file
8. Open your Google Sheet → Share → paste the service account email
   (looks like `something@yourproject.iam.gserviceaccount.com`) → Editor role

---

## Step 3 — Configure secrets

1. Create a folder called `.streamlit` inside `homework_app/`
2. Copy `secrets_TEMPLATE.toml` → `.streamlit/secrets.toml`
3. Fill in:
   - `SHEET_ID` — from Step 1
   - `DEADLINE` — e.g. `"2025-04-30 23:59"`
   - `INSTRUCTOR_PASSWORD` — something only you know
   - All fields under `[gcp_service_account]` — copy from the JSON file in Step 2

---

## Step 4 — Deploy on Streamlit Community Cloud

1. Create a free account at github.com and create a new repository
2. Upload all files in `homework_app/` to the repository
   **IMPORTANT**: Do NOT upload `.streamlit/secrets.toml` to GitHub
   Add `.streamlit/secrets.toml` to a `.gitignore` file
3. Go to share.streamlit.io → Sign in with GitHub → New app
4. Select your repository, branch (main), and set Main file path to `Home.py`
5. Click Advanced → Secrets → paste the entire contents of your `secrets.toml`
6. Click Deploy
7. Your app will have a URL like `https://yourapp.streamlit.app`
   Share this URL with students

---

## Changing the deadline

- On Streamlit Cloud: App settings → Secrets → change `DEADLINE` value → Save
- The app updates immediately — no redeployment needed

---

## What students see

1. Go to the URL → enter name + school email → click Confirm
2. Navigate to Q3 and Q9 using the left sidebar
3. Each question shows their unique parameters (different per student)
4. Submit button appears only when all fields are filled
5. After submit: score + worked solution shown, answers locked
6. If they close and return: answers are pre-filled from the Sheet

## What you see (Instructor Dashboard)

- Go to the app URL → click "Instructor Dashboard" in sidebar
- Enter your instructor password
- See: submission counts, score distributions, copying flags, late submissions
- Download full CSV of all responses

---

## Troubleshooting

**"Could not connect to Google Sheet"**
→ Check that the service account email has Editor access on the Sheet
→ Check that Google Sheets API and Drive API are enabled in your project

**Students see wrong answers / different questions**
→ Each student's parameters are seeded by their email — this is intentional

**Deadline not updating**
→ Change the DEADLINE value in Streamlit Cloud secrets panel
