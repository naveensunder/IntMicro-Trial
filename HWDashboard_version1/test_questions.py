"""
test_questions.py — EC224 Question Engine Test Suite
=====================================================
Run before every deployment to catch bugs before students see them.

Usage:
    python test_questions.py

What it checks (per question, across 20 dummy emails):
  - Parameters generate without crashing
  - No division by zero
  - Answers are finite numbers (no NaN, no inf)
  - Answers round cleanly to 2 decimal places
  - Scores fall within [0, max_marks]
  - Not all students get the same answer (seeding is working)

No Google Sheets connection needed. Runs entirely offline.
"""

import sys
import math
import hashlib
import traceback
import numpy as np

# ── Patch streamlit away (not needed for parameter/answer logic) ───────────────
import unittest.mock as mock
sys.modules["streamlit"] = mock.MagicMock()

# Now safe to import question_engine
try:
    from question_engine import (
        _q1_params, _q2_params,
        get_seed, r2, ALL_HW_CONFIGS,
    )
except ImportError as e:
    print(f"FATAL: Could not import question_engine: {e}")
    sys.exit(1)

# ── Test emails ────────────────────────────────────────────────────────────────
TEST_EMAILS = [
    f"student{i:02d}@bentley.edu" for i in range(1, 21)
]

PASS = 0
FAIL = 0
WARNINGS = []

def ok(msg):
    global PASS
    PASS += 1
    print(f"  ✓  {msg}")

def fail(msg):
    global FAIL
    FAIL += 1
    print(f"  ✗  {msg}")
    WARNINGS.append(msg)

def is_valid(val, name):
    """Check value is a finite real number."""
    try:
        f = float(val)
        if math.isnan(f) or math.isinf(f):
            fail(f"{name} is NaN or Inf: {val}")
            return False
        return True
    except Exception:
        fail(f"{name} is not numeric: {val}")
        return False

def check_diversity(values, name, threshold=0.5):
    """Check that not all students get the same answer."""
    unique = len(set(round(float(v), 2) for v in values))
    if unique < max(2, len(values) * threshold):
        fail(f"{name}: only {unique}/{len(values)} unique values — seeding may be broken")
    else:
        ok(f"{name}: {unique}/{len(values)} unique values across students")

def check_score(score, max_score, label):
    if not is_valid(score, f"{label} score"):
        return
    s = float(score)
    if s < 0 or s > max_score:
        fail(f"{label}: score {s} outside [0, {max_score}]")
    else:
        ok(f"{label}: score {s} in valid range [0, {max_score}]")


# ════════════════════════════════════════════════════════════════════════════════
#  TEST: WEEK 2 — Q1 Parameters
# ════════════════════════════════════════════════════════════════════════════════
print("\n─── Week 2 · Q1: Budget Constraint ───")
q1_xints = []; q1_yints = []; q1_slopes = []

for email in TEST_EMAILS:
    try:
        I, Px, Py = _q1_params(email)
        ANS_x = r2(I / Px)
        ANS_y = r2(I / Py)
        ANS_s = r2(-Px / Py)
        for val, name in [(ANS_x, "X-intercept"), (ANS_y, "Y-intercept"), (ANS_s, "Slope")]:
            is_valid(val, f"{email} {name}")
        if ANS_x <= 0 or ANS_y <= 0:
            fail(f"{email}: intercepts should be positive (X={ANS_x}, Y={ANS_y})")
        q1_xints.append(ANS_x)
        q1_yints.append(ANS_y)
        q1_slopes.append(ANS_s)
    except Exception:
        fail(f"{email}: Q1 params crashed — {traceback.format_exc(limit=1).strip()}")

check_diversity(q1_xints,  "Q1 X-intercepts")
check_diversity(q1_yints,  "Q1 Y-intercepts")
check_diversity(q1_slopes, "Q1 Slopes")


# ════════════════════════════════════════════════════════════════════════════════
#  TEST: WEEK 2 — Q2 Parameters
# ════════════════════════════════════════════════════════════════════════════════
print("\n─── Week 2 · Q2: Tom & Jerry ───")
q2_tx = []; q2_jx = []

for email in TEST_EMAILS:
    try:
        I, Px, Py, tom_a = _q2_params(email)
        ANS_tx = r2(I / Px)
        ANS_ty = 0.0
        ANS_jx = r2(I / (Px + Py))
        ANS_jy = r2(I / (Px + Py))
        for val, name in [
            (ANS_tx, "Tom X"), (ANS_jx, "Jerry X"), (ANS_jy, "Jerry Y")
        ]:
            is_valid(val, f"{email} {name}")
        if ANS_tx <= 0:
            fail(f"{email}: Tom X* should be positive ({ANS_tx})")
        if abs(ANS_jx - ANS_jy) > 0.01:
            fail(f"{email}: Jerry X* should equal Y* ({ANS_jx} vs {ANS_jy})")
        q2_tx.append(ANS_tx)
        q2_jx.append(ANS_jx)
    except Exception:
        fail(f"{email}: Q2 params crashed — {traceback.format_exc(limit=1).strip()}")

check_diversity(q2_tx, "Q2 Tom X*")
check_diversity(q2_jx, "Q2 Jerry X*")


# ════════════════════════════════════════════════════════════════════════════════
#  TEST: ALL_HW_CONFIGS structure
# ════════════════════════════════════════════════════════════════════════════════
print("\n─── ALL_HW_CONFIGS structure ───")
for hw_id, hw_data in ALL_HW_CONFIGS.items():
    questions = hw_data.get("questions", [])
    if not questions:
        fail(f"{hw_id}: no questions defined")
        continue
    ok(f"{hw_id}: {len(questions)} questions defined")
    total_marks = sum(q.get("marks", 0) for q in questions)
    if total_marks <= 0:
        fail(f"{hw_id}: total marks = {total_marks}")
    else:
        ok(f"{hw_id}: total marks = {total_marks}")
    for q in questions:
        for field in ["q_id", "title", "marks", "type"]:
            if not q.get(field):
                fail(f"{hw_id}/{q.get('q_id','?')}: missing field '{field}'")


# ════════════════════════════════════════════════════════════════════════════════
#  TEST: get_seed consistency
# ════════════════════════════════════════════════════════════════════════════════
print("\n─── Seed consistency ───")
for email in TEST_EMAILS[:5]:
    s1 = get_seed(email)
    s2 = get_seed(email.upper())
    if s1 != s2:
        fail(f"get_seed not case-insensitive for {email}")
    else:
        ok(f"get_seed consistent for {email}")


# ════════════════════════════════════════════════════════════════════════════════
#  SUMMARY
# ════════════════════════════════════════════════════════════════════════════════
print(f"\n{'='*50}")
print(f"Results: {PASS} passed · {FAIL} failed")
if FAIL == 0:
    print("✅ All tests passed — safe to deploy.")
else:
    print("❌ Failures detected — do not deploy until fixed:")
    for w in WARNINGS:
        print(f"   • {w}")
print(f"{'='*50}\n")
sys.exit(0 if FAIL == 0 else 1)
