import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(page_title="Comparing Budget Lines", layout="wide")

st.title("Comparing Budget Lines")
st.caption("Compare two budget constraints side-by-side to explore the effect of price or income changes.")

def _bline(I, Px, Py, n=400):
    x_int = I / Px
    xs = np.linspace(0, x_int, n)
    ys = (I - Px * xs) / Py
    return xs, ys, x_int, I / Py

col_l, col_r = st.columns(2)
with col_l:
    st.subheader("Budget Line 1")
    Px1 = st.slider("Pₓ₁", 1, 20, 5,  key="Px1")
    Py1 = st.slider("Pᵧ₁", 1, 20, 10, key="Py1")
    I1  = st.slider("I₁",  50, 500, 100, step=10, key="I1")

with col_r:
    st.subheader("Budget Line 2")
    show2 = st.checkbox("Show second budget line", value=True)
    Px2 = st.slider("Pₓ₂", 1, 20, 2,   key="Px2", disabled=not show2)
    Py2 = st.slider("Pᵧ₂", 1, 20, 5,   key="Py2", disabled=not show2)
    I2  = st.slider("I₂",  50, 500, 150, step=10, key="I2", disabled=not show2)

xs1, ys1, xint1, yint1 = _bline(I1, Px1, Py1)
xint2 = yint2 = 0
if show2:
    xs2, ys2, xint2, yint2 = _bline(I2, Px2, Py2)

x_max = 1.3 * max(xint1, xint2, 1)
y_max = 1.3 * max(yint1, yint2, 1)

fig, ax = plt.subplots(figsize=(10, 6.5))
ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)

# Line 1
ax.plot(xs1, ys1, color="darkred", lw=3, label="Budget line 1")
ax.fill_between(xs1, 0, ys1, color="lightcoral", alpha=0.25)
ax.scatter([xint1, 0], [0, yint1], color="darkred", s=60, zorder=5)
ax.text(0.46, 0.78,
        f"Line 1\nI={I1}, Pₓ={Px1}, Pᵧ={Py1}\nx-int = {xint1:.2f}\ny-int = {yint1:.2f}",
        transform=ax.transAxes, va="top", fontsize=11,
        bbox=dict(boxstyle="round", facecolor="white", edgecolor="darkred", alpha=0.95))

# Line 2
if show2:
    ax.plot(xs2, ys2, color="navy", lw=3, label="Budget line 2")
    ax.fill_between(xs2, 0, ys2, color="lightskyblue", alpha=0.22)
    ax.scatter([xint2, 0], [0, yint2], color="navy", s=60, zorder=5)
    ax.text(0.46, 0.57,
            f"Line 2\nI={I2}, Pₓ={Px2}, Pᵧ={Py2}\nx-int = {xint2:.2f}\ny-int = {yint2:.2f}",
            transform=ax.transAxes, va="top", fontsize=11,
            bbox=dict(boxstyle="round", facecolor="white", edgecolor="navy", alpha=0.95))

ax.text(0.02, 0.87, "Shaded regions = affordable bundles (spending ≤ income)",
        transform=ax.transAxes, fontsize=10, color="dimgray")

ax.set_xlim(0, x_max); ax.set_ylim(0, y_max)
ax.set_xlabel("Quantity of Good X", fontsize=12)
ax.set_ylabel("Quantity of Good Y", fontsize=12)
ax.set_title("Budget Constraints and Affordable Bundles", fontsize=14)
ax.grid(alpha=0.28); ax.legend(loc="upper right", fontsize=10)

st.pyplot(fig, use_container_width=True)
plt.close(fig)

# Summary table
st.markdown("---")
rows = [{"": "Budget Line 1", "Income (I)": I1, "Pₓ": Px1, "Pᵧ": Py1,
         "X-intercept": f"{xint1:.2f}", "Y-intercept": f"{yint1:.2f}", "Slope": f"{-Px1/Py1:.3f}"}]
if show2:
    rows.append({"": "Budget Line 2", "Income (I)": I2, "Pₓ": Px2, "Pᵧ": Py2,
                 "X-intercept": f"{xint2:.2f}", "Y-intercept": f"{yint2:.2f}", "Slope": f"{-Px2/Py2:.3f}"})
st.table(rows)
