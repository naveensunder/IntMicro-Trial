import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(page_title="Budget Line", layout="wide")

st.title("Budget Line Illustration")
st.caption("Adjust prices and income to see how the budget constraint changes.")

col_ctrl, col_chart = st.columns([1, 2])

with col_ctrl:
    Px = st.slider("Price of X  (Pₓ)", 1, 20, 5)
    Py = st.slider("Price of Y  (Pᵧ)", 1, 20, 4)
    I  = st.slider("Income  (I)", 50, 500, 100, step=10)
    st.markdown("---")
    x_int = I / Px
    y_int = I / Py
    st.metric("X-intercept  I/Pₓ", f"{x_int:.2f}")
    st.metric("Y-intercept  I/Pᵧ", f"{y_int:.2f}")
    st.metric("Slope  −Pₓ/Pᵧ", f"{-Px/Py:.3f}")

x_vals = np.linspace(0, x_int, 400)
y_vals = (I - Px * x_vals) / Py

x_max = 1.25 * x_int
y_max = 1.25 * y_int

fig, ax = plt.subplots(figsize=(8, 6))
ax.plot(x_vals, y_vals, color="#3d5a80", lw=2.8, label="Budget line")
ax.fill_between(x_vals, 0, y_vals, alpha=0.15, color="#3d5a80", label="Affordable bundles")
ax.scatter([x_int], [0], color="#c0392b", s=80, zorder=5)
ax.scatter([0], [y_int], color="#c0392b", s=80, zorder=5)
ax.annotate(
    f"All income on X\n(x = I/Pₓ = {x_int:.1f})",
    (x_int, 0), xytext=(0.78*x_int, 0.18*y_int),
    arrowprops=dict(arrowstyle="->", color="#555"),
    fontsize=10, ha="center",
    bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="#cccccc"),
)
ax.annotate(
    f"All income on Y\n(y = I/Pᵧ = {y_int:.1f})",
    (0, y_int), xytext=(0.22*x_int, 0.88*y_int),
    arrowprops=dict(arrowstyle="->", color="#555"),
    fontsize=10, ha="center",
    bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="#cccccc"),
)
ax.set_xlim(0, x_max); ax.set_ylim(0, y_max)
ax.set_xlabel("Quantity of Good X", fontsize=12)
ax.set_ylabel("Quantity of Good Y", fontsize=12)
ax.set_title(f"Budget Line  —  I = {I},  Pₓ = {Px},  Pᵧ = {Py}", fontsize=13)
ax.grid(alpha=0.3); ax.legend(loc="upper right", frameon=True)
ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)

with col_chart:
    st.pyplot(fig, use_container_width=True)
plt.close(fig)
