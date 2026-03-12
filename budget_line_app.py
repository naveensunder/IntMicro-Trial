import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

st.set_page_config(page_title="Budget Line Illustration", layout="centered")

st.title("Budget Line Illustration")
st.markdown("Adjust the sliders below and the chart updates instantly.")

# ── Sidebar controls ──────────────────────────────────────────────────────────
Px = st.slider("Price of Good X (Pₓ)", min_value=1, max_value=20, value=5, step=1)
Py = st.slider("Price of Good Y (Pᵧ)", min_value=1, max_value=20, value=4, step=1)
I  = st.slider("Income (I)", min_value=50, max_value=500, value=100, step=10)

# ── Plot ──────────────────────────────────────────────────────────────────────
x_int = I / Px
y_int = I / Py
x_max = 1.25 * x_int
y_max = 1.25 * y_int

x_vals = np.linspace(0, x_int, 400)
y_vals = (I - Px * x_vals) / Py

fig, ax = plt.subplots(figsize=(8, 6))

ax.plot(x_vals, y_vals, linewidth=2.5, color="steelblue", label="Budget line")
ax.fill_between(x_vals, 0, y_vals, alpha=0.25, color="steelblue",
                label="Affordable bundles")

ax.scatter([x_int], [0], color="red", zorder=5)
ax.scatter([0], [y_int], color="red", zorder=5)

ax.annotate(
    f"All income on X\n(x = I/Pₓ = {x_int:.1f})",
    (x_int, 0),
    xytext=(0.85 * x_int, 0.18 * y_int),
    arrowprops=dict(arrowstyle="->"),
    fontsize=10,
)
ax.annotate(
    f"All income on Y\n(y = I/Pᵧ = {y_int:.1f})",
    (0, y_int),
    xytext=(0.15 * x_int, 0.88 * y_int),
    arrowprops=dict(arrowstyle="->"),
    fontsize=10,
)

ax.set_xlabel("Quantity of Good X", fontsize=12)
ax.set_ylabel("Quantity of Good Y", fontsize=12)
ax.set_title(
    f"Budget Line and Affordable Set\n"
    f"Income = {I},  Pₓ = {Px},  Pᵧ = {Py}",
    fontsize=13,
)
ax.set_xlim(0, x_max)
ax.set_ylim(0, y_max)
ax.grid(alpha=0.3)
ax.legend(loc="upper right", fontsize=10)

st.pyplot(fig)

# ── Summary ───────────────────────────────────────────────────────────────────
st.markdown("---")
col1, col2, col3 = st.columns(3)
col1.metric("X-intercept (I/Pₓ)", f"{x_int:.2f}")
col2.metric("Y-intercept (I/Pᵧ)", f"{y_int:.2f}")
col3.metric("Slope (−Pₓ/Pᵧ)", f"{-Px/Py:.2f}")
