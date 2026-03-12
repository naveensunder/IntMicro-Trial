import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(page_title="Optimal Bundle", layout="wide")

UTILITY_OPTIONS = [
    "Cobb–Douglas",
    "Quasi-linear  (ln(x)+y)",
    "Perfect complements 1:1",
    "Perfect complements 2:1",
    "Perfect complements 3:1",
    "Perfect substitutes 1:1",
    "Perfect substitutes 2:1",
    "Perfect substitutes 3:1",
]
ALPHA_OPTIONS = [0.2, 0.4, 0.5, 0.6, 0.8, 0.9]

def utility_params(u_label, alpha):
    if "Cobb" in u_label:      return {"type":"CD","alpha":float(alpha)}
    if "Quasi" in u_label:     return {"type":"QL"}
    if "complements" in u_label:
        r = int(u_label.split()[-1].split(":")[0]); return {"type":"PC","r":r}
    if "substitutes" in u_label:
        a = int(u_label.split()[-1].split(":")[0]); return {"type":"PS","a":a,"b":1}

def U_val(par, x, y):
    t=par["type"]
    if t=="CD": a=par["alpha"]; return (x**a)*(y**(1-a))
    if t=="QL": return np.log(max(x,1e-9))+y
    if t=="PS": return par["a"]*x+par["b"]*y
    if t=="PC": return min(x/par["r"],y)

def closed_form(par, I, Px, Py):
    t=par["type"]
    if t=="CD":
        a=par["alpha"]; return a*I/Px, (1-a)*I/Py
    if t=="QL":
        xs=Py/Px; ys=(I-Px*xs)/Py
        if ys<0: return I/Px, 0.0
        return xs, ys
    if t=="PS":
        a,b=par["a"],par["b"]; bx,by=a/Px,b/Py
        if abs(bx-by)<1e-12: return (I/2)/Px,(I/2)/Py
        return (I/Px,0.0) if bx>by else (0.0,I/Py)
    if t=="PC":
        r=par["r"]; t_=I/(Px*r+Py); return r*t_,t_

def ic_pts(par, u_level, x_max, y_max, n=900):
    t=par["type"]; xs=np.linspace(1e-3,x_max,n)
    if t=="CD":
        a=par["alpha"]; ys=(u_level/(xs**a))**(1/(1-a))
        return [(xs, np.where(np.isfinite(ys)&(ys>=0)&(ys<=y_max),ys,np.nan))]
    if t=="QL":
        ys=u_level-np.log(xs)
        return [(xs, np.where(np.isfinite(ys)&(ys>=0)&(ys<=y_max),ys,np.nan))]
    if t=="PS":
        a,b=par["a"],par["b"]; ys=(u_level-a*xs)/b
        return [(xs, np.where(np.isfinite(ys)&(ys>=0)&(ys<=y_max),ys,np.nan))]
    if t=="PC":
        r=par["r"]; xk,yk=r*u_level,u_level
        xs1=np.linspace(xk,x_max,70); ys1=np.full_like(xs1,yk)
        ys2=np.linspace(yk,y_max,70); xs2=np.full_like(ys2,xk)
        return [(xs1,np.where(ys1<=y_max,ys1,np.nan)),(xs2,np.where(xs2<=x_max,xs2,np.nan))]
    return []

def slope_ic(par, x_star, y_star):
    t=par["type"]
    if t=="CD": a=par["alpha"]; return -(a/(1-a))*(y_star/x_star)
    if t=="QL": return -1.0/x_star
    if t=="PS": return -par["a"]/par["b"]
    return None

def solution_text(par, I, Px, Py, xs, ys):
    t=par["type"]
    s=-Px/Py
    lines=[]
    if t=="CD":
        a=par["alpha"]; b=1-a
        mrs=(a/b)*(ys/xs)
        lines=[
            f"**Step 1 — Budget constraint:** {I} = {Px}·x + {Py}·y  →  slope = −{Px}/{Py} = {s:.2f}",
            f"**Step 2 — Tangency condition:** MRS = (α/β)·(y/x) = Pₓ/Pᵧ",
            f"**Step 3 — Demand shares:** x* = α·I/Pₓ = {a:.1f}×{I}/{Px} = **{xs:.2f}**",
            f"y* = β·I/Pᵧ = {b:.1f}×{I}/{Py} = **{ys:.2f}**",
            f"**Tangency check:** MRS = {mrs:.3f},  Pₓ/Pᵧ = {Px/Py:.3f}",
        ]
    elif t=="QL":
        lines=[
            f"**Step 1 — Budget:** {I} = {Px}·x + {Py}·y",
            f"**Step 2 — Interior tangency:** MUₓ/Pₓ = MUᵧ/Pᵧ  →  (1/x)/Pₓ = 1/Pᵧ  →  x* = Pᵧ/Pₓ = {Py}/{Px} = **{xs:.2f}**",
            f"**Step 3:** y* = (I − Pₓ·x*)/Pᵧ = **{ys:.2f}**",
        ]
    elif t=="PS":
        a,b=par["a"],par["b"]; bx,by=a/Px,b/Py
        lines=[
            f"**Step 1 — Budget:** {I} = {Px}·x + {Py}·y",
            f"**Step 2 — Bang-for-buck:** MUₓ/Pₓ = {a}/{Px} = {bx:.3f},   MUᵧ/Pᵧ = {b}/{Py} = {by:.3f}",
            f"**Step 3 — Corner solution:** Spend all income on {'X' if bx>by else 'Y'}",
            f"x* = **{xs:.2f}**,  y* = **{ys:.2f}**",
        ]
    elif t=="PC":
        r=par["r"]
        lines=[
            f"**Step 1 — Budget:** {I} = {Px}·x + {Py}·y",
            f"**Step 2 — Kink condition:** x = {r}·y",
            f"**Step 3 — Substitute:** {Px}·({r}y) + {Py}·y = {I}  →  y* = I/(Pₓ·{r}+Pᵧ) = **{ys:.2f}**",
            f"x* = {r}·y* = **{xs:.2f}**",
        ]
    return lines

# ── UI ─────────────────────────────────────────────────────────────────────────
st.title("Finding the Optimal Consumption Bundle")

c_ctrl, c_chart = st.columns([1, 2])

with c_ctrl:
    u_sel   = st.selectbox("Utility function", UTILITY_OPTIONS)
    alpha   = st.selectbox("α (Cobb-Douglas only)", ALPHA_OPTIONS, index=2,
                           disabled="Cobb" not in u_sel)
    Px      = st.number_input("Price Pₓ  (1–20)", 1, 20, 5, step=1)
    Py      = st.number_input("Price Pᵧ  (1–20)", 1, 20, 4, step=1)
    I       = st.number_input("Income I  (50–500)", 50, 500, 100, step=10)
    compute = st.button("Compute Optimal Bundle", type="primary")

if compute or True:   # always show on load
    par = utility_params(u_sel, alpha)
    x_star, y_star = closed_form(par, I, Px, Py)
    x_star = max(float(x_star), 1e-3)
    y_star = max(float(y_star), 0.0)

    with c_ctrl:
        st.success(f"**x\* = {x_star:.3f}**    |    **y\* = {y_star:.3f}**")
        st.markdown("#### Step-by-step solution")
        for line in solution_text(par, I, Px, Py, x_star, y_star):
            st.markdown(f"- {line}")

    # Chart
    x_int, y_int = I/Px, I/Py
    x_max = max(x_int, x_star) * 1.35
    y_max = max(y_int, y_star) * 1.35

    xs_b = np.linspace(0, x_max, 600)
    ys_b = (I - Px*xs_b)/Py; ys_b = np.where(ys_b>=0, ys_b, np.nan)

    u_level = U_val(par, x_star, y_star)
    segs = ic_pts(par, u_level, x_max, y_max)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    ax.plot(xs_b, ys_b, color="#1a1a2a", lw=2.2, label="Budget line", zorder=3)
    ax.fill_between(xs_b, 0, np.nan_to_num(ys_b), color="gray", alpha=0.08)

    ic_handle = None
    for seg in segs:
        xs_ic, ys_ic = seg
        h, = ax.plot(xs_ic, ys_ic, color="#3d5a80", lw=2.2, alpha=0.85, zorder=2)
        if ic_handle is None: ic_handle = h

    ax.scatter([x_star],[y_star], color="#c0392b", s=100, zorder=6,
               label=f"Optimum  ({x_star:.2f}, {y_star:.2f})")

    m = slope_ic(par, x_star, y_star)
    if m is not None and np.isfinite(m):
        dx=0.10*x_max; x0=max(0,x_star-0.5*dx); x1=min(x_max,x_star+0.5*dx)
        y0=y_star+m*(x0-x_star); y1=y_star+m*(x1-x_star)
        ax.plot([x0,x1],[y0,y1], color="#8b59a0", lw=2.0, ls="--", alpha=0.85)

    ax.set_xlim(0, x_max); ax.set_ylim(0, y_max)
    ax.set_xlabel("Good X", fontsize=12); ax.set_ylabel("Good Y", fontsize=12)
    ax.set_title(f"Optimal Bundle — {u_sel}", fontsize=13)
    ax.grid(alpha=0.28)

    handles = [ax.lines[0], ax.get_children()[3] if ic_handle else ax.lines[1]]
    leg_items = [ax.lines[0]]
    if ic_handle: leg_items.append(ic_handle)
    leg_items.append(ax.collections[-1])
    ax.legend(handles=leg_items,
              labels=["Budget line","Indifference curve","Optimum bundle"],
              loc="upper right", fontsize=10, frameon=True)

    with c_chart:
        st.pyplot(fig, use_container_width=True)
    plt.close(fig)
