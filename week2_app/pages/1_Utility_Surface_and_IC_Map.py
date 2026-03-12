import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from mpl_toolkits.mplot3d import Axes3D  # noqa
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import warnings
warnings.filterwarnings("ignore")

st.set_page_config(page_title="Utility Surface & IC Map", layout="wide")

# ── Palette ────────────────────────────────────────────────────────────────────
C = dict(bg="white", ink="#1a1a2a", accent="#3d5a80",
         faint="#a0a8b8", rule="#cccccc")
_BLUES = LinearSegmentedColormap.from_list("nb_blues", ["#dce8f5","#3d5a80"], N=256)

plt.rcParams.update({
    "font.family":"sans-serif","font.sans-serif":["DejaVu Sans","Arial"],
    "text.color":C["ink"],"axes.labelcolor":C["ink"],
    "xtick.color":C["ink"],"ytick.color":C["ink"],"axes.edgecolor":C["rule"],
    "figure.dpi":100,"axes.spines.top":False,"axes.spines.right":False,
    "axes.grid":True,"grid.alpha":0.30,"grid.linestyle":"-","grid.color":"#dddddd",
})

# ── Maths ──────────────────────────────────────────────────────────────────────
def _safe(X): return np.where(X > 0, X, 1e-9)

def _cd_ic_Y(X, U, p):
    a, b = p["alpha"], 1.0 - p["alpha"]
    return (U / _safe(X)**a)**(1.0/b)

def _cd_surf(Xg, Yg, p):
    a, b = p["alpha"], 1.0 - p["alpha"]
    return (np.maximum(Xg,1e-9)**a) * (np.maximum(Yg,1e-9)**b)

def _ps_ic_Y(X, U, p): return (U - p["a"]*X) / p["b"]
def _ps_surf(Xg, Yg, p): return p["a"]*np.maximum(Xg,0) + p["b"]*np.maximum(Yg,0)
def _pc_surf(Xg, Yg, p): return np.minimum(Xg/p["a"], Yg/p["b"])
def _ql_ic_Y(X, U, p): return U - np.sqrt(_safe(X))
def _ql_surf(Xg, Yg, p): return np.sqrt(np.maximum(Xg,0)) + np.maximum(Yg,0)

# ── Registry ───────────────────────────────────────────────────────────────────
REGISTRY = {}
for _a in [round(x,1) for x in np.arange(0.1,1.0,0.1)]:
    _b = round(1.0-_a,10)
    _key = f"Cobb-Douglas  (α={_a:.1f}, β={_b:.1f})"
    REGISTRY[_key] = dict(
        params={"type":"cd","alpha":_a}, ic_Y=_cd_ic_Y, surface_Z=_cd_surf,
        levels=[0.5,1.0,2.0,3.0,4.0,5.0,6.0,7.0], ax_max=9.0,
        title_2d=rf"$U = X^{{{_a:.1f}}} \cdot Y^{{{_b:.1f}}}$",
        title_3d=rf"$U(X,Y) = X^{{{_a:.1f}}} \cdot Y^{{{_b:.1f}}}$", is_pc=False,
    )
for _lbl,(_a,_b) in [("1:1",(1,1)),("2:1",(2,1)),("3:1",(3,1))]:
    _key = f"Perfect Substitutes {_lbl}  U = {_a}X + {_b}Y"
    REGISTRY[_key] = dict(
        params={"type":"ps","a":_a,"b":_b}, ic_Y=_ps_ic_Y, surface_Z=_ps_surf,
        levels=[1,2,4,6,8,10,12,14], ax_max=10.0,
        title_2d=rf"$U = {_a}X + {_b}Y$", title_3d=rf"$U(X,Y) = {_a}X + {_b}Y$", is_pc=False,
    )
for _lbl,(_a,_b) in [("1:1",(1,1)),("2:1",(2,1)),("3:1",(3,1))]:
    _key = f"Perfect Complements {_lbl}  U = min(X/{_a}, Y/{_b})"
    REGISTRY[_key] = dict(
        params={"type":"pc","a":_a,"b":_b}, ic_Y=lambda X,U,p: np.full_like(X,np.nan),
        surface_Z=_pc_surf, levels=[1,2,3,4,5,6,7,8], ax_max=10.0,
        title_2d=rf"$U = \min(X/{_a},\;Y/{_b})$",
        title_3d=rf"$U(X,Y) = \min(X/{_a},\;Y/{_b})$",
        is_pc=True, pc_a=_a, pc_b=_b,
    )
REGISTRY["Quasi-linear  U = √X + Y"] = dict(
    params={"type":"ql"}, ic_Y=_ql_ic_Y, surface_Z=_ql_surf,
    levels=[1,2,3,4,5,6,7,8], ax_max=10.0,
    title_2d=r"$U = \sqrt{X} + Y$", title_3d=r"$U(X,Y) = \sqrt{X} + Y$", is_pc=False,
)
KEYS = list(REGISTRY.keys())

# ── 2-D panel ──────────────────────────────────────────────────────────────────
def draw_2d(ax, entry):
    p, ic_Y, levels = entry["params"], entry["ic_Y"], entry["levels"]
    is_pc, ax_max   = entry["is_pc"], entry["ax_max"]
    X = np.linspace(0.02, ax_max, 1200)
    ax.set_facecolor("white"); ax.set_xlim(0,ax_max); ax.set_ylim(0,ax_max)
    ax.set_xlabel("Quantity of Good X", fontsize=12)
    ax.set_ylabel("Quantity of Good Y", fontsize=12)
    ax.set_title(entry["title_2d"], color=C["accent"], fontsize=13, fontweight="bold", pad=8)
    for U in levels:
        col, al = C["faint"], 0.75
        if is_pc:
            a_pc,b_pc = entry["pc_a"],entry["pc_b"]
            cx,cy = a_pc*U, b_pc*U
            if cx<=ax_max and cy<=ax_max:
                ax.plot([cx,ax_max],[cy,cy], color=col, lw=3.0, alpha=al)
                ax.plot([cx,cx],[cy,ax_max], color=col, lw=3.0, alpha=al)
                ax.scatter([cx],[cy], color=col, s=45, alpha=al, zorder=4)
                ax.text(cx+ax_max*0.02, cy+ax_max*0.02, f"U={U:.1g}",
                        fontsize=10, color=col, va="bottom",
                        bbox=dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor=col, alpha=0.9))
        else:
            Yv = ic_Y(X, U, p)
            mask = np.isfinite(Yv)&(Yv>=0)&(Yv<=ax_max*1.02)
            if mask.any():
                ax.plot(X[mask], Yv[mask], color=col, lw=3.0, alpha=al)
                idx = np.where(mask)[0]
                xi,yi = X[idx[-1]], Yv[idx[-1]]
                if xi<ax_max*0.90 and 0<yi<ax_max*0.90:
                    ax.scatter([xi],[yi], color=col, s=40, alpha=al, zorder=4)
                    ax.text(xi+ax_max*0.018, yi, f"U={U:.1g}",
                            fontsize=10, color=col, va="center",
                            bbox=dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor=col, alpha=0.9))
    ax.annotate("", xy=(ax_max*0.76,ax_max*0.74), xytext=(ax_max*0.63,ax_max*0.61),
                arrowprops=dict(arrowstyle="-|>",color=C["accent"],lw=2.0,mutation_scale=14))
    ax.text(ax_max*0.645, ax_max*0.785, "Increasing U", fontsize=10,
            color=C["accent"], fontstyle="italic",
            bbox=dict(boxstyle="round,pad=0.25",facecolor="white",edgecolor=C["accent"],alpha=0.88))

# ── 3-D panel ──────────────────────────────────────────────────────────────────
def draw_3d(ax, entry, azimuth):
    p, surf_Z = entry["params"], entry["surface_Z"]
    levels, ax_max = entry["levels"], entry["ax_max"]
    xs = np.linspace(0.05, ax_max, 70); ys = np.linspace(0.05, ax_max, 70)
    Xg,Yg = np.meshgrid(xs,ys); Zg = surf_Z(Xg,Yg,p)
    z_lo = float(np.nanpercentile(Zg,1)); z_hi = float(np.nanpercentile(Zg,99))
    Zg_c = np.clip(Zg, z_lo, z_hi)
    ax.set_facecolor("white")
    for pane in (ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane):
        pane.fill=True; pane.set_facecolor("#f5f5f5"); pane.set_edgecolor(C["rule"])
    ax.tick_params(labelsize=8)
    ax.plot_surface(Xg,Yg,Zg_c, cmap=_BLUES, edgecolor="none", alpha=0.88)
    floor_levs = [l for l in levels if z_lo<l<z_hi]
    if floor_levs:
        ax.contour(Xg,Yg,Zg_c, levels=floor_levs, cmap=_BLUES, offset=z_lo, alpha=0.55)
    ax.set_xlabel("Good X", labelpad=4, fontsize=10)
    ax.set_ylabel("Good Y", labelpad=4, fontsize=10)
    ax.set_zlabel("Utility", labelpad=3, fontsize=10)
    ax.set_title(entry["title_3d"], color=C["accent"], fontsize=12, fontweight="bold", pad=6)
    ax.view_init(elev=28, azim=azimuth)
    return z_lo, z_hi

# ── UI ─────────────────────────────────────────────────────────────────────────
st.title("Utility Surface & Indifference Map")
st.caption("Select a utility function and use the rotation slider to explore the 3-D surface.")

col_ctrl, col_main = st.columns([1, 3])
with col_ctrl:
    func_sel = st.selectbox("Utility function", KEYS, index=4)
    azimuth  = st.slider("Rotate surface", -180, 180, -60, step=5)

entry = REGISTRY[func_sel]

fig = plt.figure(figsize=(14.0, 6.0))
fig.patch.set_facecolor("white")
ax1 = fig.add_axes([0.05, 0.10, 0.40, 0.82])
draw_2d(ax1, entry)
ax2 = fig.add_axes([0.50, 0.04, 0.50, 0.92], projection="3d")
z_lo, z_hi = draw_3d(ax2, entry, azimuth)
fig.colorbar(
    plt.cm.ScalarMappable(cmap=_BLUES, norm=plt.Normalize(vmin=z_lo,vmax=z_hi)),
    ax=ax2, shrink=0.50, pad=0.08, label="Utility", fraction=0.03
)
with col_main:
    st.pyplot(fig, use_container_width=True)
plt.close(fig)
