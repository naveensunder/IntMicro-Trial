import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

st.set_page_config(page_title="IC Shape Explorer", layout="wide")

C = dict(bg="white", ink="#1a1a2a", accent="#3d5a80", alt="#6b8cae",
         faint="#a0a8b8", rule="#cccccc", hi="#c0392b")

plt.rcParams.update({
    "font.family":"sans-serif","font.sans-serif":["DejaVu Sans","Arial"],
    "text.color":C["ink"],"axes.labelcolor":C["ink"],
    "xtick.color":C["ink"],"ytick.color":C["ink"],"axes.edgecolor":C["rule"],
    "figure.dpi":100,"axes.spines.top":False,"axes.spines.right":False,
    "axes.grid":True,"grid.alpha":0.30,"grid.linestyle":"-","grid.color":"#dddddd",
})

def _sx(X): return np.where(X > 0, X, 1e-9)
def cd_Y(X,k,a): return (k/_sx(X)**a)**(1.0/(1-a))
def ql_Y(X,k): return k - np.sqrt(_sx(X))
def ces_Y(X,k,rho):
    inner = k**rho - np.maximum(X,1e-9)**rho
    with np.errstate(invalid="ignore",divide="ignore"):
        return np.where(inner>0, inner**(1.0/rho), np.nan)
def ps_Y(X,k,a,b): return (k-a*X)/b
def concave_Y(X,k):
    with np.errstate(invalid="ignore"):
        return np.where(k-X**2>=0, np.sqrt(np.maximum(k-X**2,0)), np.nan)
SG_X0,SG_Y0 = 0.5,0.5
def sg_Y(X,k,a):
    with np.errstate(invalid="ignore",divide="ignore"):
        return (k/np.maximum(X-SG_X0,1e-9)**a)**(1.0/(1-a)) + SG_Y0

_CD_L=[0.5,1,2,3,4,5,6.5,8]; _PS_L=[1,2,4,6,8,10,12,14]
_PC_L=[1,2,3,4,5,6,7,8]; _QL_L=[1.5,2.5,3.5,4.5,5.5,6.5,7.5,8.5]
_CES_L=[0.5,1,2,3,4,5,6.5,8]; _CC_L=[2,4,8,12,16,20,25,30]; _SG_L=[0.3,0.6,1,1.5,2,3,4,5]
AX=10.0

REGISTRY={}
for _a in [round(x,1) for x in np.arange(0.1,1.0,0.1)]:
    _b=round(1-_a,10); lbl=f"Cobb-Douglas  (α={_a:.1f}, β={_b:.1f})"
    REGISTRY[lbl]=(rf"$U=X^{{{_a:.1f}}}\cdot Y^{{{_b:.1f}}}$",AX,
                   lambda ax,lev,am,a=_a: _draw_cd(ax,a,lev,am),_CD_L)
for r2,(_a,_b) in [("1:1",(1,1)),("2:1",(2,1)),("3:1",(3,1)),("1:2",(1,2))]:
    lbl=f"Perfect Substitutes {r2}   U = {_a}X + {_b}Y"
    REGISTRY[lbl]=(rf"$U={_a}X+{_b}Y$",AX,
                   lambda ax,lev,am,a=_a,b=_b: _draw_ps(ax,a,b,lev,am),_PS_L)
for r2,(_a,_b) in [("1:1",(1,1)),("2:1",(2,1)),("3:1",(3,1)),("1:2",(1,2))]:
    lbl=f"Perfect Complements {r2}   U = min(X/{_a}, Y/{_b})"
    REGISTRY[lbl]=(rf"$U=\min(X/{_a},\;Y/{_b})$",AX,
                   lambda ax,lev,am,a=_a,b=_b: _draw_pc(ax,a,b,lev,am),_PC_L)
REGISTRY["Quasi-linear   U = √X + Y"]=(r"$U=\sqrt{X}+Y$",AX,
    lambda ax,lev,am: _draw_ql(ax,lev,am),_QL_L)
for rho,desc,lat in [(0.8,"CES  ρ=0.8  (near substitutes)",r"$U=(X^{0.8}+Y^{0.8})^{1/0.8}$"),
                     (-0.5,"CES  ρ=−0.5  (≈ Cobb-Douglas)",r"$U=(X^{-0.5}+Y^{-0.5})^{-2}$"),
                     (-5.0,"CES  ρ=−5   (near complements)",r"$U=(X^{-5}+Y^{-5})^{-0.2}$")]:
    REGISTRY[desc]=(lat,AX,lambda ax,lev,am,r=rho: _draw_ces(ax,r,lev,am),_CES_L)
REGISTRY["Concave   U = X² + Y²"]=(r"$U=X^2+Y^2$",8.0,
    lambda ax,lev,am: _draw_concave(ax,lev,am),_CC_L)
REGISTRY["Stone-Geary   U = (X−0.5)^0.5·(Y−0.5)^0.5"]=(
    r"$U=(X{-}0.5)^{0.5}(Y{-}0.5)^{0.5}$",AX,
    lambda ax,lev,am: _draw_sg(ax,lev,am),_SG_L)
KEYS=list(REGISTRY.keys())

def _cols(n,hi=None):
    cs=[C["accent"] if i%2==0 else C["alt"] for i in range(n)]
    al=[0.85]*n; lw=[3.0]*n
    if hi is not None:
        for i in range(n):
            if i==hi: cs[i]=C["hi"]; al[i]=1.0; lw[i]=3.8
            else: al[i]=0.35; lw[i]=2.0
    return cs,al,lw

def _lbl(ax,xi,yi,k,col,am,al=0.9,bold=False):
    if xi<am*0.90 and 0<yi<am*0.90:
        ax.scatter([xi],[yi],color=col,s=50,zorder=6,alpha=1.0 if bold else al)
        ax.text(xi+am*0.018,yi,f"U={k:.2g}",fontsize=10,color=col,va="center",zorder=7,
                fontweight="bold" if bold else "normal",
                bbox=dict(boxstyle="round,pad=0.22",facecolor="white",edgecolor=col,alpha=0.92))

def _arrow(ax,am):
    ax.annotate("",xy=(am*0.76,am*0.74),xytext=(am*0.63,am*0.61),
                arrowprops=dict(arrowstyle="-|>",color=C["accent"],lw=2.0,mutation_scale=14))
    ax.text(am*0.645,am*0.785,"Increasing U",fontsize=10,color=C["accent"],fontstyle="italic",
            bbox=dict(boxstyle="round,pad=0.25",facecolor="white",edgecolor=C["accent"],alpha=0.88))

def _draw_cd(ax,a,levels,am,hi=None):
    X=np.linspace(0.02,am,1200); cs,al,lw=_cols(len(levels),hi)
    for i,(k,col,a_,l_) in enumerate(zip(levels,cs,al,lw)):
        Yv=cd_Y(X,k,a); mask=np.isfinite(Yv)&(Yv>=0)&(Yv<=am*1.02)
        if mask.any():
            ax.plot(X[mask],Yv[mask],color=col,lw=l_,alpha=a_,zorder=3)
            idx=np.where(mask)[0]; _lbl(ax,X[idx[-1]],Yv[idx[-1]],k,col,am,a_,i==hi)
    _arrow(ax,am)

def _draw_ps(ax,a,b,levels,am,hi=None):
    X=np.linspace(0,am,800); cs,al,lw=_cols(len(levels),hi)
    for i,(k,col,a_,l_) in enumerate(zip(levels,cs,al,lw)):
        Yv=ps_Y(X,k,a,b); mask=(Yv>=0)&(Yv<=am*1.02)&(X>=0)
        if mask.any():
            ax.plot(X[mask],Yv[mask],color=col,lw=l_,alpha=a_,zorder=3)
            idx=np.where(mask)[0]; _lbl(ax,X[idx[-1]],Yv[idx[-1]],k,col,am,a_,i==hi)
    _arrow(ax,am)

def _draw_pc(ax,a,b,levels,am,hi=None):
    cs,al,lw=_cols(len(levels),hi)
    for i,(k,col,a_,l_) in enumerate(zip(levels,cs,al,lw)):
        cx,cy=a*k,b*k
        if cx<=am and cy<=am:
            ax.plot([cx,am],[cy,cy],color=col,lw=l_,alpha=a_,zorder=3)
            ax.plot([cx,cx],[cy,am],color=col,lw=l_,alpha=a_,zorder=3)
            ax.scatter([cx],[cy],color=col,s=55,zorder=6,alpha=1.0 if i==hi else a_)
            ax.text(cx+am*0.02,cy+am*0.025,f"U={k:.2g}",fontsize=10,color=col,
                    va="bottom",zorder=7,fontweight="bold" if i==hi else "normal",
                    bbox=dict(boxstyle="round,pad=0.22",facecolor="white",edgecolor=col,alpha=0.92))
    Xr=np.linspace(0,am,200); Yr=(b/a)*Xr
    ax.plot(Xr[Yr<=am],Yr[Yr<=am],color=C["rule"],lw=1.2,ls="--",zorder=1)
    _arrow(ax,am)

def _draw_ql(ax,levels,am,hi=None):
    X=np.linspace(0.02,am,1200); cs,al,lw=_cols(len(levels),hi)
    for i,(k,col,a_,l_) in enumerate(zip(levels,cs,al,lw)):
        Yv=ql_Y(X,k); mask=np.isfinite(Yv)&(Yv>=0)&(Yv<=am*1.02)
        if mask.any():
            ax.plot(X[mask],Yv[mask],color=col,lw=l_,alpha=a_,zorder=3)
            idx=np.where(mask)[0]; _lbl(ax,X[idx[-1]],Yv[idx[-1]],k,col,am,a_,i==hi)
    _arrow(ax,am)

def _draw_ces(ax,rho,levels,am,hi=None):
    X=np.linspace(0.02,am,1200); cs,al,lw=_cols(len(levels),hi)
    for i,(k,col,a_,l_) in enumerate(zip(levels,cs,al,lw)):
        Yv=ces_Y(X,k,rho); mask=np.isfinite(Yv)&(Yv>=0)&(Yv<=am*1.02)
        if mask.any():
            ax.plot(X[mask],Yv[mask],color=col,lw=l_,alpha=a_,zorder=3)
            idx=np.where(mask)[0]; _lbl(ax,X[idx[-1]],Yv[idx[-1]],k,col,am,a_,i==hi)
    _arrow(ax,am)

def _draw_concave(ax,levels,am,hi=None):
    X=np.linspace(0,am,1200); cs,al,lw=_cols(len(levels),hi)
    for i,(k,col,a_,l_) in enumerate(zip(levels,cs,al,lw)):
        Yv=concave_Y(X,k); mask=np.isfinite(Yv)&(Yv>=0)&(Yv<=am*1.02)
        if mask.any():
            ax.plot(X[mask],Yv[mask],color=col,lw=l_,alpha=a_,zorder=3)
            idx=np.where(mask)[0]; _lbl(ax,X[idx[-1]],Yv[idx[-1]],k,col,am,a_,i==hi)
    _arrow(ax,am)

def _draw_sg(ax,levels,am,hi=None):
    X=np.linspace(SG_X0+0.02,am,1200); cs,al,lw=_cols(len(levels),hi)
    for i,(k,col,a_,l_) in enumerate(zip(levels,cs,al,lw)):
        Yv=sg_Y(X,k,0.5); mask=np.isfinite(Yv)&(Yv>=SG_Y0)&(Yv<=am*1.02)
        if mask.any():
            ax.plot(X[mask],Yv[mask],color=col,lw=l_,alpha=a_,zorder=3)
            idx=np.where(mask)[0]; _lbl(ax,X[idx[-1]],Yv[idx[-1]],k,col,am,a_,i==hi)
    ax.axvline(SG_X0,color="#bbbbbb",lw=1.1,ls="--"); ax.axhline(SG_Y0,color="#bbbbbb",lw=1.1,ls="--")
    _arrow(ax,am)

def build_panel(ax, key, hi_U=0.0):
    title, ax_max, draw_fn, levels = REGISTRY[key]
    hi_idx = None
    if hi_U > 0:
        hi_idx = int(np.argmin([abs(l-hi_U) for l in levels]))
    ax.set_facecolor("white"); ax.set_xlim(0,ax_max); ax.set_ylim(0,ax_max)
    ax.set_xlabel("Quantity of Good X", fontsize=12)
    ax.set_ylabel("Quantity of Good Y", fontsize=12)
    ax.set_title(title, color=C["accent"], fontsize=13, fontweight="bold", pad=8)
    draw_fn(ax, levels, ax_max, hi_idx)
    if hi_U > 0 and hi_idx is not None:
        ax.text(0.5,1.005,f"highlighted: U = {levels[hi_idx]:.2g}",
                transform=ax.transAxes, ha="center", va="bottom",
                fontsize=10, color=C["hi"], fontstyle="italic")

# ── UI ─────────────────────────────────────────────────────────────────────────
st.title("IC Shape Explorer — Compare Two Functions")

c1,c2 = st.columns(2)
with c1:
    sel_A = st.selectbox("Left panel", KEYS, index=4, key="A")
with c2:
    default_B = next(i for i,k in enumerate(KEYS) if "Complements 1:1" in k)
    sel_B = st.selectbox("Right panel", KEYS, index=default_B, key="B")

co1,co2 = st.columns(2)
with co1:
    same_axes = st.checkbox("Use identical axis scale for both panels", value=False)
with co2:
    hi_val = st.slider("Highlight U =", 0.0, 14.0, 0.0, step=0.5,
                       help="Set to 0 to show all curves equally")

if hi_val > 0:
    _,amA,_,levA = REGISTRY[sel_A]; _,amB,_,levB = REGISTRY[sel_B]
    hiA = levA[int(np.argmin([abs(l-hi_val) for l in levA]))]
    hiB = levB[int(np.argmin([abs(l-hi_val) for l in levB]))]
    st.caption(f"Left panel: U = {hiA:.2g} highlighted   |   Right panel: U = {hiB:.2g} highlighted")

forced = None
if same_axes:
    _,amA,_,_ = REGISTRY[sel_A]; _,amB,_,_ = REGISTRY[sel_B]
    forced = max(amA,amB)

fig, axes = plt.subplots(1,2, figsize=(14.0, 6.5))
fig.patch.set_facecolor("white")
fig.subplots_adjust(wspace=0.28, left=0.07, right=0.97, top=0.88, bottom=0.11)

for ax, key in zip(axes, [sel_A, sel_B]):
    title, ax_max, draw_fn, levels = REGISTRY[key]
    if forced: ax_max = forced
    hi_idx = int(np.argmin([abs(l-hi_val) for l in levels])) if hi_val>0 else None
    ax.set_facecolor("white"); ax.set_xlim(0,ax_max); ax.set_ylim(0,ax_max)
    ax.set_xlabel("Quantity of Good X",fontsize=12); ax.set_ylabel("Quantity of Good Y",fontsize=12)
    ax.set_title(title, color=C["accent"], fontsize=13, fontweight="bold", pad=8)
    draw_fn(ax, levels, ax_max, hi_idx)
    if hi_val>0 and hi_idx is not None:
        ax.text(0.5,1.005,f"highlighted: U = {levels[hi_idx]:.2g}",
                transform=ax.transAxes,ha="center",va="bottom",
                fontsize=10,color=C["hi"],fontstyle="italic")

title_str = "Compare Indifference Curve Shapes"
if same_axes: title_str += "  (identical axis scale)"
fig.suptitle(title_str, fontsize=14, color=C["accent"], fontweight="bold", y=0.98)

st.pyplot(fig, use_container_width=True)
plt.close(fig)
