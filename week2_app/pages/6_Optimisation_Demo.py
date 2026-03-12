import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(page_title="Optimisation Demo", layout="wide")

# ── Palette & rcParams ─────────────────────────────────────────────────────────
CLR = dict(
    accent="#3d5a80", start="#2e8b57", inter="#4682b4", optimum="#c0392b",
    path="#d4813a", budget="#3d3d3d", ic="#7b68b0", bg="#fdf8f2",
)
plt.rcParams.update({
    "font.family":"serif",
    "font.serif":["Palatino Linotype","Book Antiqua","Palatino","Georgia","DejaVu Serif"],
    "axes.titlesize":14,"axes.titleweight":"bold","axes.labelsize":12,
    "axes.labelweight":"bold","xtick.labelsize":10,"ytick.labelsize":10,
    "legend.fontsize":10,"figure.dpi":110,"axes.spines.top":False,
    "axes.spines.right":False,"axes.grid":True,"grid.alpha":0.20,
    "grid.linestyle":"--","grid.color":"#c8c8c8",
})

UTILITY_DESCRIPTIONS = {
    "Cobb-Douglas": "Smooth, bowed-in curves. The consumer always buys some of both goods.",
    "Quasi-linear (ln(x)+y)": "Diminishing returns to Good X only. Optimal X is independent of income.",
    "Perfect complements 1:1": "Goods must be consumed in fixed 1:1 pairs. L-shaped indifference curves.",
    "Perfect complements 2:1": "Goods consumed in fixed 2:1 pairs. Optimum always at the kink.",
    "Perfect complements 3:1": "Goods consumed in fixed 3:1 pairs. Optimum always at the kink.",
    "Perfect substitutes 1:1": "Goods equally interchangeable. Straight-line ICs; likely corner solution.",
    "Perfect substitutes 2:1": "Good X gives twice the utility per unit. Straight-line ICs.",
    "Perfect substitutes 3:1": "Good X gives three times the utility per unit. Straight-line ICs.",
}
UTILITY_OPTIONS = list(UTILITY_DESCRIPTIONS.keys())
ALPHA_OPTIONS   = [0.2, 0.4, 0.5, 0.6, 0.8, 0.9]

def _params(u_label, alpha):
    if u_label == "Cobb-Douglas": return {"kind":"CD","alpha":float(alpha)}
    if "Quasi" in u_label: return {"kind":"QL"}
    if u_label.startswith("Perfect complements"):
        r=int(u_label.split()[-1].split(":")[0]); return {"kind":"PC","r":r}
    if u_label.startswith("Perfect substitutes"):
        a=int(u_label.split()[-1].split(":")[0]); return {"kind":"PS","a":a,"b":1}

def U(par,x,y):
    k=par["kind"]
    if k=="CD": a=par["alpha"]; return (x**a)*(y**(1-a))
    if k=="QL": return np.log(max(x,1e-9))+y
    if k=="PS": return par["a"]*x+par["b"]*y
    if k=="PC": return min(x/par["r"],y)

def MU(par,x,y):
    k=par["kind"]
    if k=="CD": a=par["alpha"]; return a*(x**(a-1))*(y**(1-a)),(1-a)*(x**a)*(y**(-a))
    if k=="QL": return 1.0/x, 1.0
    if k=="PS": return float(par["a"]),float(par["b"])
    if k=="PC":
        r=par["r"]
        if x/r<y: return 1.0/r,0.0
        if x/r>y: return 0.0,1.0
        return 1.0/r,1.0

def bfb(par,x,y,Px,Py): mux,muy=MU(par,x,y); return mux/Px,muy/Py

def optimum(par,I,Px,Py):
    k=par["kind"]
    if k=="CD": a=par["alpha"]; xs=a*I/Px; ys=(1-a)*I/Py; return xs,ys,Px*xs
    if k=="QL":
        xs=Py/Px; ys=(I-Px*xs)/Py
        if ys<0: ys=0.0; xs=I/Px
        return xs,ys,Px*xs
    if k=="PS":
        a,b=par["a"],par["b"]; bx,by=a/Px,b/Py
        sx=(float(I) if bx>by else 0.0) if abs(bx-by)>1e-12 else I/2.0
        xs=sx/Px if sx>0 else 0.0; ys=(I-sx)/Py; return xs,ys,sx
    if k=="PC":
        r=par["r"]; t=I/(Px*r+Py); return r*t,t,Px*r*t

def move_to_budget(par,I,Px,Py,x0,y0):
    x=max(float(x0),1e-3); y=max(float(y0),0.0)
    slack=max(0.0,I-Px*x-Py*y)
    if slack<=1e-12:
        y=max((I-Px*x)/Py,0.0); x=max((I-Py*y)/Px,1e-3); return x,y,slack
    bx,by=bfb(par,x,max(y,1e-12),Px,Py)
    x=x+slack/Px if bx>=by else x
    y=(I-Px*x)/Py; x=max((I-Py*y)/Px,1e-3)
    return x,y,slack

def reallocate(par,I,Px,Py,xsb,ysb):
    step=max(1.0,round(I/100.0,1))
    xs_,ys_,sxs=optimum(par,I,Px,Py)
    sx=float(min(max(Px*xsb,0.0),float(I))); sxs=float(min(max(sxs,0.0),float(I)))
    path=[(float(xsb),float(ysb))]
    for _ in range(int(np.ceil(abs(sxs-sx)/max(step,1e-9)))+10):
        if abs(sx-sxs)<1e-12: break
        x=max(sx/Px,1e-3); y=max((I-sx)/Py,0.0)
        bx,by=bfb(par,x,max(y,1e-12),Px,Py)
        dir_t=1.0 if sxs>sx else -1.0; dir_m=1.0 if bx>by else -1.0 if bx<by else dir_t
        direction=dir_m if np.sign(dir_m)==np.sign(dir_t) else dir_t
        s_=min(step,abs(sxs-sx)); sx=min(max(sx+direction*s_,0.0),float(I))
        path.append((max(sx/Px,1e-3),max((I-sx)/Py,0.0)))
    xf=max(sxs/Px,1e-3); yf=max((I-sxs)/Py,0.0)
    if abs(path[-1][0]-xf)>1e-12 or abs(path[-1][1]-yf)>1e-12: path.append((xf,yf))
    return np.array(path)

def ic_curves(par,x_max,y_max,u_levels,n=800):
    xs=np.linspace(1e-3,x_max,n); k=par["kind"]; curves=[]
    if k=="CD":
        a=par["alpha"]
        for u in u_levels:
            ys=(u/(xs**a))**(1/(1-a))
            curves.append((xs,np.where(np.isfinite(ys)&(ys>=0)&(ys<=y_max),ys,np.nan)))
    elif k=="QL":
        for u in u_levels:
            ys=u-np.log(xs)
            curves.append((xs,np.where(np.isfinite(ys)&(ys>=0)&(ys<=y_max),ys,np.nan)))
    elif k=="PS":
        a,b=par["a"],par["b"]
        for u in u_levels:
            ys=(u-a*xs)/b
            curves.append((xs,np.where(np.isfinite(ys)&(ys>=0)&(ys<=y_max),ys,np.nan)))
    elif k=="PC":
        r=par["r"]
        for u in u_levels:
            xk,yk=r*u,u
            xs1=np.linspace(xk,x_max,50); ys1=np.full_like(xs1,yk)
            ys2=np.linspace(yk,y_max,50); xs2=np.full_like(ys2,xk)
            curves.append((xs1,np.where(ys1<=y_max,ys1,np.nan)))
            curves.append((np.where(xs2<=x_max,xs2,np.nan),ys2))
    return curves

def bline(I,Px,Py,x_max,n=400):
    xs=np.linspace(0,x_max,n); ys=(I-Px*xs)/Py; return xs,np.where(ys>=0,ys,np.nan)

# ─────────────────────────────── UI ───────────────────────────────────────────
st.title("Consumer Optimisation Explorer")
st.caption("Bang-for-the-Buck Reallocation — watch the consumer step by step to the optimum.")

# Controls
cc1, cc2, cc3 = st.columns([2,2,1])
with cc1:
    u_sel   = st.selectbox("Utility type", UTILITY_OPTIONS, key="opt_u")
    alpha   = st.selectbox("α (Cobb-Douglas only)", ALPHA_OPTIONS, index=2,
                           disabled="Cobb" not in u_sel, key="opt_a")
    st.caption(f"*{UTILITY_DESCRIPTIONS[u_sel]}*")
with cc2:
    I_  = st.slider("Income  I",  20, 500, 120, step=10, key="opt_I")
    Px_ = st.slider("Price  Pₓ",   1,  20,   6, step=1,  key="opt_Px")
    Py_ = st.slider("Price  Pᵧ",   1,  20,   4, step=1,  key="opt_Py")
with cc3:
    st.markdown("**Starting point**")
    x0_ = st.number_input("x₀", 1, 200, 5, key="opt_x0")
    y0_ = st.number_input("y₀", 1, 200, 5, key="opt_y0")
    if st.button("Randomise", key="rnd"):
        rng=np.random.default_rng()
        xm=max(int(I_//Px_),2); x0_=int(rng.integers(1,xm))
        yc=max(int((I_-Px_*x0_)//Py_),1); y0_=int(rng.integers(1,yc+1))
        st.session_state["opt_x0"]=x0_; st.session_state["opt_y0"]=y0_

par      = _params(u_sel, alpha)
xsb, ysb, slack = move_to_budget(par,I_,Px_,Py_,x0_,y0_)
path     = reallocate(par,I_,Px_,Py_,xsb,ysb)
x_star,y_star,_ = optimum(par,I_,Px_,Py_)

x_int=I_/Px_; y_int=I_/Py_
x_max=max(x_int,np.max(path[:,0]),float(x0_),xsb,1.0)*1.28
y_max=max(y_int,np.max(path[:,1]),float(y0_),ysb,1.0)*1.28

u_sb   = U(par,xsb,max(ysb,1e-12))
u_opt  = U(par,x_star,max(y_star,1e-12))
u_mid  = 0.5*(u_sb+u_opt)
u_levs = [u_sb,u_mid,u_opt] if abs(u_opt-u_sb)>1e-12 else [u_sb*0.9,u_sb,u_sb*1.1]
alphas_ic=[0.38,0.55,0.88]; lws_ic=[1.4,1.7,2.4]
curves   = ic_curves(par,x_max,y_max,u_levs)
xs_b,ys_b = bline(I_,Px_,Py_,x_max)

fig,ax = plt.subplots(figsize=(10,7))
fig.patch.set_facecolor(CLR["bg"]); ax.set_facecolor(CLR["bg"])

# Budget line
ax.plot(xs_b,ys_b,color=CLR["budget"],lw=2.8,label="Budget line",zorder=3)
ax.fill_between(xs_b,0,np.nan_to_num(ys_b),color=CLR["budget"],alpha=0.05)
ax.annotate(f"I/Pₓ={x_int:.1f}",xy=(x_int,0),xytext=(0,-22),
            textcoords="offset points",fontsize=9,color=CLR["budget"],ha="center",
            arrowprops=dict(arrowstyle="-",color=CLR["budget"],lw=0.8,alpha=0.5))
ax.annotate(f"I/Pᵧ={y_int:.1f}",xy=(0,y_int),xytext=(-42,0),
            textcoords="offset points",fontsize=9,color=CLR["budget"],va="center",
            arrowprops=dict(arrowstyle="-",color=CLR["budget"],lw=0.8,alpha=0.5))

# IC curves
for i,(xs_ic,ys_ic) in enumerate(curves):
    ai=min(i,2)
    lbl="Indifference curves" if i==0 else ("Optimal IC" if i==len(u_levs)-1 else None)
    ax.plot(xs_ic,ys_ic,color=CLR["ic"],lw=lws_ic[ai],alpha=alphas_ic[ai],
            label=lbl,zorder=2)
    u_=u_levs[min(i,len(u_levs)-1)]
    v=np.where(np.isfinite(ys_ic)&(ys_ic>=0))[0]
    if len(v)>0:
        rx=xs_ic[v[-1]]; ry=ys_ic[v[-1]]
        if rx<x_max*0.97 and ry<y_max*0.97:
            ax.text(rx+x_max*0.012,ry,f"U={u_:.2f}",fontsize=8.5,
                    color=CLR["ic"],alpha=alphas_ic[ai]+0.1,va="center")

# Reallocation path
ax.plot(path[:,0],path[:,1],color=CLR["path"],lw=2.0,
        marker="o",ms=2.8,alpha=0.85,label="Reallocation path",zorder=4)
step_arr=max(1,len(path)//14)
for ki in range(0,len(path)-1,step_arr):
    x1,y1=path[ki]; x2,y2=path[ki+1]
    ax.annotate("",xy=(x2,y2),xytext=(x1,y1),
                arrowprops=dict(arrowstyle="->",color=CLR["path"],lw=1.0,alpha=0.60))

# Key points
ax.scatter(x0_,y0_,s=110,color=CLR["start"],marker="D",zorder=7,label="Step 1: Starting bundle")
ax.scatter(xsb,ysb, s=110,color=CLR["inter"],marker="o",zorder=7,label="Step 2: Move to budget line")
ax.scatter(path[-1,0],path[-1,1],s=230,color=CLR["optimum"],marker="*",zorder=8,label="Step 3: Optimal bundle")

def _bubble(xy,txt,col,offset):
    ax.annotate(txt,xy=xy,xytext=offset,textcoords="offset points",
                fontsize=9.5,fontweight="bold",color="white",
                bbox=dict(boxstyle="round,pad=0.45",fc=col,ec="none",alpha=0.90),
                arrowprops=dict(arrowstyle="-",color=col,lw=1.3,alpha=0.65))

_bubble((float(x0_),float(y0_)),f"Step 1: Start\n({float(x0_):.1f},{float(y0_):.1f})",CLR["start"],(20,14))
_bubble((xsb,ysb),f"Step 2: Budget line\n({xsb:.1f},{ysb:.1f})",CLR["inter"],(20,-30))
_bubble((path[-1,0],path[-1,1]),f"Step 3: Optimum\n({path[-1,0]:.1f},{path[-1,1]:.1f})",CLR["optimum"],(-24,20))

ax.set_xlim(0,x_max); ax.set_ylim(0,y_max)
ax.set_xlabel("Good X  (units)",labelpad=9); ax.set_ylabel("Good Y  (units)",labelpad=9)
ax.set_title(f"Bang-for-the-Buck Reallocation  ·  {u_sel}",pad=12,color=CLR["accent"])
ax.legend(loc="upper right",frameon=True,fancybox=True,framealpha=0.93,
          edgecolor="#dddddd",fontsize=9.5)
plt.tight_layout(pad=1.6)

st.pyplot(fig, use_container_width=True)
plt.close(fig)

# Explanation cards
st.markdown("---")
st.markdown(f"""
**Parameters:** I = **{I_}** | Pₓ = **{Px_}** | Pᵧ = **{Py_}** | Pₓ/Pᵧ = **{Px_/Py_:.2f}**

**① Starting Bundle** ({float(x0_):.1f}, {float(y0_):.1f}) — unspent income = **${slack:.1f}**

**② Move to Budget Line** — spend slack on whichever good has higher MU/P.  
Now at ({xsb:.1f}, {ysb:.1f}), spending full income of **${I_}**.

**③ Reach the Optimum** — reallocate spending from the lower MU/P good to the higher one 
until **MUₓ/Pₓ = MUᵧ/Pᵧ** (equal bang-for-the-buck).  
Optimal bundle: **({path[-1,0]:.2f}, {path[-1,1]:.2f})**
""")
