import streamlit as st
import numpy as np
import sympy as sp
import matplotlib.pyplot as plt

st.set_page_config(page_title="Practice Problems", layout="wide")

# ── Colours ────────────────────────────────────────────────────────────────────
CLR = dict(
    budget="#2c3e50", ic_faint="#9b8ec4", ic_opt="#5b3fa0",
    optimum="#c0392b", feasible="#b0bec5", tangent="#e67e22",
    bg="#fdf8f2", accent="#3d5a80",
)
plt.rcParams.update({
    "font.family":"serif",
    "font.serif":["Palatino Linotype","Book Antiqua","Palatino","Georgia","DejaVu Serif"],
    "axes.titlesize":13,"axes.titleweight":"bold","axes.labelsize":11,
    "axes.labelweight":"bold","xtick.labelsize":9,"ytick.labelsize":9,
    "figure.dpi":110,"axes.spines.top":False,"axes.spines.right":False,
    "axes.grid":True,"grid.alpha":0.18,"grid.linestyle":"--","grid.color":"#c8c8c8",
})

UTILITY_TYPES   = ["cobb_douglas","perfect_substitutes","perfect_complements","neutral_good"]
UTILITY_WEIGHTS = np.array([0.45,0.25,0.25,0.05])
COBB_ALPHAS_R   = [sp.Rational(k,10) for k in [2,4,5,6,8,9]]

METHOD_BLURB = {
    "cobb_douglas":
        "**Cobb-Douglas** — smooth, strictly concave utility. Optimum is an *interior solution* found by the **tangency condition**: set MRS = Pₓ/Pᵧ and combine with the budget constraint.",
    "perfect_substitutes":
        "**Perfect Substitutes** — straight-line ICs, so the consumer picks a *corner solution*: spend *all* income on whichever good gives more utility per dollar (compare MUₓ/Pₓ vs MUᵧ/Pᵧ).",
    "perfect_complements":
        "**Perfect Complements** — L-shaped ICs, so the optimum is always at the *kink*. Use the fixed-proportions condition X/a = Y/b together with the budget constraint.",
    "neutral_good":
        "**Neutral Good** — Y gives no utility. The consumer spends all income on X: X* = I/Pₓ, Y* = 0.",
}

WORKED_EXAMPLES = {
    "cobb_douglas": {
        "title":"Worked Example — Cobb-Douglas",
        "given":"U(X,Y) = X^0.5·Y^0.5,  Pₓ = 4,  Pᵧ = 2,  I = 80",
        "steps":[
            ("Budget constraint","4X + 2Y = 80  →  Y = 40 − 2X\nX-intercept = 80/4 = **20**,  Y-intercept = 80/2 = **40**"),
            ("Tangency condition","MRS = MUₓ/MUᵧ = Y/X\nSet MRS = Pₓ/Pᵧ:  Y/X = 4/2 = 2  →  Y = 2X"),
            ("Substitute into budget","4X + 2(2X) = 80  →  8X = 80  →  **X* = 10**,  **Y* = 20**"),
            ("Verify","4(10) + 2(20) = 40 + 40 = **80 = I ✓**"),
        ],
        "answer":"X* = 10,  Y* = 20",
    },
    "perfect_substitutes": {
        "title":"Worked Example — Perfect Substitutes",
        "given":"U(X,Y) = 3X + 2Y,  Pₓ = 5,  Pᵧ = 4,  I = 60",
        "steps":[
            ("Budget constraint","5X + 4Y = 60  →  X-int = 12,  Y-int = 15"),
            ("Bang-for-buck","MUₓ/Pₓ = 3/5 = **0.60**\nMUᵧ/Pᵧ = 2/4 = **0.50**\nX gives more utility per dollar → spend *all* income on X"),
            ("Corner solution","**X* = 60/5 = 12**,  **Y* = 0**"),
            ("Verify","5(12) + 4(0) = 60 = **I ✓**"),
        ],
        "answer":"X* = 12,  Y* = 0",
    },
    "perfect_complements": {
        "title":"Worked Example — Perfect Complements",
        "given":"U(X,Y) = min(X/2, Y/3),  Pₓ = 3,  Pᵧ = 2,  I = 60",
        "steps":[
            ("Budget constraint","3X + 2Y = 60"),
            ("Fixed-proportions condition","At the kink: X/2 = Y/3  →  Y = (3/2)X"),
            ("Substitute into budget","3X + 2·(3/2)X = 60  →  6X = 60  →  **X* = 10**,  Y* = 15"),
            ("Verify","3(10) + 2(15) = 30 + 30 = **60 = I ✓**"),
        ],
        "answer":"X* = 10,  Y* = 15",
    },
    "neutral_good": {
        "title":"Worked Example — Neutral Good",
        "given":"U(X,Y) = X  (Y is neutral),  Pₓ = 5,  Pᵧ = 3,  I = 50",
        "steps":[
            ("Budget constraint","5X + 3Y = 50"),
            ("Y is neutral","Y gives zero MU. Set Y* = 0 and spend all income on X."),
            ("Optimal X","**X* = 50/5 = 10**,  **Y* = 0**"),
            ("Verify","5(10) + 3(0) = 50 = **I ✓**"),
        ],
        "answer":"X* = 10,  Y* = 0",
    },
}

GRAPH_QUESTIONS = {
    "cobb_douglas":[
        "Why does the optimum sit exactly where the budget line *touches* (not crosses) the highest indifference curve?",
        "What would happen to X* and Y* if income *doubled*?",
        "The tangent dashed line has slope −Pₓ/Pᵧ. What does it mean that the IC has the same slope at the optimum?",
    ],
    "perfect_substitutes":[
        "Why is the optimum at a *corner* of the budget line rather than in the middle?",
        "Under what price condition would the consumer be *indifferent* between any point on the budget line?",
        "What would change if Pₓ fell enough to flip the bang-for-buck comparison?",
    ],
    "perfect_complements":[
        "Why is the optimum always at the *kink* of the indifference curve?",
        "Can you verify the optimum lies on the line X/a = Y/b with these numbers?",
        "How would a rise in Pₓ affect the optimal bundle? Would the ratio X*/Y* change?",
    ],
    "neutral_good":[
        "Why does the consumer set Y* = 0 even though Y is not a 'bad'?",
        "All indifference curves are vertical lines. What does that tell you about MUᵧ?",
        "What would the solution look like if *X* were the neutral good instead?",
    ],
}

# ── Helpers ────────────────────────────────────────────────────────────────────
def _clean(Px,Py,I):
    if Px<=0 or Py<=0 or I<=0: return False
    if (I%Px)!=0 or (I%Py)!=0: return False
    if I/Px>150 or I/Py>150: return False
    return True

def gen_problem(rng=None):
    rng=rng or np.random.default_rng()
    for _ in range(500):
        utype=rng.choice(UTILITY_TYPES,p=UTILITY_WEIGHTS)
        I=int(rng.choice(np.arange(50,501,10)))
        Px=int(rng.integers(1,21)); Py=int(rng.integers(1,21))
        if not _clean(Px,Py,I): continue
        p={"utility_type":utype,"Px":Px,"Py":Py,"I":I}
        if utype=="cobb_douglas":
            a=rng.choice(COBB_ALPHAS_R)
            p.update({"alpha":a,"beta":sp.Rational(1,1)-a})
        elif utype=="perfect_substitutes":
            a=int(rng.integers(1,11)); b=int(rng.integers(1,11))
            if abs(a/Px-b/Py)<0.05: continue
            p.update({"a":a,"b":b})
        elif utype=="perfect_complements":
            a=int(rng.integers(1,6)); b=int(rng.integers(1,6))
            p.update({"a":a,"b":b})
        return p
    return {"utility_type":"cobb_douglas","Px":5,"Py":4,"I":100,
            "alpha":sp.Rational(1,2),"beta":sp.Rational(1,2)}

def solve(prob):
    Px,Py,I=prob["Px"],prob["Py"],prob["I"]; t=prob["utility_type"]; D={}
    if t=="cobb_douglas":
        a,b=prob["alpha"],prob["beta"]
        xs_=float((a*sp.Rational(I,Px)).evalf()); ys_=float((b*sp.Rational(I,Py)).evalf())
        D={"method":"Tangency","alpha":a,"beta":b}
    elif t=="perfect_substitutes":
        a,b=prob["a"],prob["b"]; bx=a/Px; by=b/Py
        xs_=(I/Px if bx>by else 0.0); ys_=(0.0 if bx>by else I/Py)
        D={"method":"Corner","a":a,"b":b,"bang_x":bx,"bang_y":by,"corner":"All on X" if bx>by else "All on Y"}
    elif t=="perfect_complements":
        a,b=prob["a"],prob["b"]; s=I/(Px*a+Py*b)
        xs_=a*s; ys_=b*s; D={"method":"Kink","a":a,"b":b,"s":s}
    else:
        xs_=I/Px; ys_=0.0; D={"method":"Neutral"}
    return {"x_star":float(xs_),"y_star":float(ys_),"x_int":I/Px,"y_int":I/Py,
            "slope":-Px/Py,"details":D}

def util_str(prob):
    t=prob["utility_type"]
    if t=="cobb_douglas":
        a,b=prob["alpha"],prob["beta"]
        return f"U(X,Y) = X^{a} · Y^{b}"
    if t=="perfect_substitutes": return f"U(X,Y) = {prob['a']}X + {prob['b']}Y"
    if t=="perfect_complements": return f"U(X,Y) = min(X/{prob['a']}, Y/{prob['b']})"
    return "U(X,Y) = X  (Y is neutral)"

def solution_steps(prob, sol):
    Px,Py,I=prob["Px"],prob["Py"],prob["I"]; t=prob["utility_type"]
    D=sol["details"]; xs_,ys_=sol["x_star"],sol["y_star"]
    x_int,y_int,slope=sol["x_int"],sol["y_int"],sol["slope"]
    steps=[]
    steps.append(("Budget Constraint",
        f"{Px}X + {Py}Y = {I}\n"
        f"X-intercept: I/Pₓ = {I}/{Px} = {x_int:.4g}\n"
        f"Y-intercept: I/Pᵧ = {I}/{Py} = {y_int:.4g}\n"
        f"Slope: −Pₓ/Pᵧ = −{Px}/{Py} = {slope:.4g}"))
    if t=="cobb_douglas":
        a,b=D["alpha"],D["beta"]; af,bf=float(a),float(b)
        steps.append(("Tangency Condition",
            f"U = X^{a} · Y^{b}\n"
            f"MRS = MUₓ/MUᵧ = ({af:.1f}/{bf:.1f})·(Y/X)\n"
            f"Set MRS = Pₓ/Pᵧ, then use budget constraint:\n"
            f"X* = α·I/Pₓ = {af:.1f}×{I}/{Px} = **{xs_:.4g}**\n"
            f"Y* = β·I/Pᵧ = {bf:.1f}×{I}/{Py} = **{ys_:.4g}**"))
    elif t=="perfect_substitutes":
        steps.append(("Bang-for-Buck",
            f"U = {D['a']}X + {D['b']}Y\n"
            f"MUₓ/Pₓ = {D['a']}/{Px} = {D['bang_x']:.4g}\n"
            f"MUᵧ/Pᵧ = {D['b']}/{Py} = {D['bang_y']:.4g}\n"
            f"→ {D['corner']}\n"
            f"X* = **{xs_:.4g}**,  Y* = **{ys_:.4g}**"))
    elif t=="perfect_complements":
        a,b,s=D["a"],D["b"],D["s"]
        steps.append(("Fixed-Proportions Condition",
            f"At the kink: X/{a} = Y/{b}  →  Y = ({b}/{a})X\n"
            f"Substitute: {Px}X + {Py}·({b}/{a})X = {I}\n"
            f"s = I/(Pₓ·{a} + Pᵧ·{b}) = {s:.4f}\n"
            f"X* = {a}·s = **{xs_:.4g}**,  Y* = {b}·s = **{ys_:.4g}**"))
    else:
        steps.append(("Neutral Good",
            f"Y gives no utility → Y* = 0\n"
            f"X* = I/Pₓ = {I}/{Px} = **{xs_:.4g}**"))
    budget_check = Px*xs_+Py*ys_
    steps.append(("Budget Verification",
        f"Pₓ·X* + Pᵧ·Y* = {Px}×{xs_:.4g} + {Py}×{ys_:.4g} = {budget_check:.4g}"
        + (" = I ✓" if abs(budget_check-I)<0.01 else f"  (should be {I})")))
    return steps

def make_chart(prob, sol):
    Px,Py,I=prob["Px"],prob["Py"],prob["I"]; t=prob["utility_type"]
    xs_,ys_=sol["x_star"],sol["y_star"]; x_int,y_int=sol["x_int"],sol["y_int"]
    pad=0.28
    x_max=max(x_int,xs_,1)*(1+pad); y_max=max(y_int,ys_,1)*(1+pad)
    x_max=max(x_max,5); y_max=max(y_max,5)

    fig,ax=plt.subplots(figsize=(8,5.8))
    fig.patch.set_facecolor(CLR["bg"]); ax.set_facecolor(CLR["bg"])

    # Budget line
    ax.plot([0,x_int],[y_int,0],color=CLR["budget"],lw=2.8,label="Budget line",zorder=4)
    ax.fill([0,0,x_int],[0,y_int,0],color=CLR["feasible"],alpha=0.20,label="Feasible set")
    ax.annotate(f"I/Pₓ={x_int:.4g}",xy=(x_int,0),xytext=(0,-22),
                textcoords="offset points",fontsize=9,color=CLR["budget"],ha="center",
                bbox=dict(boxstyle="round,pad=0.22",fc="#f0f0f0",ec=CLR["budget"],alpha=0.85),
                arrowprops=dict(arrowstyle="-",color=CLR["budget"],lw=0.8,alpha=0.6))
    ax.annotate(f"I/Pᵧ={y_int:.4g}",xy=(0,y_int),xytext=(-52,0),
                textcoords="offset points",fontsize=9,color=CLR["budget"],va="center",
                bbox=dict(boxstyle="round,pad=0.22",fc="#f0f0f0",ec=CLR["budget"],alpha=0.85),
                arrowprops=dict(arrowstyle="-",color=CLR["budget"],lw=0.8,alpha=0.6))

    xs_arr=np.linspace(1e-3,x_max,700)
    styles=[(CLR["ic_faint"],0.40,1.4,"--"),(CLR["ic_faint"],0.55,1.5,"--"),(CLR["ic_opt"],0.95,2.5,"-")]

    if t=="cobb_douglas":
        af,bf=float(prob["alpha"]),float(prob["beta"])
        U_=xs_**af*ys_**bf
        for i,u in enumerate([0.70*U_,0.85*U_,U_]):
            ys=np.power(u/np.power(xs_arr,af),1/bf)
            ys=np.where(np.isfinite(ys)&(ys>=0)&(ys<=y_max),ys,np.nan)
            c,a,lw,ls=styles[i]; lbl="Optimal IC" if i==2 else ("Indifference curves" if i==0 else None)
            ax.plot(xs_arr,ys,color=c,alpha=a,lw=lw,ls=ls,label=lbl,zorder=2)
        # tangent
        half=min(x_max,y_max)*0.18; slope=-Px/Py
        x0_=max(xs_-half,0.01); x1_=min(xs_+half,x_max)
        ax.plot([x0_,x1_],[ys_+slope*(x0_-xs_),ys_+slope*(x1_-xs_)],
                color=CLR["tangent"],lw=1.6,ls="--",alpha=0.80,zorder=3,label="Tangent at optimum")
        title=f"Cobb-Douglas  U = X^{float(prob['alpha']):.1f}·Y^{float(prob['beta']):.1f}"

    elif t=="perfect_substitutes":
        a,b=prob["a"],prob["b"]; U_=a*xs_+b*ys_
        for i,u in enumerate([0.70*U_,0.85*U_,U_]):
            ys=(u-a*xs_arr)/b; ys=np.where((ys>=0)&(ys<=y_max),ys,np.nan)
            c,al,lw,ls=styles[i]; lbl="Optimal IC" if i==2 else ("Indifference curves" if i==0 else None)
            ax.plot(xs_arr,ys,color=c,alpha=al,lw=lw,ls=ls,label=lbl,zorder=2)
        title=f"Perfect Substitutes  U = {a}X + {b}Y"

    elif t=="perfect_complements":
        a,b=prob["a"],prob["b"]; U_=min(xs_/a,ys_/b)
        for i,u in enumerate([0.70*U_,0.85*U_,U_]):
            xk,yk=a*u,b*u; c,al,lw,ls=styles[i]
            lbl="Optimal IC" if i==2 else ("Indifference curves" if i==0 else None)
            if xk<=x_max and yk<=y_max:
                h,=ax.plot([xk,x_max],[yk,yk],color=c,alpha=al,lw=lw,ls=ls,zorder=2)
                ax.plot([xk,xk],[yk,y_max],color=c,alpha=al,lw=lw,ls=ls,zorder=2)
                if lbl: h.set_label(lbl)
        title=f"Perfect Complements  U = min(X/{a}, Y/{b})"

    else:
        U_=xs_
        for i,u in enumerate([0.65*U_,0.82*U_,U_]):
            if 0<=u<=x_max:
                c,al,lw,ls=styles[i]; lbl="Optimal IC" if i==2 else ("Indifference curves" if i==0 else None)
                ax.plot([u,u],[0,y_max],color=c,alpha=al,lw=lw,ls=ls,label=lbl,zorder=2)
        title="Neutral Good  U = X  (Y neutral)"

    ax.scatter([xs_],[ys_],color=CLR["optimum"],s=100,zorder=6,
               label=f"Optimum  (X*={xs_:.3g}, Y*={ys_:.3g})")
    ax.annotate(f"(X*,Y*)\n=({xs_:.3g},{ys_:.3g})",xy=(xs_,ys_),
                xytext=(14,10),textcoords="offset points",fontsize=9,
                color=CLR["optimum"],fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.35",fc="white",ec=CLR["optimum"],alpha=0.90))

    ax.set_xlim(0,x_max); ax.set_ylim(0,y_max)
    ax.set_xlabel("Quantity of X",labelpad=8); ax.set_ylabel("Quantity of Y",labelpad=8)
    ax.set_title(title,pad=10,color=CLR["accent"])
    ax.legend(loc="upper right",frameon=True,fancybox=True,
              framealpha=0.92,edgecolor="#dddddd",fontsize=9)
    plt.tight_layout(pad=1.6)
    return fig

# ─────────────────────────────── UI ───────────────────────────────────────────
st.title("Practice Problem Generator")
st.caption("A random consumer choice problem is generated each time — try to solve it yourself before checking the answer.")

# Init session state
if "pp_problem" not in st.session_state:
    st.session_state.pp_problem  = gen_problem(np.random.default_rng())
    st.session_state.pp_attempts = 0
    st.session_state.pp_show_sol = False
    st.session_state.pp_show_ex  = False

prob = st.session_state.pp_problem
sol  = solve(prob)
t    = prob["utility_type"]
Px,Py,I = prob["Px"],prob["Py"],prob["I"]

col_prob, col_ans = st.columns([1.3, 1])

with col_prob:
    st.subheader("Your Problem")
    st.info(f"""
**Utility function:** {util_str(prob)}  
**Pₓ = {Px}**  |  **Pᵧ = {Py}**  |  **I = {I}**
""")
    st.markdown(METHOD_BLURB[t])
    st.markdown("**Your tasks:** Write the budget constraint. Find the intercepts and slope. Then find (X\*, Y\*) and verify budget exhaustion.")

    if st.button("Show worked example for this type", key="ex_btn"):
        st.session_state.pp_show_ex = not st.session_state.pp_show_ex

    if st.session_state.pp_show_ex:
        ex = WORKED_EXAMPLES[t]
        st.markdown(f"##### {ex['title']}")
        st.markdown(f"*Given: {ex['given']}*")
        for sname,sbody in ex["steps"]:
            with st.expander(sname):
                st.markdown(sbody)
        st.success(f"**Answer: {ex['answer']}**")

with col_ans:
    st.subheader("Check Your Answer")
    x_input = st.text_input("X\* =", placeholder="e.g. 10  or  25/2")
    y_input = st.text_input("Y\* =", placeholder="e.g. 5  or  0")

    def try_parse(s):
        s=s.strip()
        if not s: return None
        try: return float(s)
        except:
            try:
                parts=s.split("/")
                return float(parts[0])/float(parts[1])
            except: return None

    c1,c2,c3 = st.columns(3)
    with c1:
        check = st.button("Check Answer", type="primary")
    with c2:
        show_sol = st.button("Show Solution")
    with c3:
        new_prob = st.button("New Problem")

    if check:
        st.session_state.pp_attempts += 1
        xv = try_parse(x_input); yv = try_parse(y_input)
        if xv is None or yv is None:
            st.error("Could not parse your answer. Try formats like 10, 5.5, or 25/2.")
        else:
            xok = abs(xv-sol["x_star"])<0.02; yok = abs(yv-sol["y_star"])<0.02
            if xok and yok:
                st.success(f"✅ **Correct!** (X\*, Y\*) = ({xv:.4g}, {yv:.4g})\nBudget check: {Px}×{xv:.4g} + {Py}×{yv:.4g} = {Px*xv+Py*yv:.4g}")
            else:
                st.warning("Not quite — try again.")
                if st.session_state.pp_attempts >= 2:
                    st.info(f"**Hint:** X\* = {sol['x_star']:.4g}" if not xok else f"**Hint:** Y\* = {sol['y_star']:.4g}")

    if show_sol:
        st.session_state.pp_show_sol = True
        if st.session_state.pp_attempts < 1:
            st.warning("Please attempt at least once before viewing the solution.")
            st.session_state.pp_show_sol = False

    if new_prob:
        st.session_state.pp_problem  = gen_problem(np.random.default_rng())
        st.session_state.pp_attempts = 0
        st.session_state.pp_show_sol = False
        st.session_state.pp_show_ex  = False
        st.rerun()

if st.session_state.pp_show_sol:
    st.markdown("---")
    st.subheader("Full Solution")
    steps = solution_steps(prob, sol)
    for sname, sbody in steps:
        with st.expander(sname, expanded=True):
            st.markdown(sbody)

    fig = make_chart(prob, sol)
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

    st.markdown("#### What does this graph tell you?")
    for q in GRAPH_QUESTIONS.get(t,[]):
        st.markdown(f"- {q}")
