import streamlit as st

st.set_page_config(
    page_title="Week 2 — Consumer Theory",
    page_icon="📈",
    layout="wide",
)

st.title("📈 Week 2 — Consumer Theory: Interactive Tools")
st.markdown("---")

st.markdown("""
Select a tool from the **sidebar on the left** to get started.

| Tool | What it shows |
|------|--------------|
| **1 · Utility Surface & IC Map** | How a 3-D utility surface relates to 2-D indifference curves |
| **2 · IC Shape Explorer** | Compare indifference curve shapes side-by-side |
| **3 · Budget Line** | How prices and income determine the budget set |
| **4 · Comparing Budget Lines** | See two budget lines at once to compare shifts |
| **5 · Optimal Bundle** | Find the utility-maximising bundle for different preferences |
| **6 · Optimisation Demo** | Step-by-step bang-for-the-buck reallocation to the optimum |
| **7 · Practice Problems** | Randomly generated problems with hints and solutions |
""")

st.info("👈 Use the navigation panel on the left to choose a tool.")
