import streamlit as st

st.set_page_config(
    page_title="FinTech Master Terminal",
    page_icon="🏦",
    layout="wide"
)

st.title("🏦 Institutional Financial Terminal")
st.markdown("---")

col1, col2 = st.columns(2)

with col1:
    st.image("https://cdn-icons-png.flaticon.com/512/1087/1087815.png", width=150)
    st.markdown("### Welcome, Analyst.")
    st.info("""
    This terminal aggregates real-time market data, news sentiment, 
    and quantitative risk models into a single workspace.
    """)

with col2:
    st.markdown("### 🚀 Capabilities")
    st.markdown("""
    * **📊 Valuation:** Automated Comps & Multiples Analysis.
    * **📰 News Intel:** AI-powered Sentiment & Keyword Extraction.
    * **⚡ Risk Quant:** Monte Carlo Simulations & Efficient Frontiers.
    """)

st.success("👈 Select a module from the sidebar to begin.")