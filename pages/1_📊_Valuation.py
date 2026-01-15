import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np

# --- PAGE CONFIG ---
st.set_page_config(page_title="Valuation Analysis", layout="wide", page_icon="📊")

# --- CUSTOM UTILS ---
def format_large_number(num):
    if num is None or pd.isna(num):
        return "N/A"
    if num >= 1_000_000_000_000:
        return f"{num / 1_000_000_000_000:.2f}T"
    if num >= 1_000_000_000:
        return f"{num / 1_000_000_000:.2f}B"
    if num >= 1_000_000:
        return f"{num / 1_000_000:.2f}M"
    return f"{num:.2f}"

def format_ratio(num):
    if num is None or pd.isna(num):
        return "N/A"
    return f"{num:.2f}x"

# --- SIDEBAR: GLOBAL ROUTER ---
st.sidebar.header("Global Exchange Router 🌍")
region = st.sidebar.selectbox(
    "Select Region",
    ["US (Default)", "India (NSE)", "Europe", "Japan", "Hong Kong", "London"]
)

suffix_map = {
    "US (Default)": "",
    "India (NSE)": ".NS",
    "Europe": ".PA",
    "Japan": ".T",
    "Hong Kong": ".HK",
    "London": ".L"
}
suffix = suffix_map.get(region, "")

st.sidebar.info(f"Suffix: '{suffix}'")

# --- MAIN APP ---
st.title("📊 Comparable Company Analysis (Comps)")
st.markdown("### Automated Valuation Dashboard")

# Default tickers based on region for quick start
default_tickers = "AAPL, MSFT, GOOG, NVDA"
if region == "India (NSE)":
    default_tickers = "TCS, INFY, HCLTECH, WIPRO"
elif region == "Europe":
    default_tickers = "MC, OR, SAN, AIR" # LVMH, L'Oreal, Sanofi, Airbus
    
user_input = st.text_input("Enter Tickers (comma separated):", value=default_tickers)

if st.button("Analyze Valuation 🚀"):
    tickers_raw = [t.strip().upper() for t in user_input.split(",") if t.strip()]
    
    if not tickers_raw:
        st.warning("Please enter at least one ticker.")
        st.stop()
        
    # Append suffix
    tickers = [t + suffix if not t.endswith(suffix) else t for t in tickers_raw]
    
    st.write(f"Fetching data for: {', '.join(tickers)}...")
    
    comps_data = []
    
    progress_bar = st.progress(0)
    
    for i, ticker in enumerate(tickers):
        try:
            t = yf.Ticker(ticker)
            info = t.info
            
            # Extract Metrics using .get() to avoid KeyErrors
            data_point = {
                "Ticker": ticker,
                "Company": info.get("shortName", ticker),
                "Sector": info.get("sector", "N/A"),
                "Price": info.get("currentPrice", info.get("regularMarketPreviousClose", None)),
                "Market Cap": info.get("marketCap", None),
                "Enterprise Value": info.get("enterpriseValue", None),
                "Ex-Div Date": info.get("exDividendDate", None),
                # Valuation Ratios
                "Trailing P/E": info.get("trailingPE", None),
                "Forward P/E": info.get("forwardPE", None),
                "PEG Ratio": info.get("pegRatio", None),
                "Price/Book": info.get("priceToBook", None),
                "EV/Revenue": info.get("enterpriseToRevenue", None),
                "EV/EBITDA": info.get("enterpriseToEbitda", None),
                "Profit Margins": info.get("profitMargins", None),
            }
            comps_data.append(data_point)
            
        except Exception as e:
            st.error(f"Error fetching {ticker}: {e}")
        
        progress_bar.progress((i + 1) / len(tickers))
        
    if comps_data:
        df = pd.DataFrame(comps_data)
        
        # --- DISPLAY RAW DF ---
        # Format for display
        display_df = df.copy()
        
        # Formatting Columns
        display_df["Market Cap"] = display_df["Market Cap"].apply(format_large_number)
        display_df["Enterprise Value"] = display_df["Enterprise Value"].apply(format_large_number)
        
        ratio_cols = ["Trailing P/E", "Forward P/E", "PEG Ratio", "Price/Book", "EV/Revenue", "EV/EBITDA"]
        for col in ratio_cols:
            display_df[col] = display_df[col].apply(format_ratio)

        # Percentages
        if "Profit Margins" in display_df.columns:
            display_df["Profit Margins"] = display_df["Profit Margins"].apply(lambda x: f"{x:.1%}" if x and not pd.isna(x) else "N/A")
            
        st.subheader("Financial Metrics Table")
        st.dataframe(display_df, use_container_width=True)
        
        # --- VISUALIZATION ---
        st.subheader("Relative Valuation Comparison")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Drop N/A for plotting
            plot_df = df.dropna(subset=["EV/EBITDA"])
            if not plot_df.empty:
                st.bar_chart(plot_df.set_index("Ticker")["EV/EBITDA"])
                st.caption("EV / EBITDA")
            else:
                st.info("Not enough data for EV/EBITDA chart.")
                
        with col2:
             plot_df_pe = df.dropna(subset=["Forward P/E"])
             if not plot_df_pe.empty:
                 st.bar_chart(plot_df_pe.set_index("Ticker")["Forward P/E"])
                 st.caption("Forward P/E")
             else:
                 st.info("Not enough data for P/E chart.")

    else:
        st.error("No valid data found.")
