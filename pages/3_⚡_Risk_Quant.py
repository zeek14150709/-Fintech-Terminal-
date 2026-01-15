import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta

# --- Configuration & Styling ---
st.set_page_config(page_title="Global Quant Dashboard", layout="wide", page_icon="📈")

# Custom CSS for "Institutional" look
st.markdown("""
<style>
    .stApp {
        background-color: #0e1117;
        color: #fafafa;
    }
    .metric-card {
        background-color: #1e2127;
        padding: 20px;
        border-radius: 10px;
        border: 1px solid #30333d;
        text-align: center;
    }
    h1, h2, h3 {
        font-family: 'Roboto', sans-serif;
    }
</style>
""", unsafe_allow_html=True)

st.title(" ⭐ Institutional-Grade Portfolio Optimizer")
st.markdown("### Global Quant Dashboard | Monte Carlo Simulation Engine")

# --- 1. Global Exchange Router ---
st.sidebar.header("Global Exchange Router 🌍")

region = st.sidebar.selectbox(
    "Select Region",
    ["US (Default)", "India (NSE)", "Europe", "Japan"]
)

# Logic for suffixes
suffix_map = {
    "US (Default)": "",
    "India (NSE)": ".NS",
    "Europe": ".PA", 
    "Japan": ".T"
}
suffix = suffix_map[region]

default_benchmark = "^GSPC"
if region == "India (NSE)": default_benchmark = "^NSEI"
elif region == "Europe": default_benchmark = "^STOXX50E"
elif region == "Japan": default_benchmark = "^N225"

benchmark_ticker = st.sidebar.text_input("Benchmark Ticker", value=default_benchmark)

default_tickers = "AAPL, MSFT, GOOG, AMZN" if region == "US (Default)" else "RELIANCE, TCS, INFY, HDFCBANK"
if region == "Europe": default_tickers = "MC, OR, SAN, SAP" # LVMH, L'Oreal, Sanofi, SAP
if region == "Japan": default_tickers = "7203, 6758, 9984, 6861" # Toyota, Sony, Softbank, Keyence

ticker_input = st.sidebar.text_area("Enter Tickers (comma separated)", value=default_tickers, height=150)

# Date Range
end_date = datetime.today()
start_date = end_date - timedelta(days=365*3) # Default 3 years
selected_start = st.sidebar.date_input("Start Date", start_date)
selected_end = st.sidebar.date_input("End Date", end_date)

st.sidebar.markdown("---")
run_btn = st.sidebar.button("Run Simulation 🚀")

if run_btn:
    st.session_state['run'] = True

if st.session_state.get('run', False):
    # Parse Tickers
    tickers = [t.strip().upper() + suffix for t in ticker_input.split(",") if t.strip()]
    benchmark = benchmark_ticker.strip().upper() if benchmark_ticker else None

    if len(tickers) < 2:
        st.error("Please enter at least 2 valid tickers for optimization.")
    else:
        with st.spinner("Simulating Global Scenarios... 🌍"):
            try:
                # 1. Data Fetching
                combined_tickers = tickers + ([benchmark] if benchmark else [])
                
                # Use threads=True for faster download
                data_raw = yf.download(combined_tickers, start=selected_start, end=selected_end, progress=False)
                
                if data_raw.empty:
                    st.error("No data found! Please check your tickers.")
                    st.stop()

                # Handle Adj Close vs Close issue
                if 'Adj Close' in data_raw:
                    data_close = data_raw['Adj Close']
                elif 'Close' in data_raw:
                    data_close = data_raw['Close']
                else:
                    # Fallback if structure is flat (single level)
                    data_close = data_raw
                
                # Separate Portfolio vs Benchmark
                # If only one ticker was fetched, data_close is Series. If multiple, DataFrame.
                # However, we enforced len(tickers) >= 2, so it should be DataFrame.
                
                # Identify valid columns (tickers likely to be columns)
                # If Benchmark is in columns, separate it.
                valid_tickers = [t for t in tickers if t in data_close.columns]
                
                if len(valid_tickers) < 2:
                     st.error("Could not fetch data for enough tickers. Check spelling.")
                     st.stop()

                port_data = data_close[valid_tickers].dropna()
                
                bench_data = None
                if benchmark and benchmark in data_close.columns:
                     bench_data = data_close[[benchmark]].dropna()

                # 2. Simulation Engine
                log_returns = np.log(port_data / port_data.shift(1))
                
                num_portfolios = 10000
                num_assets = len(port_data.columns)
                risk_free_rate = 0.065

                # Vectorized Simulation
                weights = np.random.random((num_portfolios, num_assets))
                weights /= np.sum(weights, axis=1)[:, np.newaxis]
                
                mean_ret = log_returns.mean() * 252
                cov_mat = log_returns.cov() * 252
                
                # Metrics
                port_returns = np.sum(weights * mean_ret.values, axis=1)
                port_volatility = np.sqrt(np.einsum('ij,ji->i', np.dot(weights, cov_mat.values), weights.T))
                sharpe_ratios = (port_returns - risk_free_rate) / port_volatility
                
                # Optimal Portfolios
                max_sharpe_idx = np.argmax(sharpe_ratios)
                min_vol_idx = np.argmin(port_volatility)
                
                max_sharpe_ret = port_returns[max_sharpe_idx]
                max_sharpe_vol = port_volatility[max_sharpe_idx]
                max_sharpe_sr = sharpe_ratios[max_sharpe_idx]
                
                # Top Section: Metrics
                st.markdown("### 🏆 Portfolio Metrics")
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Max Sharpe Ratio", f"{max_sharpe_sr:.2f}", help="Risk-Free Rate: 6.5%")
                col2.metric("Exp. Annual Return", f"{max_sharpe_ret:.1%}")
                col3.metric("Annual Volatility", f"{max_sharpe_vol:.1%}")
                
                # Calculate VaR 95% (Monthly approximation: Mean - Z*Vol/Sqrt(12) or simply historical)
                # Using Historical VaR of the Optimal Portfolio
                opt_weights = weights[max_sharpe_idx]
                opt_daily_rets = (log_returns * opt_weights).sum(axis=1)
                var_95_daily = np.percentile(opt_daily_rets, 5)
                # var_95_monthly = var_95_daily * np.sqrt(21) 
                
                col4.metric("VaR 95% (Daily)", f"{var_95_daily:.2%}", help="Worst 5% daily scenario")

                # 3. Visualizations
                st.markdown("---")
                
                col_chart1, col_chart2 = st.columns([2, 1])
                
                with col_chart1:
                    # Chart 1: Efficient Frontier
                    st.subheader("Efficient Frontier Scatter")
                    fig_ef = px.scatter(
                        x=port_volatility, y=port_returns, color=sharpe_ratios,
                        color_continuous_scale='plasma', 
                        labels={'x': 'Expected Volatility', 'y': 'Expected Return', 'color': 'Sharpe Ratio'},
                        # title="Efficient Frontier 🌌" 
                    )
                    # Add markers
                    fig_ef.add_trace(go.Scatter(x=[max_sharpe_vol], y=[max_sharpe_ret], mode='markers', marker=dict(color='gold', size=18, symbol='star'), name='Max Sharpe ⭐'))
                    fig_ef.add_trace(go.Scatter(x=[port_volatility[min_vol_idx]], y=[port_returns[min_vol_idx]], mode='markers', marker=dict(color='cyan', size=15, symbol='diamond'), name='Min Volatility 💎'))
                    
                    st.plotly_chart(fig_ef, use_container_width=True)
                
                with col_chart2:
                    # Chart 2: Optimal Allocation
                    st.subheader("Optimal Allocation")
                    # Filter out tiny weights for cleaner chart
                    clean_weights = {k:v for k,v in zip(port_data.columns, opt_weights) if v > 0.01}
                    
                    fig_donut = px.pie(
                        names=list(clean_weights.keys()), 
                        values=list(clean_weights.values()), 
                        hole=0.5
                    )
                    fig_donut.update_layout(showlegend=False)
                    st.plotly_chart(fig_donut, use_container_width=True)
                
                # Chart 3: Historical Performance
                st.subheader("🏁 Historical Performance Comparison")
                
                # Normalize Prices (Base 100)
                norm_prices = port_data / port_data.iloc[0] * 100
                
                # Optimized Portfolio Value
                opt_port_val = (norm_prices * opt_weights).sum(axis=1)
                
                # Equal Weight
                eq_weights = np.array([1/num_assets]*num_assets)
                eq_port_val = (norm_prices * eq_weights).sum(axis=1)
                
                perf_df = pd.DataFrame({
                    "Optimized Portfolio ⭐": opt_port_val,
                    "Equal Weight Portfolio": eq_port_val
                })
                
                if bench_data is not None and not bench_data.empty:
                    bench_norm = bench_data / bench_data.iloc[0] * 100
                    # Handle if benchmark has different length (join intersection)
                    # For simplicity, assign directly (index matches due to same download)
                    perf_df["Benchmark (" + benchmark + ")"] = bench_norm
                
                fig_perf = px.line(perf_df)
                fig_perf.update_layout(yaxis_title="Normalized Value (Base 100)")
                st.plotly_chart(fig_perf, use_container_width=True)

            except Exception as e:
                st.error(f"An error occurred during simulation: {e}")
                st.exception(e)

