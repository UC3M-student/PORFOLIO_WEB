import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from scipy.optimize import minimize
from scipy.cluster import hierarchy as sch
import requests
import json
import altair as alt
import matplotlib.pyplot as plt
import pandas_datareader.data as web
import datetime
from streamlit.components.v1 import html

# ─────────────────────────────────── CONFIG & GLOBAL STYLE ────────────────────────────────────

st.set_page_config(
    page_title="Portfolio Optimizer Pro",
    page_icon="Portfolio Optimizer AI 💼",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Updated Custom CSS for a polished, modern look
CUSTOM_CSS = """
<style>
/* Global Styles */
body {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    background: linear-gradient(135deg, #f3f4f6 0%, #e5e7eb 100%);
    color: #1f2937;
}

/* Header Hero Section */
.hero {
    text-align: center;
    padding: 3rem 2rem;
    background: linear-gradient(135deg, #1e40af 0%, #3b82f6 100%);
    border-radius: 12px;
    box-shadow: 0 6px 20px rgba(0, 0, 0, 0.15);
    animation: fadeIn 1s ease-in;
}
.hero h1 {
    font-size: 2.5rem;
    font-weight: 700;
    color: #ffffff;
    margin-bottom: 0.75rem;
}
.hero p {
    font-size: 1.1rem;
    color: #e5e7eb;
    margin-bottom: 1.5rem;
}
.hero .cta-button {
    background: #ffffff;
    color: #1e40af;
    font-weight: 600;
    border-radius: 8px;
    padding: 0.75rem 2rem;
    transition: all 0.3s ease;
    text-decoration: none;
}
.hero .cta-button:hover {
    background: #1e40af;
    color: #ffffff;
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.2);
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background: #1e40af;
    color: #ffffff;
    padding: 2rem;
    border-right: 2px solid #e5e7eb;
}
section[data-testid="stSidebar"] label, section[data-testid="stSidebar"] span {
    color: #ffffff;
    font-weight: 500;
}
section[data-testid="stSidebar"] .stButton > button {
    background: #ffffff;
    color: #1e40af;
    border-radius: 8px;
    font-weight: 600;
    transition: all 0.3s ease;
}
section[data-testid="stSidebar"] .stButton > button:hover {
    background: #1e40af;
    color: #ffffff;
}

/* Cards for Metrics */
.stMetric {
    background: #ffffff;
    border-radius: 10px;
    padding: 1.25rem;
    box-shadow: 0 3px 10px rgba(0, 0, 0, 0.1);
    transition: transform 0.2s ease;
}
.stMetric:hover {
    transform: translateY(-3px);
}

/* Tabs */
div[data-testid="stTabs"] button {
    background: #ffffff;
    border: 1px solid #d1d5db;
    border-radius: 8px 8px 0 0;
    padding: 0.75rem 1.5rem;
    color: #1e40af;
    font-weight: 500;
}
div[data-testid="stTabs"] button:hover {
    background: #eff6ff;
}

/* Charts */
div[role="graphics-document"] {
    background: #ffffff;
    border-radius: 10px;
    padding: 1.5rem;
    box-shadow: 0 3px 10px rgba(0, 0, 0, 0.05);
}

/* DataFrame Styling */
.dataframe {
    border: 1px solid #e2e8f0;
    border-radius: 10px;
    overflow: hidden;
}
.dataframe tbody tr:hover {
    background: #eff6ff;
}
.dataframe th, .dataframe td {
    padding: 0.75rem;
    border-bottom: 1px solid #e2e8f0;
}

/* Animations */
@keyframes fadeIn {
    from { opacity: 0; transform: translateY(10px); }
    to { opacity: 1; transform: translateY(0); }
}

/* Mobile Responsiveness */
@media (max-width: 768px) {
    .hero h1 { font-size: 1.8rem; }
    .hero p { font-size: 1rem; }
    .stMetric { padding: 0.75rem; }
    div[role="graphics-document"] { padding: 1rem; }
    .dataframe th, .dataframe td { padding: 0.5rem; }
}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# ─────────────────────────────────────────── HELPERS ───────────────────────────────────────────


def interest_rates_today_usa():
    try:
        end = datetime.datetime.now()
        start = end - datetime.timedelta(days=7)
        df = web.DataReader('DGS10', 'fred', start, end).dropna()
        return float(df.iloc[-1].values[0]) / 100
    except Exception as e:
        st.warning(f"Failed to fetch interest rate: {
                   e}. Using default rate of 3%.")
        return 0.03


RF = interest_rates_today_usa() / 52.1429  # Weekly risk-free rate
ROLL_WINDOW = 26  # Weeks (≈ six months)


@st.cache_data(show_spinner=False)
def yahoo_timeseries(tickers: list[str], period: str = "2y", interval: str = "1wk") -> pd.DataFrame:
    try:
        df = yf.download(tickers, period=period, interval=interval, threads=True)[
            "Close"].dropna()
        return df
    except Exception as e:
        st.error(f"Error fetching price data: {e}")
        return pd.DataFrame()

# ---------- ALLOCATION ENGINES ---------------------------------------------------------------


def portfolio_minimum_variance(data: pd.DataFrame) -> pd.Series:
    r = data.pct_change().dropna()
    n = r.shape[1]
    cov = np.cov(r.T)
    A = np.zeros((n + 1, n + 1))
    A[:-1, :-1] = 2 * cov
    A[-1, :-1] = 1
    A[:-1, -1] = 1
    b = np.zeros(n + 1)
    b[-1] = 1
    try:
        z = np.linalg.solve(A, b)
        return pd.Series(z[:-1], index=data.columns, name="Minimum Variance")
    except np.linalg.LinAlgError:
        st.warning("Matrix inversion failed → using equal weights")
        return equal_weight_portfolio(data).rename("Minimum Variance")


def portfolio_risk_parity(data: pd.DataFrame) -> pd.Series:
    r = data.pct_change().dropna()
    cov, corr = r.cov(), r.corr()

    def ivp(c):
        inv = 1. / np.diag(c)
        return inv / inv.sum()

    def cluster_var(c, items):
        sub = c.loc[items, items]
        w = ivp(sub).reshape(-1, 1)
        return np.sqrt((w.T @ sub.values @ w)[0, 0])

    dist = np.sqrt(0.5 * (1 - corr))
    link = sch.linkage(dist, 'single')

    def quasi_diag(link_matrix):
        link_matrix = link_matrix.astype(int)
        sort_ix = pd.Series([link_matrix[-1, 0], link_matrix[-1, 1]])
        n_items = link_matrix[-1, 3]
        while sort_ix.max() >= n_items:
            sort_ix.index = range(0, sort_ix.shape[0] * 2, 2)
            df0 = sort_ix[sort_ix >= n_items]
            i, j = df0.index, df0.values - n_items
            sort_ix[i] = link_matrix[j, 0]
            sort_ix = pd.concat(
                [sort_ix, pd.Series(link_matrix[j, 1], index=i + 1)]).sort_index()
            sort_ix.index = range(sort_ix.shape[0])
        return sort_ix.tolist()

    sort_ix = corr.index[quasi_diag(link)].tolist()
    w = pd.Series(1.0, index=sort_ix)
    clusters = [sort_ix]
    while clusters:
        clusters = [i[j:k] for i in clusters for j, k in (
            (0, len(i) // 2), (len(i) // 2, len(i))) if len(i) > 1]
        for i in range(0, len(clusters), 2):
            c0, c1 = clusters[i], clusters[i + 1]
            v0, v1 = cluster_var(cov, c0), cluster_var(cov, c1)
            alpha = 1 - v0 / (v0 + v1)
            w[c0] *= alpha
            w[c1] *= 1 - alpha
    return w.rename("Risk Parity")


def markowitz_portfolio(data: pd.DataFrame, risk_aversion_lambda: float = 1.) -> pd.Series:
    r = data.pct_change().dropna()
    mu, cov = r.mean().values, r.cov().values
    n = len(mu)

    def utility(w):
        return -(w @ mu - risk_aversion_lambda / 2 * w @ cov @ w)

    res = minimize(utility, np.full(n, 1 / n), method='SLSQP', bounds=[(0, 1)] * n,
                   constraints={'type': 'eq', 'fun': lambda w: w.sum() - 1})
    return pd.Series(res.x if res.success else np.full(n, 1 / n), index=data.columns, name=f"Markowitz (λ={risk_aversion_lambda})")


def sharpe_max_weights(data: pd.DataFrame, rf: float = RF) -> pd.Series:
    r = data.pct_change().dropna()
    mean, cov = r.mean(), r.cov()
    n = len(mean)

    def neg_sharpe(w):
        return -((w @ mean - rf / 52.14) / np.sqrt(w @ cov.values @ w))

    res = minimize(neg_sharpe, np.full(n, 1 / n), method='SLSQP', bounds=[(0, 1)] * n,
                   constraints={'type': 'eq', 'fun': lambda w: w.sum() - 1})
    return pd.Series(res.x if res.success else np.full(n, 1 / n), index=data.columns, name="Maximum Sharpe")


def equal_weight_portfolio(data: pd.DataFrame) -> pd.Series:
    return pd.Series(1 / data.shape[1], index=data.columns, name="Equal Weight")


def max_diversification_portfolio(data: pd.DataFrame) -> pd.Series:
    r = data.pct_change().dropna()
    vol, cov = r.std(), r.cov().values
    n = len(vol)

    def diversification_ratio(w):
        return -((w @ vol) / np.sqrt(w @ cov @ w))

    res = minimize(diversification_ratio, np.full(n, 1 / n), method='SLSQP', bounds=[(0, 1)] * n,
                   constraints={'type': 'eq', 'fun': lambda w: w.sum() - 1})
    return pd.Series(res.x if res.success else np.full(n, 1 / n), index=data.columns, name="Maximum Diversification")


def momentum_weights(data: pd.DataFrame, lookback: int = 12) -> pd.Series:
    r = data.pct_change(periods=lookback).iloc[-1].clip(lower=0)
    return ((r / r.sum()) if r.sum() > 0 else equal_weight_portfolio(data)).rename("Momentum")


def min_correlation_portfolio(data: pd.DataFrame) -> pd.Series:
    corr = data.pct_change().dropna().corr().abs()
    w = 1 / corr.mean()
    return (w / w.sum()).rename("Minimum Correlation")


def inverse_volatility_portfolio(data: pd.DataFrame) -> pd.Series:
    vol = data.pct_change().dropna().std()
    w = 1 / vol
    return (w / w.sum()).rename("Inverse Volatility")


def equal_risk_contribution_portfolio(data: pd.DataFrame) -> pd.Series:
    r = data.pct_change().dropna()
    cov = r.cov().values
    n = len(data.columns)

    def portfolio_risk(w):
        return np.sqrt(w @ cov @ w)

    def risk_contribution(w):
        return w * (cov @ w) / portfolio_risk(w)

    def objective(w):
        return ((risk_contribution(w) - risk_contribution(w).mean()) ** 2).sum()

    res = minimize(objective, np.full(n, 1 / n), method='SLSQP', bounds=[(0, 1)] * n,
                   constraints={'type': 'eq', 'fun': lambda w: w.sum() - 1})
    return pd.Series(res.x if res.success else np.full(n, 1 / n), index=data.columns, name="Equal Risk Contribution")


def market_cap_weight_portfolio(tickers: list[str]) -> pd.Series:
    caps = [yf.Ticker(t).fast_info.get('marketCap', np.nan) for t in tickers]
    caps = pd.Series(caps, index=tickers, dtype='float64')
    if caps.isna().all():
        st.warning('Market-cap data unavailable → equal weights applied.')
        return pd.Series(1 / len(tickers), index=tickers, name='Market Cap')
    caps = caps.fillna(caps.median())
    return (caps / caps.sum()).rename('Market Cap')


def inverse_beta_portfolio(tickers: list[str]) -> pd.Series:
    betas = [yf.Ticker(t).fast_info.get('beta', np.nan) for t in tickers]
    betas = pd.Series(betas, index=tickers, dtype='float64')
    if betas.isna().all() or (betas <= 0).all():
        # st.warning('Beta estimates unavailable → equal weights applied.')
        return pd.Series(1 / len(tickers), index=tickers, name='Inverse Beta')
    betas = betas.replace(0, np.nan).fillna(betas.median())
    inv = 1 / betas.abs()
    return (inv / inv.sum()).rename('Inverse Beta')


def max_return_portfolio(data: pd.DataFrame) -> pd.Series:
    mean_r = data.pct_change().dropna().mean()
    winner = mean_r.idxmax()
    w = pd.Series(0, index=data.columns)
    w[winner] = 1.
    return w.rename('Maximum Return')

# ---------- PERFORMANCE HELPERS ---------------------------------------------------------------


def build_weights_dataframe(w_dict: dict[str, pd.Series], order: list[str]) -> pd.DataFrame:
    return pd.DataFrame({k: v.reindex(order) for k, v in w_dict.items()}).round(3)


def portfolio_returns(data: pd.DataFrame, w: pd.Series) -> pd.Series:
    returns = data.pct_change().dropna()
    return (returns @ w).rename(w.name)


def performance_metrics(series: pd.Series, rf: float = RF) -> dict:
    cum = (1 + series).cumprod()
    total_return = cum.iloc[-1] - 1
    ann_return = (1 + total_return) ** (52 / len(series)) - \
        1  # Weekly → annual
    ann_vol = series.std() * np.sqrt(52)
    sharpe = (ann_return - rf) / ann_vol if ann_vol != 0 else np.nan
    mdd = (cum / cum.cummax() - 1).min()
    return {
        'Total Return': total_return,
        'Ann. Return': ann_return,
        'Ann. Volatility': ann_vol,
        'Sharpe': sharpe,
        'Max Drawdown': mdd
    }


@st.cache_data(show_spinner=False)
def load_sp500_tickers() -> dict[str, str]:
    url = 'https://raw.githubusercontent.com/UC3M-student/yfinance_Portfolio/refs/heads/main/SP500_Tickers/SP500_Tickers.json'
    try:
        res = requests.get(url, timeout=10)
        res.raise_for_status()
        return json.loads(res.text)
    except Exception as e:
        st.error(f'Error loading ticker list: {e}')
        return {}

# ───────────────────────────────────────────── UI ─────────────────────────────────────────────


# Onboarding Hero Section with Interactive Tutorial
if "first_load" not in st.session_state:
    st.session_state["first_load"] = True
    st.markdown(
        """
        <div class="hero">
            <h1>Portfolio Optimizer Pro 💼</h1>
            <p>Build and analyze equity portfolios with advanced weighting strategies. Optimize your investments with data-driven insights.</p>
            <a href="#configure" class="cta-button">Start Optimizing</a>
        </div>
        """,
        unsafe_allow_html=True
    )
    # # Interactive Tutorial Modal
    # html("""
    # <script>
    #     if (!localStorage.getItem('tutorialShown')) {
    #         setTimeout(() => {
    #             alert('Welcome to Portfolio Optimizer Pro! Select 2–20 S&P 500 companies in the sidebar, adjust settings, and click "Run Analysis" to explore weighting strategies. Use the tabs to dive into performance, risk, and insights.');
    #             localStorage.setItem('tutorialShown', 'true');
    #         }, 1000);
    #     }
    # </script>
    # """)

# Sidebar with Enhanced UX
with st.sidebar:
    # st.image("https://via.placeholder.com/50.png?text=💼",
    #          width=50)  # Replace with your logo
    st.markdown("<h2 style='color: #ffffff; font-size: 1.5rem;'>Portfolio Optimizer Pro</h2>",
                unsafe_allow_html=True)
    st.markdown("<p style='color: #e5e7eb;'>Select S&P 500 companies to build and analyze your portfolio.</p>",
                unsafe_allow_html=True)
    st.divider()

    # Searchable Multiselect for Companies
    mapping = load_sp500_tickers()
    companies = sorted(list(mapping.keys()))  # Sort for better UX
    selected = st.multiselect(
        "Choose companies (2–20)",
        options=companies,
        default=["Apple Inc.", "Microsoft Corporation"],
        help="Select 2–20 S&P 500 companies. Type to search.",
        max_selections=20
    )

    # Advanced Options Expander
    with st.expander("⚙️ Advanced Settings"):
        period = st.selectbox("Data Period", [
                              "1y", "2y", "5y"], index=1, help="Choose the historical data period.")
        risk_aversion = st.slider("Risk Aversion (Markowitz)", 0.1, 5.0,
                                  1.0, 0.1, help="Adjust risk aversion for Markowitz optimization.")
        lookback = st.slider("Momentum Lookback (weeks)", 4, 52,
                             12, help="Set lookback period for momentum strategy.")

    run_button = st.button(
        "🚀 Run Analysis", type="primary", use_container_width=True)

    # Buy Me a Coffee Button
    st.markdown(
        """
        <div style="text-align: center; margin-top: 1.5rem;">
            <a href="https://www.buymeacoffee.com/freeinvestmenteducation" target="_blank">
                <img src="https://cdn.buymeacoffee.com/buttons/v2/default-yellow.png" 
                     alt="Support Us" 
                     style="height: 45px; width: auto;"/>
            </a>
        </div>
        """,
        unsafe_allow_html=True
    )

# Main Content
st.markdown("<div id='configure'></div>", unsafe_allow_html=True)
st.title('Portfolio Optimizer Pro')
st.markdown(
    'Analyze and optimize equity portfolios with cutting-edge weighting strategies. '
    '<span style="font-size:0.9rem; color:#6b7280;">For educational purposes only — not investment advice.</span>',
    unsafe_allow_html=True
)

if not run_button:
    st.info('Select companies and settings in the sidebar, then click **Run Analysis** to begin.')
    st.stop()

# Input Validation
if len(selected) < 2:
    st.warning("Please select at least **two companies** to build your portfolio.")
    st.stop()
if len(selected) > 20:
    st.warning("Please select up to 20 companies for optimal performance.")
    st.stop()

# Data Fetching with Progress Bar
with st.spinner('Fetching data and optimizing portfolio …'):
    progress = st.progress(0)
    progress.progress(10)
    tickers = [mapping[n] for n in selected]
    progress.progress(30)
    price_data = yahoo_timeseries(tickers, period=period)
    benchmark_data = yahoo_timeseries(['^GSPC'], period=period)
    progress.progress(60)
    if price_data.empty or benchmark_data.empty:
        st.error(
            'No historical data available for the selected tickers or benchmark.')
        progress.empty()
        st.stop()

    # Compute Weights
    weights = {
        'Minimum Variance': portfolio_minimum_variance(price_data),
        'Risk Parity': portfolio_risk_parity(price_data),
        'Markowitz': markowitz_portfolio(price_data, risk_aversion_lambda=risk_aversion),
        'Maximum Sharpe': sharpe_max_weights(price_data),
        'Equal Weight': equal_weight_portfolio(price_data),
        'Maximum Diversification': max_diversification_portfolio(price_data),
        'Momentum': momentum_weights(price_data, lookback=lookback),
        'Minimum Correlation': min_correlation_portfolio(price_data),
        'Inverse Volatility': inverse_volatility_portfolio(price_data),
        'Equal Risk Contribution': equal_risk_contribution_portfolio(price_data),
        'Market Cap': market_cap_weight_portfolio(tickers),
        'Inverse Beta': inverse_beta_portfolio(tickers),
        'Maximum Return': max_return_portfolio(price_data),
    }
    weights_df = build_weights_dataframe(weights, tickers)

    # Portfolio Returns
    ret_dict = {name: portfolio_returns(price_data, w)
                for name, w in weights.items()}
    portfolio_returns_df = pd.concat(ret_dict.values(), axis=1)

    # Align Benchmark Returns
    a = portfolio_returns_df.index
    bmk_returns = benchmark_data.iloc[:, 0].pct_change().dropna()
    common_idx = a.intersection(bmk_returns.index)
    portfolio_returns_df = portfolio_returns_df.loc[common_idx]
    bmk_returns = bmk_returns.loc[common_idx]

    # Analytics
    rolling_sharpe_df = ((portfolio_returns_df.rolling(ROLL_WINDOW).mean() * 52 - RF) /
                         (portfolio_returns_df.rolling(ROLL_WINDOW).std() * np.sqrt(52))).dropna()
    beta_vals = {
        col: np.cov(portfolio_returns_df[col], bmk_returns)[
            0, 1] / np.var(bmk_returns)
        if np.var(bmk_returns) != 0 else np.nan
        for col in portfolio_returns_df.columns
    }
    beta_df = pd.Series(beta_vals, name='Beta').to_frame()
    rolling_corr_df = pd.DataFrame({
        col: portfolio_returns_df[col].rolling(ROLL_WINDOW).corr(bmk_returns)
        for col in portfolio_returns_df.columns
    }).dropna()
    cum_returns_df = (1 + portfolio_returns_df).cumprod()
    drawdown_df = cum_returns_df.div(cum_returns_df.cummax()).subtract(1)
    progress.progress(100)
    progress.empty()

# Save Portfolio Config
# with st.sidebar:
#     if run_button and selected:
#         config = {"tickers": selected, "period": period,
#                   "risk_aversion": risk_aversion, "lookback": lookback}
#         config_json = json.dumps(config).encode("utf-8")
#         st.download_button(
#             "💾 Save Portfolio Config",
#             config_json,
#             file_name="portfolio_config.json",
#             mime="application/json",
#             use_container_width=True
#         )

# ───────────────────────────────────────── TABS ───────────────────────────────────────────

st.markdown("### Portfolio Analysis Dashboard")
tab_groups = st.tabs(
    ["📊 Core Analysis", "⚠️ Risk Insights", "🔍 Advanced Metrics"])

# Core Analysis
with tab_groups[0]:
    core_tabs = st.tabs(["Weights", "Performance", "Price History"])
    with core_tabs[0]:
        st.subheader("Portfolio Weights by Strategy")
        st.dataframe(
            weights_df.style.format(
                "{:.2%}").background_gradient(cmap="Blues"),
            use_container_width=True,
            height=min(480, 80 + 32 * len(tickers))
        )
        long_df = weights_df.T.reset_index().melt(
            id_vars='index', var_name='Ticker', value_name='Weight')
        chart = alt.Chart(long_df).mark_bar().encode(
            x=alt.X("index:N", title="Strategy", axis=alt.Axis(labelAngle=45)),
            y=alt.Y("Weight:Q", stack="normalize",
                    title="Weight", axis=alt.Axis(format="%")),
            color=alt.Color("Ticker:N", scale=alt.Scale(scheme="tableau10")),
            tooltip=["Ticker", alt.Tooltip("Weight", format=".1%")]
        ).properties(
            width="container",
            height=450,
            title=alt.Title("Portfolio Weights",
                            subtitle="Allocation across strategies")
        ).configure_view(stroke=None).configure_axis(
            labelFont="Inter", titleFont="Inter", titleFontSize=16, titleColor="#1e40af"
        ).configure_legend(labelFont="Inter", titleFont="Inter", titleColor="#1e40af")
        st.altair_chart(chart, use_container_width=True)
        st.markdown(
            '<span style="font-size:0.9rem; color:#6b7280;">Chart shows normalized weights of each ticker across strategies.</span>',
            unsafe_allow_html=True
        )
        csv = weights_df.to_csv().encode("utf-8")
        st.download_button("📥 Download Weights (CSV)", csv,
                           file_name="weights.csv", mime="text/csv", use_container_width=True)

    with core_tabs[1]:
        st.subheader("Portfolio Performance (Static Weights)")
        cum_df = (1 + portfolio_returns_df).cumprod().reset_index().melt(
            id_vars='Date', var_name='Strategy', value_name='Cumulative Return')
        selection = alt.selection_single(fields=['Strategy'], bind='legend')
        perf_chart = alt.Chart(cum_df).mark_line().encode(
            x="Date:T",
            y="Cumulative Return:Q",
            color="Strategy:N",
            opacity=alt.condition(selection, alt.value(1.0), alt.value(0.3)),
            tooltip=["Date:T", "Strategy", alt.Tooltip(
                "Cumulative Return:Q", format=".2f")]
        ).add_selection(selection).interactive().properties(
            height=450,
            title=alt.Title("Cumulative Returns",
                            subtitle="Performance across strategies")
        ).configure_view(stroke=None).configure_axis(
            labelFont="Inter", titleFont="Inter", titleFontSize=16, titleColor="#1e40af"
        ).configure_legend(labelFont="Inter", titleFont="Inter", titleColor="#1e40af")
        st.altair_chart(perf_chart, use_container_width=True)
        st.markdown("**Key Performance Metrics**")
        cols = st.columns(5)
        for i, strategy in enumerate(portfolio_returns_df.columns):
            metrics = performance_metrics(portfolio_returns_df[strategy])
            with cols[i % 5]:
                st.metric(f"{strategy} Ann. Return", f"{
                          metrics['Ann. Return']:.2%}")
                st.metric(f"{strategy} Sharpe", f"{metrics['Sharpe']:.2f}")
                st.metric(f"{strategy} Max Drawdown", f"{
                          metrics['Max Drawdown']:.2%}")

    with core_tabs[2]:
        st.subheader("Historical Weekly Closing Prices")
        price_long = price_data.reset_index().melt(
            id_vars='Date', var_name='Ticker', value_name='Price')
        price_chart = alt.Chart(price_long).mark_line().encode(
            x="Date:T",
            y="Price:Q",
            color="Ticker:N",
            tooltip=["Date:T", "Ticker", alt.Tooltip("Price:Q", format=".2f")]
        ).interactive().properties(
            height=450,
            title=alt.Title("Historical Prices",
                            subtitle=f"Weekly closing prices for {period}")
        ).configure_view(stroke=None).configure_axis(
            labelFont="Inter", titleFont="Inter", titleFontSize=16, titleColor="#1e40af"
        ).configure_legend(labelFont="Inter", titleFont="Inter", titleColor="#1e40af")
        st.altair_chart(price_chart, use_container_width=True)

# Risk Insights
with tab_groups[1]:
    risk_tabs = st.tabs(
        ["🔗 Correlation", "⚠️ Risk Overview", "📉 Strategy Drawdowns"])
    with risk_tabs[0]:
        st.subheader("Correlation Matrix (Returns)")
        corr_matrix = price_data.pct_change().dropna().corr().round(2)
        fig, ax = plt.subplots(
            figsize=(len(tickers) * 0.6 + 2, len(tickers) * 0.6 + 2))
        img = ax.imshow(corr_matrix.values, cmap='viridis', vmin=-1, vmax=1)
        ax.set_xticks(range(len(tickers)))
        ax.set_yticks(range(len(tickers)))
        ax.set_xticklabels(tickers, rotation=45, ha='right')
        ax.set_yticklabels(tickers)
        for i in range(len(tickers)):
            for j in range(len(tickers)):
                ax.text(
                    j, i, corr_matrix.iloc[i, j], ha='center', va='center', color='white', fontsize=8)
        cbar = plt.colorbar(img, ax=ax)
        cbar.ax.set_ylabel('Correlation', rotation=270, labelpad=15)
        st.pyplot(fig, clear_figure=True)

    with risk_tabs[1]:
        st.subheader("Risk Overview")
        st.markdown("**Rolling 12-Week Volatility (Tickers)**")
        roll_vol = price_data.pct_change().rolling(12).std() * np.sqrt(52)
        roll_long = roll_vol.reset_index().melt(
            id_vars='Date', var_name='Ticker', value_name='Volatility')
        vol_chart = alt.Chart(roll_long).mark_line().encode(
            x="Date:T",
            y="Volatility:Q",
            color="Ticker:N",
            tooltip=["Date:T", "Ticker", alt.Tooltip(
                "Volatility:Q", format=".2%")]
        ).interactive().properties(
            height=400,
            title=alt.Title(
                "Rolling Volatility", subtitle="12-week annualized volatility for each ticker")
        ).configure_view(stroke=None).configure_axis(
            labelFont="Inter", titleFont="Inter", titleFontSize=16, titleColor="#1e40af"
        ).configure_legend(labelFont="Inter", titleFont="Inter", titleColor="#1e40af")
        st.altair_chart(vol_chart, use_container_width=True)
        st.markdown("**Drawdown Chart (Equal Weight Portfolio)**")
        eq_series = portfolio_returns(
            price_data, equal_weight_portfolio(price_data))
        dd = (1 + eq_series).cumprod()
        drawdown = dd / dd.cummax() - 1
        dd_df = drawdown.reset_index().rename(columns={0: 'Drawdown'})
        dd_chart = alt.Chart(dd_df).mark_area(opacity=0.7).encode(
            x="Date:T",
            y=alt.Y("Drawdown:Q", axis=alt.Axis(format='%')),
            tooltip=["Date:T", alt.Tooltip("Drawdown:Q", format=".1%")]
        ).properties(
            height=250,
            title=alt.Title("Equal Weight Portfolio Drawdown",
                            subtitle="Maximum loss from peak")
        ).configure_view(stroke=None).configure_axis(
            labelFont="Inter", titleFont="Inter", titleFontSize=16, titleColor="#1e40af"
        )
        st.altair_chart(dd_chart, use_container_width=True)

    with risk_tabs[2]:
        st.subheader("Strategy Drawdowns")
        dd_long = drawdown_df.reset_index().melt(
            id_vars='Date', var_name='Strategy', value_name='Drawdown')
        dd_chart_all = alt.Chart(dd_long).mark_area(opacity=0.7).encode(
            x="Date:T",
            y=alt.Y("Drawdown:Q", axis=alt.Axis(format='%')),
            color="Strategy:N",
            tooltip=["Date:T", "Strategy", alt.Tooltip(
                "Drawdown:Q", format=".1%")]
        ).interactive().properties(
            height=450,
            title=alt.Title("Strategy Drawdowns",
                            subtitle="Maximum loss from peak for each strategy")
        ).configure_view(stroke=None).configure_axis(
            labelFont="Inter", titleFont="Inter", titleFontSize=16, titleColor="#1e40af"
        ).configure_legend(labelFont="Inter", titleFont="Inter", titleColor="#1e40af")
        st.altair_chart(dd_chart_all, use_container_width=True)
        st.dataframe(
            drawdown_df.round(3).style.format(
                "{:.2%}").background_gradient(cmap="Reds"),
            use_container_width=True,
            height=min(400, 80 + 32 * len(drawdown_df.tail(10)))
        )

# Advanced Metrics
with tab_groups[2]:
    insight_tabs = st.tabs(["📈 Rolling Sharpe", "📊 Market Beta",
                           "🔗 Rolling Correlation", "📉 Return Distribution"])
    with insight_tabs[0]:
        st.subheader("26-Week Rolling Sharpe Ratios (Annualized)")
        rs_long = rolling_sharpe_df.reset_index().melt(
            id_vars='Date', var_name='Strategy', value_name='Sharpe')
        rs_chart = alt.Chart(rs_long).mark_line().encode(
            x="Date:T",
            y="Sharpe:Q",
            color="Strategy:N",
            tooltip=["Date:T", "Strategy",
                     alt.Tooltip("Sharpe:Q", format=".2f")]
        ).interactive().properties(
            height=450,
            title=alt.Title("Rolling Sharpe Ratios",
                            subtitle="26-week annualized Sharpe ratios")
        ).configure_view(stroke=None).configure_axis(
            labelFont="Inter", titleFont="Inter", titleFontSize=16, titleColor="#1e40af"
        ).configure_legend(labelFont="Inter", titleFont="Inter", titleColor="#1e40af")
        st.altair_chart(rs_chart, use_container_width=True)
        st.dataframe(
            rolling_sharpe_df.round(
                3).style.background_gradient(cmap="Greens"),
            use_container_width=True,
            height=min(400, 80 + 32 * len(rolling_sharpe_df.tail(10)))
        )

    with insight_tabs[1]:
        st.subheader("CAPM Beta vs S&P 500")
        beta_chart = alt.Chart(beta_df.reset_index()).mark_bar().encode(
            x=alt.X("index:N", title="Strategy", sort="-y"),
            y=alt.Y("Beta:Q"),
            tooltip=["index", alt.Tooltip("Beta:Q", format=".2f")]
        ).properties(
            height=400,
            title=alt.Title(
                "Market Beta", subtitle="CAPM beta relative to S&P 500")
        ).configure_view(stroke=None).configure_axis(
            labelFont="Inter", titleFont="Inter", titleFontSize=16, titleColor="#1e40af"
        )
        st.altair_chart(beta_chart, use_container_width=True)
        st.dataframe(
            beta_df.round(3).style.background_gradient(cmap="Purples"),
            use_container_width=True,
            height=min(400, 80 + 32 * len(beta_df))
        )

    with insight_tabs[2]:
        st.subheader("26-Week Rolling Correlation with S&P 500")
        rc_long = rolling_corr_df.reset_index().melt(
            id_vars='Date', var_name='Strategy', value_name='Correlation')
        rc_chart = alt.Chart(rc_long).mark_line().encode(
            x="Date:T",
            y=alt.Y("Correlation:Q", scale=alt.Scale(domain=[-1, 1])),
            color="Strategy:N",
            tooltip=["Date:T", "Strategy", alt.Tooltip(
                "Correlation:Q", format=".2f")]
        ).interactive().properties(
            height=450,
            title=alt.Title("Rolling Correlation",
                            subtitle="26-week correlation with S&P 500")
        ).configure_view(stroke=None).configure_axis(
            labelFont="Inter", titleFont="Inter", titleFontSize=16, titleColor="#1e40af"
        ).configure_legend(labelFont="Inter", titleFont="Inter", titleColor="#1e40af")
        st.altair_chart(rc_chart, use_container_width=True)
        st.dataframe(
            rolling_corr_df.round(3).style.background_gradient(cmap="Blues"),
            use_container_width=True,
            height=min(400, 80 + 32 * len(rolling_corr_df.tail(10)))
        )

    with insight_tabs[3]:
        st.subheader("Weekly Return Distribution")
        returns_long = portfolio_returns_df.reset_index().melt(
            id_vars='Date', var_name='Strategy', value_name='Return')
        selection = alt.selection_single(fields=['Strategy'], bind='legend')
        hist_chart = alt.Chart(returns_long).mark_bar(opacity=0.7).encode(
            x=alt.X("Return:Q", bin=alt.Bin(
                maxbins=60), title="Weekly Return"),
            y=alt.Y("count()", title="Frequency"),
            color="Strategy:N",
            tooltip=["Strategy", alt.Tooltip("count()", title="Freq")]
        ).add_selection(selection).transform_filter(selection).properties(
            height=450,
            title=alt.Title("Return Distribution",
                            subtitle="Distribution of weekly returns by strategy")
        ).configure_view(stroke=None).configure_axis(
            labelFont="Inter", titleFont="Inter", titleFontSize=16, titleColor="#1e40af"
        ).configure_legend(labelFont="Inter", titleFont="Inter", titleColor="#1e40af")
        st.altair_chart(hist_chart, use_container_width=True)

# Footer with Professional Branding
st.markdown(
    """
    <div style="text-align: center; padding: 2rem 0; color: #6b7280;">
        <span>Portfolio Optimizer Pro by Free Investment Education | © 2025 | For educational purposes only.</span><br>
        <a href="https://www.investopedia.com/terms/p/portfolio-weight.asp" style="color: #1e40af;">Learn More</a> | 
        <a href="mailto:freeinvestmenteducation@gmail.com" style="color: #1e40af;">Contact Us</a>
    </div>
    """,
    unsafe_allow_html=True
)


# Inside the `with st.sidebar:` block, after the existing content; ; BuymeaCoffe
st.markdown(
    """
    <div style="text-align: center; margin-top: 1rem;">
        <a href="https://www.buymeacoffee.com/freeinvestmenteducation" target="_blank">
            <img src="https://cdn.buymeacoffee.com/buttons/v2/default-yellow.png" 
                 alt="Buy Me a Coffee" 
                 style="height: 45px; width: auto; margin: 0 auto;"/>
        </a>
    </div>
    """,
    unsafe_allow_html=True
)
