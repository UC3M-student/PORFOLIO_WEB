import time
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
import plotly.express as px

# ─────────────────────────────────── Sitemap ────────────────────────────────────
# Mejor página hasta ahora

# ─────────────────────────────────── CONFIG & GLOBAL STYLE ────────────────────────────────────
st.set_page_config(
    page_title="Portfolio Optimizer AI",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="collapsed"
)

CUSTOM_CSS = """
<style>
body {
    font-family: 'Poppins', sans-serif;
    background: linear-gradient(135deg, #ff6b6b, #4ecdc4, #45b7d1);
    color: #ffffff;
    margin: 0;
    padding: 0;
}
.hero {
    text-align: center;
    padding: 5rem 2rem;
    background: linear-gradient(135deg, #ff4e50, #f9d423);
    border-radius: 20px;
    box-shadow: 0 10px 30px rgba(0, 0, 0, 0.3);
    animation: heroFade 1.5s ease-in-out;
    margin: 2rem 1rem;
}
.hero h1 {
    font-size: 3.5rem;
    font-weight: 900;
    color: #ffffff;
    text-shadow: 0 3px 6px rgba(0, 0, 0, 0.2);
    margin-bottom: 1.5rem;
}
.hero p {
    font-size: 1.4rem;
    color: #f1f1f1;
    margin-bottom: 2.5rem;
}
.hero .cta-button {
    background: #f9d423;
    color: #ff4e50;
    font-weight: 700;
    border-radius: 50px;
    padding: 1rem 3rem;
    text-decoration: none;
    font-size: 1.2rem;
    transition: all 0.3s ease;
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.2);
}
.hero .cta-button:hover {
    background: #ff4e50;
    color: #ffffff;
    transform: scale(1.05);
    box-shadow: 0 6px 20px rgba(0, 0, 0, 0.3);
}
.stMetric {
    background: rgba(255, 255, 255, 0.95);
    border-radius: 15px;
    padding: 1.8rem;
    box-shadow: 0 6px 20px rgba(0, 0, 0, 0.15);
    transition: transform 0.3s ease, box-shadow 0.3s ease;
}
.stMetric:hover {
    transform: translateY(-5px);
    box-shadow: 0 8px 25px rgba(0, 0, 0, 0.2);
}
div[data-testid="stTabs"] button {
    background: #ffffff;
    border: 2px solid #ff6b6b;
    border-radius: 15px 15px 0 0;
    padding: 1rem 2.5rem;
    color: #ff4e50;
    font-weight: 700;
    transition: all 0.3s ease;
}
div[data-testid="stTabs"] button:hover {
    background: #ff6b6b;
    color: #ffffff;
    transform: translateY(-2px);
}
div[role="graphics-document"] {
    background: rgba(255, 255, 255, 0.9);
    border-radius: 15px;
    padding: 2.5rem;
    box-shadow: 0 6px 20px rgba(0, 0, 0, 0.15);
}
.dataframe {
    border: 2px solid #4ecdc4;
    border-radius: 15px;
    overflow: hidden;
    background: rgba(255, 255, 255, 0.95);
}
.dataframe tbody tr:hover {
    background: #f1f1f1;
    transition: background 0.2s ease;
}
.dataframe th, .dataframe td {
    padding: 1.2rem;
    border-bottom: 1px solid #e2e8f0;
    font-size: 1rem;
    color: #2d3436;
}
@keyframes heroFade {
    from { opacity: 0; transform: translateY(20px); }
    to { opacity: 1; transform: translateY(0); }
}
@keyframes slideIn {
    from { opacity: 0; transform: translateX(-15px); }
    to { opacity: 1; transform: translateX(0); }
}
.st-expander {
    background: rgba(255, 255, 255, 0.9);
    border-radius: 15px;
    box-shadow: 0 4px 15px rgba(0, 0, 0, 0.15);
    animation: slideIn 0.6s ease-in-out;
}
.stButton>button {
    background: #ff4e50;
    color: #ffffff;
    border-radius: 50px;
    padding: 1rem 2.5rem;
    font-weight: 700;
    font-size: 1.1rem;
    transition: all 0.3s ease;
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.2);
}
.stButton>button:hover {
    background: #f9d423;
    color: #ff4e50;
    transform: scale(1.05);
    box-shadow: 0 6px 20px rgba(0, 0, 0, 0.3);
}
h2, h3 {
    color: #ff6b6b;
    font-weight: 800;
    margin-top: 2rem;
    text-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
}
.stSelectbox, .stSlider, .stNumberInput {
    background: rgba(255, 255, 255, 0.95);
    border-radius: 10px;
    padding: 0.5rem;
    box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
}
@media (max-width: 768px) {
    .hero h1 { font-size: 2.5rem; }
    .hero p { font-size: 1.1rem; }
    .stMetric { padding: 1.2rem; }
    div[role="graphics-document"] { padding: 1.5rem; }
    .dataframe th, .dataframe td { padding: 0.8rem; font-size: 0.9rem; }
    .stButton>button { padding: 0.8rem 1.5rem; font-size: 1rem; }
}
/* Pop-up Styling */
.popup {
    position: fixed;
    top: 100px;
    right: 20px;
    background: linear-gradient(135deg, #ff4e50, #f9d423);
    color: #ffffff;
    padding: 2rem;
    border-radius: 15px;
    box-shadow: 0 10px 30px rgba(0, 0, 0, 0.3);
    z-index: 1000;
    max-width: 400px;
    width: 90%;
    text-align: center;
    font-size: 1.1rem;
    font-weight: 600;
    animation: fadeIn 0.5s ease-in-out;
}
.popup button {
    background: #ffffff;
    color: #ff4e50;
    border: none;
    border-radius: 50px;
    padding: 0.8rem 2rem;
    font-weight: 700;
    font-size: 1rem;
    margin-top: 1.5rem;
    cursor: pointer;
    transition: all 0.3s ease;
}
.popup button:hover {
    background: #ff6b6b;
    color: #ffffff;
    transform: scale(1.05);
}
@keyframes fadeIn {
    from { opacity: 0; transform: translateY(-20px); }
    to { opacity: 1; transform: translateY(0); }
}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# Load custom fonts
st.markdown("""
<link href="https://fonts.googleapis.com/css2?family=Poppins:wght@400;600;700;900&display=swap" rel="stylesheet">
""", unsafe_allow_html=True)

# Popup Message with Buy Me a Coffee Button
if "popup_shown" not in st.session_state:
    st.session_state["popup_shown"] = False

if not st.session_state["popup_shown"]:
    st.markdown("""
    <div id="welcome-popup" class="popup">
        <p>Hi there! 😊 Thanks so much for stopping by — it truly means the world.
        If you've found any value in our content and you'd like to support what I do,
        a small contribution would go a long way. 💛 Every coffee helps keep Free Investment
        Education free, honest, and growing for everyone. You're amazing. Thank you!</p>
        <div style="text-align: center; margin-top: 1.5rem;">
            <a href="https://www.buymeacoffee.com/freeinvestmenteducation" target="_blank">
                <img src="https://cdn.buymeacoffee.com/buttons/v2/default-yellow.png"
                     alt="Support Us"
                     style="height: 45px; width: auto;"/>
            </a>
        </div>
    </div>
    <script>
        setTimeout(function() {
            var popup = document.getElementById('welcome-popup');
            if (popup) {
                popup.style.display = 'none';
            }
        }, 5000);
    </script>
    """, unsafe_allow_html=True)
    st.session_state["popup_shown"] = True

# ─────────────────────────────────── HELPERS ────────────────────────────────────


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
        df = yf.download(tickers, period=period, interval=interval,
                         threads=True, auto_adjust=False)["Close"].dropna()
        return df
    except Exception as e:
        st.error(f"Error fetching price data: {e}")
        return pd.DataFrame()


@st.cache_data(show_spinner=False)
def fetch_fundamentals(tickers: list[str]) -> pd.DataFrame:
    fundamentals = []
    for ticker in tickers:
        try:
            stock = yf.Ticker(ticker)
            info = stock.info
            fundamentals.append({
                'Ticker': ticker,
                'P/E Ratio': info.get('trailingPE', np.nan),
                'Dividend Yield': info.get('dividendYield', np.nan),
                'Market Cap': info.get('marketCap', np.nan),
                'Beta': info.get('beta', np.nan),
                'EPS (TTM)': info.get('trailingEps', np.nan),
                'Book Value Per Share': info.get('bookValue', np.nan),
                'Debt-to-Equity Ratio': info.get('debtToEquity', np.nan),
                'Current Ratio': info.get('currentRatio', np.nan),
                'Revenue (TTM)': info.get('totalRevenue', np.nan),
                'Profit Margin': info.get('profitMargins', np.nan),
                'Operating Margin': info.get('operatingMargins', np.nan),
                'Return on Equity': info.get('returnOnEquity', np.nan),
                'Sector': info.get('sector', 'Unknown')
            })
        except Exception as e:
            st.error(f"Error fetching fundamentals for {ticker}: {e}")
            fundamentals.append({
                'Ticker': ticker, 'P/E Ratio': np.nan, 'Dividend Yield': np.nan,
                'Market Cap': np.nan, 'Beta': np.nan, 'EPS (TTM)': np.nan,
                'Book Value Per Share': np.nan, 'Debt-to-Equity Ratio': np.nan,
                'Current Ratio': np.nan, 'Revenue (TTM)': np.nan,
                'Profit Margin': np.nan, 'Operating Margin': np.nan,
                'Return on Equity': np.nan, 'Sector': 'Unknown'
            })
    return pd.DataFrame(fundamentals)


def monte_carlo_simulation(returns: pd.DataFrame, weights: pd.Series, n_simulations: int = 500, horizon: int = 52) -> pd.DataFrame:
    mu = returns.mean()
    cov = returns.cov()
    n_assets = len(weights)
    sim_returns = np.zeros((horizon, n_simulations))
    for i in range(n_simulations):
        sim = np.random.multivariate_normal(mu, cov, horizon)
        sim_returns[:, i] = np.cumsum(sim @ weights)
    return pd.DataFrame(sim_returns, columns=[f'Sim {i+1}' for i in range(n_simulations)])

# ---------- ALLOCATION ENGINES -------------------------------------


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


def custom_allocation(tickers: list[str], custom_weights: list[float]) -> pd.Series:
    w = pd.Series(custom_weights, index=tickers, name="Custom Allocation")
    if abs(w.sum() - 1.0) > 0.01 or w.min() < 0:
        st.warning(
            "Custom weights must sum to 100% and be non-negative. Using equal weights.")
        return pd.Series(1 / len(tickers), index=tickers, name="Custom Allocation")
    return w

# ---------- NEW HELPER FUNCTIONS FOR SECTOR EXPOSURE AND RISK CONTRIBUTION ----------


def sector_exposure(weights_df: pd.DataFrame, fundamentals_df: pd.DataFrame) -> pd.DataFrame:
    sector_weights = {}
    for strategy in weights_df.columns:
        strategy_weights = weights_df[strategy]
        sector_allocation = fundamentals_df[['Ticker', 'Sector']].copy()
        sector_allocation['Weight'] = strategy_weights.values
        sector_summary = sector_allocation.groupby(
            'Sector')['Weight'].sum().reset_index()
        sector_weights[strategy] = sector_summary.set_index('Sector')['Weight']
    return pd.DataFrame(sector_weights).fillna(0)


def risk_contribution_by_asset(weights: pd.Series, data: pd.DataFrame) -> pd.Series:
    r = data.pct_change().dropna()
    cov = r.cov().values
    w = weights.values
    portfolio_vol = np.sqrt(w @ cov @ w)
    marginal_risk = (cov @ w) / portfolio_vol
    risk_contrib = w * marginal_risk
    return pd.Series(risk_contrib, index=weights.index, name=weights.name)


def compute_efficient_frontier(data: pd.DataFrame, rf: float = RF, n_points: int = 100) -> pd.DataFrame:
    r = data.pct_change().dropna()
    mu, cov = r.mean().values, r.cov().values
    n = len(mu)

    def portfolio_metrics(w):
        ret = w @ mu * 52  # Annualized return
        vol = np.sqrt(w @ cov @ w) * np.sqrt(52)  # Annualized volatility
        return ret, vol

    returns = []
    volatilities = []
    weights_list = []

    target_returns = np.linspace(min(mu * 52), max(mu * 52), n_points)

    for target in target_returns:
        def objective(w):
            return np.sqrt(w @ cov @ w) * np.sqrt(52)

        constraints = [
            {'type': 'eq', 'fun': lambda w: w.sum() - 1},
            {'type': 'eq', 'fun': lambda w: w @ mu * 52 - target}
        ]
        bounds = [(0, 1)] * n
        res = minimize(objective, np.full(n, 1/n), method='SLSQP',
                       bounds=bounds, constraints=constraints)
        if res.success:
            ret, vol = portfolio_metrics(res.x)
            returns.append(ret)
            volatilities.append(vol)
            weights_list.append(res.x)

    return pd.DataFrame({
        'Return': returns,
        'Volatility': volatilities,
        'Weights': weights_list
    })

# ---------- PERFORMANCE HELPERS ---------------------------------------------------------------


def build_weights_dataframe(w_dict: dict[str, pd.Series], order: list[str]) -> pd.DataFrame:
    return pd.DataFrame({k: v.reindex(order) for k, v in w_dict.items()}).round(3)


def portfolio_returns(data: pd.DataFrame, w: pd.Series) -> pd.Series:
    returns = data.pct_change().dropna()
    return (returns @ w).rename(w.name)


def performance_metrics(series: pd.Series, rf: float = RF) -> dict:
    cum = (1 + series).cumprod()
    total_return = cum.iloc[-1] - 1
    ann_return = (1 + total_return) ** (52 / len(series)) - 1
    ann_vol = series.std() * np.sqrt(52)
    sharpe = (ann_return - rf) / ann_vol if ann_vol != 0 else np.nan
    mdd = (cum / cum.cummax() - 1).min()
    return {'Total Return': total_return, 'Ann. Return': ann_return, 'Ann. Volatility': ann_vol, 'Sharpe': sharpe, 'Max Drawdown': mdd}


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
if "first_load" not in st.session_state:
    st.session_state["first_load"] = True
    st.markdown(
        """
        <div class="hero">
            <h1>Portfolio Optimizer AI 🤖</h1>
            <p>Unleash the power of AI to build your dream portfolio with vibrant insights!.</p>
            <a href="#configure" class="cta-button">Get Started</a>
        </div>
        """,
        unsafe_allow_html=True
    )

# Main Content - Company Selection
st.markdown("<div id='configure'></div>", unsafe_allow_html=True)
st.title('Portfolio Optimizer AI 🤖📈')
st.markdown(
    'Analyze, optimize, and simulate equity portfolios with cutting-edge strategies. '
    '<span style="font-size:0.9rem; color:#4a5568;">For educational purposes only — not investment advice.</span>',
    unsafe_allow_html=True
)

# Searchable Company Selection
mapping = load_sp500_tickers()
companies = sorted(list(mapping.keys()))
selected = st.multiselect(
    "Choose companies (2–20)",
    options=companies,
    default=["Apple Inc.", "Microsoft Corporation"],
    help="Select 2–20 S&P 500 companies. Type to search.",
    max_selections=20,
    key="main_select"
)

# Advanced Options
with st.expander("⚙️ Advanced Settings", expanded=False):
    col1, col2 = st.columns(2)
    with col1:
        period = st.selectbox("Data Period", ["1y", "2y", "5y"], index=1)
        risk_aversion = st.slider(
            "Risk Aversion (Markowitz)", 0.1, 5.0, 1.0, 0.1)
    with col2:
        lookback = st.slider("Momentum Lookback (weeks)", 4, 52, 12)
        n_simulations = st.slider("Monte Carlo Simulations", 100, 250, 500, 50)
    transaction_cost = st.slider("Transaction Cost (%)", 0.0, 1.0, 0.0, 0.01)

# Custom Allocation Input
with st.expander("🛠️ Custom Allocation", expanded=False):
    custom_weights = []
    if selected:
        st.markdown(
            "Enter weights (as decimals, e.g., 0.3 for 30%) that sum to 1.0:")
        cols = st.columns(min(len(selected), 5))
        for i, ticker in enumerate([mapping[n] for n in selected]):
            with cols[i % len(cols)]:
                weight = st.number_input(f"Weight for {
                                         ticker}", min_value=0.0, max_value=1.0, value=1.0/len(selected), step=0.01, key=f"weight_{ticker}")
                custom_weights.append(weight)
        if st.button("Apply Custom Weights", use_container_width=True):
            st.session_state["custom_weights"] = custom_weights

run_button = st.button("🚀 Run Analysis", type="primary",
                       use_container_width=True)

if not run_button:
    st.info('Choose stocks and settings, then hit **Launch Analysis** to see the magic!')
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
    fundamentals_df = fetch_fundamentals(tickers)
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
    if "custom_weights" in st.session_state and len(st.session_state["custom_weights"]) == len(tickers):
        weights['Custom Allocation'] = custom_allocation(
            tickers, st.session_state["custom_weights"])

    weights_df = build_weights_dataframe(weights, tickers)

    # Compute Sector Exposure
    sector_exposure_df = sector_exposure(weights_df, fundamentals_df)

    # Compute Risk Contributions
    risk_contributions = {name: risk_contribution_by_asset(
        w, price_data) for name, w in weights.items()}
    risk_contributions_df = pd.DataFrame(risk_contributions).round(3)

    # Compute Efficient Frontier for Markowitz
    ef_data = compute_efficient_frontier(price_data)

    # Portfolio Returns with Transaction Costs
    ret_dict = {name: portfolio_returns(
        price_data, w) * (1 - transaction_cost/100) for name, w in weights.items()}
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
    beta_vals = {col: np.cov(portfolio_returns_df[col], bmk_returns)[0, 1] / np.var(bmk_returns) if np.var(bmk_returns) != 0 else np.nan
                 for col in portfolio_returns_df.columns}
    beta_df = pd.Series(beta_vals, name='Beta').to_frame()
    rolling_corr_df = pd.DataFrame({col: portfolio_returns_df[col].rolling(ROLL_WINDOW).corr(bmk_returns)
                                   for col in portfolio_returns_df.columns}).dropna()
    cum_returns_df = (1 + portfolio_returns_df).cumprod()
    drawdown_df = cum_returns_df.div(cum_returns_df.cummax()).subtract(1)

    # Monte Carlo Simulations
    mc_results = {name: monte_carlo_simulation(price_data.pct_change().dropna(), w, n_simulations)
                  for name, w in weights.items()}
    progress.progress(100)
    progress.empty()

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

# ───────────────────────────────────────── TABS ───────────────────────────────────────────
st.markdown("### Portfolio Analysis Dashboard", unsafe_allow_html=True)
tab_groups = st.tabs(["📊 Core Analysis", "⚠️ Risk Insights",
                     "🔍 Advanced Metrics", "🔮 Simulations", "📖 Documentation"])

with tab_groups[0]:
    core_tabs = st.tabs(["Weights", "Performance", "Price History", "Fundamentals",
                        "Sector Exposure", "Risk Contribution", "Efficient Frontier"])
    with core_tabs[0]:
        st.subheader("Portfolio Weights by Strategy")
        st.dataframe(weights_df.style.format("{:.2%}").background_gradient(cmap="Blues"),
                     use_container_width=True, height=min(480, 80 + 32 * len(tickers)))
        long_df = weights_df.T.reset_index().melt(
            id_vars='index', var_name='Ticker', value_name='Weight')
        chart = alt.Chart(long_df).mark_bar().encode(
            x=alt.X("index:N", title="Strategy", axis=alt.Axis(labelAngle=45)),
            y=alt.Y("Weight:Q", stack="normalize",
                    title="Weight", axis=alt.Axis(format="%")),
            color=alt.Color("Ticker:N", scale=alt.Scale(scheme="tableau10")),
            tooltip=["Ticker", alt.Tooltip("Weight", format=".1%")]
        ).properties(width="container", height=450, title=alt.Title("Portfolio Weights", subtitle="Allocation across strategies")
                     ).configure_view(stroke=None).configure_axis(labelFont="Inter", titleFont="Inter", titleFontSize=16, titleColor="#2b6cb0"
                                                                  ).configure_legend(labelFont="Inter", titleFont="Inter", titleColor="#2b6cb0")
        st.altair_chart(chart, use_container_width=True)
        csv = weights_df.to_csv().encode("utf-8")
        st.download_button("📥 Download Weights (CSV)", csv,
                           file_name="weights.csv", mime="text/csv", use_container_width=True)

    with core_tabs[1]:
        st.subheader("Portfolio Performance (Static Weights)")
        cum_df = (1 + portfolio_returns_df).cumprod().reset_index().melt(
            id_vars='Date', var_name='Strategy', value_name='Cumulative Return')
        selection = alt.selection_point(fields=['Strategy'], bind='legend')
        perf_chart = alt.Chart(cum_df).mark_line().encode(
            x="Date:T", y="Cumulative Return:Q", color="Strategy:N",
            opacity=alt.condition(selection, alt.value(1.0), alt.value(0.3)),
            tooltip=["Date:T", "Strategy", alt.Tooltip(
                "Cumulative Return:Q", format=".2f")]
        ).add_params(selection).interactive().properties(height=450, title=alt.Title("Cumulative Returns", subtitle="Performance across strategies")
                                                         ).configure_view(stroke=None).configure_axis(labelFont="Inter", titleFont="Inter", titleFontSize=16, titleColor="#2b6cb0"
                                                                                                      ).configure_legend(labelFont="Inter", titleFont="Inter", titleColor="#2b6cb0")
        st.altair_chart(perf_chart, use_container_width=True)
        st.markdown("**Key Performance Metrics**")
        cols = st.columns(min(len(portfolio_returns_df.columns), 5))
        for i, strategy in enumerate(portfolio_returns_df.columns):
            metrics = performance_metrics(portfolio_returns_df[strategy])
            with cols[i % len(cols)]:
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
            x="Date:T", y="Price:Q", color="Ticker:N",
            tooltip=["Date:T", "Ticker", alt.Tooltip("Price:Q", format=".2f")]
        ).interactive().properties(height=450, title=alt.Title("Historical Prices", subtitle=f"Weekly closing prices for {period}")
                                   ).configure_view(stroke=None).configure_axis(labelFont="Inter", titleFont="Inter", titleFontSize=16, titleColor="#2b6cb0"
                                                                                ).configure_legend(labelFont="Inter", titleFont="Inter", titleColor="#2b6cb0")
        st.altair_chart(price_chart, use_container_width=True)

    with core_tabs[3]:
        st.subheader("Fundamental Data")
        st.dataframe(fundamentals_df.style.format({
            'P/E Ratio': "{:.2f}", 'Dividend Yield': "{:.2%}", 'Market Cap': "{:.2e}", 'Beta': "{:.2f}",
            'EPS (TTM)': "{:.2f}", 'Book Value Per Share': "{:.2f}", 'Debt-to-Equity Ratio': "{:.2f}",
            'Current Ratio': "{:.2f}", 'Revenue (TTM)': "{:.2e}", 'Profit Margin': "{:.2%}",
            'Operating Margin': "{:.2%}", 'Return on Equity': "{:.2%}"
        }).background_gradient(cmap="Blues"), use_container_width=True)

    with core_tabs[4]:
        st.subheader("Sector Exposure by Strategy")
        st.dataframe(sector_exposure_df.style.format("{:.2%}").background_gradient(cmap="Greens"),
                     use_container_width=True, height=min(480, 80 + 32 * len(sector_exposure_df)))
        sector_long = sector_exposure_df.T.reset_index().melt(
            id_vars='index', var_name='Sector', value_name='Weight')
        sector_chart = alt.Chart(sector_long).mark_bar().encode(
            x=alt.X("index:N", title="Strategy", axis=alt.Axis(labelAngle=45)),
            y=alt.Y("Weight:Q", stack="normalize",
                    title="Sector Weight", axis=alt.Axis(format="%")),
            color=alt.Color("Sector:N", scale=alt.Scale(scheme="tableau10")),
            tooltip=["Sector", alt.Tooltip("Weight", format=".1%")]
        ).properties(width="container", height=450, title=alt.Title("Sector Exposure", subtitle="Sector allocation across strategies")
                     ).configure_view(stroke=None).configure_axis(labelFont="Inter", titleFont="Inter", titleFontSize=16, titleColor="#2b6cb0"
                                                                  ).configure_legend(labelFont="Inter", titleFont="Inter", titleColor="#2b6cb0")
        st.altair_chart(sector_chart, use_container_width=True)
        csv = sector_exposure_df.to_csv().encode("utf-8")
        st.download_button("📥 Download Sector Exposure (CSV)", csv,
                           file_name="sector_exposure.csv", mime="text/csv", use_container_width=True)

    with core_tabs[5]:
        st.subheader("Risk Contribution by Asset")
        st.dataframe(risk_contributions_df.style.format("{:.2%}").background_gradient(cmap="Oranges"),
                     use_container_width=True, height=min(480, 80 + 32 * len(tickers)))
        risk_long = risk_contributions_df.T.reset_index().melt(
            id_vars='index', var_name='Ticker', value_name='Risk Contribution')
        risk_chart = alt.Chart(risk_long).mark_bar().encode(
            x=alt.X("index:N", title="Strategy", axis=alt.Axis(labelAngle=45)),
            y=alt.Y("Risk Contribution:Q", stack="normalize",
                    title="Risk Contribution", axis=alt.Axis(format="%")),
            color=alt.Color("Ticker:N", scale=alt.Scale(scheme="tableau10")),
            tooltip=["Ticker", alt.Tooltip("Risk Contribution", format=".1%")]
        ).properties(width="container", height=450, title=alt.Title("Risk Contribution by Asset", subtitle="Risk allocation across strategies")
                     ).configure_view(stroke=None).configure_axis(labelFont="Inter", titleFont="Inter", titleFontSize=16, titleColor="#2b6cb0"
                                                                  ).configure_legend(labelFont="Inter", titleFont="Inter", titleColor="#2b6cb0")
        st.altair_chart(risk_chart, use_container_width=True)
        csv = risk_contributions_df.to_csv().encode("utf-8")
        st.download_button("📥 Download Risk Contributions (CSV)", csv,
                           file_name="risk_contributions.csv", mime="text/csv", use_container_width=True)

    with core_tabs[6]:
        st.subheader("Efficient Frontier (Markowitz Portfolio)")
        ef_chart = alt.Chart(ef_data).mark_point(filled=True, size=50).encode(
            x=alt.X("Volatility:Q", title="Annualized Volatility",
                    axis=alt.Axis(format="%")),
            y=alt.Y("Return:Q", title="Annualized Return",
                    axis=alt.Axis(format="%")),
            tooltip=[alt.Tooltip("Volatility:Q", format=".2%"),
                     alt.Tooltip("Return:Q", format=".2%")]
        )

        # Add portfolio points for all strategies
        metrics_df = pd.DataFrame({strategy: performance_metrics(
            portfolio_returns_df[strategy]) for strategy in portfolio_returns_df.columns}).T
        metrics_df['Strategy'] = metrics_df.index
        portfolio_points = alt.Chart(metrics_df).mark_point(filled=True, size=100, shape='diamond').encode(
            x=alt.X("Ann. Volatility:Q", title="Annualized Volatility",
                    axis=alt.Axis(format="%")),
            y=alt.Y("Ann. Return:Q", title="Annualized Return",
                    axis=alt.Axis(format="%")),
            color=alt.Color("Strategy:N", scale=alt.Scale(scheme="tableau10")),
            tooltip=["Strategy", alt.Tooltip("Ann. Return:Q", format=".2%"), alt.Tooltip(
                "Ann. Volatility:Q", format=".2%")]
        )

        combined_chart = alt.layer(ef_chart, portfolio_points).properties(
            width="container",
            height=450,
            title=alt.Title("Efficient Frontier",
                            subtitle="Markowitz optimal portfolios")
        ).configure_view(
            stroke=None
        ).configure_axis(
            labelFont="Inter",
            titleFont="Inter",
            titleFontSize=16,
            titleColor="#2b6cb0"
        ).configure_legend(
            labelFont="Inter",
            titleFont="Inter",
            titleColor="#2b6cb0"
        ).interactive()

        st.altair_chart(combined_chart, use_container_width=True)

with tab_groups[1]:
    risk_tabs = st.tabs(["🔗 Correlation", "⚠️ Risk Overview",
                        "📉 Strategy Drawdowns", "📊 Risk-Return Scatter"])
    with risk_tabs[0]:
        st.subheader("Correlation Matrix (Returns)")
        corr_matrix = price_data.pct_change().dropna().corr().round(2)
        fig, ax = plt.subplots(
            figsize=(len(tickers) * 0.6 + 2, len(tickers) * 0.6 + 2))
        img = ax.imshow(corr_matrix.values, cmap='coolwarm', vmin=-1, vmax=1)
        ax.set_xticks(range(len(tickers)))
        ax.set_yticks(range(len(tickers)))
        ax.set_xticklabels(tickers, rotation=45, ha='right')
        ax.set_yticklabels(tickers)
        for i in range(len(tickers)):
            for j in range(len(tickers)):
                ax.text(j, i, corr_matrix.iloc[i, j], ha='center', va='center', color='white' if abs(
                    corr_matrix.iloc[i, j]) > 0.5 else 'black', fontsize=8)
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
            x="Date:T", y="Volatility:Q", color="Ticker:N",
            tooltip=["Date:T", "Ticker", alt.Tooltip(
                "Volatility:Q", format=".2%")]
        ).interactive().properties(height=400, title=alt.Title("Rolling Volatility", subtitle="12-week annualized volatility for each ticker")
                                   ).configure_view(stroke=None).configure_axis(labelFont="Inter", titleFont="Inter", titleFontSize=16, titleColor="#2b6cb0"
                                                                                ).configure_legend(labelFont="Inter", titleFont="Inter", titleColor="#2b6cb0")
        st.altair_chart(vol_chart, use_container_width=True)
        st.markdown("**Drawdown Chart (Equal Weight Portfolio)**")
        eq_series = portfolio_returns(
            price_data, equal_weight_portfolio(price_data))
        dd = (1 + eq_series).cumprod()
        drawdown = dd / dd.cummax() - 1
        dd_df = drawdown.reset_index().rename(columns={0: 'Drawdown'})
        dd_chart = alt.Chart(dd_df).mark_area(opacity=0.7).encode(
            x="Date:T", y=alt.Y("Drawdown:Q", axis=alt.Axis(format='%')),
            tooltip=["Date:T", alt.Tooltip("Drawdown:Q", format=".1%")]
        ).properties(height=250, title=alt.Title("Equal Weight Portfolio Drawdown", subtitle="Maximum loss from peak")
                     ).configure_view(stroke=None).configure_axis(labelFont="Inter", titleFont="Inter", titleFontSize=16, titleColor="#2b6cb0")
        st.altair_chart(dd_chart, use_container_width=True)

    with risk_tabs[2]:
        st.subheader("Strategy Drawdowns")
        dd_long = drawdown_df.reset_index().melt(
            id_vars='Date', var_name='Strategy', value_name='Drawdown')
        dd_chart_all = alt.Chart(dd_long).mark_area(opacity=0.7).encode(
            x="Date:T", y=alt.Y("Drawdown:Q", axis=alt.Axis(format='%')),
            color="Strategy:N", tooltip=["Date:T", "Strategy", alt.Tooltip("Drawdown:Q", format=".1%")]
        ).interactive().properties(height=450, title=alt.Title("Strategy Drawdowns", subtitle="Maximum loss from peak for each strategy")
                                   ).configure_view(stroke=None).configure_axis(labelFont="Inter", titleFont="Inter", titleFontSize=16, titleColor="#2b6cb0"
                                                                                ).configure_legend(labelFont="Inter", titleFont="Inter", titleColor="#2b6cb0")
        st.altair_chart(dd_chart_all, use_container_width=True)
        st.dataframe(drawdown_df.round(3).style.format("{:.2%}").background_gradient(cmap="Reds"),
                     use_container_width=True, height=min(400, 80 + 32 * len(drawdown_df.tail(10))))

    with risk_tabs[3]:
        st.subheader("Risk-Return Scatter Plot")
        metrics_df = pd.DataFrame({strategy: performance_metrics(portfolio_returns_df[strategy])
                                  for strategy in portfolio_returns_df.columns}).T
        metrics_df['Strategy'] = metrics_df.index
        scatter_fig = px.scatter(metrics_df, x="Ann. Volatility", y="Ann. Return", color="Strategy",
                                 size="Sharpe", hover_data=["Sharpe", "Max Drawdown"], title="Risk-Return Profile by Strategy",
                                 labels={"Ann. Volatility": "Annualized Volatility", "Ann. Return": "Annualized Return"})
        scatter_fig.update_layout(
            showlegend=True, height=450, template="plotly_white")
        st.plotly_chart(scatter_fig, use_container_width=True)

with tab_groups[2]:
    insight_tabs = st.tabs(["📈 Rolling Sharpe", "📊 Market Beta",
                           "🔗 Rolling Correlation", "📉 Return Distribution"])
    with insight_tabs[0]:
        st.subheader("26-Week Rolling Sharpe Ratios (Annualized)")
        rs_long = rolling_sharpe_df.reset_index().melt(
            id_vars='Date', var_name='Strategy', value_name='Sharpe')
        rs_chart = alt.Chart(rs_long).mark_line().encode(
            x="Date:T", y="Sharpe:Q", color="Strategy:N",
            tooltip=["Date:T", "Strategy",
                     alt.Tooltip("Sharpe:Q", format=".2f")]
        ).interactive().properties(height=450, title=alt.Title("Rolling Sharpe Ratios", subtitle="26-week annualized Sharpe ratios")
                                   ).configure_view(stroke=None).configure_axis(labelFont="Inter", titleFont="Inter", titleFontSize=16, titleColor="#2b6cb0"
                                                                                ).configure_legend(labelFont="Inter", titleFont="Inter", titleColor="#2b6cb0")
        st.altair_chart(rs_chart, use_container_width=True)
        st.dataframe(rolling_sharpe_df.round(3).style.background_gradient(cmap="Greens"),
                     use_container_width=True, height=min(400, 80 + 32 * len(rolling_sharpe_df.tail(10))))

    with insight_tabs[1]:
        st.subheader("CAPM Beta vs S&P 500")
        beta_chart = alt.Chart(beta_df.reset_index()).mark_bar().encode(
            x=alt.X("index:N", title="Strategy", sort="-y"),
            y=alt.Y("Beta:Q"), tooltip=["index", alt.Tooltip("Beta:Q", format=".2f")]
        ).properties(height=400, title=alt.Title("Market Beta", subtitle="CAPM beta relative to S&P 500")
                     ).configure_view(stroke=None).configure_axis(labelFont="Inter", titleFont="Inter", titleFontSize=16, titleColor="#2b6cb0")
        st.altair_chart(beta_chart, use_container_width=True)
        st.dataframe(beta_df.round(3).style.background_gradient(cmap="Purples"),
                     use_container_width=True, height=min(400, 80 + 32 * len(beta_df)))

    with insight_tabs[2]:
        st.subheader("26-Week Rolling Correlation with S&P 500")
        rc_long = rolling_corr_df.reset_index().melt(
            id_vars='Date', var_name='Strategy', value_name='Correlation')
        rc_chart = alt.Chart(rc_long).mark_line().encode(
            x="Date:T", y=alt.Y("Correlation:Q", scale=alt.Scale(domain=[-1, 1])),
            color="Strategy:N", tooltip=["Date:T", "Strategy", alt.Tooltip("Correlation:Q", format=".2f")]
        ).interactive().properties(height=450, title=alt.Title("Rolling Correlation", subtitle="26-week correlation with S&P 500")
                                   ).configure_view(stroke=None).configure_axis(labelFont="Inter", titleFont="Inter", titleFontSize=16, titleColor="#2b6cb0"
                                                                                ).configure_legend(labelFont="Inter", titleFont="Inter", titleColor="#2b6cb0")
        st.altair_chart(rc_chart, use_container_width=True)
        st.dataframe(rolling_corr_df.round(3).style.background_gradient(cmap="Blues"),
                     use_container_width=True, height=min(400, 80 + 32 * len(rolling_corr_df.tail(10))))

    with insight_tabs[3]:
        st.subheader("Weekly Return Distribution")
        returns_long = portfolio_returns_df.reset_index().melt(
            id_vars='Date', var_name='Strategy', value_name='Return')
        selection = alt.selection_point(fields=['Strategy'], bind='legend')
        hist_chart = alt.Chart(returns_long).mark_bar(opacity=0.7).encode(
            x=alt.X("Return:Q", bin=alt.Bin(
                maxbins=60), title="Weekly Return"),
            y=alt.Y("count()", title="Frequency"), color="Strategy:N",
            tooltip=["Strategy", alt.Tooltip("count()", title="Freq")]
        ).add_params(selection).transform_filter(selection).properties(height=450,
                                                                       title=alt.Title(
                                                                           "Return Distribution", subtitle="Distribution of weekly returns by strategy")
                                                                       ).configure_view(stroke=None).configure_axis(labelFont="Inter", titleFont="Inter", titleFontSize=16, titleColor="#2b6cb0"
                                                                                                                    ).configure_legend(labelFont="Inter", titleFont="Inter", titleColor="#2b6cb0")
        st.altair_chart(hist_chart, use_container_width=True)

with tab_groups[3]:
    st.subheader(
        "Monte Carlo Simulations (1-Year Forecast for All Strategies)")
    sim_tabs = st.tabs(list(mc_results.keys()))
    for i, strategy in enumerate(mc_results.keys()):
        with sim_tabs[i]:
            mc_df = mc_results[strategy]
            mc_long = mc_df.reset_index().melt(
                id_vars='index', var_name='Simulation', value_name='Return')
            mc_chart = alt.Chart(mc_long).mark_line(opacity=0.1).encode(
                x=alt.X("index:N", title="Weeks"), y=alt.Y("Return:Q", title="Cumulative Return"),
                color=alt.Color("Simulation:N", legend=None)
            ).properties(height=450, title=alt.Title(f"Monte Carlo Simulation: {strategy}", subtitle=f"{n_simulations} simulated paths over 1 year")
                         ).configure_view(stroke=None).configure_axis(labelFont="Inter", titleFont="Inter", titleFontSize=16, titleColor="#2b6cb0")
            st.altair_chart(mc_chart, use_container_width=True)
            final_returns = mc_df.iloc[-1]
            st.markdown(f"**Simulation Summary for {strategy}**")
            cols = st.columns(3)
            cols[0].metric("Median Annual Return", f"{
                           final_returns.median():.2%}")
            cols[1].metric("5th Percentile Return", f"{
                           final_returns.quantile(0.05):.2%}")
            cols[2].metric("95th Percentile Return", f"{
                           final_returns.quantile(0.95):.2%}")

with tab_groups[4]:
    st.subheader("Documentation")
    st.markdown("""
    This section explains the portfolio optimization strategies, performance metrics, and other key concepts used in this application. Use the tabs below to learn more about each topic.
    """)
    doc_tabs = st.tabs(["Portfolio Strategies", "Performance Metrics",
                       "Risk Metrics", "Fundamentals", "Sector & Risk Analysis"])

    with doc_tabs[0]:
        st.markdown("### Portfolio Strategies")
        st.markdown("""
        The application offers various strategies to allocate weights to the selected stocks in your portfolio. Each strategy has a unique approach to balancing risk and return:

        - **Minimum Variance**: Minimizes portfolio volatility by allocating weights to reduce the overall variance, often favoring low-volatility stocks.
        - **Risk Parity**: Allocates weights so that each asset contributes equally to the portfolio's total risk, using hierarchical clustering to account for correlations.
        - **Markowitz (Modern Portfolio Theory)**: Optimizes the portfolio by maximizing expected return for a given level of risk, adjusted by the risk aversion parameter (λ).
        - **Maximum Sharpe**: Maximizes the Sharpe Ratio, which is the portfolio's excess return per unit of risk, relative to the risk-free rate.
        - **Equal Weight**: Assigns equal weights to all assets, providing a simple and diversified approach.
        - **Maximum Diversification**: Maximizes the diversification ratio by balancing weights to reduce correlation-driven risk.
        - **Momentum**: Allocates higher weights to assets with strong recent performance (based on the lookback period).
        - **Minimum Correlation**: Assigns weights inversely proportional to the average correlation of each asset with others, aiming to reduce portfolio correlation.
        - **Inverse Volatility**: Allocates weights inversely proportional to each asset's volatility, favoring less volatile stocks.
        - **Equal Risk Contribution**: Adjusts weights so that each asset contributes equally to the portfolio's total risk, balancing risk contributions.
        - **Market Cap**: Weights assets based on their market capitalization, mimicking market-weighted indices like the S&P 500.
        - **Inverse Beta**: Allocates weights inversely proportional to each asset's beta, favoring low-beta stocks to reduce market risk exposure.
        - **Maximum Return**: Allocates 100% to the asset with the highest historical return, a high-risk strategy.
        - **Custom Allocation**: Allows users to manually specify weights, which must sum to 100% and be non-negative.
        """)

    with doc_tabs[1]:
        st.markdown("### Performance Metrics")
        st.markdown("""
        The application provides several metrics to evaluate portfolio performance:

        - **Total Return**: The overall percentage return of the portfolio over the selected period.
        - **Annualized Return**: The compounded annual return, assuming the portfolio's performance is consistent over a year.
        - **Annualized Volatility**: A measure of the portfolio's risk, calculated as the standard deviation of weekly returns annualized by multiplying by √52.
        - **Sharpe Ratio**: The excess return (return above the risk-free rate) per unit of volatility, indicating risk-adjusted performance.
        - **Maximum Drawdown**: The largest peak-to-trough decline in portfolio value, measuring the worst-case loss.
        """)

    with doc_tabs[2]:
        st.markdown("### Risk Metrics")
        st.markdown("""
        Risk metrics provide insights into the portfolio's risk profile:

        - **Correlation Matrix**: Shows the pairwise correlations between asset returns, indicating how assets move together.
        - **Rolling Volatility**: The 12-week annualized volatility of each asset, showing how risk changes over time.
        - **Drawdown**: The percentage decline from the portfolio's peak value, indicating potential losses during downturns.
        - **Beta**: Measures the portfolio's sensitivity to market movements (S&P 500), where a beta of 1 indicates market-like volatility.
        - **Rolling Correlation**: The 26-week correlation of the portfolio with the S&P 500, showing how closely it tracks the market over time.
        """)

    with doc_tabs[3]:
        st.markdown("### Fundamental Data")
        st.markdown("""
        Fundamental data provides insights into the financial health and valuation of selected companies:

        - **P/E Ratio**: Price-to-earnings ratio, indicating how much investors pay per dollar of earnings.
        - **Dividend Yield**: The annual dividend payment as a percentage of the stock price.
        - **Market Cap**: The total market value of a company's outstanding shares.
        - **Beta**: The stock's volatility relative to the market (S&P 500).
        - **EPS (TTM)**: Earnings per share over the trailing twelve months.
        - **Book Value Per Share**: The net asset value per share, calculated as total assets minus liabilities divided by shares outstanding.
        - **Debt-to-Equity Ratio**: A measure of financial leverage, comparing total debt to shareholders' equity.
        - **Current Ratio**: The ratio of current assets to current liabilities, indicating liquidity.
        - **Revenue (TTM)**: Total revenue over the trailing twelve months.
        - **Profit Margin**: Net income as a percentage of revenue, indicating profitability.
        - **Operating Margin**: Operating income as a percentage of revenue, reflecting operational efficiency.
        - **Return on Equity**: Net income divided by shareholders' equity, measuring how effectively equity is used to generate profit.
        """)

    with doc_tabs[4]:
        st.markdown("### Sector Exposure and Risk Contribution Analysis")
        st.markdown("""
        Additional analyses provide deeper insights into portfolio diversification and risk allocation:

        - **Sector Exposure**: Shows the allocation of portfolio weights across different sectors for each strategy, helping to understand diversification across industries.
        - **Risk Contribution by Asset**: Measures the contribution of each asset to the total portfolio risk, calculated as the product of the asset's weight and its marginal risk contribution (based on covariance with the portfolio).
        - **Efficient Frontier (Markowitz)**: Plots the set of optimal portfolios that offer the highest expected return for a given level of risk, based on the Markowitz model. The plot includes points for all strategies to compare their risk-return profiles.
        """)

# Footer
st.markdown(
    """
    <div style="text-align: center; padding: 2.5rem 0; color: #4a5568; background: #ffffff; border-radius: 12px; box-shadow: 0 3px 10px rgba(0, 0, 0, 0.1); margin-top: 2rem;">
        <span>Portfolio Optimizer AI by Free Investment Education | © 2025 | For educational purposes only.</span><br>
        <a href="https://www.investopedia.com/terms/p/portfolio-weight.asp" style="color: #2b6cb0;">Learn More</a> |
        <a href="mailto:freeinvestmenteducation@gmail.com" style="color: #2b6cb0;">Contact Us</a>
    </div>
    """,
    unsafe_allow_html=True
)
