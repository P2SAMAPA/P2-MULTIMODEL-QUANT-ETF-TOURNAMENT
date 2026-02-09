import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from stable_baselines3 import PPO, A2C
from stable_baselines3.common.vec_env import DummyVecEnv
import gymnasium as gym
from sklearn.preprocessing import StandardScaler
import plotly.graph_objects as go
from datetime import datetime, timedelta
import requests

# --- 1. SETTINGS & STATE ---
st.set_page_config(page_title="Quant Alpha Tournament", layout="wide")

if 'results' not in st.session_state: st.session_state.results = None
if 'forecast' not in st.session_state: st.session_state.forecast = None

TARGET_ETFS = ['TLT', 'TBT', 'VNQ', 'GLD', 'SLV']
MACRO = ['^VIX', '^TNX', 'DX-Y.NYB']
FRED_API_KEY = st.secrets.get("FRED_API_KEY")

# --- 2. ANALYTICS UTILITIES ---
def get_sofr_rate(api_key):
    """Fetches the latest SOFR (Secured Overnight Financing Rate) from FRED."""
    if not api_key: return 0.053  # Current estimated risk-free rate if no key
    url = f"https://api.stlouisfed.org/fred/series/observations?series_id=SOFR&api_key={api_key}&file_type=json"
    try:
        r = requests.get(url, timeout=10).json()
        latest_val = float(r['observations'][-1]['value'])
        return latest_val / 100
    except: return 0.053

def calculate_sharpe(returns, rf_rate):
    """Calculates the annualized Sharpe Ratio against SOFR."""
    returns = np.array(returns)
    daily_rf = rf_rate / 252
    excess_returns = returns - daily_rf
    if np.std(excess_returns) == 0: return 0
    return (np.mean(excess_returns) / np.std(excess_returns)) * np.sqrt(252)

# --- 3. RL ENVIRONMENT ---
class TradingEnv(gym.Env):
    def __init__(self, features, returns, etfs):
        super().__init__()
        self.features, self.returns, self.etfs = features, returns, etfs
        self.action_space = gym.spaces.Discrete(len(etfs))
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(features.shape[1],), dtype=np.float32)
        self.current_step = 0
    def reset(self, seed=None):
        self.current_step = 0
        return self.features[0], {}
    def step(self, action):
        reward = float(self.returns[self.current_step, action])
        self.current_step += 1
        done = self.current_step >= len(self.features) - 1
        return self.features[self.current_step], reward, done, False, {}

# --- 4. TOURNAMENT ENGINE ---
def run_full_tournament(data, rf_rate):
    rets_df = data[TARGET_ETFS].pct_change().dropna()
    feats_df = data.shift(1).dropna()
    common_idx = rets_df.index.intersection(feats_df.index)
    X, y = feats_df.loc[common_idx].values, rets_df.loc[common_idx].values
    
    scaler = StandardScaler()
    X_sc = scaler.fit_transform(X).astype(np.float32)
    split = int(len(X_sc) * 0.8)
    
    # Train Models
    env = DummyVecEnv([lambda: TradingEnv(X_sc[:split], y[:split], TARGET_ETFS)])
    ppo = PPO("MlpPolicy", env, verbose=0).learn(total_timesteps=1500)
    a2c = A2C("MlpPolicy", env, verbose=0).learn(total_timesteps=1200)
    
    # Out-of-Sample Backtest
    results = {"PPO": [], "A2C": [], "Equal-Weight": []}
    dates = common_idx[split:]
    X_test, y_test = X_sc[split:], y[split:]
    
    for i in range(len(X_test)):
        p_act, _ = ppo.predict(X_test[i], deterministic=True)
        a_act, _ = a2c.predict(X_test[i], deterministic=True)
        results["PPO"].append(y_test[i, p_act])
        results["A2C"].append(y_test[i, a_act])
        results["Equal-Weight"].append(np.mean(y_test[i]))

    # Final Forecast for the upcoming session
    latest_feat = X_sc[-1:]
    f_act, _ = ppo.predict(latest_feat[0], deterministic=True)
    
    # 15-Day Trade Audit History
    audit_data = []
    for j in range(len(X_test)-15, len(X_test)):
        idx_j = j
        act, _ = ppo.predict(X_test[idx_j], deterministic=True)
        audit_data.append({
            'Date': dates[idx_j].strftime('%Y-%m-%d'),
            'Ticker': TARGET_ETFS[act],
            'Daily Return': results["PPO"][idx_j]
        })
    history_df = pd.DataFrame(audit_data)

    return results, dates, TARGET_ETFS[f_act], history_df

# --- 5. UI & DASHBOARD ---
st.title("🏆 Advanced Alpha Tournament")

# Calculate Market Open Date
now = datetime.now()
target_date = now if now.hour < 16 else now + timedelta(days=1)
while target_date.weekday() >= 5: target_date += timedelta(days=1)
st.sidebar.subheader("📅 Target Market Session")
st.sidebar.info(f"Predictions applicable for:\n**{target_date.strftime('%A, %b %d, %Y')}**")

if st.button("🚀 Run Analysis & Backtest"):
    with st.status("Analyzing Market Regimes...") as status:
        rf = get_sofr_rate(FRED_API_KEY)
        data = yf.download(TARGET_ETFS + MACRO, start="2018-01-01", progress=False)['Close'].ffill().dropna()
        res, dates, fc_ticker, hist = run_full_tournament(data, rf)
        st.session_state.results = (res, dates, rf)
        st.session_state.forecast = (fc_ticker, hist)
        status.update(label="Analysis Complete!", state="complete")
    st.rerun()

if st.session_state.results:
    res, dates, rf = st.session_state.results
    fc_ticker, hist = st.session_state.forecast
    
    # Summary Metrics
    st.header(f"🎯 Next Trade: {fc_ticker}")
    
    col1, col2, col3 = st.columns(3)
    ppo_rets = np.array(res["PPO"])
    annualized_return = (np.prod(1 + ppo_rets)**(252/len(ppo_rets)) - 1)
    
    col1.metric("PPO Annualized Return", f"{annualized_return:.2%}", delta="Best Performer")
    col2.metric("Sharpe Ratio (vs SOFR)", f"{calculate_sharpe(res['PPO'], rf):.2f}")
    col3.metric("Risk-Free Rate (SOFR)", f"{rf:.2%}")

    # Sharpe Leaderboard
    st.subheader("📊 Efficiency Comparison")
    l_cols = st.columns(3)
    for i, (name, r) in enumerate(res.items()):
        l_cols[i].metric(f"{name} Sharpe", f"{calculate_sharpe(r, rf):.2f}")

    # Chart
    fig = go.Figure()
    for name, r in res.items():
        fig.add_trace(go.Scatter(x=dates, y=np.cumprod(1 + np.array(r)), name=name))
    fig.update_layout(title="Equity Curves (Out-of-Sample)", template="plotly_dark", height=500)
    st.plotly_chart(fig, use_container_width=True)

    # 15-Day Audit
    st.subheader("📅 Last 15 Sessions: Trade Audit")
    st.table(hist.sort_values('Date', ascending=False).style.format({'Daily Return': '{:.2%}'}))
