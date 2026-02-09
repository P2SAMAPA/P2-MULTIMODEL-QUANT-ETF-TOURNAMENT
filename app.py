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
# Ensure you add your FRED API Key to Streamlit Secrets
FRED_API_KEY = st.secrets.get("FRED_API_KEY")

# --- 2. ANALYTICS UTILITIES ---
def get_sofr_rate(api_key):
    """Fetches latest SOFR from FRED. Falls back to 5.3% if API fails."""
    if not api_key: return 0.053 
    url = f"https://api.stlouisfed.org/fred/series/observations?series_id=SOFR&api_key={api_key}&file_type=json"
    try:
        r = requests.get(url, timeout=10).json()
        latest_val = float(r['observations'][-1]['value'])
        return latest_val / 100
    except: return 0.053

def calculate_sharpe(returns, rf_rate):
    """Calculates annualized Sharpe Ratio against the SOFR risk-free rate."""
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
        # Fixed: Observation shape must match feature count to prevent ValueErrors
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

# --- 4. ENGINE ---
def run_full_tournament(data, rf_rate):
    # Check for empty data download
    if data.empty or len(data) < 50:
        st.error("Error: Yahoo Finance returned insufficient data. Please wait a minute and try again.")
        return None, None, None, None

    rets_df = data[TARGET_ETFS].pct_change().dropna()
    feats_df = data.shift(1).dropna()
    common_idx = rets_df.index.intersection(feats_df.index)
    
    if len(common_idx) < 30:
        st.error("Error: Data alignment failed. No overlapping dates found between macro and ETFs.")
        return None, None, None, None

    X, y = feats_df.loc[common_idx].values, rets_df.loc[common_idx].values
    
    scaler = StandardScaler()
    X_sc = scaler.fit_transform(X).astype(np.float32)
    split = int(len(X_sc) * 0.8)
    
    # Train Agents
    env = DummyVecEnv([lambda: TradingEnv(X_sc[:split], y[:split], TARGET_ETFS)])
    ppo = PPO("MlpPolicy", env, verbose=0).learn(total_timesteps=1500)
    a2c = A2C("MlpPolicy", env, verbose=0).learn(total_timesteps=1200)
    
    # Backtest
    results = {"PPO": [], "A2C": [], "Equal-Weight": []}
    dates = common_idx[split:]
    X_test, y_test = X_sc[split:], y[split:]
    
    for i in range(len(X_test)):
        p_act, _ = ppo.predict(X_test[i], deterministic=True)
        a_act, _ = a2c.predict(X_test[i], deterministic=True)
        results["PPO"].append(y_test[i, p_act])
        results["A2C"].append(y_test[i, a_act])
        results["Equal-Weight"].append(np.mean(y_test[i]))

    # Tomorrow's Forecast
    latest_feat = X_sc[-1:]
    f_act, _ = ppo.predict(latest_feat[0], deterministic=True)
    
    # Audit History
    audit_data = []
    for j in range(max(0, len(X_test)-15), len(X_test)):
        act, _ = ppo.predict(X_test[j], deterministic=True)
        audit_data.append({
            'Date': dates[j].strftime('%Y-%m-%d'),
            'Ticker': TARGET_ETFS[act],
            'Daily Return': results["PPO"][j]
        })
    history_df = pd.DataFrame(audit_data)

    return results, dates, TARGET_ETFS[f_act], history_df

# --- 5. UI ---
st.title("🏆 Quant Alpha Tournament")

# Logic to find next market session
now = datetime.now()
target_date = now if now.hour < 16 else now + timedelta(days=1)
while target_date.weekday() >= 5: target_date += timedelta(days=1)
st.sidebar.subheader("📅 Target Session")
st.sidebar.info(f"Analysis for:\n**{target_date.strftime('%A, %b %d, %Y')}**")

if st.button("🚀 Run Backtest & Tournament"):
    with st.status("Training Agents & Fetching SOFR...") as status:
        rf = get_sofr_rate(FRED_API_KEY)
        data = yf.download(TARGET_ETFS + MACRO, start="2018-01-01", progress=False)['Close'].ffill().dropna()
        res, dates, fc_ticker, hist = run_full_tournament(data, rf)
        if res:
            st.session_state.results = (res, dates, rf)
            st.session_state.forecast = (fc_ticker, hist)
            status.update(label="Tournament Complete!", state="complete")
    st.rerun()

if st.session_state.results:
    res, dates, rf = st.session_state.results
    fc_ticker, hist = st.session_state.forecast
    
    # 1. Prediction Hero Section
    st.header(f"🎯 Forecast for {target_date.strftime('%b %d')}: BUY {fc_ticker}")
    
    m1, m2, m3 = st.columns(3)
    ppo_rets = np.array(res["PPO"])
    ann_ret = (np.prod(1 + ppo_rets)**(252/len(ppo_rets)) - 1)
    
    m1.metric("PPO Annualized Return", f"{ann_ret:.2%}", "Out-of-Sample")
    m2.metric("PPO Sharpe (vs SOFR)", f"{calculate_sharpe(res['PPO'], rf):.2f}")
    m3.metric("SOFR Rate", f"{rf:.2%}")

    # 2. Sharpe Comparison
    st.subheader("📊 Efficiency Leaderboard")
    s_cols = st.columns(3)
    for i, (name, r) in enumerate(res.items()):
        s_val = calculate_sharpe(r, rf)
        s_cols[i].metric(name, f"{s_val:.2f}", "Sharpe")

    # 3. Chart
    fig = go.Figure()
    for name, r in res.items():
        fig.add_trace(go.Scatter(x=dates, y=np.cumprod(1 + np.array(r)), name=name))
    fig.update_layout(title="Out-of-Sample Equity Curves", template="plotly_dark")
    st.plotly_chart(fig, use_container_width=True)

    # 4. Audit Table
    st.subheader("📅 Last 15 Sessions Audit")
    st.table(hist.sort_values('Date', ascending=False).style.format({'Daily Return': '{:.2%}'}))
