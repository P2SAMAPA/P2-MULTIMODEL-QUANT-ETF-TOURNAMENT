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

# Persistent state for results
if 'results' not in st.session_state: st.session_state.results = None

TARGET_ETFS = ['TLT', 'TBT', 'VNQ', 'GLD', 'SLV']
MACRO = ['^VIX', '^TNX', 'DX-Y.NYB']
FRED_API_KEY = st.secrets.get("FRED_API_KEY")

# --- 2. ANALYTICS & DATA UTILITIES ---
def get_sofr_rate(api_key):
    """Fetches latest SOFR from FRED. Falls back to 3.6% if API fails."""
    if not api_key: return 0.036 
    url = f"https://api.stlouisfed.org/fred/series/observations?series_id=SOFR&api_key={api_key}&file_type=json"
    try:
        r = requests.get(url, timeout=10).json()
        return float(r['observations'][-1]['value']) / 100
    except: return 0.036

def calculate_sharpe(returns, rf_rate):
    returns = np.array(returns)
    daily_rf = rf_rate / 252
    excess = returns - daily_rf
    if np.std(excess) == 0: return 0
    return (np.mean(excess) / np.std(excess)) * np.sqrt(252)

# --- 3. RL ENVIRONMENT ---
class TradingEnv(gym.Env):
    def __init__(self, features, returns, etfs):
        super().__init__()
        self.features, self.returns, self.etfs = features, returns, etfs
        self.action_space = gym.spaces.Discrete(len(etfs))
        self.observation_space = gym.spaces.Box(low=-10, high=10, shape=(features.shape[1],), dtype=np.float32)
        self.current_step = 0
    def reset(self, seed=None):
        self.current_step = 0
        return self.features[0], {}
    def step(self, action):
        reward = float(self.returns[self.current_step, action])
        self.current_step += 1
        done = self.current_step >= len(self.features) - 1
        obs = self.features[self.current_step] if not done else self.features[-1]
        return obs, reward, done, False, {}

# --- 4. ENGINE (With 7-Day Retraining Cache) ---
@st.cache_resource(ttl=604800) # 7 Days in seconds
def train_and_backtest(data_json, rf_rate):
    data = pd.read_json(data_json)
    rets_df = data[TARGET_ETFS].pct_change().dropna()
    feats_df = data.shift(1).dropna()
    common_idx = rets_df.index.intersection(feats_df.index)
    
    X, y = feats_df.loc[common_idx].values, rets_df.loc[common_idx].values
    scaler = StandardScaler()
    X_sc = scaler.fit_transform(X).astype(np.float32)
    split = int(len(X_sc) * 0.8)
    
    # Train Models
    env = DummyVecEnv([lambda: TradingEnv(X_sc[:split], y[:split], TARGET_ETFS)])
    ppo_model = PPO("MlpPolicy", env, verbose=0).learn(total_timesteps=2000)
    a2c_model = A2C("MlpPolicy", env, verbose=0).learn(total_timesteps=2000)
    
    # Out-of-Sample Evaluation
    results = {"PPO": [], "A2C": [], "Equal-Weight": []}
    dates = common_idx[split:]
    X_test, y_test = X_sc[split:], y[split:]
    
    for i in range(len(X_test)):
        p_act, _ = ppo_model.predict(X_test[i], deterministic=True)
        a_act, _ = a2c_model.predict(X_test[i], deterministic=True)
        results["PPO"].append(y_test[i, p_act])
        results["A2C"].append(y_test[i, a_act])
        results["Equal-Weight"].append(np.mean(y_test[i]))

    # Identify Champion based on Cumulative Return
    perf = {k: np.prod(1 + np.array(v)) for k, v in results.items()}
    champion_name = max(perf, key=perf.get)
    
    # Forecast with Champion
    latest_feat = X_sc[-1]
    if champion_name == "PPO":
        f_act, _ = ppo_model.predict(latest_feat, deterministic=True)
    elif champion_name == "A2C":
        f_act, _ = a2c_model.predict(latest_feat, deterministic=True)
    else: # Equal Weight fallback
        f_act = np.argmax(latest_feat[:len(TARGET_ETFS)]) 

    # Audit for last 15 sessions
    audit_data = []
    for j in range(max(0, len(X_test)-15), len(X_test)):
        # Dynamic audit based on current champion
        if champion_name == "A2C": act, _ = a2c_model.predict(X_test[j], deterministic=True)
        else: act, _ = ppo_model.predict(X_test[j], deterministic=True)
        
        audit_data.append({
            'Date': dates[j].strftime('%Y-%m-%d'),
            'Ticker': TARGET_ETFS[act],
            'Daily Return': results[champion_name][j]
        })

    return results, dates, TARGET_ETFS[f_act], champion_name, pd.DataFrame(audit_data)

# --- 5. UI ---
st.title("🏆 Quant Alpha Tournament")

# Market session logic
now = datetime.now()
target_date = now if now.hour < 16 else now + timedelta(days=1)
while target_date.weekday() >= 5: target_date += timedelta(days=1)

if st.button("🚀 Run Tournament"):
    with st.status("Fetching Data & Training Models...") as status:
        rf = get_sofr_rate(FRED_API_KEY)
        raw_data = yf.download(TARGET_ETFS + MACRO, start="2019-01-01", progress=False)['Close'].ffill().dropna()
        
        # We pass data as JSON to satisfy the streamlit cache requirements
        res, dates, ticker, champ, audit = train_and_backtest(raw_data.to_json(), rf)
        
        st.session_state.results = {
            "res": res, "dates": dates, "rf": rf, 
            "ticker": ticker, "champ": champ, "audit": audit
        }
        status.update(label=f"Tournament Complete! Champion: {champ}", state="complete")

if st.session_state.results:
    s = st.session_state.results
    
    st.header(f"🎯 Forecast for {target_date.strftime('%b %d')}: BUY {s['ticker']}")
    st.subheader(f"Current Model Champion: :green[{s['champ']}]")
    
    # Metrics
    m1, m2, m3 = st.columns(3)
    champ_rets = np.array(s['res'][s['champ']])
    ann_ret = (np.prod(1 + champ_rets)**(252/len(champ_rets)) - 1)
    
    m1.metric(f"{s['champ']} Annualized", f"{ann_ret:.2%}")
    m2.metric(f"{s['champ']} Sharpe", f"{calculate_sharpe(champ_rets, s['rf']):.2f}")
    m3.metric("SOFR Rate", f"{s['rf']:.2%}")

    # Chart
    fig = go.Figure()
    for name, r in s['res'].items():
        fig.add_trace(go.Scatter(x=s['dates'], y=np.cumprod(1 + np.array(r)), name=name))
    fig.update_layout(
        title="Out of Sample Cumulative Return", 
        template="plotly_dark",
        xaxis_title="Date",
        yaxis_title="Growth of $1"
    )
    st.plotly_chart(fig, use_container_width=True)

    # Audit
    st.subheader(f"📅 Last 15 Sessions Audit ({s['champ']})")
    st.table(s['audit'].sort_values('Date', ascending=False).style.format({'Daily Return': '{:.2%}'}))
