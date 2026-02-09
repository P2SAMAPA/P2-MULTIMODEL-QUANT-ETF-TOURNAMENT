import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import requests
import plotly.graph_objects as go
from stable_baselines3 import PPO, A2C
from stable_baselines3.common.vec_env import DummyVecEnv
import gymnasium as gym
from sklearn.preprocessing import StandardScaler
from datetime import datetime, timedelta
import pytz
import time

# --- 1. CONFIGURATION ---
st.set_page_config(page_title="Advanced Alpha Tournament", layout="wide")

TARGET_ETFS = ['TLT', 'TBT', 'VNQ', 'GLD', 'SLV']
YAHOO_MACRO = ['^VIX', '^TNX', 'DX-Y.NYB']
FRED_API_KEY = st.secrets.get("FRED_API_KEY")

# Initialize session state for persistent results
if 'tournament_results' not in st.session_state:
    st.session_state.tournament_results = None
if 'dates' not in st.session_state:
    st.session_state.dates = None

# --- 2. DATA ENGINE ---
def get_next_market_date():
    tz = pytz.timezone('US/Eastern')
    now = datetime.now(tz)
    next_date = now if now.hour < 16 else now + timedelta(days=1)
    while next_date.weekday() >= 5:
        next_date += timedelta(days=1)
    return next_date.strftime('%A, %b %d, %Y')

@st.cache_data(ttl=3600)
def get_master_data(api_key):
    all_tickers = TARGET_ETFS + YAHOO_MACRO
    try:
        raw = yf.download(all_tickers, start="2015-01-01", auto_adjust=True, progress=False)
        prices = raw['Close'] if isinstance(raw.columns, pd.MultiIndex) else raw
        return prices.ffill().dropna()
    except:
        return pd.DataFrame()

# --- 3. REINFORCEMENT LEARNING ENV ---
class TradingEnv(gym.Env):
    def __init__(self, df, etfs, feature_cols):
        super().__init__()
        self.df = df.reset_index(drop=True)
        self.etfs = etfs
        self.feature_cols = feature_cols
        self.action_space = gym.spaces.Discrete(len(etfs))
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(len(feature_cols),), dtype=np.float32)
        self.current_step = 0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        return self.df[self.feature_cols].iloc[0].values.astype(np.float32), {}

    def step(self, action):
        reward = float(self.df[self.etfs[action]].iloc[self.current_step])
        self.current_step += 1
        done = self.current_step >= len(self.df) - 1
        obs = self.df[self.feature_cols].iloc[self.current_step].values.astype(np.float32)
        return obs, reward, done, False, {}

# --- 4. TOURNAMENT ENGINE ---
def run_tournament(data):
    # Reduced sample size slightly for Streamlit stability
    data = data.tail(1500) 
    rets = data[TARGET_ETFS].pct_change().dropna()
    features = data.shift(1).dropna()
    idx = rets.index.intersection(features.index)
    X, y = features.loc[idx], rets.loc[idx]
    
    feature_cols = X.columns.tolist()
    scaler = StandardScaler()
    X_sc = scaler.fit_transform(X).astype(np.float32)
    y_vals = y.values
    
    split = int(len(X_sc) * 0.8)
    X_train, X_live = X_sc[:split], X_sc[split:]
    y_train, y_live = y_vals[:split], y_vals[split:]

    # RL Models
    train_df = pd.DataFrame(X_train, columns=feature_cols)
    for i, col in enumerate(TARGET_ETFS): train_df[col] = y_train[:, i]
    
    env = DummyVecEnv([lambda: TradingEnv(train_df, TARGET_ETFS, feature_cols)])
    ppo = PPO("MlpPolicy", env).learn(total_timesteps=1000)
    
    ppo_rets = []
    for obs in X_live:
        act, _ = ppo.predict(obs, deterministic=True)
        ppo_rets.append(y_live[len(ppo_rets), act])
        
    return {"PPO-Regime-RL": ppo_rets}, idx[split:]

# --- 5. UI ---
st.title("🏆 Alpha Tournament")

df_raw = get_master_data(FRED_API_KEY)

if df_raw.empty:
    st.error("Data Fetch Failed.")
else:
    if st.button("🚀 Run Analysis"):
        with st.status("Computing...", expanded=True):
            res, dates = run_tournament(df_raw)
            st.session_state.tournament_results = res
            st.session_state.dates = dates
        st.rerun() # Force UI to show stored results

if st.session_state.tournament_results:
    res = st.session_state.tournament_results
    dates = st.session_state.dates
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("Leaderboard")
        for name, r in res.items():
            st.metric(name, f"{(np.prod(1+np.array(r))-1):.2%}")
            
    with col2:
        fig = go.Figure()
        for name, r in res.items():
            fig.add_trace(go.Scatter(x=dates, y=np.cumprod(1+np.array(r)), name=name))
        st.plotly_chart(fig, use_container_width=True)
