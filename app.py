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

# --- 1. SETTINGS & STATE ---
st.set_page_config(page_title="Quant Tournament", layout="wide")

if 'results' not in st.session_state: st.session_state.results = None
if 'forecast' not in st.session_state: st.session_state.forecast = None

TARGET_ETFS = ['TLT', 'TBT', 'VNQ', 'GLD', 'SLV']
MACRO = ['^VIX', '^TNX', 'DX-Y.NYB']

# --- 2. MODEL ARCHITECTURES ---
class CNN_LSTM(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.conv = nn.Conv1d(in_dim, 32, 3, padding=1)
        self.lstm = nn.LSTM(32, 64, batch_first=True)
        self.fc = nn.Linear(64, out_dim)
    def forward(self, x):
        # x shape: [batch, seq, features] -> transpose for CNN: [batch, features, seq]
        x = x.transpose(1, 2)
        x = torch.relu(self.conv(x)).transpose(1, 2)
        _, (hn, _) = self.lstm(x)
        return self.fc(hn[-1])

# --- 3. RL ENVIRONMENT ---
class TradingEnv(gym.Env):
    def __init__(self, features, returns, etfs):
        super().__init__()
        self.features = features
        self.returns = returns
        self.etfs = etfs
        self.action_space = gym.spaces.Discrete(len(etfs))
        # Fixed dimensioning: the observation space must match X_sc width
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(features.shape[1],), dtype=np.float32)
        self.current_step = 0
    def reset(self, seed=None):
        self.current_step = 0
        return self.features[0], {}
    def step(self, action):
        reward = float(self.returns[self.current_step, action])
        self.current_step += 1
        done = self.current_step >= len(self.features) - 1
        obs = self.features[self.current_step]
        return obs, reward, done, False, {}

# --- 4. ENGINE ---
def run_full_tournament(data):
    # Calculate returns and lagged features
    rets_df = data[TARGET_ETFS].pct_change().dropna()
    feats_df = data.shift(1).dropna()
    common_idx = rets_df.index.intersection(feats_df.index)
    
    X = feats_df.loc[common_idx].values
    y = rets_df.loc[common_idx].values
    
    scaler = StandardScaler()
    X_sc = scaler.fit_transform(X).astype(np.float32)
    
    split = int(len(X_sc) * 0.8)
    X_train, X_test = X_sc[:split], X_sc[split:]
    y_train, y_test = y[:split], y[split:]

    # RL Models (PPO & A2C)
    def make_env(): return TradingEnv(X_train, y_train, TARGET_ETFS)
    env = DummyVecEnv([make_env])
    
    ppo = PPO("MlpPolicy", env, verbose=0).learn(total_timesteps=1200)
    a2c = A2C("MlpPolicy", env, verbose=0).learn(total_timesteps=1200)
    
    # DL Model (CNN-LSTM)
    model = CNN_LSTM(X.shape[1], len(TARGET_ETFS))
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
    X_t = torch.tensor(X_train).unsqueeze(1) # [batch, seq=1, features]
    y_t = torch.tensor(y_train).float()
    
    for _ in range(30):
        optimizer.zero_grad()
        loss = nn.MSELoss()(model(X_t), y_t)
        loss.backward()
        optimizer.step()

    # Backtest Loop
    results = {"PPO": [], "A2C": [], "CNN-LSTM": []}
    for i in range(len(X_test)):
        # PPO & A2C Predictions
        p_act, _ = ppo.predict(X_test[i], deterministic=True)
        a_act, _ = a2c.predict(X_test[i], deterministic=True)
        
        # CNN-LSTM Prediction
        with torch.no_grad():
            c_out = model(torch.tensor(X_test[i]).reshape(1, 1, -1))
            c_act = torch.argmax(c_out).item()
            
        results["PPO"].append(y_test[i, p_act])
        results["A2C"].append(y_test[i, a_act])
        results["CNN-LSTM"].append(y_test[i, c_act])

    # Final Forecast for Tomorrow
    latest_feat = X_sc[-1:]
    final_act, _ = ppo.predict(latest_feat[0], deterministic=True)
    forecast = {"ETF": TARGET_ETFS[final_act], "Model": "PPO (Best Adaptive)"}
    
    return results, common_idx[split:], forecast

# --- 5. UI ---
st.title("🏆 Advanced Multi-Model Alpha Tournament")

@st.cache_data
def load_market_data():
    return yf.download(TARGET_ETFS + MACRO, start="2018-01-01", progress=False)['Close'].ffill().dropna()

data = load_market_data()

if st.button("🚀 Run Tournament"):
    with st.status("Training RL Agents and Neural Networks...") as status:
        res, dates, fc = run_full_tournament(data)
        st.session_state.results = (res, dates)
        st.session_state.forecast = fc
        status.update(label="Tournament Analysis Finished!", state="complete")
    st.rerun()

if st.session_state.results:
    res, dates = st.session_state.results
    fc = st.session_state.forecast

    # 1. Prediction for Tomorrow
    st.header(f"🎯 Forecast for Next Session: {fc['ETF']}")
    st.success(f"The **{fc['Model']}** recommends a long position in **{fc['ETF']}** based on current macro regimes.")

    # 2. Results Comparison
    st.subheader("📊 Performance Leaderboard")
    cols = st.columns(3)
    for i, (name, r) in enumerate(res.items()):
        cum_ret = (np.prod(1 + np.array(r)) - 1)
        cols[i].metric(name, f"{cum_ret:.2%}")

    # 3. Charting
    fig = go.Figure()
    for name, r in res.items():
        fig.add_trace(go.Scatter(x=dates, y=np.cumprod(1 + np.array(r)), name=name))
    fig.update_layout(title="Equity Curves (Out-of-Sample)", template="plotly_dark")
    st.plotly_chart(fig, use_container_width=True)
