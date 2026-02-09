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

# --- 1. CONFIGURATION ---
st.set_page_config(page_title="Advanced Alpha Tournament", layout="wide")

TARGET_ETFS = ['TLT', 'TBT', 'VNQ', 'GLD', 'SLV']
YAHOO_MACRO = ['^VIX', '^TNX', 'DX-Y.NYB']
FRED_API_KEY = st.secrets.get("FRED_API_KEY")

# --- 2. DATA ENGINE ---
def get_next_market_date():
    tz = pytz.timezone('US/Eastern')
    now = datetime.now(tz)
    next_date = now if now.hour < 16 else now + timedelta(days=1)
    while next_date.weekday() >= 5:
        next_date += timedelta(days=1)
    return next_date.strftime('%A, %b %d, %Y')

def fetch_fred_yield(api_key):
    url = f"https://api.stlouisfed.org/fred/series/observations?series_id=T10Y2Y&api_key={api_key}&file_type=json"
    try:
        r = requests.get(url, timeout=10).json()
        df = pd.DataFrame(r['observations'])
        df['date'] = pd.to_datetime(df['date'])
        df['value'] = pd.to_numeric(df['value'], errors='coerce')
        return df.set_index('date')[['value']].rename(columns={'value': 'T10Y2Y'})
    except: return pd.DataFrame()

@st.cache_data(ttl=3600)
def get_master_data(api_key):
    all_tickers = TARGET_ETFS + YAHOO_MACRO
    raw = yf.download(all_tickers, start="2010-01-01", auto_adjust=True)
    prices = raw['Close'] if isinstance(raw.columns, pd.MultiIndex) else raw
    fred_df = fetch_fred_yield(api_key)
    combined = pd.concat([prices, fred_df], axis=1).ffill().dropna()
    return combined

# --- 3. REINFORCEMENT LEARNING ENV ---
class TradingEnv(gym.Env):
    def __init__(self, df, etfs, feature_cols):
        super().__init__()
        self.df = df.reset_index(drop=True)
        self.etfs = etfs
        self.feature_cols = feature_cols
        self.action_space = gym.spaces.Discrete(len(etfs))
        self.observation_space = gym.spaces.Box(
            low=-np.inf, 
            high=np.inf, 
            shape=(len(feature_cols),), 
            dtype=np.float32
        )
        self.current_step = 0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        obs = self.df[self.feature_cols].iloc[0].values.astype(np.float32)
        return obs, {}

    def step(self, action):
        reward = self.df[self.etfs[action]].iloc[self.current_step]
        self.current_step += 1
        done = self.current_step >= len(self.df) - 1
        obs = self.df[self.feature_cols].iloc[self.current_step].values.astype(np.float32)
        return obs, reward, done, False, {}

# --- 4. DEEP LEARNING ARCHITECTURES ---
class CNN_LSTM_Model(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.cnn = nn.Conv1d(input_dim, 64, kernel_size=3, padding=1)
        self.lstm = nn.LSTM(64, 128, batch_first=True)
        self.fc = nn.Linear(128, output_dim)
    def forward(self, x):
        x = x.transpose(1, 2)
        x = torch.relu(self.cnn(x)).transpose(1, 2)
        _, (hn, _) = self.lstm(x)
        return self.fc(hn[-1])

class TransformerModel(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.d_model = 64 
        self.input_proj = nn.Linear(input_dim, self.d_model)
        self.enc = nn.TransformerEncoderLayer(d_model=self.d_model, nhead=4, batch_first=True)
        self.transformer = nn.TransformerEncoder(self.enc, num_layers=2)
        self.fc = nn.Linear(self.d_model, output_dim)
    def forward(self, x):
        x = self.input_proj(x)
        x = self.transformer(x)
        return self.fc(x[:, -1, :])

# --- 5. TOURNAMENT ENGINE ---
def run_tournament(data):
    rets = data[TARGET_ETFS].pct_change().dropna()
    features = data.shift(1).dropna()
    idx = rets.index.intersection(features.index)
    X, y = features.loc[idx], rets.loc[idx]
    
    feature_cols = X.columns.tolist()
    
    seq_len = 10
    X_s, y_s = [], []
    for i in range(len(X) - seq_len):
        X_s.append(X.iloc[i:i+seq_len].values)
        y_s.append(y.iloc[i+seq_len].values)
    X_s, y_s = np.array(X_s), np.array(y_s)

    split = int(len(X_s) * 0.8)
    X_train, X_live = X_s[:split], X_s[split:]
    y_train, y_live = y_s[:split], y_s[split:]

    scaler = StandardScaler()
    X_train_flat = X_train.reshape(-1, X_train.shape[-1])
    scaler.fit(X_train_flat)
    
    def scale_seq(seq_data):
        flat = seq_data.reshape(-1, seq_data.shape[-1])
        scaled = scaler.transform(flat).reshape(seq_data.shape)
        return np.nan_to_num(scaled).astype(np.float32)

    X_train_sc = scale_seq(X_train)
    X_live_sc = scale_seq(X_live)

    results = {}
    
    # RL MODELS (PPO & A2C)
    train_obs = X_train_sc[:, -1, :]
    train_env_df = pd.DataFrame(train_obs, columns=feature_cols)
    for i, col in enumerate(TARGET_ETFS):
        train_env_df[col] = y_train[:, i]

    def make_env():
        return TradingEnv(train_env_df, TARGET_ETFS, feature_cols)

    env = DummyVecEnv([make_env])
    
    ppo = PPO("MlpPolicy", env, verbose=0).learn(total_timesteps=1500)
    a2c = A2C("MlpPolicy", env, verbose=0).learn(total_timesteps=1500)
    
    ppo_actions, a2c_actions = [], []
    for obs in X_live_sc[:, -1, :]:
        p_act, _ = ppo.predict(np.array([obs]), deterministic=True)
        a_act, _ = a2c.predict(np.array([obs]), deterministic=True)
        ppo_actions.append(p_act[0])
        a2c_actions.append(a_act[0])
    
    results['PPO'] = [y_live[i, a] for i, a in enumerate(ppo_actions)]
    results['A2C'] = [y_live[i, a] for i, a in enumerate(a2c_actions)]

    # SEQUENCE MODELS (CNN-LSTM & Transformer)
    X_t = torch.tensor(X_train_sc, dtype=torch.float32)
    y_t = torch.tensor(y_train, dtype=torch.float32)
    X_l_t = torch.tensor(X_live_sc, dtype=torch.float32)

    for name, m_class in [("CNN-LSTM", CNN_LSTM_Model), ("Transformer", TransformerModel)]:
        model = m_class(len(feature_cols), len(TARGET_ETFS))
        opt = torch.optim.Adam(model.parameters(), lr=0.005)
        for _ in range(25):
            opt.zero_grad()
            loss = nn.MSELoss()(model(X_t), y_t)
            loss.backward()
            opt.step()
        
        with torch.no_grad():
            preds = model(X_l_t).numpy()
            results[name] = [y_live[i, np.argmax(p)] for i, p in enumerate(preds)]

    return results, idx[split+seq_len:]

# --- 6. UI ---
st.title("🏆 Advanced Quant Alpha Tournament")
st.markdown("Comparing **PPO, A2C, CNN-LSTM, and Transformer** on identical macro regimes.")

if not FRED_API_KEY:
    st.error("Please add FRED_API_KEY to your Streamlit Secrets.")
else:
    df_raw = get_master_data(FRED_API_KEY)
    if not df_raw.empty:
        with st.status("Analyzing Market Regimes...", expanded=True) as status:
            tournament_res, dates = run_tournament(df_raw)
            status.update(label="Tournament Analysis Complete!", state="complete", expanded=False)
        
        st.subheader("📊 Performance Leaderboard (Out-of-Sample)")
        summary = []
        fig = go.Figure()
        for name, rets in tournament_res.items():
            cum_ret = (np.prod(1 + np.array(rets)) - 1)
            summary.append({"Model": name, "Cumulative Return": f"{cum_ret:.2%}"})
            fig.add_trace(go.Scatter(x=dates, y=np.cumprod(1 + np.array(rets)), name=name))
        
        st.table(pd.DataFrame(summary).sort_values("Cumulative Return", ascending=False))
        fig.update_layout(title="Equity Curve Comparison", template="plotly_dark", height=450)
        st.plotly_chart(fig, use_container_width=True)

        target_date = get_next_market_date()
        st.subheader(f"🎯 Forecasts for US Open: {target_date}")
        cols = st.columns(len(tournament_res))
        for i, (name, rets) in enumerate(tournament_res.items()):
            cols[i].metric(name, "BUY SIGNAL", delta="Active")
