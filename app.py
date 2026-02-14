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
import pandas_market_calendars as mcal

# --- 1. SETTINGS & STATE ---
st.set_page_config(page_title="Alpha Tournament Pro", layout="wide")

if 'results' not in st.session_state: st.session_state.results = None

TARGET_ETFS = ['TLT', 'TBT', 'VNQ', 'GLD', 'SLV']
MACRO = ['^VIX', '^TNX', 'DX-Y.NYB']
FRED_API_KEY = st.secrets.get("FRED_API_KEY")

# --- 2. MODEL ARCHITECTURES ---
class CNN_LSTM_Model(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.conv = nn.Conv1d(input_dim, 64, kernel_size=3, padding=1)
        self.lstm = nn.LSTM(64, 128, batch_first=True)
        self.fc = nn.Linear(128, output_dim)
    def forward(self, x):
        x = x.transpose(1, 2)
        x = torch.relu(self.conv(x)).transpose(1, 2)
        _, (hn, _) = self.lstm(x)
        return self.fc(hn[-1])

class TransformerModel(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, 64)
        encoder_layer = nn.TransformerEncoderLayer(d_model=64, nhead=4, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.fc = nn.Linear(64, output_dim)
    def forward(self, x):
        x = self.input_proj(x)
        x = self.transformer(x)
        return self.fc(x[:, -1, :])

# --- 3. UTILITIES ---
def get_sofr_rate(api_key):
    if not api_key: return 0.053 
    url = f"https://api.stlouisfed.org/fred/series/observations?series_id=SOFR&api_key={api_key}&file_type=json"
    try:
        r = requests.get(url, timeout=10).json()
        return float(r['observations'][-1]['value']) / 100
    except: return 0.053

def get_next_trading_day():
    nyse = mcal.get_calendar('NYSE')
    today = pd.Timestamp.now(tz='America/New_York').normalize()
    schedule = nyse.schedule(start_date=today, end_date=today + timedelta(days=10))
    valid_days = mcal.date_range(schedule, frequency='1D')
    for day in valid_days:
        if day.normalize() > today:
            return day.strftime('%Y-%m-%d')
    return (today + timedelta(days=1)).strftime('%Y-%m-%d')

class TradingEnv(gym.Env):
    def __init__(self, features, returns, etfs, tcost_bps):
        super().__init__()
        self.features, self.returns, self.etfs = features, returns, etfs
        self.tcost = tcost_bps / 10000 
        self.action_space = gym.spaces.Discrete(len(etfs))
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(features.shape[1],), dtype=np.float32)
        self.current_step = 0
        self.last_action = None
    def reset(self, seed=None):
        self.current_step = 0
        self.last_action = None
        return self.features[0], {}
    def step(self, action):
        raw_reward = float(self.returns[self.current_step, action])
        penalty = 0
        if self.last_action is not None and action != self.last_action:
            penalty = self.tcost
        reward = raw_reward - penalty
        self.last_action = action
        self.current_step += 1
        done = self.current_step >= len(self.features) - 1
        return self.features[self.current_step], reward, done, False, {}

# --- 4. ENGINE ---
@st.cache_resource(ttl=604800)
def run_tournament_engine(data_json, rf_rate, tcost_bps, start_year):
    data = pd.read_json(data_json)
    rets_df = data[TARGET_ETFS].pct_change().dropna()
    feats_df = data.shift(1).dropna()
    common_idx = rets_df.index.intersection(feats_df.index)
    X, y = feats_df.loc[common_idx].values, rets_df.loc[common_idx].values
    
    scaler = StandardScaler()
    X_sc = scaler.fit_transform(X).astype(np.float32)
    split = int(len(X_sc) * 0.8) 
    X_train, X_test, y_train, y_test = X_sc[:split], X_sc[split:], y[:split], y[split:]

    env = DummyVecEnv([lambda: TradingEnv(X_train, y_train, TARGET_ETFS, tcost_bps)])
    ppo = PPO("MlpPolicy", env, verbose=0).learn(5000)
    a2c = A2C("MlpPolicy", env, verbose=0).learn(5000)
    
    dl_models = {}
    for name, m_class in [("CNN-LSTM", CNN_LSTM_Model), ("Transformer", TransformerModel)]:
        model = m_class(X.shape[1], len(TARGET_ETFS))
        opt = torch.optim.Adam(model.parameters(), lr=0.005)
        X_t, y_t = torch.tensor(X_train).unsqueeze(1), torch.tensor(y_train).float()
        for _ in range(50): 
            opt.zero_grad()
            nn.MSELoss()(model(X_t), y_t).backward()
            opt.step()
        dl_models[name] = model

    results = {"PPO": [], "A2C": [], "CNN-LSTM": [], "Transformer": []}
    dates = common_idx[split:]
    tcost_dec = tcost_bps / 10000

    for name in results.keys():
        last_pick = None
        for i in range(len(X_test)):
            if name == "PPO": act, _ = ppo.predict(X_test[i], deterministic=True)
            elif name == "A2C": act, _ = a2c.predict(X_test[i], deterministic=True)
            else:
                with torch.no_grad():
                    out = dl_models[name](torch.tensor(X_test[i]).reshape(1, 1, -1))
                    act = torch.argmax(out).item()
            day_ret = y_test[i, act]
            if last_pick is not None and act != last_pick: day_ret -= tcost_dec
            results[name].append(day_ret)
            last_pick = act

    # OOS period calculation
    oos_start_year = dates[0].year
    oos_end_year = dates[-1].year
    oos_years = f"{oos_start_year}-{oos_end_year}" if oos_start_year != oos_end_year else str(oos_start_year)

    # Logic for ranking
    recency_window = 15
    recency_scores = {n: np.sum(np.array(r[-recency_window:]) > 0) / recency_window for n, r in results.items()}
    perf = {k: ((np.prod(1 + np.array(results[k])) - 1) * 0.7) + (recency_scores[k] * 0.3) for k in results.keys()}
    
    # Sort to find Champion and Runner-Up
    sorted_models = sorted(perf.items(), key=lambda x: x[1], reverse=True)
    champ, runner_up = sorted_models[0][0], sorted_models[1][0]
    
    # Forecasts for both
    forecasts = {}
    latest_feat = X_sc[-1:]
    for m in [champ, runner_up]:
        if m == "PPO": act, _ = ppo.predict(latest_feat[0], deterministic=True)
        elif m == "A2C": act, _ = a2c.predict(latest_feat[0], deterministic=True)
        else:
            with torch.no_grad():
                f_out = dl_models[m](torch.tensor(latest_feat).reshape(1, 1, -1))
                act = torch.argmax(f_out).item()
        forecasts[m] = TARGET_ETFS[act]

    # Process Table (Champion only as requested)
    champ_series = pd.Series(results[champ], index=dates)
    monthly_rets = champ_series.groupby([champ_series.index.year, champ_series.index.month]).apply(lambda x: np.prod(1+x)-1)
    m_table = monthly_rets.unstack().fillna(0)
    m_table.columns = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'][:len(m_table.columns)]
    m_table['Yearly Total'] = m_table.apply(lambda row: np.prod(1 + row) - 1, axis=1)

    return results, dates, forecasts, champ, runner_up, m_table, recency_scores, oos_years

# --- 5. UI ---
st.title("Alpha Tournament Pro: Multi-model ETF Forecast")

with st.sidebar:
    st.header("Tournament Configuration")
    start_year = st.selectbox("Select Training Start Year", options=["2007", "2010", "2015", "2019", "2021"], index=0)
    t_cost = st.slider("Transaction Cost (bps)", min_value=0, max_value=100, value=10, step=5)
    run_btn = st.button("🚀 Execute Alpha Tournament")

if run_btn:
    with st.status(f"Training Tournament Models...") as status:
        rf = get_sofr_rate(FRED_API_KEY)
        raw_data = yf.download(TARGET_ETFS + MACRO, start=f"{start_year}-01-01", progress=False)['Close'].ffill().dropna()
        res, dates, fcasts, champ, runner, m_table, r_scores, oos_years = run_tournament_engine(raw_data.to_json(), rf, t_cost, start_year)
        next_trade_day = get_next_trading_day()
        st.session_state.results = {"res": res, "dates": dates, "fcasts": fcasts, "champ": champ, "runner": runner, "rf": rf, "monthly": m_table, "recency": r_scores, "t_cost": t_cost, "oos_years": oos_years, "next_day": next_trade_day}
        status.update(label=f"Tournament Complete!", state="complete")
    st.rerun()

if st.session_state.results:
    s = st.session_state.results
    
    # --- CHAMPION ROW ---
    st.subheader(f"🏆 Champion: {s['champ']}")
    c1, c2, c3, c4 = st.columns(4)
    c_rets = np.array(s['res'][s['champ']])
    c1.metric(f"PREDICTION", s['fcasts'][s['champ']], delta=f"Valid: {s['next_day']}")
    c2.metric("Total Return (Net)", f"{(np.prod(1+c_rets)-1):.2%}", delta=f"OOS: {s['oos_years']}")
    c3.metric("Sharpe (Annualized)", f"{((np.mean(c_rets)-(s['rf']/252))/np.std(c_rets)*np.sqrt(252)):.2f}", delta=f"SOFR: {s['rf']:.2%}", delta_color="normal")
    c4.metric("Recency Score (15d)", f"{s['recency'][s['champ']]:.0%}")

    # --- RUNNER UP ROW ---
    st.subheader(f"🥈 Runner-Up: {s['runner']}")
    r1, r2, r3, r4 = st.columns(4)
    r_rets = np.array(s['res'][s['runner']])
    r1.metric(f"PREDICTION", s['fcasts'][s['runner']], delta=f"Valid: {s['next_day']}")
    r2.metric("Total Return (Net)", f"{(np.prod(1+r_rets)-1):.2%}", delta=f"OOS: {s['oos_years']}")
    r3.metric("Sharpe (Annualized)", f"{((np.mean(r_rets)-(s['rf']/252))/np.std(r_rets)*np.sqrt(252)):.2f}", delta=f"SOFR: {s['rf']:.2%}", delta_color="normal")
    r4.metric("Recency Score (15d)", f"{s['recency'][s['runner']]:.0%}")

    st.divider()
    # Charts and Tables follow for Champion (as original UI)
    fig = go.Figure()
    for name, r in s['res'].items(): fig.add_trace(go.Scatter(x=s['dates'], y=np.cumprod(1 + np.array(r)), name=name))
    fig.update_layout(title="Net Return Performance", template="plotly_dark", height=400)
    st.plotly_chart(fig, use_container_width=True)

    st.subheader(f"📅 Monthly Matrix ({s['champ']})")
    st.dataframe(s['monthly'].style.format("{:.2%}"), use_container_width=True)

    st.divider()
    st.header("🔍 Methodology")
    st.info("""
    **Recency Score (15d):** The 'Hit Rate' of a model over the last 15 trading sessions (% of positive days). 
    The engine blends this (30%) with long-term OOS performance (70%) to rank the models.
    """)
    
    st.subheader("🤖 Model Descriptions")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**PPO (Proximal Policy Optimization)**")
        st.caption("A state-of-the-art reinforcement learning algorithm that learns optimal trading policies through trial and error. PPO uses an actor-critic architecture and clips policy updates to ensure stable learning. Trained for 5,000 timesteps on historical data.")
        
        st.markdown("**CNN-LSTM**")
        st.caption("A hybrid deep learning architecture combining Convolutional Neural Networks (for pattern extraction) with Long Short-Term Memory networks (for sequential dependencies). Captures both spatial and temporal features in market data. Trained for 50 epochs.")
    
    with col2:
        st.markdown("**A2C (Advantage Actor-Critic)**")
        st.caption("A reinforcement learning algorithm that simultaneously learns a value function (critic) and policy (actor). More sample-efficient than standard policy gradients and learns to maximize risk-adjusted returns. Trained for 5,000 timesteps.")
        
        st.markdown("**Transformer**")
        st.caption("An attention-based neural network architecture that weighs the importance of different time steps in the input sequence. Uses multi-head self-attention mechanisms to capture complex temporal relationships in market data. Trained for 50 epochs.")
