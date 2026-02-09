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
st.set_page_config(page_title="Multi-model ETF Tournament", layout="wide")

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

# --- 3. ANALYTICS UTILITIES ---
def get_sofr_rate(api_key):
    """Fetches live SOFR from FRED API or falls back to current bank rate."""
    if not api_key: return 0.053 
    url = f"https://api.stlouisfed.org/fred/series/observations?series_id=SOFR&api_key={api_key}&file_type=json"
    try:
        r = requests.get(url, timeout=10).json()
        return float(r['observations'][-1]['value']) / 100
    except: return 0.053

# --- 4. RL ENVIRONMENT ---
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

# --- 5. TOURNAMENT ENGINE ---
@st.cache_resource(ttl=604800) # Auto-retrain every 7 days
def execute_tournament(data_json, rf_rate):
    data = pd.read_json(data_json)
    rets_df = data[TARGET_ETFS].pct_change().dropna()
    feats_df = data.shift(1).dropna()
    common_idx = rets_df.index.intersection(feats_df.index)
    X, y = feats_df.loc[common_idx].values, rets_df.loc[common_idx].values
    
    scaler = StandardScaler()
    X_sc = scaler.fit_transform(X).astype(np.float32)
    split = int(len(X_sc) * 0.8)
    X_train, X_test, y_train, y_test = X_sc[:split], X_sc[split:], y[:split], y[split:]

    # A. Train RL Models
    env = DummyVecEnv([lambda: TradingEnv(X_train, y_train, TARGET_ETFS)])
    ppo = PPO("MlpPolicy", env, verbose=0).learn(1500)
    a2c = A2C("MlpPolicy", env, verbose=0).learn(1500)

    # B. Train DL Models
    dl_models = {}
    for name, m_class in [("CNN-LSTM", CNN_LSTM_Model), ("Transformer", TransformerModel)]:
        model = m_class(X.shape[1], len(TARGET_ETFS))
        optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
        X_t, y_t = torch.tensor(X_train).unsqueeze(1), torch.tensor(y_train).float()
        for _ in range(30):
            optimizer.zero_grad()
            nn.MSELoss()(model(X_t), y_t).backward()
            optimizer.step()
        dl_models[name] = model

    # C. Out-of-Sample Performance
    results = {"PPO": [], "A2C": [], "CNN-LSTM": [], "Transformer": []}
    dates = common_idx[split:]
    
    # Store actions for the audit
    actions_audit = {"PPO": [], "A2C": [], "CNN-LSTM": [], "Transformer": []}

    for i in range(len(X_test)):
        p_act, _ = ppo.predict(X_test[i], deterministic=True)
        a_act, _ = a2c.predict(X_test[i], deterministic=True)
        results["PPO"].append(y_test[i, p_act]); actions_audit["PPO"].append(TARGET_ETFS[p_act])
        results["A2C"].append(y_test[i, a_act]); actions_audit["A2C"].append(TARGET_ETFS[a_act])
        
        with torch.no_grad():
            x_in = torch.tensor(X_test[i]).reshape(1, 1, -1)
            for name in ["CNN-LSTM", "Transformer"]:
                out = dl_models[name](x_in)
                act = torch.argmax(out).item()
                results[name].append(y_test[i, act])
                actions_audit[name].append(TARGET_ETFS[act])

    # D. Selection: Winner takes all
    perf = {k: np.prod(1 + np.array(v)) for k, v in results.items()}
    champ = max(perf, key=perf.get)
    
    # E. Next-Day Forecast
    latest_feat = X_sc[-1:]
    if champ == "PPO": final_act, _ = ppo.predict(latest_feat[0], deterministic=True)
    elif champ == "A2C": final_act, _ = a2c.predict(latest_feat[0], deterministic=True)
    else: 
        with torch.no_grad():
            f_out = dl_models[champ](torch.tensor(latest_feat).reshape(1, 1, -1))
            final_act = torch.argmax(f_out).item()

    # F. 15-Day Audit Table
    audit_list = []
    for j in range(len(X_test)-15, len(X_test)):
        audit_list.append({
            'Date': dates[j].strftime('%Y-%m-%d'),
            'Model Pick': actions_audit[champ][j],
            'Outcome Return': results[champ][j]
        })

    return results, dates, TARGET_ETFS[final_act], champ, pd.DataFrame(audit_list)

# --- 6. UI ---
st.title("Multi-model (DL & ML) Tournament for Prediction of ETF Returns")

st.markdown("""
### Model Descriptions
* **PPO (Proximal Policy Optimization):** An RL agent that learns a robust trading policy by balancing exploration and stable updates.
* **A2C (Advantage Actor-Critic):** A synchronous RL model that uses an 'Actor' to trade and a 'Critic' to evaluate and improve performance.
* **CNN-LSTM:** A hybrid Deep Learning model where CNNs extract local patterns in price data and LSTMs capture long-term trends.
* **Transformer:** Uses self-attention mechanisms to weigh the importance of different historical macro-economic signals simultaneously.
""")

# Market Date Calculation
now = datetime.now()
target_date = now if now.hour < 16 else now + timedelta(days=1)
while target_date.weekday() >= 5: target_date += timedelta(days=1)

if st.button("🚀 Execute Alpha Tournament"):
    with st.status("Analyzing Market Dynamics...") as status:
        rf = get_sofr_rate(FRED_API_KEY)
        raw_data = yf.download(TARGET_ETFS + MACRO, start="2019-01-01", progress=False)['Close'].ffill().dropna()
        res, dates, ticker, champ, audit = execute_tournament(raw_data.to_json(), rf)
        st.session_state.results = {"res": res, "dates": dates, "ticker": ticker, "champ": champ, "audit": audit, "rf": rf}
        status.update(label=f"Champion Identified: {champ}", state="complete")
    st.rerun()

if st.session_state.results:
    s = st.session_state.results
    st.header(f"🎯 Forecast for {target_date.strftime('%b %d')}: BUY {s['ticker']}")
    st.caption("Sharpe Ratio is calculated using live SOFR from FRED")

    m1, m2, m3 = st.columns(3)
    champ_rets = np.array(s['res'][s['champ']])
    m1.metric("Champion Return (OOS)", f"{(np.prod(1+champ_rets)-1):.2%}", delta=s['champ'])
    m2.metric("Champion Sharpe Ratio", f"{((np.mean(champ_rets)-(s['rf']/252))/np.std(champ_rets)*np.sqrt(252)):.2f}")
    m3.metric("Live SOFR Rate", f"{s['rf']:.2%}")

    fig = go.Figure()
    for name, r in s['res'].items():
        fig.add_trace(go.Scatter(x=s['dates'], y=np.cumprod(1 + np.array(r)), name=name))
    fig.update_layout(title="Out of Sample Cumulative Return", template="plotly_dark", xaxis_title="Date", yaxis_title="Growth of $1")
    st.plotly_chart(fig, use_container_width=True)

    st.subheader(f"📅 Last 15 Sessions Audit ({s['champ']})")
    st.table(s['audit'].sort_values('Date', ascending=False).style.format({'Outcome Return': '{:.2%}'}))
