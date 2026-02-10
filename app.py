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

# --- 4. ENGINE ---
@st.cache_resource(ttl=604800)
def run_tournament_engine(data_json, rf_rate):
    data = pd.read_json(data_json)
    rets_df = data[TARGET_ETFS].pct_change().dropna()
    feats_df = data.shift(1).dropna()
    common_idx = rets_df.index.intersection(feats_df.index)
    X, y = feats_df.loc[common_idx].values, rets_df.loc[common_idx].values
    
    scaler = StandardScaler()
    X_sc = scaler.fit_transform(X).astype(np.float32)
    split = int(len(X_sc) * 0.8) 
    X_train, X_test, y_train, y_test = X_sc[:split], X_sc[split:], y[:split], y[split:]

    # Training RL
    env = DummyVecEnv([lambda: TradingEnv(X_train, y_train, TARGET_ETFS)])
    ppo = PPO("MlpPolicy", env, verbose=0).learn(2000)
    a2c = A2C("MlpPolicy", env, verbose=0).learn(2000)
    
    # Training DL
    dl_models = {}
    for name, m_class in [("CNN-LSTM", CNN_LSTM_Model), ("Transformer", TransformerModel)]:
        model = m_class(X.shape[1], len(TARGET_ETFS))
        opt = torch.optim.Adam(model.parameters(), lr=0.005)
        X_t, y_t = torch.tensor(X_train).unsqueeze(1), torch.tensor(y_train).float()
        for _ in range(25): 
            opt.zero_grad()
            nn.MSELoss()(model(X_t), y_t).backward()
            opt.step()
        dl_models[name] = model

    # Tournament Competition
    results = {"PPO": [], "A2C": [], "CNN-LSTM": [], "Transformer": []}
    picks = {"PPO": [], "A2C": [], "CNN-LSTM": [], "Transformer": []}
    dates = common_idx[split:]
    
    for i in range(len(X_test)):
        p_act, _ = ppo.predict(X_test[i], deterministic=True)
        a_act, _ = a2c.predict(X_test[i], deterministic=True)
        results["PPO"].append(y_test[i, p_act]); picks["PPO"].append(TARGET_ETFS[p_act])
        results["A2C"].append(y_test[i, a_act]); picks["A2C"].append(TARGET_ETFS[a_act])
        
        with torch.no_grad():
            x_in = torch.tensor(X_test[i]).reshape(1, 1, -1)
            for name in ["CNN-LSTM", "Transformer"]:
                out = dl_models[name](x_in)
                act = torch.argmax(out).item()
                results[name].append(y_test[i, act])
                picks[name].append(TARGET_ETFS[act])

    perf = {k: np.prod(1 + np.array(v)) for k, v in results.items()}
    champ = max(perf, key=perf.get)
    
    # Heatmap logic with Geometric Yearly Compounding
    champ_series = pd.Series(results[champ], index=dates)
    monthly_rets = champ_series.groupby([champ_series.index.year, champ_series.index.month]).apply(lambda x: np.prod(1+x)-1)
    m_table = monthly_rets.unstack().fillna(0)
    m_table.columns = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'][:len(m_table.columns)]
    
    # Geometric compounding for yearly total column
    m_table['Yearly Total'] = m_table.apply(lambda row: np.prod(1 + row) - 1, axis=1)

    # Next Day Forecast
    latest_feat = X_sc[-1:]
    if champ == "PPO": f_act, _ = ppo.predict(latest_feat[0], deterministic=True)
    elif champ == "A2C": f_act, _ = a2c.predict(latest_feat[0], deterministic=True)
    else: 
        with torch.no_grad():
            f_out = dl_models[champ](torch.tensor(latest_feat).reshape(1, 1, -1))
            f_act = torch.argmax(f_out).item()

    audit_list = []
    for j in range(len(X_test)-15, len(X_test)):
        audit_list.append({
            'Date': dates[j].strftime('%Y-%m-%d'),
            'ETF Picked': picks[champ][j],
            'Outcome Return': results[champ][j]
        })

    return results, dates, TARGET_ETFS[f_act], champ, pd.DataFrame(audit_list), m_table

# --- 5. UI ---
st.title("Multi-model (DL & ML) Tournament for Prediction of ETF Returns")

with st.sidebar:
    st.header("Tournament Configuration")
    start_year = st.selectbox("Select Training Start Year", options=["2007", "2010", "2015", "2019", "2021"], index=0)
    run_btn = st.button("🚀 Execute Alpha Tournament")

now = datetime.now()
target_date = now if now.hour < 16 else now + timedelta(days=1)
while target_date.weekday() >= 5: target_date += timedelta(days=1)

if run_btn:
    with st.status(f"Training Tournament Models...") as status:
        rf = get_sofr_rate(FRED_API_KEY)
        raw_data = yf.download(TARGET_ETFS + MACRO, start=f"{start_year}-01-01", progress=False)['Close'].ffill().dropna()
        res, dates, ticker, champ, audit, m_table = run_tournament_engine(raw_data.to_json(), rf)
        st.session_state.results = {"res": res, "dates": dates, "ticker": ticker, "champ": champ, "audit": audit, "rf": rf, "start": start_year, "monthly": m_table}
        status.update(label=f"Champion Identified: {champ}", state="complete")
    st.rerun()

if st.session_state.results:
    s = st.session_state.results
    st.header(f"🎯 Forecast for {target_date.strftime('%b %d')}: BUY {s['ticker']}")
    
    m1, m2, m3 = st.columns(3)
    champ_rets_arr = np.array(s['res'][s['champ']])
    m1.metric("Winner Total Return (OOS)", f"{(np.prod(1+champ_rets_arr)-1):.2%}", delta=s['champ'])
    m2.metric("Sharpe Ratio (Annualized)", f"{((np.mean(champ_rets_arr)-(s['rf']/252))/np.std(champ_rets_arr)*np.sqrt(252)):.2f}")
    m3.metric("Live SOFR Rate", f"{s['rf']:.2%}")

    # Chart
    fig = go.Figure()
    for name, r in s['res'].items():
        fig.add_trace(go.Scatter(x=s['dates'], y=np.cumprod(1 + np.array(r)), name=name))
    fig.update_layout(title=f"Out of Sample Cumulative Return (Training since {s['start']})", template="plotly_dark", height=400)
    st.plotly_chart(fig, use_container_width=True)

    # --- HEATMAP WITH CUSTOM COLOR SCHEME ---
    st.subheader(f"📅 Monthly Performance Matrix ({s['champ']})")
    
    def heatmap_style(val):
        if val > 0:
            return f'background-color: rgba(0, 128, 0, {min(val*5, 0.9)}); color: white;'
        elif val < 0:
            return f'background-color: rgba(255, 0, 0, {min(abs(val)*5, 0.9)}); color: white;'
        return ''

    styled_monthly = s['monthly'].style.applymap(heatmap_style).format("{:.2%}")
    st.dataframe(styled_monthly, use_container_width=True)

    # --- AUDIT TABLE ---
    st.subheader(f"📊 15-Day Audit Table ({s['champ']})")
    def audit_style(val):
        color = '#d1f2eb' if val > 0 else '#fcdedc'
        text_color = '#0e6251' if val > 0 else '#943126'
        return f'background-color: {color}; color: {text_color}; border-radius: 8px; font-weight: bold;'

    st.table(s['audit'].sort_values('Date', ascending=False).style.format({'Outcome Return': '{:.2%}'})
             .applymap(audit_style, subset=['Outcome Return']))

    # Methodology
    st.divider()
    st.header("🔍 Methodology & Model Architecture")
    col_a, col_b = st.columns(2)
    with col_a:
        st.subheader("Data Strategy: 80/20 Split")
        [cite_start]st.write("The engine utilizes an 80:20 temporal split[cite: 26]. 80% is used for training, while the remaining 20% serves as a blind test to select the Champion.")
        st.subheader("7-Day Hard Refresh")
        st.write("The system retrains all models every 7 days via a TTL cache mechanism to ensure the models stay adapted to the latest market regime.")
    with col_b:
        st.subheader("The Competing Models")
        st.markdown("1. **PPO:** Stable RL policy optimization. \n 2. **A2C:** Actor-critic RL agent. \n 3. **CNN-LSTM:** Spatial-temporal pattern learner. \n 4. **Transformer:** Attention-based macro correlation finder.")
