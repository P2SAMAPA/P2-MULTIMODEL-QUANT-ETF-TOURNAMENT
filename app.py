import streamlit as st
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
from datasets import load_dataset
import os
from io import StringIO

# --- 1. SETTINGS & STATE ---
st.set_page_config(page_title="Alpha Tournament Pro", layout="wide")

if 'results' not in st.session_state: st.session_state.results = None

TARGET_ETFS = ['TLT', 'TBT', 'VNQ', 'GLD', 'SLV']

# Get secrets from HF Spaces
FRED_API_KEY = os.environ.get("FRED_API_KEY")
HF_TOKEN = os.environ.get("HF_KEY")
HF_DATASET_REPO = "P2SAMAPA/my-etf-data"

# --- 2. MODEL ARCHITECTURES ---
class CNN_LSTM_Model(nn.Module):
    def __init__(self, input_dim, output_dim, seq_len):
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
    def __init__(self, input_dim, output_dim, seq_len):
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
    try:
        nyse = mcal.get_calendar('NYSE')
        today = pd.Timestamp.now(tz='America/New_York').normalize()
        schedule = nyse.schedule(start_date=today, end_date=today + timedelta(days=10))
        valid_days = mcal.date_range(schedule, frequency='1D')
        for day in valid_days:
            if day.normalize() > today:
                return day.strftime('%Y-%m-%d')
    except:
        pass
    return (pd.Timestamp.now() + timedelta(days=1)).strftime('%Y-%m-%d')

def analyze_period_characteristics(returns_df, test_start_idx):
    """Analyze the OOS period to understand if returns are reasonable"""
    test_returns = returns_df.iloc[test_start_idx:]
    
    stats = {}
    for etf in returns_df.columns:
        etf_rets = test_returns[etf]
        stats[etf] = {
            'mean_daily': etf_rets.mean(),
            'std_daily': etf_rets.std(),
            'sharpe': etf_rets.mean() / etf_rets.std() * np.sqrt(252) if etf_rets.std() > 0 else 0,
            'max_daily': etf_rets.max(),
            'min_daily': etf_rets.min(),
            'total_return': (1 + etf_rets).prod() - 1
        }
    
    # Check TLT vs TBT correlation (should be negative)
    if 'TLT' in returns_df.columns and 'TBT' in returns_df.columns:
        tlt_tbt_corr = test_returns['TLT'].corr(test_returns['TBT'])
        stats['tlt_tbt_correlation'] = tlt_tbt_corr
    
    return stats

def calculate_hold_period_returns(predictions, returns_df, tcost_bps, hold_periods=[1, 3, 5]):
    """Calculate annualized returns for different hold periods accounting for transaction costs"""
    tcost_dec = tcost_bps / 10000
    hold_returns = {}
    
    for hold_days in hold_periods:
        period_returns = []
        i = 0
        while i < len(predictions) - hold_days:
            etf = predictions[i]
            # Get the next hold_days returns for this ETF
            future_rets = returns_df[etf].iloc[i:i+hold_days]
            total_ret = np.prod(1 + future_rets) - 1
            # Subtract transaction cost for this trade
            net_ret = total_ret - tcost_dec
            period_returns.append(net_ret)
            i += hold_days  # Skip to next period
        
        if period_returns:
            avg_period_return = np.mean(period_returns)
            # Calculate annualized return: (1 + avg_return)^(252/hold_days) - 1
            num_periods_per_year = 252 / hold_days
            annualized_return = (1 + avg_period_return) ** num_periods_per_year - 1
            
            hold_returns[hold_days] = {
                'avg_return': avg_period_return,
                'annualized': annualized_return,
                'num_trades_per_year': num_periods_per_year
            }
        else:
            hold_returns[hold_days] = {
                'avg_return': 0,
                'annualized': 0,
                'num_trades_per_year': 0
            }
    
    # Find optimal hold period (highest annualized return)
    optimal_period = max(hold_returns.keys(), key=lambda k: hold_returns[k]['annualized'])
    
    return hold_returns, optimal_period

def load_data_from_hf(start_year, hf_token, dataset_repo):
    """Load data from HuggingFace dataset"""
    try:
        # Load the dataset from HF
        dataset = load_dataset(dataset_repo, split='train', token=hf_token)
        
        # Convert to pandas DataFrame
        df = dataset.to_pandas()
        
        # Set Date as index
        if 'Date' in df.columns:
            df['Date'] = pd.to_datetime(df['Date'])
            df = df.set_index('Date')
        elif 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])
            df = df.set_index('date')
        
        # Filter data from start_year onwards
        df = df[df.index >= f'{start_year}-01-01']
        df = df.sort_index()
        
        # Select the columns we need: returns and features
        ret_cols = [f'{etf}_Ret' for etf in TARGET_ETFS]
        
        # Feature columns: ETF technical indicators + macro signals
        feature_cols = []
        for etf in TARGET_ETFS:
            feature_cols.extend([f'{etf}_MA20', f'{etf}_Vol'])
        
        # Add macro/market signals
        macro_cols = ['UNRATE', 'CPI', 'VIX', 'TNX', 'DXY', 'AU_CU_Ratio', 'AU_CU_Trend']
        feature_cols.extend(macro_cols)
        
        # Check all columns exist
        all_cols = ret_cols + feature_cols
        missing = [c for c in all_cols if c not in df.columns]
        if missing:
            st.warning(f"Missing columns in dataset: {missing}")
            # Remove missing columns from feature list
            feature_cols = [c for c in feature_cols if c in df.columns]
        
        # Create returns dataframe (for targets)
        returns_df = df[ret_cols].copy()
        returns_df.columns = TARGET_ETFS  # Rename to simple ETF names
        
        # Create features dataframe
        features_df = df[feature_cols].copy()
        
        # Forward fill and drop NaN
        returns_df = returns_df.ffill().dropna()
        features_df = features_df.ffill().dropna()
        
        # Align indices
        common_idx = returns_df.index.intersection(features_df.index)
        returns_df = returns_df.loc[common_idx]
        features_df = features_df.loc[common_idx]
        
        if len(returns_df) < 100:
            st.error(f"Insufficient data after filtering: {len(returns_df)} rows")
            return None, None
        
        return (features_df, returns_df), "HuggingFace Dataset"
        
    except Exception as e:
        st.error(f"Error loading from HF dataset: {str(e)}")
        import traceback
        st.code(traceback.format_exc())
        return None, None

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
def run_tournament_engine(features_json, returns_json, rf_rate, tcost_bps, start_year, data_source):
    features_df = pd.read_json(StringIO(features_json))
    returns_df = pd.read_json(StringIO(returns_json))
    
    # Track data transformations for diagnostics
    raw_start = features_df.index[0].strftime('%Y-%m-%d')
    raw_end = features_df.index[-1].strftime('%Y-%m-%d')
    raw_rows = len(features_df)
    
    # CRITICAL: Clean infinities and NaNs BEFORE any processing
    features_df = features_df.replace([np.inf, -np.inf], np.nan)
    returns_df = returns_df.replace([np.inf, -np.inf], np.nan)
    features_df = features_df.ffill().bfill().dropna()
    returns_df = returns_df.ffill().bfill().dropna()
    
    # Align after cleaning
    common_idx = features_df.index.intersection(returns_df.index)
    features_df = features_df.loc[common_idx]
    returns_df = returns_df.loc[common_idx]
    
    if len(features_df) < 100:
        raise ValueError(f"Insufficient data after cleaning: {len(features_df)} rows")
    
    # Calculate momentum features for different lookback periods
    lookback_periods = [30, 45, 60]
    momentum_dict = {}
    
    for period in lookback_periods:
        momentum = features_df.pct_change(period)
        momentum = momentum.replace([np.inf, -np.inf], np.nan)
        momentum = momentum.ffill().bfill()
        momentum_dict[f'momentum_{period}d'] = momentum
    
    # Test each lookback period to find best performing one
    best_lookback = None
    best_score = -np.inf
    
    for period in lookback_periods:
        momentum_data = momentum_dict[f'momentum_{period}d']
        combined_data = pd.concat([features_df, momentum_data.add_suffix(f'_mom{period}')], axis=1)
        combined_data = combined_data.replace([np.inf, -np.inf], np.nan).dropna()
        
        if len(combined_data) < 100:
            continue
        
        # Quick validation score using correlation
        aligned_returns = returns_df.loc[combined_data.index]
        mom_cols = [col for col in combined_data.columns if f'_mom{period}' in col]
        if len(aligned_returns) > 0 and len(mom_cols) > 0:
            corr = combined_data[mom_cols].corrwith(aligned_returns.iloc[:, 0])
            corr = corr.replace([np.inf, -np.inf], 0).fillna(0)
            score = np.abs(corr).mean()
            if score > best_score:
                best_score = score
                best_lookback = period
    
    # Use best lookback period or default to 45
    if best_lookback is None:
        best_lookback = 45
    
    # Build final dataset with best lookback
    momentum_data = momentum_dict[f'momentum_{best_lookback}d']
    features_with_momentum = pd.concat([features_df, momentum_data.add_suffix(f'_mom{best_lookback}')], axis=1)
    features_with_momentum = features_with_momentum.replace([np.inf, -np.inf], np.nan).dropna()
    
    # Align features and returns
    common_idx = features_with_momentum.index.intersection(returns_df.index)
    X = features_with_momentum.loc[common_idx].values
    y = returns_df.loc[common_idx].values
    
    # Final check for inf/nan
    if not np.isfinite(X).all():
        st.warning("Cleaning remaining non-finite values in features...")
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    
    if not np.isfinite(y).all():
        st.warning("Cleaning remaining non-finite values in returns...")
        y = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Diagnostics
    after_processing_start = common_idx[0].strftime('%Y-%m-%d')
    after_processing_rows = len(common_idx)
    
    scaler = StandardScaler()
    X_sc = scaler.fit_transform(X).astype(np.float32)
    split = int(len(X_sc) * 0.8)
    split_date = common_idx[split].strftime('%Y-%m-%d')
    
    # Analyze OOS period characteristics
    period_stats = analyze_period_characteristics(returns_df.loc[common_idx], split)
    
    # Create sequences for deep learning models
    seq_len = best_lookback
    X_seq = []
    y_seq = []
    for i in range(seq_len, len(X_sc)):
        X_seq.append(X_sc[i-seq_len:i])
        y_seq.append(y[i])
    X_seq = np.array(X_seq)
    y_seq = np.array(y_seq)
    
    # Adjust split for sequences
    split_seq = int(len(X_seq) * 0.8)
    X_train, X_test = X_seq[:split_seq], X_seq[split_seq:]
    y_train, y_test = y_seq[:split_seq], y_seq[split_seq:]
    
    # For RL models, use flattened current features
    X_train_flat = X_sc[:split]
    X_test_flat = X_sc[split:]
    y_train_rl = y[:split]
    y_test_rl = y[split:]
    
    env = DummyVecEnv([lambda: TradingEnv(X_train_flat, y_train_rl, TARGET_ETFS, tcost_bps)])
    ppo = PPO("MlpPolicy", env, verbose=0).learn(5000)
    a2c = A2C("MlpPolicy", env, verbose=0).learn(5000)
    
    dl_models = {}
    for name, m_class in [("CNN-LSTM", CNN_LSTM_Model), ("Transformer", TransformerModel)]:
        model = m_class(X.shape[1], len(TARGET_ETFS), seq_len)
        opt = torch.optim.Adam(model.parameters(), lr=0.005)
        X_t, y_t = torch.tensor(X_train).float(), torch.tensor(y_train).float()
        for _ in range(50): 
            opt.zero_grad()
            nn.MSELoss()(model(X_t), y_t).backward()
            opt.step()
        dl_models[name] = model

    results = {"PPO": [], "A2C": [], "CNN-LSTM": [], "Transformer": []}
    predictions = {"PPO": [], "A2C": [], "CNN-LSTM": [], "Transformer": []}
    test_dates = common_idx[split:]
    tcost_dec = tcost_bps / 10000

    # PPO and A2C predictions
    for name in ["PPO", "A2C"]:
        last_pick = None
        for i in range(len(X_test_flat)):
            if name == "PPO": 
                act, _ = ppo.predict(X_test_flat[i], deterministic=True)
            else:
                act, _ = a2c.predict(X_test_flat[i], deterministic=True)
            
            predictions[name].append(TARGET_ETFS[act])
            day_ret = y_test_rl[i, act]
            if last_pick is not None and act != last_pick: 
                day_ret -= tcost_dec
            results[name].append(day_ret)
            last_pick = act
    
    # Deep learning model predictions
    for name in ["CNN-LSTM", "Transformer"]:
        last_pick = None
        for i in range(len(X_test)):
            with torch.no_grad():
                out = dl_models[name](torch.tensor(X_test[i]).unsqueeze(0).float())
                act = torch.argmax(out).item()
            
            predictions[name].append(TARGET_ETFS[act])
            day_ret = y_test[i, act]
            if last_pick is not None and act != last_pick:
                day_ret -= tcost_dec
            results[name].append(day_ret)
            last_pick = act

    # OOS period calculation
    oos_start_year = test_dates[0].year
    oos_end_year = test_dates[-1].year
    oos_years = f"{oos_start_year}-{oos_end_year}" if oos_start_year != oos_end_year else str(oos_start_year)

    # Logic for ranking
    recency_window = 15
    recency_scores = {n: np.sum(np.array(r[-recency_window:]) > 0) / recency_window for n, r in results.items()}
    perf = {k: ((np.prod(1 + np.array(results[k])) - 1) * 0.7) + (recency_scores[k] * 0.3) for k in results.keys()}
    
    # Sort to find Champion and Runner-Up
    sorted_models = sorted(perf.items(), key=lambda x: x[1], reverse=True)
    champ, runner_up = sorted_models[0][0], sorted_models[1][0]
    
    # Forecasts for both with optimal hold period
    forecasts = {}
    latest_feat_flat = X_sc[-1:]
    latest_feat_seq = X_seq[-1:]
    
    # Create returns dataframe for hold period calculation (aligned with test dates)
    oos_returns_df = returns_df.loc[test_dates]
    
    for m in [champ, runner_up]:
        if m == "PPO": 
            act, _ = ppo.predict(latest_feat_flat[0], deterministic=True)
        elif m == "A2C": 
            act, _ = a2c.predict(latest_feat_flat[0], deterministic=True)
        else:
            with torch.no_grad():
                f_out = dl_models[m](torch.tensor(latest_feat_seq).float())
                act = torch.argmax(f_out).item()
        
        etf_pred = TARGET_ETFS[act]
        
        # Calculate hold period returns and find optimal
        hold_stats, optimal_hold = calculate_hold_period_returns(predictions[m], oos_returns_df, tcost_bps, [1, 3, 5])
        
        forecasts[m] = {
            'etf': etf_pred,
            'hold_periods': hold_stats,
            'optimal_hold': optimal_hold
        }

    # Process Table (Champion only)
    champ_series = pd.Series(results[champ], index=test_dates)
    monthly_rets = champ_series.groupby([champ_series.index.year, champ_series.index.month]).apply(lambda x: np.prod(1+x)-1)
    m_table = monthly_rets.unstack().fillna(0)
    m_table.columns = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'][:len(m_table.columns)]
    m_table['Yearly Total'] = m_table.apply(lambda row: np.prod(1 + row) - 1, axis=1)

    # Diagnostics info
    diagnostics = {
        'requested_start': f"{start_year}-01-01",
        'raw_start': raw_start,
        'raw_end': raw_end,
        'raw_rows': raw_rows,
        'actual_start': after_processing_start,
        'processed_rows': after_processing_rows,
        'split_date': split_date,
        'data_source': data_source,
        'best_lookback': best_lookback,
        'num_features': X.shape[1],
        'train_test_split': '80/20',
        'period_stats': period_stats
    }

    return results, test_dates, forecasts, champ, runner_up, m_table, recency_scores, oos_years, diagnostics

# --- 5. UI ---
st.title("Alpha Tournament Pro: Multi-model ETF Forecast")

with st.sidebar:
    st.header("Tournament Configuration")
    
    start_year = st.slider(
        "Select Training Start Year", 
        min_value=2008, 
        max_value=2024,
        value=2015,
        step=1,
        help="Choose the year from which to start training the models"
    )
    
    t_cost = st.slider("Transaction Cost (bps)", min_value=0, max_value=100, value=10, step=5)
    run_btn = st.button("🚀 Execute Alpha Tournament")

if run_btn:
    with st.status(f"Training Tournament Models...") as status:
        try:
            rf = get_sofr_rate(FRED_API_KEY)
            data_tuple = load_data_from_hf(start_year, HF_TOKEN, HF_DATASET_REPO)
            
            if data_tuple[0] is None:
                st.error("Failed to load data from HuggingFace dataset")
                st.stop()
            
            (features_df, returns_df), data_src = data_tuple
            
            res, dates, fcasts, champ, runner, m_table, r_scores, oos_years, diag = run_tournament_engine(
                features_df.to_json(), 
                returns_df.to_json(), 
                rf, t_cost, start_year, data_src
            )
            next_trade_day = get_next_trading_day()
            st.session_state.results = {
                "res": res, "dates": dates, "fcasts": fcasts, "champ": champ, "runner": runner, 
                "rf": rf, "monthly": m_table, "recency": r_scores, "t_cost": t_cost, 
                "oos_years": oos_years, "next_day": next_trade_day, "diagnostics": diag
            }
            status.update(label=f"Tournament Complete!", state="complete")
        except Exception as e:
            status.update(label=f"Error: {str(e)}", state="error")
            st.error(f"Failed to run tournament: {str(e)}")
            import traceback
            st.code(traceback.format_exc())
            st.stop()
    st.rerun()

if st.session_state.results:
    s = st.session_state.results
    
    # --- CHAMPION ROW ---
    st.subheader(f"🏆 Champion: {s['champ']}")
    c1, c2, c3, c4 = st.columns(4)
    c_rets = np.array(s['res'][s['champ']])
    
    # Calculate annualized return using ACTUAL trading days
    total_return_c = np.prod(1+c_rets) - 1
    num_trading_days_c = len(c_rets)
    annualized_return_c = (1 + total_return_c) ** (252 / num_trading_days_c) - 1
    
    # Get optimal hold period
    optimal_hold_c = s['fcasts'][s['champ']]['optimal_hold']
    
    c1.metric(f"PREDICTION", s['fcasts'][s['champ']]['etf'], delta=f"Hold: {optimal_hold_c}d")
    c2.metric("Annualized Return (Net)", f"{annualized_return_c:.2%}", delta=f"OOS: {s['oos_years']}")
    c3.metric("Sharpe (Annualized)", f"{((np.mean(c_rets)-(s['rf']/252))/np.std(c_rets)*np.sqrt(252)):.2f}", delta=f"SOFR: {s['rf']:.2%}", delta_color="normal")
    c4.metric("Recency Score (15d)", f"{s['recency'][s['champ']]:.0%}")

    # --- RUNNER UP ROW ---
    st.subheader(f"🥈 Runner-Up: {s['runner']}")
    r1, r2, r3, r4 = st.columns(4)
    r_rets = np.array(s['res'][s['runner']])
    
    # Calculate annualized return using ACTUAL trading days
    total_return_r = np.prod(1+r_rets) - 1
    num_trading_days_r = len(r_rets)
    annualized_return_r = (1 + total_return_r) ** (252 / num_trading_days_r) - 1
    
    # Get optimal hold period
    optimal_hold_r = s['fcasts'][s['runner']]['optimal_hold']
    
    r1.metric(f"PREDICTION", s['fcasts'][s['runner']]['etf'], delta=f"Hold: {optimal_hold_r}d")
    r2.metric("Annualized Return (Net)", f"{annualized_return_r:.2%}", delta=f"OOS: {s['oos_years']}")
    r3.metric("Sharpe (Annualized)", f"{((np.mean(r_rets)-(s['rf']/252))/np.std(r_rets)*np.sqrt(252)):.2f}", delta=f"SOFR: {s['rf']:.2%}", delta_color="normal")
    r4.metric("Recency Score (15d)", f"{s['recency'][s['runner']]:.0%}")

    st.divider()
    # Charts and Tables
    fig = go.Figure()
    for name, r in s['res'].items(): fig.add_trace(go.Scatter(x=s['dates'], y=np.cumprod(1 + np.array(r)), name=name))
    fig.update_layout(title="Net Return Performance", template="plotly_dark", height=400)
    st.plotly_chart(fig, width='stretch')

    st.subheader(f"📅 Monthly Matrix ({s['champ']})")
    st.dataframe(s['monthly'].style.format("{:.2%}"), width='stretch')

    st.divider()
    st.header("🔍 Methodology")
    st.info("""
    **Recency Score (15d):** The 'Hit Rate' of a model over the last 15 trading sessions (% of positive days). 
    The engine blends this (30%) with long-term OOS performance (70%) to rank the models.
    
    **Data Split:** 80% Training / 20% Out-of-Sample Testing (no validation set)
    
    **Annualized Return:** Calculated using actual trading days: (1 + total_return)^(252/num_days) - 1
    
    **Optimal Hold Period:** Model tests 1-day, 3-day, and 5-day hold periods, calculating annualized returns after transaction costs for each. The period with the highest net annualized return is recommended.
    
    **Returns Calculation:** Dataset uses close-to-close daily returns: (Today's Close - Yesterday's Close) / Yesterday's Close
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
    
    # Data Diagnostics
    if 'diagnostics' in s:
        st.divider()
        st.subheader("📊 Data Diagnostics & Period Analysis")
        diag = s['diagnostics']
        
        # Period analysis
        if 'period_stats' in diag:
            pstats = diag['period_stats']
            st.write("**OOS Period Characteristics:**")
            
            # TLT/TBT correlation check
            if 'tlt_tbt_correlation' in pstats:
                corr_val = pstats['tlt_tbt_correlation']
                if corr_val < -0.8:
                    st.success(f"✅ TLT/TBT Correlation: {corr_val:.2f} (Strong negative - ideal for switching strategy)")
                elif corr_val < -0.5:
                    st.info(f"ℹ️ TLT/TBT Correlation: {corr_val:.2f} (Moderate negative)")
                else:
                    st.warning(f"⚠️ TLT/TBT Correlation: {corr_val:.2f} (Weak relationship - reduces switching advantage)")
            
            # Individual ETF stats
            st.write("**Individual ETF Performance (OOS Period):**")
            etf_df = pd.DataFrame({
                etf: {
                    'Total Return': f"{stats['total_return']:.2%}",
                    'Sharpe Ratio': f"{stats['sharpe']:.2f}",
                    'Daily Volatility': f"{stats['std_daily']:.3%}"
                }
                for etf, stats in pstats.items() if etf != 'tlt_tbt_correlation'
            }).T
            st.dataframe(etf_df)
        
        if diag['requested_start'] != diag['actual_start']:
            st.warning(f"⚠️ **Data Availability Notice:** Data requested from {diag['requested_start']}, but actual usable data starts from {diag['actual_start']} due to momentum calculation requirements.")
        
        col_a, col_b, col_c = st.columns(3)
        with col_a:
            st.metric("Data Source", diag['data_source'])
            st.metric("Requested Start", diag['requested_start'])
        with col_b:
            st.metric("Actual Data Start", diag['actual_start'])
            st.metric("Training/OOS Split", diag.get('train_test_split', '80/20'))
        with col_c:
            st.metric("Total Data Rows", f"{diag['processed_rows']:,}")
            st.metric("Features Used", f"{diag.get('num_features', 'N/A')}")
