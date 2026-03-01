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
from collections import Counter

# --- 1. SETTINGS & STATE ---
st.set_page_config(page_title="Alpha Tournament Pro", layout="wide")

if 'results' not in st.session_state: st.session_state.results = None

TARGET_ETFS = ['TLT', 'LQD', 'HYG', 'VCIT', 'VNQ', 'GLD', 'SLV']
DEFAULT_ENSEMBLE_YEARS = [2008, 2010, 2013, 2015, 2019, 2021]

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
            future_rets = returns_df[etf].iloc[i:i+hold_days]
            total_ret = np.prod(1 + future_rets) - 1
            net_ret = total_ret - tcost_dec
            period_returns.append(net_ret)
            i += hold_days
        
        if period_returns:
            avg_period_return = np.mean(period_returns)
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
    
    optimal_period = max(hold_returns.keys(), key=lambda k: hold_returns[k]['annualized'])
    
    return hold_returns, optimal_period

@st.cache_data(ttl=3600)
def load_data_from_hf(start_year, hf_token, dataset_repo):
    """Load data from HuggingFace dataset"""
    try:
        dataset = load_dataset(dataset_repo, split='train', token=hf_token)
        df = dataset.to_pandas()
        
        if 'Date' in df.columns:
            df['Date'] = pd.to_datetime(df['Date'])
            df = df.set_index('Date')
        elif 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])
            df = df.set_index('date')
        
        df = df[df.index >= f'{start_year}-01-01']
        df = df.sort_index()
        
        # Debug: Show available columns
        # st.write(f"Available columns: {list(df.columns)}")
        
        # Check for return columns - try different naming conventions
        ret_cols = []
        for etf in TARGET_ETFS:
            possible_names = [f'{etf}_Ret', f'{etf}_ret', f'{etf}_return', f'{etf}_Return', f'{etf}', f'ret_{etf}']
            for name in possible_names:
                if name in df.columns:
                    ret_cols.append(name)
                    break
        
        if len(ret_cols) != len(TARGET_ETFS):
            missing = [etf for etf in TARGET_ETFS if not any(f'{etf}' in col for col in ret_cols)]
            st.error(f"Missing return columns for: {missing}")
            return None, None
        
        # Build feature columns - be flexible with naming
        feature_cols = []
        for etf in TARGET_ETFS:
            # Try to find price-related columns for momentum calculation
            possible_price = [f'{etf}_Close', f'{etf}_close', f'{etf}_price', f'{etf}_Price', f'{etf}']
            price_col = None
            for name in possible_price:
                if name in df.columns:
                    price_col = name
                    break
            
            # Add technical indicators if they exist
            for suffix in ['_MA20', '_Vol', '_vol', '_volume', '_Volume', '_MA10', '_MA50', '_RSI']:
                col_name = f'{etf}{suffix}'
                if col_name in df.columns:
                    feature_cols.append(col_name)
        
        # Add macro columns
        macro_cols = ['UNRATE', 'CPI', 'VIX', 'TNX', 'DXY', 'AU_CU_Ratio', 'AU_CU_Trend',
                      'unrate', 'cpi', 'vix', 'tnx', 'dxy', 'au_cu_ratio']
        feature_cols.extend([col for col in macro_cols if col in df.columns])
        
        # If no feature columns found, use all numeric columns except returns
        if len(feature_cols) == 0:
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            feature_cols = [c for c in numeric_cols if c not in ret_cols]
            st.warning(f"Using {len(feature_cols)} numeric columns as features")
        
        returns_df = df[ret_cols].copy()
        returns_df.columns = TARGET_ETFS
        
        features_df = df[feature_cols].copy()
        
        # Clean data
        returns_df = returns_df.replace([np.inf, -np.inf], np.nan).ffill().bfill().dropna()
        features_df = features_df.replace([np.inf, -np.inf], np.nan).ffill().bfill().dropna()
        
        common_idx = returns_df.index.intersection(features_df.index)
        returns_df = returns_df.loc[common_idx]
        features_df = features_df.loc[common_idx]
        
        if len(returns_df) < 100:
            st.error(f"Insufficient data: {len(returns_df)} rows")
            return None, None
        
        return (features_df, returns_df), "HuggingFace Dataset"
        
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
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
        super().reset(seed=seed)
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
        truncated = False
        return self.features[self.current_step], reward, done, truncated, {}

# --- 4. ENGINE ---
@st.cache_data(ttl=86400, show_spinner=False)
def run_tournament_engine(_features_df, _returns_df, rf_rate, tcost_bps, start_year):
    """Run tournament for a single training period - cached separately"""
    
    features_df = _features_df.copy()
    returns_df = _returns_df.copy()
    
    # Clean data
    features_df = features_df.replace([np.inf, -np.inf], np.nan).ffill().bfill().dropna()
    returns_df = returns_df.replace([np.inf, -np.inf], np.nan).ffill().bfill().dropna()
    
    common_idx = features_df.index.intersection(returns_df.index)
    features_df = features_df.loc[common_idx]
    returns_df = returns_df.loc[common_idx]
    
    if len(features_df) < 100:
        return None
    
    # Calculate momentum features from returns (cumulative product to get price-like series)
    lookback_periods = [30, 45, 60]
    momentum_dict = {}
    
    # Reconstruct price-like series from returns for momentum calculation
    price_like = (1 + returns_df).cumprod()
    
    for period in lookback_periods:
        momentum = price_like.pct_change(period).replace([np.inf, -np.inf], np.nan).ffill().bfill()
        momentum_dict[f'momentum_{period}d'] = momentum
    
    # Find best lookback
    best_lookback = 45
    best_score = -np.inf
    
    for period in lookback_periods:
        momentum_data = momentum_dict[f'momentum_{period}d']
        # Only use momentum columns that align with features
        aligned_features = features_df.loc[momentum_data.index]
        
        if len(aligned_features) < 100:
            continue
        
        # Calculate correlation between momentum and future returns
        future_rets = returns_df.shift(-1).loc[momentum_data.index]  # Next day returns
        
        for i, etf in enumerate(TARGET_ETFS):
            if etf in momentum_data.columns and etf in future_rets.columns:
                corr = momentum_data[etf].corr(future_rets[etf])
                if not np.isnan(corr):
                    score = abs(corr)
                    if score > best_score:
                        best_score = score
                        best_lookback = period
    
    # Build final dataset
    momentum_data = momentum_dict[f'momentum_{best_lookback}d']
    
    # Align all data
    common_idx = features_df.index.intersection(momentum_data.index).intersection(returns_df.index)
    features_aligned = features_df.loc[common_idx]
    momentum_aligned = momentum_data.loc[common_idx]
    returns_aligned = returns_df.loc[common_idx]
    
    # Combine features
    features_with_momentum = pd.concat([features_aligned, momentum_aligned.add_suffix(f'_mom{best_lookback}')], axis=1)
    features_with_momentum = features_with_momentum.replace([np.inf, -np.inf], np.nan).ffill().bfill().dropna()
    
    # Final alignment
    final_idx = features_with_momentum.index.intersection(returns_aligned.index)
    X = features_with_momentum.loc[final_idx].values
    y = returns_aligned.loc[final_idx].values
    
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    y = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)
    
    if len(X) < 100:
        return None
    
    scaler = StandardScaler()
    X_sc = scaler.fit_transform(X).astype(np.float32)
    split = int(len(X_sc) * 0.8)
    
    if split < 50 or (len(X_sc) - split) < 20:
        return None
    
    period_stats = analyze_period_characteristics(returns_aligned.loc[final_idx], split)
    
    # Create sequences
    seq_len = min(best_lookback, 30)  # Cap sequence length
    if len(X_sc) <= seq_len:
        return None
        
    X_seq = []
    y_seq = []
    for i in range(seq_len, len(X_sc)):
        X_seq.append(X_sc[i-seq_len:i])
        y_seq.append(y[i])
    X_seq = np.array(X_seq)
    y_seq = np.array(y_seq)
    
    split_seq = int(len(X_seq) * 0.8)
    X_train, X_test = X_seq[:split_seq], X_seq[split_seq:]
    y_train, y_test = y_seq[:split_seq], y_seq[split_seq:]
    
    X_train_flat = X_sc[seq_len:seq_len+split_seq]
    X_test_flat = X_sc[seq_len+split_seq:]
    y_train_rl = y[seq_len:seq_len+split_seq]
    y_test_rl = y[seq_len+split_seq:]
    
    # Train models (reduced iterations for speed)
    try:
        env = DummyVecEnv([lambda: TradingEnv(X_train_flat, y_train_rl, TARGET_ETFS, tcost_bps)])
        ppo = PPO("MlpPolicy", env, verbose=0, n_steps=2048, batch_size=64).learn(3000)
        a2c = A2C("MlpPolicy", env, verbose=0).learn(3000)
    except Exception as e:
        st.error(f"RL training failed: {e}")
        return None
    
    dl_models = {}
    for name, m_class in [("CNN-LSTM", CNN_LSTM_Model), ("Transformer", TransformerModel)]:
        try:
            model = m_class(X.shape[1], len(TARGET_ETFS), seq_len)
            opt = torch.optim.Adam(model.parameters(), lr=0.005)
            X_t, y_t = torch.tensor(X_train).float(), torch.tensor(y_train).float()
            for _ in range(30):
                opt.zero_grad()
                nn.MSELoss()(model(X_t), y_t).backward()
                opt.step()
            dl_models[name] = model
        except Exception as e:
            st.error(f"DL model {name} training failed: {e}")
            continue

    results = {"PPO": [], "A2C": [], "CNN-LSTM": [], "Transformer": []}
    predictions = {"PPO": [], "A2C": [], "CNN-LSTM": [], "Transformer": []}
    test_dates = final_idx[split_seq + seq_len:]
    tcost_dec = tcost_bps / 10000

    # Generate predictions
    for name in ["PPO", "A2C"]:
        last_pick = None
        for i in range(len(X_test_flat)):
            try:
                if name == "PPO": 
                    act, _ = ppo.predict(X_test_flat[i], deterministic=True)
                else:
                    act, _ = a2c.predict(X_test_flat[i], deterministic=True)
                
                act = int(act) if isinstance(act, (np.ndarray, list)) else act
                
                predictions[name].append(TARGET_ETFS[act])
                day_ret = y_test_rl[i, act]
                if last_pick is not None and act != last_pick: 
                    day_ret -= tcost_dec
                results[name].append(day_ret)
                last_pick = act
            except Exception as e:
                results[name].append(0)
                predictions[name].append(TARGET_ETFS[0])
    
    for name in ["CNN-LSTM", "Transformer"]:
        if name not in dl_models:
            continue
        last_pick = None
        for i in range(len(X_test)):
            try:
                with torch.no_grad():
                    out = dl_models[name](torch.tensor(X_test[i]).unsqueeze(0).float())
                    act = torch.argmax(out).item()
                
                predictions[name].append(TARGET_ETFS[act])
                day_ret = y_test[i, act]
                if last_pick is not None and act != last_pick:
                    day_ret -= tcost_dec
                results[name].append(day_ret)
                last_pick = act
            except Exception as e:
                results[name].append(0)
                predictions[name].append(TARGET_ETFS[0])

    # Check if we have valid results
    valid_models = [k for k, v in results.items() if len(v) > 0]
    if len(valid_models) < 2:
        return None

    oos_start_year = test_dates[0].year
    oos_end_year = test_dates[-1].year
    oos_years = f"{oos_start_year}-{oos_end_year}" if oos_start_year != oos_end_year else str(oos_start_year)

    recency_window = min(15, len(results[valid_models[0]]) - 1)
    recency_scores = {}
    for n in valid_models:
        if len(results[n]) >= recency_window:
            recency_scores[n] = np.sum(np.array(results[n][-recency_window:]) > 0) / recency_window
        else:
            recency_scores[n] = 0.5
    
    perf = {k: ((np.prod(1 + np.array(results[k])) - 1) * 0.7) + (recency_scores.get(k, 0) * 0.3) 
            for k in valid_models}
    
    sorted_models = sorted(perf.items(), key=lambda x: x[1], reverse=True)
    champ, runner_up = sorted_models[0][0], sorted_models[1][0]
    
    forecasts = {}
    latest_feat_flat = X_sc[-1:]
    latest_feat_seq = X_seq[-1:]
    
    oos_returns_df = returns_aligned.loc[final_idx[split_seq + seq_len:]]
    
    for m in [champ, runner_up]:
        try:
            if m == "PPO": 
                act, _ = ppo.predict(latest_feat_flat[0], deterministic=True)
                act = int(act) if isinstance(act, (np.ndarray, list)) else act
            elif m == "A2C": 
                act, _ = a2c.predict(latest_feat_flat[0], deterministic=True)
                act = int(act) if isinstance(act, (np.ndarray, list)) else act
            else:
                with torch.no_grad():
                    f_out = dl_models[m](torch.tensor(latest_feat_seq).float())
                    act = torch.argmax(f_out).item()
            
            etf_pred = TARGET_ETFS[act]
            hold_stats, optimal_hold = calculate_hold_period_returns(predictions[m], oos_returns_df, tcost_bps, [1, 3, 5])
            
            forecasts[m] = {
                'etf': etf_pred,
                'hold_periods': hold_stats,
                'optimal_hold': optimal_hold
            }
        except Exception as e:
            forecasts[m] = {'etf': TARGET_ETFS[0], 'hold_periods': {}, 'optimal_hold': 1}

    champ_series = pd.Series(results[champ], index=test_dates)
    monthly_rets = champ_series.groupby([champ_series.index.year, champ_series.index.month]).apply(lambda x: np.prod(1+x)-1)
    m_table = monthly_rets.unstack().fillna(0)
    m_table.columns = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'][:len(m_table.columns)]
    m_table['Yearly Total'] = m_table.apply(lambda row: np.prod(1 + row) - 1, axis=1)

    c_rets = np.array(results[champ])
    total_return_c = np.prod(1+c_rets) - 1
    num_trading_days_c = len(c_rets)
    annualized_return_c = (1 + total_return_c) ** (252 / num_trading_days_c) - 1

    return {
        'champion': champ,
        'runner_up': runner_up,
        'champion_prediction': forecasts[champ]['etf'],
        'runner_up_prediction': forecasts[runner_up]['etf'],
        'champion_hold': forecasts[champ]['optimal_hold'],
        'runner_up_hold': forecasts[runner_up]['optimal_hold'],
        'annualized_return': annualized_return_c,
        'sharpe': (np.mean(c_rets)-(rf_rate/252))/np.std(c_rets)*np.sqrt(252) if np.std(c_rets) > 0 else 0,
        'recency': recency_scores[champ],
        'oos_years': oos_years,
        'monthly_table': m_table,
        'period_stats': period_stats,
        'results': results,
        'test_dates': test_dates
    }

# --- 5. UI ---
st.title("Alpha Tournament Pro: Multi-model ETF Forecast")

with st.sidebar:
    st.header("Tournament Configuration")
    t_cost = st.slider("Transaction Cost (bps)", min_value=0, max_value=100, value=10, step=5)
    
    st.divider()
    st.subheader("Training Periods")
    
    # Option A or B selection
    period_option = st.radio(
        "Select Training Period Option:",
        options=["Option A (Default)", "Option B (Custom)"],
        help="Option A uses predefined optimal periods. Option B lets you choose your own 6 periods."
    )
    
    if period_option == "Option A (Default)":
        ensemble_years = DEFAULT_ENSEMBLE_YEARS
        st.info(f"**Using Years:** {', '.join(map(str, ensemble_years))}")
    else:
        st.caption("**Select 6 Years:**")
        current_year = datetime.now().year
        available_years = list(range(2008, current_year))
        
        # Create 6 dropdowns for year selection
        selected_years = []
        for i in range(6):
            year = st.selectbox(
                f"Period {i+1}",
                options=available_years,
                index=min(i * 2, len(available_years) - 1),
                key=f"year_{i}"
            )
            selected_years.append(year)
        
        ensemble_years = sorted(list(set(selected_years)))  # Remove duplicates and sort
        
        if len(ensemble_years) < len(set(selected_years)):
            st.warning(f"⚠️ Duplicate years selected. Using {len(ensemble_years)} unique periods.")
    
    run_btn = st.button("🚀 Execute Ensemble Tournament")

if run_btn:
    with st.status(f"Running Ensemble Tournament...") as status:
        try:
            rf = get_sofr_rate(FRED_API_KEY)
            
            ensemble_results = {}
            progress_container = st.empty()
            
            for idx, start_year in enumerate(ensemble_years):
                progress_container.info(f"Training period {start_year}... ({idx+1}/{len(ensemble_years)})")
                
                data_tuple = load_data_from_hf(start_year, HF_TOKEN, HF_DATASET_REPO)
                
                if data_tuple[0] is None:
                    st.warning(f"Skipping {start_year}: No data available")
                    continue
                
                (features_df, returns_df), data_src = data_tuple
                
                result = run_tournament_engine(features_df, returns_df, rf, t_cost, start_year)
                
                if result is not None:
                    ensemble_results[start_year] = result
                    progress_container.success(f"✓ Period {start_year} complete. Champion: {result['champion']}")
                else:
                    progress_container.warning(f"✗ Period {start_year} failed to produce results")
            
            progress_container.empty()
            
            if len(ensemble_results) == 0:
                st.error("Failed to run any training periods. Check data availability and column names.")
                st.stop()
            
            # Ensemble voting logic
            champion_votes = Counter([r['champion_prediction'] for r in ensemble_results.values()])
            runner_up_votes = Counter([r['runner_up_prediction'] for r in ensemble_results.values()])
            
            agreement_count = sum(1 for r in ensemble_results.values() 
                                if r['champion_prediction'] == r['runner_up_prediction'])
            
            consensus_etf = champion_votes.most_common(1)[0][0]
            consensus_votes = champion_votes[consensus_etf]
            
            # Calculate consensus hold period (most common among periods that voted for consensus ETF)
            consensus_holds = []
            for r in ensemble_results.values():
                if r['champion_prediction'] == consensus_etf:
                    consensus_holds.append(r['champion_hold'])
            
            if consensus_holds:
                consensus_hold = Counter(consensus_holds).most_common(1)[0][0]
            else:
                consensus_hold = 1  # Default to 1 day
            
            if consensus_votes >= 4:
                confidence = "HIGH"
            elif consensus_votes == 3:
                confidence = "MEDIUM"
            else:
                confidence = "LOW"
            
            best_period = max(ensemble_results.keys(), 
                            key=lambda k: ensemble_results[k]['annualized_return'])
            best_result = ensemble_results[best_period]
            
            next_trade_day = get_next_trading_day()
            
            st.session_state.results = {
                "ensemble_results": ensemble_results,
                "consensus_etf": consensus_etf,
                "consensus_hold": consensus_hold,
                "consensus_votes": consensus_votes,
                "total_periods": len(ensemble_results),
                "champion_votes": dict(champion_votes),
                "runner_up_votes": dict(runner_up_votes),
                "agreement_count": agreement_count,
                "confidence": confidence,
                "best_result": best_result,
                "best_period": best_period,
                "rf": rf,
                "next_day": next_trade_day,
                "ensemble_years": ensemble_years
            }
            
            status.update(label=f"Ensemble Tournament Complete!", state="complete")
        except Exception as e:
            status.update(label=f"Error: {str(e)}", state="error")
            st.error(f"Failed to run tournament: {str(e)}")
            import traceback
            st.code(traceback.format_exc())
            st.stop()
    st.rerun()

if st.session_state.results:
    s = st.session_state.results
    
    # --- ENSEMBLE CONSENSUS ---
    st.subheader("🎯 Ensemble Consensus Prediction")
    
    cons1, cons2, cons3 = st.columns(3)
    
    # Display ETF with hold period
    etf_with_hold = f"{s['consensus_etf']} (Hold: {s['consensus_hold']}d)"
    
    if s['confidence'] == "HIGH":
        cons1.success(f"**{etf_with_hold}**")
        cons1.caption(f"Consensus: {s['consensus_votes']}/{s['total_periods']} periods")
    elif s['confidence'] == "MEDIUM":
        cons1.info(f"**{etf_with_hold}**")
        cons1.caption(f"Consensus: {s['consensus_votes']}/{s['total_periods']} periods")
    else:
        cons1.warning(f"**{etf_with_hold}**")
        cons1.caption(f"Weak consensus: {s['consensus_votes']}/{s['total_periods']} periods")
    
    cons2.metric("Confidence Level", s['confidence'], 
                 delta=f"Valid: {s['next_day']}")
    
    cons3.metric("Model Agreement", 
                f"{s['agreement_count']}/{s['total_periods']} periods",
                delta="Champion = Runner-up")
    
    # Show which periods were used
    st.caption(f"**Training Periods Used:** {', '.join(map(str, s['ensemble_years']))}")
    
    # Voting breakdown
    st.divider()
    st.subheader("📊 Voting Breakdown by Training Period")
    
    vote_data = []
    for year, result in s['ensemble_results'].items():
        vote_data.append({
            'Training Period': f"{year}-2026",
            'Champion': result['champion'],
            'Champion Predicts': result['champion_prediction'],
            'Hold': f"{result['champion_hold']}d",
            'Runner-Up': result['runner_up'],
            'Runner-Up Predicts': result['runner_up_prediction'],
            'Agreement': '✅' if result['champion_prediction'] == result['runner_up_prediction'] else '❌'
        })
    
    vote_df = pd.DataFrame(vote_data)
    st.dataframe(vote_df, use_container_width=True)
    
    # Best performing period details
    st.divider()
    st.subheader(f"📈 Best Performing Period: {s['best_period']}-2026")
    
    br = s['best_result']
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Champion Model", br['champion'])
    col2.metric("Annualized Return", f"{br['annualized_return']:.2%}", delta=f"OOS: {br['oos_years']}")
    col3.metric("Sharpe Ratio", f"{br['sharpe']:.2f}", delta=f"SOFR: {s['rf']:.2%}")
    col4.metric("Recency Score (15d)", f"{br['recency']:.0%}")
    
    # Performance chart - CHANGED HERE
    st.divider()
    fig = go.Figure()
    for name, r in br['results'].items():
        if len(r) > 0:
            fig.add_trace(go.Scatter(x=br['test_dates'][:len(r)], y=np.cumprod(1 + np.array(r)), name=name))
    fig.update_layout(title=f"Net Return Performance - {br['oos_years']} OOS Period", 
                     template="plotly_dark", height=400)
    st.plotly_chart(fig, use_container_width=True)
    
    # Monthly matrix
    st.subheader(f"📅 Monthly Matrix - {s['best_period']} Period ({br['champion']})")
    st.dataframe(br['monthly_table'].style.format("{:.2%}"), use_container_width=True)
    
    # Methodology
    st.divider()
    st.header("🔍 Methodology")
    st.info("""
    **Ensemble Voting System:**
    
    The Alpha Tournament uses an ensemble approach to reduce overfitting and increase prediction robustness:
    
    1. **Multiple Training Periods:** Models are trained on 6 different historical periods, each using an 80/20 train/test split up to 2026.
    
    2. **Independent Tournaments:** Each training period runs a complete tournament with 4 model architectures (PPO, A2C, CNN-LSTM, Transformer) competing against each other.
    
    3. **Consensus Voting:** The final prediction is determined by majority vote across all training periods. Each period's champion model casts one vote.
    
    4. **Confidence Scoring:**
       - **HIGH:** 4+ periods agree (≥67% consensus)
       - **MEDIUM:** 3 periods agree (50% consensus)
       - **LOW:** Split vote (no clear majority)
    
    5. **Internal Agreement Check:** Within each period, we verify if the champion and runner-up models agree, providing an additional confidence signal.
    
    **Trading Recommendation:** Only trade when confidence is HIGH or when internal agreement is strong across multiple periods. This approach significantly reduces the risk of following overfitted predictions from any single training period.
    
    **Hold Period Optimization:** Each model tests 1-day, 3-day, and 5-day hold periods, selecting the one with the highest annualized return after transaction costs. The consensus hold period is the most common recommendation among periods that voted for the consensus ETF.
    """)
    
    st.subheader("🤖 Model Descriptions")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**PPO (Proximal Policy Optimization)**")
        st.caption("Reinforcement learning algorithm trained for 3,000 timesteps.")
        
        st.markdown("**CNN-LSTM**")
        st.caption("Hybrid deep learning architecture trained for 30 epochs.")
    
    with col2:
        st.markdown("**A2C (Advantage Actor-Critic)**")
        st.caption("Reinforcement learning algorithm trained for 3,000 timesteps.")
        
        st.markdown("**Transformer**")
        st.caption("Attention-based neural network trained for 30 epochs.")
    
    # Period analysis
    if 'period_stats' in br:
        st.divider()
        st.subheader(f"📊 OOS Period Analysis - {s['best_period']} Training")
                     
        st.write("**Individual ETF Performance (OOS Period):**")
        etf_df = pd.DataFrame({
            etf: {
                'Total Return': f"{stats['total_return']:.2%}",
                'Sharpe Ratio': f"{stats['sharpe']:.2f}",
                'Daily Volatility': f"{stats['std_daily']:.3%}"
            }
            for etf, stats in br['period_stats'].items()
        }).T
        st.dataframe(etf_df, use_container_width=True)
