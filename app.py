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
from huggingface_hub import list_repo_files

# --- 1. SETTINGS & STATE ---
st.set_page_config(page_title="Alpha Tournament Pro", layout="wide")

if 'results_fi' not in st.session_state:
    st.session_state.results_fi = None
if 'results_eq' not in st.session_state:
    st.session_state.results_eq = None

# ── Universe definitions ──────────────────────────────────────────────────────
FI_ETFS  = ['TLT', 'LQD', 'HYG', 'VCIT', 'VNQ', 'GLD', 'SLV']
EQ_ETFS  = ['XME', 'XLF', 'XLV', 'QQQ', 'XLP', 'XLI', 'XLK', 'XLU', 'XLY', 'GDX', 'XLE']

DEFAULT_ENSEMBLE_YEARS = [
    2008, 2009, 2010, 2011, 2012, 2013, 2014,
    2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024, 2025
]

# Secrets
FRED_API_KEY   = os.environ.get("FRED_API_KEY")
HF_TOKEN       = os.environ.get("HF_KEY")
HF_DATASET_REPO = "P2SAMAPA/my-etf-data"

# --- 2. MODEL ARCHITECTURES ---
class CNN_LSTM_Model(nn.Module):
    def __init__(self, input_dim, output_dim, seq_len):
        super().__init__()
        self.conv = nn.Conv1d(input_dim, 64, kernel_size=3, padding=1)
        self.lstm = nn.LSTM(64, 128, batch_first=True)
        self.fc   = nn.Linear(128, output_dim)

    def forward(self, x):
        x = x.transpose(1, 2)
        x = torch.relu(self.conv(x)).transpose(1, 2)
        _, (hn, _) = self.lstm(x)
        return self.fc(hn[-1])


class TransformerModel(nn.Module):
    def __init__(self, input_dim, output_dim, seq_len):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, 64)
        encoder_layer   = nn.TransformerEncoderLayer(d_model=64, nhead=4, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.fc = nn.Linear(64, output_dim)

    def forward(self, x):
        x = self.input_proj(x)
        x = self.transformer(x)
        return self.fc(x[:, -1, :])


# --- 3. UTILITIES ---
def get_sofr_rate(api_key):
    if not api_key:
        return 0.053
    url = (f"https://api.stlouisfed.org/fred/series/observations"
           f"?series_id=SOFR&api_key={api_key}&file_type=json")
    try:
        r = requests.get(url, timeout=10).json()
        return float(r['observations'][-1]['value']) / 100
    except:
        return 0.053


def get_next_trading_day():
    try:
        nyse  = mcal.get_calendar('NYSE')
        today = pd.Timestamp.now(tz='America/New_York').normalize()
        schedule  = nyse.schedule(start_date=today, end_date=today + timedelta(days=10))
        valid_days = mcal.date_range(schedule, frequency='1D')
        for day in valid_days:
            if day.normalize() > today:
                return day.strftime('%Y-%m-%d')
    except:
        pass
    return (pd.Timestamp.now() + timedelta(days=1)).strftime('%Y-%m-%d')


def analyze_period_characteristics(returns_df, test_start_idx):
    test_returns = returns_df.iloc[test_start_idx:]
    stats = {}
    for etf in returns_df.columns:
        r = test_returns[etf]
        stats[etf] = {
            'mean_daily':   r.mean(),
            'std_daily':    r.std(),
            'sharpe':       r.mean() / r.std() * np.sqrt(252) if r.std() > 0 else 0,
            'max_daily':    r.max(),
            'min_daily':    r.min(),
            'total_return': (1 + r).prod() - 1,
        }
    return stats


def calculate_hold_period_returns(predictions, returns_df, tcost_bps, hold_periods=[1, 3, 5]):
    tcost_dec   = tcost_bps / 10000
    hold_returns = {}
    for hold_days in hold_periods:
        period_returns = []
        i = 0
        while i < len(predictions) - hold_days:
            etf         = predictions[i]
            future_rets = returns_df[etf].iloc[i:i + hold_days]
            total_ret   = np.prod(1 + future_rets) - 1
            period_returns.append(total_ret - tcost_dec)
            i += hold_days
        if period_returns:
            avg = np.mean(period_returns)
            hold_returns[hold_days] = {
                'avg_return':         avg,
                'annualized':         (1 + avg) ** (252 / hold_days) - 1,
                'num_trades_per_year': 252 / hold_days,
            }
        else:
            hold_returns[hold_days] = {'avg_return': 0, 'annualized': 0, 'num_trades_per_year': 0}
    optimal = max(hold_returns, key=lambda k: hold_returns[k]['annualized'])
    return hold_returns, optimal


# ── Data loader (universe-aware) ──────────────────────────────────────────────
@st.cache_data(ttl=3600)
def load_data_from_hf(start_year, hf_token, dataset_repo, universe_etfs):
    """Load data for a given ETF universe from HuggingFace."""
    try:
        dataset = load_dataset(dataset_repo, split='train', token=hf_token)
        df = dataset.to_pandas()
    except Exception as e:
        st.warning(f"Standard dataset load failed: {e}. Trying direct CSV download...")
        try:
            from huggingface_hub import hf_hub_download
            csv_path = hf_hub_download(
                repo_id=dataset_repo,
                filename="etf_data.csv",
                repo_type="dataset",
                token=hf_token,
            )
            df = pd.read_csv(csv_path)
        except Exception as e2:
            st.error(f"Direct CSV download also failed: {e2}")
            import traceback; st.code(traceback.format_exc())
            return None, None

    # Date handling
    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date']); df = df.set_index('Date')
    elif 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date']); df = df.set_index('date')
    else:
        st.error("No date column found in dataset.")
        return None, None

    df = df[df.index >= f'{start_year}-01-01'].sort_index()

    # Return columns
    ret_cols = []
    for etf in universe_etfs:
        for name in [f'{etf}_Ret', f'{etf}_ret', f'{etf}_return', f'{etf}_Return', f'{etf}', f'ret_{etf}']:
            if name in df.columns:
                ret_cols.append(name)
                break

    if len(ret_cols) != len(universe_etfs):
        missing = [etf for etf in universe_etfs
                   if not any(etf in col for col in ret_cols)]
        st.error(f"Missing return columns for: {missing}")
        return None, None

    # Feature columns
    feature_cols = []
    for etf in universe_etfs:
        for suffix in ['_MA20', '_Vol', '_vol', '_volume', '_Volume', '_MA10', '_MA50', '_RSI']:
            col_name = f'{etf}{suffix}'
            if col_name in df.columns:
                feature_cols.append(col_name)

    macro_cols = ['UNRATE', 'CPI', 'VIX', 'TNX', 'DXY', 'AU_CU_Ratio', 'AU_CU_Trend',
                  'unrate', 'cpi', 'vix', 'tnx', 'dxy', 'au_cu_ratio']
    feature_cols.extend([c for c in macro_cols if c in df.columns])

    if not feature_cols:
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        feature_cols = [c for c in numeric_cols if c not in ret_cols]
        st.warning(f"Using {len(feature_cols)} numeric columns as features")

    returns_df  = df[ret_cols].copy()
    returns_df.columns = universe_etfs
    features_df = df[feature_cols].copy()

    returns_df  = returns_df.replace([np.inf, -np.inf], np.nan).ffill().bfill().dropna()
    features_df = features_df.replace([np.inf, -np.inf], np.nan).ffill().bfill().dropna()

    common_idx  = returns_df.index.intersection(features_df.index)
    returns_df  = returns_df.loc[common_idx]
    features_df = features_df.loc[common_idx]

    if len(returns_df) < 100:
        st.error(f"Insufficient data: {len(returns_df)} rows")
        return None, None

    return (features_df, returns_df), "HuggingFace Dataset"


# ── Trading environment (universe-aware) ──────────────────────────────────────
class TradingEnv(gym.Env):
    def __init__(self, features, returns, etfs, tcost_bps):
        super().__init__()
        self.features, self.returns, self.etfs = features, returns, etfs
        self.tcost = tcost_bps / 10000
        self.action_space      = gym.spaces.Discrete(len(etfs))
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(features.shape[1],), dtype=np.float32)
        self.current_step = 0
        self.last_action  = None

    def reset(self, seed=None):
        super().reset(seed=seed)
        self.current_step = 0
        self.last_action  = None
        return self.features[0], {}

    def step(self, action):
        raw_reward = float(self.returns[self.current_step, action])
        penalty    = self.tcost if (self.last_action is not None and action != self.last_action) else 0
        reward     = raw_reward - penalty
        self.last_action   = action
        self.current_step += 1
        done = self.current_step >= len(self.features) - 1
        return self.features[self.current_step], reward, done, False, {}


# --- 4. SHARED TOURNAMENT ENGINE ---
@st.cache_data(ttl=86400, show_spinner=False)
def run_tournament_engine(_features_df, _returns_df, rf_rate, tcost_bps, start_year, universe_etfs):
    features_df = _features_df.copy()
    returns_df  = _returns_df.copy()

    features_df = features_df.replace([np.inf, -np.inf], np.nan).ffill().bfill().dropna()
    returns_df  = returns_df.replace([np.inf, -np.inf], np.nan).ffill().bfill().dropna()

    common_idx  = features_df.index.intersection(returns_df.index)
    features_df = features_df.loc[common_idx]
    returns_df  = returns_df.loc[common_idx]

    if len(features_df) < 100:
        return None

    lookback_periods = [30, 45, 60]
    price_like       = (1 + returns_df).cumprod()
    momentum_dict    = {
        f'momentum_{p}d': price_like.pct_change(p).replace([np.inf, -np.inf], np.nan).ffill().bfill()
        for p in lookback_periods
    }

    best_lookback = 45
    best_score    = -np.inf
    for period in lookback_periods:
        momentum_data = momentum_dict[f'momentum_{period}d']
        future_rets   = returns_df.shift(-1).loc[momentum_data.index]
        for etf in universe_etfs:
            if etf in momentum_data.columns and etf in future_rets.columns:
                corr = momentum_data[etf].corr(future_rets[etf])
                if not np.isnan(corr) and abs(corr) > best_score:
                    best_score    = abs(corr)
                    best_lookback = period

    momentum_data = momentum_dict[f'momentum_{best_lookback}d']
    common_idx    = features_df.index.intersection(momentum_data.index).intersection(returns_df.index)

    features_aligned  = features_df.loc[common_idx]
    momentum_aligned  = momentum_data.loc[common_idx]
    returns_aligned   = returns_df.loc[common_idx]

    features_with_momentum = pd.concat(
        [features_aligned, momentum_aligned.add_suffix(f'_mom{best_lookback}')], axis=1)
    features_with_momentum = features_with_momentum.replace([np.inf, -np.inf], np.nan).ffill().bfill().dropna()

    final_idx = features_with_momentum.index.intersection(returns_aligned.index)
    X = features_with_momentum.loc[final_idx].values
    y = returns_aligned.loc[final_idx].values
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    y = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)

    if len(X) < 100:
        return None

    scaler = StandardScaler()
    X_sc   = scaler.fit_transform(X).astype(np.float32)
    split  = int(len(X_sc) * 0.8)
    if split < 50 or (len(X_sc) - split) < 20:
        return None

    period_stats = analyze_period_characteristics(returns_aligned.loc[final_idx], split)

    seq_len = min(best_lookback, 30)
    if len(X_sc) <= seq_len:
        return None

    X_seq, y_seq = [], []
    for i in range(seq_len, len(X_sc)):
        X_seq.append(X_sc[i - seq_len:i])
        y_seq.append(y[i])
    X_seq = np.array(X_seq)
    y_seq = np.array(y_seq)

    split_seq       = int(len(X_seq) * 0.8)
    X_train, X_test = X_seq[:split_seq], X_seq[split_seq:]
    y_train, y_test = y_seq[:split_seq], y_seq[split_seq:]

    X_train_flat = X_sc[seq_len:seq_len + split_seq]
    X_test_flat  = X_sc[seq_len + split_seq:]
    y_train_rl   = y[seq_len:seq_len + split_seq]
    y_test_rl    = y[seq_len + split_seq:]

    # ── RL models ──
    try:
        env = DummyVecEnv([lambda: TradingEnv(X_train_flat, y_train_rl, universe_etfs, tcost_bps)])
        ppo = PPO("MlpPolicy", env, verbose=0, n_steps=2048, batch_size=64).learn(3000)
        a2c = A2C("MlpPolicy", env, verbose=0).learn(3000)
    except Exception as e:
        st.error(f"RL training failed: {e}")
        return None

    # ── DL models ──
    dl_models = {}
    for name, m_class in [("CNN-LSTM", CNN_LSTM_Model), ("Transformer", TransformerModel)]:
        try:
            model = m_class(X.shape[1], len(universe_etfs), seq_len)
            opt   = torch.optim.Adam(model.parameters(), lr=0.005)
            X_t   = torch.tensor(X_train).float()
            y_t   = torch.tensor(y_train).float()
            for _ in range(30):
                opt.zero_grad()
                nn.MSELoss()(model(X_t), y_t).backward()
                opt.step()
            dl_models[name] = model
        except Exception as e:
            st.error(f"DL model {name} training failed: {e}")

    results     = {"PPO": [], "A2C": [], "CNN-LSTM": [], "Transformer": []}
    predictions = {"PPO": [], "A2C": [], "CNN-LSTM": [], "Transformer": []}
    test_dates  = final_idx[split_seq + seq_len:]
    tcost_dec   = tcost_bps / 10000

    for name in ["PPO", "A2C"]:
        last_pick = None
        for i in range(len(X_test_flat)):
            try:
                if name == "PPO":
                    act, _ = ppo.predict(X_test_flat[i], deterministic=True)
                else:
                    act, _ = a2c.predict(X_test_flat[i], deterministic=True)
                act = int(act) if isinstance(act, (np.ndarray, list)) else act
                predictions[name].append(universe_etfs[act])
                day_ret = y_test_rl[i, act]
                if last_pick is not None and act != last_pick:
                    day_ret -= tcost_dec
                results[name].append(day_ret)
                last_pick = act
            except:
                results[name].append(0)
                predictions[name].append(universe_etfs[0])

    for name in ["CNN-LSTM", "Transformer"]:
        if name not in dl_models:
            continue
        last_pick = None
        for i in range(len(X_test)):
            try:
                with torch.no_grad():
                    out = dl_models[name](torch.tensor(X_test[i]).unsqueeze(0).float())
                act = torch.argmax(out).item()
                predictions[name].append(universe_etfs[act])
                day_ret = y_test[i, act]
                if last_pick is not None and act != last_pick:
                    day_ret -= tcost_dec
                results[name].append(day_ret)
                last_pick = act
            except:
                results[name].append(0)
                predictions[name].append(universe_etfs[0])

    valid_models = [k for k, v in results.items() if len(v) > 0]
    if len(valid_models) < 2:
        return None

    oos_start = test_dates[0].year
    oos_end   = test_dates[-1].year
    oos_years = f"{oos_start}-{oos_end}" if oos_start != oos_end else str(oos_start)

    recency_window = min(15, len(results[valid_models[0]]) - 1)
    recency_scores = {
        n: np.sum(np.array(results[n][-recency_window:]) > 0) / recency_window
        if len(results[n]) >= recency_window else 0.5
        for n in valid_models
    }

    perf = {
        k: ((np.prod(1 + np.array(results[k])) - 1) * 0.7) + (recency_scores.get(k, 0) * 0.3)
        for k in valid_models
    }
    sorted_models = sorted(perf.items(), key=lambda x: x[1], reverse=True)
    champ, runner_up = sorted_models[0][0], sorted_models[1][0]

    forecasts = {}
    latest_feat_flat = X_sc[-1:]
    latest_feat_seq  = X_seq[-1:]
    oos_returns_df   = returns_aligned.loc[final_idx[split_seq + seq_len:]]

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
            etf_pred = universe_etfs[act]
            hold_stats, optimal_hold = calculate_hold_period_returns(
                predictions[m], oos_returns_df, tcost_bps, [1, 3, 5])
            forecasts[m] = {'etf': etf_pred, 'hold_periods': hold_stats, 'optimal_hold': optimal_hold}
        except:
            forecasts[m] = {'etf': universe_etfs[0], 'hold_periods': {}, 'optimal_hold': 1}

    champ_series  = pd.Series(results[champ], index=test_dates)
    monthly_rets  = champ_series.groupby([champ_series.index.year,
                                          champ_series.index.month]).apply(lambda x: np.prod(1 + x) - 1)
    m_table       = monthly_rets.unstack().fillna(0)
    m_table.columns = ['Jan','Feb','Mar','Apr','May','Jun',
                        'Jul','Aug','Sep','Oct','Nov','Dec'][:len(m_table.columns)]
    m_table['Yearly Total'] = m_table.apply(lambda row: np.prod(1 + row) - 1, axis=1)

    c_rets = np.array(results[champ])
    total_return_c    = np.prod(1 + c_rets) - 1
    num_days_c        = len(c_rets)
    annualized_return = (1 + total_return_c) ** (252 / num_days_c) - 1

    return {
        'champion':            champ,
        'runner_up':           runner_up,
        'champion_prediction': forecasts[champ]['etf'],
        'runner_up_prediction': forecasts[runner_up]['etf'],
        'champion_hold':       forecasts[champ]['optimal_hold'],
        'runner_up_hold':      forecasts[runner_up]['optimal_hold'],
        'annualized_return':   annualized_return,
        'sharpe': (np.mean(c_rets) - (0.053 / 252)) / np.std(c_rets) * np.sqrt(252)
                  if np.std(c_rets) > 0 else 0,
        'recency':       recency_scores[champ],
        'oos_years':     oos_years,
        'monthly_table': m_table,
        'period_stats':  period_stats,
        'results':       results,
        'test_dates':    test_dates,
    }


# --- 5. SHARED RESULTS RENDERER ---
def render_results(s, universe_label):
    """Render ensemble results for any universe."""
    st.subheader(f"🎯 {universe_label} — Ensemble Consensus Prediction")
    cons1, cons2, cons3 = st.columns(3)
    etf_with_hold = f"{s['consensus_etf']} (Hold: {s['consensus_hold']}d)"
    if s['confidence'] == "HIGH":
        cons1.success(f"**{etf_with_hold}**")
    elif s['confidence'] == "MEDIUM":
        cons1.info(f"**{etf_with_hold}**")
    else:
        cons1.warning(f"**{etf_with_hold}**")
    cons1.caption(f"Consensus: {s['consensus_votes']}/{s['total_periods']} periods")
    cons2.metric("Confidence Level", s['confidence'], delta=f"Valid: {s['next_day']}")
    cons3.metric("Model Agreement",
                 f"{s['agreement_count']}/{s['total_periods']} periods",
                 delta="Champion = Runner-up")
    st.caption(f"**Training Periods Used:** {', '.join(map(str, s['ensemble_years']))}")

    # Voting breakdown
    st.divider()
    st.subheader("📊 Voting Breakdown by Training Period")
    vote_df = pd.DataFrame([{
        'Training Period':    f"{year}-2026",
        'Champion':          r['champion'],
        'Champion Predicts': r['champion_prediction'],
        'Hold':              f"{r['champion_hold']}d",
        'Runner-Up':         r['runner_up'],
        'Runner-Up Predicts': r['runner_up_prediction'],
        'Agreement':         '✅' if r['champion_prediction'] == r['runner_up_prediction'] else '❌',
    } for year, r in s['ensemble_results'].items()])
    st.dataframe(vote_df, use_container_width=True)

    # Best period
    br  = s['best_result']
    bpy = s['best_period']
    st.divider()
    st.subheader(f"📈 Best Performing Period: {bpy}-2026")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Champion Model",   br['champion'])
    c2.metric("Annualized Return", f"{br['annualized_return']:.2%}", delta=f"OOS: {br['oos_years']}")
    c3.metric("Sharpe Ratio",     f"{br['sharpe']:.2f}",           delta=f"SOFR: {s['rf']:.2%}")
    c4.metric("Recency Score (15d)", f"{br['recency']:.0%}")

    # Performance chart
    st.divider()
    fig = go.Figure()
    for name, r in br['results'].items():
        if len(r) > 0:
            fig.add_trace(go.Scatter(
                x=br['test_dates'][:len(r)],
                y=np.cumprod(1 + np.array(r)),
                name=name))
    fig.update_layout(
        title=f"{universe_label} — Net Return Performance ({br['oos_years']} OOS)",
        template="plotly_dark", height=400)
    st.plotly_chart(fig, use_container_width=True)

    # Monthly matrix
    st.subheader(f"📅 Monthly Matrix — {bpy} Period ({br['champion']})")
    st.dataframe(br['monthly_table'].style.format("{:.2%}"), use_container_width=True)

    # OOS period analysis
    if 'period_stats' in br:
        st.divider()
        st.subheader(f"📊 OOS Period Analysis — {bpy} Training")
        etf_df = pd.DataFrame({
            etf: {
                'Total Return':    f"{stats['total_return']:.2%}",
                'Sharpe Ratio':    f"{stats['sharpe']:.2f}",
                'Daily Volatility': f"{stats['std_daily']:.3%}",
            }
            for etf, stats in br['period_stats'].items()
        }).T
        st.dataframe(etf_df, use_container_width=True)

    # Methodology
    st.divider()
    st.header("🔍 Methodology")
    st.info("""
**Ensemble Voting System:**

1. **Multiple Training Periods:** Models train on historical periods using an 80/20 train/test split.
2. **Independent Tournaments:** Each period runs 4 architectures (PPO, A2C, CNN-LSTM, Transformer).
3. **Consensus Voting:** Final prediction is the majority vote across all periods.
4. **Confidence Scoring:**
   - **HIGH:** 4+ periods agree (≥67%)
   - **MEDIUM:** 3 periods agree (50%)
   - **LOW:** Split vote
5. **Hold Period Optimization:** Tests 1-, 3-, and 5-day holds; selects highest annualized return after costs.
""")


# --- 6. SHARED TOURNAMENT RUNNER ---
def run_ensemble(universe_etfs, ensemble_years, t_cost, session_key):
    """Run ensemble tournament for a universe and store in session_state."""
    with st.status("Running Ensemble Tournament...") as status:
        try:
            rf               = get_sofr_rate(FRED_API_KEY)
            ensemble_results = {}
            prog             = st.empty()

            for idx, start_year in enumerate(ensemble_years):
                prog.info(f"Training period {start_year}... ({idx+1}/{len(ensemble_years)})")
                data_tuple = load_data_from_hf(start_year, HF_TOKEN, HF_DATASET_REPO, tuple(universe_etfs))
                if data_tuple[0] is None:
                    st.warning(f"Skipping {start_year}: No data available")
                    continue
                (features_df, returns_df), _ = data_tuple
                result = run_tournament_engine(
                    features_df, returns_df, rf, t_cost, start_year, tuple(universe_etfs))
                if result is not None:
                    ensemble_results[start_year] = result
                    prog.success(f"✓ {start_year} complete. Champion: {result['champion']}")
                else:
                    prog.warning(f"✗ {start_year} failed")

            prog.empty()

            if not ensemble_results:
                st.error("Failed to run any training periods.")
                st.stop()

            champion_votes  = Counter([r['champion_prediction'] for r in ensemble_results.values()])
            runner_up_votes = Counter([r['runner_up_prediction'] for r in ensemble_results.values()])
            agreement_count = sum(1 for r in ensemble_results.values()
                                  if r['champion_prediction'] == r['runner_up_prediction'])

            consensus_etf   = champion_votes.most_common(1)[0][0]
            consensus_votes = champion_votes[consensus_etf]

            consensus_holds = [r['champion_hold'] for r in ensemble_results.values()
                               if r['champion_prediction'] == consensus_etf]
            consensus_hold  = Counter(consensus_holds).most_common(1)[0][0] if consensus_holds else 1

            confidence = "HIGH" if consensus_votes >= 4 else ("MEDIUM" if consensus_votes == 3 else "LOW")

            best_period = max(ensemble_results, key=lambda k: ensemble_results[k]['annualized_return'])

            st.session_state[session_key] = {
                "ensemble_results": ensemble_results,
                "consensus_etf":    consensus_etf,
                "consensus_hold":   consensus_hold,
                "consensus_votes":  consensus_votes,
                "total_periods":    len(ensemble_results),
                "champion_votes":   dict(champion_votes),
                "runner_up_votes":  dict(runner_up_votes),
                "agreement_count":  agreement_count,
                "confidence":       confidence,
                "best_result":      ensemble_results[best_period],
                "best_period":      best_period,
                "rf":               rf,
                "next_day":         get_next_trading_day(),
                "ensemble_years":   ensemble_years,
            }
            status.update(label="Ensemble Tournament Complete!", state="complete")
        except Exception as e:
            status.update(label=f"Error: {str(e)}", state="error")
            st.error(f"Failed to run tournament: {str(e)}")
            import traceback; st.code(traceback.format_exc())
            st.stop()
    st.rerun()


# ─────────────────────────────────────────────────────────────────────────────
# --- 7. UI ---
# ─────────────────────────────────────────────────────────────────────────────
st.title("Alpha Tournament Pro: Multi-model ETF Forecast")

# ── Sidebar (shared controls) ─────────────────────────────────────────────────
with st.sidebar:
    st.header("Tournament Configuration")
    t_cost = st.slider("Transaction Cost (bps)", 0, 100, 10, 5)
    st.divider()
    st.subheader("Training Periods")

    period_option = st.radio(
        "Select Training Period Option:",
        ["Option A (Default)", "Option B (Custom)"],
    )
    if period_option == "Option A (Default)":
        ensemble_years = DEFAULT_ENSEMBLE_YEARS
        st.info(f"**Using Years:** {', '.join(map(str, ensemble_years))}")
    else:
        st.caption("**Select 6 Years:**")
        current_year    = datetime.now().year
        available_years = list(range(2008, current_year))
        selected_years  = [
            st.selectbox(f"Period {i+1}", available_years,
                         index=min(i * 2, len(available_years) - 1),
                         key=f"year_{i}")
            for i in range(6)
        ]
        ensemble_years = sorted(set(selected_years))
        if len(ensemble_years) < len(set(selected_years)):
            st.warning(f"⚠️ Duplicate years removed. Using {len(ensemble_years)} unique periods.")

    st.divider()
    run_fi_btn = st.button("🏦 Run FI Tournament",  use_container_width=True)
    run_eq_btn = st.button("📈 Run Equity Tournament", use_container_width=True)

# ── Trigger runs ──────────────────────────────────────────────────────────────
if run_fi_btn:
    run_ensemble(FI_ETFS, ensemble_years, t_cost, "results_fi")

if run_eq_btn:
    run_ensemble(EQ_ETFS, ensemble_years, t_cost, "results_eq")

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab_fi, tab_eq = st.tabs(["🏦  Fixed Income (FI)", "📈  Equities"])

with tab_fi:
    if st.session_state.results_fi:
        render_results(st.session_state.results_fi, "Fixed Income ETFs")
    else:
        st.info(
            "No FI results yet. Configure the sidebar and click **🏦 Run FI Tournament**.\n\n"
            f"**Universe:** {', '.join(FI_ETFS)}"
        )

with tab_eq:
    if st.session_state.results_eq:
        render_results(st.session_state.results_eq, "Equity ETFs")
    else:
        st.info(
            "No Equity results yet. Configure the sidebar and click **📈 Run Equity Tournament**.\n\n"
            f"**Universe:** {', '.join(EQ_ETFS)}"
        )
