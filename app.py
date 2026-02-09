# --- UPDATED TOURNAMENT ENGINE ---
def run_tournament(data):
    # Prep Features & Labels
    rets = data[TARGET_ETFS].pct_change().dropna()
    features = data.shift(1).dropna()
    idx = rets.index.intersection(features.index)
    X, y = features.loc[idx], rets.loc[idx]
    
    # Sequence Processing (10-day lookback)
    seq_len = 10
    X_s, y_s = [], []
    for i in range(len(X) - seq_len):
        X_s.append(X.iloc[i:i+seq_len].values)
        y_s.append(y.iloc[i+seq_len].values)
    X_s, y_s = np.array(X_s), np.array(y_s)

    # Split 80/20
    split = int(len(X_s) * 0.8)
    X_train, X_live = X_s[:split], X_s[split:]
    y_train, y_live = y_s[:split], y_s[split:]

    # Scaling with NaN protection
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
    rl_train_df = pd.DataFrame(X_train_sc[:, -1, :]).join(pd.DataFrame(y_train, columns=TARGET_ETFS))
    env = DummyVecEnv([lambda: TradingEnv(rl_train_df, TARGET_ETFS)])
    
    ppo = PPO("MlpPolicy", env, verbose=0).learn(total_timesteps=1500)
    a2c = A2C("MlpPolicy", env, verbose=0).learn(total_timesteps=1500)
    
    # FIXED PREDICTION LOOP
    ppo_actions, a2c_actions = [], []
    for obs in X_live_sc[:, -1, :]:
        # Ensure obs is a 2D batch [1, features] and float32
        obs_input = np.array([obs], dtype=np.float32) 
        p_act, _ = ppo.predict(obs_input, deterministic=True)
        a_act, _ = a2c.predict(obs_input, deterministic=True)
        
        # SB3 predict returns an array for vectorized envs, so we take [0]
        ppo_actions.append(p_act[0])
        a2c_actions.append(a_act[0])
    
    results['PPO'] = [y_live[i, a] for i, a in enumerate(ppo_actions)]
    results['A2C'] = [y_live[i, a] for i, a in enumerate(a2c_actions)]

    # SEQUENCE MODELS (CNN-LSTM & Transformer)
    X_t = torch.tensor(X_train_sc, dtype=torch.float32)
    y_t = torch.tensor(y_train, dtype=torch.float32)
    X_l_t = torch.tensor(X_live_sc, dtype=torch.float32)

    for name, m_class in [("CNN-LSTM", CNN_LSTM_Model), ("Transformer", TransformerModel)]:
        model = m_class(X_t.shape[2], len(TARGET_ETFS))
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
