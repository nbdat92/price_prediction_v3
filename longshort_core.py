# longshort_core.py
import os
import numpy as np
import pandas as pd

# ---------- LOAD CSV ----------
def load_mt5_csv(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"CSV not found: {path}")
    try:
        df = pd.read_csv(path, sep=';')
        if df.shape[1] == 1:
            df = pd.read_csv(path)
    except Exception:
        df = pd.read_csv(path)
    df.columns = [c.strip().lower() for c in df.columns]
    m = {}
    for a, b in [('time', 'Datetime'), ('date', 'Datetime'), ('datetime', 'Datetime'),
                 ('open', 'Open'), ('high', 'High'), ('low', 'Low'), ('close', 'Close')]:
        if a in df.columns:
            m[a] = b
    df = df.rename(columns=m)
    if 'Datetime' not in df.columns:
        raise RuntimeError("CSV cần có cột thời gian (time/date/datetime).")
    df['Datetime'] = pd.to_datetime(df['Datetime'])
    df = df.set_index('Datetime').sort_index()
    return df[['Open', 'High', 'Low', 'Close']].dropna()

# ---------- FEATURES ----------
def rsi(s: pd.Series, n: int = 14) -> pd.Series:
    d = s.diff()
    up = d.clip(lower=0).rolling(n).mean()
    dn = (-d.clip(upper=0)).rolling(n).mean()
    rs = up / dn.replace(0, np.nan)
    return 100 - (100 / (1 + rs))

def ema(s: pd.Series, n: int) -> pd.Series:
    return s.ewm(span=n, adjust=False).mean()

def sma(s: pd.Series, n: int) -> pd.Series:
    return s.rolling(n).mean()

def atr(h: pd.Series, l: pd.Series, c: pd.Series, n: int = 14) -> pd.Series:
    tr = pd.concat([(h - l).abs(), (h - c.shift(1)).abs(), (l - c.shift(1)).abs()], axis=1).max(axis=1)
    return tr.rolling(n).mean()

def boll(c: pd.Series, n: int = 20, k: float = 2.0):
    m = c.rolling(n).mean()
    sd = c.rolling(n).std()
    return m - k * sd, m, m + k * sd

def stochastic(h, l, c, k: int = 14, d: int = 3):
    ll = l.rolling(k).min()
    hh = h.rolling(k).max()
    kf = 100 * (c - ll) / (hh - ll)
    return kf, kf.rolling(d).mean()

FEATURES = [
    'ret_1', 'ret_5', 'ret_10',
    'sma10', 'sma20', 'sma50', 'ema12', 'ema26',
    'atr14', 'bb_low', 'bb_mid', 'bb_up', 'bb_width',
    'rsi14', 'stoch_k', 'stoch_d', 'dow', 'hour'
]

def make_features(df: pd.DataFrame) -> pd.DataFrame:
    o = df.copy()
    o['ret_1'] = o['Close'].pct_change(1)
    o['ret_5'] = o['Close'].pct_change(5)
    o['ret_10'] = o['Close'].pct_change(10)
    o['sma10'] = sma(o['Close'], 10)
    o['sma20'] = sma(o['Close'], 20)
    o['sma50'] = sma(o['Close'], 50)
    o['ema12'] = ema(o['Close'], 12)
    o['ema26'] = ema(o['Close'], 26)
    o['atr14'] = atr(o['High'], o['Low'], o['Close'], 14)
    bl, bm, bu = boll(o['Close'], 20, 2.0)
    o['bb_low'], o['bb_mid'], o['bb_up'] = bl, bm, bu
    o['bb_width'] = (bu - bl) / bm
    o['rsi14'] = rsi(o['Close'], 14)
    kf, ds = stochastic(o['High'], o['Low'], o['Close'], 14, 3)
    o['stoch_k'], o['stoch_d'] = kf, ds
    o['dow'] = o.index.dayofweek
    o['hour'] = o.index.hour
    return o.dropna()

# ---------- THRESHOLDS ----------
def auto_thresholds(summary_path: str = "data/wf_summary_longshort.csv"):
    if os.path.exists(summary_path):
        s = pd.read_csv(summary_path)
        if {'thr_long', 'thr_short'}.issubset(s.columns):
            return float(np.median(s['thr_long'])), float(np.median(s['thr_short'])), 0.02
    return 0.58, 0.42, 0.02

# ---------- MODEL ----------
def load_or_train_model(X: pd.DataFrame, y: pd.Series, model_path: str = "models/live_lgb.txt"):
    import lightgbm as lgb
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    if os.path.exists(model_path):
        return lgb.Booster(model_file=model_path)
    dtrain = lgb.Dataset(X, label=y.astype(int))
    params = dict(
        objective='binary',
        metric='binary_logloss',
        learning_rate=0.05,
        num_leaves=127,
        min_data_in_leaf=50,
        feature_fraction=0.9,
        bagging_fraction=0.9,
        bagging_freq=1,
        verbose=-1
    )
    model = lgb.train(params, dtrain, num_boost_round=1200, callbacks=[lgb.log_evaluation(0)])
    model.save_model(model_path)
    return model

# ---------- PAYLOAD CHO CHART ----------
def build_payload_for_chart(df: pd.DataFrame, prob: pd.Series | None,
                            thrL: float, thrS: float, dz: float, n_last: int = 300, stride: int = 1, smooth: int = 5):
    use_idx = df.index
    if prob is not None and smooth > 1:
        prob = prob.rolling(smooth, min_periods=1).mean()
        use_idx = prob.index

    tail_idx = use_idx[-n_last::stride]
    use = df.loc[tail_idx]

    payload = {
        "time": use.index.strftime("%Y-%m-%d %H:%M:%S").tolist(),
        "open": use['Open'].tolist(),
        "high": use['High'].tolist(),
        "low": use['Low'].tolist(),
        "close": use['Close'].tolist(),
        "thrL": float(thrL), "thrS": float(thrS), "dz": float(dz),
        "prob": None, "long_x": [], "long_y": [], "short_x": [], "short_y": []
    }

    if prob is not None:
        pr = prob.loc[tail_idx]
        payload["prob"] = pr.round(4).tolist()

        sig = np.where(np.abs(pr - 0.5) < dz, 0,
                       np.where(pr > thrL, 1,
                                np.where(pr < thrS, -1, 0)))
        sig_series = pd.Series(sig, index=tail_idx)
        change_idx = sig_series[sig_series.diff().fillna(0) != 0].index

        payload["long_x"] = [t.strftime("%Y-%m-%d %H:%M:%S") for t in change_idx if sig_series.loc[t] == 1]
        payload["long_y"] = [float(df.loc[t, 'Close']) for t in change_idx if sig_series.loc[t] == 1]
        payload["short_x"] = [t.strftime("%Y-%m-%d %H:%M:%S") for t in change_idx if sig_series.loc[t] == -1]
        payload["short_y"] = [float(df.loc[t, 'Close']) for t in change_idx if sig_series.loc[t] == -1]

    return payload
