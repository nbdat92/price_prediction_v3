# app.py
import os
from datetime import datetime, timedelta

from flask import Flask, render_template, jsonify, request
from apscheduler.schedulers.background import BackgroundScheduler

import MetaTrader5 as mt5
import pandas as pd
import numpy as np

from longshort_core import (
    load_mt5_csv, make_features, FEATURES, auto_thresholds,
    load_or_train_model, build_payload_for_chart
)

app = Flask(__name__)

# ---------- CẤU HÌNH & MAP TIMEFRAME ----------
TF_MAP = {
    "M1":  (mt5.TIMEFRAME_M1, 1),
    "M5":  (mt5.TIMEFRAME_M5, 5),
    "M15": (mt5.TIMEFRAME_M15, 15),
    "M30": (mt5.TIMEFRAME_M30, 30),
    "H1":  (mt5.TIMEFRAME_H1, 60),
    "H4":  (mt5.TIMEFRAME_H4, 240),
    "D1":  (mt5.TIMEFRAME_D1, 1440),
}

SETTINGS = {
    "symbol": "XAUUSDm",
    "timeframe": "M15",
    "from": "2024-01-01",
    "to": None,        # None = đến hiện tại
    "auto_job": True
}

def csv_path_for(tf: str) -> str:
    return f"data/{SETTINGS['symbol']}_{tf}.csv"

def _ensure_mt5():
    if not mt5.initialize():
        raise RuntimeError(f"Initialize MT5 failed: {mt5.last_error()}")

# ---------- FETCH CSV TỪ MT5 ----------
def fetch_csv_from_mt5():
    from datetime import timezone

    def _floor_open_utc(dt_utc, tf_minutes):
        tf_sec = tf_minutes * 60
        ts = int(dt_utc.timestamp())
        open_ts = (ts // tf_sec) * tf_sec
        return datetime.fromtimestamp(open_ts, tz=timezone.utc)

    _ensure_mt5()
    symbol = SETTINGS["symbol"]
    tf_str = SETTINGS["timeframe"]
    tf_enum, tf_minutes = TF_MAP[tf_str]

    if not mt5.symbol_select(symbol, True):
        mt5.shutdown()
        raise RuntimeError(f"Cannot select symbol {symbol}")

    # --- FROM/TO chuẩn hoá về UTC (có tz) ---
    dt_from = datetime.fromisoformat(SETTINGS["from"])
    if dt_from.tzinfo is None:
        dt_from = dt_from.replace(tzinfo=timezone.utc)
    else:
        dt_from = dt_from.astimezone(timezone.utc)

    if SETTINGS["to"] is None:
        dt_to_raw = datetime.now(timezone.utc)
    else:
        dt_to_raw = datetime.fromisoformat(SETTINGS["to"])
        if dt_to_raw.tzinfo is None:
            dt_to_raw = dt_to_raw.replace(tzinfo=timezone.utc)
        else:
            dt_to_raw = dt_to_raw.astimezone(timezone.utc)

    # --- CHỐT mốc TO = open(nến hiện tại) - 1 block (tương đương shift=1) ---
    now_utc = datetime.now(timezone.utc)
    current_open = _floor_open_utc(now_utc, tf_minutes)
    last_closed_open = current_open - timedelta(minutes=tf_minutes)

    # Nếu user đặt "to" xa hơn nến đã đóng -> hạ xuống last_closed_open
    dt_to_eff = min(dt_to_raw, last_closed_open)

    # Trường hợp cấu hình quá hẹp
    if dt_to_eff <= dt_from:
        # lấy tối thiểu 1 block
        dt_to_eff = (dt_from if dt_from <= last_closed_open else last_closed_open)

    # --- LẤY DỮ LIỆU ---
    rates = mt5.copy_rates_range(symbol, tf_enum, dt_from, dt_to_eff)
    mt5.shutdown()

    if rates is None or len(rates) == 0:
        raise RuntimeError("copy_rates_range returned empty")

    df = pd.DataFrame(rates)
    # time MT5 là epoch UTC -> giữ UTC có tzinfo
    df['Datetime'] = pd.to_datetime(df['time'], unit='s', utc=True)
    df = df.set_index('Datetime').sort_index()

    # --- PHÒNG HỜ: nếu lỡ dính nến hiện tại thì bỏ hàng cuối ---
    # nến hiện tại có index == current_open
    if not df.empty and df.index[-1] == current_open:
        df = df.iloc[:-1]

    df = df[['open', 'high', 'low', 'close']].rename(columns=str.title)

    path = csv_path_for(tf_str)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, sep=';')
    return path, len(df)


#----------- TRAINING -----------------------------
def tune_thresholds(y_true, y_prob, rets, costs_bp=2.0, deadzone=0.02):
    cost = costs_bp/10000.0
    best_thr, best_exp = 0.58, -1e9
    for thr in np.linspace(0.52, 0.65, 14):
        sig_long  = (y_prob > thr).astype(int)
        sig_short = (y_prob < (1-thr)).astype(int) * (-1)
        sig = sig_long + sig_short
        sig = np.where(np.abs(y_prob-0.5)<deadzone, 0, sig)
        pnl = sig[:-1]*rets[1:] - (sig[:-1]!=0)*cost
        expc = pnl.mean()
        if expc > best_exp:
            best_exp, best_thr = expc, thr
    return float(best_thr), float(1-best_thr), float(deadzone)

def backtest_ls(prob_series, close_series, thrL, thrS, dz, costs_bp=2.0):
    ret = close_series.pct_change().fillna(0.0)
    pr  = prob_series.reindex(close_series.index).ffill().fillna(0.5)
    sig = (pr>thrL).astype(int) - (pr<thrS).astype(int)
    sig = np.where(np.abs(pr-0.5)<dz, 0, sig)
    pos = pd.Series(sig, index=close_series.index).shift(1).fillna(0)
    fee = (pos!=0)*(costs_bp/10000.0)
    net = pos*ret - fee
    eq  = (1+net).cumprod()
    return pd.DataFrame({"equity":eq, "net_ret":net})

# ---------- TRAIN NHANH THEO TF ----------
def train_quick_for_tf():
    tf = SETTINGS["timeframe"]
    sym = SETTINGS["symbol"]
    path = csv_path_for(tf)

    df   = load_mt5_csv(path)
    feat = make_features(df)
    ybin = (feat['Close'].shift(-1)/feat['Close'] - 1.0 > 0).astype(int)
    data = feat.join(ybin.rename('y')).dropna()
    X, y = data[FEATURES], data['y']
    model = load_or_train_model(X, y, model_path=f"models/live_lgb_{sym}_{tf}.txt")

    # --- tính prob, tune ngưỡng, backtest và LƯU file ---
    prob = pd.Series(model.predict(X), index=X.index)
    thrL, thrS, dz = tune_thresholds(
        y_true=y.values,
        y_prob=prob.values,
        rets=data['Close'].pct_change().values,
        costs_bp=2.0, deadzone=0.02
    )
    bt = backtest_ls(prob, data['Close'], thrL, thrS, dz, costs_bp=2.0)

    # lưu SUMMARY & EQUITY kèm symbol/tf
    summ_path = f"data/{sym}_{tf}_wf_summary_longshort.csv"
    eq_path   = f"data/{sym}_{tf}_wf_equity_curve_longshort.csv"
    pd.DataFrame([{
        "symbol": sym, "tf": tf, "rows": len(X),
        "thr_long": round(thrL,4), "thr_short": round(thrS,4), "deadzone": dz,
        "equity_end": round(bt['equity'].iloc[-1],4)
    }]).to_csv(summ_path, index=False)
    bt[['equity']].to_csv(eq_path)

    return len(X), thrL, thrS, dz, summ_path, eq_path


# ---------- SCHEDULER ----------
scheduler = BackgroundScheduler(daemon=True)

def job_fetch_and_train():
    try:
        p, n = fetch_csv_from_mt5()
        m = train_quick_for_tf()
        print(f"[JOB] updated {p} rows={n}, trained_samples={m}")
    except Exception as e:
        print("[JOB ERROR]", e)

def schedule_job():
    scheduler.remove_all_jobs()
    tf = SETTINGS["timeframe"]
    interval_min = TF_MAP[tf][1]
    if SETTINGS["auto_job"]:
        scheduler.add_job(job_fetch_and_train, 'interval', minutes=interval_min, id='job')
    if not scheduler.running:
        scheduler.start()

# ---------- ROUTES ----------
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/equity")
def equity():
    return render_template("equity.html")

@app.route("/api/candles")
def api_candles():
    n = int(request.args.get("n", 50)) #số cây nến
    smooth = int(request.args.get("smooth", 5))
    stride = int(request.args.get("stride", 1))

    path = csv_path_for(SETTINGS["timeframe"])
    df = load_mt5_csv(path)
    feat = make_features(df)
    y = (feat['Close'].shift(-1)/feat['Close'] - 1.0 > 0).astype(int)
    data = feat.join(y.rename('y')).dropna()

    X = data[FEATURES]; y = data['y']
    thrL, thrS, dz = auto_thresholds(f"data/{SETTINGS['symbol']}_{SETTINGS['timeframe']}_wf_summary_longshort.csv")  # nếu có
    model = load_or_train_model(X, y, model_path=f"models/live_lgb_{SETTINGS['timeframe']}.txt")
    prob = pd.Series(model.predict(X), index=X.index)

    payload = build_payload_for_chart(df.loc[X.index], prob, thrL, thrS, dz,
                                      n_last=n, stride=stride, smooth=smooth)
    return jsonify(payload)

@app.route("/api/latest")
def api_latest():
    path = csv_path_for(SETTINGS["timeframe"])
    df = load_mt5_csv(path)
    feat = make_features(df)
    y = (feat['Close'].shift(-1)/feat['Close'] - 1.0 > 0).astype(int)
    data = feat.join(y.rename('y')).dropna()
    X = data[FEATURES]; y = data['y']

    thrL, thrS, dz = auto_thresholds(f"data/{SETTINGS['symbol']}_{SETTINGS['timeframe']}_wf_summary_longshort.csv")
    model = load_or_train_model(X, y, model_path=f"models/live_lgb_{SETTINGS['timeframe']}.txt")
    prob = float(model.predict(X.iloc[[-1]])[0])
    ts = X.index[-1].strftime("%Y-%m-%d %H:%M:%S")
    close_last = float(data['Close'].iloc[-1])

    sig = 0
    if abs(prob - 0.5) < dz: sig = 0
    elif prob > thrL: sig = 1
    elif prob < thrS: sig = -1

    return jsonify(dict(time=ts, close=close_last, prob_up=prob,
                        thrL=float(thrL), thrS=float(thrS), dz=float(dz),
                        signal=int(sig), tf=SETTINGS["timeframe"], symbol=SETTINGS["symbol"]))

@app.route("/api/equity")
def api_equity():
    p = f"data/{SETTINGS['symbol']}_{SETTINGS['timeframe']}_wf_equity_curve_longshort.csv"
    if not os.path.exists(p): return jsonify(dict(time=[], equity=[]))
    eq = pd.read_csv(p)
    return jsonify(dict(time=eq.iloc[:,0].tolist(), equity=eq.iloc[:,1].round(4).tolist()))

@app.route("/api/settings", methods=["GET", "POST"])
def api_settings():
    global SETTINGS
    if request.method == "GET":
        return jsonify(SETTINGS)
    data = request.get_json(force=True)
    if "symbol" in data: SETTINGS["symbol"] = data["symbol"]
    if "timeframe" in data and data["timeframe"] in TF_MAP: SETTINGS["timeframe"] = data["timeframe"]
    if "from" in data: SETTINGS["from"] = data["from"]
    if "to" in data: SETTINGS["to"] = data["to"]
    if "auto_job" in data: SETTINGS["auto_job"] = bool(data["auto_job"])
    schedule_job()
    return jsonify({"ok": True, "settings": SETTINGS})

@app.route("/api/train-now", methods=["POST"])
def api_train_now():
    p, n = fetch_csv_from_mt5()
    rows, thrL, thrS, dz, summ_path, eq_path = train_quick_for_tf()
    return jsonify({
        "ok": True,
        "csv": p, "rows_csv": n,
        "trained_samples": rows,
        "thrL": thrL, "thrS": thrS, "dz": dz,
        "summary": summ_path, "equity_curve": eq_path
    })

if __name__ == "__main__":
    os.makedirs("data", exist_ok=True)
    os.makedirs("models", exist_ok=True)
    schedule_job()  # bật job định kỳ theo TF
    # lần đầu: fetch + train ngay để có dữ liệu
    try:
        if not os.path.exists(csv_path_for(SETTINGS["timeframe"])):
            fetch_csv_from_mt5()
        train_quick_for_tf()
    except Exception as e:
        print("[INIT WARN]", e)
    app.run(debug=True)
