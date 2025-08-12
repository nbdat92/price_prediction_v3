# app.py
import os
from datetime import datetime, timedelta

from flask import Flask, render_template, jsonify, request
from apscheduler.schedulers.background import BackgroundScheduler

import MetaTrader5 as mt5
import pandas as pd

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
    "H1":  (mt5.TIMEFRAME_H1, 60),
    "H4":  (mt5.TIMEFRAME_H4, 240),
}

SETTINGS = {
    "symbol": "XAUUSD",
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
    _ensure_mt5()
    symbol = SETTINGS["symbol"]
    tf_str = SETTINGS["timeframe"]
    tf_enum, _ = TF_MAP[tf_str]

    if not mt5.symbol_select(symbol, True):
        mt5.shutdown()
        raise RuntimeError(f"Cannot select symbol {symbol}")

    dt_from = datetime.fromisoformat(SETTINGS["from"])
    dt_to = datetime.now() if SETTINGS["to"] is None else datetime.fromisoformat(SETTINGS["to"])
    dt_to_eff = dt_to + timedelta(hours=12)  # tránh hụt phiên mở / múi giờ

    rates = mt5.copy_rates_range(symbol, tf_enum, dt_from, dt_to_eff)
    mt5.shutdown()

    if rates is None or len(rates) == 0:
        raise RuntimeError("copy_rates_range returned empty")

    df = pd.DataFrame(rates)
    df['Datetime'] = pd.to_datetime(df['time'], unit='s')
    df = df.set_index('Datetime').sort_index()
    df = df[['open', 'high', 'low', 'close']].rename(columns=str.title)

    path = csv_path_for(tf_str)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, sep=';')
    return path, len(df)

# ---------- TRAIN NHANH THEO TF ----------
def train_quick_for_tf():
    path = csv_path_for(SETTINGS["timeframe"])
    df = load_mt5_csv(path)
    feat = make_features(df)
    y = (feat['Close'].shift(-1) / feat['Close'] - 1.0 > 0).astype(int)
    data = feat.join(y.rename('y')).dropna()
    X = data[FEATURES]; y = data['y']
    model = load_or_train_model(X, y, model_path=f"models/live_lgb_{SETTINGS['timeframe']}.txt")
    return len(X)

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
    n = int(request.args.get("n", 300))
    smooth = int(request.args.get("smooth", 5))
    stride = int(request.args.get("stride", 1))

    path = csv_path_for(SETTINGS["timeframe"])
    df = load_mt5_csv(path)
    feat = make_features(df)
    y = (feat['Close'].shift(-1)/feat['Close'] - 1.0 > 0).astype(int)
    data = feat.join(y.rename('y')).dropna()

    X = data[FEATURES]; y = data['y']
    thrL, thrS, dz = auto_thresholds(f"data/wf_summary_longshort.csv")  # nếu có
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

    thrL, thrS, dz = auto_thresholds(f"data/wf_summary_longshort.csv")
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
    # Nếu đã có file equity của backtest long-short thì dùng, không thì trả rỗng
    p1 = "data/wf_equity_curve_longshort.csv"
    p2 = "data/wf_equity_curve.csv"
    if os.path.exists(p1):
        eq = pd.read_csv(p1)
    elif os.path.exists(p2):
        eq = pd.read_csv(p2)
    else:
        return jsonify(dict(time=[], equity=[]))
    return jsonify(dict(time=eq.iloc[:, 0].tolist(), equity=eq.iloc[:, 1].round(4).tolist()))

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
    m = train_quick_for_tf()
    return jsonify({"ok": True, "csv": p, "rows": n, "trained_samples": m})

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
