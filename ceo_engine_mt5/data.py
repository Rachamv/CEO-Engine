"""
The CEO Protocol — Layer 1: Data Module
============================================
Instrument-agnostic OHLCV fetcher.
Supports: yfinance (stocks, forex, indices), ccxt (crypto), CSV, MT5
"""

import pandas as pd
from datetime import datetime

_YF_TF_MAP = {
    "1m":"1m","2m":"2m","5m":"5m","15m":"15m","30m":"30m",
    "1h":"1h","60m":"1h","90m":"90m","2h":"2h","4h":"4h",
    "6h":"6h","8h":"8h","12h":"12h","1d":"1d","d":"1d",
    "5d":"5d","1w":"1wk","w":"1wk","1mo":"1mo","mo":"1mo",
}
_CCXT_TF_MAP = {
    "1m":"1m","3m":"3m","5m":"5m","15m":"15m","30m":"30m",
    "1h":"1h","60m":"1h","2h":"2h","4h":"4h","6h":"6h",
    "8h":"8h","12h":"12h","1d":"1d","d":"1d","3d":"3d",
    "1w":"1w","w":"1w","1mo":"1M","mo":"1M",
}
_RESAMPLE_MAP = {
    "1m":"1min","2m":"2min","3m":"3min","5m":"5min",
    "15m":"15min","30m":"30min","1h":"1h","60m":"1h",
    "2h":"2h","4h":"4h","6h":"6h","8h":"8h",
    "12h":"12h","1d":"1D","5d":"5D","1w":"1W","1mo":"1ME",
}

def _normalise_tf(tf, target):
    tf = tf.lower().strip()
    mapping = _YF_TF_MAP if target == "yfinance" else _CCXT_TF_MAP
    if tf not in mapping:
        raise ValueError(f"Unsupported timeframe '{tf}' for {target}.")
    return mapping[tf]

def _to_utc(dt_index):
    if dt_index.tz is None:
        return dt_index.tz_localize("UTC")
    return dt_index.tz_convert("UTC")

def _estimate_tick_size(close_series):
    sample = close_series.dropna().head(100)
    if sample.empty: return 0.00001
    price = sample.median()
    if price < 0.01: return 0.0000001
    elif price < 1:  return 0.00001
    elif price < 10: return 0.0001
    elif price < 100: return 0.01
    elif price < 1000: return 0.1
    elif price < 10000: return 1.0
    else: return 10.0

def _clean(df, symbol):
    df.columns = [c.lower() for c in df.columns]
    if "close" not in df.columns and "adj close" in df.columns:
        df = df.rename(columns={"adj close": "close"})
    required = ["open","high","low","close"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")
    if "volume" not in df.columns:
        df["volume"] = 0.0
    df = df[["open","high","low","close","volume"]].copy()
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["open","high","low","close"])
    df = df[df["close"] > 0]
    df.index = _to_utc(df.index)
    df.index.name = "datetime"
    df = df.sort_index()
    df.attrs["symbol"] = symbol
    # Only estimate tick_size if the fetcher hasn't already supplied a real one
    # (e.g. _fetch_mt5() sets it from the broker's mt5.symbol_info() — that's
    # more accurate than a price-magnitude guess and must not be clobbered here).
    if not df.attrs.get("tick_size"):
        df.attrs["tick_size"] = _estimate_tick_size(df["close"])
    return df

def _fetch_yfinance(symbol, timeframe, start=None, end=None):
    try:
        import yfinance as yf
    except ImportError:
        raise ImportError("Run: pip install yfinance")
    interval = _normalise_tf(timeframe, "yfinance")
    ticker = yf.Ticker(symbol)
    df = ticker.history(interval=interval, start=start, end=end,
                        auto_adjust=True, actions=False)
    if df.empty:
        raise ValueError(f"yfinance returned no data for '{symbol}' [{timeframe}].")
    return df

def _fetch_ccxt(symbol, timeframe, start=None, end=None,
                exchange="binance", limit_per_call=1000):
    try:
        import ccxt
    except ImportError:
        raise ImportError("Run: pip install ccxt")
    tf = _normalise_tf(timeframe, "ccxt")
    try:
        ex = getattr(ccxt, exchange)({"enableRateLimit": True})
    except AttributeError:
        raise ValueError(f"Unknown ccxt exchange: '{exchange}'")
    ex.load_markets()
    since  = int(pd.Timestamp(start, tz="UTC").timestamp()*1000) if start else None
    end_ms = int(pd.Timestamp(end,   tz="UTC").timestamp()*1000) if end   else None
    all_rows = []
    fetch_since = since
    while True:
        ohlcv = ex.fetch_ohlcv(symbol, timeframe=tf, since=fetch_since,
                                limit=limit_per_call)
        if not ohlcv: break
        all_rows.extend(ohlcv)
        last_ts = ohlcv[-1][0]
        if len(ohlcv) < limit_per_call: break
        if end_ms and last_ts >= end_ms: break
        fetch_since = last_ts + 1
    if not all_rows:
        raise ValueError(f"ccxt returned no data for '{symbol}' [{timeframe}].")
    df = pd.DataFrame(all_rows,
                      columns=["datetime","open","high","low","close","volume"])
    df["datetime"] = pd.to_datetime(df["datetime"], unit="ms", utc=True)
    df = df.set_index("datetime")
    if end_ms:
        df = df[df.index <= pd.Timestamp(end, tz="UTC")]
    return df

def _fetch_csv(filepath, datetime_col="datetime"):
    df = pd.read_csv(filepath)
    dt_candidates = [c for c in df.columns
                     if any(k in c.lower() for k in ["date","time","timestamp"])]
    if datetime_col in df.columns:
        dt_col = datetime_col
    elif dt_candidates:
        dt_col = dt_candidates[0]
    else:
        dt_col = df.columns[0]
    df[dt_col] = pd.to_datetime(df[dt_col], utc=True)
    df = df.set_index(dt_col)
    df.index.name = "datetime"
    return df

def _fetch_mt5(symbol, timeframe, start=None, end=None, n_bars=5000):
    from .mt5_connect import MT5Connection, _require_mt5
    _require_mt5()
    conn = MT5Connection()
    conn.connect()
    try:
        sym_info = conn.symbol_info(symbol)
        tick_sz  = sym_info["tick_size"]
        if start and end:
            start_dt = datetime.strptime(start, "%Y-%m-%d")
            end_dt   = datetime.strptime(end,   "%Y-%m-%d")
            rates    = conn.fetch_rates(symbol, timeframe,
                                        start=start_dt, end=end_dt)
        elif start:
            start_dt = datetime.strptime(start, "%Y-%m-%d")
            rates    = conn.fetch_rates(symbol, timeframe,
                                        start=start_dt, n_bars=n_bars)
        else:
            rates = conn.fetch_rates(symbol, timeframe, n_bars=n_bars)
        df = pd.DataFrame(rates)
        df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
        df = df.rename(columns={"time":"datetime","tick_volume":"volume"})
        df = df.set_index("datetime")
        df.attrs["tick_size"] = tick_sz
        return df[["open","high","low","close","volume"]]
    finally:
        conn.disconnect()

def resample_ohlcv(df, target_tf):
    tf = target_tf.lower().strip()
    if tf not in _RESAMPLE_MAP:
        raise ValueError(f"Unsupported resample timeframe '{tf}'.")
    rule = _RESAMPLE_MAP[tf]
    resampled = df.resample(rule, label="left", closed="left").agg({
        "open":"first","high":"max","low":"min",
        "close":"last","volume":"sum",
    }).dropna(subset=["open","close"])
    resampled.attrs = df.attrs.copy()
    return resampled

def fetch_ohlcv(symbol, timeframe="1h", source="yfinance",
                start=None, end=None, exchange="binance",
                filepath=None, n_bars=5000):
    source = source.lower().strip()
    if source == "yfinance":
        raw = _fetch_yfinance(symbol, timeframe, start=start, end=end)
    elif source == "ccxt":
        raw = _fetch_ccxt(symbol, timeframe, start=start, end=end,
                          exchange=exchange)
    elif source == "csv":
        if not filepath:
            raise ValueError("filepath required for source='csv'")
        raw = _fetch_csv(filepath)
    elif source == "mt5":
        raw = _fetch_mt5(symbol, timeframe, start=start, end=end,
                         n_bars=n_bars)
    else:
        raise ValueError(f"Unknown source '{source}'. Use yfinance/ccxt/csv/mt5.")
    df = _clean(raw, symbol)
    print(f"✅  {symbol} [{timeframe}] — {len(df):,} bars | "
          f"{df.index[0].strftime('%Y-%m-%d')} → "
          f"{df.index[-1].strftime('%Y-%m-%d')} | "
          f"tick_size={df.attrs['tick_size']}")
    return df
