"""
The CEO Protocol — Layer 2: Indicator Calculations  v2.0
=============================================================
Fix: candle_parts() enforces OHLC integrity — body_ratio capped at 1.0,
     true high/low derived from all four prices to handle broker feed
     glitches or synthetic data where open/close fall outside high/low.
"""

import pandas as pd
import numpy as np


def rma(series, length):
    result = np.full(len(series), np.nan)
    values = series.values.astype(float)
    first_valid = np.where(~np.isnan(values))[0]
    if len(first_valid) < length:
        return pd.Series(result, index=series.index)
    start    = first_valid[0]
    seed_end = start + length
    if seed_end > len(values):
        return pd.Series(result, index=series.index)
    result[seed_end - 1] = np.nanmean(values[start:seed_end])
    alpha = 1.0 / length
    for i in range(seed_end, len(values)):
        if np.isnan(values[i]):
            result[i] = result[i - 1]
        else:
            result[i] = result[i-1] * (1 - alpha) + values[i] * alpha
    return pd.Series(result, index=series.index)

def ema(series, length):
    return series.ewm(span=length, adjust=False, min_periods=length).mean()

def sma(series, length):
    return series.rolling(window=length, min_periods=length).mean()

def atr(df, length=14):
    high       = df["high"]
    low        = df["low"]
    prev_close = df["close"].shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low  - prev_close).abs(),
    ], axis=1).max(axis=1)
    return rma(tr, length)

def rsi(series, length=14):
    delta     = series.diff()
    gains     = delta.clip(lower=0)
    losses    = (-delta).clip(lower=0)
    avg_gain  = rma(gains, length)
    avg_loss  = rma(losses, length)
    rs        = avg_gain / avg_loss.replace(0, np.nan)
    result    = 100.0 - (100.0 / (1.0 + rs))
    result    = result.where(avg_loss != 0, 100.0)
    return result

def adx(df, length=14):
    high  = df["high"].values.astype(float)
    low   = df["low"].values.astype(float)
    close = df["close"].values.astype(float)
    up_move   = np.diff(high,  prepend=high[0])
    down_move = np.diff(low,   prepend=low[0]) * -1
    plus_dm   = np.where((up_move > down_move)   & (up_move > 0),   up_move,   0.0)
    minus_dm  = np.where((down_move > up_move)   & (down_move > 0), down_move, 0.0)
    prev_close = np.concatenate([[close[0]], close[:-1]])
    tr = np.maximum.reduce([
        high - low,
        np.abs(high - prev_close),
        np.abs(low  - prev_close),
    ])
    tr_rma    = rma(pd.Series(tr,       index=df.index), length)
    plus_rma  = rma(pd.Series(plus_dm,  index=df.index), length)
    minus_rma = rma(pd.Series(minus_dm, index=df.index), length)
    plus_di   = (plus_rma  / tr_rma.replace(0, np.nan) * 100.0).fillna(0.0)
    minus_di  = (minus_rma / tr_rma.replace(0, np.nan) * 100.0).fillna(0.0)
    di_sum    = plus_di + minus_di
    dx        = (plus_di - minus_di).abs() / di_sum.replace(0, np.nan) * 100.0
    dx        = dx.fillna(0.0)
    adx_val   = rma(dx, length)
    return pd.DataFrame({"adx":adx_val,"plus_di":plus_di,"minus_di":minus_di},
                        index=df.index)

def pivot_high(series, left, right):
    values = series.values.astype(float)
    n      = len(values)
    result = np.full(n, np.nan)
    for i in range(left, n - right):
        center   = values[i]
        if np.isnan(center): continue
        left_ok  = all(values[i-j] <= center for j in range(1, left+1))
        right_ok = all(values[i+j] <= center for j in range(1, right+1))
        if left_ok and right_ok:
            result[i + right] = center
    return pd.Series(result, index=series.index)

def pivot_low(series, left, right):
    values = series.values.astype(float)
    n      = len(values)
    result = np.full(n, np.nan)
    for i in range(left, n - right):
        center   = values[i]
        if np.isnan(center): continue
        left_ok  = all(values[i-j] >= center for j in range(1, left+1))
        right_ok = all(values[i+j] >= center for j in range(1, right+1))
        if left_ok and right_ok:
            result[i + right] = center
    return pd.Series(result, index=series.index)

def candle_parts(df):
    """
    Candle anatomy with OHLC integrity enforcement.
    Body ratio capped at 1.0. True high/low derived from all four prices
    to handle malformed candles from broker feeds or synthetic data.
    """
    tick = df.attrs.get("tick_size", 1e-5)
    # Use true extremes — not just reported high/low
    true_high       = df[["high","open","close"]].max(axis=1)
    true_low        = df[["low", "open","close"]].min(axis=1)
    candle_range    = (true_high - true_low).clip(lower=tick)
    body_size       = (df["close"] - df["open"]).abs()
    body_ratio      = (body_size / candle_range).clip(upper=1.0)  # BUG FIX: cap at 1
    upper_wick      = (true_high - df[["open","close"]].max(axis=1)).clip(lower=0.0)
    lower_wick      = (df[["open","close"]].min(axis=1) - true_low).clip(lower=0.0)
    upper_rejection = upper_wick / candle_range
    lower_rejection = lower_wick / candle_range
    return pd.DataFrame({
        "candle_range":    candle_range,
        "body_size":       body_size,
        "body_ratio":      body_ratio,
        "upper_wick":      upper_wick,
        "lower_wick":      lower_wick,
        "upper_rejection": upper_rejection,
        "lower_rejection": lower_rejection,
    }, index=df.index)

def fvg(df, atr_series, atr_min=0.05, recent_bars=6):
    min_size       = atr_series * atr_min
    bull_fvg_raw   = (df["low"] > df["high"].shift(2)) & \
                     ((df["low"] - df["high"].shift(2)) >= min_size)
    bear_fvg_raw   = (df["high"] < df["low"].shift(2)) & \
                     ((df["low"].shift(2) - df["high"]) >= min_size)
    def barssince_le(s, n):
        return s.rolling(window=n, min_periods=1).max().astype(bool)
    return pd.DataFrame({
        "bull_fvg":        bull_fvg_raw,
        "bear_fvg":        bear_fvg_raw,
        "bull_fvg_recent": barssince_le(bull_fvg_raw, recent_bars),
        "bear_fvg_recent": barssince_le(bear_fvg_raw, recent_bars),
    }, index=df.index)

def displacement(df, atr_series, candle, atr_min=0.20, body_ratio_min=0.50):
    min_body = atr_series * atr_min
    body_ok  = (candle["body_size"] >= min_body) & \
               (candle["body_ratio"] >= body_ratio_min)
    return pd.DataFrame({
        "bull_displacement": (df["close"] > df["open"]) & body_ok,
        "bear_displacement": (df["close"] < df["open"]) & body_ok,
    }, index=df.index)

DEFAULT_PARAMS = {
    "atr_len":14,"ema_fast_len":50,"ema_slow_len":200,
    "rsi_len":14,"rsi_os":35.0,"rsi_ob":65.0,
    "vol_len":20,"vol_mult":1.30,
    "fvg_atr_min":0.05,"fvg_recent_bars":6,
    "disp_atr_min":0.20,"disp_body_ratio_min":0.50,
    "pivot_len":5,
    "adx_len":14,"adx_trend_min":20.0,"adx_range_max":16.0,
    "atr_regime_len":50,"high_vol_mult":1.25,"low_vol_mult":0.75,
    "chop_body_max":0.35,
}

def calc_all(df, params=None):
    p   = {**DEFAULT_PARAMS, **(params or {})}
    out = df.copy()
    out["atr"]      = atr(df, p["atr_len"])
    out["ema_fast"] = ema(df["close"], p["ema_fast_len"])
    out["ema_slow"] = ema(df["close"], p["ema_slow_len"])
    out["vol_sma"]  = sma(df["volume"], p["vol_len"])
    out["rsi_val"]  = rsi(df["close"], p["rsi_len"])
    out["trend_long"]  = out["ema_fast"] > out["ema_slow"]
    out["trend_short"] = out["ema_fast"] < out["ema_slow"]
    vol_ok           = df["volume"].notna() & out["vol_sma"].notna()
    out["vol_spike"] = vol_ok & (df["volume"] > out["vol_sma"] * p["vol_mult"])
    out["rsi_long"]  = out["rsi_val"] <= p["rsi_os"]
    out["rsi_short"] = out["rsi_val"] >= p["rsi_ob"]
    c = candle_parts(df)
    for col in c.columns:
        out[col] = c[col]
    f = fvg(df, out["atr"], atr_min=p["fvg_atr_min"],
            recent_bars=p["fvg_recent_bars"])
    for col in f.columns:
        out[col] = f[col]
    d = displacement(df, out["atr"], c,
                     atr_min=p["disp_atr_min"],
                     body_ratio_min=p["disp_body_ratio_min"])
    for col in d.columns:
        out[col] = d[col]
    adx_df = adx(df, p["adx_len"])
    for col in adx_df.columns:
        out[col] = adx_df[col]
    pl = p["pivot_len"]
    out["pivot_high"] = pivot_high(df["high"], pl, pl)
    out["pivot_low"]  = pivot_low(df["low"],   pl, pl)
    out["atr_regime_avg"] = sma(out["atr"], p["atr_regime_len"])
    out["avg_body_ratio"] = sma(out["body_ratio"], 20)
    out["high_vol_regime"] = (out["atr_regime_avg"].notna() &
        (out["atr"] > out["atr_regime_avg"] * p["high_vol_mult"]))
    out["low_vol_regime"]  = (out["atr_regime_avg"].notna() &
        (out["atr"] < out["atr_regime_avg"] * p["low_vol_mult"]))
    out["regime_trend_up"]   = ((out["adx"] >= p["adx_trend_min"]) &
        out["trend_long"] & (out["plus_di"] > out["minus_di"]))
    out["regime_trend_down"] = ((out["adx"] >= p["adx_trend_min"]) &
        out["trend_short"] & (out["minus_di"] > out["plus_di"]))
    out["regime_range"]  = ((out["adx"] <= p["adx_range_max"]) &
        ~out["high_vol_regime"])
    out["regime_choppy"] = (~out["regime_trend_up"] & ~out["regime_trend_down"] &
        (out["avg_body_ratio"] <= p["chop_body_max"]) & ~out["high_vol_regime"])
    conditions = [out["regime_trend_up"],out["regime_trend_down"],
                  out["regime_choppy"],out["high_vol_regime"],
                  out["low_vol_regime"],out["regime_range"]]
    choices    = ["Trend Up","Trend Down","Choppy","High Vol","Low Vol","Range"]
    out["regime_name"] = np.select(conditions, choices, default="Neutral")
    return out
