"""
The CEO Protocol MT5 — Live Monitor Utilities
==================================================
Pure, low-level helpers shared by mt5_live_signals.py, mt5_live_session.py,
and mt5_live.py: lot-size math, dedup IDs, signal text formatting, sound
alerts, CSV logging, MT5-rates-to-DataFrame conversion, and the
structural SL/TP calculators.

Split out of mt5_live.py (which had grown to 1580 lines) purely for
file-size/maintainability reasons — no behavior changed. See CHANGELOG.
This file has no dependency on mt5_live_signals.py or mt5_live_session.py,
so it's always safe to import from either of them.
"""

import os, time
from datetime import datetime, timezone
from typing import Optional, Set

import pandas as pd
import numpy as np

from ceo_engine_mt5.ceo_logging import get_logger
logger = get_logger(__name__)

from .mt5_connect import MT5Connection


# ─────────────────────────────────────────────────────────────────────────────
# HTF auto-selection
# ─────────────────────────────────────────────────────────────────────────────

_HTF_AUTO = {
    "m1":"m15","m2":"m15","m3":"m15","m5":"m15",
    "m15":"h1","m30":"h1","h1":"h4","h2":"h4",
    "h4":"d1","h6":"d1","h8":"d1","h12":"d1","d1":"w1",
    "1m":"15m","5m":"15m","15m":"1h","30m":"1h",
    "1h":"4h","4h":"1d","1d":"1w",
}


def _auto_htf(tf):
    return _HTF_AUTO.get(tf.lower(), "d1")


# ─────────────────────────────────────────────────────────────────────────────
# Legacy lot-size helper (kept for backward compat — RiskEngine is preferred)
# ─────────────────────────────────────────────────────────────────────────────

def calc_lot_size(account_balance, risk_pct, sl_distance_price,
                  tick_value, tick_size,
                  volume_min=0.01, volume_max=100.0, volume_step=0.01):
    if sl_distance_price <= 0 or tick_value <= 0 or tick_size <= 0:
        return {"lots": volume_min, "error": "invalid SL or tick data"}
    risk_amount  = account_balance * (risk_pct / 100.0)
    ticks_in_sl  = sl_distance_price / tick_size
    risk_per_lot = ticks_in_sl * tick_value
    if risk_per_lot <= 0:
        return {"lots": volume_min, "error": "risk_per_lot <= 0"}
    raw_lots = risk_amount / risk_per_lot
    if volume_step > 0:
        lots = round(round(raw_lots / volume_step) * volume_step, 8)
    else:
        lots = raw_lots
    lots = max(volume_min, min(volume_max, lots))
    return {
        "lots":         round(lots, 2),
        "risk_amount":  round(risk_amount, 2),
        "risk_per_lot": round(risk_per_lot, 4),
        "sl_ticks":     round(ticks_in_sl, 1),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Deduplication
# ─────────────────────────────────────────────────────────────────────────────

def _load_seen_ids(log_path: Optional[str]) -> Set[str]:
    seen = set()
    if not log_path or not os.path.exists(log_path):
        return seen
    try:
        df = pd.read_csv(log_path)
        if {"symbol","timeframe","datetime"}.issubset(df.columns):
            for _, row in df.iterrows():
                seen.add(f"{row['symbol']}_{row['timeframe']}_{row['datetime']}")
        logger.info("Loaded %s seen signal IDs from %s", len(seen), log_path)
    except Exception as e:
        logger.warning("Could not read log %s: %s", log_path, e)
    return seen

def _bar_id(symbol, tf, bar_time, direction):
    return f"{symbol}_{tf.upper()}_{bar_time.strftime('%Y-%m-%d %H:%M:%S')}_{direction}"


# ─────────────────────────────────────────────────────────────────────────────
# Signal formatter
# ─────────────────────────────────────────────────────────────────────────────

def _pips(diff, digits):
    pips = diff * (10 ** (digits - 1)) if digits >= 4 else diff
    return f"{pips:+.1f} pips"

def _format_signal(symbol, tf, direction, bar_time, close, atr,
                   model_name, quality, regime, alignment, htf_label,
                   conf_count, sl, tp1, tp2, tp3, digits,
                   lot_info=None, ceo_valid=False, bos=False,
                   in_discount=False, pat_name="", session=""):
    arrow = "▲" if direction == "LONG" else "▼"
    fmt   = f".{digits}f"

    # CEO structure flags
    flags = []
    if ceo_valid:   flags.append("✅ CEO sequence")
    if bos:         flags.append("✅ BOS confirmed")
    if in_discount: flags.append("✅ Discount zone")
    if pat_name:    flags.append(f"📐 {pat_name}")
    flags_str = "  " + "  ".join(flags) if flags else "  ⚡ Base sweep"

    lines = [
        f"\n{'═'*60}",
        f"[{bar_time.strftime('%Y-%m-%d %H:%M:%S')} UTC]  "
        f"{symbol} {tf.upper()}  {arrow} {direction}",
        f"{'═'*60}",
        f"  Model      : {model_name}",
        f"  Quality    : {quality:.1f} / 100",
        f"  Session    : {session.title() or '—'}",
        f"  Regime     : {regime}",
        f"  Alignment  : {alignment}",
        f"  HTF Bias   : {htf_label}",
        f"  Confluence : {conf_count} / 16 models",
        f"  ─────────────────────────────────────────────",
        flags_str,
        f"  ─────────────────────────────────────────────",
        f"  Entry ref  : {close:{fmt}}",
        f"  SL    ref  : {sl:{fmt}}  ({_pips(sl - close, digits)})",
        f"  TP1   ref  : {tp1:{fmt}}  ({_pips(tp1 - close, digits)})",
        f"  TP2   ref  : {tp2:{fmt}}  ({_pips(tp2 - close, digits)})",
        f"  TP3   ref  : {tp3:{fmt}}  ({_pips(tp3 - close, digits)})",
    ]
    if lot_info and "lots" in lot_info and "error" not in lot_info:
        lines += [
            f"  ─────────────────────────────────────────────",
            f"  Lot size   : {lot_info['lots']} lots",
            f"  Risk $     : ${lot_info['risk_amount']:,.2f}",
        ]
    lines.append(f"{'═'*60}")
    return "\n".join(lines)


def _play_alert(direction):
    try:
        import winsound
        freq = 1200 if direction == "LONG" else 800
        winsound.Beep(freq, 300)
        time.sleep(0.15)
        winsound.Beep(freq, 300)
    except Exception as e:
        logger.debug("Sound alert failed: %s", e)

def _log_signal(log_path, row):
    write_header = not os.path.exists(log_path)
    pd.DataFrame([row]).to_csv(log_path, mode="a",
                               header=write_header, index=False)

def _rates_to_df(rates, symbol, tick_sz):
    df = pd.DataFrame(rates)
    df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
    df = df.rename(columns={"time":"datetime","tick_volume":"volume"})
    df = df.set_index("datetime")
    df = df[["open","high","low","close","volume"]].copy()
    df.attrs["symbol"]    = symbol
    df.attrs["tick_size"] = tick_sz
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Structural SL from CEO structure
# ─────────────────────────────────────────────────────────────────────────────

def _structural_sl(last: pd.Series, direction: str, atr: float,
                   sl_atr_mult: float = 1.5, sl_buffer: float = 0.10) -> float:
    """
    Use the swept pivot low/high as SL (CEO method).
    Falls back to ATR-based SL if structural level not available.
    """
    if direction == "LONG":
        struct_sl = last.get("last_swing_low", np.nan)
        if not np.isnan(struct_sl) and struct_sl > 0:
            return struct_sl - atr * sl_buffer   # buffer below the pivot
    else:
        struct_sl = last.get("last_swing_high", np.nan)
        if not np.isnan(struct_sl) and struct_sl > 0:
            return struct_sl + atr * sl_buffer

    # Fallback: ATR-based
    sign = 1 if direction == "LONG" else -1
    return float(last["close"]) - sign * atr * sl_atr_mult


# ─────────────────────────────────────────────────────────────────────────────
# TP targeting from unmitigated levels or fixed R
# ─────────────────────────────────────────────────────────────────────────────

def _tp_levels(last: pd.Series, direction: str, close: float,
               sl: float, tp1_r: float, tp2_r: float, tp3_r: float
               ) -> tuple:
    """
    Use nearest unmitigated structural level as TP3 if available.
    TP1 and TP2 remain at fixed R multiples for partial-close management.
    """
    sign    = 1 if direction == "LONG" else -1
    sl_dist = abs(close - sl)

    tp1 = close + sign * sl_dist * tp1_r
    tp2 = close + sign * sl_dist * tp2_r

    # Try structural TP3
    if direction == "LONG":
        struct_tp = last.get("nearest_unmit_res", np.nan)
    else:
        struct_tp = last.get("nearest_unmit_sup", np.nan)

    if not np.isnan(struct_tp) and abs(struct_tp - close) > sl_dist * tp2_r:
        tp3 = float(struct_tp)
    else:
        tp3 = close + sign * sl_dist * tp3_r

    return tp1, tp2, tp3


def _try_connect(login, password, server, retry_interval=30):
    while True:
        try:
            conn = MT5Connection(login=login, password=password, server=server)
            conn.connect()
            return conn
        except (ConnectionError, RuntimeError) as e:
            logger.warning("MT5 connection failed: %s", e)
            logger.info("Retrying in %s seconds...", retry_interval)
            time.sleep(retry_interval)

def _seconds_to_bar_close(tf):
    secs = {"m1":60,"1m":60,"m5":300,"5m":300,"m15":900,"15m":900,
            "m30":1800,"30m":1800,"h1":3600,"1h":3600,
            "h4":14400,"4h":14400,"d1":86400,"1d":86400}
    period = secs.get(tf.lower(), 3600)
    now    = int(datetime.now(timezone.utc).timestamp())
    return period - (now % period)

