"""
The CEO Protocol — Layer 4: Backtest Runner  v2.0
Improvements: #1 Sweep SL, #2 Session filter, #4 Spread cost
Bug fix: gap-open SL protection (SL never on wrong side of entry)
"""

import pandas as pd
import numpy as np
from .signals import MODEL_NAMES, NUM_MODELS, _col
from .session_filter import SESSION_WINDOWS, TRADE_ANYTIME

# Convert session_filter.py's canonical dtime windows into the (start_hour,
# end_hour) int tuples build_session_mask() works with. This is the ONLY
# place backtest.py defines session hours — risk_engine.py imports the same
# SESSION_WINDOWS, so backtest and live can never disagree on session times again.
_SESSIONS_HOURS = {
    name: (start.hour, end.hour if end.minute == 0 else end.hour + 1)
    for name, (start, end) in SESSION_WINDOWS.items()
}

DEFAULT_BT_PARAMS = {
    "sl_mode":           "sweep",
    "sl_atr_mult":       1.50,
    "sl_buffer_atr":     0.10,
    "sl_max_atr_mult":   3.00,
    "tp1_r":             1.00,
    "tp2_r":             2.00,
    "tp3_r":             3.00,
    "tp1_weight":        0.333,
    "tp2_weight":        0.333,
    "tp3_weight":        0.334,
    "max_bars":          30,
    "conservative_both": True,
    "spread_mode":       "fixed_r",
    "commission_r":      0.05,
    "spread_points":     0.0,
    "point_size":        0.00001,
    "session_filter":    False,
    "sessions": _SESSIONS_HOURS,
    "active_sessions": [TRADE_ANYTIME],  # "all" = no restriction, trade anytime
}

def build_session_mask(df, p):
    if not p.get("session_filter", False):
        return np.ones(len(df), dtype=bool)
    sessions        = p.get("sessions",        DEFAULT_BT_PARAMS["sessions"])
    active_sessions = p.get("active_sessions", DEFAULT_BT_PARAMS["active_sessions"])
    if TRADE_ANYTIME in active_sessions:
        return np.ones(len(df), dtype=bool)
    idx   = df.index
    if idx.tz is None: idx = idx.tz_localize("UTC")
    else:              idx = idx.tz_convert("UTC")
    hours = idx.hour
    mask  = np.zeros(len(df), dtype=bool)
    for name in active_sessions:
        if name not in sessions: continue
        start, end = sessions[name]
        if start <= end: mask |= (hours >= start) & (hours < end)
        else:            mask |= (hours >= start) | (hours < end)
    return mask

def _calc_sl(direction, entry_price, sig_bar_low, sig_bar_high, atr_val, p):
    """
    Sweep-based SL placement with gap-open protection.
    Falls back to ATR if:
      (a) sweep depth exceeds sl_max_atr_mult
      (b) gap-open puts entry on wrong side of sweep SL
    """
    sl_mode    = p.get("sl_mode",         "sweep")
    buffer_atr = p.get("sl_buffer_atr",   0.10)
    max_mult   = p.get("sl_max_atr_mult", 3.00)
    atr_mult   = p.get("sl_atr_mult",     1.50)
    buffer     = atr_val * buffer_atr

    if sl_mode == "sweep":
        sl   = (sig_bar_low  - buffer) if direction == 1 else (sig_bar_high + buffer)
        risk = abs(entry_price - sl)
        # Cap (a): sweep too deep
        if risk > atr_val * max_mult:
            risk = atr_val * atr_mult
            sl   = (entry_price - risk) if direction == 1 else (entry_price + risk)
        # Cap (b): gap-open put entry on wrong side — BUG FIX
        wrong_side = (direction == 1 and sl >= entry_price) or \
                     (direction == -1 and sl <= entry_price)
        if wrong_side:
            risk = atr_val * atr_mult
            sl   = (entry_price - risk) if direction == 1 else (entry_price + risk)
    else:
        risk = max(atr_val * atr_mult, 1e-10)
        sl   = (entry_price - risk) if direction == 1 else (entry_price + risk)

    return sl, max(risk, 1e-10)

def _spread_cost_r(risk, p):
    if p.get("spread_mode","fixed_r") == "price":
        sp = p.get("spread_points", 0.0)
        pt = p.get("point_size",    0.00001)
        return (sp * pt) / risk if risk > 0 else 0.0
    return p.get("commission_r", 0.05)

def _check_exit_hits(direction, h, lo, sl, tp1, tp2, tp3):
    """Returns (hit_tp1, hit_tp2, hit_tp3, hit_sl) for one bar's high/low."""
    hit_tp1 = (direction == 1 and h >= tp1) or (direction == -1 and lo <= tp1)
    hit_tp2 = (direction == 1 and h >= tp2) or (direction == -1 and lo <= tp2)
    hit_tp3 = (direction == 1 and h >= tp3) or (direction == -1 and lo <= tp3)
    hit_sl  = (direction == 1 and lo <= sl) or (direction == -1 and h >= sl)
    return hit_tp1, hit_tp2, hit_tp3, hit_sl


def _compute_r_result(hit_tp3, hit_sl, conservative, tp1_hit, tp2_hit,
                       tp1_r, tp2_r, tp3_r, tp1_w, tp2_w, tp3_w,
                       close_price, entry_price, direction, risk):
    """
    R-multiple outcome for one closed trade, in the same branch order as
    the original inline logic:
      1. TP3 and SL both hit the same bar, conservative mode -> full loss
      2. TP3 hit -> full weighted R across all three targets
      3. SL hit -> partial R from any TPs already banked, minus the
         unrealized portion of size still open at SL
      4. Time-expired with neither TP3 nor SL hit -> partial R + open R
         (mark-to-close) on the remaining size
    """
    if hit_tp3 and hit_sl and conservative:
        return -1.0
    if hit_tp3:
        return tp1_r*tp1_w + tp2_r*tp2_w + tp3_r*tp3_w
    if hit_sl:
        r_result = 0.0
        if tp1_hit: r_result += tp1_r*tp1_w
        if tp2_hit: r_result += tp2_r*tp2_w
        r_result -= (1.0 - (tp1_w if tp1_hit else 0.)
                         - (tp2_w if tp2_hit else 0.))
        return r_result
    pnl    = (close_price - entry_price) * direction
    open_r = pnl / risk if risk > 0 else 0.0
    partial = (tp1_r*tp1_w if tp1_hit else 0.) + \
              (tp2_r*tp2_w if tp2_hit else 0.)
    rem = 1.0-(tp1_w if tp1_hit else 0.)-(tp2_w if tp2_hit else 0.)
    return partial + open_r * rem


def _try_enter_trade(i, sig_bar, atrs, session_mask, long_sig, short_sig,
                      last_dir, opens, lows, highs, tp1_r, tp2_r, tp3_r, p):
    """
    Checks whether a new trade opens at bar i's open, based on the signal
    at sig_bar (i-1). Returns None for any disqualifying condition (bad
    ATR, session blocked, no signal, same-direction re-entry guard) —
    every None case is a `continue` in the original inline version, which
    is equivalent here since this is the last block in the loop body.
    """
    atr_val = atrs[sig_bar]
    if np.isnan(atr_val) or atr_val <= 0:
        return None
    if not session_mask[i]:
        return None
    go_long  = bool(long_sig[sig_bar])  and last_dir != 1
    go_short = bool(short_sig[sig_bar]) and last_dir != -1
    if not (go_long or go_short):
        return None

    direction   = 1 if go_long else -1
    entry_price = opens[i]
    sl, risk    = _calc_sl(direction, entry_price,
                           lows[sig_bar], highs[sig_bar],
                           atr_val, p)
    if direction == 1:
        tp1 = entry_price + risk*tp1_r
        tp2 = entry_price + risk*tp2_r
        tp3 = entry_price + risk*tp3_r
    else:
        tp1 = entry_price - risk*tp1_r
        tp2 = entry_price - risk*tp2_r
        tp3 = entry_price - risk*tp3_r

    return {"direction": direction, "entry_price": entry_price, "sl": sl,
            "risk": risk, "tp1": tp1, "tp2": tp2, "tp3": tp3, "entry_bar": i}


def _simulate_model(opens, highs, lows, closes, atrs,
                    long_sig, short_sig, session_mask, p, index):
    n            = len(opens)
    tp1_r        = p["tp1_r"];  tp2_r = p["tp2_r"];  tp3_r = p["tp3_r"]
    max_bars     = p["max_bars"]
    conservative = p["conservative_both"]
    tp1_w        = p["tp1_weight"]
    tp2_w        = p["tp2_weight"]
    tp3_w        = p["tp3_weight"]
    trades    = []
    in_trade  = False
    direction = entry_price = sl = tp1 = tp2 = tp3 = risk = 0.0
    entry_bar = 0
    tp1_hit   = tp2_hit = False
    last_dir  = 0

    for i in range(1, n):
        sig_bar = i - 1
        if in_trade:
            h = highs[i]; lo = lows[i]; c = closes[i]
            hit_tp1, hit_tp2, hit_tp3, hit_sl = _check_exit_hits(
                direction, h, lo, sl, tp1, tp2, tp3)
            expired = (i - entry_bar) >= max_bars
            if hit_tp1 and not tp1_hit: tp1_hit = True
            if hit_tp2 and not tp2_hit: tp2_hit = True
            if hit_tp3 or hit_sl or expired:
                r_result = _compute_r_result(
                    hit_tp3, hit_sl, conservative, tp1_hit, tp2_hit,
                    tp1_r, tp2_r, tp3_r, tp1_w, tp2_w, tp3_w,
                    c, entry_price, direction, risk)
                r_result -= _spread_cost_r(risk, p)
                trades.append({
                    "entry_bar": index[entry_bar],
                    "exit_bar":  index[i],
                    "direction": "LONG" if direction==1 else "SHORT",
                    "entry": round(entry_price,6), "sl": round(sl,6),
                    "tp1":   round(tp1,6),         "tp2": round(tp2,6),
                    "tp3":   round(tp3,6),
                    "risk":  round(risk,6),
                    "sl_mode": p.get("sl_mode","sweep"),
                    "bars_held": i - entry_bar,
                    "tp1_hit": tp1_hit, "tp2_hit": tp2_hit,
                    "tp3_hit": hit_tp3, "sl_hit": hit_sl,
                    "expired": expired and not hit_tp3 and not hit_sl,
                    "r_result": round(r_result, 4),
                    "win": r_result > 0,
                })
                last_dir = direction
                in_trade = False; direction = 0

        if not in_trade:
            entry = _try_enter_trade(
                i, sig_bar, atrs, session_mask, long_sig, short_sig,
                last_dir, opens, lows, highs, tp1_r, tp2_r, tp3_r, p)
            if entry is not None:
                direction     = entry["direction"]
                entry_price   = entry["entry_price"]
                sl            = entry["sl"]
                risk          = entry["risk"]
                tp1, tp2, tp3 = entry["tp1"], entry["tp2"], entry["tp3"]
                entry_bar     = entry["entry_bar"]
                in_trade      = True
                tp1_hit = tp2_hit = False

    return pd.DataFrame(trades)

def run_backtest(df, params=None):
    p            = {**DEFAULT_BT_PARAMS, **(params or {})}
    opens        = df["open"].values.astype(float)
    highs        = df["high"].values.astype(float)
    lows         = df["low"].values.astype(float)
    closes       = df["close"].values.astype(float)
    atrs         = df["atr"].values.astype(float)
    idx          = df.index
    session_mask = build_session_mask(df, p)
    sl_label     = p.get("sl_mode","sweep")
    sess_label   = (",".join(p.get("active_sessions",["all"]))
                   if p.get("session_filter") else "all")
    print(f"Running backtest — SL:{sl_label}  sessions:{sess_label}")
    print(f"  Session mask: {session_mask.sum():,}/{len(session_mask):,} bars\n")
    results = {}
    for m in range(NUM_MODELS):
        ls = df[_col(m,"long")].values.astype(bool)
        ss = df[_col(m,"short")].values.astype(bool)
        t  = _simulate_model(opens,highs,lows,closes,atrs,ls,ss,session_mask,p,idx)
        results[MODEL_NAMES[m]] = t
        line = f"  {MODEL_NAMES[m]:<35} → {len(t):>4} trades"
        if len(t):
            line += f"  |  WR {t['win'].mean()*100:>5.1f}%  |  Net R {t['r_result'].sum():>+7.2f}R"
        print(line)
    cl = df["confluence_long_fired"].values.astype(bool)
    cs = df["confluence_short_fired"].values.astype(bool)
    ct = _simulate_model(opens,highs,lows,closes,atrs,cl,cs,session_mask,p,idx)
    results["Confluence"] = ct
    line = f"  {'Confluence':<35} → {len(ct):>4} trades"
    if len(ct):
        line += f"  |  WR {ct['win'].mean()*100:>5.1f}%  |  Net R {ct['r_result'].sum():>+7.2f}R"
    print(line)
    return results

def results_table(bt):
    rows = []
    for name, trades in bt.items():
        if trades.empty:
            rows.append({"Model":name,"Trades":0,"Wins":0,"Losses":0,
                         "Win Rate":np.nan,"Profit Factor":np.nan,
                         "Avg R":np.nan,"Net R":np.nan,
                         "Max Win R":np.nan,"Max Loss R":np.nan,
                         "Avg Bars":np.nan,"SL Mode":"—"})
            continue
        wins   = trades[trades["r_result"] > 0]
        losses = trades[trades["r_result"] <= 0]
        gp  = wins["r_result"].sum()   if len(wins)   > 0 else 0.
        gl  = losses["r_result"].abs().sum() if len(losses) > 0 else 0.
        pf  = (gp/gl) if gl > 0 else (999. if gp > 0 else np.nan)
        slm = trades["sl_mode"].iloc[0] if "sl_mode" in trades.columns else "atr"
        rows.append({
            "Model": name, "Trades": len(trades),
            "Wins": len(wins), "Losses": len(losses),
            "Win Rate":      round(trades["win"].mean()*100, 1),
            "Profit Factor": round(pf, 2),
            "Avg R":         round(trades["r_result"].mean(), 3),
            "Net R":         round(trades["r_result"].sum(),  2),
            "Max Win R":     round(trades["r_result"].max(),  3),
            "Max Loss R":    round(trades["r_result"].min(),  3),
            "Avg Bars":      round(trades["bars_held"].mean(),1),
            "SL Mode":       slm,
        })
    return pd.DataFrame(rows).set_index("Model")

def equity_curve(trades, starting_r=0.0):
    if trades.empty: return pd.Series(dtype=float)
    return trades.set_index("entry_bar")["r_result"].cumsum() + starting_r

def compare_sl_modes(df, params=None):
    p = {**DEFAULT_BT_PARAMS, **(params or {})}
    print("─── SL Mode: sweep ───")
    bt_s = run_backtest(df, {**p, "sl_mode":"sweep"})
    tbl_s = results_table(bt_s)
    tbl_s.columns = [f"sweep_{c}" for c in tbl_s.columns]
    print("\n─── SL Mode: atr ───")
    bt_a = run_backtest(df, {**p, "sl_mode":"atr"})
    tbl_a = results_table(bt_a)
    tbl_a.columns = [f"atr_{c}" for c in tbl_a.columns]
    combined = pd.concat([tbl_s, tbl_a], axis=1)
    combined["net_r_diff"] = combined["sweep_Net R"] - combined["atr_Net R"]
    return combined
