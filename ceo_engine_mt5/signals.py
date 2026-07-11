"""
The CEO Protocol — Layer 3: Signal Logic  v2.0
Improvements: #3 FVG fill tracking, #8 Multi-swing pool
"""

import pandas as pd
import numpy as np

MODEL_NAMES = {
    0:"LQ",1:"LQ + Trend",2:"LQ + FVG",3:"LQ + Volume",
    4:"LQ + RSI",5:"LQ + Displacement",6:"LQ + Trend + FVG",
    7:"LQ + Trend + Volume",8:"LQ + Trend + RSI",
    9:"LQ + Trend + Displacement",10:"LQ + FVG + Volume",
    11:"LQ + FVG + RSI",12:"LQ + FVG + Displacement",
    13:"LQ + Volume + RSI",14:"LQ + Trend + FVG + Volume",
    15:"LQ + All Filters",
}
MODEL_USES = {
    "trend":        {1,6,7,8,9,14,15},
    "fvg":          {2,6,10,11,12,14,15},
    "volume":       {3,7,10,13,14,15},
    "rsi":          {4,8,11,13,15},
    "displacement": {5,9,12,15},
}
NUM_MODELS = 16

def _col(m, side):
    return f"m{m:02d}_{side}"

def apply_htf_bias(df, htf_df, htf_ema_fast=50, htf_ema_slow=200):
    if htf_df is None:
        df["htf_bullish"] = True
        df["htf_bearish"] = True
        return df
    from .indicators import ema as calc_ema
    htf = htf_df.copy()
    htf["ema_fast"]    = calc_ema(htf["close"], htf_ema_fast)
    htf["ema_slow"]    = calc_ema(htf["close"], htf_ema_slow)
    htf["htf_bullish"] = htf["ema_fast"] > htf["ema_slow"]
    htf["htf_bearish"] = htf["ema_fast"] < htf["ema_slow"]
    merged = pd.merge_asof(
        df.reset_index(),
        htf[["htf_bullish","htf_bearish"]].reset_index(),
        on="datetime", direction="backward",
    ).set_index("datetime")
    # Guard: merge_asof may not produce the column if htf is too small
    df["htf_bullish"] = merged["htf_bullish"].fillna(False).values \
        if "htf_bullish" in merged.columns else False
    df["htf_bearish"] = merged["htf_bearish"].fillna(False).values \
        if "htf_bearish" in merged.columns else False
    return df

def _track_fvg_fills(df, recent_bars: int = 6):
    """
    Improvement #3: tracks FVG fill status.
    A bull FVG (gap_bottom=high[i-2], gap_top=low[i]) is unfilled while
    price has not traded back into the gap zone AND it was formed within
    recent_bars bars.  A gap is considered filled when either:
      - low[i] <= gap_bottom  (price blew straight through — filled or broken)
      - low[i] <= gap_top     (price entered the gap from above)
    Symmetric logic applies for bear FVGs.

    Parameters
    ----------
    df          : DataFrame with bull_fvg / bear_fvg columns
    recent_bars : max bars a gap stays active before expiring (default 6).
                  Prefer passing this explicitly rather than relying on
                  df.attrs, since attrs are not preserved through most
                  pandas operations (merge, groupby, copy with certain flags).
    """
    # df.attrs is used as a fallback for callers that don't pass recent_bars
    # explicitly. The explicit parameter always takes precedence over attrs.
    if "fvg_recent_bars" in df.attrs and recent_bars == 6:
        # Only read attrs when the caller left the default in place
        recent_bars = int(df.attrs["fvg_recent_bars"])
    n            = len(df)
    high         = df["high"].values
    low          = df["low"].values
    bull_fvg_raw = df["bull_fvg"].values
    bear_fvg_raw = df["bear_fvg"].values
    bull_unfilled = np.zeros(n, dtype=bool)
    bear_unfilled = np.zeros(n, dtype=bool)
    active_bull   = []   # (formed_bar, gap_bottom, gap_top)
    active_bear   = []
    for i in range(n):
        if bull_fvg_raw[i] and i >= 2:
            active_bull.append((i, high[i-2], low[i]))
        if bear_fvg_raw[i] and i >= 2:
            active_bear.append((i, high[i], low[i-2]))
        # Bull gap survives while: within age limit AND price has not entered
        # the gap zone. A gap is entered when low[i] <= gap_top (price is at
        # or below the top of the gap). We keep the gap while low > gap_bottom
        # (not blown through) AND low >= gap_top (not entered from above).
        # Note: low[i] == gap_top means price is touching the top edge but has
        # not yet traded into the gap — treated as unfilled, not yet filled.
        active_bull = [
            (b, gb, gt) for b, gb, gt in active_bull
            if (i - b) <= recent_bars and low[i] > gb and low[i] >= gt
        ]
        # Bear gap survives while: within age limit AND price has not entered
        # the gap zone. A gap is entered when high[i] >= gap_bottom (price is
        # at or above the bottom of the gap). We keep the gap while high < gap_top
        # (not blown through) AND high <= gap_bottom (not entered from below).
        active_bear = [
            (b, gb, gt) for b, gb, gt in active_bear
            if (i - b) <= recent_bars and high[i] < gt and high[i] <= gb
        ]
        bull_unfilled[i] = len(active_bull) > 0
        bear_unfilled[i] = len(active_bear) > 0
    df = df.copy()
    df["bull_fvg_recent_unfilled"] = bull_unfilled
    df["bear_fvg_recent_unfilled"] = bear_unfilled
    return df

def _update_swing_pool(pool: list, new_val: float, pool_size: int) -> None:
    """Appends a freshly-confirmed pivot to the pool, capping at pool_size (mutates in place)."""
    if not np.isnan(new_val):
        pool.append(new_val)
        if len(pool) > pool_size:
            pool.pop(0)


def _check_sweep_long(pool_lows, low_i, close_i, atr_i, max_depth_atr, min_rejection, lower_rej_i):
    """Checks the low-side pivot pool for a sweep-and-reject (returns the swept level or NaN)."""
    for level in reversed(pool_lows):
        sell_depth = (level - low_i) / atr_i
        if (low_i < level and close_i > level and
            sell_depth <= max_depth_atr and
            lower_rej_i >= min_rejection):
            return level
    return np.nan


def _check_sweep_short(pool_highs, high_i, close_i, atr_i, max_depth_atr, min_rejection, upper_rej_i):
    """Checks the high-side pivot pool for a sweep-and-reject (returns the swept level or NaN)."""
    for level in reversed(pool_highs):
        buy_depth = (high_i - level) / atr_i
        if (high_i > level and close_i < level and
            buy_depth <= max_depth_atr and
            upper_rej_i >= min_rejection):
            return level
    return np.nan


def detect_sweeps(df, params=None):
    """
    Improvement #8: multi-swing pool sweep detection.
    Maintains last N swing levels, checks sweep against all of them.
    """
    p             = params or {}
    max_depth_atr = p.get("max_sweep_depth_atr", 0.80)
    min_rejection = p.get("min_rejection_ratio",  0.20)
    pool_size     = p.get("swing_pool_size",       3)
    high      = df["high"].values
    low       = df["low"].values
    close     = df["close"].values
    atr_vals  = df["atr"].values
    ph        = df["pivot_high"].values
    pl        = df["pivot_low"].values
    upper_rej = df["upper_rejection"].values
    lower_rej = df["lower_rejection"].values
    n         = len(df)
    last_swing_high  = np.full(n, np.nan)
    last_swing_low   = np.full(n, np.nan)
    swept_level_high = np.full(n, np.nan)
    swept_level_low  = np.full(n, np.nan)
    base_long        = np.zeros(n, dtype=bool)
    base_short       = np.zeros(n, dtype=bool)
    pool_highs = []
    pool_lows  = []
    for i in range(n):
        _update_swing_pool(pool_highs, ph[i], pool_size)
        _update_swing_pool(pool_lows,  pl[i], pool_size)
        last_swing_high[i] = pool_highs[-1] if pool_highs else np.nan
        last_swing_low[i]  = pool_lows[-1]  if pool_lows  else np.nan
        atr_i = atr_vals[i]
        if np.isnan(atr_i) or atr_i <= 0:
            continue
        swept_low = _check_sweep_long(pool_lows, low[i], close[i], atr_i,
                                       max_depth_atr, min_rejection, lower_rej[i])
        if not np.isnan(swept_low):
            base_long[i]        = True
            swept_level_low[i]  = swept_low
        swept_high = _check_sweep_short(pool_highs, high[i], close[i], atr_i,
                                         max_depth_atr, min_rejection, upper_rej[i])
        if not np.isnan(swept_high):
            base_short[i]        = True
            swept_level_high[i]  = swept_high
    df = df.copy()
    df["last_swing_high"]  = last_swing_high
    df["last_swing_low"]   = last_swing_low
    df["swept_level_high"] = swept_level_high
    df["swept_level_low"]  = swept_level_low
    df["base_long"]        = base_long
    df["base_short"]       = base_short
    return df

def build_signals(df, use_filled_fvg=True):
    df  = df.copy()
    bl  = df["base_long"].values
    bs  = df["base_short"].values
    tl  = df["trend_long"].values
    ts  = df["trend_short"].values
    vs  = df["vol_spike"].values
    rl  = df["rsi_long"].values
    rs  = df["rsi_short"].values
    bdl = df["bull_displacement"].values
    bds = df["bear_displacement"].values
    if use_filled_fvg and "bull_fvg_recent_unfilled" in df.columns:
        bfr = df["bull_fvg_recent_unfilled"].values
        nfr = df["bear_fvg_recent_unfilled"].values
    else:
        bfr = df["bull_fvg_recent"].values
        nfr = df["bear_fvg_recent"].values
    longs  = {0:bl,1:bl&tl,2:bl&bfr,3:bl&vs,4:bl&rl,5:bl&bdl,
              6:bl&tl&bfr,7:bl&tl&vs,8:bl&tl&rl,9:bl&tl&bdl,
              10:bl&bfr&vs,11:bl&bfr&rl,12:bl&bfr&bdl,13:bl&vs&rl,
              14:bl&tl&bfr&vs,15:bl&tl&bfr&vs&rl&bdl}
    shorts = {0:bs,1:bs&ts,2:bs&nfr,3:bs&vs,4:bs&rs,5:bs&bds,
              6:bs&ts&nfr,7:bs&ts&vs,8:bs&ts&rs,9:bs&ts&bds,
              10:bs&nfr&vs,11:bs&nfr&rs,12:bs&nfr&bds,13:bs&vs&rs,
              14:bs&ts&nfr&vs,15:bs&ts&nfr&vs&rs&bds}
    for m in range(NUM_MODELS):
        df[_col(m,"long")]  = longs[m]
        df[_col(m,"short")] = shorts[m]
    return df

def quality_scores(df):
    df  = df.copy()
    tl  = df["trend_long"].values.astype(float)
    ts  = df["trend_short"].values.astype(float)
    vs  = df["vol_spike"].values.astype(float)
    rl  = df["rsi_long"].values.astype(float)
    rs  = df["rsi_short"].values.astype(float)
    bdl = df["bull_displacement"].values.astype(float)
    bds = df["bear_displacement"].values.astype(float)
    if "bull_fvg_recent_unfilled" in df.columns:
        bfr = df["bull_fvg_recent_unfilled"].values.astype(float)
        nfr = df["bear_fvg_recent_unfilled"].values.astype(float)
    else:
        bfr = df["bull_fvg_recent"].values.astype(float)
        nfr = df["bear_fvg_recent"].values.astype(float)
    rup   = df["regime_trend_up"].values
    rdown = df["regime_trend_down"].values
    rng   = df["regime_range"].values
    chop  = df["regime_choppy"].values
    hvol  = df["high_vol_regime"].values
    rlb = np.where(rup,15.,np.where(rdown,-15.,np.where(rng,5.,np.where(chop,-10.,np.where(hvol,5.,0.)))))
    rsb = np.where(rdown,15.,np.where(rup,-15.,np.where(rng,5.,np.where(chop,-10.,np.where(hvol,5.,0.)))))
    for m in range(NUM_MODELS):
        sl = np.full(len(df), 25.0)
        ss = np.full(len(df), 25.0)
        if m in MODEL_USES["trend"]:
            sl += np.where(tl,15.,-10.); ss += np.where(ts,15.,-10.)
        if m in MODEL_USES["fvg"]:
            sl += np.where(bfr,15.,-10.); ss += np.where(nfr,15.,-10.)
        if m in MODEL_USES["volume"]:
            sl += np.where(vs,10.,-6.); ss += np.where(vs,10.,-6.)
        if m in MODEL_USES["rsi"]:
            sl += np.where(rl,10.,-5.); ss += np.where(rs,10.,-5.)
        if m in MODEL_USES["displacement"]:
            sl += np.where(bdl,15.,-10.); ss += np.where(bds,15.,-10.)
        sl += rlb; ss += rsb
        df[f"m{m:02d}_quality_long"]  = np.clip(sl, 0., 100.)
        df[f"m{m:02d}_quality_short"] = np.clip(ss, 0., 100.)
    df["quality_long"]  = df["m00_quality_long"]
    df["quality_short"] = df["m00_quality_short"]
    return df

def regime_alignment(df):
    df    = df.copy()
    rup   = df["regime_trend_up"]
    rdown = df["regime_trend_down"]
    chop  = df["regime_choppy"]
    df["alignment_long"]  = np.where(rup,"Aligned",np.where(rdown,"Against",np.where(chop,"Avoid","Neutral")))
    df["alignment_short"] = np.where(rdown,"Aligned",np.where(rup,"Against",np.where(chop,"Avoid","Neutral")))
    return df

def confluence_signals(df, min_count=3, min_quality=45.0, require_align=True, mode="sweep"):
    """
    Multi-model liquidity-sweep confluence signal.

    mode controls what "confluence" actually requires:
        "sweep"          >= min_count of the 16 filter-models agree
                          simultaneously (default -- this is the
                          original behavior, unchanged).
        "ceo_structure"  the full CEO sequence validates
                          (ceo_long_valid/ceo_short_valid) instead of a
                          model count.
        "full"           both at once: >= min_count models agree AND
                          the full CEO sequence validates.

    Every mode still applies the same quality/alignment/HTF-bias gates
    on top -- the mode only changes what counts as the base "agreement"
    condition.

    "ceo_structure" and "full" require ceo_long_valid/ceo_short_valid to
    already be columns on df, i.e. ceo_structure.build_ceo_structure()
    must have run first (true at every real pipeline site except
    walkforward.py's per-window rebuild -- see signals.build_all()'s
    docstring). Raises a clear error rather than silently treating the
    gate as always-False when those columns are missing: a silently dead
    signal is exactly the bug this mode system exists to not repeat.
    """
    valid_modes = ("sweep", "ceo_structure", "full")
    if mode not in valid_modes:
        raise ValueError(f"confluence mode must be one of {valid_modes}, got {mode!r}")
    if mode in ("ceo_structure", "full"):
        missing = [c for c in ("ceo_long_valid", "ceo_short_valid") if c not in df.columns]
        if missing:
            raise ValueError(
                f"confluence_mode={mode!r} requires {missing}, which come from "
                f"ceo_structure.build_ceo_structure() -- call that before "
                f"confluence_signals()/build_confluence(), or use mode='sweep'."
            )

    df    = df.copy()
    lc    = np.zeros(len(df), dtype=int)
    sc    = np.zeros(len(df), dtype=int)
    for m in range(NUM_MODELS):
        lc += df[_col(m,"long")].values.astype(int)
        sc += df[_col(m,"short")].values.astype(int)
    df["confluence_long_count"]  = lc
    df["confluence_short_count"] = sc

    sweep_gate_l = lc >= min_count
    sweep_gate_s = sc >= min_count
    if mode == "sweep":
        gate_l, gate_s = sweep_gate_l, sweep_gate_s
    elif mode == "ceo_structure":
        gate_l = df["ceo_long_valid"].values.astype(bool)
        gate_s = df["ceo_short_valid"].values.astype(bool)
    else:   # "full"
        gate_l = sweep_gate_l & df["ceo_long_valid"].values.astype(bool)
        gate_s = sweep_gate_s & df["ceo_short_valid"].values.astype(bool)

    qgl = df["m00_quality_long"]  >= min_quality
    qgs = df["m00_quality_short"] >= min_quality
    if require_align:
        al = ~df["alignment_long"].isin(["Against","Avoid"])
        as_ = ~df["alignment_short"].isin(["Against","Avoid"])
    else:
        al = pd.Series(True, index=df.index)
        as_ = pd.Series(True, index=df.index)
    htf_l = df["htf_bullish"] if "htf_bullish" in df.columns else pd.Series(True, index=df.index)
    htf_s = df["htf_bearish"] if "htf_bearish" in df.columns else pd.Series(True, index=df.index)
    df["confluence_long_fired"]  = gate_l & qgl & al  & htf_l
    df["confluence_short_fired"] = gate_s & qgs & as_ & htf_s
    return df

def build_all(df, htf_df=None, params=None):
    """
    Runs sweep detection -> per-model signals -> quality scores -> regime
    alignment -> HTF bias.

    Deliberately does NOT call confluence_signals() anymore. Confluence's
    quality gate reads m00_quality_long/short (the bare LQ model, no
    filters -> base 25 +/- a 15-point regime swing, so it's capped at
    40). ceo_structure.py's validate_ceo_sequence() adds a further 0-30
    structural bonus to that same column -- but only later in the
    pipeline. Calling confluence_signals() here, before that bonus
    exists, meant the quality gate (45 by default here, 50-60 in every
    real run.py/mt5_live.py/launcher.py config) could never be satisfied,
    so the Confluence model could never fire -- confirmed via backtest
    (0 trades, always) before this was fixed. Call confluence_signals()
    explicitly after build_ceo_structure() in your pipeline instead --
    see check_symbol() in mt5_live_signals.py or run.py for the pattern.
    (walkforward.py's per-window rebuild is a known exception: it doesn't
    run build_ceo_structure() at all, so its Confluence evaluation still
    can't clear the quality gate. Flagged, not silently worked around --
    fixing that means adding the structure/pattern stages to walk-forward
    too, which is a separate, real performance tradeoff to weigh since
    it'd run per rolling window.)
    """
    p  = params or {}
    df = df.copy()
    fvg_recent_bars = p.get("fvg_recent_bars", 6)
    df.attrs["fvg_recent_bars"] = fvg_recent_bars   # keep for introspection
    sweep_p = {
        "max_sweep_depth_atr": p.get("max_sweep_depth_atr", 0.80),
        "min_rejection_ratio": p.get("min_rejection_ratio",  0.20),
        "swing_pool_size":     p.get("swing_pool_size",       3),
    }
    df = detect_sweeps(df, sweep_p)
    df = _track_fvg_fills(df, recent_bars=fvg_recent_bars)
    df = build_signals(df, use_filled_fvg=True)
    df = quality_scores(df)
    df = regime_alignment(df)
    df = apply_htf_bias(df, htf_df,
                        htf_ema_fast=p.get("htf_ema_fast", 50),
                        htf_ema_slow=p.get("htf_ema_slow", 200))
    return df


def build_confluence(df, params=None):
    """
    Call this after build_ceo_structure() -- see build_all()'s docstring
    for why the ordering matters. Thin wrapper around confluence_signals()
    that reads the same params dict key names build_all() used to consume
    internally (confluence_min_count/min_quality_score/htf_require_align),
    plus confluence_mode ("sweep" | "ceo_structure" | "full"), so existing
    signal_params dicts don't need to change shape to keep working.
    """
    p = params or {}
    return confluence_signals(df,
                              min_count    = p.get("confluence_min_count", 3),
                              min_quality  = p.get("min_quality_score", 45.0),
                              require_align= p.get("htf_require_align", True),
                              mode         = p.get("confluence_mode", "sweep"))
