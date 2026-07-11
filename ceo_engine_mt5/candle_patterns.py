"""
The CEO Protocol — Layer 2.5: Candle Pattern Detection
===========================================================
Detects all 19 CEO Method candlestick patterns plus key confirmation
patterns used for POI reaction timing.

All detections are fully vectorised (numpy/pandas) — no loops.
Inputs are the enriched DataFrame from indicators.calc_all().

Required columns (already produced by indicators.py)
-----------------------------------------------------
    open, high, low, close          OHLCV
    body_ratio                      abs(close-open) / candle_range
    upper_rejection                 upper_wick / candle_range
    lower_rejection                 lower_wick / candle_range
    body_size                       abs(close-open)
    candle_range                    high - low
    atr                             ATR(14)
    bull_displacement               from indicators
    bear_displacement               from indicators

Output columns (one bool Series per pattern, True on the signal bar)
---------------------------------------------------------------------

── Single-candle ────────────────────────────────────────────────────
    cp_hammer                   Hammer
    cp_hanging_man              Hanging Man
    cp_shooting_star            Shooting Star
    cp_inverted_hammer          Inverted Hammer
    cp_bull_marubozu            Bullish Marubozu
    cp_bear_marubozu            Bearish Marubozu
    cp_doji                     Doji
    cp_dragonfly_doji           Dragonfly Doji
    cp_gravestone_doji          Gravestone Doji
    cp_long_legged_doji         Long-Legged Doji
    cp_spinning_top             Spinning Top

── Two-candle ───────────────────────────────────────────────────────
    cp_bull_engulfing           Bullish Engulfing
    cp_bear_engulfing           Bearish Engulfing
    cp_piercing_line            Piercing Line
    cp_dark_cloud_cover         Dark Cloud Cover
    cp_bull_harami              Bullish Harami
    cp_bear_harami              Bearish Harami
    cp_tweezers_top             Tweezers Top
    cp_tweezers_bottom          Tweezers Bottom
    cp_bull_meeting_lines       Bullish Meeting Lines
    cp_bear_meeting_lines       Bearish Meeting Lines
    cp_bull_belt_hold           Belt Hold (Bull)
    cp_bear_belt_hold           Belt Hold (Bear)

── Three-candle ─────────────────────────────────────────────────────
    cp_morning_star             Morning Star
    cp_evening_star             Evening Star
    cp_bull_harami_cross        Bullish Harami Cross
    cp_bear_harami_cross        Bearish Harami Cross
    cp_three_white_soldiers     Three White Soldiers
    cp_three_black_crows        Three Black Crows
    cp_three_inside_up          Three Inside Up
    cp_three_inside_down        Three Inside Down
    cp_three_outside_up         Three Outside Up
    cp_three_outside_down       Three Outside Down
    cp_morning_doji_star        Morning Doji Star
    cp_evening_doji_star        Evening Doji Star

── Composite signals (used by confluence engine) ────────────────────
    cp_bull_any                 Any bullish candle pattern on this bar
    cp_bear_any                 Any bearish candle pattern on this bar
    cp_bull_strong              High-reliability bull pattern (rel >= 4)
    cp_bear_strong              High-reliability bear pattern (rel >= 4)
    cp_bull_confirmation        Bull pattern confirming a sweep signal
    cp_bear_confirmation        Bear pattern confirming a sweep signal

Functions
---------
    detect_candle_patterns(df, params)   → enriched DataFrame
    pattern_summary(df)                  → DataFrame of pattern counts
"""

import pandas as pd
import numpy as np
from typing import Optional


# ─────────────────────────────────────────────────────────────────────────────
# Default thresholds
# ─────────────────────────────────────────────────────────────────────────────

DEFAULT_CP_PARAMS = {
    # Doji: body_ratio below this = doji
    "doji_body_ratio":          0.05,

    # Spinning top: body small but not quite doji
    "spinning_top_max_body":    0.35,
    "spinning_top_min_wick":    0.20,   # both wicks must be at least this

    # Hammer / Shooting Star
    "hs_max_body_ratio":        0.35,   # body must be small
    "hs_min_long_wick":         0.55,   # long wick must be at least this
    "hs_max_short_wick":        0.15,   # short wick must be small

    # Marubozu: nearly no wicks
    "marub_body_ratio":         0.90,
    "marub_max_wick":           0.05,

    # Engulfing: how much the body must exceed prior body
    "engulf_body_mult":         1.00,   # engulfing body >= 1x prior body

    # Piercing / Dark Cloud: midpoint penetration
    "piercing_midpoint":        0.50,

    # Harami: inner candle body must be within outer body
    "harami_inner_ratio":       0.60,   # inner body <= 60% of outer body

    # Tweezers: how close highs/lows must be (fraction of ATR)
    "tweezers_atr_tol":         0.15,

    # Meeting lines: close equality tolerance
    "meeting_lines_atr_tol":    0.10,

    # Morning/Evening Star: gap requirement (relaxed for forex — no true gaps)
    "star_gap_required":        False,  # True = require gap, False = allow no-gap
    "star_middle_body_ratio":   0.30,   # middle candle body must be small

    # Three White Soldiers / Black Crows
    "soldiers_body_ratio":      0.55,   # each candle body must be substantial
    "soldiers_open_in_body":    True,   # each opens within prior body

    # Belt Hold
    "belt_hold_body_ratio":     0.70,   # strong body
    "belt_hold_max_shadow":     0.05,   # almost no shadow on entry side

    # Trend context for context-sensitive patterns
    # (hammer valid only in downtrend, hanging man only in uptrend, etc.)
    "use_trend_context":        True,
    "trend_lookback":           10,     # bars to determine local trend
}


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _is_bull(o: np.ndarray, c: np.ndarray) -> np.ndarray:
    return c > o

def _is_bear(o: np.ndarray, c: np.ndarray) -> np.ndarray:
    return c < o

def _body_top(o: np.ndarray, c: np.ndarray) -> np.ndarray:
    return np.maximum(o, c)

def _body_bot(o: np.ndarray, c: np.ndarray) -> np.ndarray:
    return np.minimum(o, c)

def _shift(arr: np.ndarray, n: int, fill=np.nan) -> np.ndarray:
    """Shift array by n bars (positive = look back)."""
    result = np.empty_like(arr, dtype=float)
    if n > 0:
        result[:n] = fill
        result[n:] = arr[:-n]
    elif n < 0:
        result[n:] = fill
        result[:n] = arr[-n:]
    else:
        result = arr.copy().astype(float)
    return result

def _local_trend(close: np.ndarray, lookback: int) -> np.ndarray:
    """
    Returns 1 (uptrend), -1 (downtrend), 0 (neutral)
    based on whether close is above/below close N bars ago.
    """
    prev = _shift(close, lookback, fill=close[0])
    trend = np.where(close > prev, 1, np.where(close < prev, -1, 0))
    return trend


# ─────────────────────────────────────────────────────────────────────────────
# Main detection function
# ─────────────────────────────────────────────────────────────────────────────

def detect_candle_patterns(
    df: pd.DataFrame,
    params: Optional[dict] = None,
) -> pd.DataFrame:
    """
    Detect all candle patterns on an indicators-enriched DataFrame.
    Returns df with cp_* boolean columns added.
    """
    p = {**DEFAULT_CP_PARAMS, **(params or {})}
    df = df.copy()

    # ── Raw arrays ────────────────────────────────────────────────────────────
    o   = df["open"].values.astype(float)
    h   = df["high"].values.astype(float)
    lo  = df["low"].values.astype(float)
    c   = df["close"].values.astype(float)
    atr = df["atr"].values.astype(float)

    br  = df["body_ratio"].values.astype(float)       # body / range
    ur  = df["upper_rejection"].values.astype(float)  # upper wick / range
    lr  = df["lower_rejection"].values.astype(float)  # lower wick / range
    bs  = df["body_size"].values.astype(float)        # abs(close - open)

    bull = _is_bull(o, c)
    bear = _is_bear(o, c)

    # Local trend context
    trend = _local_trend(c, p["trend_lookback"])

    # Prior bar values
    o1 = _shift(o, 1, fill=o[0])
    h1 = _shift(h, 1, fill=h[0])
    l1 = _shift(lo, 1, fill=lo[0])
    c1 = _shift(c, 1, fill=c[0])
    br1 = _shift(br, 1, fill=br[0])
    bs1 = _shift(bs, 1, fill=bs[0])
    bull1 = _is_bull(o1, c1)
    bear1 = _is_bear(o1, c1)

    # Two bars ago
    o2 = _shift(o, 2, fill=o[0])
    c2 = _shift(c, 2, fill=c[0])
    br2 = _shift(br, 2, fill=br[0])
    bs2 = _shift(bs, 2, fill=bs[0])
    bull2 = _is_bull(o2, c2)
    bear2 = _is_bear(o2, c2)

    atr1 = _shift(atr, 1, fill=atr[0])
    ref_atr = np.where(np.isnan(atr), atr1, atr)  # fallback if atr NaN

    # ─────────────────────────────────────────────────────────────────────────
    # SINGLE CANDLE PATTERNS
    # ─────────────────────────────────────────────────────────────────────────

    doji_thresh  = p["doji_body_ratio"]
    hs_max_br    = p["hs_max_body_ratio"]
    hs_min_long  = p["hs_min_long_wick"]
    hs_max_short = p["hs_max_short_wick"]

    # Doji — open ≈ close
    is_doji = br <= doji_thresh

    # Dragonfly Doji — doji with long lower wick
    df["cp_dragonfly_doji"] = (
        is_doji &
        (lr >= 0.60) &
        (ur <= 0.10)
    )

    # Gravestone Doji — doji with long upper wick
    df["cp_gravestone_doji"] = (
        is_doji &
        (ur >= 0.60) &
        (lr <= 0.10)
    )

    # Long-Legged Doji — doji with both wicks long
    df["cp_long_legged_doji"] = (
        is_doji &
        (ur >= 0.30) &
        (lr >= 0.30)
    )

    # Plain Doji — indecision (not dragonfly or gravestone)
    df["cp_doji"] = (
        is_doji &
        ~df["cp_dragonfly_doji"].values &
        ~df["cp_gravestone_doji"].values &
        ~df["cp_long_legged_doji"].values
    )

    # Spinning Top — small body, wicks both sides
    df["cp_spinning_top"] = (
        (br <= p["spinning_top_max_body"]) &
        (br > doji_thresh) &
        (ur >= p["spinning_top_min_wick"]) &
        (lr >= p["spinning_top_min_wick"])
    )

    # Marubozu — almost no wicks
    df["cp_bull_marubozu"] = (
        bull &
        (br >= p["marub_body_ratio"]) &
        (ur <= p["marub_max_wick"]) &
        (lr <= p["marub_max_wick"])
    )
    df["cp_bear_marubozu"] = (
        bear &
        (br >= p["marub_body_ratio"]) &
        (ur <= p["marub_max_wick"]) &
        (lr <= p["marub_max_wick"])
    )

    # Hammer — small body, long lower wick, small upper wick
    # Valid in downtrend context
    hammer_shape = (
        (br <= hs_max_br) &
        (lr >= hs_min_long) &
        (ur <= hs_max_short)
    )
    if p["use_trend_context"]:
        df["cp_hammer"]       = hammer_shape & (trend == -1)
        df["cp_hanging_man"]  = hammer_shape & (trend == 1)
    else:
        df["cp_hammer"]       = hammer_shape
        df["cp_hanging_man"]  = hammer_shape

    # Shooting Star / Inverted Hammer — small body, long upper wick, small lower wick
    star_shape = (
        (br <= hs_max_br) &
        (ur >= hs_min_long) &
        (lr <= hs_max_short)
    )
    if p["use_trend_context"]:
        df["cp_shooting_star"]     = star_shape & (trend == 1)
        df["cp_inverted_hammer"]   = star_shape & (trend == -1)
    else:
        df["cp_shooting_star"]     = star_shape
        df["cp_inverted_hammer"]   = star_shape

    # ─────────────────────────────────────────────────────────────────────────
    # TWO CANDLE PATTERNS
    # ─────────────────────────────────────────────────────────────────────────

    # Bullish Engulfing — bearish prior, bullish current that engulfs prior body
    df["cp_bull_engulfing"] = (
        bear1 &
        bull &
        (o <= _body_bot(o1, c1)) &   # opens at or below prior body bottom
        (c >= _body_top(o1, c1)) &   # closes at or above prior body top
        (bs >= bs1 * p["engulf_body_mult"])
    )

    # Bearish Engulfing
    df["cp_bear_engulfing"] = (
        bull1 &
        bear &
        (o >= _body_top(o1, c1)) &
        (c <= _body_bot(o1, c1)) &
        (bs >= bs1 * p["engulf_body_mult"])
    )

    # Piercing Line — bear then bull opening below prior low, closing above prior midpoint
    prior_mid_bear = (o1 + c1) / 2
    df["cp_piercing_line"] = (
        bear1 &
        bull &
        (o < l1) &                          # opens below prior low
        (c > prior_mid_bear) &              # closes above prior midpoint
        (c < o1)                            # but not above prior open
    )

    # Dark Cloud Cover — bull then bear opening above prior high, closing below midpoint
    prior_mid_bull = (o1 + c1) / 2
    df["cp_dark_cloud_cover"] = (
        bull1 &
        bear &
        (o > h1) &                          # opens above prior high
        (c < prior_mid_bull) &              # closes below prior midpoint
        (c > c1)                            # but not below prior close
    )

    # Bullish Harami — large bear candle, small bull candle inside
    df["cp_bull_harami"] = (
        bear1 &
        bull &
        (o > _body_bot(o1, c1)) &          # open inside prior body
        (c < _body_top(o1, c1)) &          # close inside prior body
        (bs <= bs1 * p["harami_inner_ratio"])
    )

    # Bearish Harami
    df["cp_bear_harami"] = (
        bull1 &
        bear &
        (o < _body_top(o1, c1)) &
        (c > _body_bot(o1, c1)) &
        (bs <= bs1 * p["harami_inner_ratio"])
    )

    # Harami Cross (doji inside prior candle)
    df["cp_bull_harami_cross"] = (
        bear1 &
        is_doji &
        (h < _body_top(o1, c1)) &
        (lo > _body_bot(o1, c1))
    )
    df["cp_bear_harami_cross"] = (
        bull1 &
        is_doji &
        (h < _body_top(o1, c1)) &
        (lo > _body_bot(o1, c1))
    )

    # Tweezers Top — equal highs after uptrend
    tweezer_tol = ref_atr * p["tweezers_atr_tol"]
    df["cp_tweezers_top"] = (
        (trend == 1) &
        (np.abs(h - h1) <= tweezer_tol) &
        bull1 &
        bear
    )

    # Tweezers Bottom — equal lows after downtrend
    df["cp_tweezers_bottom"] = (
        (trend == -1) &
        (np.abs(lo - l1) <= tweezer_tol) &
        bear1 &
        bull
    )

    # Meeting Lines — opposite candles closing at same level
    meeting_tol = ref_atr * p["meeting_lines_atr_tol"]
    df["cp_bull_meeting_lines"] = (
        bear1 &
        bull &
        (np.abs(c - c1) <= meeting_tol)
    )
    df["cp_bear_meeting_lines"] = (
        bull1 &
        bear &
        (np.abs(c - c1) <= meeting_tol)
    )

    # Belt Hold Bull — opens at low (no lower wick), strong bullish body
    df["cp_bull_belt_hold"] = (
        bull &
        (lr <= p["belt_hold_max_shadow"]) &
        (br >= p["belt_hold_body_ratio"]) &
        (trend == -1)
    )

    # Belt Hold Bear — opens at high (no upper wick), strong bearish body
    df["cp_bear_belt_hold"] = (
        bear &
        (ur <= p["belt_hold_max_shadow"]) &
        (br >= p["belt_hold_body_ratio"]) &
        (trend == 1)
    )

    # ─────────────────────────────────────────────────────────────────────────
    # THREE CANDLE PATTERNS
    # ─────────────────────────────────────────────────────────────────────────

    star_mid_br = p["star_middle_body_ratio"]

    # Morning Star — bearish, small middle, bullish
    df["cp_morning_star"] = (
        bear2 &                              # first: bearish
        (br1 <= star_mid_br) &              # middle: small body
        bull &                               # third: bullish
        (c > (o2 + c2) / 2) &              # third closes above first midpoint
        (bs2 > bs1) &                       # first body larger than middle
        (trend == -1)
    )

    # Evening Star — bullish, small middle, bearish
    df["cp_evening_star"] = (
        bull2 &
        (br1 <= star_mid_br) &
        bear &
        (c < (o2 + c2) / 2) &
        (bs2 > bs1) &
        (trend == 1)
    )

    # Morning Doji Star — bearish, doji middle, bullish
    doji1 = _shift(is_doji.astype(float), 1, fill=0).astype(bool)
    df["cp_morning_doji_star"] = (
        bear2 &
        doji1 &
        bull &
        (c > (o2 + c2) / 2) &
        (trend == -1)
    )

    # Evening Doji Star
    df["cp_evening_doji_star"] = (
        bull2 &
        doji1 &
        bear &
        (c < (o2 + c2) / 2) &
        (trend == 1)
    )

    # Three White Soldiers
    bull2_arr = _is_bull(o2, c2)
    soldiers_br = p["soldiers_body_ratio"]
    df["cp_three_white_soldiers"] = (
        bull2_arr &
        bull1 &
        bull &
        (br2 >= soldiers_br) &
        (br1 >= soldiers_br) &
        (br  >= soldiers_br) &
        # each opens within prior body
        (o1 >= _body_bot(o2, c2)) & (o1 <= _body_top(o2, c2)) &
        (o  >= _body_bot(o1, c1)) & (o  <= _body_top(o1, c1)) &
        # each closes higher
        (c1 > c2) & (c > c1)
    )

    # Three Black Crows
    bear2_arr = _is_bear(o2, c2)
    df["cp_three_black_crows"] = (
        bear2_arr &
        bear1 &
        bear &
        (br2 >= soldiers_br) &
        (br1 >= soldiers_br) &
        (br  >= soldiers_br) &
        (o1 >= _body_bot(o2, c2)) & (o1 <= _body_top(o2, c2)) &
        (o  >= _body_bot(o1, c1)) & (o  <= _body_top(o1, c1)) &
        (c1 < c2) & (c < c1)
    )

    # Three Inside Up — harami then confirmation
    bull_harami1 = _shift(df["cp_bull_harami"].values.astype(float), 1, fill=0).astype(bool)
    df["cp_three_inside_up"] = (
        bull_harami1 &
        bull &
        (c > c1)
    )

    # Three Inside Down
    bear_harami1 = _shift(df["cp_bear_harami"].values.astype(float), 1, fill=0).astype(bool)
    df["cp_three_inside_down"] = (
        bear_harami1 &
        bear &
        (c < c1)
    )

    # Three Outside Up — engulfing then confirmation
    bull_eng1 = _shift(df["cp_bull_engulfing"].values.astype(float), 1, fill=0).astype(bool)
    df["cp_three_outside_up"] = (
        bull_eng1 &
        bull &
        (c > c1)
    )

    # Three Outside Down
    bear_eng1 = _shift(df["cp_bear_engulfing"].values.astype(float), 1, fill=0).astype(bool)
    df["cp_three_outside_down"] = (
        bear_eng1 &
        bear &
        (c < c1)
    )

    # ─────────────────────────────────────────────────────────────────────────
    # COMPOSITE SIGNALS
    # ─────────────────────────────────────────────────────────────────────────

    BULL_PATTERNS = [
        "cp_hammer", "cp_inverted_hammer", "cp_dragonfly_doji",
        "cp_bull_marubozu", "cp_bull_engulfing", "cp_piercing_line",
        "cp_bull_harami", "cp_bull_harami_cross", "cp_tweezers_bottom",
        "cp_bull_meeting_lines", "cp_bull_belt_hold",
        "cp_morning_star", "cp_morning_doji_star",
        "cp_three_white_soldiers", "cp_three_inside_up", "cp_three_outside_up",
    ]

    BEAR_PATTERNS = [
        "cp_hanging_man", "cp_shooting_star", "cp_gravestone_doji",
        "cp_bear_marubozu", "cp_bear_engulfing", "cp_dark_cloud_cover",
        "cp_bear_harami", "cp_bear_harami_cross", "cp_tweezers_top",
        "cp_bear_meeting_lines", "cp_bear_belt_hold",
        "cp_evening_star", "cp_evening_doji_star",
        "cp_three_black_crows", "cp_three_inside_down", "cp_three_outside_down",
    ]

    # High reliability patterns (reliability >= 4 from reference tool)
    BULL_STRONG = [
        "cp_bull_engulfing", "cp_morning_star", "cp_three_white_soldiers",
        "cp_three_outside_up", "cp_bull_marubozu",
    ]
    BEAR_STRONG = [
        "cp_bear_engulfing", "cp_evening_star", "cp_three_black_crows",
        "cp_three_outside_down", "cp_bear_marubozu",
    ]

    df["cp_bull_any"] = np.zeros(len(df), dtype=bool)
    df["cp_bear_any"] = np.zeros(len(df), dtype=bool)
    df["cp_bull_strong"] = np.zeros(len(df), dtype=bool)
    df["cp_bear_strong"] = np.zeros(len(df), dtype=bool)

    for col in BULL_PATTERNS:
        if col in df.columns:
            df["cp_bull_any"] = df["cp_bull_any"] | df[col]
    for col in BEAR_PATTERNS:
        if col in df.columns:
            df["cp_bear_any"] = df["cp_bear_any"] | df[col]
    for col in BULL_STRONG:
        if col in df.columns:
            df["cp_bull_strong"] = df["cp_bull_strong"] | df[col]
    for col in BEAR_STRONG:
        if col in df.columns:
            df["cp_bear_strong"] = df["cp_bear_strong"] | df[col]

    # Confirmation signals — candle pattern aligning with base sweep
    # (base_long / base_short come from signals.detect_sweeps)
    if "base_long" in df.columns:
        df["cp_bull_confirmation"] = df["cp_bull_any"] & df["base_long"]
        df["cp_bear_confirmation"] = df["cp_bear_any"] & df["base_short"]
    else:
        df["cp_bull_confirmation"] = df["cp_bull_any"]
        df["cp_bear_confirmation"] = df["cp_bear_any"]

    return df


# ─────────────────────────────────────────────────────────────────────────────
# Pattern summary utility
# ─────────────────────────────────────────────────────────────────────────────

def pattern_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Returns a summary DataFrame showing count and last occurrence
    of each detected pattern.
    """
    cp_cols = [c for c in df.columns if c.startswith("cp_") and
               c not in ("cp_bull_any", "cp_bear_any",
                         "cp_bull_strong", "cp_bear_strong",
                         "cp_bull_confirmation", "cp_bear_confirmation")]

    rows = []
    for col in cp_cols:
        count = int(df[col].sum())
        last_idx = df.index[df[col]].max() if count > 0 else None
        direction = "Bull" if any(x in col for x in [
            "hammer", "inverted", "dragonfly", "bull", "morning",
            "piercing", "tweezers_bot", "belt_hold_bull", "soldiers",
            "inside_up", "outside_up", "meeting_bull",
        ]) else "Bear"
        rows.append({
            "pattern":   col.replace("cp_", "").replace("_", " ").title(),
            "column":    col,
            "direction": direction,
            "count":     count,
            "last_bar":  last_idx,
        })

    return pd.DataFrame(rows).sort_values("count", ascending=False).reset_index(drop=True)


# ─────────────────────────────────────────────────────────────────────────────
# Integration: add candle pattern quality bonus to existing quality scores
# ─────────────────────────────────────────────────────────────────────────────

def apply_pattern_quality_bonus(
    df: pd.DataFrame,
    strong_bonus:  float = 12.0,
    any_bonus:     float =  6.0,
    confirm_bonus: float = 15.0,
) -> pd.DataFrame:
    """
    Boosts existing quality_long / quality_short scores based on
    candle pattern confirmation at the signal bar.

    strong_bonus   : added when a high-reliability pattern fires
    any_bonus      : added when any pattern fires
    confirm_bonus  : added when pattern directly confirms a sweep signal
    """
    df = df.copy()

    if "quality_long" not in df.columns:
        return df

    # Long quality boost
    long_bonus = np.zeros(len(df))
    if "cp_bull_confirmation" in df.columns:
        long_bonus += np.where(df["cp_bull_confirmation"].values, confirm_bonus, 0)
    if "cp_bull_strong" in df.columns:
        long_bonus += np.where(df["cp_bull_strong"].values, strong_bonus, 0)
    elif "cp_bull_any" in df.columns:
        long_bonus += np.where(df["cp_bull_any"].values, any_bonus, 0)

    # Short quality boost
    short_bonus = np.zeros(len(df))
    if "cp_bear_confirmation" in df.columns:
        short_bonus += np.where(df["cp_bear_confirmation"].values, confirm_bonus, 0)
    if "cp_bear_strong" in df.columns:
        short_bonus += np.where(df["cp_bear_strong"].values, strong_bonus, 0)
    elif "cp_bear_any" in df.columns:
        short_bonus += np.where(df["cp_bear_any"].values, any_bonus, 0)

    # Apply to all model quality scores
    q_long_cols  = [c for c in df.columns if c.endswith("_quality_long")]
    q_short_cols = [c for c in df.columns if c.endswith("_quality_short")]

    for col in q_long_cols:
        df[col] = np.clip(df[col].values + long_bonus, 0, 100)
    for col in q_short_cols:
        df[col] = np.clip(df[col].values + short_bonus, 0, 100)

    # Update top-level quality columns
    if "quality_long" in df.columns:
        df["quality_long"]  = np.clip(df["quality_long"].values  + long_bonus,  0, 100)
        df["quality_short"] = np.clip(df["quality_short"].values + short_bonus, 0, 100)

    return df


# ─────────────────────────────────────────────────────────────────────────────
# Full pipeline wrapper
# ─────────────────────────────────────────────────────────────────────────────

def build_candle_patterns(
    df: pd.DataFrame,
    params: Optional[dict] = None,
    apply_quality_bonus: bool = True,
) -> pd.DataFrame:
    """
    One-call wrapper: detect patterns then optionally boost quality scores.
    Call this after indicators.calc_all() and signals.detect_sweeps().
    """
    df = detect_candle_patterns(df, params)
    if apply_quality_bonus:
        df = apply_pattern_quality_bonus(df)
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Quick self-test
# ─────────────────────────────────────────────────────────────────────────────
