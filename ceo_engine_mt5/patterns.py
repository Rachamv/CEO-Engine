"""
The CEO Protocol — Phase 3: Geometric Pattern Detection
============================================================
Detects classical chart patterns using the pivot_high / pivot_low
arrays already produced by indicators.calc_all().

All detection is pivot-based — patterns are identified by the
geometric relationships between recent confirmed swing points,
not by curve-fitting or ML.

Patterns detected
-----------------
    Reversal patterns (high-reliability, used as confluence signals)
        Head & Shoulders          (bearish)
        Inverse Head & Shoulders  (bullish)
        Double Top                (bearish)
        Double Bottom             (bullish)
        Triple Top                (bearish)
        Triple Bottom             (bullish)
        Rising Wedge              (bearish)
        Falling Wedge             (bullish)

    Continuation patterns (trend confirmation)
        Bull Flag                 (bullish continuation)
        Bear Flag                 (bearish continuation)
        Ascending Triangle        (bullish breakout)
        Descending Triangle       (bearish breakout)
        Symmetrical Triangle      (neutral — breakout either direction)
        Bull Pennant              (bullish continuation)
        Bear Pennant              (bearish continuation)
        Rectangle / Range         (neutral — trade the breakout)

Output columns
--------------
    pat_hs              bool  — Head & Shoulders confirmed
    pat_ihs             bool  — Inverse H&S confirmed
    pat_double_top      bool  — Double Top
    pat_double_bottom   bool  — Double Bottom
    pat_triple_top      bool  — Triple Top
    pat_triple_bottom   bool  — Triple Bottom
    pat_rising_wedge    bool  — Rising Wedge
    pat_falling_wedge   bool  — Falling Wedge
    pat_bull_flag       bool  — Bull Flag
    pat_bear_flag       bool  — Bear Flag
    pat_asc_triangle    bool  — Ascending Triangle
    pat_desc_triangle   bool  — Descending Triangle
    pat_sym_triangle    bool  — Symmetrical Triangle
    pat_bull_pennant    bool  — Bull Pennant
    pat_bear_pennant    bool  — Bear Pennant
    pat_rectangle       bool  — Rectangle / Range

    pat_bull_any        bool  — any bullish pattern on this bar
    pat_bear_any        bool  — any bearish pattern on this bar
    pat_neutral_any     bool  — any neutral pattern on this bar
    pat_name            str   — name of detected pattern (or "")
    pat_quality         float — pattern quality score 0-100

Functions
---------
    detect_patterns(df, params)     → enriched DataFrame
    build_patterns(df, params)      → detect + apply quality bonus
"""

import numpy as np
import pandas as pd
from typing import Optional, List, Tuple


# ─────────────────────────────────────────────────────────────────────────────
# Default parameters
# ─────────────────────────────────────────────────────────────────────────────

DEFAULT_PAT_PARAMS = {
    # Pivot collection
    "pivot_lookback":       100,    # bars to collect pivots from
    "min_pivots":           4,      # minimum pivots needed to detect any pattern

    # Equality tolerance — two levels are "equal" if within this * ATR
    "equal_atr_tol":        0.30,

    # Head & Shoulders
    "hs_shoulder_sym":      0.25,   # shoulders must be within 25% height of each other
    "hs_neckline_slope":    0.15,   # neckline must be near-flat (slope tolerance)

    # Double / Triple top/bottom
    "dt_atr_tol":           0.35,   # tops/bottoms must be within this * ATR

    # Wedge
    "wedge_min_pivots":     4,      # need at least 2 highs + 2 lows
    "wedge_converge_tol":   0.10,   # trendlines must converge (not diverge beyond this)

    # Flag / Pennant
    "flag_impulse_bars":    5,      # impulse must form within this many bars
    "flag_impulse_atr":     2.5,    # impulse must be at least this * ATR
    "flag_channel_bars":    20,     # flag channel max duration
    "flag_retrace_max":     0.50,   # flag retraces at most 50% of impulse

    # Triangle
    "tri_min_pivots":       4,      # 2 highs + 2 lows minimum
    "tri_max_bars":         80,     # triangle must form within this many bars

    # Rectangle
    "rect_min_touches":     2,      # minimum touches per side
    "rect_atr_band":        0.40,   # highs/lows must be within this * ATR

    # Pattern quality bonuses (added to quality score)
    "hs_quality":           85.0,
    "dt_quality":           75.0,
    "wedge_quality":        70.0,
    "flag_quality":         65.0,
    "triangle_quality":     60.0,
    "rectangle_quality":    50.0,
}


# ─────────────────────────────────────────────────────────────────────────────
# Pivot collector
# ─────────────────────────────────────────────────────────────────────────────

def _build_pivot_index(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Pre-extract all pivot highs/lows once as parallel (index, value) arrays.
    Avoids re-scanning a rolling window with .iloc on every bar.
    """
    ph_vals = df["pivot_high"].values
    pl_vals = df["pivot_low"].values

    ph_idx = np.where(~np.isnan(ph_vals))[0]
    pl_idx = np.where(~np.isnan(pl_vals))[0]

    return ph_idx, ph_vals[ph_idx], pl_idx, pl_vals[pl_idx]


def _window_pivots(
    bar_idx:  int,
    lookback: int,
    idx_arr:  np.ndarray,
    val_arr:  np.ndarray,
) -> List[Tuple[int, float]]:
    """
    Return pivots within [bar_idx - lookback, bar_idx] using binary search
    on the pre-sorted pivot index array (O(log n) instead of O(lookback)).
    """
    if len(idx_arr) == 0:
        return []
    start = bar_idx - lookback
    lo = np.searchsorted(idx_arr, start, side="left")
    hi = np.searchsorted(idx_arr, bar_idx, side="right")
    if lo >= hi:
        return []
    return list(zip(idx_arr[lo:hi].tolist(), val_arr[lo:hi].tolist()))


def _atr_at(atr_arr: np.ndarray, idx: int) -> float:
    val = atr_arr[idx]
    return float(val) if not np.isnan(val) else 1.0


def _near(a: float, b: float, tol: float) -> bool:
    """True if a and b are within tol of each other."""
    return abs(a - b) <= tol


def _slope(x1: int, y1: float, x2: int, y2: float) -> float:
    if x2 == x1:
        return 0.0
    return (y2 - y1) / (x2 - x1)


# ─────────────────────────────────────────────────────────────────────────────
# Individual pattern detectors
# ─────────────────────────────────────────────────────────────────────────────

def _detect_hs(
    ph: List[Tuple[int, float]],
    pl: List[Tuple[int, float]],
    atr: float,
    p:   dict,
) -> Tuple[bool, bool]:
    """
    Returns (hs_detected, ihs_detected).
    H&S: left shoulder high, head high (higher), right shoulder high (lower than head, near left)
    IHS: left shoulder low, head low (lower), right shoulder low (higher than head, near left)
    """
    hs = ihs = False
    tol = atr * p["hs_shoulder_sym"]

    # Need at least 3 pivot highs for H&S
    if len(ph) >= 3:
        ls, head, rs = ph[-3], ph[-2], ph[-1]
        head_higher  = head[1] > ls[1] and head[1] > rs[1]
        shoulders_sym = _near(ls[1], rs[1], tol)
        # Neckline: connect the lows between shoulders — check slope is flat
        neckline_lows = [pt for pt in pl if ls[0] < pt[0] < rs[0]]
        if neckline_lows and head_higher and shoulders_sym:
            hs = True

    # Need at least 3 pivot lows for IHS
    if len(pl) >= 3:
        ls, head, rs = pl[-3], pl[-2], pl[-1]
        head_lower   = head[1] < ls[1] and head[1] < rs[1]
        shoulders_sym = _near(ls[1], rs[1], tol)
        neckline_highs = [h for h in ph if ls[0] < h[0] < rs[0]]
        if neckline_highs and head_lower and shoulders_sym:
            ihs = True

    return hs, ihs


def _detect_double(
    ph: List[Tuple[int, float]],
    pl: List[Tuple[int, float]],
    atr: float,
    p:   dict,
) -> Tuple[bool, bool, bool, bool]:
    """Returns (double_top, double_bottom, triple_top, triple_bottom)."""
    tol = atr * p["dt_atr_tol"]
    dt = db = tt = tb = False

    if len(ph) >= 2:
        if _near(ph[-1][1], ph[-2][1], tol):
            dt = True
    if len(ph) >= 3:
        if _near(ph[-1][1], ph[-2][1], tol) and _near(ph[-2][1], ph[-3][1], tol):
            tt = True

    if len(pl) >= 2:
        if _near(pl[-1][1], pl[-2][1], tol):
            db = True
    if len(pl) >= 3:
        if _near(pl[-1][1], pl[-2][1], tol) and _near(pl[-2][1], pl[-3][1], tol):
            tb = True

    return dt, db, tt, tb


def _detect_wedge(
    ph: List[Tuple[int, float]],
    pl: List[Tuple[int, float]],
    atr: float,
    p:   dict,
) -> Tuple[bool, bool]:
    """
    Rising wedge: both trendlines slope up, upper line flatter than lower (converging).
    Falling wedge: both slope down, lower line flatter than upper (converging).
    Returns (rising_wedge, falling_wedge).
    """
    rw = fw = False

    if len(ph) < 2 or len(pl) < 2:
        return rw, fw

    upper_slope = _slope(ph[-2][0], ph[-2][1], ph[-1][0], ph[-1][1])
    lower_slope = _slope(pl[-2][0], pl[-2][1], pl[-1][0], pl[-1][1])
    tol = p["wedge_converge_tol"]

    # Rising wedge: both slopes positive, upper slope < lower slope (converging upward)
    if upper_slope > 0 and lower_slope > 0:
        if lower_slope > upper_slope + tol:
            rw = True

    # Falling wedge: both slopes negative, upper slope < lower slope (converging downward)
    if upper_slope < 0 and lower_slope < 0:
        if upper_slope < lower_slope - tol:
            fw = True

    return rw, fw


def _detect_triangle(
    ph:   List[Tuple[int, float]],
    pl:   List[Tuple[int, float]],
    atr:  float,
    p:    dict,
    high: np.ndarray,
    low:  np.ndarray,
    idx:  int,
) -> Tuple[bool, bool, bool]:
    """
    Ascending triangle:   flat resistance + rising support
    Descending triangle:  flat support + falling resistance
    Symmetrical triangle: falling resistance + rising support
    Returns (asc, desc, sym).
    """
    asc = desc = sym = False

    if len(ph) < 2 or len(pl) < 2:
        return asc, desc, sym

    upper_slope = _slope(ph[-2][0], ph[-2][1], ph[-1][0], ph[-1][1])
    lower_slope = _slope(pl[-2][0], pl[-2][1], pl[-1][0], pl[-1][1])
    flat_tol    = atr * 0.05   # slope near zero

    # Ascending: flat top + rising bottom
    if abs(upper_slope) <= flat_tol and lower_slope > flat_tol:
        asc = True

    # Descending: falling top + flat bottom
    if upper_slope < -flat_tol and abs(lower_slope) <= flat_tol:
        desc = True

    # Symmetrical: falling top + rising bottom (converging)
    if upper_slope < -flat_tol and lower_slope > flat_tol:
        sym = True

    return asc, desc, sym


def _detect_flag(
    close: np.ndarray,
    high:  np.ndarray,
    low:   np.ndarray,
    idx:   int,
    atr:   float,
    p:     dict,
) -> Tuple[bool, bool]:
    """
    Bull flag: strong up impulse followed by tight downward channel.
    Bear flag: strong down impulse followed by tight upward channel.
    Returns (bull_flag, bear_flag).
    """
    bf = bef = False
    impulse_bars  = p["flag_impulse_bars"]
    impulse_atr   = p["flag_impulse_atr"]
    channel_bars  = p["flag_channel_bars"]
    retrace_max   = p["flag_retrace_max"]

    if idx < impulse_bars + channel_bars:
        return bf, bef

    # Look for bull flag impulse (strong up move)
    impulse_start = idx - channel_bars - impulse_bars
    impulse_end   = idx - channel_bars
    if impulse_start < 0:
        return bf, bef

    impulse_move_up   = close[impulse_end] - close[impulse_start]
    impulse_move_down = close[impulse_start] - close[impulse_end]

    # Bull flag
    if impulse_move_up >= atr * impulse_atr:
        # Check channel: recent bars form a gentle downward drift
        channel_high = max(high[impulse_end: idx + 1])
        channel_low  = min(low[impulse_end: idx + 1])
        channel_size = channel_high - channel_low
        retrace      = (channel_high - close[idx]) / impulse_move_up if impulse_move_up > 0 else 1.0
        if retrace <= retrace_max and channel_size <= atr * 2.0:
            bf = True

    # Bear flag
    if impulse_move_down >= atr * impulse_atr:
        channel_high = max(high[impulse_end: idx + 1])
        channel_low  = min(low[impulse_end: idx + 1])
        channel_size = channel_high - channel_low
        retrace      = (close[idx] - channel_low) / impulse_move_down if impulse_move_down > 0 else 1.0
        if retrace <= retrace_max and channel_size <= atr * 2.0:
            bef = True

    return bf, bef


def _detect_pennant(
    close: np.ndarray,
    idx:   int,
    atr:   float,
    p:     dict,
    ph:   List[Tuple[int, float]],
    pl:   List[Tuple[int, float]],
) -> Tuple[bool, bool]:
    """
    Pennant = impulse + miniature symmetrical triangle.
    Bull pennant: up impulse + converging triangle.
    Bear pennant: down impulse + converging triangle.
    """
    bp = bep = False
    impulse_atr  = p["flag_impulse_atr"]
    impulse_bars = p["flag_impulse_bars"]

    if idx < impulse_bars + 4:
        return bp, bep

    impulse_start = idx - impulse_bars - 4
    if impulse_start < 0:
        return bp, bep

    impulse_up   = close[idx - impulse_bars] - close[impulse_start]
    impulse_down = close[impulse_start] - close[idx - impulse_bars]

    # Need converging pivots after the impulse
    recent_ph = [x for x in ph if x[0] >= idx - 8]
    recent_pl = [x for x in pl if x[0] >= idx - 8]

    if len(recent_ph) >= 2 and len(recent_pl) >= 2:
        upper_slope = _slope(recent_ph[-2][0], recent_ph[-2][1], recent_ph[-1][0], recent_ph[-1][1])
        lower_slope = _slope(recent_pl[-2][0], recent_pl[-2][1], recent_pl[-1][0], recent_pl[-1][1])
        converging  = upper_slope < 0 and lower_slope > 0

        if impulse_up >= atr * impulse_atr and converging:
            bp = True
        if impulse_down >= atr * impulse_atr and converging:
            bep = True

    return bp, bep


def _detect_rectangle(
    ph:  List[Tuple[int, float]],
    pl:  List[Tuple[int, float]],
    atr: float,
    p:   dict,
) -> bool:
    """Rectangle: flat resistance + flat support with multiple touches."""
    tol = atr * p["rect_atr_band"]
    min_touches = p["rect_min_touches"]

    if len(ph) < min_touches or len(pl) < min_touches:
        return False

    # Check if recent pivot highs are all near the same level
    recent_highs = [x[1] for x in ph[-min_touches:]]
    recent_lows  = [x[1] for x in pl[-min_touches:]]

    high_range = max(recent_highs) - min(recent_highs)
    low_range  = max(recent_lows)  - min(recent_lows)

    return high_range <= tol and low_range <= tol


# ─────────────────────────────────────────────────────────────────────────────
# Main detection function
# ─────────────────────────────────────────────────────────────────────────────

def _detect_patterns_for_bar(
    i, lb, min_piv, ph_idx_all, ph_val_all, pl_idx_all, pl_val_all,
    atr_arr, close_arr, high, low, p,
    pat_hs, pat_ihs, pat_double_top, pat_double_bot, pat_triple_top, pat_triple_bot,
    pat_rising_wedge, pat_falling_wedge, pat_bull_flag, pat_bear_flag,
    pat_asc_tri, pat_desc_tri, pat_sym_tri, pat_bull_pen, pat_bear_pen,
    pat_rect, pat_quality, pat_name,
) -> None:
    """
    Runs every geometric pattern detector for one bar and writes results
    into the output arrays by index. Pulled verbatim out of detect_patterns()'s
    loop body — no detection logic changed, purely a complexity-reduction
    extraction (see CHANGELOG).
    """
    ph = _window_pivots(i, lb, ph_idx_all, ph_val_all)
    pl = _window_pivots(i, lb, pl_idx_all, pl_val_all)

    if len(ph) + len(pl) < min_piv:
        return

    atr = _atr_at(atr_arr, i)

    # ── H&S / IHS ────────────────────────────────────────────────────────
    hs, ihs = _detect_hs(ph, pl, atr, p)
    if hs:
        pat_hs[i]   = True
        pat_quality[i] = max(pat_quality[i], p["hs_quality"])
        pat_name[i] = "Head & Shoulders"
    if ihs:
        pat_ihs[i]  = True
        pat_quality[i] = max(pat_quality[i], p["hs_quality"])
        pat_name[i] = "Inv. Head & Shoulders"

    # ── Double / Triple ───────────────────────────────────────────────────
    dt, db, tt, tb = _detect_double(ph, pl, atr, p)
    if tt:
        pat_triple_top[i] = True
        pat_quality[i] = max(pat_quality[i], p["dt_quality"] + 5)
        pat_name[i] = "Triple Top"
    elif dt:
        pat_double_top[i] = True
        pat_quality[i] = max(pat_quality[i], p["dt_quality"])
        pat_name[i] = "Double Top"
    if tb:
        pat_triple_bot[i] = True
        pat_quality[i] = max(pat_quality[i], p["dt_quality"] + 5)
        pat_name[i] = "Triple Bottom"
    elif db:
        pat_double_bot[i] = True
        pat_quality[i] = max(pat_quality[i], p["dt_quality"])
        pat_name[i] = "Double Bottom"

    # ── Wedge ─────────────────────────────────────────────────────────────
    rw, fw = _detect_wedge(ph, pl, atr, p)
    if rw:
        pat_rising_wedge[i] = True
        pat_quality[i] = max(pat_quality[i], p["wedge_quality"])
        pat_name[i] = "Rising Wedge"
    if fw:
        pat_falling_wedge[i] = True
        pat_quality[i] = max(pat_quality[i], p["wedge_quality"])
        pat_name[i] = "Falling Wedge"

    # ── Triangle ──────────────────────────────────────────────────────────
    asc, desc, sym = _detect_triangle(ph, pl, atr, p, high, low, i)
    if asc:
        pat_asc_tri[i] = True
        pat_quality[i] = max(pat_quality[i], p["triangle_quality"])
        pat_name[i] = "Ascending Triangle"
    if desc:
        pat_desc_tri[i] = True
        pat_quality[i] = max(pat_quality[i], p["triangle_quality"])
        pat_name[i] = "Descending Triangle"
    if sym:
        pat_sym_tri[i] = True
        pat_quality[i] = max(pat_quality[i], p["triangle_quality"])
        pat_name[i] = "Symmetrical Triangle"

    # ── Flag ─────────────────────────────────────────────────────────────
    bf, bef = _detect_flag(close_arr, high, low, i, atr, p)
    if bf:
        pat_bull_flag[i] = True
        pat_quality[i] = max(pat_quality[i], p["flag_quality"])
        pat_name[i] = "Bull Flag"
    if bef:
        pat_bear_flag[i] = True
        pat_quality[i] = max(pat_quality[i], p["flag_quality"])
        pat_name[i] = "Bear Flag"

    # ── Pennant ───────────────────────────────────────────────────────────
    bp, bep = _detect_pennant(close_arr, i, atr, p, ph, pl)
    if bp:
        pat_bull_pen[i] = True
        pat_quality[i] = max(pat_quality[i], p["flag_quality"])
        pat_name[i] = "Bull Pennant"
    if bep:
        pat_bear_pen[i] = True
        pat_quality[i] = max(pat_quality[i], p["flag_quality"])
        pat_name[i] = "Bear Pennant"

    # ── Rectangle ────────────────────────────────────────────────────────
    rect = _detect_rectangle(ph, pl, atr, p)
    if rect:
        pat_rect[i] = True
        pat_quality[i] = max(pat_quality[i], p["rectangle_quality"])
        if not pat_name[i]:
            pat_name[i] = "Rectangle"


def detect_patterns(
    df:     pd.DataFrame,
    params: Optional[dict] = None,
) -> pd.DataFrame:
    """
    Detect all geometric patterns on an indicators-enriched DataFrame.
    Returns df with pat_* columns added.

    Call after indicators.calc_all() — requires pivot_high, pivot_low, atr.
    """
    p  = {**DEFAULT_PAT_PARAMS, **(params or {})}
    df = df.copy()
    n  = len(df)

    # Initialise output arrays
    pat_hs          = np.zeros(n, dtype=bool)
    pat_ihs         = np.zeros(n, dtype=bool)
    pat_double_top  = np.zeros(n, dtype=bool)
    pat_double_bot  = np.zeros(n, dtype=bool)
    pat_triple_top  = np.zeros(n, dtype=bool)
    pat_triple_bot  = np.zeros(n, dtype=bool)
    pat_rising_wedge  = np.zeros(n, dtype=bool)
    pat_falling_wedge = np.zeros(n, dtype=bool)
    pat_bull_flag   = np.zeros(n, dtype=bool)
    pat_bear_flag   = np.zeros(n, dtype=bool)
    pat_asc_tri     = np.zeros(n, dtype=bool)
    pat_desc_tri    = np.zeros(n, dtype=bool)
    pat_sym_tri     = np.zeros(n, dtype=bool)
    pat_bull_pen    = np.zeros(n, dtype=bool)
    pat_bear_pen    = np.zeros(n, dtype=bool)
    pat_rect        = np.zeros(n, dtype=bool)
    pat_quality     = np.zeros(n)
    pat_name        = [""] * n

    high  = df["high"].values
    low   = df["low"].values
    close_arr = df["close"].values
    atr_arr = df["atr"].values

    lb = p["pivot_lookback"]
    min_piv = p["min_pivots"]

    ph_idx_all, ph_val_all, pl_idx_all, pl_val_all = _build_pivot_index(df)

    for i in range(lb, n):
        _detect_patterns_for_bar(
            i, lb, min_piv, ph_idx_all, ph_val_all, pl_idx_all, pl_val_all,
            atr_arr, close_arr, high, low, p,
            pat_hs, pat_ihs, pat_double_top, pat_double_bot, pat_triple_top, pat_triple_bot,
            pat_rising_wedge, pat_falling_wedge, pat_bull_flag, pat_bear_flag,
            pat_asc_tri, pat_desc_tri, pat_sym_tri, pat_bull_pen, pat_bear_pen,
            pat_rect, pat_quality, pat_name,
        )

    # ── Write to DataFrame ────────────────────────────────────────────────────
    df["pat_hs"]             = pat_hs
    df["pat_ihs"]            = pat_ihs
    df["pat_double_top"]     = pat_double_top
    df["pat_double_bottom"]  = pat_double_bot
    df["pat_triple_top"]     = pat_triple_top
    df["pat_triple_bottom"]  = pat_triple_bot
    df["pat_rising_wedge"]   = pat_rising_wedge
    df["pat_falling_wedge"]  = pat_falling_wedge
    df["pat_bull_flag"]      = pat_bull_flag
    df["pat_bear_flag"]      = pat_bear_flag
    df["pat_asc_triangle"]   = pat_asc_tri
    df["pat_desc_triangle"]  = pat_desc_tri
    df["pat_sym_triangle"]   = pat_sym_tri
    df["pat_bull_pennant"]   = pat_bull_pen
    df["pat_bear_pennant"]   = pat_bear_pen
    df["pat_rectangle"]      = pat_rect
    df["pat_quality"]        = pat_quality
    df["pat_name"]           = pat_name

    BULL_PATS = ["pat_ihs", "pat_double_bottom", "pat_triple_bottom",
                 "pat_falling_wedge", "pat_bull_flag", "pat_bull_pennant",
                 "pat_asc_triangle"]
    BEAR_PATS = ["pat_hs", "pat_double_top", "pat_triple_top",
                 "pat_rising_wedge", "pat_bear_flag", "pat_bear_pennant",
                 "pat_desc_triangle"]
    NEUT_PATS = ["pat_sym_triangle", "pat_rectangle"]

    df["pat_bull_any"]    = np.zeros(n, dtype=bool)
    df["pat_bear_any"]    = np.zeros(n, dtype=bool)
    df["pat_neutral_any"] = np.zeros(n, dtype=bool)

    for col in BULL_PATS:
        df["pat_bull_any"] = df["pat_bull_any"] | df[col]
    for col in BEAR_PATS:
        df["pat_bear_any"] = df["pat_bear_any"] | df[col]
    for col in NEUT_PATS:
        df["pat_neutral_any"] = df["pat_neutral_any"] | df[col]

    return df


# ─────────────────────────────────────────────────────────────────────────────
# Quality bonus application
# ─────────────────────────────────────────────────────────────────────────────

def apply_pattern_quality_bonus(
    df:          pd.DataFrame,
    bull_bonus:  float = 8.0,
    bear_bonus:  float = 8.0,
) -> pd.DataFrame:
    """Boosts quality scores when a geometric pattern aligns with a signal."""
    df = df.copy()
    if "quality_long" not in df.columns:
        return df

    pat_q   = df["pat_quality"].values
    bull_on = df["pat_bull_any"].values
    bear_on = df["pat_bear_any"].values

    long_bonus  = np.where(bull_on, np.maximum(pat_q * 0.10, bull_bonus), 0)
    short_bonus = np.where(bear_on, np.maximum(pat_q * 0.10, bear_bonus), 0)

    for col in [c for c in df.columns if c.endswith("_quality_long")]:
        df[col] = np.clip(df[col].values + long_bonus, 0, 100)
    for col in [c for c in df.columns if c.endswith("_quality_short")]:
        df[col] = np.clip(df[col].values + short_bonus, 0, 100)

    if "quality_long" in df.columns:
        df["quality_long"]  = np.clip(df["quality_long"].values  + long_bonus,  0, 100)
        df["quality_short"] = np.clip(df["quality_short"].values + short_bonus, 0, 100)

    return df


def build_patterns(
    df:     pd.DataFrame,
    params: Optional[dict] = None,
    apply_bonus: bool = True,
) -> pd.DataFrame:
    """One-call wrapper: detect patterns + optionally boost quality scores."""
    df = detect_patterns(df, params)
    if apply_bonus:
        df = apply_pattern_quality_bonus(df)
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Quick self-test
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys, os
    from .data       import fetch_ohlcv, resample_ohlcv
    from .indicators import calc_all
    from .signals    import build_all

    print("=" * 60)
    print("CEO Engine — Geometric Pattern Detection Test")
    print("=" * 60)

    np.random.seed(99)
    n = 3000
    dates = pd.date_range("2023-01-01", periods=n, freq="1h", tz="UTC")
    close = 1900 + np.cumsum(np.random.randn(n) * 2.0)
    close = np.maximum(close, 1000)

    raw = pd.DataFrame({
        "datetime": dates,
        "open":   close * (1 + np.random.randn(n) * 0.0003),
        "high":   close * (1 + np.abs(np.random.randn(n)) * 0.0008),
        "low":    close * (1 - np.abs(np.random.randn(n)) * 0.0008),
        "close":  close,
        "volume": np.random.randint(500, 8000, n).astype(float),
    })
    raw.to_csv("/tmp/test_xauusd_pat.csv", index=False)

    df  = fetch_ohlcv("XAUUSD", source="csv", filepath="/tmp/test_xauusd_pat.csv")
    df  = calc_all(df)
    htf = resample_ohlcv(df, "4h")
    df  = build_all(df, htf_df=htf)
    df  = build_patterns(df)

    pat_cols = [c for c in df.columns if c.startswith("pat_") and
                c not in ("pat_quality", "pat_name", "pat_bull_any",
                          "pat_bear_any", "pat_neutral_any")]

    print(f"\n── Pattern Counts ──")
    for col in pat_cols:
        count = int(df[col].sum())
        if count > 0:
            print(f"  {col:<25} : {count}")

    named = df[df["pat_name"] != ""]
    print(f"\n── Named pattern bars: {len(named)} ──")
    if len(named) > 0:
        print(named[["pat_name", "pat_quality"]].tail(10).to_string())

    print(f"\n  pat_bull_any    : {df['pat_bull_any'].sum()}")
    print(f"  pat_bear_any    : {df['pat_bear_any'].sum()}")
    print(f"  pat_neutral_any : {df['pat_neutral_any'].sum()}")

    print("\n✅  Geometric Pattern Detection complete.")
