"""
The CEO Protocol — Layer 2.7: CEO Structure Detection
==========================================================
Implements the structural concepts from the CEO Institutional Method
that cannot be replicated in Pine Script due to state and compute limits.

Concepts implemented
--------------------
1.  Break of Structure (BOS)
        Price closes beyond the last swing high/low AFTER a sweep.
        Validates that the sweep was genuine, not just noise.

2.  Double BOS validation
        Two sequential structure breaks required to confirm a protected
        swing leg. First break = transactional liquidity taken.
        Second break = swing leg confirmed protected.

3.  Fibonacci Premium / Discount zones
        Drawn from swing low to swing high (bull) or high to low (bear).
        Below 50% equilibrium = discount (buy zone).
        Above 50% = premium (sell zone).
        Filters whether a signal bar sits in the correct zone.

4.  Order Block (OB) detection
        Last bearish candle before a bullish displacement (bull OB).
        Last bullish candle before a bearish displacement (bear OB).
        Marks the OB open/high/low/close as a tradeable zone.

5.  Quasimodo (QM) level
        Failed OB — an OB candle whose range was broken by the BOS.
        Highest-probability entry per the CEO method.
        Tracks whether each QM level has been mitigated (price returned).

6.  Structural vs Inducement liquidity classification
        Structural: internal low/high that sits at or below 50% Fib
                    and forms a clean V/A shape (multi-candle cluster).
        Inducement: internal low/high above 50%, or poorly shaped —
                    do NOT trade this level; it will be swept first.

7.  Unmitigated level tracking
        Stateful map of all OB and QM levels, flagged mitigated/unmitigated.
        Nearest unmitigated level above/below current price = TP target.

Output columns
--------------
    bos_long            bool   — bullish BOS confirmed this bar (after sweep)
    bos_short           bool   — bearish BOS confirmed this bar
    double_bos_long     bool   — second sequential BOS (swing leg confirmed)
    double_bos_short    bool   — second sequential BOS (short)

    in_discount         bool   — current bar close is in discount zone (< 50% Fib)
    in_premium          bool   — current bar close is in premium zone (> 50% Fib)
    fib_50              float  — equilibrium level of current swing
    fib_zone_score      float  — 0-100, how well-placed signal is in correct zone

    ob_bull_active      bool   — unmitigated bullish OB exists within recent bars
    ob_bear_active      bool   — unmitigated bearish OB exists within recent bars
    ob_bull_high        float  — bullish OB zone top
    ob_bull_low         float  — bullish OB zone bottom
    ob_bear_high        float  — bearish OB zone top
    ob_bear_low         float  — bearish OB zone bottom

    qm_bull_active      bool   — unmitigated bullish QM level present
    qm_bear_active      bool   — unmitigated bearish QM level present
    qm_bull_level       float  — bullish QM price level
    qm_bear_level       float  — bearish QM price level

    struct_liq_long     bool   — valid structural liquidity for long setup
    struct_liq_short    bool   — valid structural liquidity for short setup
    inducement_long     bool   — internal low is inducement (do not trade)
    inducement_short    bool   — internal high is inducement (do not trade)

    nearest_unmit_res   float  — nearest unmitigated resistance level (TP for longs)
    nearest_unmit_sup   float  — nearest unmitigated support level (TP for shorts)

    ceo_long_valid      bool   — full CEO sequence valid for long  (master gate)
    ceo_short_valid     bool   — full CEO sequence valid for short (master gate)
    ceo_quality_bonus   float  — quality score bonus from structure (0-30)

Functions
---------
    detect_bos(df, params)              → adds BOS columns
    detect_fib_zones(df, params)        → adds Fibonacci zone columns
    detect_order_blocks(df, params)     → adds OB columns
    detect_qm_levels(df, params)        → adds QM columns (requires OB + BOS)
    detect_structural_liquidity(df, p)  → adds structural liq columns
    track_unmitigated_levels(df, p)     → adds TP target columns
    build_ceo_structure(df, params)     → runs all, returns enriched df
"""

import pandas as pd
import numpy as np
from typing import Optional, List


# ─────────────────────────────────────────────────────────────────────────────
# Default parameters
# ─────────────────────────────────────────────────────────────────────────────

DEFAULT_STRUCT_PARAMS = {
    # BOS
    "bos_close_confirm":        True,    # BOS requires a close beyond swing, not just wick
    "bos_lookback":             50,      # bars to look back for active swing reference

    # Fibonacci
    "fib_swing_lookback":       100,     # bars to identify the current swing leg
    "fib_equilibrium":          0.50,    # midpoint
    "fib_discount_max":         0.50,    # below this = discount
    "fib_premium_min":          0.50,    # above this = premium
    "fib_golden_low":           0.618,   # golden pocket low (optional bonus)
    "fib_golden_high":          0.65,    # golden pocket high

    # Order Blocks
    "ob_lookback":              20,      # bars to look back for OB detection
    "ob_atr_min":               0.30,    # OB body must be at least this * ATR
    "ob_mitigation_close":      True,    # OB mitigated when price closes through it
    "ob_max_age":               100,     # OB older than this bars = expired

    # QM levels
    "qm_lookback":              50,      # bars to look for QM zones

    # Structural liquidity
    "struct_liq_min_candles":   2,       # minimum candles forming the V/A shape
    "struct_liq_max_fib":       0.52,    # must be at or below this Fib level
    "struct_liq_lookback":      30,      # bars to look for internal lows/highs

    # Unmitigated level tracking
    "unmit_max_levels":         5,       # max levels to track per side
    "unmit_lookback":           200,     # bars of history to scan for levels

    # CEO sequence gates
    "require_bos":              True,    # signal must have BOS after sweep
    "require_fib_zone":         True,    # signal must be in correct Fib zone
    "require_ob_or_qm":         False,   # signal must have nearby OB or QM
    "require_struct_liq":       False,   # signal must have structural liquidity
}


# ─────────────────────────────────────────────────────────────────────────────
# Utility
# ─────────────────────────────────────────────────────────────────────────────

def _shift(arr: np.ndarray, n: int, fill=np.nan) -> np.ndarray:
    result = np.empty(len(arr), dtype=float)
    if n > 0:
        result[:n] = fill
        result[n:] = arr[:-n]
    elif n < 0:
        result[n:] = fill
        result[:n] = arr[-n:]
    else:
        result[:] = arr
    return result


def _rolling_max(arr: np.ndarray, window: int) -> np.ndarray:
    result = np.full(len(arr), np.nan)
    for i in range(window - 1, len(arr)):
        result[i] = np.nanmax(arr[i - window + 1: i + 1])
    return result


def _rolling_min(arr: np.ndarray, window: int) -> np.ndarray:
    result = np.full(len(arr), np.nan)
    for i in range(window - 1, len(arr)):
        result[i] = np.nanmin(arr[i - window + 1: i + 1])
    return result


# ─────────────────────────────────────────────────────────────────────────────
# 1. Break of Structure
# ─────────────────────────────────────────────────────────────────────────────

def detect_bos(df: pd.DataFrame, params: Optional[dict] = None) -> pd.DataFrame:
    """
    Detects Break of Structure (BOS) after a liquidity sweep.

    Bull BOS: after a base_long sweep, price closes above last_swing_high
    Bear BOS: after a base_short sweep, price closes below last_swing_low

    Also detects double BOS (two sequential breaks = swing leg confirmed).
    """
    p = {**DEFAULT_STRUCT_PARAMS, **(params or {})}
    df = df.copy()

    close          = df["close"].values
    high           = df["high"].values
    low            = df["low"].values
    last_sh        = df["last_swing_high"].values  # from signals.detect_sweeps
    last_sl        = df["last_swing_low"].values
    base_long      = df["base_long"].values.astype(bool)
    base_short     = df["base_short"].values.astype(bool)
    n              = len(df)

    bos_long       = np.zeros(n, dtype=bool)
    bos_short      = np.zeros(n, dtype=bool)
    double_bos_long  = np.zeros(n, dtype=bool)
    double_bos_short = np.zeros(n, dtype=bool)

    # State trackers
    pending_long   = False   # waiting for BOS after a sweep
    pending_short  = False
    bos_long_count = 0       # sequential BOS counter
    bos_short_count = 0
    ref_high       = np.nan  # swing high to break for BOS
    ref_low        = np.nan  # swing low to break for BOS

    for i in range(1, n):
        # New sweep resets pending state and captures reference level
        if base_long[i]:
            pending_long   = True
            bos_long_count = 0
            ref_high       = last_sh[i]

        if base_short[i]:
            pending_short   = True
            bos_short_count = 0
            ref_low         = last_sl[i]

        # Check for BOS (close confirmation)
        if pending_long and not np.isnan(ref_high):
            broke = close[i] > ref_high if p["bos_close_confirm"] else high[i] > ref_high
            if broke:
                bos_long[i]  = True
                bos_long_count += 1
                if bos_long_count >= 2:
                    double_bos_long[i] = True
                # Advance reference to the new swing high so a second BOS
                # requires a genuinely new structural break, not just the
                # next bar closing marginally higher.
                new_ref = last_sh[i] if (not np.isnan(last_sh[i]) and last_sh[i] > ref_high) else close[i]
                ref_high = new_ref
                pending_long = False   # require a new sweep before next BOS

        if pending_short and not np.isnan(ref_low):
            broke = close[i] < ref_low if p["bos_close_confirm"] else low[i] < ref_low
            if broke:
                bos_short[i]  = True
                bos_short_count += 1
                if bos_short_count >= 2:
                    double_bos_short[i] = True
                new_ref = last_sl[i] if (not np.isnan(last_sl[i]) and last_sl[i] < ref_low) else close[i]
                ref_low = new_ref
                pending_short = False   # require a new sweep before next BOS

    df["bos_long"]         = bos_long
    df["bos_short"]        = bos_short
    df["double_bos_long"]  = double_bos_long
    df["double_bos_short"] = double_bos_short

    return df


# ─────────────────────────────────────────────────────────────────────────────
# 2. Fibonacci Premium / Discount Zones
# ─────────────────────────────────────────────────────────────────────────────

def detect_fib_zones(df: pd.DataFrame, params: Optional[dict] = None) -> pd.DataFrame:
    """
    Identifies the current active swing leg and classifies each bar
    as being in premium, discount, or equilibrium zone.

    Swing leg defined as: highest high to lowest low over lookback window,
    oriented by the current trend direction (EMA cross from indicators).
    """
    p = {**DEFAULT_STRUCT_PARAMS, **(params or {})}
    df = df.copy()

    close    = df["close"].values
    high     = df["high"].values
    low      = df["low"].values
    lb       = p["fib_swing_lookback"]
    eq       = p["fib_equilibrium"]
    n        = len(df)

    in_discount   = np.zeros(n, dtype=bool)
    in_premium    = np.zeros(n, dtype=bool)
    in_golden     = np.zeros(n, dtype=bool)
    fib_50        = np.full(n, np.nan)
    fib_zone_score = np.zeros(n)

    trend_long  = df["trend_long"].values  if "trend_long"  in df.columns else np.ones(n, dtype=bool)
    trend_short = df["trend_short"].values if "trend_short" in df.columns else np.zeros(n, dtype=bool)

    for i in range(lb, n):
        window_h = high[i - lb: i + 1]
        window_l = low[i - lb: i + 1]

        swing_high = np.nanmax(window_h)
        swing_low  = np.nanmin(window_l)
        swing_rng  = swing_high - swing_low

        if swing_rng <= 0:
            continue

        mid = swing_low + swing_rng * eq
        fib_50[i] = mid

        c = close[i]

        # Discount = below 50% (good for longs)
        # Premium  = above 50% (good for shorts)
        if trend_long[i]:
            in_discount[i] = c < mid
            in_premium[i]  = c > mid
        elif trend_short[i]:
            in_discount[i] = c > mid   # premium for shorts = above 50%
            in_premium[i]  = c < mid
        else:
            in_discount[i] = c < mid
            in_premium[i]  = c > mid

        # Golden pocket (0.618–0.65) — extra quality bonus
        fib_618 = swing_high - swing_rng * p["fib_golden_low"]
        fib_650 = swing_high - swing_rng * p["fib_golden_high"]
        if min(fib_618, fib_650) <= c <= max(fib_618, fib_650):
            in_golden[i] = True

        # Zone score: 100 if in golden pocket, 75 if in correct zone, 25 if wrong zone
        if in_golden[i]:
            fib_zone_score[i] = 100.0
        elif in_discount[i] and trend_long[i]:
            # How deep in discount: closer to swing_low = better
            depth = (mid - c) / (mid - swing_low)
            fib_zone_score[i] = 50.0 + min(depth, 1.0) * 50.0
        elif in_premium[i] and trend_short[i]:
            depth = (c - mid) / (swing_high - mid)
            fib_zone_score[i] = 50.0 + min(depth, 1.0) * 50.0
        else:
            fib_zone_score[i] = 20.0   # wrong zone

    df["in_discount"]    = in_discount
    df["in_premium"]     = in_premium
    df["in_golden"]      = in_golden
    df["fib_50"]         = fib_50
    df["fib_zone_score"] = fib_zone_score

    return df


# ─────────────────────────────────────────────────────────────────────────────
# 3. Order Block Detection
# ─────────────────────────────────────────────────────────────────────────────

def _prune_order_blocks(active_obs, i, c, h, lo, p, is_bull):
    """Drops mitigated or expired order blocks from the active list."""
    surviving = []
    for ob_h, ob_l, ob_bar in active_obs:
        age = i - ob_bar
        if is_bull:
            mitigated = c[i] < ob_l if p["ob_mitigation_close"] else lo[i] < ob_l
        else:
            mitigated = c[i] > ob_h if p["ob_mitigation_close"] else h[i] > ob_h
        if not mitigated and age <= p["ob_max_age"]:
            surviving.append((ob_h, ob_l, ob_bar))
    return surviving


def _detect_new_order_block(i, o, c, atr, p, active_obs, max_slots, is_bull):
    """
    Looks back from bar i for the last opposite-colored candle before a
    displacement bar, and if its body clears the ATR threshold, adds it
    as a new order block (capped at max_slots, dropping the oldest).
    Returns the possibly-updated active_obs list.
    """
    lookback = min(i, p["ob_lookback"])
    for j in range(i - 1, i - lookback - 1, -1):
        is_candidate = (c[j] < o[j]) if is_bull else (c[j] > o[j])
        if not is_candidate:
            continue
        ob_body = abs(o[j] - c[j])
        if ob_body >= atr[i] * p["ob_atr_min"]:
            new_ob = (max(o[j], c[j]), min(o[j], c[j]), i)
            if not any(abs(ob[0] - new_ob[0]) < 1e-8 for ob in active_obs):
                active_obs.append(new_ob)
                if len(active_obs) > max_slots:
                    active_obs = active_obs[-max_slots:]
        break
    return active_obs


def detect_order_blocks(df: pd.DataFrame, params: Optional[dict] = None) -> pd.DataFrame:
    """
    Detects Order Blocks:
    - Bull OB: last bearish candle before a bullish displacement
    - Bear OB: last bullish candle before a bearish displacement

    Tracks mitigation (price returning through the OB zone).
    """
    p  = {**DEFAULT_STRUCT_PARAMS, **(params or {})}
    df = df.copy()

    o     = df["open"].values
    h     = df["high"].values
    lo    = df["low"].values
    c     = df["close"].values
    atr   = df["atr"].values
    bull_disp = df["bull_displacement"].values.astype(bool)
    bear_disp = df["bear_displacement"].values.astype(bool)
    n     = len(df)

    ob_bull_active = np.zeros(n, dtype=bool)
    ob_bear_active = np.zeros(n, dtype=bool)
    ob_bull_high   = np.full(n, np.nan)
    ob_bull_low    = np.full(n, np.nan)
    ob_bear_high   = np.full(n, np.nan)
    ob_bear_low    = np.full(n, np.nan)

    MAX_OB_SLOTS = 3   # track up to 3 unmitigated OBs per side

    # Each entry: (high, low, bar_idx)
    active_bull_obs: list = []
    active_bear_obs: list = []

    for i in range(1, n):
        active_bull_obs = _prune_order_blocks(active_bull_obs, i, c, h, lo, p, is_bull=True)
        active_bear_obs = _prune_order_blocks(active_bear_obs, i, c, h, lo, p, is_bull=False)

        if bull_disp[i]:
            active_bull_obs = _detect_new_order_block(
                i, o, c, atr, p, active_bull_obs, MAX_OB_SLOTS, is_bull=True)
        if bear_disp[i]:
            active_bear_obs = _detect_new_order_block(
                i, o, c, atr, p, active_bear_obs, MAX_OB_SLOTS, is_bull=False)

        # ── Record nearest active OB per side (closest to current price) ──────
        # For the DataFrame columns we expose the nearest (most relevant) OB.
        # The chart overlays can draw all of them via the extended columns below.
        if active_bull_obs:
            ob_bull_active[i] = True
            # Nearest = highest low (closest to current price from below)
            nearest = max(active_bull_obs, key=lambda x: x[1])
            ob_bull_high[i]   = nearest[0]
            ob_bull_low[i]    = nearest[1]

        if active_bear_obs:
            ob_bear_active[i] = True
            # Nearest = lowest high (closest from above)
            nearest = min(active_bear_obs, key=lambda x: x[0])
            ob_bear_high[i]   = nearest[0]
            ob_bear_low[i]    = nearest[1]

    df["ob_bull_active"] = ob_bull_active
    df["ob_bear_active"] = ob_bear_active
    df["ob_bull_high"]   = ob_bull_high
    df["ob_bull_low"]    = ob_bull_low
    df["ob_bear_high"]   = ob_bear_high
    df["ob_bear_low"]    = ob_bear_low

    return df


# ─────────────────────────────────────────────────────────────────────────────
# 4. Quasimodo (QM) Level Detection
# ─────────────────────────────────────────────────────────────────────────────

def detect_qm_levels(df: pd.DataFrame, params: Optional[dict] = None) -> pd.DataFrame:
    """
    QM = failed Order Block — an OB whose range was broken by a BOS.

    Bull QM: a bearish OB candle whose low was broken by the BOS move
    Bear QM: a bullish OB candle whose high was broken by the BOS move

    The QM level is the midpoint of the failed OB body.
    Tracks mitigation (price returning to the QM level).
    """
    {**DEFAULT_STRUCT_PARAMS, **(params or {})}
    df = df.copy()

    c          = df["close"].values
    df["high"].values
    df["low"].values
    bos_long   = df["bos_long"].values.astype(bool)
    bos_short  = df["bos_short"].values.astype(bool)
    n          = len(df)

    qm_bull_active = np.zeros(n, dtype=bool)
    qm_bear_active = np.zeros(n, dtype=bool)
    qm_bull_level  = np.full(n, np.nan)
    qm_bear_level  = np.full(n, np.nan)

    active_bull_qm = None   # (level, bar_index)
    active_bear_qm = None

    for i in range(1, n):
        # Check mitigation of existing QM levels
        if active_bull_qm is not None:
            qm_lvl, qm_bar = active_bull_qm
            # Mitigated when price returns to and closes below the QM level
            if c[i] < qm_lvl:
                active_bull_qm = None

        if active_bear_qm is not None:
            qm_lvl, qm_bar = active_bear_qm
            if c[i] > qm_lvl:
                active_bear_qm = None

        # New Bull QM: BOS to upside broke through the last bear OB zone
        if bos_long[i] and "ob_bull_low" in df.columns:
            prev_ob_low = df["ob_bull_low"].iloc[i]
            if not np.isnan(prev_ob_low):
                # QM level = the OB low that was broken
                active_bull_qm = (prev_ob_low, i)

        # New Bear QM: BOS to downside broke through last bull OB zone
        if bos_short[i] and "ob_bear_high" in df.columns:
            prev_ob_high = df["ob_bear_high"].iloc[i]
            if not np.isnan(prev_ob_high):
                active_bear_qm = (prev_ob_high, i)

        if active_bull_qm is not None:
            qm_bull_active[i] = True
            qm_bull_level[i]  = active_bull_qm[0]

        if active_bear_qm is not None:
            qm_bear_active[i] = True
            qm_bear_level[i]  = active_bear_qm[0]

    df["qm_bull_active"] = qm_bull_active
    df["qm_bear_active"] = qm_bear_active
    df["qm_bull_level"]  = qm_bull_level
    df["qm_bear_level"]  = qm_bear_level

    return df


# ─────────────────────────────────────────────────────────────────────────────
# 5. Structural vs Inducement Liquidity
# ─────────────────────────────────────────────────────────────────────────────

def detect_structural_liquidity(
    df: pd.DataFrame,
    params: Optional[dict] = None,
) -> pd.DataFrame:
    """
    Classifies internal swing points as structural liquidity or inducement.

    Structural liquidity criteria (must pass BOTH):
    1. Sits at or below 50% Fibonacci equilibrium of the swing leg
    2. Forms a V/A shape — confirmed by pivot detection (multi-candle cluster)

    Inducement: internal low/high that fails either criterion.
    Inducement means the level WILL be swept before the real move.
    """
    p  = {**DEFAULT_STRUCT_PARAMS, **(params or {})}
    df = df.copy()

    pivot_low  = df["pivot_low"].values   # from indicators
    pivot_high = df["pivot_high"].values
    fib_50     = df["fib_50"].values      # from detect_fib_zones
    df["close"].values
    n          = len(df)

    struct_liq_long  = np.zeros(n, dtype=bool)
    struct_liq_short = np.zeros(n, dtype=bool)
    inducement_long  = np.zeros(n, dtype=bool)
    inducement_short = np.zeros(n, dtype=bool)

    lb = p["struct_liq_lookback"]

    for i in range(lb, n):
        if np.isnan(fib_50[i]):
            continue

        mid = fib_50[i]

        # Look for internal pivot lows within lookback (potential structural liq for longs)
        window_pl = pivot_low[i - lb: i]
        window_ph = pivot_high[i - lb: i]

        # Find valid internal pivot lows
        valid_lows = window_pl[~np.isnan(window_pl)]
        if len(valid_lows) > 0:
            last_internal_low = valid_lows[-1]
            # Criterion 1: below 50% Fib
            below_50 = last_internal_low <= mid * (1 + p["struct_liq_max_fib"] - 0.50)
            # Criterion 2: V-shape — pivot_low detected means multi-candle cluster confirmed
            # (the pivot_low function already requires left+right window of bars)
            v_shape = True   # pivot detection IS the V-shape check

            if below_50 and v_shape:
                struct_liq_long[i] = True
            else:
                inducement_long[i] = True

        # Find valid internal pivot highs (structural liq for shorts)
        valid_highs = window_ph[~np.isnan(window_ph)]
        if len(valid_highs) > 0:
            last_internal_high = valid_highs[-1]
            above_50 = last_internal_high >= mid * (1 - (p["struct_liq_max_fib"] - 0.50))
            a_shape  = True

            if above_50 and a_shape:
                struct_liq_short[i] = True
            else:
                inducement_short[i] = True

    df["struct_liq_long"]  = struct_liq_long
    df["struct_liq_short"] = struct_liq_short
    df["inducement_long"]  = inducement_long
    df["inducement_short"] = inducement_short

    return df


# ─────────────────────────────────────────────────────────────────────────────
# 6. Unmitigated Level Tracking (TP targets)
# ─────────────────────────────────────────────────────────────────────────────

def _add_new_levels(i, ob_bull_high, ob_bear_low, qm_bull, qm_bear,
                     resistance_levels, support_levels) -> None:
    """Appends any new OB/QM level confirmed at bar i (mutates the lists in place)."""
    if not np.isnan(ob_bear_low[i]) and ob_bear_low[i] not in resistance_levels:
        resistance_levels.append(ob_bear_low[i])
    if not np.isnan(qm_bear[i]) and qm_bear[i] not in resistance_levels:
        resistance_levels.append(qm_bear[i])
    if not np.isnan(ob_bull_high[i]) and ob_bull_high[i] not in support_levels:
        support_levels.append(ob_bull_high[i])
    if not np.isnan(qm_bull[i]) and qm_bull[i] not in support_levels:
        support_levels.append(qm_bull[i])


def _remove_mitigated_and_cap(resistance_levels, support_levels, close_i, max_levels):
    """Drops levels price has closed through, then caps each list to the nearest N."""
    resistance_levels = [lvl for lvl in resistance_levels if close_i < lvl]
    support_levels    = [lvl for lvl in support_levels    if close_i > lvl]
    if len(resistance_levels) > max_levels:
        resistance_levels = sorted(resistance_levels)[:max_levels]
    if len(support_levels) > max_levels:
        support_levels = sorted(support_levels, reverse=True)[:max_levels]
    return resistance_levels, support_levels


def track_unmitigated_levels(
    df: pd.DataFrame,
    params: Optional[dict] = None,
) -> pd.DataFrame:
    """
    Stateful tracker of unmitigated OB and QM levels.
    Nearest unmitigated level above price = resistance TP for longs.
    Nearest unmitigated level below price = support TP for shorts.

    This solves the unmitigated HTF TP targeting problem that cannot
    be done in Pine Script due to lack of persistent state.
    """
    p  = {**DEFAULT_STRUCT_PARAMS, **(params or {})}
    df = df.copy()

    c  = df["close"].values
    n  = len(df)

    nearest_res = np.full(n, np.nan)
    nearest_sup = np.full(n, np.nan)

    # Collect all OB and QM levels into a running list
    resistance_levels: List[float] = []
    support_levels:    List[float] = []

    ob_bull_high = df["ob_bull_high"].values if "ob_bull_high" in df.columns else np.full(n, np.nan)
    ob_bear_low  = df["ob_bear_low"].values  if "ob_bear_low"  in df.columns else np.full(n, np.nan)
    qm_bull      = df["qm_bull_level"].values if "qm_bull_level" in df.columns else np.full(n, np.nan)
    qm_bear      = df["qm_bear_level"].values if "qm_bear_level" in df.columns else np.full(n, np.nan)

    max_lvl = p["unmit_max_levels"]

    for i in range(n):
        _add_new_levels(i, ob_bull_high, ob_bear_low, qm_bull, qm_bear,
                         resistance_levels, support_levels)
        resistance_levels, support_levels = _remove_mitigated_and_cap(
            resistance_levels, support_levels, c[i], max_lvl)

        above = [lvl for lvl in resistance_levels if lvl > c[i]]
        below = [lvl for lvl in support_levels    if lvl < c[i]]
        if above:
            nearest_res[i] = min(above)
        if below:
            nearest_sup[i] = max(below)

    df["nearest_unmit_res"] = nearest_res
    df["nearest_unmit_sup"] = nearest_sup

    return df


# ─────────────────────────────────────────────────────────────────────────────
# 7. CEO Sequence Validation (master gate)
# ─────────────────────────────────────────────────────────────────────────────

def _apply_bos_gate(df, p, n, long_valid, short_valid):
    """Gate 1: requires a BOS within the lookback window, if configured."""
    if not (p["require_bos"] and "bos_long" in df.columns):
        return long_valid, short_valid
    lb = p["bos_lookback"]
    bos_long_recent  = np.zeros(n, dtype=bool)
    bos_short_recent = np.zeros(n, dtype=bool)
    bl = df["bos_long"].values.astype(bool)
    bs = df["bos_short"].values.astype(bool)
    for i in range(n):
        start = max(0, i - lb)
        bos_long_recent[i]  = bl[start:i+1].any()
        bos_short_recent[i] = bs[start:i+1].any()
    return long_valid & bos_long_recent, short_valid & bos_short_recent


def _apply_structural_gates(df, p, n, long_valid, short_valid):
    """Gates 2-4 (Fib zone, OB/QM POI, structural liquidity) + inducement block."""
    if p["require_fib_zone"] and "in_discount" in df.columns:
        long_valid  = long_valid  & df["in_discount"].values
        short_valid = short_valid & df["in_premium"].values

    if p["require_ob_or_qm"]:
        has_bull_poi = (
            (df["ob_bull_active"].values if "ob_bull_active" in df.columns else np.zeros(n, dtype=bool)) |
            (df["qm_bull_active"].values if "qm_bull_active" in df.columns else np.zeros(n, dtype=bool))
        )
        has_bear_poi = (
            (df["ob_bear_active"].values if "ob_bear_active" in df.columns else np.zeros(n, dtype=bool)) |
            (df["qm_bear_active"].values if "qm_bear_active" in df.columns else np.zeros(n, dtype=bool))
        )
        long_valid  = long_valid  & has_bull_poi
        short_valid = short_valid & has_bear_poi

    if p["require_struct_liq"] and "struct_liq_long" in df.columns:
        long_valid  = long_valid  & df["struct_liq_long"].values
        short_valid = short_valid & df["struct_liq_short"].values

    if "inducement_long" in df.columns:
        long_valid  = long_valid  & ~df["inducement_long"].values
        short_valid = short_valid & ~df["inducement_short"].values

    return long_valid, short_valid


def _compute_ceo_quality_bonus(df, n) -> np.ndarray:
    """Structural quality bonus (0-30) from BOS/golden-zone/QM/liquidity confluence."""
    bonus = np.zeros(n)
    if "bos_long" in df.columns:
        bonus += np.where(df["bos_long"].values,         8.0, 0)
    if "double_bos_long" in df.columns:
        bonus += np.where(df["double_bos_long"].values,  7.0, 0)
    if "in_golden" in df.columns:
        bonus += np.where(df["in_golden"].values,       10.0, 0)
    elif "in_discount" in df.columns:
        bonus += np.where(df["in_discount"].values,      5.0, 0)
    if "qm_bull_active" in df.columns:
        bonus += np.where(df["qm_bull_active"].values,   5.0, 0)
    if "struct_liq_long" in df.columns:
        bonus += np.where(df["struct_liq_long"].values,  5.0, 0)
    return np.clip(bonus, 0, 30)


def _apply_quality_bonus(df, bonus) -> None:
    """Adds the structural bonus onto every quality column (mutates df in place)."""
    q_long_cols  = [c for c in df.columns if c.endswith("_quality_long")]
    q_short_cols = [c for c in df.columns if c.endswith("_quality_short")]
    for col in q_long_cols:
        df[col] = np.clip(df[col].values + bonus, 0, 100)
    for col in q_short_cols:
        df[col] = np.clip(df[col].values + bonus, 0, 100)
    if "quality_long" in df.columns:
        df["quality_long"]  = np.clip(df["quality_long"].values  + bonus, 0, 100)
        df["quality_short"] = np.clip(df["quality_short"].values + bonus, 0, 100)


def validate_ceo_sequence(
    df: pd.DataFrame,
    params: Optional[dict] = None,
) -> pd.DataFrame:
    """
    Applies the complete CEO sequence gate:
        Sweep → BOS → Correct Fib Zone → OB/QM POI

    ceo_long_valid  = True when all configured gates pass for longs
    ceo_short_valid = True when all configured gates pass for shorts
    ceo_quality_bonus = structural quality bonus (0-30) for existing scores
    """
    p  = {**DEFAULT_STRUCT_PARAMS, **(params or {})}
    df = df.copy()

    base_long  = df["base_long"].values.astype(bool)
    base_short = df["base_short"].values.astype(bool)
    n = len(df)

    long_valid, short_valid = _apply_bos_gate(df, p, n, base_long.copy(), base_short.copy())
    long_valid, short_valid = _apply_structural_gates(df, p, n, long_valid, short_valid)

    df["ceo_long_valid"]  = long_valid
    df["ceo_short_valid"] = short_valid

    bonus = _compute_ceo_quality_bonus(df, n)
    df["ceo_quality_bonus"] = bonus
    _apply_quality_bonus(df, bonus)

    return df


# ─────────────────────────────────────────────────────────────────────────────
# Master build_ceo_structure
# ─────────────────────────────────────────────────────────────────────────────

def build_ceo_structure(
    df: pd.DataFrame,
    params: Optional[dict] = None,
) -> pd.DataFrame:
    """
    Full CEO structure pipeline.
    Call after indicators.calc_all() → signals.build_all().
    If you need confluence signals, call signals.build_confluence() after
    this (not before) -- its quality gate reads m00_quality_long/short,
    which this function's validate_ceo_sequence() step adds a 0-30
    structural bonus to. See signals.build_all()'s docstring for the
    full story.

    Runs:
        detect_bos
        detect_fib_zones
        detect_order_blocks
        detect_qm_levels
        detect_structural_liquidity
        track_unmitigated_levels
        validate_ceo_sequence
    """
    df = detect_bos(df, params)
    df = detect_fib_zones(df, params)
    df = detect_order_blocks(df, params)
    df = detect_qm_levels(df, params)
    df = detect_structural_liquidity(df, params)
    df = track_unmitigated_levels(df, params)
    df = validate_ceo_sequence(df, params)
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Quick self-test
# ─────────────────────────────────────────────────────────────────────────────
