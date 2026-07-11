"""
The CEO Protocol — Phase 3: Session Filter
===============================================
Provides session-aware filtering for the signal pipeline.
Works as a DataFrame-level filter (adds session columns to df)
AND as a real-time gate (checks a single bar_time).

Separate from the RiskEngine's SessionFilter (which gates live orders).
This module adds session context to the backtest DataFrame so the
backtest accurately reflects session-filtered performance.

Sessions (all times UTC)
------------------------
    Asian      00:00 – 09:00   Low liquidity, wider spreads, avoid
    London     07:00 – 16:00   High liquidity, strong directional moves
    New York   12:00 – 21:00   High liquidity, volatile, trend continuation
    Overlap    12:00 – 16:00   Highest liquidity of the day
    Pre-London 05:00 – 07:00   London setup window — valid for entry prep
    Post-NY    21:00 – 00:00   Avoid — low liquidity, spread widens

Key output columns (added to DataFrame)
-----------------------------------------
    sess_asian          bool  — bar falls in Asian session
    sess_london         bool  — bar falls in London session
    sess_new_york       bool  — bar falls in New York session
    sess_overlap        bool  — bar falls in London/NY overlap
    sess_active         bool  — bar falls in any allowed session
    sess_name           str   — session name string ("London", "Overlap", etc.)
    sess_quality_mult   float — quality multiplier for this session (0.0–1.0)
    sess_spread_risk    str   — "low" / "medium" / "high"
    sess_valid_signal   bool  — signal is valid in this session context

Functions
---------
    add_session_columns(df, allowed, tz)    → enriched DataFrame
    is_valid_session(bar_time, allowed)     → (bool, str)
    session_stats(df)                       → summary of signals per session
"""

import pandas as pd
import numpy as np
from datetime import datetime, time as dtime, timezone
from typing import List, Optional, Tuple


# ─────────────────────────────────────────────────────────────────────────────
# Session definitions (UTC)
# ─────────────────────────────────────────────────────────────────────────────

SESSION_WINDOWS = {
    "asian":      (dtime(0,  0), dtime(9,  0)),
    "pre_london": (dtime(5,  0), dtime(7,  0)),
    "london":     (dtime(7,  0), dtime(16, 0)),
    "overlap":    (dtime(12, 0), dtime(16, 0)),
    "new_york":   (dtime(12, 0), dtime(21, 0)),
    "post_ny":    (dtime(21, 0), dtime(23, 59)),
}

# Canonical session name list — every other module (backtest.py, risk_engine.py)
# imports this instead of keeping its own copy, so the windows can't drift apart.
# asian + london + new_york + post_ny already covers the full 24h on their own;
# pre_london/overlap are informational sub-windows for quality scoring.
ALL_SESSIONS = list(SESSION_WINDOWS.keys())

# Sentinel accepted by build_session_mask() / SessionFilter to mean
# "no session restriction — trade anytime the market is open".
TRADE_ANYTIME = "all"

# Quality multiplier per session — reduces signal quality score during
# low-liquidity windows, boosts during peak liquidity
SESSION_QUALITY_MULT = {
    "asian":      0.50,    # noisy, institutional traps common
    "pre_london": 0.70,    # setup window, not execution
    "london":     1.00,    # full quality
    "overlap":    1.10,    # highest liquidity — slight boost
    "new_york":   1.00,    # full quality
    "post_ny":    0.40,    # avoid
    "weekend":    0.00,    # market closed / gaps
    "none":       0.30,    # outside all sessions
}

# Spread risk per session
SESSION_SPREAD_RISK = {
    "asian":      "high",
    "pre_london": "medium",
    "london":     "low",
    "overlap":    "low",
    "new_york":   "low",
    "post_ny":    "high",
    "weekend":    "high",
    "none":       "high",
}

# Default allowed sessions for execution
DEFAULT_ALLOWED = [TRADE_ANYTIME]


# ─────────────────────────────────────────────────────────────────────────────
# Core session classifier
# ─────────────────────────────────────────────────────────────────────────────

def _classify_closure(weekday: int, t) -> Optional[Tuple[dict, str, float, str]]:
    """
    Special-cases real market closure (weekend, Friday post-20:00 close).
    Returns the full _classify_bar tuple if closed, or None if the market
    is open and the caller should proceed to normal session classification.
    """
    if weekday == 5 or (weekday == 6 and t < dtime(5, 0)):
        return (
            {s: False for s in SESSION_WINDOWS},
            "weekend",
            SESSION_QUALITY_MULT["weekend"],
            SESSION_SPREAD_RISK["weekend"],
        )
    if weekday == 4 and t >= dtime(20, 0):
        return (
            {s: False for s in SESSION_WINDOWS},
            "post_ny",
            SESSION_QUALITY_MULT["post_ny"],
            SESSION_SPREAD_RISK["post_ny"],
        )
    return None


def _determine_primary_session(sessions: dict) -> str:
    """
    Resolves which single session "owns" this bar when multiple windows
    overlap, by priority: overlap > london > new_york > pre_london >
    asian > post_ny > none.
    """
    if sessions.get("overlap"):
        return "overlap"
    if sessions.get("london") and not sessions.get("new_york"):
        return "london"
    if sessions.get("new_york") and not sessions.get("london"):
        return "new_york"
    if sessions.get("london") and sessions.get("new_york"):
        return "overlap"
    if sessions.get("pre_london"):
        return "pre_london"
    if sessions.get("asian"):
        return "asian"
    if sessions.get("post_ny"):
        return "post_ny"
    return "none"


def _classify_bar(dt: datetime) -> Tuple[dict, str, float, str]:
    """
    Classify a single bar datetime into session membership.
    Returns (session_bools, primary_name, quality_mult, spread_risk).
    """
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)

    weekday = dt.weekday()   # 0=Mon … 4=Fri, 5=Sat, 6=Sun
    t       = dt.time()

    closure = _classify_closure(weekday, t)
    if closure is not None:
        return closure

    sessions = {
        name: (start <= t < end)
        for name, (start, end) in SESSION_WINDOWS.items()
    }

    primary     = _determine_primary_session(sessions)
    mult        = SESSION_QUALITY_MULT.get(primary, 0.30)
    spread_risk = SESSION_SPREAD_RISK.get(primary, "high")

    return sessions, primary, mult, spread_risk


# ─────────────────────────────────────────────────────────────────────────────
# DataFrame-level session columns
# ─────────────────────────────────────────────────────────────────────────────

def add_session_columns(
    df:      pd.DataFrame,
    allowed: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Add session classification columns to an OHLCV DataFrame.
    Requires df.index to be a DatetimeTzAware index (UTC).

    Parameters
    ----------
    df      : DataFrame with DatetimeIndex (from fetch_ohlcv)
    allowed : sessions in which signals are valid
              default: ["london", "new_york", "overlap"]

    Returns df with session columns added.

    NOTE: Call this LAST in the pipeline — it multiplies quality scores
    by a session weight. Calling before candle_patterns or ceo_structure
    will cause those modules to then add bonuses on top of already-reduced
    scores, leading to incorrect final values.
    """
    # Pipeline stage guard — warn if called before other enrichment stages
    _EXPECTED_BEFORE_SESSION = ["cp_bull_any", "bos_long", "pat_bull_any"]
    missing = [c for c in _EXPECTED_BEFORE_SESSION if c not in df.columns]
    if missing:
        import warnings
        warnings.warn(
            f"add_session_columns() called before expected pipeline stages. "
            f"Missing columns: {missing}. "
            f"Call order should be: indicators → signals → candle_patterns → "
            f"ceo_structure → patterns → session_filter (last).",
            stacklevel=2,
        )
    if allowed is None:
        allowed = DEFAULT_ALLOWED

    df = df.copy()
    n  = len(df)

    sess_asian      = np.zeros(n, dtype=bool)
    sess_london     = np.zeros(n, dtype=bool)
    sess_new_york   = np.zeros(n, dtype=bool)
    sess_overlap    = np.zeros(n, dtype=bool)
    sess_pre_london = np.zeros(n, dtype=bool)
    sess_post_ny    = np.zeros(n, dtype=bool)
    sess_active     = np.zeros(n, dtype=bool)
    sess_name       = np.empty(n, dtype=object)
    sess_q_mult     = np.zeros(n)
    sess_spread     = np.empty(n, dtype=object)
    sess_valid      = np.zeros(n, dtype=bool)

    index = df.index
    if not hasattr(index, "tz") or index.tz is None:
        index = index.tz_localize("UTC")
    elif str(index.tz) != "UTC":
        index = index.tz_convert("UTC")

    for i, dt in enumerate(index):
        sessions, primary, q_mult, spread_risk = _classify_bar(dt.to_pydatetime())

        sess_asian[i]      = sessions.get("asian",      False)
        sess_london[i]     = sessions.get("london",     False)
        sess_new_york[i]   = sessions.get("new_york",   False)
        sess_overlap[i]    = sessions.get("overlap",    False)
        sess_pre_london[i] = sessions.get("pre_london", False)
        sess_post_ny[i]    = sessions.get("post_ny",    False)
        sess_name[i]       = primary
        sess_q_mult[i]     = q_mult
        sess_spread[i]     = spread_risk

        if TRADE_ANYTIME in allowed:
            # Trade anytime the market is open — only real closure (weekend)
            # blocks a bar; the Friday-close/Sunday-open special cases in
            # _classify_bar() already collapse into "weekend" or "post_ny"
            # appropriately above.
            is_active = primary != "weekend"
        else:
            is_active = primary in allowed
        sess_active[i] = is_active
        sess_valid[i]  = is_active

    df["sess_asian"]       = sess_asian
    df["sess_london"]      = sess_london
    df["sess_new_york"]    = sess_new_york
    df["sess_overlap"]     = sess_overlap
    df["sess_pre_london"]  = sess_pre_london
    df["sess_post_ny"]     = sess_post_ny
    df["sess_active"]      = sess_active
    df["sess_name"]        = sess_name
    df["sess_quality_mult"]= sess_q_mult
    df["sess_spread_risk"] = sess_spread
    df["sess_valid_signal"]= sess_valid

    # Apply quality multiplier to existing quality scores
    q_long_cols  = [c for c in df.columns if c.endswith("_quality_long")]
    q_short_cols = [c for c in df.columns if c.endswith("_quality_short")]

    for col in q_long_cols:
        df[col] = np.clip(df[col].values * sess_q_mult, 0, 100)
    for col in q_short_cols:
        df[col] = np.clip(df[col].values * sess_q_mult, 0, 100)

    if "quality_long" in df.columns:
        df["quality_long"]  = np.clip(df["quality_long"].values  * sess_q_mult, 0, 100)
        df["quality_short"] = np.clip(df["quality_short"].values * sess_q_mult, 0, 100)

    # Block signals outside allowed sessions
    if "base_long" in df.columns:
        df["base_long"]  = df["base_long"]  & df["sess_valid_signal"]
        df["base_short"] = df["base_short"] & df["sess_valid_signal"]

    if "ceo_long_valid" in df.columns:
        df["ceo_long_valid"]  = df["ceo_long_valid"]  & df["sess_valid_signal"]
        df["ceo_short_valid"] = df["ceo_short_valid"] & df["sess_valid_signal"]

    return df


# ─────────────────────────────────────────────────────────────────────────────
# Real-time gate (for live trading)
# ─────────────────────────────────────────────────────────────────────────────

def is_valid_session(
    bar_time: datetime,
    allowed:  Optional[List[str]] = None,
) -> Tuple[bool, str, float]:
    """
    Real-time session check for a single bar.
    Returns (is_valid, session_name, quality_multiplier).

    Use this in mt5_live.py before acting on a signal.
    """
    if allowed is None:
        allowed = DEFAULT_ALLOWED

    _, primary, q_mult, _ = _classify_bar(bar_time)

    if TRADE_ANYTIME in allowed:
        is_valid = primary != "weekend"
    else:
        is_valid = primary in allowed
    return is_valid, primary, q_mult


def session_at(bar_time: datetime) -> dict:
    """Returns full session classification for a datetime."""
    sessions, primary, q_mult, spread_risk = _classify_bar(bar_time)
    return {
        "sessions":     sessions,
        "primary":      primary,
        "quality_mult": q_mult,
        "spread_risk":  spread_risk,
        "is_london":    sessions.get("london", False),
        "is_new_york":  sessions.get("new_york", False),
        "is_overlap":   sessions.get("overlap", False),
        "is_asian":     sessions.get("asian", False),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Session statistics
# ─────────────────────────────────────────────────────────────────────────────

def session_stats(df: pd.DataFrame) -> pd.DataFrame:
    """
    Returns a summary of signal counts and win rates per session.
    Useful for identifying which sessions the CEO method performs best in.

    Requires: sess_name, base_long/base_short columns.
    """
    if "sess_name" not in df.columns:
        raise ValueError("Run add_session_columns() first.")

    rows = []
    for sess in ["asian", "pre_london", "london", "overlap", "new_york", "post_ny"]:
        mask = df["sess_name"] == sess
        total_bars = int(mask.sum())
        if total_bars == 0:
            continue

        long_signals  = int((mask & df["base_long"]).sum())  if "base_long"  in df.columns else 0
        short_signals = int((mask & df["base_short"]).sum()) if "base_short" in df.columns else 0

        rows.append({
            "session":       sess,
            "total_bars":    total_bars,
            "long_signals":  long_signals,
            "short_signals": short_signals,
            "total_signals": long_signals + short_signals,
            "signals_pct":   round((long_signals + short_signals) / total_bars * 100, 1),
            "quality_mult":  SESSION_QUALITY_MULT[sess],
            "spread_risk":   SESSION_SPREAD_RISK[sess],
        })

    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────────────
# Quick self-test
# ─────────────────────────────────────────────────────────────────────────────
