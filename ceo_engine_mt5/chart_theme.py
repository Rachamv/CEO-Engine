"""
The CEO Protocol — Chart Theme & Shared Data Prep
======================================================
The color theme and the small DataFrame-slicing helpers shared by both
chart_png.py (matplotlib) and chart_lwc.py (Lightweight Charts). Split out of the
original monolithic chart.py purely for file-size/maintainability
reasons — no rendering logic lives here. See CHANGELOG.
"""

import numpy as np
import pandas as pd


# ─────────────────────────────────────────────────────────────────────────────
# Shared theme
# ─────────────────────────────────────────────────────────────────────────────

THEME = {
    "bg":           "#0a0e12",
    "panel":        "#0f1519",
    "border":       "#1e2830",
    "text":         "#e8f0f5",
    "muted":        "#5a7080",
    "green":        "#00e676",
    "red":          "#ff3d57",
    "amber":        "#ffb300",
    "purple":       "#9c6fff",
    "teal":         "#00bcd4",
    "blue":         "#2196f3",
    "grid":         "#1a2530",
    "candle_bull":  "#00e676",
    "candle_bear":  "#ff3d57",
    "ema_fast":     "#ffb300",
    "ema_slow":     "#9c6fff",
    "pivot_high":   "#ff3d57",
    "pivot_low":    "#00e676",
    "ob_bull":      "#00e676",
    "ob_bear":      "#ff3d57",
    "qm_bull":      "#00bcd4",
    "qm_bear":      "#ff6090",
    "bos":          "#ffb300",
    "fib":          "#5a7080",
    "fvg_bull":     "#00bcd433",
    "fvg_bear":     "#ff609033",
    "session_london":  "#1a2a1a",
    "session_ny":      "#1a1a2a",
    "session_overlap": "#1a2520",
    "session_asian":   "#0d1015",
    "signal_long":  "#00e676",
    "signal_short": "#ff3d57",
    "cp_bull":      "#00e676",
    "cp_bear":      "#ff3d57",
    "pat_line":     "#9c6fff",
}


# ─────────────────────────────────────────────────────────────────────────────
# Shared data preparation
# ─────────────────────────────────────────────────────────────────────────────

def _prep_slice(df: pd.DataFrame, bars: int) -> pd.DataFrame:
    """Return the last N bars, reset to integer index."""
    sl = df.tail(bars).copy()
    sl = sl.reset_index(drop=False)
    return sl


def _col(df: pd.DataFrame, name: str, default=None):
    """Safe column getter."""
    if name in df.columns:
        return df[name].values
    n = len(df)
    if default is None:
        return np.full(n, np.nan)
    if isinstance(default, bool):
        return np.zeros(n, dtype=bool)
    return np.full(n, default)


# ─────────────────────────────────────────────────────────────────────────────
# ── OPTION A: Static PNG (matplotlib) ────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────────────

