"""
The CEO Protocol — Static PNG Chart Renderer
==================================================
plot_chart_png() and every matplotlib-specific drawing helper it calls
(candles, EMAs, pivots, volume, CEO structure overlays, geometric pattern
trendlines, signal markers). Used for Telegram alert chart attachments.

Split out of the original monolithic chart.py purely for file-size/
maintainability reasons — no rendering logic changed. See CHANGELOG.
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D

from .chart_theme import THEME, _prep_slice, _col


def _setup_png_figure(show_volume: bool):
    """Creates the figure/axes layout (1 or 2 rows depending on show_volume)."""
    height_ratios = [4, 1] if show_volume else [1]
    nrows = 2 if show_volume else 1
    fig = plt.figure(figsize=(20, 10 if show_volume else 8), facecolor=THEME["bg"])
    gs  = gridspec.GridSpec(nrows, 1, hspace=0.04, height_ratios=height_ratios)

    ax  = fig.add_subplot(gs[0])
    axv = fig.add_subplot(gs[1], sharex=ax) if show_volume else None

    for a in ([ax, axv] if axv else [ax]):
        a.set_facecolor(THEME["panel"])
        a.tick_params(colors=THEME["muted"], labelsize=7)
        for sp in a.spines.values():
            sp.set_edgecolor(THEME["border"])
        a.grid(color=THEME["grid"], linewidth=0.4, alpha=0.6, axis="y")

    return fig, ax, axv


def _draw_session_shading_png(ax, sl, n) -> None:
    """Shades the background by session (London/NY/overlap/Asian)."""
    if "sess_name" not in sl.columns:
        return
    sess_colors = {
        "london":    THEME["session_london"],
        "new_york":  THEME["session_ny"],
        "overlap":   THEME["session_overlap"],
        "asian":     THEME["session_asian"],
    }
    prev_sess = None
    seg_start = 0
    for i, sess in enumerate(sl["sess_name"].values):
        if sess != prev_sess or i == n - 1:
            if prev_sess in sess_colors and i > seg_start:
                ax.axvspan(seg_start - 0.5, i - 0.5,
                           color=sess_colors[prev_sess], alpha=1.0, zorder=0)
            seg_start = i
            prev_sess = sess


def _draw_candles_png(ax, x, o, h, lo, c) -> None:
    """Draws candlestick wicks and bodies."""
    width     = 0.6
    half      = width / 2
    bull_mask = c >= o
    bear_mask = ~bull_mask

    for mask, body_col, wick_col in [
        (bull_mask, THEME["candle_bull"], THEME["candle_bull"]),
        (bear_mask, THEME["candle_bear"], THEME["candle_bear"]),
    ]:
        xi = x[mask]
        ax.vlines(xi, lo[mask], h[mask], colors=wick_col, linewidth=0.8, zorder=2)
        bottoms = np.minimum(o[mask], c[mask])
        heights = np.abs(c[mask] - o[mask])
        heights = np.maximum(heights, (h[mask] - lo[mask]) * 0.01)  # min height
        for xi_, bot_, ht_ in zip(xi, bottoms, heights):
            rect = plt.Rectangle((xi_ - half, bot_), width, ht_, color=body_col, zorder=3)
            ax.add_patch(rect)


def _draw_emas_png(ax, sl, x) -> None:
    """Draws EMA fast/slow lines if present and not all-NaN."""
    ema_fast = _col(sl, "ema_fast")
    ema_slow = _col(sl, "ema_slow")
    if not np.all(np.isnan(ema_fast)):
        ax.plot(x, ema_fast, color=THEME["ema_fast"],
                linewidth=1.2, alpha=0.9, zorder=4, label="EMA Fast")
    if not np.all(np.isnan(ema_slow)):
        ax.plot(x, ema_slow, color=THEME["ema_slow"],
                linewidth=1.2, alpha=0.9, zorder=4, label="EMA Slow")


def _draw_pivot_markers_png(ax, sl, x, h, lo) -> None:
    """Draws pivot-high/pivot-low triangle markers."""
    ph = _col(sl, "pivot_high")
    pl = _col(sl, "pivot_low")
    ph_mask = ~np.isnan(ph)
    pl_mask = ~np.isnan(pl)
    ax.scatter(x[ph_mask], h[ph_mask] + _atr_frac(sl, 0.3)[ph_mask],
               marker="v", color=THEME["pivot_high"], s=30, zorder=5, alpha=0.8)
    ax.scatter(x[pl_mask], lo[pl_mask] - _atr_frac(sl, 0.3)[pl_mask],
               marker="^", color=THEME["pivot_low"],  s=30, zorder=5, alpha=0.8)


def _draw_volume_png(axv, sl, x, o, c, n) -> None:
    """Draws the volume subplot if present."""
    if axv is None or "volume" not in sl.columns:
        return
    vol = sl["volume"].values
    vol_colors = [THEME["candle_bull"] if c[i] >= o[i]
                  else THEME["candle_bear"] for i in range(n)]
    axv.bar(x, vol, color=vol_colors, alpha=0.6, width=0.8, zorder=2)
    axv.set_ylabel("Volume", color=THEME["muted"], fontsize=7)


def _finalize_png_chart(fig, ax, sl, symbol, tf, c) -> None:
    """Adds the title (with last-signal flag) and legend."""
    last_c   = c[-1]
    last_sig = ""
    if "base_long" in sl.columns and sl["base_long"].iloc[-1]:
        last_sig = "  ▲ LONG SIGNAL"
    elif "base_short" in sl.columns and sl["base_short"].iloc[-1]:
        last_sig = "  ▼ SHORT SIGNAL"

    fig.suptitle(
        f"CEO Engine  |  {symbol}  {tf}  |  {last_c:.2f}{last_sig}",
        color=THEME["text"], fontsize=12, fontweight="bold",
        x=0.02, ha="left",
    )

    legend_elements = [
        Line2D([0],[0], color=THEME["ema_fast"],  linewidth=1.5, label="EMA Fast"),
        Line2D([0],[0], color=THEME["ema_slow"],  linewidth=1.5, label="EMA Slow"),
        Line2D([0],[0], marker="^", color=THEME["pivot_low"],  markersize=6,
               linestyle="None", label="Pivot Low"),
        Line2D([0],[0], marker="v", color=THEME["pivot_high"], markersize=6,
               linestyle="None", label="Pivot High"),
    ]
    ax.legend(handles=legend_elements, loc="upper left",
              facecolor=THEME["panel"], edgecolor=THEME["border"],
              labelcolor=THEME["muted"], fontsize=7, framealpha=0.8)


def plot_chart_png(
    df:          pd.DataFrame,
    symbol:      str  = "XAUUSD",
    tf:          str  = "M15",
    bars:        int  = 150,
    save_path:   str  = "ceo_chart.png",
    show_volume: bool = True,
    show_struct: bool = True,
    show_patterns: bool = True,
    show_sessions: bool = True,
    dpi:         int  = 150,
) -> str:
    """
    Render a static candlestick PNG chart with all overlays.
    Returns save_path on completion.
    """
    sl = _prep_slice(df, bars)
    n  = len(sl)
    x  = np.arange(n)

    o = sl["open"].values
    h = sl["high"].values
    lo = sl["low"].values
    c = sl["close"].values

    fig, ax, axv = _setup_png_figure(show_volume)

    if show_sessions:
        _draw_session_shading_png(ax, sl, n)

    _draw_candles_png(ax, x, o, h, lo, c)
    _draw_emas_png(ax, sl, x)
    _draw_pivot_markers_png(ax, sl, x, h, lo)

    if show_struct:
        _draw_structure_png(ax, sl, x, h, lo, c)
    if show_patterns:
        _draw_patterns_png(ax, sl, x, h, lo)

    _draw_signals_png(ax, sl, x, h, lo, c)
    _draw_volume_png(axv, sl, x, o, c, n)

    _set_time_labels(ax, sl, x, n)
    if axv:
        plt.setp(ax.get_xticklabels(), visible=False)

    ax.set_ylabel("Price", color=THEME["muted"], fontsize=8)
    ax.yaxis.set_label_position("right")
    ax.yaxis.tick_right()

    _finalize_png_chart(fig, ax, sl, symbol, tf, c)

    plt.savefig(save_path, dpi=dpi, bbox_inches="tight", facecolor=THEME["bg"])
    plt.close(fig)
    return save_path


# ── PNG overlay helpers ───────────────────────────────────────────────────────

def _atr_frac(sl: pd.DataFrame, frac: float) -> np.ndarray:
    atr = _col(sl, "atr")
    atr = np.where(np.isnan(atr), np.nanmean(atr) if not np.all(np.isnan(atr)) else 1.0, atr)
    return atr * frac


def _draw_structure_png(ax, sl, x, h, lo, c):
    n   = len(sl)
    _atr_frac(sl, 1.0)

    # OB zones
    ob_bull_active = _col(sl, "ob_bull_active", False).astype(bool)
    ob_bull_high   = _col(sl, "ob_bull_high")
    ob_bull_low    = _col(sl, "ob_bull_low")
    ob_bear_active = _col(sl, "ob_bear_active", False).astype(bool)
    ob_bear_high   = _col(sl, "ob_bear_high")
    ob_bear_low    = _col(sl, "ob_bear_low")

    drawn_bull_ob = set()
    drawn_bear_ob = set()
    for i in range(n):
        if ob_bull_active[i] and not np.isnan(ob_bull_high[i]):
            key = round(ob_bull_high[i], 1)
            if key not in drawn_bull_ob:
                ax.axhspan(ob_bull_low[i], ob_bull_high[i],
                           xmin=i/n, xmax=1.0,
                           color=THEME["ob_bull"], alpha=0.08, zorder=1)
                ax.axhline(ob_bull_high[i], color=THEME["ob_bull"],
                           linewidth=0.6, linestyle="--", alpha=0.5, zorder=1)
                ax.text(x[-1], ob_bull_high[i], " OB",
                        color=THEME["ob_bull"], fontsize=6, va="bottom",
                        ha="left", zorder=6)
                drawn_bull_ob.add(key)

        if ob_bear_active[i] and not np.isnan(ob_bear_high[i]):
            key = round(ob_bear_high[i], 1)
            if key not in drawn_bear_ob:
                ax.axhspan(ob_bear_low[i], ob_bear_high[i],
                           xmin=i/n, xmax=1.0,
                           color=THEME["ob_bear"], alpha=0.08, zorder=1)
                ax.axhline(ob_bear_low[i], color=THEME["ob_bear"],
                           linewidth=0.6, linestyle="--", alpha=0.5, zorder=1)
                ax.text(x[-1], ob_bear_low[i], " OB",
                        color=THEME["ob_bear"], fontsize=6, va="top",
                        ha="left", zorder=6)
                drawn_bear_ob.add(key)

    # QM levels
    qm_bull = _col(sl, "qm_bull_level")
    qm_bear = _col(sl, "qm_bear_level")
    drawn_qm = set()
    for i in range(n):
        if not np.isnan(qm_bull[i]):
            key = round(qm_bull[i], 1)
            if key not in drawn_qm:
                ax.axhline(qm_bull[i], color=THEME["qm_bull"],
                           linewidth=0.8, linestyle=":", alpha=0.7, zorder=1)
                ax.text(x[-1], qm_bull[i], " QM",
                        color=THEME["qm_bull"], fontsize=6, va="bottom", zorder=6)
                drawn_qm.add(key)
        if not np.isnan(qm_bear[i]):
            key = round(qm_bear[i], 1)
            if key not in drawn_qm:
                ax.axhline(qm_bear[i], color=THEME["qm_bear"],
                           linewidth=0.8, linestyle=":", alpha=0.7, zorder=1)
                ax.text(x[-1], qm_bear[i], " QM",
                        color=THEME["qm_bear"], fontsize=6, va="top", zorder=6)
                drawn_qm.add(key)

    # BOS markers — position at actual price extremes, not axis limits
    # (get_ylim() before full layout is unreliable and the truthy-check
    # on a float is always True, so the h.max()/lo.min() fallback never fired)
    bos_long  = _col(sl, "bos_long",  False).astype(bool)
    bos_short = _col(sl, "bos_short", False).astype(bool)
    chart_top    = np.nanmax(h)
    chart_bottom = np.nanmin(lo)
    pad = (chart_top - chart_bottom) * 0.02

    for i in x[bos_long]:
        ax.axvline(i, color=THEME["bos"], linewidth=0.6,
                   linestyle="--", alpha=0.5, zorder=1)
        ax.text(i, chart_top + pad,
                "BOS↑", color=THEME["bos"], fontsize=5,
                ha="center", va="bottom", zorder=6)
    for i in x[bos_short]:
        ax.axvline(i, color=THEME["bos"], linewidth=0.6,
                   linestyle="--", alpha=0.5, zorder=1)
        ax.text(i, chart_bottom - pad,
                "BOS↓", color=THEME["bos"], fontsize=5,
                ha="center", va="top", zorder=6)

    # Fibonacci 50% level
    fib_50 = _col(sl, "fib_50")
    last_fib = fib_50[~np.isnan(fib_50)]
    if len(last_fib) > 0:
        fv = last_fib[-1]
        ax.axhline(fv, color=THEME["fib"], linewidth=0.8,
                   linestyle="-.", alpha=0.6, zorder=1)
        ax.text(x[0], fv, "50% ", color=THEME["fib"],
                fontsize=6, va="bottom", ha="right", zorder=6)


def _draw_patterns_png(ax, sl, x, h, lo):
    """Draw geometric pattern trendlines."""
    # Triangles and wedges — draw converging trendlines from last 2 pivot highs/lows
    ph = _col(sl, "pivot_high")
    pl = _col(sl, "pivot_low")

    pat_cols_draw = {
        "pat_asc_triangle":  THEME["pat_line"],
        "pat_desc_triangle": THEME["pat_line"],
        "pat_sym_triangle":  THEME["pat_line"],
        "pat_rising_wedge":  THEME["red"],
        "pat_falling_wedge": THEME["green"],
    }

    for pat_col, color in pat_cols_draw.items():
        pat = _col(sl, pat_col, False).astype(bool)
        if not pat.any():
            continue

        # Find last active bar of this pattern
        last_i = np.where(pat)[0][-1]

        # Get last 2 pivot highs and lows before that bar
        ph_idx = [i for i in range(last_i + 1) if not np.isnan(ph[i])][-2:]
        pl_idx = [i for i in range(last_i + 1) if not np.isnan(pl[i])][-2:]

        if len(ph_idx) == 2:
            ax.plot([ph_idx[0], ph_idx[1]], [h[ph_idx[0]], h[ph_idx[1]]],
                    color=color, linewidth=1.0, linestyle="--",
                    alpha=0.7, zorder=4)
        if len(pl_idx) == 2:
            ax.plot([pl_idx[0], pl_idx[1]], [lo[pl_idx[0]], lo[pl_idx[1]]],
                    color=color, linewidth=1.0, linestyle="--",
                    alpha=0.7, zorder=4)

    # Double top / bottom — horizontal level
    for col, lvl_arr, color, label in [
        ("pat_double_top",    h, THEME["red"],   "DT"),
        ("pat_double_bottom", lo, THEME["green"], "DB"),
        ("pat_triple_top",    h, THEME["red"],   "3T"),
        ("pat_triple_bottom", lo, THEME["green"], "3B"),
    ]:
        pat = _col(sl, col, False).astype(bool)
        if not pat.any():
            continue
        last_i = np.where(pat)[0][-1]
        level  = lvl_arr[last_i]
        ax.axhline(level, color=color, linewidth=0.8,
                   linestyle="--", alpha=0.6, zorder=1)
        ax.text(x[-1], level, f" {label}",
                color=color, fontsize=6, va="bottom", zorder=6)

    # H&S neckline
    for col, color, label in [
        ("pat_hs",  THEME["red"],   "H&S"),
        ("pat_ihs", THEME["green"], "IHS"),
    ]:
        pat = _col(sl, col, False).astype(bool)
        if not pat.any():
            continue
        last_i = np.where(pat)[0][-1]
        pl_idx = [i for i in range(last_i + 1) if not np.isnan(pl[i])]
        if len(pl_idx) >= 2:
            i1, i2 = pl_idx[-2], pl_idx[-1]
            ax.plot([i1, last_i], [lo[i1], lo[i2]],
                    color=color, linewidth=1.2, alpha=0.8, zorder=4)
            ax.text(last_i, lo[last_i], f" {label}",
                    color=color, fontsize=7, fontweight="bold", va="top", zorder=6)


def _draw_signals_png(ax, sl, x, h, lo, c):
    """Draw entry signal markers and SL/TP lines."""
    atr = _atr_frac(sl, 1.0)

    base_long  = _col(sl, "base_long",  False).astype(bool)
    base_short = _col(sl, "base_short", False).astype(bool)
    cp_bull    = _col(sl, "cp_bull_confirmation", False).astype(bool)
    cp_bear    = _col(sl, "cp_bear_confirmation", False).astype(bool)

    # Sweep signals
    for i in np.where(base_long)[0]:
        ax.scatter(x[i], lo[i] - atr[i] * 0.5,
                   marker="^", color=THEME["signal_long"],
                   s=80, zorder=7, linewidths=0)
    for i in np.where(base_short)[0]:
        ax.scatter(x[i], h[i] + atr[i] * 0.5,
                   marker="v", color=THEME["signal_short"],
                   s=80, zorder=7, linewidths=0)

    # Candle pattern confirmations — smaller markers
    for i in np.where(cp_bull)[0]:
        ax.scatter(x[i], lo[i] - atr[i] * 1.0,
                   marker="^", color=THEME["cp_bull"],
                   s=30, zorder=7, alpha=0.7, linewidths=0)
    for i in np.where(cp_bear)[0]:
        ax.scatter(x[i], h[i] + atr[i] * 1.0,
                   marker="v", color=THEME["cp_bear"],
                   s=30, zorder=7, alpha=0.7, linewidths=0)

    # Pattern name labels
    if "pat_name" in sl.columns:
        for i, name in enumerate(sl["pat_name"].values):
            if name and i < len(x):
                ax.text(x[i], h[i] + atr[i] * 0.8, name,
                        color=THEME["pat_line"], fontsize=5,
                        rotation=45, ha="left", va="bottom", zorder=6, alpha=0.8)


def _set_time_labels(ax, sl, x, n):
    """Set readable datetime labels on x-axis."""
    step  = max(1, n // 10)
    ticks = x[::step]

    labels = []
    if "datetime" in sl.columns:
        for i in range(0, n, step):
            dt = sl["datetime"].iloc[i]
            if hasattr(dt, "strftime"):
                labels.append(dt.strftime("%m/%d %H:%M"))
            else:
                labels.append(str(dt)[:13])
    else:
        labels = [str(i) for i in ticks]

    ax.set_xticks(ticks)
    ax.set_xticklabels(labels, rotation=30, ha="right",
                       color=THEME["muted"], fontsize=6)


