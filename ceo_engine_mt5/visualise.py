"""
The CEO Protocol — Layer 5: Visualisation
==============================================
Produces all charts from backtest results.

Charts
------
1. Model Comparison Bar Chart  — Net R per model, colour-coded win/loss
2. Equity Curves               — cumulative R over time, all models
3. Win Rate vs Profit Factor   — scatter, bubble = trade count
4. Drawdown Chart              — max drawdown per model (top models only)
5. Trade Distribution          — R result histogram per model
6. Signal Quality Heatmap      — regime × model win rate matrix
7. Summary Dashboard           — single-page overview combining key panels

All charts saved as PNG. Call plot_all() to generate everything.
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import warnings
warnings.filterwarnings("ignore")
from typing import Dict, Optional
from .backtest import results_table, equity_curve


# ─────────────────────────────────────────────────────────────────────────────
# Theme
# ─────────────────────────────────────────────────────────────────────────────

THEME = {
    "bg":        "#0a0e12",
    "panel":     "#0f1519",
    "border":    "#1e2830",
    "text":      "#e8f0f5",
    "muted":     "#5a7080",
    "green":     "#00e676",
    "red":       "#ff3d57",
    "amber":     "#ffb300",
    "purple":    "#9c6fff",
    "teal":      "#00bcd4",
    "grid":      "#1a2530",
}

SHORT_NAMES = {
    "LQ":                           "LQ",
    "LQ + Trend":                   "LQ+T",
    "LQ + FVG":                     "LQ+F",
    "LQ + Volume":                  "LQ+V",
    "LQ + RSI":                     "LQ+R",
    "LQ + Displacement":            "LQ+D",
    "LQ + Trend + FVG":             "LQ+TF",
    "LQ + Trend + Volume":          "LQ+TV",
    "LQ + Trend + RSI":             "LQ+TR",
    "LQ + Trend + Displacement":    "LQ+TD",
    "LQ + FVG + Volume":            "LQ+FV",
    "LQ + FVG + RSI":               "LQ+FR",
    "LQ + FVG + Displacement":      "LQ+FD",
    "LQ + Volume + RSI":            "LQ+VR",
    "LQ + Trend + FVG + Volume":    "LQ+TFV",
    "LQ + All Filters":             "LQ+ALL",
    "Confluence":                   "CONF",
}

def _apply_theme(fig, axes=None):
    fig.patch.set_facecolor(THEME["bg"])
    if axes is None:
        return
    for ax in (axes if hasattr(axes, "__iter__") else [axes]):
        ax.set_facecolor(THEME["panel"])
        ax.tick_params(colors=THEME["muted"], labelsize=8)
        ax.xaxis.label.set_color(THEME["muted"])
        ax.yaxis.label.set_color(THEME["muted"])
        for spine in ax.spines.values():
            spine.set_edgecolor(THEME["border"])
        ax.grid(color=THEME["grid"], linewidth=0.5, alpha=0.7)


def _bar_color(val):
    if pd.isna(val) or val == 0:
        return THEME["muted"]
    return THEME["green"] if val > 0 else THEME["red"]


# ─────────────────────────────────────────────────────────────────────────────
# 1. Model Comparison Bar Chart
# ─────────────────────────────────────────────────────────────────────────────

def plot_model_comparison(tbl: pd.DataFrame, save_path: str = "chart_comparison.png"):
    """Net R bar chart for all models."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    _apply_theme(fig, axes)
    fig.suptitle("CEO Engine — Model Comparison", color=THEME["text"],
                 fontsize=14, fontweight="bold", y=1.01)

    # Left: Net R
    ax = axes[0]
    labels = [SHORT_NAMES.get(n, n) for n in tbl.index]
    net_r  = tbl["Net R"].fillna(0).values
    colors = [_bar_color(v) for v in net_r]
    bars   = ax.barh(labels, net_r, color=colors, alpha=0.85, height=0.65)
    ax.axvline(0, color=THEME["muted"], linewidth=1, linestyle="--")
    ax.set_xlabel("Net R")
    ax.set_title("Net R per Model", color=THEME["text"], fontsize=11)
    for bar, val in zip(bars, net_r):
        if not np.isnan(val):
            ax.text(val + (0.05 if val >= 0 else -0.05), bar.get_y() + bar.get_height()/2,
                    f"{val:+.2f}R", va="center",
                    ha="left" if val >= 0 else "right",
                    color=THEME["text"], fontsize=7)

    # Right: Win Rate
    ax2 = axes[1]
    wr     = tbl["Win Rate"].fillna(0).values
    colors2 = [THEME["green"] if w >= 50 else THEME["amber"] if w >= 40
               else THEME["red"] for w in wr]
    ax2.barh(labels, wr, color=colors2, alpha=0.85, height=0.65)
    ax2.axvline(50, color=THEME["muted"], linewidth=1, linestyle="--")
    ax2.set_xlabel("Win Rate %")
    ax2.set_title("Win Rate per Model", color=THEME["text"], fontsize=11)
    for i, (w, t) in enumerate(zip(wr, tbl["Trades"].fillna(0).values)):
        ax2.text(max(w + 0.5, 1), i, f"{w:.1f}%  ({int(t)}T)",
                 va="center", color=THEME["text"], fontsize=7)

    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight",
                facecolor=THEME["bg"])
    plt.close(fig)
    print(f"  ✅  Saved: {save_path}")


# ─────────────────────────────────────────────────────────────────────────────
# 2. Equity Curves
# ─────────────────────────────────────────────────────────────────────────────

def plot_equity_curves(
    bt: Dict[str, pd.DataFrame],
    min_trades: int = 5,
    save_path: str = "chart_equity.png",
):
    """Cumulative R equity curves for all models with enough trades."""
    eligible = {k: v for k, v in bt.items()
                if not v.empty and len(v) >= min_trades}

    if not eligible:
        print("  ⚠️   No models with enough trades for equity chart.")
        return

    fig, ax = plt.subplots(figsize=(16, 7))
    _apply_theme(fig, ax)
    fig.suptitle("CEO Engine — Equity Curves (Cumulative R)",
                 color=THEME["text"], fontsize=14, fontweight="bold")

    palette = [THEME["green"], THEME["teal"], THEME["purple"], THEME["amber"],
               "#4fc3f7", "#f06292", "#aed581", "#ff8a65",
               "#80cbc4", "#ce93d8", "#ffcc02", "#81d4fa"]

    best_net = {k: v["r_result"].sum() for k, v in eligible.items()}
    sorted_models = sorted(eligible.keys(), key=lambda k: best_net[k], reverse=True)

    for i, name in enumerate(sorted_models):
        trades  = eligible[name]
        curve   = equity_curve(trades)
        color   = palette[i % len(palette)]
        lw      = 2.5 if i == 0 else 1.2
        alpha   = 1.0 if i == 0 else 0.55
        label   = f"{SHORT_NAMES.get(name, name)}  ({best_net[name]:+.1f}R)"
        ax.plot(curve.index, curve.values, color=color,
                linewidth=lw, alpha=alpha, label=label)

    ax.axhline(0, color=THEME["muted"], linewidth=1, linestyle="--", alpha=0.6)
    ax.fill_between(curve.index, 0, 0, alpha=0)   # dummy for spacing
    ax.set_xlabel("Date")
    ax.set_ylabel("Cumulative R")
    ax.legend(loc="upper left", fontsize=7, framealpha=0.3,
              facecolor=THEME["panel"], edgecolor=THEME["border"],
              labelcolor=THEME["text"])
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:+.1f}R"))

    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight",
                facecolor=THEME["bg"])
    plt.close(fig)
    print(f"  ✅  Saved: {save_path}")


# ─────────────────────────────────────────────────────────────────────────────
# 3. Win Rate vs Profit Factor Scatter
# ─────────────────────────────────────────────────────────────────────────────

def plot_wr_vs_pf(
    tbl: pd.DataFrame,
    min_trades: int = 3,
    save_path: str = "chart_wr_pf.png",
):
    fig, ax = plt.subplots(figsize=(10, 7))
    _apply_theme(fig, ax)
    fig.suptitle("CEO Engine — Win Rate vs Profit Factor",
                 color=THEME["text"], fontsize=14, fontweight="bold")

    sub = tbl[(tbl["Trades"] >= min_trades) & tbl["Win Rate"].notna()
              & tbl["Profit Factor"].notna()].copy()
    sub["PF_capped"] = sub["Profit Factor"].clip(upper=10.0)

    for name, row in sub.iterrows():
        x      = row["Win Rate"]
        y      = row["PF_capped"]
        size   = max(row["Trades"] * 8, 40)
        color  = THEME["green"] if row["Net R"] > 0 else THEME["red"]
        ax.scatter(x, y, s=size, color=color, alpha=0.75, zorder=3)
        ax.annotate(SHORT_NAMES.get(name, name),
                    (x, y), textcoords="offset points",
                    xytext=(6, 4), fontsize=7.5, color=THEME["text"])

    ax.axvline(50, color=THEME["muted"], linewidth=1, linestyle="--", alpha=0.5)
    ax.axhline(1.0, color=THEME["muted"], linewidth=1, linestyle="--", alpha=0.5)
    ax.set_xlabel("Win Rate %")
    ax.set_ylabel("Profit Factor (capped at 10)")
    ax.text(50.5, ax.get_ylim()[1] * 0.97, "50% WR", color=THEME["muted"],
            fontsize=8, va="top")
    ax.text(sub["Win Rate"].min(), 1.02, "PF = 1.0", color=THEME["muted"],
            fontsize=8)

    # Quadrant shading
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    ax.fill_betweenx([1.0, ylim[1]], 50, xlim[1],
                     color=THEME["green"], alpha=0.04)
    ax.fill_betweenx([ylim[0], 1.0], xlim[0], 50,
                     color=THEME["red"], alpha=0.04)

    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight",
                facecolor=THEME["bg"])
    plt.close(fig)
    print(f"  ✅  Saved: {save_path}")


# ─────────────────────────────────────────────────────────────────────────────
# 4. Drawdown Chart
# ─────────────────────────────────────────────────────────────────────────────

def _max_drawdown(trades: pd.DataFrame) -> pd.Series:
    """Returns drawdown series (negative values)."""
    curve  = trades["r_result"].cumsum()
    peak   = curve.cummax()
    return curve - peak


def plot_drawdown(
    bt: Dict[str, pd.DataFrame],
    top_n: int = 6,
    min_trades: int = 5,
    save_path: str = "chart_drawdown.png",
):
    eligible = {k: v for k, v in bt.items()
                if not v.empty and len(v) >= min_trades}
    if not eligible:
        print("  ⚠️   No models with enough trades for drawdown chart.")
        return

    # Pick top N by Net R
    sorted_models = sorted(eligible.keys(),
                           key=lambda k: eligible[k]["r_result"].sum(),
                           reverse=True)[:top_n]

    fig, ax = plt.subplots(figsize=(16, 5))
    _apply_theme(fig, ax)
    fig.suptitle(f"CEO Engine — Drawdown (Top {top_n} Models by Net R)",
                 color=THEME["text"], fontsize=14, fontweight="bold")

    palette = [THEME["green"], THEME["teal"], THEME["purple"],
               THEME["amber"], "#4fc3f7", "#f06292"]

    for i, name in enumerate(sorted_models):
        trades = eligible[name]
        dd     = _max_drawdown(trades)
        color  = palette[i % len(palette)]
        label  = f"{SHORT_NAMES.get(name, name)}  (MDD {dd.min():.2f}R)"
        ax.fill_between(range(len(dd)), dd.values, 0,
                        color=color, alpha=0.25)
        ax.plot(dd.values, color=color, linewidth=1.5,
                alpha=0.85, label=label)

    ax.axhline(0, color=THEME["muted"], linewidth=1, linestyle="--")
    ax.set_xlabel("Trade #")
    ax.set_ylabel("Drawdown (R)")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.1f}R"))
    ax.legend(loc="lower left", fontsize=8, framealpha=0.3,
              facecolor=THEME["panel"], edgecolor=THEME["border"],
              labelcolor=THEME["text"])

    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight",
                facecolor=THEME["bg"])
    plt.close(fig)
    print(f"  ✅  Saved: {save_path}")


# ─────────────────────────────────────────────────────────────────────────────
# 5. Trade Distribution Histogram
# ─────────────────────────────────────────────────────────────────────────────

def plot_trade_distribution(
    bt: Dict[str, pd.DataFrame],
    models: Optional[list] = None,
    save_path: str = "chart_distribution.png",
):
    """R result histogram for selected models."""
    if models is None:
        # Pick top 6 by trade count
        models = sorted(
            [k for k, v in bt.items() if not v.empty],
            key=lambda k: len(bt[k]), reverse=True
        )[:6]

    n_models = len(models)
    if n_models == 0:
        print("  ⚠️   No models with trades for distribution chart.")
        return

    cols = min(3, n_models)
    rows = (n_models + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 5, rows * 4))
    if n_models == 1:
        axes = np.array([[axes]])
    elif rows == 1:
        axes = axes.reshape(1, -1)
    _apply_theme(fig, [ax for row in axes for ax in row])
    fig.suptitle("CEO Engine — R Result Distribution",
                 color=THEME["text"], fontsize=14, fontweight="bold")

    for idx, name in enumerate(models):
        r   = idx // cols
        c   = idx % cols
        ax  = axes[r][c]
        trades = bt[name]
        if trades.empty:
            ax.set_visible(False)
            continue
        results = trades["r_result"].values
        bins    = np.linspace(results.min() - 0.1, results.max() + 0.1, 25)
        wins    = results[results > 0]
        losses  = results[results <= 0]
        ax.hist(losses, bins=bins, color=THEME["red"],   alpha=0.75, label="Loss")
        ax.hist(wins,   bins=bins, color=THEME["green"], alpha=0.75, label="Win")
        ax.axvline(0,            color=THEME["muted"], linewidth=1, linestyle="--")
        ax.axvline(results.mean(), color=THEME["amber"], linewidth=1.5,
                   linestyle="-", label=f"Avg {results.mean():+.2f}R")
        ax.set_title(SHORT_NAMES.get(name, name), color=THEME["text"], fontsize=10)
        ax.set_xlabel("R Result")
        ax.set_ylabel("Count")
        ax.legend(fontsize=7, framealpha=0.3, facecolor=THEME["panel"],
                  edgecolor=THEME["border"], labelcolor=THEME["text"])

    # Hide unused subplots
    for idx in range(n_models, rows * cols):
        axes[idx // cols][idx % cols].set_visible(False)

    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight",
                facecolor=THEME["bg"])
    plt.close(fig)
    print(f"  ✅  Saved: {save_path}")


# ─────────────────────────────────────────────────────────────────────────────
# 6. Summary Dashboard
# ─────────────────────────────────────────────────────────────────────────────

def plot_summary_dashboard(
    bt: Dict[str, pd.DataFrame],
    tbl: pd.DataFrame,
    symbol: str = "Instrument",
    save_path: str = "chart_dashboard.png",
):
    """
    Single-page overview:
    Top row    : Net R bar chart + Win Rate bar chart
    Middle row : Equity curves (top 8 models)
    Bottom row : WR vs PF scatter + Drawdown (top 5)
    """
    fig = plt.figure(figsize=(20, 14))
    fig.patch.set_facecolor(THEME["bg"])
    gs  = gridspec.GridSpec(3, 2, figure=fig, hspace=0.42, wspace=0.28)

    ax_netR  = fig.add_subplot(gs[0, 0])
    ax_wr    = fig.add_subplot(gs[0, 1])
    ax_eq    = fig.add_subplot(gs[1, :])
    ax_scat  = fig.add_subplot(gs[2, 0])
    ax_dd    = fig.add_subplot(gs[2, 1])

    all_axes = [ax_netR, ax_wr, ax_eq, ax_scat, ax_dd]
    _apply_theme(fig, all_axes)

    fig.suptitle(f"The CEO Protocol — Full Backtest Report  |  {symbol}",
                 color=THEME["text"], fontsize=15, fontweight="bold", y=1.01)

    labels = [SHORT_NAMES.get(n, n) for n in tbl.index]

    # ── Net R bars ────────────────────────────────────────────────────────────
    net_r  = tbl["Net R"].fillna(0).values
    colors = [_bar_color(v) for v in net_r]
    ax_netR.barh(labels, net_r, color=colors, alpha=0.85, height=0.65)
    ax_netR.axvline(0, color=THEME["muted"], linewidth=1, linestyle="--")
    ax_netR.set_xlabel("Net R")
    ax_netR.set_title("Net R per Model", color=THEME["text"], fontsize=10)

    # ── Win Rate bars ─────────────────────────────────────────────────────────
    wr      = tbl["Win Rate"].fillna(0).values
    colors2 = [THEME["green"] if w >= 50 else THEME["amber"] if w >= 40
               else THEME["red"] for w in wr]
    ax_wr.barh(labels, wr, color=colors2, alpha=0.85, height=0.65)
    ax_wr.axvline(50, color=THEME["muted"], linewidth=1, linestyle="--")
    ax_wr.set_xlabel("Win Rate %")
    ax_wr.set_title("Win Rate per Model", color=THEME["text"], fontsize=10)

    # ── Equity curves ─────────────────────────────────────────────────────────
    eligible = {k: v for k, v in bt.items() if not v.empty and len(v) >= 3}
    sorted_m = sorted(eligible.keys(),
                      key=lambda k: eligible[k]["r_result"].sum(),
                      reverse=True)[:8]
    palette  = [THEME["green"], THEME["teal"], THEME["purple"], THEME["amber"],
                "#4fc3f7", "#f06292", "#aed581", "#ff8a65"]

    for i, name in enumerate(sorted_m):
        trades = eligible[name]
        curve  = equity_curve(trades)
        nr     = trades["r_result"].sum()
        lw     = 2.2 if i == 0 else 1.1
        alpha  = 1.0 if i == 0 else 0.5
        ax_eq.plot(curve.index, curve.values,
                   color=palette[i % len(palette)], linewidth=lw,
                   alpha=alpha, label=f"{SHORT_NAMES.get(name, name)} ({nr:+.1f}R)")

    ax_eq.axhline(0, color=THEME["muted"], linewidth=1, linestyle="--", alpha=0.5)
    ax_eq.set_ylabel("Cumulative R")
    ax_eq.set_title("Equity Curves — Top 8 Models", color=THEME["text"], fontsize=10)
    ax_eq.legend(loc="upper left", fontsize=7, ncol=4, framealpha=0.3,
                 facecolor=THEME["panel"], edgecolor=THEME["border"],
                 labelcolor=THEME["text"])
    ax_eq.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:+.1f}R"))

    # ── WR vs PF scatter ──────────────────────────────────────────────────────
    sub = tbl[(tbl["Trades"] >= 3) & tbl["Win Rate"].notna()
              & tbl["Profit Factor"].notna()].copy()
    sub["PF_capped"] = sub["Profit Factor"].clip(upper=8.0)
    for name, row in sub.iterrows():
        x     = row["Win Rate"]
        y     = row["PF_capped"]
        size  = max(row["Trades"] * 10, 40)
        color = THEME["green"] if row["Net R"] > 0 else THEME["red"]
        ax_scat.scatter(x, y, s=size, color=color, alpha=0.75, zorder=3)
        ax_scat.annotate(SHORT_NAMES.get(name, name), (x, y),
                         textcoords="offset points", xytext=(5, 3),
                         fontsize=7, color=THEME["text"])
    ax_scat.axvline(50,  color=THEME["muted"], linewidth=1, linestyle="--", alpha=0.4)
    ax_scat.axhline(1.0, color=THEME["muted"], linewidth=1, linestyle="--", alpha=0.4)
    ax_scat.set_xlabel("Win Rate %")
    ax_scat.set_ylabel("Profit Factor")
    ax_scat.set_title("Win Rate vs Profit Factor", color=THEME["text"], fontsize=10)

    # ── Drawdown ──────────────────────────────────────────────────────────────
    dd_models = sorted_m[:5]
    for i, name in enumerate(dd_models):
        trades = eligible.get(name)
        if trades is None or trades.empty:
            continue
        dd    = _max_drawdown(trades)
        color = palette[i % len(palette)]
        ax_dd.fill_between(range(len(dd)), dd.values, 0,
                           color=color, alpha=0.2)
        ax_dd.plot(dd.values, color=color, linewidth=1.4,
                   alpha=0.8,
                   label=f"{SHORT_NAMES.get(name, name)} MDD{dd.min():.1f}R")
    ax_dd.axhline(0, color=THEME["muted"], linewidth=1, linestyle="--")
    ax_dd.set_xlabel("Trade #")
    ax_dd.set_ylabel("Drawdown (R)")
    ax_dd.set_title("Drawdown — Top 5 Models", color=THEME["text"], fontsize=10)
    ax_dd.legend(loc="lower left", fontsize=7, framealpha=0.3,
                 facecolor=THEME["panel"], edgecolor=THEME["border"],
                 labelcolor=THEME["text"])
    ax_dd.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.1f}R"))

    plt.savefig(save_path, dpi=150, bbox_inches="tight",
                facecolor=THEME["bg"])
    plt.close(fig)
    print(f"  ✅  Saved: {save_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Master plot_all
# ─────────────────────────────────────────────────────────────────────────────

def plot_all(
    bt: Dict[str, pd.DataFrame],
    symbol: str = "Instrument",
    output_dir: str = ".",
    min_trades: int = 3,
):
    """
    Generate all charts and save to output_dir.
    Returns list of saved file paths.
    """
    import os
    os.makedirs(output_dir, exist_ok=True)
    tbl = results_table(bt)

    def path(name):
        return os.path.join(output_dir, name)

    print("\nGenerating charts...")
    saved = []

    plot_model_comparison(tbl, save_path=path("chart_comparison.png"))
    saved.append(path("chart_comparison.png"))

    plot_equity_curves(bt, min_trades=min_trades,
                       save_path=path("chart_equity.png"))
    saved.append(path("chart_equity.png"))

    plot_wr_vs_pf(tbl, min_trades=min_trades,
                  save_path=path("chart_wr_pf.png"))
    saved.append(path("chart_wr_pf.png"))

    plot_drawdown(bt, min_trades=min_trades,
                  save_path=path("chart_drawdown.png"))
    saved.append(path("chart_drawdown.png"))

    plot_trade_distribution(bt, save_path=path("chart_distribution.png"))
    saved.append(path("chart_distribution.png"))

    plot_summary_dashboard(bt, tbl, symbol=symbol,
                           save_path=path("chart_dashboard.png"))
    saved.append(path("chart_dashboard.png"))

    print(f"\n✅  All charts saved to: {output_dir}")
    return saved


# ─────────────────────────────────────────────────────────────────────────────
# Quick test
# ─────────────────────────────────────────────────────────────────────────────
