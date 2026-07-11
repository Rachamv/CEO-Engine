"""
The CEO Protocol — Chart Module
=====================================
Two chart renderers sharing the same enriched DataFrame:

    plot_chart_png(df, ...)    — static matplotlib PNG (for Telegram alerts)
    plot_chart_html(df, ...)   — interactive Lightweight Charts HTML (for dashboard)

Both draw the same layers:
    • OHLC candlesticks
    • EMA fast / slow
    • Volume bars
    • Pivot high / low markers
    • Signal markers (sweep entry points)
    • Candle pattern markers
    • Geometric pattern trendlines / levels
    • CEO structure overlays (OB zones, QM levels, BOS, Fib levels)
    • Session background shading (PNG renderer only -- see chart_lwc.py's
      module docstring for why the HTML renderer drops this one layer)

This file is the public entry point — the actual rendering code lives in:
    chart_theme.py   — color theme + shared DataFrame-slicing helpers
    chart_png.py     — matplotlib (static PNG) renderer
    chart_lwc.py     — Lightweight Charts (interactive HTML) renderer
(split out purely for file-size/maintainability — see CHANGELOG)

Usage
-----
    from .chart import plot_chart_png, plot_chart_html

    # After full pipeline:
    plot_chart_png(df,  symbol="XAUUSD", tf="M15", bars=150,
                   save_path="signal_alert.png")

    html_str = plot_chart_html(df, symbol="XAUUSD", tf="M15", bars=200)
    with open("chart.html", "w") as f:
        f.write(html_str)
"""

import numpy as np
import pandas as pd

from .chart_png import plot_chart_png
from .chart_lwc import plot_chart_html

__all__ = ["plot_chart_png", "plot_chart_html"]
