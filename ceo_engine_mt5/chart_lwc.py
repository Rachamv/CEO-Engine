"""
The CEO Protocol — Interactive HTML Chart Renderer (Lightweight Charts)
=============================================================================
plot_chart_html() and the drawing helpers it calls, rendering the CEO
structure chart with TradingView's Lightweight Charts JS library instead
of plotly -- the same library the live dashboard already loads for
/api/candles, so the whole project now uses one charting approach and
zero Python-side charting dependencies for this renderer (Chart.js for
report.py's statistics, Lightweight Charts for candlestick + overlays,
matplotlib only for the static Telegram-alert PNG in chart_png.py).

Draws the same layers the previous plotly version did:
    - OHLC candlesticks
    - EMA fast / slow
    - Volume histogram (bottom overlay pane)
    - Pivot high / low markers
    - Signal markers (sweep entry points, CEO full-sequence entries,
      candle pattern confirmations)
    - Geometric pattern trendlines
    - CEO structure overlays (order block zones, fib 50% level,
      double/triple top-bottom levels)

One accepted simplification versus the plotly version: order-block zones
are drawn as dashed top/bottom boundary lines rather than a filled
rectangle, and the session background tint is dropped -- Lightweight
Charts v4 (the version already pinned in the dashboard template) has no
built-in rectangle/region-shading primitive without pulling in its
newer v5 plugin API, which the live dashboard doesn't use. Everything
that carries trading-decision information (structure levels, signals,
patterns) is preserved; only the two purely-cosmetic layers are simplified.

plot_chart_html()'s signature and return contract (a full standalone
HTML document containing a literal "<body>...</body>") are unchanged
from the plotly version, so every caller (dashboard.py's legacy chart
endpoint, report.py's _embed_ceo_chart, mt5_live_signals.py's fallback,
run.py's --ceo-chart CLI flag) needs no changes at all.
"""

import json
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from .chart_theme import THEME, _prep_slice, _col

LIGHTWEIGHT_CHARTS_CDN = "https://unpkg.com/lightweight-charts@4.1.3/dist/lightweight-charts.standalone.production.js"


def _json_default(o):
    if hasattr(o, "item"):
        return o.item()
    return str(o)


def _to_unix_seconds(times: pd.Series) -> np.ndarray:
    """
    Lightweight Charts wants UNIX seconds (a plain number) for intraday
    time axes. Deliberately does NOT do `.astype("int64") // 10**9` --
    that assumes datetime64[ns] storage, but pandas (from 2.x's
    non-nanosecond dtypes onward, and more so as of 3.x) can store
    datetime64 at second/microsecond/etc resolution depending on how the
    Series was built, which silently produces wrong timestamps off by a
    factor of 1000/1e6/etc if assumed. Subtracting the epoch and dividing
    by a Timedelta is resolution-independent.
    """
    dt = pd.to_datetime(times)
    if dt.dt.tz is not None:
        dt = dt.dt.tz_convert("UTC").dt.tz_localize(None)
    seconds = (dt - pd.Timestamp("1970-01-01")) // pd.Timedelta(seconds=1)
    return seconds.to_numpy()


def _build_times_index(sl: pd.DataFrame, n: int) -> pd.Series:
    """Datetime x-axis values, or a defensive integer-sequence fallback
    (see note below)."""
    if "datetime" in sl.columns:
        return pd.to_datetime(sl["datetime"])
    # Defensive fallback for a caller that passes a DataFrame without a
    # "datetime" column (every real path in this codebase sets one via
    # data.py/_rates_to_df, so this is normally unreachable).
    return pd.Series(pd.date_range("2000-01-01", periods=n, freq="min"))


# ─────────────────────────────────────────────────────────────────────────────
# Series/marker/price-line builders -- each returns plain Python data
# structures (JSON-serializable), not JS. The JS is templated once at the
# very end from these.
# ─────────────────────────────────────────────────────────────────────────────

def _build_candle_data(times_unix, o, h, lo, c) -> list:
    return [{"time": int(t), "open": float(oo), "high": float(hh),
              "low": float(ll), "close": float(cc)}
            for t, oo, hh, ll, cc in zip(times_unix, o, h, lo, c)]


def _build_line_data(times_unix, values) -> list:
    return [{"time": int(t), "value": float(v)}
            for t, v in zip(times_unix, values) if not np.isnan(v)]


def _build_ema_series_specs(sl, times_unix) -> list:
    specs = []
    ema_fast = _col(sl, "ema_fast")
    ema_slow = _col(sl, "ema_slow")
    if not np.all(np.isnan(ema_fast)):
        specs.append({"color": THEME["ema_fast"], "data": _build_line_data(times_unix, ema_fast)})
    if not np.all(np.isnan(ema_slow)):
        specs.append({"color": THEME["ema_slow"], "data": _build_line_data(times_unix, ema_slow)})
    return specs


def _build_volume_data(times_unix, o, c, vol) -> list:
    return [{"time": int(t), "value": float(v),
              "color": THEME["candle_bull"] if cc >= oo else THEME["candle_bear"]}
            for t, oo, cc, v in zip(times_unix, o, c, vol)]


def _build_pivot_markers(sl, times_unix, atr_arr) -> list:
    ph = _col(sl, "pivot_high")
    pl = _col(sl, "pivot_low")
    markers = []
    for i, t in enumerate(times_unix):
        if not np.isnan(ph[i]):
            markers.append({"time": int(t), "position": "aboveBar", "color": THEME["pivot_high"],
                             "shape": "arrowDown", "size": 0.7, "text": ""})
        if not np.isnan(pl[i]):
            markers.append({"time": int(t), "position": "belowBar", "color": THEME["pivot_low"],
                             "shape": "arrowUp", "size": 0.7, "text": ""})
    return markers


def _build_signal_markers(sl, times_unix) -> list:
    """CEO full-sequence entries, base sweep signals, candle pattern
    confirmations -- all merged into one list since Lightweight Charts'
    setMarkers() replaces the whole marker set per call."""
    base_long  = _col(sl, "base_long",  False).astype(bool)
    base_short = _col(sl, "base_short", False).astype(bool)
    cp_bull    = _col(sl, "cp_bull_confirmation", False).astype(bool)
    cp_bear    = _col(sl, "cp_bear_confirmation", False).astype(bool)
    ceo_long   = _col(sl, "ceo_long_valid",  False).astype(bool)
    ceo_short  = _col(sl, "ceo_short_valid", False).astype(bool)

    markers = []
    for i, t in enumerate(times_unix):
        t = int(t)
        if ceo_long[i]:
            markers.append({"time": t, "position": "belowBar", "color": THEME["signal_long"],
                             "shape": "arrowUp", "size": 2.2, "text": "CEO"})
        if ceo_short[i]:
            markers.append({"time": t, "position": "aboveBar", "color": THEME["signal_short"],
                             "shape": "arrowDown", "size": 2.2, "text": "CEO"})
        if base_long[i] and not ceo_long[i]:
            markers.append({"time": t, "position": "belowBar", "color": THEME["signal_long"],
                             "shape": "arrowUp", "size": 1, "text": ""})
        if base_short[i] and not ceo_short[i]:
            markers.append({"time": t, "position": "aboveBar", "color": THEME["signal_short"],
                             "shape": "arrowDown", "size": 1, "text": ""})
        if cp_bull[i]:
            markers.append({"time": t, "position": "belowBar", "color": THEME["cp_bull"],
                             "shape": "circle", "size": 0.6, "text": ""})
        if cp_bear[i]:
            markers.append({"time": t, "position": "aboveBar", "color": THEME["cp_bear"],
                             "shape": "circle", "size": 0.6, "text": ""})
    markers.sort(key=lambda m: m["time"])
    return markers


def _build_bos_markers(sl, times_unix) -> list:
    """Break-of-structure events -- the first CEO-structure signal that
    fires chronologically, well before the full ceo_long_valid/
    ceo_short_valid gate (which also needs the Fib zone and OB/QM to
    line up). Marked distinctly from the entry-signal markers so a
    trader can see structure building up, not just the final entry."""
    bos_long    = _col(sl, "bos_long",         False).astype(bool)
    bos_short   = _col(sl, "bos_short",        False).astype(bool)
    dbos_long   = _col(sl, "double_bos_long",  False).astype(bool)
    dbos_short  = _col(sl, "double_bos_short", False).astype(bool)

    markers = []
    for i, t in enumerate(times_unix):
        t = int(t)
        if dbos_long[i]:
            markers.append({"time": t, "position": "belowBar", "color": THEME["bos"],
                             "shape": "arrowUp", "size": 1.4, "text": "BOS2"})
        elif bos_long[i]:
            markers.append({"time": t, "position": "belowBar", "color": THEME["bos"],
                             "shape": "circle", "size": 0.8, "text": "BOS"})
        if dbos_short[i]:
            markers.append({"time": t, "position": "aboveBar", "color": THEME["bos"],
                             "shape": "arrowDown", "size": 1.4, "text": "BOS2"})
        elif bos_short[i]:
            markers.append({"time": t, "position": "aboveBar", "color": THEME["bos"],
                             "shape": "circle", "size": 0.8, "text": "BOS"})
    return markers


def _build_structure_overlays(sl, times_unix, c, atr) -> dict:
    """
    Order-block zones (as dashed top/bottom boundary line series -- see
    module docstring for why this replaces plotly's filled rectangles)
    and horizontal price lines (fib 50%, double/triple top-bottom).
    """
    n = len(sl)
    zone_lines = []   # each: {"color":..., "data":[{time,value}, {time,value}]}  (2-point line = flat boundary)
    price_lines = []  # each: {"price":..., "color":..., "title":..., "dash": bool}

    ob_bull_active = _col(sl, "ob_bull_active", False).astype(bool)
    ob_bull_high   = _col(sl, "ob_bull_high")
    ob_bull_low    = _col(sl, "ob_bull_low")
    ob_bear_active = _col(sl, "ob_bear_active", False).astype(bool)
    ob_bear_high   = _col(sl, "ob_bear_high")
    ob_bear_low    = _col(sl, "ob_bear_low")

    drawn = set()
    last_t = int(times_unix[-1])
    for i in range(n):
        if ob_bull_active[i] and not np.isnan(ob_bull_high[i]):
            key = ("bull", round(float(ob_bull_high[i]), 1))
            if key not in drawn:
                t0 = int(times_unix[i])
                zone_lines.append({"color": THEME["ob_bull"],
                                   "data": [{"time": t0, "value": float(ob_bull_high[i])},
                                            {"time": last_t, "value": float(ob_bull_high[i])}]})
                zone_lines.append({"color": THEME["ob_bull"],
                                   "data": [{"time": t0, "value": float(ob_bull_low[i])},
                                            {"time": last_t, "value": float(ob_bull_low[i])}]})
                drawn.add(key)
        if ob_bear_active[i] and not np.isnan(ob_bear_high[i]):
            key = ("bear", round(float(ob_bear_high[i]), 1))
            if key not in drawn:
                t0 = int(times_unix[i])
                zone_lines.append({"color": THEME["ob_bear"],
                                   "data": [{"time": t0, "value": float(ob_bear_high[i])},
                                            {"time": last_t, "value": float(ob_bear_high[i])}]})
                zone_lines.append({"color": THEME["ob_bear"],
                                   "data": [{"time": t0, "value": float(ob_bear_low[i])},
                                            {"time": last_t, "value": float(ob_bear_low[i])}]})
                drawn.add(key)

    # Fib 50%
    fib_50 = _col(sl, "fib_50")
    last_fib = fib_50[~np.isnan(fib_50)]
    if len(last_fib) > 0:
        price_lines.append({"price": float(last_fib[-1]), "color": THEME["fib"], "title": "Fib 50%"})

    # Quasimodo (QM) levels -- highest-probability entry per the CEO
    # method (a failed order block). Drawn as dashed price lines rather
    # than zones since a QM is a single reaction level, not a range.
    qm_bull_active = _col(sl, "qm_bull_active", False).astype(bool)
    qm_bull_level  = _col(sl, "qm_bull_level")
    qm_bear_active = _col(sl, "qm_bear_active", False).astype(bool)
    qm_bear_level  = _col(sl, "qm_bear_level")
    if qm_bull_active.any() and not np.isnan(qm_bull_level[qm_bull_active][-1]):
        price_lines.append({"price": float(qm_bull_level[qm_bull_active][-1]),
                             "color": THEME["qm_bull"], "title": "QM (bull)", "dash": True})
    if qm_bear_active.any() and not np.isnan(qm_bear_level[qm_bear_active][-1]):
        price_lines.append({"price": float(qm_bear_level[qm_bear_active][-1]),
                             "color": THEME["qm_bear"], "title": "QM (bear)", "dash": True})

    # Fair Value Gaps -- 3-candle imbalance zones. Bull FVG: gap between
    # bar i-2's high and bar i's low. Bear FVG: gap between bar i-2's low
    # and bar i's high. Drawn the same way as OB zones (top/bottom
    # boundary lines) so both structure concepts read consistently.
    bull_fvg = _col(sl, "bull_fvg", False).astype(bool)
    bear_fvg = _col(sl, "bear_fvg", False).astype(bool)
    h_arr = _col(sl, "high")
    lo_arr = _col(sl, "low")
    for i in range(2, len(sl)):
        if bull_fvg[i]:
            top, bot = float(lo_arr[i]), float(h_arr[i - 2])
            t0, t1 = int(times_unix[i - 2]), int(times_unix[i])
            zone_lines.append({"color": THEME["fvg_bull"],
                               "data": [{"time": t0, "value": top}, {"time": t1, "value": top}]})
            zone_lines.append({"color": THEME["fvg_bull"],
                               "data": [{"time": t0, "value": bot}, {"time": t1, "value": bot}]})
        if bear_fvg[i]:
            top, bot = float(lo_arr[i - 2]), float(h_arr[i])
            t0, t1 = int(times_unix[i - 2]), int(times_unix[i])
            zone_lines.append({"color": THEME["fvg_bear"],
                               "data": [{"time": t0, "value": top}, {"time": t1, "value": top}]})
            zone_lines.append({"color": THEME["fvg_bear"],
                               "data": [{"time": t0, "value": bot}, {"time": t1, "value": bot}]})

    # Double/triple top-bottom levels
    h = _col(sl, "high")
    lo = _col(sl, "low")
    for col, lvl_arr, color, label in [
        ("pat_double_top",    h,  THEME["red"],   "Double Top"),
        ("pat_double_bottom", lo, THEME["green"], "Double Bottom"),
        ("pat_triple_top",    h,  THEME["red"],   "Triple Top"),
        ("pat_triple_bottom", lo, THEME["green"], "Triple Bottom"),
    ]:
        pat = _col(sl, col, False).astype(bool)
        if not pat.any():
            continue
        last_i = int(np.where(pat)[0][-1])
        price_lines.append({"price": float(lvl_arr[last_i]), "color": color, "title": label})

    return {"zone_lines": zone_lines, "price_lines": price_lines}


def _build_pattern_trendlines(sl, times_unix, h, lo) -> list:
    """Geometric pattern trendlines (triangles/wedges): a straight
    2-point line series per detected pattern, same as the plotly version."""
    ph = _col(sl, "pivot_high")
    pl = _col(sl, "pivot_low")

    triangle_pats = {
        "pat_asc_triangle":   THEME["pat_line"],
        "pat_desc_triangle":  THEME["pat_line"],
        "pat_sym_triangle":   THEME["pat_line"],
        "pat_rising_wedge":   THEME["red"],
        "pat_falling_wedge":  THEME["green"],
    }

    lines = []
    for pat_col, color in triangle_pats.items():
        pat = _col(sl, pat_col, False).astype(bool)
        if not pat.any():
            continue
        last_i = int(np.where(pat)[0][-1])
        ph_idx = [i for i in range(last_i + 1) if not np.isnan(ph[i])][-2:]
        pl_idx = [i for i in range(last_i + 1) if not np.isnan(pl[i])][-2:]
        name = pat_col.replace("pat_", "").replace("_", " ").title()

        if len(ph_idx) == 2:
            lines.append({"color": color, "name": name, "data": [
                {"time": int(times_unix[ph_idx[0]]), "value": float(h[ph_idx[0]])},
                {"time": int(times_unix[ph_idx[1]]), "value": float(h[ph_idx[1]])},
            ]})
        if len(pl_idx) == 2:
            lines.append({"color": color, "name": f"{name} (lower)", "data": [
                {"time": int(times_unix[pl_idx[0]]), "value": float(lo[pl_idx[0]])},
                {"time": int(times_unix[pl_idx[1]]), "value": float(lo[pl_idx[1]])},
            ]})
    return lines


# ─────────────────────────────────────────────────────────────────────────────
# Public entry point
# ─────────────────────────────────────────────────────────────────────────────

def build_structure_payload(df: pd.DataFrame, bars: int = 300) -> dict:
    """
    JSON-serializable CEO structure overlay for the *live* dashboard chart
    (the native LightweightCharts candlestick view fed by /api/candles,
    which renders client-side from raw JSON -- unlike plot_chart_html(),
    which returns a complete server-rendered HTML document). Pushed via
    dashboard.update_structure() alongside dashboard.update_candles() so
    the browser can draw order block / FVG zones, Fib 50%, and QM levels
    as price/line series, and BOS + CEO-valid events as markers.

    Returns: {"zoneLines": [...], "priceLines": [...], "markers": [...]}
    """
    sl = _prep_slice(df, bars)
    n = len(sl)
    if n == 0:
        return {"zoneLines": [], "priceLines": [], "markers": []}

    times = _build_times_index(sl, n)
    times_unix = _to_unix_seconds(times)
    c = sl["close"].values
    atr_arr = _col(sl, "atr")
    atr_arr = np.where(np.isnan(atr_arr),
                        np.nanmean(atr_arr) if not np.all(np.isnan(atr_arr)) else 1.0,
                        atr_arr)

    structure = _build_structure_overlays(sl, times_unix, c, atr_arr)
    markers = _build_bos_markers(sl, times_unix)
    markers += [m for m in _build_signal_markers(sl, times_unix) if m["text"] == "CEO"]
    markers.sort(key=lambda m: m["time"])

    return {
        "zoneLines": structure["zone_lines"],
        "priceLines": structure["price_lines"],
        "markers": markers,
    }


def plot_chart_html(
    df:            pd.DataFrame,
    symbol:        str  = "XAUUSD",
    tf:            str  = "M15",
    bars:          int  = 200,
    show_volume:   bool = True,
    show_struct:   bool = True,
    show_patterns: bool = True,
    show_sessions: bool = True,   # accepted for signature compatibility; see module docstring
    height:        int  = 700,
) -> str:
    """
    Render an interactive Lightweight Charts candlestick chart.
    Returns the full HTML string (embed in dashboard or save to file) --
    same contract as the plotly version this replaced.
    """
    sl = _prep_slice(df, bars)
    n  = len(sl)

    times = _build_times_index(sl, n)
    times_unix = _to_unix_seconds(times)

    o  = sl["open"].values
    h  = sl["high"].values
    lo = sl["low"].values
    c  = sl["close"].values

    atr_arr = _col(sl, "atr")
    atr_arr = np.where(np.isnan(atr_arr),
                        np.nanmean(atr_arr) if not np.all(np.isnan(atr_arr)) else 1.0,
                        atr_arr)

    candle_data = _build_candle_data(times_unix, o, h, lo, c)
    ema_specs   = _build_ema_series_specs(sl, times_unix)

    volume_data = None
    if show_volume and "volume" in sl.columns:
        volume_data = _build_volume_data(times_unix, o, c, sl["volume"].values)

    markers = _build_pivot_markers(sl, times_unix, atr_arr)
    markers += _build_signal_markers(sl, times_unix)
    if show_struct:
        markers += _build_bos_markers(sl, times_unix)
    markers.sort(key=lambda m: m["time"])

    structure = {"zone_lines": [], "price_lines": []}
    if show_struct:
        structure = _build_structure_overlays(sl, times_unix, c, atr_arr)

    pattern_lines = []
    if show_patterns:
        pattern_lines = _build_pattern_trendlines(sl, times_unix, h, lo)

    last_price = float(c[-1])
    last_sig = ""
    if "base_long" in sl.columns and bool(sl["base_long"].iloc[-1]):
        last_sig = " \u25b2 LONG"
    elif "base_short" in sl.columns and bool(sl["base_short"].iloc[-1]):
        last_sig = " \u25bc SHORT"
    title = f"CEO Engine | {symbol} {tf} | {last_price:.2f}{last_sig}"

    payload = {
        "theme": THEME,
        "title": title,
        "height": height,
        "candles": candle_data,
        "emaSeries": ema_specs,
        "volume": volume_data,
        "markers": markers,
        "zoneLines": structure["zone_lines"],
        "priceLines": structure["price_lines"],
        "patternLines": pattern_lines,
    }
    payload_json = json.dumps(payload, default=_json_default).replace("</", "<\\/")

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>{title}</title>
<script src="{LIGHTWEIGHT_CHARTS_CDN}"></script>
<style>
  html, body {{ margin:0; padding:0; background:{THEME["bg"]}; }}
  #chart-title {{ font-family:sans-serif; color:{THEME["text"]}; font-size:14px;
                  padding:8px 12px; background:{THEME["bg"]}; }}
  #chart-container {{ width:100%; }}
</style>
</head>
<body>
<div id="chart-title">{title}</div>
<div id="chart-container"></div>
<script>
(function() {{
  const P = {payload_json};
  const container = document.getElementById("chart-container");

  const chart = LightweightCharts.createChart(container, {{
    width: container.clientWidth || 900,
    height: P.height,
    layout: {{ background: {{ color: P.theme.bg }}, textColor: P.theme.text }},
    grid: {{
      vertLines: {{ color: P.theme.grid }},
      horzLines: {{ color: P.theme.grid }},
    }},
    crosshair: {{ mode: LightweightCharts.CrosshairMode.Normal }},
    rightPriceScale: {{ borderColor: P.theme.border }},
    timeScale: {{ borderColor: P.theme.border, timeVisible: true, secondsVisible: false }},
  }});

  const candleSeries = chart.addCandlestickSeries({{
    upColor: P.theme.candle_bull, downColor: P.theme.candle_bear,
    borderUpColor: P.theme.candle_bull, borderDownColor: P.theme.candle_bear,
    wickUpColor: P.theme.candle_bull, wickDownColor: P.theme.candle_bear,
  }});
  candleSeries.setData(P.candles);

  P.emaSeries.forEach(function(spec) {{
    const s = chart.addLineSeries({{ color: spec.color, lineWidth: 1.5, priceLineVisible: false, crosshairMarkerVisible: false }});
    s.setData(spec.data);
  }});

  if (P.volume) {{
    const volSeries = chart.addHistogramSeries({{
      priceFormat: {{ type: "volume" }}, priceScaleId: "",
    }});
    volSeries.priceScale().applyOptions({{ scaleMargins: {{ top: 0.82, bottom: 0 }} }});
    volSeries.setData(P.volume);
  }}

  P.zoneLines.forEach(function(spec) {{
    const s = chart.addLineSeries({{
      color: spec.color, lineWidth: 1, lineStyle: LightweightCharts.LineStyle.Dotted,
      priceLineVisible: false, lastValueVisible: false, crosshairMarkerVisible: false,
    }});
    s.setData(spec.data);
  }});

  P.patternLines.forEach(function(spec) {{
    const s = chart.addLineSeries({{
      color: spec.color, lineWidth: 1.5, lineStyle: LightweightCharts.LineStyle.Dashed,
      priceLineVisible: false, lastValueVisible: false, crosshairMarkerVisible: false,
      title: spec.name,
    }});
    s.setData(spec.data);
  }});

  P.priceLines.forEach(function(spec) {{
    candleSeries.createPriceLine({{
      price: spec.price, color: spec.color, lineWidth: 1,
      lineStyle: LightweightCharts.LineStyle.DashDot,
      axisLabelVisible: true, title: spec.title,
    }});
  }});

  if (P.markers.length) {{
    candleSeries.setMarkers(P.markers);
  }}

  chart.timeScale().fitContent();

  window.addEventListener("resize", function() {{
    chart.applyOptions({{ width: container.clientWidth || 900 }});
  }});
}})();
</script>
</body>
</html>"""
