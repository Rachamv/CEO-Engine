"""
The CEO Protocol — Backtest HTML Report
=============================================
Generates a single self-contained HTML file combining everything
needed to evaluate a symbol/timeframe backtest at a glance:

    - Header: symbol, timeframe, date range, best model
    - Equity curve (best model, interactive)
    - Model comparison table (all 17 models, sortable visually)
    - Trade distribution (R-multiple histogram)
    - Win rate vs Profit Factor scatter
    - Session breakdown (which sessions the model performs best in)
    - Drawdown chart
    - CEO candlestick chart (last 200 bars with all overlays)
    - Walk-forward consistency (if provided)
    - Journal performance stats (if provided — for live-vs-backtest comparison)

One HTML file, opens in any browser, no server required. Easy to
archive, share, or attach to a journal entry before going live with
a new symbol/timeframe combination.

Usage
-----
    from .report import generate_report

    path = generate_report(
        df, bt, tbl,
        symbol="XAUUSD", tf="M15",
        wf_summary=wf["summary"],          # optional
        journal=journal,                    # optional
        out_path="ceo_report_XAUUSD_M15.html",
    )
"""

import os
import json
from datetime import datetime, timezone
from typing import Dict, Optional

import numpy as np
import pandas as pd

from .backtest import equity_curve
from .chart_lwc import LIGHTWEIGHT_CHARTS_CDN

from ceo_engine_mt5.ceo_logging import get_logger
logger = get_logger(__name__)


def _json_default(o):
    """Lets json.dumps handle numpy/pandas scalar types (int64, float32,
    Timestamp, etc.) that turn up in DataFrame-derived chart data."""
    if hasattr(o, "item"):
        return o.item()
    return str(o)


# ─────────────────────────────────────────────────────────────────────────────
# Theme (matches chart.py / dashboard.py)
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
}


CHARTJS_CDN = "https://cdn.jsdelivr.net/npm/chart.js@4.4.4/dist/chart.umd.min.js"

_chart_counter = {"n": 0}


def _render_chart(config: Optional[dict], empty_message: Optional[str] = None,
                   height: int = 320) -> str:
    """
    Generic Chart.js chart embed. Renders a themed 'no data' placeholder
    when there's nothing to plot, otherwise a <canvas> + inline <script>
    that constructs the chart client-side. Chart.js itself is loaded once
    via CDN in the document <head> -- no Python charting library needed
    at all (this replaced a top-level `import plotly`, which meant the
    whole report generator failed to even import when plotly wasn't
    installed, regardless of whether a chart was actually requested).
    """
    if config is None or empty_message:
        msg = empty_message or "No data"
        return (f'<div style="height:{height}px;display:flex;align-items:center;'
                f'justify-content:center;color:{THEME["muted"]};font-size:13px;">{msg}</div>')

    _chart_counter["n"] += 1
    canvas_id = f"chart-{_chart_counter['n']}"
    config_json = json.dumps(config, default=_json_default).replace("</", "<\\/")
    return f"""<div style="position:relative;height:{height}px;">
      <canvas id="{canvas_id}"></canvas>
    </div>
    <script>new Chart(document.getElementById("{canvas_id}"), {config_json});</script>"""


def _themed_axis(title: str = "", **overrides) -> dict:
    axis = {
        "grid":  {"color": THEME["grid"]},
        "ticks": {"color": THEME["muted"]},
    }
    if title:
        axis["title"] = {"display": True, "text": title, "color": THEME["muted"]}
    axis.update(overrides)
    return axis


# ─────────────────────────────────────────────────────────────────────────────
# Session classification (lightweight, mirrors session_filter.py)
# ─────────────────────────────────────────────────────────────────────────────

def _session_for_hour(hour_utc: int) -> str:
    if 12 <= hour_utc < 16:
        return "overlap"
    if 7 <= hour_utc < 16:
        return "london"
    if 12 <= hour_utc < 21:
        return "new_york"
    if 0 <= hour_utc < 9:
        return "asian"
    return "other"


# ─────────────────────────────────────────────────────────────────────────────
# Individual chart builders (each returns a Chart.js-ready HTML string)
# ─────────────────────────────────────────────────────────────────────────────

def _chart_equity_curve(trades: pd.DataFrame, model_name: str) -> str:
    eq = equity_curve(trades)
    if eq.empty:
        return _render_chart(None, empty_message="No trades", height=350)

    labels = [str(i) for i in range(len(eq))]
    config = {
        "type": "line",
        "data": {
            "labels": labels,
            "datasets": [
                {
                    "label": "Cumulative R",
                    "data": [round(float(v), 4) for v in eq.values],
                    "borderColor": THEME["teal"],
                    "backgroundColor": "rgba(0,188,212,0.08)",
                    "fill": "origin",
                    "borderWidth": 2,
                    "pointRadius": 0,
                    "tension": 0.05,
                },
                {
                    "label": "Zero",
                    "data": [0] * len(eq),
                    "borderColor": THEME["muted"],
                    "borderWidth": 1,
                    "borderDash": [4, 4],
                    "pointRadius": 0,
                    "fill": False,
                },
            ],
        },
        "options": {
            "responsive": True, "maintainAspectRatio": False,
            "plugins": {
                "legend": {"display": False},
                "title": {"display": True, "text": f"Equity Curve — {model_name}",
                          "color": THEME["text"], "font": {"size": 14}},
            },
            "scales": {
                "x": _themed_axis(**{"display": False}),
                "y": _themed_axis("Cumulative R"),
            },
        },
    }
    return _render_chart(config, height=350)


def _chart_model_comparison(tbl: pd.DataFrame) -> str:
    valid = tbl.dropna(subset=["Net R"]).sort_values("Net R", ascending=True)
    if valid.empty:
        return _render_chart(None, empty_message="No data", height=350)

    colors = [THEME["green"] if v >= 0 else THEME["red"] for v in valid["Net R"]]
    config = {
        "type": "bar",
        "data": {
            "labels": list(valid.index),
            "datasets": [{
                "label": "Net R",
                "data": [round(float(v), 3) for v in valid["Net R"]],
                "backgroundColor": colors,
            }],
        },
        "options": {
            "indexAxis": "y",
            "responsive": True, "maintainAspectRatio": False,
            "plugins": {
                "legend": {"display": False},
                "title": {"display": True, "text": "Model Comparison — Net R",
                          "color": THEME["text"], "font": {"size": 14}},
            },
            "scales": {
                "x": _themed_axis("Net R"),
                "y": {"ticks": {"color": THEME["muted"], "font": {"size": 10}}},
            },
        },
    }
    return _render_chart(config, height=max(350, len(valid) * 26))


def _chart_trade_distribution(trades: pd.DataFrame) -> str:
    if trades.empty:
        return _render_chart(None, empty_message="No trades", height=350)

    r = trades["r_result"].dropna()
    if r.empty:
        return _render_chart(None, empty_message="No trades", height=350)

    # Chart.js has no built-in histogram binning -- bucket the R-multiples
    # into 20 equal-width bins here, same as plotly's nbinsx=20 default.
    counts, edges = np.histogram(r, bins=20)
    labels = [f"{edges[i]:.1f}" for i in range(len(edges) - 1)]
    bin_mids = (edges[:-1] + edges[1:]) / 2
    win_counts, loss_counts = [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        bucket = r[(r >= lo) & (r < hi)] if hi < edges[-1] else r[(r >= lo) & (r <= hi)]
        win_counts.append(int((bucket > 0).sum()))
        loss_counts.append(int((bucket <= 0).sum()))

    config = {
        "type": "bar",
        "data": {
            "labels": labels,
            "datasets": [
                {"label": "Wins", "data": win_counts,
                 "backgroundColor": THEME["green"], "stack": "r"},
                {"label": "Losses", "data": loss_counts,
                 "backgroundColor": THEME["red"], "stack": "r"},
            ],
        },
        "options": {
            "responsive": True, "maintainAspectRatio": False,
            "plugins": {
                "legend": {"labels": {"color": THEME["muted"]}},
                "title": {"display": True, "text": "Trade R-Multiple Distribution",
                          "color": THEME["text"], "font": {"size": 14}},
            },
            "scales": {
                "x": _themed_axis("R Multiple", stacked=True),
                "y": _themed_axis("Count", stacked=True),
            },
        },
    }
    return _render_chart(config, height=350)


def _chart_drawdown(trades: pd.DataFrame) -> str:
    if trades.empty:
        return _render_chart(None, empty_message="No trades", height=300)

    eq = equity_curve(trades)
    running_max = eq.cummax()
    dd = eq - running_max
    config = {
        "type": "line",
        "data": {
            "labels": [str(i) for i in range(len(dd))],
            "datasets": [{
                "label": "Drawdown",
                "data": [round(float(v), 4) for v in dd.values],
                "borderColor": THEME["red"],
                "backgroundColor": "rgba(255,61,87,0.12)",
                "fill": "origin",
                "borderWidth": 1.5,
                "pointRadius": 0,
            }],
        },
        "options": {
            "responsive": True, "maintainAspectRatio": False,
            "plugins": {
                "legend": {"display": False},
                "title": {"display": True, "text": "Drawdown (R)",
                          "color": THEME["text"], "font": {"size": 14}},
            },
            "scales": {
                "x": _themed_axis(**{"display": False}),
                "y": _themed_axis("Drawdown (R)"),
            },
        },
    }
    return _render_chart(config, height=300)


def _chart_session_breakdown(trades: pd.DataFrame) -> str:
    if trades.empty or "entry_bar" not in trades.columns:
        return _render_chart(None, empty_message="No trades", height=350)

    t = trades.copy()
    t["session"] = pd.to_datetime(t["entry_bar"]).dt.hour.apply(_session_for_hour)

    rows = []
    for sess in ["asian", "pre_london", "london", "overlap", "new_york", "other"]:
        sub = t[t["session"] == sess]
        if len(sub) == 0:
            continue
        wr = (sub["win"].sum() / len(sub) * 100) if len(sub) else 0
        rows.append({"session": sess, "trades": len(sub),
                     "win_rate": wr, "net_r": sub["r_result"].sum()})

    if not rows:
        return _render_chart(None, empty_message="No session data", height=300)

    sess_df = pd.DataFrame(rows)
    colors  = [THEME["green"] if v >= 0 else THEME["red"] for v in sess_df["net_r"]]

    config = {
        "type": "bar",
        "data": {
            "labels": list(sess_df["session"]),
            "datasets": [{
                "label": "Net R",
                "data": [round(float(v), 3) for v in sess_df["net_r"]],
                "backgroundColor": colors,
                # Carried alongside the chart data purely so the tooltip
                # callback below can show trade count / win rate on hover
                # (this replaces plotly's always-visible outside-bar labels).
                "meta": [{"trades": int(t_), "win_rate": round(float(w), 1)}
                         for t_, w in zip(sess_df["trades"], sess_df["win_rate"])],
            }],
        },
        "options": {
            "responsive": True, "maintainAspectRatio": False,
            "plugins": {
                "legend": {"display": False},
                "title": {"display": True, "text": "Performance by Session",
                          "color": THEME["text"], "font": {"size": 14}},
                "tooltip": {"callbacks": {
                    "label": "__SESSION_TOOLTIP__",
                }},
            },
            "scales": {
                "x": {"ticks": {"color": THEME["muted"]}},
                "y": _themed_axis("Net R"),
            },
        },
    }
    html = _render_chart(config, height=350)
    # Chart.js tooltip callbacks must be real JS functions, which JSON can't
    # represent -- render the config as JSON first, then splice the
    # function in as raw JS text.
    tooltip_fn = (
        "function(ctx){"
        "var m=ctx.dataset.meta[ctx.dataIndex];"
        "return 'Net R: '+ctx.parsed.y.toFixed(2)+' | '+m.trades+' trades | '+m.win_rate+'% WR';"
        "}"
    )
    return html.replace('"__SESSION_TOOLTIP__"', tooltip_fn)


def _chart_wr_vs_pf(tbl: pd.DataFrame) -> str:
    valid = tbl.dropna(subset=["Win Rate", "Profit Factor"])
    if valid.empty:
        return _render_chart(None, empty_message="No data", height=380)

    net_r  = valid["Net R"].fillna(0)
    trades = valid["Trades"].fillna(1).clip(lower=1)
    points = []
    for name, wr, pf, n, r in zip(valid.index, valid["Win Rate"], valid["Profit Factor"],
                                   trades, net_r):
        points.append({
            "x": round(float(wr), 2), "y": round(float(pf), 2),
            "r": round(float(n) ** 0.5 * 2.5, 1),
            "label": str(name),
        })
    colors = [THEME["green"] if v >= 0 else THEME["red"] for v in net_r]

    x_max = max(100.0, float(valid["Win Rate"].max()) + 5)
    config = {
        "type": "bubble",
        "data": {"datasets": [
            {
                "label": "Models",
                "data": points,
                "backgroundColor": colors,
                "borderColor": THEME["text"],
                "borderWidth": 1,
            },
            {
                "type": "line",
                "label": "PF = 1.0",
                "data": [{"x": 0, "y": 1.0}, {"x": x_max, "y": 1.0}],
                "borderColor": THEME["amber"],
                "borderDash": [4, 4],
                "borderWidth": 1,
                "pointRadius": 0,
                "fill": False,
            },
        ]},
        "options": {
            "responsive": True, "maintainAspectRatio": False,
            "plugins": {
                "legend": {"display": False},
                "title": {"display": True,
                          "text": "Win Rate vs Profit Factor (bubble = trade count)",
                          "color": THEME["text"], "font": {"size": 14}},
                "tooltip": {"callbacks": {"label": "__WRPF_TOOLTIP__"}},
            },
            "scales": {
                "x": _themed_axis("Win Rate %"),
                "y": _themed_axis("Profit Factor"),
            },
        },
    }
    html = _render_chart(config, height=380)
    tooltip_fn = (
        "function(ctx){"
        "if(ctx.dataset.label!=='Models'){return 'PF = 1.0';}"
        "var p=ctx.raw;"
        "return p.label+': WR '+p.x+'%, PF '+p.y;"
        "}"
    )
    return html.replace('"__WRPF_TOOLTIP__"', tooltip_fn)


# ─────────────────────────────────────────────────────────────────────────────
# HTML assembly
# ─────────────────────────────────────────────────────────────────────────────

def _stat_card(label: str, value: str, color: str = None) -> str:
    color = color or THEME["text"]
    return f"""
    <div class="stat-card">
      <div class="stat-value" style="color:{color}">{value}</div>
      <div class="stat-label">{label}</div>
    </div>"""


def _table_html(tbl: pd.DataFrame) -> str:
    rows = []
    for name, row in tbl.iterrows():
        if pd.isna(row.get("Trades")) or row.get("Trades", 0) == 0:
            continue
        net_r = row.get("Net R", 0)
        flag  = THEME["green"] if (not pd.isna(net_r) and net_r > 0) else THEME["red"]
        rows.append(f"""
        <tr>
          <td>{name}</td>
          <td>{int(row['Trades'])}</td>
          <td>{row['Win Rate']:.1f}%</td>
          <td>{row['Profit Factor']:.2f}</td>
          <td>{row['Avg R']:+.3f}</td>
          <td style="color:{flag};font-weight:600">{net_r:+.2f}R</td>
          <td>{row.get('Max DD R', 0):.2f}R</td>
        </tr>""")
    return "".join(rows)


def _embed_ceo_chart(df, symbol, tf) -> str:
    """
    Renders the CEO candlestick chart and extracts just its <body> content
    for inline embedding (plot_chart_html() returns a full standalone
    document with its own <html>/<head>/<body>, which would otherwise
    nest html/head/body tags). The Lightweight Charts CDN <script> tag
    itself lives in that document's <head> and gets dropped here along
    with the rest of <head> -- generate_report()'s own <head> loads the
    same CDN script once for the whole report page, so the embedded
    chart's inline <script> still has `LightweightCharts` available by
    the time it runs.
    """
    try:
        from .chart import plot_chart_html
        full_chart_doc = plot_chart_html(df, symbol=symbol, tf=tf, bars=200)
        body_start = full_chart_doc.find("<body>")
        body_end   = full_chart_doc.find("</body>")
        if body_start == -1 or body_end == -1:
            return full_chart_doc
        return full_chart_doc[body_start + len("<body>"):body_end]
    except Exception as e:
        logger.warning("CEO candlestick chart embed failed for HTML report: %s", e)
        return ""


def _build_wf_section_html(wf_summary: Optional[pd.DataFrame]) -> str:
    """Walk-forward consistency table, or '' if no walk-forward data was passed."""
    if wf_summary is None or wf_summary.empty:
        return ""
    wf_rows = []
    sort_col = "Net R (avg)" if "Net R (avg)" in wf_summary.columns else wf_summary.columns[0]
    for name, row in wf_summary.sort_values(sort_col, ascending=False).head(8).iterrows():
        pw  = row.get("Positive Windows", 0)
        vw  = row.get("Valid Windows", 0)
        consistency = (pw / vw * 100) if vw else 0
        net_r_avg = row.get("Net R (avg)", np.nan)
        color = THEME["green"] if consistency >= 60 else (THEME["amber"] if consistency >= 40 else THEME["red"])
        net_r_str = f"{net_r_avg:+.2f}R" if not pd.isna(net_r_avg) else "—"
        wf_rows.append(f"""
        <tr>
          <td>{name}</td>
          <td>{int(vw)}</td>
          <td style="color:{color};font-weight:600">{pw}/{vw} ({consistency:.0f}%)</td>
          <td>{net_r_str}</td>
        </tr>""")
    return f"""
    <div class="panel">
      <div class="panel-header">🔁 Walk-Forward Consistency</div>
      <div class="panel-body">
        <table class="data-table">
          <thead><tr><th>Model</th><th>Windows</th><th>Profitable Windows</th><th>Avg Net R</th></tr></thead>
          <tbody>{"".join(wf_rows)}</tbody>
        </table>
      </div>
    </div>"""


def _build_journal_section_html(journal_stats: Optional[dict], best_wr: float) -> str:
    """Live-vs-backtest comparison panel, or '' if no journal data was passed."""
    if not journal_stats or journal_stats.get("trades", 0) <= 0:
        return ""
    live_wr = journal_stats.get("win_rate", 0)
    diff    = live_wr - best_wr
    diff_color = THEME["green"] if diff >= -5 else THEME["red"]
    return f"""
    <div class="panel">
      <div class="panel-header">📒 Live Journal Comparison</div>
      <div class="panel-body">
        <div class="stat-grid">
          {_stat_card("Live Trades", str(journal_stats.get("trades",0)))}
          {_stat_card("Live Win Rate", f"{live_wr:.1f}%")}
          {_stat_card("Backtest Win Rate", f"{best_wr:.1f}%")}
          {_stat_card("Divergence", f"{diff:+.1f}pp", diff_color)}
        </div>
      </div>
    </div>"""


def generate_report(
    df:            pd.DataFrame,
    bt:            Dict[str, pd.DataFrame],
    tbl:           pd.DataFrame,
    symbol:        str,
    tf:            str,
    wf_summary:    Optional[pd.DataFrame] = None,
    journal_stats: Optional[dict] = None,
    out_path:      Optional[str] = None,
    include_ceo_chart: bool = True,
) -> str:
    """
    Generate a single self-contained HTML backtest report.

    Parameters
    ----------
    df             : the fully enriched DataFrame (225 columns) used for backtest
    bt             : dict of {model_name: trades_df} from run_backtest()
    tbl            : results_table(bt) output
    symbol, tf     : for the report header
    wf_summary     : optional walk_forward()["summary"] DataFrame
    journal_stats  : optional Journal.performance_stats() dict (live comparison)
    out_path       : output file path (default: ceo_report_{symbol}_{tf}.html)
    include_ceo_chart : whether to embed the candlestick chart (slower to build)

    Returns the path to the saved HTML file.
    """
    out_path = out_path or f"ceo_report_{symbol}_{tf}.html"

    valid_tbl  = tbl.dropna(subset=["Net R"])
    best_name  = valid_tbl["Net R"].idxmax() if not valid_tbl.empty else None
    best_trades= bt.get(best_name, pd.DataFrame()) if best_name else pd.DataFrame()

    # ── Build all figures ────────────────────────────────────────────────────
    chart_equity   = _chart_equity_curve(best_trades, best_name or "—")
    chart_models   = _chart_model_comparison(tbl)
    chart_dist     = _chart_trade_distribution(best_trades)
    chart_dd       = _chart_drawdown(best_trades)
    chart_session  = _chart_session_breakdown(best_trades)
    chart_wrpf     = _chart_wr_vs_pf(tbl)

    if include_ceo_chart:
        try:
            chart_html = _embed_ceo_chart(df, symbol, tf)
        except Exception as e:
            logger.warning("CEO candlestick chart embed raised unexpectedly "
                            "(outside its own guard): %s", e)
            chart_html = ""
    else:
        chart_html = ""

    # ── Header stats ──────────────────────────────────────────────────────────
    n_bars   = len(df)
    date_range = f"{df.index[0].strftime('%Y-%m-%d')} → {df.index[-1].strftime('%Y-%m-%d')}"
    total_trades_all = int(tbl["Trades"].fillna(0).sum())

    best_wr  = valid_tbl.loc[best_name, "Win Rate"]   if best_name else 0
    best_pf  = valid_tbl.loc[best_name, "Profit Factor"] if best_name else 0
    best_netr= valid_tbl.loc[best_name, "Net R"]      if best_name else 0
    best_trd = int(valid_tbl.loc[best_name, "Trades"]) if best_name else 0

    wf_html      = _build_wf_section_html(wf_summary)
    journal_html = _build_journal_section_html(journal_stats, best_wr)

    # ── Assemble HTML ─────────────────────────────────────────────────────────
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>CEO Engine Report — {symbol} {tf}</title>
<script src="{CHARTJS_CDN}"></script>
<script src="{LIGHTWEIGHT_CHARTS_CDN}"></script>
<style>
  :root {{
    --bg: {THEME['bg']}; --panel: {THEME['panel']}; --border: {THEME['border']};
    --text: {THEME['text']}; --muted: {THEME['muted']};
    --green: {THEME['green']}; --red: {THEME['red']}; --amber: {THEME['amber']};
  }}
  * {{ box-sizing: border-box; margin:0; padding:0; }}
  body {{ background:var(--bg); color:var(--text); font-family:'Segoe UI',system-ui,sans-serif; font-size:13px; }}
  .header {{ padding:20px 24px; background:var(--panel); border-bottom:1px solid var(--border); }}
  .header h1 {{ font-size:20px; font-weight:700; }}
  .header h1 span {{ color:var(--amber); }}
  .header .meta {{ color:var(--muted); font-size:12px; margin-top:4px; }}
  .container {{ max-width:1400px; margin:0 auto; padding:20px; }}
  .stat-grid {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(140px,1fr)); gap:12px; margin-bottom:20px; }}
  .stat-card {{ background:var(--panel); border:1px solid var(--border); border-radius:8px; padding:16px; text-align:center; }}
  .stat-value {{ font-size:24px; font-weight:700; margin-bottom:4px; }}
  .stat-label {{ font-size:10px; color:var(--muted); text-transform:uppercase; letter-spacing:0.06em; }}
  .panel {{ background:var(--panel); border:1px solid var(--border); border-radius:8px; margin-bottom:16px; overflow:hidden; }}
  .panel-header {{ padding:12px 16px; border-bottom:1px solid var(--border); font-weight:600; font-size:13px; }}
  .panel-body {{ padding:14px 16px; }}
  .grid-2 {{ display:grid; grid-template-columns:1fr 1fr; gap:16px; }}
  @media (max-width:900px) {{ .grid-2 {{ grid-template-columns:1fr; }} }}
  .data-table {{ width:100%; border-collapse:collapse; }}
  .data-table th {{ text-align:left; padding:8px 10px; font-size:10px; color:var(--muted);
                    text-transform:uppercase; border-bottom:1px solid var(--border); }}
  .data-table td {{ padding:8px 10px; border-bottom:1px solid var(--border); font-size:12px; }}
  .data-table tr:hover td {{ background:rgba(255,255,255,0.02); }}
  .footer {{ text-align:center; color:var(--muted); font-size:11px; padding:24px; }}
</style>
</head>
<body>

<div class="header">
  <h1>🧠 CEO <span>ENGINE</span> — Backtest Report</h1>
  <div class="meta">{symbol} &nbsp;|&nbsp; {tf.upper()} &nbsp;|&nbsp; {n_bars:,} bars &nbsp;|&nbsp; {date_range}
  &nbsp;|&nbsp; Generated {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}</div>
</div>

<div class="container">

  <div class="stat-grid">
    {_stat_card("Best Model", best_name or "—")}
    {_stat_card("Trades", str(best_trd))}
    {_stat_card("Win Rate", f"{best_wr:.1f}%")}
    {_stat_card("Profit Factor", f"{best_pf:.2f}")}
    {_stat_card("Net R", f"{best_netr:+.2f}R", THEME["green"] if best_netr >= 0 else THEME["red"])}
    {_stat_card("Total Trades (all models)", str(total_trades_all))}
  </div>

  <div class="panel">
    <div class="panel-header">📈 Equity Curve — {best_name or "—"}</div>
    <div class="panel-body">{chart_equity}</div>
  </div>

  <div class="grid-2">
    <div class="panel">
      <div class="panel-header">📊 Trade Distribution</div>
      <div class="panel-body">{chart_dist}</div>
    </div>
    <div class="panel">
      <div class="panel-header">📉 Drawdown</div>
      <div class="panel-body">{chart_dd}</div>
    </div>
  </div>

  <div class="panel">
    <div class="panel-header">🕐 Performance by Session</div>
    <div class="panel-body">{chart_session}</div>
  </div>

  <div class="panel">
    <div class="panel-header">🏆 Model Comparison</div>
    <div class="panel-body">{chart_models}</div>
  </div>

  <div class="panel">
    <div class="panel-header">🎯 Win Rate vs Profit Factor</div>
    <div class="panel-body">{chart_wrpf}</div>
  </div>

  <div class="panel">
    <div class="panel-header">📋 Full Results Table</div>
    <div class="panel-body">
      <table class="data-table">
        <thead><tr><th>Model</th><th>Trades</th><th>Win Rate</th><th>Profit Factor</th>
        <th>Avg R</th><th>Net R</th><th>Max DD</th></tr></thead>
        <tbody>{_table_html(tbl)}</tbody>
      </table>
    </div>
  </div>

  {wf_html}
  {journal_html}

  {f'''<div class="panel">
    <div class="panel-header">📈 CEO Candlestick Chart (last 200 bars)</div>
    <div class="panel-body" style="padding:0">{chart_html}</div>
  </div>''' if chart_html else ""}

  <div class="footer">
    The CEO Protocol — Generated automatically. Backtest results do not
    guarantee future performance. Always forward-test before risking capital.
  </div>

</div>
</body>
</html>"""

    html = html.encode("ascii", "xmlcharrefreplace").decode("ascii")
    with open(out_path, "w", encoding="ascii") as f:
        f.write(html)

    return out_path


# ─────────────────────────────────────────────────────────────────────────────
# Self-test
# ─────────────────────────────────────────────────────────────────────────────
