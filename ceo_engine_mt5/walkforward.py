"""
The CEO Protocol — Walk-Forward Validator  v2.0
====================================================
Splits historical data into N equal windows and runs the full
backtest pipeline on each window independently.

Answers: is the best model genuinely robust, or curve-fitted?

Usage
-----
from .walkforward import walk_forward, print_wf_report

results = walk_forward(df, n_windows=6, min_trades=5)
print_wf_report(results)

CLI:
python walkforward.py --source csv --file eurusd.csv --windows 6
"""

import pandas as pd
import numpy as np
from typing import Optional, Dict

from .indicators import calc_all
from .signals    import build_all, build_confluence

from ceo_engine_mt5.ceo_logging import get_logger
logger = get_logger(__name__)
from .backtest   import run_backtest, results_table, DEFAULT_BT_PARAMS
from .data       import resample_ohlcv


# ─────────────────────────────────────────────────────────────────────────────
# Core
# ─────────────────────────────────────────────────────────────────────────────

def walk_forward(
    df:            pd.DataFrame,
    n_windows:     int   = 6,
    min_trades:    int   = 5,
    metric:        str   = "Net R",
    ind_params:    Optional[dict] = None,
    signal_params: Optional[dict] = None,
    bt_params:     Optional[dict] = None,
    htf_tf:        Optional[str]  = None,
    verbose:       bool  = True,
) -> Dict:
    """
    Walk-forward validation across N equal time windows.

    Parameters
    ----------
    df            : OHLCV DataFrame from fetch_ohlcv()
    n_windows     : number of windows to split into
    min_trades    : minimum trades to consider a model valid
    metric        : selection metric — "Net R"|"Profit Factor"|"Win Rate"|"Avg R"
    ind_params    : indicator param overrides
    signal_params : signal param overrides
    bt_params     : backtest param overrides
    htf_tf        : HTF timeframe for bias filter (auto-resample if provided)
    verbose       : print progress

    Returns
    -------
    dict with keys:
        windows       : list of per-window result dicts
        summary       : DataFrame — per-model stability across windows
        best_overall  : name of most consistently best model
        consistency   : float 0–1, how often best model repeats
        metric        : metric used
        n_windows     : n_windows
    """
    p_bt = {**DEFAULT_BT_PARAMS, **(bt_params or {})}

    bars_per_window = len(df) // n_windows
    if bars_per_window < 200:
        raise ValueError(
            f"Too few bars per window ({bars_per_window}). "
            f"Use fewer windows or more historical data."
        )

    if verbose:
        print(f"\n{'═'*60}")
        print(f"  Walk-Forward Validation")
        print(f"  Windows   : {n_windows}  |  Bars/window : {bars_per_window:,}")
        print(f"  Metric    : {metric}  |  Min trades : {min_trades}")
        print(f"  Range     : {df.index[0].date()} → {df.index[-1].date()}")
        print(f"{'═'*60}\n")

    windows = []

    # This per-window rebuild never runs build_ceo_structure() (see the
    # long comment at the build_confluence() call below), so
    # ceo_long_valid/ceo_short_valid never exist here -- confluence_mode
    # "ceo_structure"/"full" would raise on every single window otherwise.
    # Forced back to "sweep" here specifically, with a one-time warning,
    # rather than letting the global CLI setting silently crash walk-forward.
    wf_signal_params = dict(signal_params or {})
    requested_mode = wf_signal_params.get("confluence_mode", "sweep")
    if requested_mode != "sweep":
        logger.warning(
            "confluence_mode=%r requires build_ceo_structure(), which "
            "walk-forward's per-window rebuild doesn't run -- using "
            "'sweep' for walk-forward's Confluence evaluation instead.",
            requested_mode)
        wf_signal_params["confluence_mode"] = "sweep"

    for w in range(n_windows):
        start_idx  = w * bars_per_window
        end_idx    = (start_idx + bars_per_window
                      if w < n_windows - 1 else len(df))
        window_df  = df.iloc[start_idx:end_idx].copy()
        start_date = window_df.index[0].strftime("%Y-%m-%d")
        end_date   = window_df.index[-1].strftime("%Y-%m-%d")

        if verbose:
            print(f"  Window {w+1}/{n_windows}: {start_date} → {end_date} "
                  f"({len(window_df):,} bars)", end="  ")

        window_df = calc_all(window_df, params=ind_params)

        htf_df = None
        if htf_tf:
            try:
                htf_df = resample_ohlcv(window_df, htf_tf)
            except Exception as e:
                logger.warning("HTF resample failed for walk-forward window, "
                                "continuing without HTF bias: %s", e)

        window_df = build_all(window_df, htf_df=htf_df, params=wf_signal_params)
        # NOTE: this window-rebuild loop does not run build_ceo_structure()
        # (candle patterns / CEO sequence / geometric patterns), unlike
        # every other full-pipeline call site in this codebase -- so
        # Confluence's quality gate here still can't clear the same way
        # it now can elsewhere (see signals.build_all()'s docstring for
        # the full story). Calling build_confluence() right after
        # build_all(), in the same position it used to run internally,
        # preserves walk-forward's existing behavior (the column exists,
        # the "Confluence" row still applies, it just still won't produce
        # trades in "sweep" mode) rather than crashing run_backtest() below
        # on a missing column. Fixing that properly means adding
        # build_ceo_structure() (and ideally the pattern stages) to this
        # per-window rebuild too -- a real performance tradeoff worth
        # deciding deliberately, since it'd then run once per rolling
        # window instead of once.
        window_df = build_confluence(window_df, params=wf_signal_params)
        bt        = run_backtest(window_df, params=p_bt)
        tbl       = results_table(bt)

        valid = tbl[tbl["Trades"] >= min_trades].dropna(subset=[metric])
        if valid.empty:
            best_model = "—"
            best_value = np.nan
        else:
            best_model = valid[metric].idxmax()
            best_value = valid.loc[best_model, metric]

        windows.append({
            "window":     w + 1,
            "start":      start_date,
            "end":        end_date,
            "bars":       len(window_df),
            "best_model": best_model,
            "best_value": best_value,
            "metric":     metric,
            "table":      tbl,
        })

        if verbose:
            if pd.isna(best_value):
                print(f"→ no valid model (min {min_trades} trades)")
            else:
                print(f"→ {best_model}  ({metric}: {best_value:+.2f})")

    summary                   = _build_summary(windows, min_trades, metric)
    best_overall, consistency = _find_consistent_winner(windows)

    if verbose:
        print(f"\n{'─'*60}")
        print(f"  Best overall : {best_overall}")
        print(f"  Consistency  : {consistency:.0%} of windows")
        print(f"{'─'*60}")

    return {
        "windows":      windows,
        "summary":      summary,
        "best_overall": best_overall,
        "consistency":  consistency,
        "metric":       metric,
        "n_windows":    n_windows,
    }


def _build_summary(windows, min_trades, metric):
    from .signals import MODEL_NAMES
    all_models = list(MODEL_NAMES.values()) + ["Confluence"]
    rows = []
    for name in all_models:
        net_rs = []; win_rates = []; trade_counts = []
        for w in windows:
            tbl = w["table"]
            if name not in tbl.index: continue
            row = tbl.loc[name]
            t   = row["Trades"]
            if pd.isna(t) or t < min_trades: continue
            if not pd.isna(row["Net R"]):    net_rs.append(row["Net R"])
            if not pd.isna(row["Win Rate"]): win_rates.append(row["Win Rate"])
            trade_counts.append(t)
        n_valid = len(net_rs)
        n_pos   = sum(1 for v in net_rs if v > 0)
        rows.append({
            "Model":            name,
            "Valid Windows":    n_valid,
            "Positive Windows": n_pos,
            "Win Rate (avg)":   round(float(np.nanmean(win_rates)), 1) if win_rates else np.nan,
            "Net R (avg)":      round(float(np.nanmean(net_rs)),    2) if net_rs    else np.nan,
            "Net R (min)":      round(float(np.nanmin(net_rs)),     2) if net_rs    else np.nan,
            "Net R (max)":      round(float(np.nanmax(net_rs)),     2) if net_rs    else np.nan,
            "Net R (std)":      round(float(np.nanstd(net_rs)),     2) if net_rs    else np.nan,
            "Avg Trades/Win":   round(float(np.nanmean(trade_counts)), 1) if trade_counts else np.nan,
        })
    return pd.DataFrame(rows).set_index("Model")


def _find_consistent_winner(windows):
    from collections import Counter
    winners = [w["best_model"] for w in windows if w["best_model"] != "—"]
    if not winners:
        return "—", 0.0
    counts      = Counter(winners)
    best        = counts.most_common(1)[0][0]
    consistency = counts[best] / len(windows)
    return best, consistency


# ─────────────────────────────────────────────────────────────────────────────
# Report printer
# ─────────────────────────────────────────────────────────────────────────────

def print_wf_report(results: dict, top_n: int = 8):
    windows  = results["windows"]
    summary  = results["summary"]
    metric   = results["metric"]

    print(f"\n{'═'*68}")
    print(f"  WALK-FORWARD REPORT  |  {results['n_windows']} windows  |  metric: {metric}")
    print(f"{'═'*68}")

    print(f"\n  {'Win':>3}  {'Period':<23}  {'Bars':>5}  {'Best Model':<33}  {metric:>8}")
    print(f"  {'─'*3}  {'─'*23}  {'─'*5}  {'─'*33}  {'─'*8}")
    for w in windows:
        val    = f"{w['best_value']:+.2f}" if not pd.isna(w.get("best_value", np.nan)) else "—"
        period = f"{w['start']} → {w['end']}"
        print(f"  {w['window']:>3}  {period:<23}  {w['bars']:>5}  "
              f"{w['best_model']:<33}  {val:>8}")

    print(f"\n  Best overall : {results['best_overall']}")
    print(f"  Consistency  : {results['consistency']:.0%} of windows")

    print(f"\n{'─'*68}")
    print(f"  Per-Model Stability — top {top_n} by avg Net R")
    print(f"{'─'*68}")

    valid = summary.dropna(subset=["Net R (avg)"])
    valid = valid[valid["Valid Windows"] > 0]
    top   = valid.sort_values("Net R (avg)", ascending=False).head(top_n)

    print(f"  {'Model':<33}  {'Valid':>5}  {'Pos':>4}  "
          f"{'WR%':>5}  {'AvgR':>6}  {'MinR':>6}  {'MaxR':>6}  {'Std':>5}")
    print(f"  {'─'*33}  {'─'*5}  {'─'*4}  {'─'*5}  {'─'*6}  {'─'*6}  {'─'*6}  {'─'*5}")

    for name, row in top.iterrows():
        wr  = f"{row['Win Rate (avg)']:.1f}" if not pd.isna(row["Win Rate (avg)"]) else "—"
        avr = f"{row['Net R (avg)']:+.2f}"   if not pd.isna(row["Net R (avg)"])   else "—"
        mnr = f"{row['Net R (min)']:+.2f}"   if not pd.isna(row["Net R (min)"])   else "—"
        mxr = f"{row['Net R (max)']:+.2f}"   if not pd.isna(row["Net R (max)"])   else "—"
        std = f"{row['Net R (std)']:.2f}"    if not pd.isna(row["Net R (std)"])   else "—"
        all_pos = (not pd.isna(row["Net R (avg)"]) and
                   row["Positive Windows"] == row["Valid Windows"] and
                   row["Valid Windows"] > 0)
        flag = "✅" if all_pos else "  "
        print(f"  {flag} {name:<31}  "
              f"{int(row['Valid Windows'] or 0):>5}  "
              f"{int(row['Positive Windows'] or 0):>4}  "
              f"{wr:>5}  {avr:>6}  {mnr:>6}  {mxr:>6}  {std:>5}")

    print(f"\n  ✅ = positive Net R in ALL valid windows (most robust)")
    print(f"{'═'*68}\n")


# ─────────────────────────────────────────────────────────────────────────────
# CSV export
# ─────────────────────────────────────────────────────────────────────────────

def save_wf_results(results: dict, out_dir: str = "."):
    import os
    os.makedirs(out_dir, exist_ok=True)
    win_rows = [{
        "window":     w["window"],
        "start":      w["start"],
        "end":        w["end"],
        "bars":       w["bars"],
        "best_model": w["best_model"],
        "best_value": w.get("best_value", np.nan),
        "metric":     w["metric"],
    } for w in results["windows"]]
    pd.DataFrame(win_rows).to_csv(
        os.path.join(out_dir, "wf_windows.csv"), index=False)
    results["summary"].to_csv(
        os.path.join(out_dir, "wf_summary.csv"))
    print(f"  💾  {out_dir}/wf_windows.csv")
    print(f"  💾  {out_dir}/wf_summary.csv")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse, sys
    from .data import fetch_ohlcv

    p = argparse.ArgumentParser(description="CEO Engine — Walk-Forward Validator")
    p.add_argument("--symbol",     default="EURUSD")
    p.add_argument("--tf",         default="1h")
    p.add_argument("--source",     default="csv",
                   choices=["yfinance","ccxt","csv","mt5"])
    p.add_argument("--file",       default=None)
    p.add_argument("--start",      default="2022-01-01")
    p.add_argument("--end",        default=None)
    p.add_argument("--exchange",   default="binance")
    p.add_argument("--windows",    type=int,   default=6)
    p.add_argument("--min-trades", type=int,   default=5)
    p.add_argument("--metric",     default="Net R",
                   choices=["Net R","Profit Factor","Win Rate","Avg R"])
    p.add_argument("--htf",        default=None)
    p.add_argument("--out",        default="wf_results")
    p.add_argument("--sl",         type=float, default=1.5)
    p.add_argument("--sl-mode",    default="sweep", choices=["sweep","atr"])
    p.add_argument("--session-filter", action="store_true")
    args = p.parse_args()

    df = fetch_ohlcv(symbol=args.symbol, timeframe=args.tf,
                     source=args.source, start=args.start, end=args.end,
                     exchange=args.exchange, filepath=args.file)

    results = walk_forward(
        df,
        n_windows     = args.windows,
        min_trades    = args.min_trades,
        metric        = args.metric,
        bt_params     = {"sl_mode": args.sl_mode,
                         "sl_atr_mult": args.sl,
                         "session_filter": args.session_filter},
        htf_tf        = args.htf,
        verbose       = True,
    )
    print_wf_report(results)
    save_wf_results(results, out_dir=args.out)
    print(f"Results saved to: {args.out}/")
