"""
Live MT5 runner for the CEO Protocol v3.0.

Launches the MT5 live signal monitor with optional auto-trade,
news filtering, Telegram alerts and dashboard integration.

Usage
-----
python run.py --symbol XAUUSD --tf M15 --live
python run.py --symbol XAUUSD --tf M15 --live --auto-trade
python run.py --symbol XAUUSD --tf M15 --live --dashboard-port 5000
python run.py --symbol XAUUSD --tf M15 --live --telegram-token TOKEN --telegram-chat CHAT_ID
"""

import argparse
import os
import sys
import time

import pandas as pd

# ── Core pipeline ─────────────────────────────────────────────────────────────
from ceo_engine_mt5.data       import fetch_ohlcv, resample_ohlcv
from ceo_engine_mt5.indicators import calc_all
from ceo_engine_mt5.signals    import build_all, build_confluence
from ceo_engine_mt5.backtest   import run_backtest, results_table
from ceo_engine_mt5.visualise  import plot_all

# ── Phase 1 ──────────────────────────────────────────────────────────────────
from ceo_engine_mt5.candle_patterns import build_candle_patterns
from ceo_engine_mt5.ceo_structure   import build_ceo_structure

# ── Phase 3 ──────────────────────────────────────────────────────────────────
from ceo_engine_mt5.patterns       import build_patterns
from ceo_engine_mt5.session_filter import add_session_columns

from ceo_engine_mt5.ceo_logging import get_logger
logger = get_logger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# HTF auto-select
# ─────────────────────────────────────────────────────────────────────────────

_HTF_AUTO = {
    "1m":"15m","2m":"15m","3m":"15m","5m":"15m",
    "15m":"1h","30m":"1h","1h":"4h","2h":"4h",
    "4h":"1d","6h":"1d","8h":"1d","12h":"1d","1d":"1w",
    "m1":"m15","m5":"m15","m15":"h1","m30":"h1",
    "h1":"h4","h4":"d1","d1":"w1",
}



# ─────────────────────────────────────────────────────────────────────────────
# Console helpers
# ─────────────────────────────────────────────────────────────────────────────

def _banner(text, width=62):
    print("\n" + "═"*width)
    print(f"  {text}")
    print("═"*width)

def _section(text):
    print(f"\n── {text} {'─'*max(0,53-len(text))}")

def _print_results(tbl):
    _section("Model Performance")
    print(f"  {'Model':<35} {'Trades':>6}  {'WR%':>6}  {'PF':>6}  {'AvgR':>7}  {'NetR':>8}  {'SL':>6}")
    print("  " + "─"*78)
    for name, row in tbl.iterrows():
        if pd.isna(row["Trades"]) or row["Trades"] == 0:
            print(f"  {'—':1} {name:<33} {'—':>6}  {'—':>6}  {'—':>6}  {'—':>7}  {'—':>8}")
            continue
        wr  = f"{row['Win Rate']:.1f}%"      if not pd.isna(row["Win Rate"])      else "—"
        pf  = f"{row['Profit Factor']:.2f}"  if not pd.isna(row["Profit Factor"]) else "—"
        avgr= f"{row['Avg R']:+.3f}R"        if not pd.isna(row["Avg R"])         else "—"
        netr= f"{row['Net R']:+.2f}R"        if not pd.isna(row["Net R"])         else "—"
        slm = row.get("SL Mode","—")
        flag= "✅" if (not pd.isna(row["Net R"]) and row["Net R"] > 0) else "❌"
        print(f"  {flag} {name:<33} {int(row['Trades']):>6}  {wr:>6}  "
              f"{pf:>6}  {avgr:>7}  {netr:>8}  {slm:>6}")

def _print_best(tbl, min_trades=10):
    _section("Best Systems")
    valid = tbl[tbl["Trades"] >= min_trades].dropna(subset=["Net R"])
    if valid.empty:
        print(f"  No models with >= {min_trades} trades.")
        return
    for metric in ["Net R","Profit Factor","Win Rate","Avg R"]:
        col = valid[metric].dropna()
        if col.empty: continue
        name = col.idxmax()
        val  = col.max()
        print(f"  🏆  Best {metric:<16}: {name}  ({val:.2f})")

def _save_csv(bt, tbl, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    tbl.to_csv(os.path.join(out_dir, "results_summary.csv"))
    print(f"  💾  results_summary.csv")
    all_trades = []
    for name, trades in bt.items():
        if not trades.empty:
            t = trades.copy(); t.insert(0, "model", name)
            all_trades.append(t)
    if all_trades:
        combined = pd.concat(all_trades, ignore_index=True)
        combined.to_csv(os.path.join(out_dir, "all_trades.csv"), index=False)
        print(f"  💾  all_trades.csv  ({len(combined)} trades)")



def _build_parser():
    p = argparse.ArgumentParser(
        description="The CEO Protocol v3 — Master Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    g = p.add_argument_group("Data")
    g.add_argument("--symbol",   nargs="*",
                   help="One or more symbols e.g. XAUUSD EURUSD GBPUSD (omit to start dashboard only)")
    g.add_argument("--tf",       default="1h",
                   help="Timeframe: M5 M15 M30 H1 H4 D1")

    g = p.add_argument_group("MT5 / Live")
    g.add_argument("--live",           action="store_true",
                   help="Launch live MT5 monitor")
    g.add_argument("--mt5-login",      type=int,   default=None)
    g.add_argument("--mt5-password",   type=str,   default=None)
    g.add_argument("--mt5-server",     type=str,   default=None)
    g.add_argument("--reconnect-wait", type=int,   default=30)
    g.add_argument("--bars",           type=int,   default=1000)
    g.add_argument("--sound",          action="store_true")
    g.add_argument("--log",            type=str,   default=None,
                   help="CSV signal log path")
    g.add_argument("--poll",           type=int,   default=5)
    g.add_argument("--model",          type=int,   default=0)

    g = p.add_argument_group("Signal mode")
    g.add_argument("--confluence", action="store_true",
                   help="Use confluence mode (multiple model agreement)")
    g.add_argument("--ceo-only",   action="store_true",
                   help="Only fire when full CEO sequence is valid (highest quality "
                        "gate). Exclusive: overrides --model and --confluence, which "
                        "have no effect while this is set (a warning is logged if "
                        "--confluence is also passed).")
    g.add_argument("--confluence-mode", choices=["sweep", "ceo_structure", "full"],
                   default="sweep",
                   help="What 'confluence' requires: 'sweep' = N of the 16 "
                        "filter-models agree (default); 'ceo_structure' = the "
                        "full CEO sequence validates instead of a model count; "
                        "'full' = both at once. 'ceo_structure'/'full' have no "
                        "effect in --walkforward mode (its per-window rebuild "
                        "doesn't run the CEO structure stage -- falls back to "
                        "'sweep' there with a one-time warning).")
    g.add_argument("--min-models", type=int,   default=3)
    g.add_argument("--min-quality",type=float, default=50.0)
    g.add_argument("--no-align-block", action="store_true")

    g = p.add_argument_group("HTF Bias")
    g.add_argument("--htf",          default=None)
    g.add_argument("--no-htf",       action="store_true")
    g.add_argument("--htf-ema-fast", type=int, default=50)
    g.add_argument("--htf-ema-slow", type=int, default=200)

    g = p.add_argument_group("Indicators")
    g.add_argument("--atr-len",   type=int,   default=14)
    g.add_argument("--ema-fast",  type=int,   default=50)
    g.add_argument("--ema-slow",  type=int,   default=200)
    g.add_argument("--rsi-len",   type=int,   default=14)
    g.add_argument("--rsi-os",    type=float, default=35.0)
    g.add_argument("--rsi-ob",    type=float, default=65.0)
    g.add_argument("--pivot-len", type=int,   default=5)
    g.add_argument("--vol-mult",  type=float, default=1.30)
    g.add_argument("--pool-size", type=int,   default=3)

    g = p.add_argument_group("Sweep Detection")
    g.add_argument("--max-depth", type=float, default=0.80)
    g.add_argument("--min-rej",   type=float, default=0.20)

    g = p.add_argument_group("Session")
    g.add_argument("--sessions", nargs="+",
                   default=["all"],
                   choices=["all","london","new_york","asian","overlap",
                            "pre_london","post_ny"],
                   help="Session names to allow, or 'all' (default) to "
                        "trade anytime the market is open.")

    g = p.add_argument_group("Risk / Backtest")
    g.add_argument("--sl-mode",    default="sweep", choices=["sweep","atr"])
    g.add_argument("--sl",         type=float, default=1.50)
    g.add_argument("--sl-buffer",  type=float, default=0.10)
    g.add_argument("--sl-max",     type=float, default=3.00)
    g.add_argument("--tp1",        type=float, default=1.00)
    g.add_argument("--tp2",        type=float, default=2.00)
    g.add_argument("--tp3",        type=float, default=3.00)
    g.add_argument("--maxbars",    type=int,   default=30)
    g.add_argument("--commission", type=float, default=0.05)
    g.add_argument("--spread-mode",default="fixed_r",
                   choices=["fixed_r","price"])
    g.add_argument("--spread-points",type=float, default=0.0)

    g = p.add_argument_group("Phase 2 — Auto-trade + Risk")
    g.add_argument("--auto-trade",      action="store_true",
                   help="Place trades via MT5 (live mode only)")
    g.add_argument("--risk-pct",        type=float, default=0.0,
                   help="Pct of balance to risk per trade (0 = disabled)")
    g.add_argument("--account-size",    type=float, default=10_000.0)
    g.add_argument("--max-sl-pips",     type=float, default=2000.0)
    g.add_argument("--min-consistency", type=float, default=0.0,
                   help="Min walk-forward consistency fraction (e.g. 0.6)")
    g.add_argument("--daily-loss-pct",  type=float, default=5.0)
    g.add_argument("--max-dd-pct",      type=float, default=10.0)
    g.add_argument("--consistency-pct", type=float, default=0.0,
                   help="Prop firm: max single-day pct of total profits (0=off)")

    g = p.add_argument_group("Phase 4 — Output")
    g.add_argument("--journal",         type=str, default=None,
                   help="SQLite journal path, e.g. ceo_journal.db")
    g.add_argument("--telegram-token",  type=str, default=None)
    g.add_argument("--telegram-chat",   type=str, default=None)
    g.add_argument("--dashboard-port",  type=int, default=None,
                   help="Start web dashboard on this port")

    g = p.add_argument_group("News Filter (live mode)")
    g.add_argument("--no-news-filter",  action="store_true",
                   help="Disable news blocking (not recommended for funded accounts)")
    g.add_argument("--news-pre-mins",   type=int, default=30,
                   help="Minutes to block BEFORE high-impact events (default 30)")
    g.add_argument("--news-post-mins",  type=int, default=15,
                   help="Minutes to block AFTER high-impact events (default 15)")
    g.add_argument("--news-medium",     action="store_true",
                   help="Also block medium-impact events")
    g.add_argument("--fcsapi-key",      default="",
                   help="FCS API key for calendar fallback (optional)")

    g = p.add_argument_group("Multi-Timeframe Stack (live mode)")
    g.add_argument("--mtf",             nargs="+", default=None,
                   help="Enable multi-TF stack, e.g. --mtf H4 H1 M15 "
                        "(lowest TF = entry/execution timeframe)")
    g.add_argument("--mtf-mode",        default="bias",
                   choices=["bias","sweep","ceo","cascade"])
    g.add_argument("--mtf-min-tfs",     type=int, default=None)
    g.add_argument("--mtf-min-score",   type=float, default=40.0)

    g = p.add_argument_group("Performance Feedback Loop (live mode)")
    g.add_argument("--perf-feedback",      action="store_true",
                   help="Enable rolling performance feedback (requires --journal and --risk-pct)")
    g.add_argument("--perf-window",        type=int, default=10)
    g.add_argument("--perf-loss-streak",   type=int, default=3)
    g.add_argument("--perf-wr-floor",      type=float, default=35.0)

    return p


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = _build_parser()
    args   = parser.parse_args()
    t0     = time.time()

    symbols = args.symbol

    ind_params = {
        "atr_len":    args.atr_len,
        "ema_fast_len":args.ema_fast,
        "ema_slow_len":args.ema_slow,
        "rsi_len":    args.rsi_len,
        "rsi_os":     args.rsi_os,
        "rsi_ob":     args.rsi_ob,
        "vol_mult":   args.vol_mult,
        "pivot_len":  args.pivot_len,
    }
    signal_params = {
        "max_sweep_depth_atr": args.max_depth,
        "min_rejection_ratio": args.min_rej,
        "htf_ema_fast":        args.htf_ema_fast,
        "htf_ema_slow":        args.htf_ema_slow,
        "confluence_min_count":args.min_models,
        "min_quality_score":   args.min_quality,
        "htf_require_align":   not args.no_align_block,
        "confluence_mode":     args.confluence_mode,
        "swing_pool_size":     args.pool_size,
    }
    bt_params = {
        "sl_mode":        args.sl_mode,
        "sl_atr_mult":    args.sl,
        "sl_buffer_atr":  args.sl_buffer,
        "sl_max_atr_mult":args.sl_max,
        "tp1_r":          args.tp1,
        "tp2_r":          args.tp2,
        "tp3_r":          args.tp3,
        "max_bars":       args.maxbars,
        "commission_r":   args.commission,
        "spread_mode":    args.spread_mode,
        "spread_points":  args.spread_points,
        "session_filter": len(args.sessions) > 0,
        "active_sessions":args.sessions,
    }

    # If no symbols were provided, start the dashboard-only mode so the
    # app behaves like a standalone dashboard that detects MT5 and
    # populates available symbols without running the engine pipeline.
    if not args.symbol or len(args.symbol) == 0:
        from ceo_engine_mt5 import dashboard
        port = args.dashboard_port or 5000
        dashboard.start_dashboard(port=port)
        print(f"✅  Dashboard started at http://localhost:{port} — MT5 detection and market ticks will populate if MT5 is running.")
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("Stopping dashboard")
        return

    # ── Live mode ──────────────────────────────────────────────────────────────
    if not args.live:
        print("❌  Live mode required when specifying symbols; add --live to run the MT5 pipeline")
        sys.exit(1)

    from ceo_engine_mt5.mt5_live import run_live
    symbols = args.symbol   # already a list — supports 1 to N symbols
    run_live(
        symbols         = symbols,
        tf              = args.tf,
        params          = {
            "model_id": args.model,
            "no_htf":   args.no_htf,
            "htf_tf":   args.htf,
            "sessions": args.sessions,
            "ceo_only": args.ceo_only,
            "confluence": args.confluence,
        },
        signal_params   = signal_params,
        bt_params       = bt_params,
        log_path        = args.log,
        sound           = args.sound,
        poll_interval   = args.poll,
        n_bars          = args.bars,
        risk_pct        = args.risk_pct,
        login           = args.mt5_login,
        password        = args.mt5_password,
        server          = args.mt5_server,
        reconnect_wait  = args.reconnect_wait,
        auto_trade      = args.auto_trade,
        account_size    = args.account_size,
        max_sl_pips     = args.max_sl_pips,
        min_quality     = args.min_quality,
        min_consistency = args.min_consistency,
        daily_loss_pct  = args.daily_loss_pct,
        max_dd_pct      = args.max_dd_pct,
        consistency_pct = args.consistency_pct,
        journal_path    = args.journal,
        telegram_token  = args.telegram_token,
        telegram_chat   = args.telegram_chat,
        dashboard_port  = args.dashboard_port,
        block_news      = not getattr(args, 'no_news_filter', False),
        news_pre_mins   = getattr(args, 'news_pre_mins', 30),
        news_post_mins  = getattr(args, 'news_post_mins', 15),
        news_medium     = getattr(args, 'news_medium', False),
        fcsapi_key      = getattr(args, 'fcsapi_key', ""),
        mtf_tfs         = getattr(args, 'mtf', None),
        mtf_mode        = getattr(args, 'mtf_mode', "bias"),
        mtf_min_tfs     = getattr(args, 'mtf_min_tfs', None),
        mtf_min_score   = getattr(args, 'mtf_min_score', 40.0),
        perf_feedback         = getattr(args, 'perf_feedback', False),
        perf_rolling_window   = getattr(args, 'perf_window', 10),
        perf_loss_streak      = getattr(args, 'perf_loss_streak', 3),
        perf_win_rate_floor   = getattr(args, 'perf_wr_floor', 35.0),
    )
    return


# ─────────────────────────────────────────────────────────────────────────────
# Programmatic API
# ─────────────────────────────────────────────────────────────────────────────

def _run_walkforward_validation(df, wf_windows, ind_params, signal_params, bt_params,
                                  htf_tf, tf, verbose, out, risk_pct, min_trades,
                                  min_consistency, max_sl_pips, sessions, tbl, symbol):
    """Runs walk-forward validation and, if risk-sizing is enabled, registers
    the result with a RiskEngine so live sizing reflects walk-forward consistency."""
    if verbose:
        _section("Walk-Forward")
    from ceo_engine_mt5.walkforward import walk_forward, print_wf_report, save_wf_results
    wf = walk_forward(
        df, n_windows=wf_windows,
        ind_params=ind_params, signal_params=signal_params,
        bt_params=bt_params,
        htf_tf=htf_tf or _auto_htf(tf),
        min_trades=5,
        verbose=verbose,
    )
    if verbose:
        print_wf_report(wf)
    save_wf_results(wf, out_dir=out)

    if risk_pct > 0:
        from ceo_engine_mt5.risk_engine import RiskEngine
        engine = RiskEngine(
            risk_pct        = risk_pct,
            min_trades      = min_trades,
            min_consistency = min_consistency,
            max_sl_pips     = max_sl_pips,
            sessions        = sessions,
        )
        engine.register_walkforward(tbl, wf["summary"], symbol, tf)

    return wf


def _generate_html_report_safe(df, bt, tbl, symbol, tf, wf, journal_stats, out, verbose):
    """Generates the HTML report, logging (not raising) on failure since a
    report-generation problem shouldn't take down the rest of the run."""
    if verbose:
        _section("HTML Report")
    try:
        from ceo_engine_mt5.report import generate_report
        report_path = os.path.join(
            out, f"ceo_report_{symbol.replace('/','_').replace('=','')}_{tf}.html"
        )
        generate_report(
            df, bt, tbl,
            symbol=symbol, tf=tf,
            wf_summary=wf["summary"] if wf else None,
            journal_stats=journal_stats,
            out_path=report_path,
            include_ceo_chart=True,
        )
        if verbose:
            print(f"  💾  {report_path}")
    except Exception as e:
        logger.warning("HTML report generation failed: %s", e)
        if verbose:
            print(f"  ⚠️  HTML report generation failed: {e}")


def run(
    symbol,
    tf            = "1h",
    source        = "yfinance",
    start         = "2022-01-01",
    end           = None,
    exchange      = "binance",
    filepath      = None,
    htf_tf        = None,
    no_htf        = False,
    ind_params    = None,
    signal_params = None,
    bt_params     = None,
    sessions      = None,
    out_dir       = None,
    no_charts     = False,
    no_csv        = False,
    min_trades    = 10,
    walkforward   = False,
    wf_windows    = 6,
    risk_pct      = 0.0,
    min_consistency = 0.0,
    max_sl_pips   = 2000.0,
    html_report   = False,
    journal_stats = None,
    verbose       = True,
):
    """
    Programmatic entry point. Runs the full Phase 1-3 pipeline
    + backtest and returns (bt_dict, results_table, enriched_df).

    Example
    -------
    from run import run
    bt, tbl, df = run("XAUUSD", tf="H1", source="yfinance",
                       start="2022-01-01", sessions=["london","new_york"])
    """
    t0  = time.time()
    out = out_dir or f"results_{symbol.replace('/','_').replace('=','')}_{tf}"
    os.makedirs(out, exist_ok=True)
    if sessions is None:
        sessions = ["all"]   # trade anytime the market is open

    if verbose:
        _banner(f"CEO Engine v3  |  {symbol} [{tf}]")
        _section("Data")

    df = fetch_ohlcv(symbol=symbol, timeframe=tf, source=source,
                     start=start, end=end, exchange=exchange, filepath=filepath)

    if verbose:
        print(f"  Bars: {len(df):,}  |  "
              f"{df.index[0].date()} → {df.index[-1].date()}")

    htf_df = None
    if not no_htf:
        htf_timeframe = htf_tf or _auto_htf(tf)
        if verbose: _section(f"HTF [{htf_timeframe}]")
        htf_df = resample_ohlcv(df, htf_timeframe)

    # Full Phase 1-3 pipeline
    df = build_pipeline(
        df,
        htf_df        = htf_df,
        ind_params    = ind_params,
        signal_params = signal_params,
        sessions      = sessions,
        verbose       = verbose,
    )

    if verbose: _section("Backtest")
    print()
    bt  = run_backtest(df, params=bt_params)
    tbl = results_table(bt)
    if verbose:
        _print_results(tbl)
        _print_best(tbl, min_trades=min_trades)

    wf = None
    if walkforward:
        wf = _run_walkforward_validation(
            df, wf_windows, ind_params, signal_params, bt_params,
            htf_tf, tf, verbose, out, risk_pct, min_trades,
            min_consistency, max_sl_pips, sessions, tbl, symbol,
        )

    if html_report:
        _generate_html_report_safe(df, bt, tbl, symbol, tf, wf, journal_stats, out, verbose)

    if not no_csv:
        _save_csv(bt, tbl, out)
    if not no_charts:
        plot_all(bt, symbol=f"{symbol} [{tf}]", output_dir=out)

    if verbose:
        _banner(f"Done in {time.time()-t0:.1f}s  |  {out}/")

    return bt, tbl, df
