"""
The CEO Protocol MT5 — Live Signal Monitor  v3.0
=====================================================
Full integration of all Phase 1-4 modules:

  Phase 1 — candle_patterns.py    36 candle pattern detectors
             ceo_structure.py     BOS, OB, QM, Fib zones, structural SL
  Phase 2 — risk_engine.py        position sizing + model selection gate
             funded_account_guard  prop firm rules + correlation block
             executor.py          MT5 order placement + TP/SL management
  Phase 3 — patterns.py           16 geometric chart patterns
             session_filter.py    session quality multipliers
  Phase 4 — journal.py            SQLite trade logging
             alerts.py            Telegram alerts
             dashboard.py         Web UI
             chart.py             PNG + HTML charts

Signal pipeline (every bar close):
    data → indicators → signals → candle_patterns → ceo_structure
    → patterns → session_filter → RiskEngine gate → FundedAccountGuard gate
    → Executor.place_trade() → Journal + Alert + Dashboard

This file is the CLI entry point and run_live() orchestrator. The actual
implementation is split across:
    mt5_live_utils.py    — shared low-level helpers (lot sizing, dedup IDs,
                            signal formatting, structural SL/TP)
    mt5_live_signals.py  — check_symbol() and its risk/guard gates + the
                            fired-signal routing to journal/executor/alerts
    mt5_live_session.py  — component init, model registration, symbol
                            validation, MTF signal handling, trade
                            management tick, shutdown summary
(split out purely for file-size/maintainability — see CHANGELOG)

Usage
-----
# Basic signal monitor (no auto-trade)
python mt5_live.py --symbol XAUUSD --tf M15

# Full auto-trade with all Phase 1-4 modules
python mt5_live.py --symbol XAUUSD --tf M15 --auto-trade \\
    --risk-pct 1.0 --account-size 10000 \\
    --telegram-token YOUR_TOKEN --telegram-chat YOUR_CHAT \\
    --journal ceo_journal.db --dashboard

# Multi-symbol with funded account guard
python mt5_live.py --symbol XAUUSD EURUSD --tf H1 --auto-trade \\
    --risk-pct 1.0 --account-size 10000 \\
    --daily-loss-pct 5.0 --max-dd-pct 10.0 --consistency-pct 15.0

# Signal-only with walk-forward validated model selection
python mt5_live.py --symbol XAUUSD --tf M15 --min-consistency 0.6
"""

import sys, os, time, argparse
from datetime import datetime, timezone
from typing import Set

from ceo_engine_mt5.ceo_logging import get_logger
logger = get_logger(__name__)

# ── Split-out implementation modules (see module docstring above) ───────────
from .mt5_connect import get_mt5_timeframe
from .mt5_live_utils import _load_seen_ids, _try_connect, _seconds_to_bar_close
from .mt5_live_signals import check_symbol
from .mt5_live_session import (
    _register_models_for_symbols, _validate_symbols,
    _trade_management_tick, _init_live_components, _handle_bar_close,
    _print_shutdown_summary,
)


def _print_startup_banner(symbols, tf, sessions, auto_trade, risk_pct, block_news,
                            perf_monitor, log_path, journal_path, alerts_obj,
                            dashboard_port) -> None:
    """Prints the live-monitor startup summary (symbols, sessions, which Phase 2-4
    components are active) before the per-symbol threads launch."""
    print(f"\n{'═'*60}")
    print(f"  The CEO Protocol v3.0 — Live Monitor")
    print(f"  Symbols   : {', '.join(symbols)}  ({len(symbols)} threads)")
    print(f"  Timeframe : {tf.upper()}")
    print(f"  Sessions  : {', '.join(sessions)}")
    print(f"  Auto-trade: {'ON' if auto_trade else 'OFF'}")
    if risk_pct > 0:
        print(f"  Risk      : {risk_pct}% per trade")
    print(f"  News gate : {'ON' if block_news else 'OFF'}")
    print(f"  Perf loop : {'ON' if perf_monitor else 'OFF'}")
    print(f"  Log       : {log_path or 'disabled'}")
    print(f"  Journal   : {journal_path or 'disabled'}")
    print(f"  Telegram  : {'on' if alerts_obj else 'off'}")
    print(f"  Dashboard : {'http://localhost:' + str(dashboard_port) if dashboard_port else 'off'}")
    print(f"  Ctrl+C to stop")
    print(f"{'═'*60}\n")


def run_live(
    symbols,
    tf,
    params,
    signal_params,
    bt_params,
    log_path       = None,
    sound          = False,
    poll_interval  = 5,
    n_bars         = 1000,
    risk_pct       = 0.0,
    login          = None,
    password       = None,
    server         = None,
    reconnect_wait = 30,
    # Phase 2
    auto_trade     = False,
    account_size   = 10_000.0,
    max_sl_pips    = 2000.0,
    min_quality    = 50.0,
    min_consistency= 0.0,
    daily_loss_pct = 5.0,
    max_dd_pct     = 10.0,
    consistency_pct= 0.0,
    # News filter
    block_news     = True,
    news_pre_mins  = 30,
    news_post_mins = 15,
    news_medium    = False,
    fcsapi_key     = "",
    # Multi-timeframe stack
    mtf_tfs        = None,     # e.g. ["H4","H1","M15"] — None = single-TF mode
    mtf_mode       = "bias",
    mtf_min_tfs    = None,
    mtf_min_score  = 40.0,
    # Performance feedback loop
    perf_feedback         = False,
    perf_rolling_window   = 10,
    perf_loss_streak      = 3,
    perf_win_rate_floor   = 35.0,
    perf_check_every_bars = 1,
    # Phase 4
    journal_path   = None,
    telegram_token = None,
    telegram_chat  = None,
    dashboard_port = None,
):
    import MetaTrader5 as mt5

    # ── Initialise Phase 2-4 components ──────────────────────────────────────
    components = _init_live_components(
        params=params, symbols=symbols, tf=tf, signal_params=signal_params,
        risk_pct=risk_pct, min_quality=min_quality, min_consistency=min_consistency,
        max_sl_pips=max_sl_pips,
        auto_trade=auto_trade, account_size=account_size,
        daily_loss_pct=daily_loss_pct, max_dd_pct=max_dd_pct,
        consistency_pct=consistency_pct,
        journal_path=journal_path, telegram_token=telegram_token,
        telegram_chat=telegram_chat, dashboard_port=dashboard_port,
        block_news=block_news, news_pre_mins=news_pre_mins,
        news_post_mins=news_post_mins, news_medium=news_medium,
        fcsapi_key=fcsapi_key,
        mtf_tfs=mtf_tfs, mtf_mode=mtf_mode, mtf_min_tfs=mtf_min_tfs,
        mtf_min_score=mtf_min_score,
        perf_feedback=perf_feedback, perf_rolling_window=perf_rolling_window,
        perf_loss_streak=perf_loss_streak, perf_win_rate_floor=perf_win_rate_floor,
    )
    risk_engine  = components["risk_engine"]
    guard        = components["guard"]
    executor     = components["executor"]
    journal      = components["journal"]
    alerts_obj   = components["alerts_obj"]
    news_filter  = components["news_filter"]
    mtf_stack    = components["mtf_stack"]
    perf_monitor = components["perf_monitor"]
    sessions     = components["sessions"]

    # ── Print startup banner ──────────────────────────────────────────────────
    _print_startup_banner(symbols, tf, sessions, auto_trade, risk_pct, block_news,
                           perf_monitor, log_path, journal_path, alerts_obj,
                           dashboard_port)

    seen_ids: Set[str] = _load_seen_ids(log_path)
    total_signals  = 0

    # ── Shared state for multi-symbol threading ───────────────────────────────
    import threading
    _lock          = threading.Lock()
    _last_bar      = {s: None for s in symbols}
    _sym_signals   = {s: 0    for s in symbols}
    _stop_event    = threading.Event()

    def _monitor_symbol(symbol: str, conn_local):
        """
        Per-symbol monitoring thread.
        Each symbol gets its own bar-close detection loop.
        MT5 read calls (copy_rates_from_pos) are thread-safe.
        Pipeline CPU work runs independently per thread.
        Shared resources (journal, guard, executor, alerts)
        are protected via _lock where needed.
        """
        nonlocal total_signals
        tf_const = get_mt5_timeframe(tf)

        while not _stop_event.is_set():
            try:
                rates = mt5.copy_rates_from_pos(symbol, tf_const, 0, 2)
                if rates is None or len(rates) == 0:
                    time.sleep(poll_interval)
                    continue

                cur_bar = datetime.fromtimestamp(rates[0]["time"], timezone.utc)

                with _lock:
                    already_seen = (_last_bar[symbol] == cur_bar)

                if not already_seen:
                    with _lock:
                        _last_bar[symbol] = cur_bar

                    now = datetime.now(timezone.utc)
                    print(f"[{now.strftime('%H:%M:%S')}]  "
                          f"{symbol} {tf.upper()} bar closed "
                          f"@ {cur_bar.strftime('%Y-%m-%d %H:%M')}",
                          end="", flush=True)

                    n = _handle_bar_close(
                        symbol=symbol, conn_local=conn_local, cur_bar=cur_bar, tf=tf,
                        mtf_stack=mtf_stack, bt_params=bt_params, seen_ids=seen_ids,
                        risk_engine=risk_engine, guard=guard, journal=journal,
                        alerts_obj=alerts_obj, executor=executor, risk_pct=risk_pct,
                        params=params, signal_params=signal_params, log_path=log_path,
                        sound=sound, news_filter=news_filter, n_bars=n_bars,
                        perf_monitor=perf_monitor, lock=_lock,
                    )
                    with _lock:
                        total_signals        += n
                        _sym_signals[symbol]  += n

                # Sleep until near the next bar close for this symbol
                remaining  = _seconds_to_bar_close(tf)
                sleep_time = min(remaining - 15, 55) if remaining > 20 else poll_interval
                time.sleep(max(sleep_time, 1))

            except Exception as e:
                logger.warning("[%s] Monitor loop iteration failed: %s", symbol, e)
                print(f"\n  ⚠️  [{symbol}] {e}")
                time.sleep(poll_interval)

    while True:
        conn = _try_connect(login, password, server,
                            retry_interval=reconnect_wait)

        # Wire executor to live MT5 connection
        if executor:
            executor.conn = conn
            executor.simulation = False

        # Give the dashboard a live executor + connection reference for its
        # manual trade routes. Done here (not in _init_live_components)
        # because `conn` only exists once the connection above succeeds --
        # also re-run on every reconnect so the dashboard never holds a
        # stale connection.
        if dashboard_port and executor:
            import ceo_engine_mt5.dashboard as dash
            dash.set_executor(executor, conn)

        # Register models from a quick backtest if risk_engine is present
        if risk_engine and not risk_engine.model_selector._registry:
            _register_models_for_symbols(
                conn=conn, symbols=symbols, tf=tf, signal_params=signal_params,
                sessions=sessions, risk_engine=risk_engine, perf_monitor=perf_monitor,
                dashboard_port=dashboard_port,
            )

        # Validate symbols
        if not _validate_symbols(conn, symbols):
            return
        print()

        try:
            # ── Launch one thread per symbol ──────────────────────────────────
            _stop_event.clear()
            threads = []
            for symbol in symbols:
                t = threading.Thread(
                    target=_monitor_symbol,
                    args=(symbol, conn),
                    name=f"ceo-{symbol}",
                    daemon=True,
                )
                t.start()
                threads.append(t)
                time.sleep(0.2)   # stagger thread starts to avoid MT5 burst

            if len(symbols) > 1:
                print(f"  🔀  {len(symbols)} symbol threads running concurrently")

            # Main thread: manage open trades every 2s and push to dashboard.
            # This is independent of per-symbol bar-close cadence so floating
            # P&L stays current even between bar closes on XAUUSD.
            while all(t.is_alive() for t in threads):
                _trade_management_tick(executor, journal, guard, alerts_obj, dashboard_port, perf_monitor=perf_monitor)
                time.sleep(2)

        except KeyboardInterrupt:
            _stop_event.set()
            _print_shutdown_summary(
                total_signals=total_signals, symbols=symbols,
                sym_signals=_sym_signals, log_path=log_path,
                journal=journal, alerts_obj=alerts_obj, guard=guard, conn=conn,
            )
            conn.disconnect()
            break

        except Exception as e:
            print(f"\n⚠️  Connection lost: {e}")
            print(f"   Stopping threads and reconnecting in {reconnect_wait}s...")
            _stop_event.set()
            try:
                conn.disconnect()
            except Exception as e:
                logger.warning("disconnect() during reconnect failed (continuing): %s", e)
            time.sleep(reconnect_wait)




def check_symbol_wrapper(*args, **kwargs):
    """Alias for backward compatibility."""
    return check_symbol(*args, **kwargs)
