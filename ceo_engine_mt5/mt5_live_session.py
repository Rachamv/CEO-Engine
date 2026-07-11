"""
The CEO Protocol MT5 — Live Session Lifecycle
==================================================
Everything run_live() needs to set up and run a live monitoring session:
component initialization (RiskEngine/Guard/Executor/Journal/Alerts/
Dashboard/NewsFilter/MTF-stack/PerformanceMonitor), per-symbol model
registration via a quick backtest, symbol validation, the multi-timeframe
signal path, the per-bar-close dispatch (single-TF or MTF), the open-
trade management tick, and the shutdown summary.

Split out of mt5_live.py (which had grown to 1580 lines) purely for
file-size/maintainability reasons — no behavior changed. See CHANGELOG.
Depends on mt5_live_utils.py and mt5_live_signals.py (for check_symbol);
mt5_live.py imports from here for run_live(), so this file must never
import from mt5_live.py (that would create a circular import).
"""

import os
from datetime import datetime, timezone
from typing import Set

from ceo_engine_mt5.ceo_logging import get_logger
logger = get_logger(__name__)

from .indicators import calc_all
from .signals import build_all, build_confluence
from .candle_patterns import build_candle_patterns
from .ceo_structure import build_ceo_structure
from .patterns import build_patterns
from .session_filter import add_session_columns

from .risk_engine import RiskEngine
from .funded_account_guard import FundedAccountGuard
from .news_filter import make_news_filter
from .multi_tf import make_mtf_stack
from .performance_monitor import PerformanceMonitor

from .mt5_live_utils import _auto_htf, _bar_id, _rates_to_df, _structural_sl, _tp_levels
from .mt5_live_signals import check_symbol


def _handle_mtf_signal(
    mtf_stack, symbol: str, conn_local, tf: str, bt_params: dict,
    cur_bar, seen_ids: Set[str],
    risk_engine, guard, journal, alerts_obj, executor, risk_pct: float,
    news_filter=None,
) -> list:
    """
    Runs one multi-timeframe confirmation check for `symbol` and, if a
    valid cross-TF signal fires, computes SL/TP, sizes it, and routes it
    through journal/alerts/executor exactly as the single-TF path does.

    Returns the list of fired MTFResult objects (empty if none/invalid).
    Pulled out of the `_monitor_symbol` thread loop in run_live() so that
    loop stays short enough to actually read.
    """
    result = mtf_stack.check(symbol, conn_local, verbose=False)
    fired: list = []

    # Push candle data to dashboard chart on EVERY check, not just when a
    # signal validates — otherwise the chart stays empty for a symbol in
    # MTF mode until the first signal fires (could be hours/days).
    if result.entry_df is not None and len(result.entry_df) > 0:
        try:
            import ceo_engine_mt5.dashboard as dash
            bars = [
                {
                    "time":   int(ts.timestamp()),
                    "open":   float(row["open"]),
                    "high":   float(row["high"]),
                    "low":    float(row["low"]),
                    "close":  float(row["close"]),
                    "volume": float(row.get("volume", 0) or 0),
                }
                for ts, row in result.entry_df.tail(500).iterrows()
            ]
            dash.update_candles(symbol, result.entry_tf, bars)
            try:
                from .chart_lwc import build_structure_payload
                dash.update_structure(symbol, result.entry_tf,
                                       build_structure_payload(result.entry_df))
            except Exception as e:
                logger.debug("[%s] Dashboard structure push failed (MTF): %s", symbol, e)
        except Exception as e:
            logger.debug("[%s] Dashboard candle push failed (MTF): %s", symbol, e)

    if not result.valid:
        return fired

    # ── News filter gate — must run here too, not just the single-TF path,
    # otherwise enabling MTF silently bypasses news protection entirely.
    if news_filter is not None:
        entry_bar_time = (result.entry_last.name.to_pydatetime()
                          if result.entry_last is not None and hasattr(result.entry_last, "name")
                          else datetime.now(timezone.utc))
        news_blocked, news_reason = news_filter.is_blocked(
            bar_time=entry_bar_time, symbol=symbol
        )
        if news_blocked:
            print(f"\n  🚫  [{symbol}] {news_reason} (MTF)")
            return fired

    print(result.summary())

    # Build SL/TP from entry_tf bar (structural)
    entry_last = result.entry_last
    close      = float(entry_last["close"]) if entry_last is not None else 0.0
    atr_val    = float(entry_last.get("atr", 0.0)) if entry_last is not None else 0.0
    sl_mult    = bt_params.get("sl_atr_mult", 1.5)
    sl_buf     = bt_params.get("sl_buffer", 0.10)
    tp1_r      = bt_params.get("tp1_r", 1.0)
    tp2_r      = bt_params.get("tp2_r", 2.0)
    tp3_r      = bt_params.get("tp3_r", 3.0)

    sl = _structural_sl(entry_last, result.direction, atr_val, sl_mult, sl_buf)
    tp1, tp2, tp3 = _tp_levels(entry_last, result.direction, close, sl,
                               tp1_r, tp2_r, tp3_r)

    bid = _bar_id(symbol, tf, bar_time=cur_bar, direction=result.direction)
    if bid in seen_ids:
        return fired
    seen_ids.add(bid)

    account_info = {}
    if risk_engine or guard:
        try:
            account_info = conn_local.account_info()
        except Exception as e:
            logger.warning("[%s] account_info() failed during MTF signal: %s", symbol, e)

    sym_info = conn_local.symbol_info(symbol)
    lots = 0.0
    if risk_engine and account_info:
        lots, _ = risk_engine.evaluate(
            symbol=symbol, tf=tf,
            direction=result.direction.lower(),
            entry=close, sl=sl, quality=result.score,
            account=account_info, sym_info=sym_info,
            bar_time=result.bar_time,
            spread_pips=float(sym_info.get("spread", 0)),
            verbose=False,
        )

    if journal:
        try:
            journal.log_signal(
                symbol=symbol, tf=tf,
                direction=result.direction.lower(),
                quality=result.score,
                model=f"MTF-{result.mode}-{'+'.join(result.tfs)}",
                bar_time=result.bar_time,
                entry=close, sl=sl, tp1=tp1, tp2=tp2, tp3=tp3,
                traded=lots > 0,
            )
        except Exception as e:
            logger.warning("[%s] journal.log_signal() failed: %s", symbol, e)

    if alerts_obj and (lots > 0 or risk_pct == 0):
        try:
            alerts_obj.signal(
                symbol=symbol, tf=tf,
                direction=result.direction.lower(),
                entry=close, sl=sl, tp1=tp1, tp2=tp2, tp3=tp3,
                quality=result.score,
                model=f"MTF {'+'.join(result.tfs)}",
                session="",
            )
        except Exception as e:
            logger.warning("[%s] Telegram alert failed: %s", symbol, e)

    # Push to dashboard Signal cards + Market Watch — without this, MTF mode
    # signals never appear in the dashboard even though they're journaled
    # and alerted correctly. Mirrors the single-TF push in mt5_live_signals.py.
    try:
        import ceo_engine_mt5.dashboard as dash
        _daily_chg = None
        try:
            day_open = float(cur_bar.get("open", close)) if hasattr(cur_bar, "get") else close
            if day_open:
                _daily_chg = round((close - day_open) / day_open * 100, 2)
        except Exception:
            pass
        dash.update_signal({
            "symbol": symbol, "tf": tf, "direction": result.direction.lower(),
            "quality": result.score, "model": f"MTF {'+'.join(result.tfs)}",
            "session": "", "bar_time": result.bar_time.isoformat()
                if hasattr(result.bar_time, "isoformat") else str(result.bar_time),
            "entry": close, "sl": sl,
            "tp1": tp1, "tp2": tp2, "tp3": tp3,
            "daily_change_pct": _daily_chg,
            "current_price": close,
        })
        if account_info:
            dash.update_account(account_info)
        if guard:
            dash.update_guard(guard.status(account_info))
    except Exception as e:
        logger.debug("[%s] Dashboard signal push failed (MTF): %s", symbol, e)

    if executor and lots > 0 and account_info:
        simulation = getattr(executor, "simulation", False)
        is_connected = getattr(executor, "is_connected", lambda: True)
        if not simulation and not is_connected():
            logger.warning("[%s] executor skipped auto-trade because MT5 connection is not available", symbol)
        else:
            try:
                executor.place_trade(
                    symbol=symbol, tf=tf,
                    direction=result.direction.lower(),
                    entry=close, sl=sl, tp1=tp1, tp2=tp2, tp3=tp3,
                    quality=result.score,
                    model=f"MTF {'+'.join(result.tfs)}",
                    bar_time=result.bar_time, atr=atr_val,
                )
            except Exception as e:
                logger.error("[%s] executor.place_trade() failed: %s", symbol, e, exc_info=True)

    return [result]


def _register_models_for_symbols(conn, symbols, tf, signal_params, sessions,
                                  risk_engine, perf_monitor,
                                  dashboard_port=None) -> None:
    """
    Runs a quick backtest per symbol to seed RiskEngine.model_selector with
    a starting best-model choice before live trading begins. Failures for
    one symbol don't block the others — logged and skipped instead.
    """
    logger.info("Running quick backtest to select best model...")
    try:
        from .backtest import run_backtest, results_table
    except Exception as e:
        logger.error("Quick backtest setup failed (import error): %s", e, exc_info=True)
        return

    for sym in symbols:
        try:
            rates  = conn.fetch_rates(sym, tf, n_bars=1000)
            sym_i  = conn.symbol_info(sym)
            df_bt  = _rates_to_df(rates, sym, sym_i["tick_size"])
            df_bt  = calc_all(df_bt)
            htf_tf = _auto_htf(tf)
            try:
                htf_r = conn.fetch_rates(sym, htf_tf, n_bars=500)
                htf_d = _rates_to_df(htf_r, sym, sym_i["tick_size"])
            except Exception as e:
                logger.info("[%s] HTF fetch for model registration failed, "
                            "continuing without HTF bias: %s", sym, e)
                htf_d = None
            df_bt = build_all(df_bt, htf_df=htf_d, params=signal_params)
            df_bt = build_candle_patterns(df_bt)
            df_bt = build_ceo_structure(df_bt)
            # See signals.build_all()'s docstring: confluence needs the
            # CEO structural bonus build_ceo_structure() just applied.
            df_bt = build_confluence(df_bt, params=signal_params)
            df_bt = build_patterns(df_bt)
            df_bt = add_session_columns(df_bt, allowed=sessions)
            tbl   = results_table(run_backtest(df_bt))
            selected = risk_engine.register_backtest(tbl, sym, tf)
            if perf_monitor and selected and selected in tbl.index:
                perf_monitor.register_backtest_winrate(
                    sym, tf, float(tbl.loc[selected, "Win Rate"])
                )
            # Push backtest results to dashboard Backtest tab
            if dashboard_port:
                try:
                    import ceo_engine_mt5.dashboard as dash
                    rows = tbl.reset_index().to_dict(orient="records")
                    dash.update_backtest(sym, tf, rows)
                except Exception as _de:
                    logger.debug("Dashboard backtest push failed: %s", _de)
        except Exception as e:
            logger.warning("Quick backtest failed for %s: %s", sym, e)


def _validate_symbols(conn, symbols) -> bool:
    """
    Confirms every symbol resolves on the connected broker before starting
    live monitor threads. Returns False (and disconnects) on the first
    symbol that doesn't exist — there's no useful partial-success mode
    here, since a typo'd symbol means a thread would spin forever.
    """
    for sym in symbols:
        try:
            info = conn.symbol_info(sym)
            print(f"  ✅  {sym:<12} digits={info['digits']}  "
                  f"spread={info['spread']}  tick={info['tick_size']}")
        except ValueError as e:
            logger.error("Symbol validation failed for %s: %s", sym, e)
            print(f"  ❌  {sym}: {e}")
            conn.disconnect()
            return False
    return True


def _trade_management_tick(executor, journal, guard, alerts_obj, dashboard_port, perf_monitor=None) -> None:
    """
    One management pass over open trades: TP1/TP2/SL trail via the
    executor, then fan the resulting close events out to journal/guard/
    alerts/dashboard. Runs on a fixed 2s cadence from the main thread,
    independent of per-symbol bar-close timing.
    """
    if executor is None:
        return

    try:
        events = executor.manage_open_trades()
    except Exception as e:
        logger.error("executor.manage_open_trades() failed: %s", e, exc_info=True)
        return

    for ev in events:
        tr = ev.get("trade")
        if not tr:
            continue
        if journal:
            try:
                journal.log_trade_close(tr.ticket, tr.close_price,
                    tr.close_time, tr.close_reason, tr.pnl or 0.0,
                    tr.tp1_hit, tr.tp2_hit)
            except Exception as e:
                logger.warning("journal.log_trade_close() failed for ticket %s: %s",
                                tr.ticket, e)
        if guard:
            try:
                guard.record_closed_trade(tr.pnl or 0.0,
                    trade_date=tr.close_time, verbose=False)
            except Exception as e:
                logger.warning("guard.record_closed_trade() failed for ticket %s: %s",
                                tr.ticket, e)
        if alerts_obj:
            try:
                alerts_obj.trade_closed(tr.ticket, tr.symbol,
                    tr.direction, tr.close_reason,
                    tr.close_price, tr.pnl or 0.0,
                    tr.tp1_hit, tr.tp2_hit)
            except Exception as e:
                logger.warning("Telegram trade_closed alert failed for ticket %s: %s",
                                tr.ticket, e)

    # Push performance feedback status to dashboard (risk_pct/quality
    # adjustments made by PerformanceMonitor.update(), which runs in the
    # per-symbol bar-close path — this just surfaces the latest state).
    if dashboard_port and perf_monitor is not None:
        try:
            import ceo_engine_mt5.dashboard as dash
            dash.update_stats(perf_monitor.status())
        except Exception as e:
            logger.debug("Dashboard perf_monitor push failed: %s", e)

    # Detect guard halt transition and fire alert exactly once
    if guard and alerts_obj:
        try:
            g_status = guard.status()
            was_halted = getattr(_trade_management_tick, "_last_halt_state", False)
            if g_status.get("halted") and not was_halted:
                alerts_obj.guard_halt(
                    reason         = g_status.get("halt_reason", "Risk limit breached"),
                    daily_loss_pct = g_status.get("daily_loss_used_pct", 0),
                    dd_pct         = g_status.get("drawdown_used_pct", 0),
                )
            _trade_management_tick._last_halt_state = g_status.get("halted", False)
        except Exception as e:
            logger.debug("Guard halt alert check failed: %s", e)

    if dashboard_port:
        try:
            import ceo_engine_mt5.dashboard as dash
            dash.update_trades(executor.get_open_trades())
        except Exception as e:
            logger.warning("Dashboard update_trades() failed: %s", e)
        # Push live bid/ask ticks for Market Watch from MT5 directly
        if executor and executor.conn:
            try:
                import ceo_engine_mt5.dashboard as dash
                open_trades = executor.get_open_trades()
                syms_to_tick = {t.get("symbol") for t in open_trades if t.get("symbol")}
                for sym in syms_to_tick:
                    tick = executor.conn.symbol_info_tick(sym)
                    if tick:
                        dash.update_market_tick(sym, tick["bid"], tick["ask"])
            except Exception as e:
                logger.debug("Market tick push failed: %s", e)


def _init_live_components(
    params, symbols, tf, signal_params,
    risk_pct, min_quality, min_consistency, max_sl_pips,
    auto_trade, account_size, daily_loss_pct, max_dd_pct, consistency_pct,
    journal_path, telegram_token, telegram_chat, dashboard_port,
    block_news, news_pre_mins, news_post_mins, news_medium, fcsapi_key,
    mtf_tfs, mtf_mode, mtf_min_tfs, mtf_min_score,
    perf_feedback, perf_rolling_window, perf_loss_streak, perf_win_rate_floor,
) -> dict:
    """
    Builds every opt-in Phase 2-4 component for run_live() from CLI/config
    values, printing a status line for each one that gets enabled. Each
    component is independently gated — this function does the gating so
    run_live() itself doesn't have to.

    Returns a dict with keys: risk_engine, guard, executor, journal,
    alerts_obj, news_filter, mtf_stack, perf_monitor, sessions. Any
    component not enabled is None in the returned dict.
    """
    risk_engine = None
    guard       = None
    executor    = None
    journal     = None
    alerts_obj  = None
    news_filter = None
    mtf_stack   = None
    perf_monitor = None

    sessions = params.get("sessions", ["all"])

    if risk_pct > 0:
        risk_engine = RiskEngine(
            risk_pct    = risk_pct,
            sessions    = sessions,
            min_quality = min_quality,
            min_consistency = min_consistency,
            max_sl_pips = max_sl_pips,
        )
        print(f"  ✅  RiskEngine: {risk_pct}% per trade")

    if auto_trade or daily_loss_pct > 0:
        guard = FundedAccountGuard(
            account_size          = account_size,
            daily_loss_limit_pct  = daily_loss_pct,
            max_drawdown_pct      = max_dd_pct,
            consistency_pct       = consistency_pct,
        )
        print(f"  ✅  FundedAccountGuard: DL={daily_loss_pct}% DD={max_dd_pct}%")

    if telegram_token and telegram_chat:
        from .alerts import AlertSystem
        import os, json as _json
        # Load per-event alert toggles from dashboard config if present —
        # the Settings tab writes these; CLI users get sensible defaults.
        alert_cfg = {}
        try:
            cfg_path = "ceo_engine_config.json"
            if os.path.exists(cfg_path):
                with open(cfg_path) as f:
                    alert_cfg = _json.load(f)
        except Exception:
            pass
        alerts_obj = AlertSystem.from_config({
            "telegram_token": telegram_token,
            "telegram_chat":  telegram_chat,
            **alert_cfg,
        })
        alerts_obj.system_alert("CEO Engine live monitor started", level="info")
        print(f"  ✅  Telegram alerts: connected")

    if auto_trade and risk_engine and guard:
        from .executor import Executor
        executor = Executor(
            connection  = None,    # MT5Connection set after connect
            risk_engine = risk_engine,
            guard       = guard,
            journal_file= journal_path.replace(".db",".csv") if journal_path else None,
            alerts_obj  = alerts_obj,   # may be None if Telegram not configured
        )
        print(f"  ✅  Executor: auto-trade enabled")

    if journal_path:
        from .journal import Journal
        journal = Journal(journal_path)
        print(f"  ✅  Journal: {journal_path}")


    if dashboard_port:
        import ceo_engine_mt5.dashboard as dash
        if journal_path:
            dash.set_journal(journal_path)
        dash.start_dashboard(port=dashboard_port)
        print(f"  ✅  Dashboard: http://localhost:{dashboard_port}")
        # Note: the executor is wired to the dashboard (dash.set_executor)
        # by the caller once the live MT5 connection exists -- `conn` isn't
        # available yet at this point in startup.

    # ── News filter ───────────────────────────────────────────────────────────
    if block_news:
        news_filter = make_news_filter(
            symbols      = symbols,
            block_high   = True,
            block_medium = news_medium,
            pre_mins     = news_pre_mins,
            post_mins    = news_post_mins,
            fcsapi_key   = fcsapi_key,
            offline      = False,
            verbose      = True,
        )
        print(f"  ✅  News filter: HIGH events blocked "
              f"(-{news_pre_mins}min / +{news_post_mins}min post)")
    else:
        print(f"  ⚠️  News filter: DISABLED (not recommended for funded accounts)")

    # ── Multi-timeframe stack ─────────────────────────────────────────────────
    if mtf_tfs:
        mtf_stack = make_mtf_stack(
            tfs           = mtf_tfs,
            mode          = mtf_mode,
            min_tfs       = mtf_min_tfs,
            entry_tf      = tf,
            sessions      = sessions,
            min_score     = mtf_min_score,
            signal_params = signal_params,
        )
        print(f"  ✅  Multi-TF stack: {' → '.join(mtf_stack.tfs)}  "
              f"mode={mtf_mode}  min_tfs={mtf_stack.min_tfs}  "
              f"min_score={mtf_min_score}")

    # ── Performance feedback loop ────────────────────────────────────────────
    if perf_feedback and journal and risk_engine:
        perf_monitor = PerformanceMonitor(
            journal              = journal,
            risk_engine          = risk_engine,
            baseline_risk_pct    = risk_pct,
            rolling_window       = perf_rolling_window,
            loss_streak_trigger  = perf_loss_streak,
            win_rate_floor       = perf_win_rate_floor,
        )
        print(f"  ✅  Performance feedback: window={perf_rolling_window} trades, "
              f"loss-streak-trigger={perf_loss_streak}, "
              f"WR-floor={perf_win_rate_floor}%")
    elif perf_feedback:
        print(f"  ⚠️  Performance feedback requires both --journal and "
              f"--risk-pct > 0 — skipping")

    return {
        "risk_engine": risk_engine, "guard": guard, "executor": executor,
        "journal": journal, "alerts_obj": alerts_obj, "news_filter": news_filter,
        "mtf_stack": mtf_stack, "perf_monitor": perf_monitor, "sessions": sessions,
    }


def _handle_bar_close(
    symbol, conn_local, cur_bar, tf, mtf_stack, bt_params, seen_ids,
    risk_engine, guard, journal, alerts_obj, executor, risk_pct,
    params, signal_params, log_path, sound, news_filter, n_bars,
    perf_monitor, lock,
) -> int:
    """
    Runs the signal check for one freshly-closed bar (MTF-stack mode or
    single-TF mode, whichever is configured), then the performance
    feedback update. Returns the number of signals that fired, for the
    caller to fold into its running totals under its own lock.
    """
    if mtf_stack is not None:
        fired = _handle_mtf_signal(
            mtf_stack=mtf_stack, symbol=symbol, conn_local=conn_local,
            tf=tf, bt_params=bt_params, cur_bar=cur_bar,
            seen_ids=seen_ids, risk_engine=risk_engine, guard=guard,
            journal=journal, alerts_obj=alerts_obj, executor=executor,
            risk_pct=risk_pct, news_filter=news_filter,
        )
        if not fired:
            print("  — no MTF confirmation")
        n = len(fired)
    else:
        # Single-TF mode (existing pipeline)
        fired = check_symbol(
            conn=conn_local, symbol=symbol, tf=tf,
            params=params,
            signal_params=signal_params,
            bt_params=bt_params,
            log_path=log_path,
            sound=sound,
            seen_ids=seen_ids,
            risk_engine=risk_engine,
            guard=guard,
            executor=executor,
            journal=journal,
            alerts=alerts_obj,
            news_filter=news_filter,
            n_bars=n_bars,
        )
        if fired:
            print(f"  ← {len(fired)} signal(s)!")
        else:
            print("  — no signal")
        n = len(fired)

    # Performance feedback update — runs after every bar-close check
    # regardless of mode, so risk_pct/min_quality stay current with
    # live results.
    if perf_monitor is not None:
        with lock:
            perf_monitor.update(symbol=symbol, tf=tf, verbose=True)

    return n


def _print_shutdown_summary(total_signals, symbols, sym_signals, log_path,
                             journal, alerts_obj, guard, conn) -> None:
    """Prints the Ctrl+C shutdown summary and sends a final Telegram digest."""
    print(f"\n\nStopped. Total signals this session: {total_signals}")

    if len(symbols) > 1:
        print(f"\n── Signals per symbol ──")
        for sym, count in sorted(sym_signals.items(), key=lambda x: x[1], reverse=True):
            print(f"  {sym:<15}: {count}")

    if log_path and os.path.exists(log_path):
        print(f"\nLog: {log_path}")
    if journal:
        journal.print_summary()
    if alerts_obj:
        try:
            acct = conn.account_info()
            if journal:
                alerts_obj.daily_summary(
                    stats=journal.performance_stats(),
                    guard_status=guard.status(acct) if guard else None,
                )
        except Exception as e:
            logger.warning("Shutdown daily-summary alert failed: %s", e)

