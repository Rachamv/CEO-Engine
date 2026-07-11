"""
The CEO Protocol MT5 — Live Signal Evaluation
==================================================
The per-bar signal pipeline: check_symbol() runs every Phase 1-4 gate for
one symbol/timeframe and fires confirmed signals through console/journal/
CSV/executor/alerts/dashboard via the two helpers below it.

Split out of mt5_live.py (which had grown to 1580 lines) purely for
file-size/maintainability reasons — no behavior changed. See CHANGELOG.
Depends only on mt5_live_utils.py — mt5_live_session.py and mt5_live.py
both import check_symbol() from here, so this file must never import
from either of them (that would create a circular import).
"""

from typing import Optional

from ceo_engine_mt5.ceo_logging import get_logger
logger = get_logger(__name__)

from .indicators import calc_all
from .signals import build_all, build_confluence, MODEL_NAMES
from .candle_patterns import build_candle_patterns
from .ceo_structure import build_ceo_structure
from .patterns import build_patterns
from .session_filter import is_valid_session, add_session_columns
from .risk_engine import RiskEngine
from .funded_account_guard import FundedAccountGuard
from .news_filter import NewsFilter

from .mt5_live_utils import (
    _bar_id, _format_signal, _log_signal, _play_alert,
    _structural_sl, _tp_levels, _rates_to_df, _auto_htf,
)


def _apply_risk_and_guard_gates(
    symbol, tf, direction, close, sl, quality, account_info, sym_info,
    bar_time_utc, risk_engine, guard, open_trade_details, log_path, bar_time, model_label,
):
    """
    Runs the RiskEngine and FundedAccountGuard gates for one candidate
    signal. Returns (lots, lot_info, blocked) — if blocked is True, the
    caller should treat the signal as fully rejected (already CSV-logged
    here); lots/lot_info are only meaningful when blocked is False.
    """
    lot_info = None
    lots = 0.0

    if risk_engine and account_info:
        lots, risk_report = risk_engine.evaluate(
            symbol=symbol, tf=tf, direction=direction.lower(),
            entry=close, sl=sl, quality=quality,
            account=account_info, sym_info=sym_info,
            bar_time=bar_time_utc,
            spread_pips=float(sym_info.get("spread", 0)),
            verbose=False,
        )
        if lots <= 0:
            blocked_by = risk_report.get("blocked_by", "risk_engine")
            if log_path:
                _log_signal(log_path, {
                    "datetime": bar_time.strftime("%Y-%m-%d %H:%M:%S"),
                    "symbol": symbol, "timeframe": tf.upper(),
                    "direction": direction, "model": model_label,
                    "quality": round(quality, 1), "status": "BLOCKED",
                    "block_reason": blocked_by,
                })
            return 0.0, None, True
        lot_info = {"lots": lots,
                    "risk_amount": risk_report.get("sizing", {}).get("risk_amount", 0)}

    if guard and account_info:
        guard_ok, guard_reason = guard.pre_trade_check(
            account=account_info,
            open_trades=len(open_trade_details),
            has_sl=True,
            bar_time=bar_time_utc,
            symbol=symbol,
            direction=direction.lower(),
            open_trade_details=open_trade_details,
            verbose=False,
        )
        if not guard_ok:
            if log_path:
                _log_signal(log_path, {
                    "datetime": bar_time.strftime("%Y-%m-%d %H:%M:%S"),
                    "symbol": symbol, "timeframe": tf.upper(),
                    "direction": direction, "model": model_label,
                    "quality": round(quality, 1), "status": "BLOCKED",
                    "block_reason": guard_reason[:80],
                })
            return lots, lot_info, True

    return lots, lot_info, False


def _route_fired_signal(sig, df, last, lot_info, sound, journal, log_path,
                          executor, alerts, guard, account_info) -> None:
    """
    Fans a confirmed, gate-passed signal out to every configured output:
    console line, optional sound, journal, legacy CSV log, executor
    (auto-trade), Telegram alert, and dashboard. Each output is
    independent — a failure in one (logged, not raised) never blocks the
    others. Pulled out of check_symbol()'s `_fire` closure so each
    concern lives in one readable place.
    """
    symbol, tf, direction = sig["symbol"], sig["tf"], sig["direction"]
    bar_time, bar_time_utc = sig["bar_time"], sig["bar_time_utc"]
    close, atr_val = sig["close"], sig["atr"]
    model_label, quality = sig["model"], sig["quality"]
    sl, tp1, tp2, tp3 = sig["sl"], sig["tp1"], sig["tp2"], sig["tp3"]
    digits = sig["digits"]
    session_name = sig["session"]
    ceo_valid, bos = sig["ceo_valid"], sig["bos"]
    in_discount, pat_name = sig["in_discount"], sig["pat_name"]

    print(_format_signal(
        symbol, tf, direction, bar_time, close, atr_val,
        model_label, quality, str(last["regime_name"]),
        sig["alignment"], sig["htf_label"], sig["conf_count"],
        sl, tp1, tp2, tp3, digits, lot_info,
        ceo_valid=ceo_valid, bos=bos,
        in_discount=in_discount, pat_name=pat_name,
        session=session_name,
    ))

    if sound:
        _play_alert(direction)

    # ── Phase 4: Journal — log signal ───────────────────────────────────────
    if journal:
        try:
            journal.log_signal(
                symbol=symbol, tf=tf, direction=direction.lower(),
                quality=quality, model=model_label,
                bar_time=bar_time_utc,
                entry=close, sl=sl, tp1=tp1, tp2=tp2, tp3=tp3,
                traded=lot_info is not None,
                session=session_name, pat_name=pat_name,
                ceo_valid=ceo_valid, bos_confirmed=bos,
                in_discount=in_discount,
            )
        except Exception as e:
            logger.warning("[%s] journal.log_signal() failed: %s", symbol, e)


    # ── Phase 4: CSV log (legacy) ────────────────────────────────────────────
    if log_path:
        _log_signal(log_path, {
            "datetime":   bar_time.strftime("%Y-%m-%d %H:%M:%S"),
            "symbol":     symbol, "timeframe": tf.upper(),
            "direction":  direction, "model": model_label,
            "quality":    round(quality, 1),
            "session":    session_name,
            "regime":     str(last["regime_name"]),
            "alignment":  sig["alignment"], "htf_bias": sig["htf_label"],
            "conf_count": sig["conf_count"],
            "ceo_valid":  int(ceo_valid), "bos": int(bos),
            "in_discount": int(in_discount), "pat_name": pat_name,
            "entry_ref":  close, "sl_ref": round(sl, 6),
            "tp1_ref":    round(tp1, 6), "tp2_ref": round(tp2, 6),
            "tp3_ref":    round(tp3, 6),
            "lots":       sig.get("lots", ""),
            "status":     "FIRED",
        })

    # ── Phase 2: Executor — place trade ─────────────────────────────────────
    if executor and lot_info and account_info:
        simulation = getattr(executor, "simulation", False)
        is_connected = getattr(executor, "is_connected", lambda: True)
        if not simulation and not is_connected():
            logger.warning("[%s] executor skipped auto-trade because MT5 connection is not available", symbol)
        else:
            try:
                ticket = executor.place_trade(
                    symbol=symbol, tf=tf,
                    direction=direction.lower(),
                    entry=close, sl=sl,
                    tp1=tp1, tp2=tp2, tp3=tp3,
                    quality=quality, model=model_label,
                    bar_time=bar_time_utc, atr=atr_val,
                )
                if ticket and journal:
                    journal.log_trade_open(
                        ticket=ticket, symbol=symbol, tf=tf,
                        direction=direction.lower(),
                        entry=close, sl=sl,
                        tp1=tp1, tp2=tp2, tp3=tp3,
                        lots=sig.get("lots", 0),
                        quality=quality, model=model_label,
                        bar_time=bar_time_utc, session=session_name,
                    )
            except RuntimeError as e:
                if "mt5 disconnected" in str(e).lower():
                    logger.warning("[%s] executor.place_trade() failed due to MT5 disconnect: %s", symbol, e)
                else:
                    logger.error("[%s] executor.place_trade() failed: %s", symbol, e, exc_info=True)
            except Exception as e:
                logger.error("[%s] executor.place_trade() failed: %s", symbol, e, exc_info=True)

    # ── Phase 4: Telegram alert ──────────────────────────────────────────────
    if alerts:
        try:
            chart_path = None
            try:
                from .chart import plot_chart_png
                chart_path = f"/tmp/ceo_{symbol}_{tf}.png"
                plot_chart_png(df, symbol=symbol, tf=tf, bars=150,
                               save_path=chart_path, dpi=100)
            except Exception as e:
                logger.info("[%s] Chart generation for alert failed, "
                            "sending alert without chart: %s", symbol, e)
                chart_path = None

            alerts.signal(
                symbol=symbol, tf=tf, direction=direction.lower(),
                entry=close, sl=sl, tp1=tp1, tp2=tp2, tp3=tp3,
                quality=quality, model=model_label,
                session=session_name, pat_name=pat_name,
                ceo_valid=ceo_valid, bos=bos,
                in_discount=in_discount,
                chart_path=chart_path,
            )
        except Exception as e:
            logger.warning("[%s] Telegram alert failed: %s", symbol, e)

    # ── Phase 4: Dashboard update ────────────────────────────────────────────
    try:
        import ceo_engine_mt5.dashboard as dash
        # Compute daily change pct from the last bar (close vs open of day)
        _daily_chg = None
        try:
            if len(df) >= 2:
                _day_open = float(df["open"].iloc[0])
                if _day_open and _day_open != 0:
                    _daily_chg = round((close - _day_open) / _day_open * 100, 2)
        except Exception:
            pass
        dash.update_signal({
            "symbol": symbol, "tf": tf, "direction": direction.lower(),
            "quality": quality, "model": model_label,
            "session": session_name, "bar_time": bar_time_utc.isoformat(),
            "entry": close, "sl": sl,
            "tp1": tp1, "tp2": tp2, "tp3": tp3,
            "ceo_valid": ceo_valid, "bos": bos,
            "pat_name": pat_name,
            "daily_change_pct": _daily_chg,
            "current_price": close,
        })
        if account_info:
            dash.update_account(account_info)
        if guard:
            dash.update_guard(guard.status(account_info))

        # ── Lightweight Charts candle feed (primary) ─────────────
        try:
            bars = [
                {
                    "time":   int(ts.timestamp()),
                    "open":   float(row["open"]),
                    "high":   float(row["high"]),
                    "low":    float(row["low"]),
                    "close":  float(row["close"]),
                    "volume": float(row.get("volume", 0) or 0),
                }
                for ts, row in df.tail(500).iterrows()
            ]
            dash.update_candles(symbol, tf, bars)
            try:
                from .chart_lwc import build_structure_payload
                dash.update_structure(symbol, tf, build_structure_payload(df))
            except Exception as e:
                logger.warning("[%s] Dashboard structure update failed: %s", symbol, e)
        except Exception as e:
            logger.warning("[%s] Dashboard candle update failed: %s", symbol, e)
            # ── Fallback: legacy HTML chart ───────────────────
            try:
                from .chart import plot_chart_html
                html = plot_chart_html(df, symbol=symbol, tf=tf, bars=150)
                dash.update_chart(symbol, tf, html)
            except Exception as e2:
                logger.warning("[%s] Dashboard chart fallback also failed: %s", symbol, e2)
    except Exception as e:
        logger.warning("[%s] Dashboard update failed: %s", symbol, e)


def check_symbol(conn, symbol, tf, params, signal_params, bt_params,
                 log_path, sound, seen_ids,
                 risk_engine: Optional[RiskEngine] = None,
                 guard: Optional[FundedAccountGuard] = None,
                 executor=None,
                 journal=None,
                 alerts=None,
                 news_filter: Optional[NewsFilter] = None,
                 n_bars: int = 1000):
    """
    Full bar-close pipeline for one symbol.
    Returns list of fired signal dicts.
    """
    sym_info = conn.symbol_info(symbol)
    digits   = sym_info["digits"]
    tick_sz  = sym_info["tick_size"]

    rates = conn.fetch_rates(symbol, tf, n_bars=n_bars)
    df    = _rates_to_df(rates, symbol, tick_sz)

    # HTF data
    htf_tf = params.get("htf_tf") or _auto_htf(tf)
    htf_df = None
    if not params.get("no_htf"):
        try:
            htf_rates = conn.fetch_rates(symbol, htf_tf, n_bars=500)
            htf_df    = _rates_to_df(htf_rates, symbol, tick_sz)
        except Exception as e:
            logger.info("[%s] HTF fetch failed, continuing without HTF bias: %s",
                        symbol, e)

    # ── Full Phase 1-3 pipeline ───────────────────────────────────────────────
    df = calc_all(df, params=params.get("ind_params"))
    df = build_all(df, htf_df=htf_df, params=signal_params)
    df = build_candle_patterns(df)
    df = build_ceo_structure(df)
    # Confluence's quality gate needs the CEO structural bonus that
    # build_ceo_structure() just applied to m00_quality_long/short -- see
    # build_all()'s docstring in signals.py for why this can't run earlier.
    df = build_confluence(df, params=signal_params)
    df = build_patterns(df)
    df = add_session_columns(df, allowed=params.get("sessions", ["all"]))

    last     = df.iloc[-1]
    bar_time = df.index[-1].to_pydatetime().replace(tzinfo=None)
    bar_time_utc = df.index[-1].to_pydatetime()
    close    = float(last["close"])
    atr_val  = float(last["atr"])

    # Account info for live gates
    account_info = {}
    if risk_engine or guard:
        try:
            account_info = conn.account_info()
        except Exception as e:
            logger.warning("[%s] account_info() failed — risk gate will be skipped "
                            "this bar: %s", symbol, e)

    # Open trades for correlation check
    open_trade_details = []
    if executor and guard:
        open_trade_details = [
            {"symbol": t.symbol, "direction": t.direction}
            for t in executor._open_trades.values()
        ]

    # Session check (real-time)
    sess_valid, session_name, q_mult = is_valid_session(
        bar_time_utc,
        allowed=params.get("sessions", ["all"]),
    )

    fired = []
    sl_mult  = bt_params.get("sl_atr_mult", 1.5)
    sl_buf   = bt_params.get("sl_buffer",   0.10)
    tp1_r    = bt_params.get("tp1_r", 1.0)
    tp2_r    = bt_params.get("tp2_r", 2.0)
    tp3_r    = bt_params.get("tp3_r", 3.0)
    model_id = params.get("model_id", 0)

    def _fire(direction, model_label, quality, alignment, conf_count):
        bid = _bar_id(symbol, tf, bar_time, direction)
        if bid in seen_ids:
            return None

        # Session gate (real-time check redundant with session_filter
        # but provides the session name for logging)
        if not sess_valid:
            return None

        # ── News filter gate ──────────────────────────────────────────────────
        if news_filter is not None:
            news_blocked, news_reason = news_filter.is_blocked(
                bar_time=bar_time_utc, symbol=symbol
            )
            if news_blocked:
                logger.warning("[%s] News blocked: %s", symbol, news_reason)
                if log_path:
                    _log_signal(log_path, {
                        "datetime": bar_time.strftime("%Y-%m-%d %H:%M:%S"),
                        "symbol": symbol, "timeframe": tf.upper(),
                        "direction": direction, "model": model_label,
                        "quality": round(quality, 1),
                        "status": "BLOCKED", "block_reason": news_reason[:80],
                    })
                return None

        # Structural SL (CEO method — last swept pivot)
        sl = _structural_sl(last, direction, atr_val, sl_mult, sl_buf)

        # TP levels (structural TP3 if available)
        tp1, tp2, tp3 = _tp_levels(last, direction, close, sl,
                                    tp1_r, tp2_r, tp3_r)

        # CEO structure context
        ceo_valid    = bool(last.get("ceo_long_valid" if direction=="LONG"
                                     else "ceo_short_valid", False))
        bos          = bool(last.get("bos_long" if direction=="LONG"
                                     else "bos_short", False))
        in_discount  = bool(last.get("in_discount", False))
        pat_name     = str(last.get("pat_name", ""))
        cp_confirm   = bool(last.get("cp_bull_confirmation" if direction=="LONG"
                                     else "cp_bear_confirmation", False))

        htf_label = ("Bullish" if last.get("htf_bullish", True)
                     else "Bearish" if last.get("htf_bearish", False)
                     else "Neutral")

        sig = dict(
            symbol=symbol, tf=tf, direction=direction,
            bar_time=bar_time, bar_time_utc=bar_time_utc,
            close=close, atr=atr_val,
            model=model_label, quality=quality,
            regime=str(last["regime_name"]),
            alignment=alignment, htf_label=htf_label,
            conf_count=conf_count, session=session_name,
            sl=sl, tp1=tp1, tp2=tp2, tp3=tp3,
            digits=digits,
            ceo_valid=ceo_valid, bos=bos,
            in_discount=in_discount, pat_name=pat_name,
            cp_confirm=cp_confirm,
        )

        # ── Phase 2: RiskEngine + FundedAccountGuard gates ───────────────────
        lots, lot_info, blocked = _apply_risk_and_guard_gates(
            symbol=symbol, tf=tf, direction=direction, close=close, sl=sl,
            quality=quality, account_info=account_info, sym_info=sym_info,
            bar_time_utc=bar_time_utc, risk_engine=risk_engine, guard=guard,
            open_trade_details=open_trade_details, log_path=log_path,
            bar_time=bar_time, model_label=model_label,
        )
        if blocked:
            return None
        if lot_info:
            sig["lots"] = lots

        # ── Route to console / journal / CSV / executor / alerts / dashboard ─
        _route_fired_signal(
            sig=sig, df=df, last=last, lot_info=lot_info, sound=sound,
            journal=journal, log_path=log_path, executor=executor,
            alerts=alerts, guard=guard, account_info=account_info,
        )

        seen_ids.add(bid)
        return sig

    # ── Fire signals ──────────────────────────────────────────────────────────
    # Each "mode" is a gate column + quality column per direction, plus how
    # to label the model in output. Unified into one loop instead of three
    # near-identical copies (single-model / confluence / CEO-only).
    #
    # ceo_only is exclusive, matching its name and help text ("Only fire
    # when full CEO sequence is valid"): when set, it's the ONLY mode
    # evaluated -- the base single-model mode and Confluence (even if
    # --confluence was also passed) are both skipped, not just
    # supplemented. Previously ceo_only just appended CEO Full Sequence
    # as a third mode without excluding anything else, which didn't
    # match what "only" promised.
    if params.get("ceo_only", False):
        if params.get("confluence", False):
            logger.warning(
                "[%s] both --ceo-only and --confluence are set -- "
                "ceo_only is exclusive, so only CEO Full Sequence will "
                "be evaluated here; --confluence has no effect while "
                "ceo_only is on.", symbol)
        modes = [
            ({"LONG": "ceo_long_valid", "SHORT": "ceo_short_valid"},
             {"LONG": "quality_long",   "SHORT": "quality_short"},
             lambda cc: "CEO Full Sequence"),
        ]
    else:
        modes = [
            ({"LONG": f"m{model_id:02d}_long",        "SHORT": f"m{model_id:02d}_short"},
             {"LONG": f"m{model_id:02d}_quality_long", "SHORT": f"m{model_id:02d}_quality_short"},
             lambda cc: MODEL_NAMES[model_id]),
        ]
        if params.get("confluence", False):
            modes.append(
                ({"LONG": "confluence_long_fired", "SHORT": "confluence_short_fired"},
                 {"LONG": "m00_quality_long",      "SHORT": "m00_quality_short"},
                 lambda cc: f"Confluence ({cc}M)")
            )

    for gate_cols, qual_cols, label_fn in modes:
        for direction in ("LONG", "SHORT"):
            if not last.get(gate_cols[direction], False):
                continue
            quality   = float(last.get(qual_cols[direction], 0))
            alignment = str(last.get("alignment_long" if direction == "LONG"
                                      else "alignment_short", ""))
            cc        = int(last.get("confluence_long_count" if direction == "LONG"
                                      else "confluence_short_count", 0))
            s = _fire(direction, label_fn(cc), quality, alignment, cc)
            if s and s not in fired:
                fired.append(s)

    # Trade management moved to run_live() main thread (every 2s) so
    # floating P&L stays current between bar closes.

    return fired

