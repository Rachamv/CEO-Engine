"""
Tests for the live MT5 glue layer — mt5_live_signals.py and
mt5_live_session.py. These modules talk to a real MT5 terminal in
production, so (following the same injected-dependency pattern
test_executor.py already uses for Executor) they're exercised here via
small fake/mock connection, risk-engine, guard, journal, and executor
objects rather than a live MT5 session.

Lower priority than the pure decision-logic modules (ceo_structure,
candle_patterns, patterns, multi_tf) since this layer is mostly routing
and error containment around calls into code that's already tested
elsewhere (RiskEngine, FundedAccountGuard, Executor). The focus here is
on that routing/error-containment behavior itself:
  - _apply_risk_and_guard_gates: risk + guard gate sequencing, CSV-log
    on block, "blocked" short-circuit semantics
  - _validate_symbols: fail-fast on first bad symbol, disconnect on failure
  - _trade_management_tick: fans close events out to journal/guard/alerts
    without raising even when any one downstream component blows up
"""

from datetime import datetime, timezone
from types import SimpleNamespace

import pytest

import pandas as pd

from ceo_engine_mt5 import mt5_live_session as mls
from ceo_engine_mt5.mt5_live_signals import _apply_risk_and_guard_gates
from ceo_engine_mt5.mt5_live_session import (
    _validate_symbols, _trade_management_tick,
    _register_models_for_symbols, _init_live_components,
)


# ─────────────────────────────────────────────────────────────────────────────
# _apply_risk_and_guard_gates
# ─────────────────────────────────────────────────────────────────────────────

class _FakeRiskEngine:
    def __init__(self, lots=0.10, blocked_by=None):
        self._lots = 0.0 if blocked_by else lots
        self._blocked_by = blocked_by

    def evaluate(self, **kwargs):
        report = {"blocked_by": self._blocked_by, "sizing": {"risk_amount": 25.0}}
        return self._lots, report


class _FakeGuard:
    def __init__(self, ok=True, reason=""):
        self._ok = ok
        self._reason = reason

    def pre_trade_check(self, **kwargs):
        return self._ok, self._reason


def _gate_kwargs(**overrides):
    kwargs = dict(
        symbol="XAUUSD", tf="H1", direction="long", close=2350.0, sl=2345.0,
        quality=72.0, account_info={"balance": 10000.0}, sym_info={"spread": 1.2},
        bar_time_utc=datetime.now(timezone.utc), risk_engine=None, guard=None,
        open_trade_details=[], log_path=None,
        bar_time=datetime.now(timezone.utc), model_label="LQ",
    )
    kwargs.update(overrides)
    return kwargs


class TestApplyRiskAndGuardGates:
    def test_no_engines_configured_passes_through_unblocked(self):
        lots, lot_info, blocked = _apply_risk_and_guard_gates(**_gate_kwargs())
        assert blocked is False
        assert lots == 0.0
        assert lot_info is None

    def test_risk_engine_sizes_the_trade_when_approved(self):
        lots, lot_info, blocked = _apply_risk_and_guard_gates(
            **_gate_kwargs(risk_engine=_FakeRiskEngine(lots=0.25)))
        assert blocked is False
        assert lots == 0.25
        assert lot_info["lots"] == 0.25
        assert lot_info["risk_amount"] == 25.0

    def test_risk_engine_rejection_blocks_before_guard_runs(self):
        guard = _FakeGuard(ok=True)
        lots, lot_info, blocked = _apply_risk_and_guard_gates(
            **_gate_kwargs(risk_engine=_FakeRiskEngine(blocked_by="max_spread"), guard=guard))
        assert blocked is True
        assert lots == 0.0

    def test_guard_rejection_blocks_even_with_valid_risk_sizing(self):
        lots, lot_info, blocked = _apply_risk_and_guard_gates(**_gate_kwargs(
            risk_engine=_FakeRiskEngine(lots=0.10),
            guard=_FakeGuard(ok=False, reason="Daily loss limit reached"),
        ))
        assert blocked is True
        # lots from the risk engine are still returned even though blocked,
        # in case the caller wants to log what would have been sized.
        assert lots == 0.10

    def test_guard_approval_after_risk_sizing_passes_through(self):
        lots, lot_info, blocked = _apply_risk_and_guard_gates(**_gate_kwargs(
            risk_engine=_FakeRiskEngine(lots=0.10),
            guard=_FakeGuard(ok=True),
        ))
        assert blocked is False
        assert lots == 0.10

    def test_risk_engine_skipped_without_account_info(self):
        lots, lot_info, blocked = _apply_risk_and_guard_gates(**_gate_kwargs(
            risk_engine=_FakeRiskEngine(lots=0.10), account_info=None,
        ))
        # account_info is None -> "if risk_engine and account_info" is falsy,
        # risk engine never runs, no blocking.
        assert blocked is False
        assert lots == 0.0

    def test_block_writes_csv_log_when_log_path_given(self, tmp_path):
        log_path = str(tmp_path / "signals.csv")
        _apply_risk_and_guard_gates(**_gate_kwargs(
            risk_engine=_FakeRiskEngine(blocked_by="max_daily_trades"),
            log_path=log_path,
        ))
        import os
        assert os.path.exists(log_path)
        content = open(log_path).read()
        assert "BLOCKED" in content
        assert "max_daily_trades" in content


# ─────────────────────────────────────────────────────────────────────────────
# _validate_symbols
# ─────────────────────────────────────────────────────────────────────────────

class _FakeConnAllValid:
    def __init__(self):
        self.disconnected = False

    def symbol_info(self, sym):
        return {"digits": 2, "spread": 1.5, "tick_size": 0.01}

    def disconnect(self):
        self.disconnected = True


class _FakeConnOneInvalid:
    def __init__(self, bad_symbol):
        self.bad_symbol = bad_symbol
        self.disconnected = False

    def symbol_info(self, sym):
        if sym == self.bad_symbol:
            raise ValueError(f"Unknown symbol: {sym}")
        return {"digits": 2, "spread": 1.5, "tick_size": 0.01}

    def disconnect(self):
        self.disconnected = True


class TestValidateSymbols:
    def test_all_valid_symbols_returns_true(self):
        conn = _FakeConnAllValid()
        assert _validate_symbols(conn, ["XAUUSD", "GBPUSD"]) is True
        assert conn.disconnected is False

    def test_invalid_symbol_returns_false_and_disconnects(self):
        conn = _FakeConnOneInvalid(bad_symbol="FAKEUSD")
        assert _validate_symbols(conn, ["XAUUSD", "FAKEUSD", "GBPUSD"]) is False
        assert conn.disconnected is True

    def test_stops_at_first_invalid_symbol_without_checking_rest(self):
        checked = []
        class _Conn:
            def symbol_info(self, sym):
                checked.append(sym)
                if sym == "BAD":
                    raise ValueError("nope")
                return {"digits": 2, "spread": 1.0, "tick_size": 0.01}
            def disconnect(self):
                pass
        _validate_symbols(_Conn(), ["BAD", "NEVER_CHECKED"])
        assert checked == ["BAD"]

    def test_empty_symbol_list_returns_true(self):
        assert _validate_symbols(_FakeConnAllValid(), []) is True


# ─────────────────────────────────────────────────────────────────────────────
# _trade_management_tick
# ─────────────────────────────────────────────────────────────────────────────

def _close_event(ticket=1, pnl=15.0):
    trade = SimpleNamespace(
        ticket=ticket, close_price=2360.0, close_time=datetime.now(timezone.utc),
        close_reason="tp1", pnl=pnl, tp1_hit=True, tp2_hit=False,
        symbol="XAUUSD", direction="long",
    )
    return {"trade": trade}


class _FakeExecutor:
    def __init__(self, events=None, raise_on_manage=False, open_trades=None, conn=None):
        self._events = events or []
        self._raise = raise_on_manage
        self._open_trades = open_trades or []
        self.conn = conn

    def manage_open_trades(self):
        if self._raise:
            raise RuntimeError("MT5 connection dropped")
        return self._events

    def get_open_trades(self):
        return self._open_trades


class _RecordingJournal:
    def __init__(self):
        self.closed = []

    def log_trade_close(self, ticket, price, time_, reason, pnl, tp1, tp2):
        self.closed.append(ticket)


class _RaisingJournal:
    def log_trade_close(self, *a, **kw):
        raise RuntimeError("DB locked")


class _RecordingGuard:
    def __init__(self):
        self.recorded = []

    def record_closed_trade(self, pnl, trade_date=None, verbose=False):
        self.recorded.append(pnl)

    def status(self):
        return {"halted": False}


class _RecordingAlerts:
    def __init__(self):
        self.closed = []

    def trade_closed(self, ticket, symbol, direction, reason, price, pnl, tp1, tp2):
        self.closed.append(ticket)

    def guard_halt(self, **kwargs):
        pass


class TestTradeManagementTick:
    def test_none_executor_is_a_safe_noop(self):
        _trade_management_tick(None, None, None, None, dashboard_port=None)  # must not raise

    def test_executor_exception_is_contained(self):
        executor = _FakeExecutor(raise_on_manage=True)
        # must not propagate
        _trade_management_tick(executor, None, None, None, dashboard_port=None)

    def test_close_event_fans_out_to_journal_guard_and_alerts(self):
        executor = _FakeExecutor(events=[_close_event(ticket=42, pnl=15.0)])
        journal = _RecordingJournal()
        guard = _RecordingGuard()
        alerts = _RecordingAlerts()
        _trade_management_tick(executor, journal, guard, alerts, dashboard_port=None)
        assert journal.closed == [42]
        assert guard.recorded == [15.0]
        assert alerts.closed == [42]

    def test_journal_failure_does_not_block_guard_or_alerts(self):
        executor = _FakeExecutor(events=[_close_event(ticket=7, pnl=-5.0)])
        guard = _RecordingGuard()
        alerts = _RecordingAlerts()
        # journal raises -> guard and alerts must still run
        _trade_management_tick(executor, _RaisingJournal(), guard, alerts, dashboard_port=None)
        assert guard.recorded == [-5.0]
        assert alerts.closed == [7]

    def test_events_with_no_trade_key_are_skipped(self):
        executor = _FakeExecutor(events=[{"trade": None}, {}])
        journal = _RecordingJournal()
        _trade_management_tick(executor, journal, None, None, dashboard_port=None)
        assert journal.closed == []


# ─────────────────────────────────────────────────────────────────────────────
# _register_models_for_symbols — regression tests for an undefined
# `dashboard_port` reference that used to raise NameError on every call
# where a dashboard was configured (silently swallowed by the per-symbol
# except, so it looked like "quick backtest failed" instead of a crash).
# ─────────────────────────────────────────────────────────────────────────────

class _FakeConnForRegistration:
    """Rates/symbol_info content doesn't matter -- the whole indicator/
    signal pipeline is monkeypatched to pass a trivial frame straight
    through, so this only needs to satisfy the call signatures."""

    def fetch_rates(self, symbol, tf, n_bars):
        return []

    def symbol_info(self, symbol):
        return {"tick_size": 0.01}


class _FakeRiskEngineForRegistration:
    def __init__(self):
        self.registered = []

    def register_backtest(self, tbl, sym, tf):
        self.registered.append((sym, tf))
        return "Model A"


def _passthrough(df, *args, **kwargs):
    return df


class TestRegisterModelsForSymbolsDashboardPush:
    def _patch_pipeline(self, monkeypatch):
        trivial_df = pd.DataFrame({"close": [1.0, 2.0]})
        monkeypatch.setattr(mls, "_rates_to_df", lambda rates, sym, tick_size: trivial_df)
        monkeypatch.setattr(mls, "_auto_htf", lambda tf: "H4")
        monkeypatch.setattr(mls, "calc_all", _passthrough)
        monkeypatch.setattr(mls, "build_all", _passthrough)
        monkeypatch.setattr(mls, "build_candle_patterns", _passthrough)
        monkeypatch.setattr(mls, "build_ceo_structure", _passthrough)
        monkeypatch.setattr(mls, "build_confluence", _passthrough)
        monkeypatch.setattr(mls, "build_patterns", _passthrough)
        monkeypatch.setattr(mls, "add_session_columns", _passthrough)

        import ceo_engine_mt5.backtest as bt
        tbl = pd.DataFrame({"Win Rate": [55.0]}, index=["Model A"])
        monkeypatch.setattr(bt, "run_backtest", lambda df: object())
        monkeypatch.setattr(bt, "results_table", lambda res: tbl)

    def test_dashboard_port_pushes_backtest_results_without_crashing(self, monkeypatch):
        self._patch_pipeline(monkeypatch)
        import ceo_engine_mt5.dashboard as dash
        pushed = {}
        monkeypatch.setattr(dash, "update_backtest",
                             lambda sym, tf, rows: pushed.setdefault(sym, rows))

        risk_engine = _FakeRiskEngineForRegistration()
        _register_models_for_symbols(
            conn=_FakeConnForRegistration(), symbols=["XAUUSD"], tf="H1",
            signal_params={}, sessions=None, risk_engine=risk_engine,
            perf_monitor=None, dashboard_port=5000,
        )

        # Before the fix, referencing the undefined `dashboard_port` name
        # raised NameError here -- caught by the per-symbol except, so
        # registration looked "failed" and the dashboard never got data.
        assert risk_engine.registered == [("XAUUSD", "H1")]
        assert "XAUUSD" in pushed

    def test_no_dashboard_port_skips_push_without_crashing(self, monkeypatch):
        self._patch_pipeline(monkeypatch)
        import ceo_engine_mt5.dashboard as dash
        called = []
        monkeypatch.setattr(dash, "update_backtest", lambda sym, tf, rows: called.append(sym))

        risk_engine = _FakeRiskEngineForRegistration()
        _register_models_for_symbols(
            conn=_FakeConnForRegistration(), symbols=["XAUUSD"], tf="H1",
            signal_params={}, sessions=None, risk_engine=risk_engine,
            perf_monitor=None, dashboard_port=None,
        )

        assert risk_engine.registered == [("XAUUSD", "H1")]
        assert called == []


# ─────────────────────────────────────────────────────────────────────────────
# _init_live_components — regression test for an undefined `conn` reference
# that used to raise NameError on every startup where a dashboard and
# auto-trade were both enabled (the live MT5 connection doesn't exist yet
# at this point in startup; wiring the dashboard to it now happens in
# mt5_live.py once the connection is actually established).
# ─────────────────────────────────────────────────────────────────────────────

class TestInitLiveComponentsDashboardAutoTrade:
    def test_dashboard_and_auto_trade_together_do_not_crash(self, monkeypatch, tmp_path):
        import ceo_engine_mt5.dashboard as dash
        monkeypatch.setattr(dash, "start_dashboard", lambda port: None)
        set_executor_calls = []
        monkeypatch.setattr(dash, "set_executor",
                             lambda executor, conn: set_executor_calls.append((executor, conn)))

        components = _init_live_components(
            params={}, symbols=["XAUUSD"], tf="H1", signal_params={},
            risk_pct=1.0, min_quality=50.0, min_consistency=0.0, max_sl_pips=2000.0,
            auto_trade=True, account_size=10_000, daily_loss_pct=5.0, max_dd_pct=10.0,
            consistency_pct=0.0, journal_path=None, telegram_token=None,
            telegram_chat=None, dashboard_port=5000,
            block_news=False, news_pre_mins=30, news_post_mins=15, news_medium=False,
            fcsapi_key="", mtf_tfs=None, mtf_mode="bias", mtf_min_tfs=None,
            mtf_min_score=40.0, perf_feedback=False, perf_rolling_window=10,
            perf_loss_streak=3, perf_win_rate_floor=35.0,
        )

        # Before the fix, this raised NameError: name 'conn' is not defined
        # as soon as dashboard_port + executor were both truthy.
        assert components["executor"] is not None
        # The dashboard/executor wiring now happens in mt5_live.py once a
        # real connection exists, not inside this function.
        assert set_executor_calls == []

    def _base_kwargs(self, **overrides):
        kwargs = dict(
            params={}, symbols=["XAUUSD"], tf="H1", signal_params={},
            risk_pct=0.0, min_quality=50.0, min_consistency=0.0, max_sl_pips=2000.0,
            auto_trade=False, account_size=10_000, daily_loss_pct=0.0, max_dd_pct=0.0,
            consistency_pct=0.0, journal_path=None, telegram_token=None,
            telegram_chat=None, dashboard_port=None,
            block_news=False, news_pre_mins=30, news_post_mins=15, news_medium=False,
            fcsapi_key="", mtf_tfs=None, mtf_mode="bias", mtf_min_tfs=None,
            mtf_min_score=40.0, perf_feedback=False, perf_rolling_window=10,
            perf_loss_streak=3, perf_win_rate_floor=35.0,
        )
        kwargs.update(overrides)
        return kwargs

    def test_telegram_alerts_configured_when_token_and_chat_present(self, monkeypatch, tmp_path):
        import ceo_engine_mt5.alerts as alerts_mod
        sent = []
        monkeypatch.setattr(alerts_mod.AlertSystem, "system_alert",
                             lambda self, message, level="info": sent.append(message))
        # Run from a tmp cwd so an unrelated ceo_engine_config.json in the
        # real project root doesn't get picked up by this test.
        monkeypatch.chdir(tmp_path)

        components = _init_live_components(**self._base_kwargs(
            telegram_token="123:ABC", telegram_chat="999"))

        assert components["alerts_obj"] is not None
        assert sent  # system_alert() was called on startup

    def test_journal_created_when_path_given(self, tmp_path):
        journal_path = str(tmp_path / "journal.db")
        components = self_journal = _init_live_components(**self._base_kwargs(
            journal_path=journal_path))
        assert components["journal"] is not None

    def test_dashboard_with_journal_path_calls_set_journal(self, monkeypatch, tmp_path):
        import ceo_engine_mt5.dashboard as dash
        monkeypatch.setattr(dash, "start_dashboard", lambda port: None)
        set_journal_calls = []
        monkeypatch.setattr(dash, "set_journal", lambda path: set_journal_calls.append(path))

        journal_path = str(tmp_path / "journal.db")
        _init_live_components(**self._base_kwargs(
            journal_path=journal_path, dashboard_port=5000))

        assert set_journal_calls == [journal_path]

    def test_news_filter_enabled_when_block_news_true(self):
        components = _init_live_components(**self._base_kwargs(block_news=True))
        assert components["news_filter"] is not None

    def test_news_filter_none_when_block_news_false(self):
        components = _init_live_components(**self._base_kwargs(block_news=False))
        assert components["news_filter"] is None

    def test_mtf_stack_built_when_mtf_tfs_given(self):
        components = _init_live_components(**self._base_kwargs(
            mtf_tfs=["H4", "H1", "M15"]))
        assert components["mtf_stack"] is not None
        assert components["mtf_stack"].tfs == ["H4", "H1", "M15"]

    def test_perf_feedback_builds_monitor_when_journal_and_risk_present(self, tmp_path):
        journal_path = str(tmp_path / "journal.db")
        components = _init_live_components(**self._base_kwargs(
            journal_path=journal_path, risk_pct=1.0, perf_feedback=True))
        assert components["perf_monitor"] is not None

    def test_perf_feedback_skipped_with_warning_when_missing_journal(self, capsys):
        components = _init_live_components(**self._base_kwargs(
            perf_feedback=True, risk_pct=1.0, journal_path=None))
        assert components["perf_monitor"] is None
        assert "skipping" in capsys.readouterr().out


# ─────────────────────────────────────────────────────────────────────────────
# _handle_bar_close — dispatches to MTF or single-TF path, then runs perf
# feedback. Fakes both check_symbol and _handle_mtf_signal at the module
# level so this test focuses purely on the dispatch/print/perf-update logic.
# ─────────────────────────────────────────────────────────────────────────────

import threading
from ceo_engine_mt5.mt5_live_session import _handle_bar_close


class _FakePerfMonitor:
    def __init__(self):
        self.updates = []

    def update(self, symbol, tf, verbose=True):
        self.updates.append((symbol, tf))


class TestHandleBarClose:
    def _kwargs(self, **overrides):
        kwargs = dict(
            symbol="EURUSD", conn_local=object(), cur_bar=object(), tf="H1",
            mtf_stack=None, bt_params={}, seen_ids=set(),
            risk_engine=None, guard=None, journal=None, alerts_obj=None,
            executor=None, risk_pct=1.0, params={}, signal_params={},
            log_path=None, sound=False, news_filter=None, n_bars=500,
            perf_monitor=None, lock=threading.Lock(),
        )
        kwargs.update(overrides)
        return kwargs

    def test_single_tf_mode_calls_check_symbol(self, monkeypatch, capsys):
        monkeypatch.setattr(mls, "check_symbol", lambda **kwargs: [{"id": 1}])
        n = _handle_bar_close(**self._kwargs())
        assert n == 1
        assert "1 signal(s)!" in capsys.readouterr().out

    def test_single_tf_mode_prints_no_signal(self, monkeypatch, capsys):
        monkeypatch.setattr(mls, "check_symbol", lambda **kwargs: [])
        n = _handle_bar_close(**self._kwargs())
        assert n == 0
        assert "no signal" in capsys.readouterr().out

    def test_mtf_mode_calls_handle_mtf_signal(self, monkeypatch, capsys):
        monkeypatch.setattr(mls, "_handle_mtf_signal", lambda **kwargs: [{"id": 1}, {"id": 2}])
        n = _handle_bar_close(**self._kwargs(mtf_stack=object()))
        assert n == 2

    def test_mtf_mode_prints_no_confirmation(self, monkeypatch, capsys):
        monkeypatch.setattr(mls, "_handle_mtf_signal", lambda **kwargs: [])
        n = _handle_bar_close(**self._kwargs(mtf_stack=object()))
        assert n == 0
        assert "no MTF confirmation" in capsys.readouterr().out

    def test_perf_monitor_updated_after_check(self, monkeypatch):
        monkeypatch.setattr(mls, "check_symbol", lambda **kwargs: [])
        perf = _FakePerfMonitor()
        _handle_bar_close(**self._kwargs(perf_monitor=perf))
        assert perf.updates == [("EURUSD", "H1")]

    def test_no_perf_monitor_is_fine(self, monkeypatch):
        monkeypatch.setattr(mls, "check_symbol", lambda **kwargs: [])
        n = _handle_bar_close(**self._kwargs(perf_monitor=None))
        assert n == 0


# ─────────────────────────────────────────────────────────────────────────────
# _print_shutdown_summary
# ─────────────────────────────────────────────────────────────────────────────

from ceo_engine_mt5.mt5_live_session import _print_shutdown_summary


class _FakeJournalForShutdown:
    def __init__(self):
        self.printed = False

    def print_summary(self):
        self.printed = True

    def performance_stats(self):
        return {"trades": 5}


class _FakeAlertsForShutdown:
    def __init__(self, raise_error=False):
        self.daily_summaries = []
        self._raise = raise_error

    def daily_summary(self, stats, guard_status=None):
        if self._raise:
            raise RuntimeError("boom")
        self.daily_summaries.append((stats, guard_status))


class _FakeGuardForShutdown:
    def status(self, acct):
        return {"blocked": False}


class _FakeConnForShutdown:
    def __init__(self, raise_error=False):
        self._raise = raise_error

    def account_info(self):
        if self._raise:
            raise RuntimeError("acct boom")
        return {"balance": 10_000.0}


class TestPrintShutdownSummary:
    def test_prints_total_and_per_symbol_counts(self, capsys):
        _print_shutdown_summary(
            total_signals=7, symbols=["EURUSD", "XAUUSD"],
            sym_signals={"EURUSD": 4, "XAUUSD": 3}, log_path=None,
            journal=None, alerts_obj=None, guard=None, conn=None,
        )
        out = capsys.readouterr().out
        assert "Total signals this session: 7" in out
        assert "Signals per symbol" in out
        assert "EURUSD" in out and "XAUUSD" in out

    def test_single_symbol_skips_per_symbol_breakdown(self, capsys):
        _print_shutdown_summary(
            total_signals=3, symbols=["EURUSD"], sym_signals={"EURUSD": 3},
            log_path=None, journal=None, alerts_obj=None, guard=None, conn=None,
        )
        out = capsys.readouterr().out
        assert "Signals per symbol" not in out

    def test_log_path_printed_when_exists(self, tmp_path, capsys):
        log_file = tmp_path / "log.txt"
        log_file.write_text("x")
        _print_shutdown_summary(
            total_signals=0, symbols=["EURUSD"], sym_signals={},
            log_path=str(log_file), journal=None, alerts_obj=None,
            guard=None, conn=None,
        )
        assert str(log_file) in capsys.readouterr().out

    def test_journal_summary_printed_when_present(self):
        journal = _FakeJournalForShutdown()
        _print_shutdown_summary(
            total_signals=0, symbols=["EURUSD"], sym_signals={},
            log_path=None, journal=journal, alerts_obj=None,
            guard=None, conn=None,
        )
        assert journal.printed is True

    def test_alerts_daily_summary_sent_with_journal_and_guard(self):
        journal = _FakeJournalForShutdown()
        alerts = _FakeAlertsForShutdown()
        guard = _FakeGuardForShutdown()
        conn = _FakeConnForShutdown()
        _print_shutdown_summary(
            total_signals=0, symbols=["EURUSD"], sym_signals={},
            log_path=None, journal=journal, alerts_obj=alerts,
            guard=guard, conn=conn,
        )
        assert len(alerts.daily_summaries) == 1
        assert alerts.daily_summaries[0][1] == {"blocked": False}

    def test_alerts_without_journal_sends_nothing(self):
        alerts = _FakeAlertsForShutdown()
        conn = _FakeConnForShutdown()
        _print_shutdown_summary(
            total_signals=0, symbols=["EURUSD"], sym_signals={},
            log_path=None, journal=None, alerts_obj=alerts,
            guard=None, conn=conn,
        )
        assert alerts.daily_summaries == []

    def test_account_info_failure_does_not_propagate(self):
        journal = _FakeJournalForShutdown()
        alerts = _FakeAlertsForShutdown()
        conn = _FakeConnForShutdown(raise_error=True)
        # Should not raise even though conn.account_info() blows up.
        _print_shutdown_summary(
            total_signals=0, symbols=["EURUSD"], sym_signals={},
            log_path=None, journal=journal, alerts_obj=alerts,
            guard=None, conn=conn,
        )
        assert alerts.daily_summaries == []

    def test_alerts_failure_does_not_propagate(self):
        journal = _FakeJournalForShutdown()
        alerts = _FakeAlertsForShutdown(raise_error=True)
        conn = _FakeConnForShutdown()
        _print_shutdown_summary(
            total_signals=0, symbols=["EURUSD"], sym_signals={},
            log_path=None, journal=journal, alerts_obj=alerts,
            guard=None, conn=conn,
        )  # no exception expected
