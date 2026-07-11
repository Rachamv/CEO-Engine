"""
Tests for mt5_live.py -- previously 0% covered. run_live() is the CLI
orchestrator: connect -> init components -> validate symbols -> spawn one
monitor thread per symbol -> run a 2s trade-management loop on the main
thread until Ctrl+C or a connection error. This suite fakes every
collaborator (MetaTrader5 itself, _try_connect, _init_live_components,
_validate_symbols, _handle_bar_close, _trade_management_tick,
_print_shutdown_summary) so the tests exercise run_live()'s own control
flow -- not the modules it delegates to, which already have their own
test files.
"""

import sys
import types

import pytest

import ceo_engine_mt5.mt5_live as mt5_live_mod
from ceo_engine_mt5.mt5_live import _print_startup_banner, check_symbol_wrapper, run_live


# ─────────────────────────────────────────────────────────────────────────────
# _print_startup_banner -- pure print function
# ─────────────────────────────────────────────────────────────────────────────

class TestPrintStartupBanner:
    def test_prints_key_fields(self, capsys):
        _print_startup_banner(
            symbols=["EURUSD", "XAUUSD"], tf="h1", sessions=["london", "ny"],
            auto_trade=True, risk_pct=1.5, block_news=True,
            perf_monitor=object(), log_path="log.txt", journal_path="j.db",
            alerts_obj=object(), dashboard_port=5000,
        )
        out = capsys.readouterr().out
        assert "EURUSD" in out and "XAUUSD" in out
        assert "H1" in out
        assert "Auto-trade: ON" in out
        assert "Risk      : 1.5%" in out
        assert "News gate : ON" in out
        assert "Perf loop : ON" in out
        assert "http://localhost:5000" in out

    def test_off_states_print_off(self, capsys):
        _print_startup_banner(
            symbols=["EURUSD"], tf="m15", sessions=["all"],
            auto_trade=False, risk_pct=0.0, block_news=False,
            perf_monitor=None, log_path=None, journal_path=None,
            alerts_obj=None, dashboard_port=None,
        )
        out = capsys.readouterr().out
        assert "Auto-trade: OFF" in out
        assert "News gate : OFF" in out
        assert "Perf loop : OFF" in out
        assert "Dashboard : off" in out
        assert "Journal   : disabled" in out
        assert "Telegram  : off" in out

    def test_zero_risk_pct_line_omitted(self, capsys):
        _print_startup_banner(
            symbols=["EURUSD"], tf="m15", sessions=["all"],
            auto_trade=False, risk_pct=0.0, block_news=False,
            perf_monitor=None, log_path=None, journal_path=None,
            alerts_obj=None, dashboard_port=None,
        )
        out = capsys.readouterr().out
        assert "Risk      :" not in out


# ─────────────────────────────────────────────────────────────────────────────
# check_symbol_wrapper -- thin alias
# ─────────────────────────────────────────────────────────────────────────────

class TestCheckSymbolWrapper:
    def test_delegates_to_check_symbol(self, monkeypatch):
        calls = []
        monkeypatch.setattr(mt5_live_mod, "check_symbol",
                             lambda *a, **k: calls.append((a, k)) or ["result"])
        out = check_symbol_wrapper(1, 2, foo="bar")
        assert out == ["result"]
        assert calls == [((1, 2), {"foo": "bar"})]


# ─────────────────────────────────────────────────────────────────────────────
# run_live() -- control flow with every collaborator faked out
# ─────────────────────────────────────────────────────────────────────────────

class _FakeConnForRunLive:
    def __init__(self):
        self.disconnected = False

    def disconnect(self):
        self.disconnected = True


def _base_components(**overrides):
    components = {
        "risk_engine": None, "guard": None, "executor": None,
        "journal": None, "alerts_obj": None, "news_filter": None,
        "mtf_stack": None, "perf_monitor": None, "sessions": ["all"],
    }
    components.update(overrides)
    return components


def _install_fake_mt5(monkeypatch, copy_rates_from_pos=None):
    fake_mod = types.SimpleNamespace()
    for name in ["M1","M2","M3","M4","M5","M6","M10","M12","M15","M20","M30",
                 "H1","H2","H3","H4","H6","H8","H12","D1","W1","MN1"]:
        setattr(fake_mod, f"TIMEFRAME_{name}", name)
    fake_mod.copy_rates_from_pos = copy_rates_from_pos or (
        lambda symbol, tf, start_pos, count: [
            {"time": 1704067200, "open": 1.0, "high": 1.0, "low": 1.0,
             "close": 1.0, "tick_volume": 1}])
    monkeypatch.setitem(sys.modules, "MetaTrader5", fake_mod)
    import ceo_engine_mt5.mt5_connect as mtc
    monkeypatch.setattr(mtc, "MT5_AVAILABLE", True)
    return fake_mod


class TestRunLiveShutdown:
    def test_keyboard_interrupt_triggers_clean_shutdown_and_disconnect(self, monkeypatch):
        _install_fake_mt5(monkeypatch)
        fake_conn = _FakeConnForRunLive()

        monkeypatch.setattr(mt5_live_mod, "_try_connect", lambda *a, **k: fake_conn)
        monkeypatch.setattr(mt5_live_mod, "_init_live_components",
                             lambda **k: _base_components())
        monkeypatch.setattr(mt5_live_mod, "_validate_symbols", lambda conn, symbols: True)
        monkeypatch.setattr(mt5_live_mod, "_handle_bar_close", lambda **k: 0)
        monkeypatch.setattr(mt5_live_mod, "_seconds_to_bar_close", lambda tf: 0)
        monkeypatch.setattr(mt5_live_mod, "_load_seen_ids", lambda log_path: set())
        monkeypatch.setattr(mt5_live_mod.time, "sleep", lambda s: None)

        shutdown_calls = []
        monkeypatch.setattr(mt5_live_mod, "_print_shutdown_summary",
                             lambda **k: shutdown_calls.append(k))

        def _raise_keyboard_interrupt(*a, **k):
            raise KeyboardInterrupt()
        monkeypatch.setattr(mt5_live_mod, "_trade_management_tick", _raise_keyboard_interrupt)

        # run_live() catches KeyboardInterrupt internally and returns --
        # it must NOT propagate out of this call.
        run_live(symbols=["EURUSD"], tf="H1", params={}, signal_params={}, bt_params={})

        assert len(shutdown_calls) == 1
        assert shutdown_calls[0]["total_signals"] == 0
        assert fake_conn.disconnected is True

    def test_invalid_symbols_returns_without_starting_threads(self, monkeypatch):
        _install_fake_mt5(monkeypatch)
        fake_conn = _FakeConnForRunLive()

        monkeypatch.setattr(mt5_live_mod, "_try_connect", lambda *a, **k: fake_conn)
        monkeypatch.setattr(mt5_live_mod, "_init_live_components",
                             lambda **k: _base_components())
        monkeypatch.setattr(mt5_live_mod, "_validate_symbols", lambda conn, symbols: False)
        monkeypatch.setattr(mt5_live_mod, "_load_seen_ids", lambda log_path: set())

        tick_calls = []
        monkeypatch.setattr(mt5_live_mod, "_trade_management_tick",
                             lambda *a, **k: tick_calls.append(1))

        # Should return (not hang, not raise) as soon as validation fails --
        # the trade-management loop should never even start.
        run_live(symbols=["BADSYM"], tf="H1", params={}, signal_params={}, bt_params={})

        assert tick_calls == []

    def test_dashboard_and_auto_trade_wire_executor_to_connection(self, monkeypatch):
        """Regression check for the mt5_live_session.py fix: with a
        dashboard_port and an executor both present, run_live() must wire
        dash.set_executor(executor, conn) itself once conn exists, and must
        not crash doing so."""
        _install_fake_mt5(monkeypatch)
        fake_conn = _FakeConnForRunLive()

        class _FakeExecutor:
            def __init__(self):
                self.conn = None
                self.simulation = True

        executor = _FakeExecutor()

        import ceo_engine_mt5.dashboard as dash
        set_executor_calls = []
        monkeypatch.setattr(dash, "set_executor",
                             lambda ex, conn: set_executor_calls.append((ex, conn)))

        monkeypatch.setattr(mt5_live_mod, "_try_connect", lambda *a, **k: fake_conn)
        monkeypatch.setattr(mt5_live_mod, "_init_live_components",
                             lambda **k: _base_components(executor=executor))
        monkeypatch.setattr(mt5_live_mod, "_validate_symbols", lambda conn, symbols: True)
        monkeypatch.setattr(mt5_live_mod, "_handle_bar_close", lambda **k: 0)
        monkeypatch.setattr(mt5_live_mod, "_seconds_to_bar_close", lambda tf: 0)
        monkeypatch.setattr(mt5_live_mod, "_load_seen_ids", lambda log_path: set())
        monkeypatch.setattr(mt5_live_mod.time, "sleep", lambda s: None)
        monkeypatch.setattr(mt5_live_mod, "_print_shutdown_summary", lambda **k: None)
        monkeypatch.setattr(mt5_live_mod, "_register_models_for_symbols", lambda **k: None)

        def _raise_keyboard_interrupt(*a, **k):
            raise KeyboardInterrupt()
        monkeypatch.setattr(mt5_live_mod, "_trade_management_tick", _raise_keyboard_interrupt)

        run_live(symbols=["EURUSD"], tf="H1", params={}, signal_params={}, bt_params={},
                 auto_trade=True, dashboard_port=5000)

        assert executor.conn is fake_conn
        assert executor.simulation is False
        assert set_executor_calls == [(executor, fake_conn)]

    def test_connection_error_triggers_reconnect_wait_not_crash(self, monkeypatch):
        """A mid-session exception (e.g. MT5 connection drop) should be
        caught, disconnect attempted, and the outer loop retried -- not
        propagate out of run_live(). We let it retry exactly once, then
        stop via a KeyboardInterrupt on the second pass."""
        _install_fake_mt5(monkeypatch)
        fake_conn = _FakeConnForRunLive()

        monkeypatch.setattr(mt5_live_mod, "_try_connect", lambda *a, **k: fake_conn)
        monkeypatch.setattr(mt5_live_mod, "_init_live_components",
                             lambda **k: _base_components())
        monkeypatch.setattr(mt5_live_mod, "_validate_symbols", lambda conn, symbols: True)
        monkeypatch.setattr(mt5_live_mod, "_handle_bar_close", lambda **k: 0)
        monkeypatch.setattr(mt5_live_mod, "_seconds_to_bar_close", lambda tf: 0)
        monkeypatch.setattr(mt5_live_mod, "_load_seen_ids", lambda log_path: set())
        monkeypatch.setattr(mt5_live_mod.time, "sleep", lambda s: None)
        monkeypatch.setattr(mt5_live_mod, "_print_shutdown_summary", lambda **k: None)

        calls = {"n": 0}

        def _tick(*a, **k):
            calls["n"] += 1
            if calls["n"] == 1:
                raise RuntimeError("connection dropped")
            raise KeyboardInterrupt()

        monkeypatch.setattr(mt5_live_mod, "_trade_management_tick", _tick)

        run_live(symbols=["EURUSD"], tf="H1", params={}, signal_params={}, bt_params={},
                 reconnect_wait=0)

        assert calls["n"] == 2  # first raised RuntimeError, second stopped cleanly
