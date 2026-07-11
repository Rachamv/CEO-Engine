"""
Tests for mt5_connect.py -- the MT5 terminal connection wrapper.

The real MetaTrader5 package only exists on Windows and this suite runs
on Linux, so every test here injects a small fake `MetaTrader5` module
into sys.modules (matching the shape mt5_connect.py actually uses:
TIMEFRAME_* constants, initialize/shutdown, terminal_info/account_info,
symbol_info/symbol_select, copy_rates_*, symbol_info_tick, symbols_get)
and monkeypatches MT5_AVAILABLE so _require_mt5() doesn't reject the
call before it ever reaches the fake.
"""

import sys
import types
from datetime import datetime

import pytest

import ceo_engine_mt5.mt5_connect as mtc
from ceo_engine_mt5.mt5_connect import MT5Connection, get_mt5_timeframe, _require_mt5


def _make_fake_mt5(**overrides):
    """Builds a fake MetaTrader5 module with sane defaults, overridable
    per test via keyword functions/values."""
    mod = types.SimpleNamespace()

    # Timeframe constants -- values don't matter, just need to exist.
    for name in ["M1","M2","M3","M4","M5","M6","M10","M12","M15","M20","M30",
                 "H1","H2","H3","H4","H6","H8","H12","D1","W1","MN1"]:
        setattr(mod, f"TIMEFRAME_{name}", name)

    mod.initialize = overrides.get("initialize", lambda **kwargs: True)
    mod.last_error = overrides.get("last_error", lambda: (0, "no error"))
    mod.terminal_info = overrides.get(
        "terminal_info", lambda: types.SimpleNamespace(name="MetaTrader 5", build=1234))
    mod.account_info = overrides.get(
        "account_info",
        lambda: types.SimpleNamespace(
            login=12345, server="Broker-Demo", name="Test Account",
            currency="USD", balance=10000.0, equity=10000.0, margin=0.0,
            margin_free=10000.0, leverage=100, profit=0.0))
    mod.shutdown = overrides.get("shutdown", lambda: None)
    mod.symbol_select = overrides.get("symbol_select", lambda symbol, enable: True)
    mod.symbol_info = overrides.get(
        "symbol_info",
        lambda symbol: types.SimpleNamespace(
            name=symbol, description="desc", digits=5, trade_tick_size=0.00001,
            trade_tick_value=1.0, trade_contract_size=100000, spread=2,
            currency_base="EUR", currency_profit="USD", point=0.00001,
            volume_min=0.01, volume_max=100.0, volume_step=0.01))
    mod.copy_rates_from_pos = overrides.get(
        "copy_rates_from_pos",
        lambda symbol, tf, start_pos, count: [
            {"time": 1704067200, "open": 1.0, "high": 1.5, "low": 0.5,
             "close": 1.2, "tick_volume": 100}])
    mod.copy_rates_from = overrides.get("copy_rates_from", mod.copy_rates_from_pos)
    mod.copy_rates_range = overrides.get(
        "copy_rates_range",
        lambda symbol, tf, start, end: mod.copy_rates_from_pos(symbol, tf, 0, 1))
    mod.symbol_info_tick = overrides.get(
        "symbol_info_tick",
        lambda symbol: types.SimpleNamespace(bid=1.1, ask=1.1002, last=1.1001,
                                              volume=5, time=1704067200))
    mod.symbols_get = overrides.get(
        "symbols_get",
        lambda filter_str=None: [types.SimpleNamespace(name="EURUSD"),
                                  types.SimpleNamespace(name="XAUUSD")])
    return mod


@pytest.fixture
def fake_mt5(monkeypatch):
    """Injects a fresh fake MetaTrader5 module and enables MT5_AVAILABLE."""
    mod = _make_fake_mt5()
    monkeypatch.setitem(sys.modules, "MetaTrader5", mod)
    monkeypatch.setattr(mtc, "MT5_AVAILABLE", True)
    return mod


# ─────────────────────────────────────────────────────────────────────────────
# _require_mt5 / platform gating
# ─────────────────────────────────────────────────────────────────────────────

class TestRequireMt5:
    def test_raises_with_windows_only_message_off_windows(self, monkeypatch):
        monkeypatch.setattr(mtc, "MT5_AVAILABLE", False)
        monkeypatch.setattr(sys, "platform", "linux")
        with pytest.raises(RuntimeError, match="only available on Windows"):
            _require_mt5()

    def test_raises_with_install_message_on_windows_without_package(self, monkeypatch):
        monkeypatch.setattr(mtc, "MT5_AVAILABLE", False)
        monkeypatch.setattr(sys, "platform", "win32")
        with pytest.raises(RuntimeError, match="pip install MetaTrader5"):
            _require_mt5()

    def test_does_not_raise_when_available(self, monkeypatch):
        monkeypatch.setattr(mtc, "MT5_AVAILABLE", True)
        _require_mt5()  # should not raise


# ─────────────────────────────────────────────────────────────────────────────
# get_mt5_timeframe
# ─────────────────────────────────────────────────────────────────────────────

class TestGetMt5Timeframe:
    def test_known_alias_maps_correctly(self, fake_mt5):
        assert get_mt5_timeframe("h1") == fake_mt5.TIMEFRAME_H1
        assert get_mt5_timeframe("1h") == fake_mt5.TIMEFRAME_H1

    def test_case_insensitive(self, fake_mt5):
        assert get_mt5_timeframe("H4") == get_mt5_timeframe("h4")

    def test_unknown_timeframe_raises(self, fake_mt5):
        with pytest.raises(ValueError, match="Unknown MT5 timeframe"):
            get_mt5_timeframe("7m")


# ─────────────────────────────────────────────────────────────────────────────
# MT5Connection.connect / disconnect / context manager
# ─────────────────────────────────────────────────────────────────────────────

class TestConnect:
    def test_successful_connect_returns_true(self, fake_mt5):
        conn = MT5Connection()
        assert conn.connect() is True
        assert conn._connected is True

    def test_failed_initialize_raises_connection_error(self, monkeypatch):
        mod = _make_fake_mt5(initialize=lambda **kwargs: False,
                              last_error=lambda: (1, "terminal not found"))
        monkeypatch.setitem(sys.modules, "MetaTrader5", mod)
        monkeypatch.setattr(mtc, "MT5_AVAILABLE", True)
        conn = MT5Connection()
        with pytest.raises(ConnectionError, match="MT5 initialisation failed"):
            conn.connect()

    def test_login_credentials_passed_through_when_all_present(self, fake_mt5, monkeypatch):
        captured = {}
        monkeypatch.setattr(fake_mt5, "initialize",
                             lambda **kwargs: captured.update(kwargs) or True)
        conn = MT5Connection(login=123, password="pw", server="Broker-Demo")
        conn.connect()
        assert captured["login"] == 123
        assert captured["password"] == "pw"
        assert captured["server"] == "Broker-Demo"

    def test_partial_credentials_are_not_passed(self, fake_mt5, monkeypatch):
        captured = {}
        monkeypatch.setattr(fake_mt5, "initialize",
                             lambda **kwargs: captured.update(kwargs) or True)
        conn = MT5Connection(login=123)  # missing password/server
        conn.connect()
        assert "login" not in captured

    def test_path_is_passed_when_given(self, fake_mt5, monkeypatch):
        captured = {}
        monkeypatch.setattr(fake_mt5, "initialize",
                             lambda **kwargs: captured.update(kwargs) or True)
        conn = MT5Connection(path="/opt/mt5/terminal64.exe")
        conn.connect()
        assert captured["path"] == "/opt/mt5/terminal64.exe"

    def test_disconnect_calls_shutdown_when_connected(self, fake_mt5, monkeypatch):
        shutdown_calls = []
        monkeypatch.setattr(fake_mt5, "shutdown", lambda: shutdown_calls.append(True))
        conn = MT5Connection()
        conn.connect()
        conn.disconnect()
        assert shutdown_calls == [True]
        assert conn._connected is False

    def test_disconnect_is_noop_when_never_connected(self, fake_mt5, monkeypatch):
        shutdown_calls = []
        monkeypatch.setattr(fake_mt5, "shutdown", lambda: shutdown_calls.append(True))
        conn = MT5Connection()
        conn.disconnect()
        assert shutdown_calls == []

    def test_context_manager_connects_and_disconnects(self, fake_mt5, monkeypatch):
        shutdown_calls = []
        monkeypatch.setattr(fake_mt5, "shutdown", lambda: shutdown_calls.append(True))
        with MT5Connection() as conn:
            assert conn._connected is True
        assert shutdown_calls == [True]


# ─────────────────────────────────────────────────────────────────────────────
# symbol_info
# ─────────────────────────────────────────────────────────────────────────────

class TestSymbolInfo:
    def test_returns_expected_fields(self, fake_mt5):
        conn = MT5Connection()
        info = conn.symbol_info("EURUSD")
        assert info["symbol"] == "EURUSD"
        assert info["tick_size"] == 0.00001
        assert info["volume_min"] == 0.01

    def test_falls_back_to_symbol_select_when_not_found(self, fake_mt5, monkeypatch):
        calls = {"n": 0}

        def flaky_symbol_info(symbol):
            calls["n"] += 1
            if calls["n"] == 1:
                return None  # not visible in MarketWatch yet
            return types.SimpleNamespace(
                name=symbol, description="d", digits=2, trade_tick_size=0.01,
                trade_tick_value=1.0, trade_contract_size=100, spread=1,
                currency_base="XAU", currency_profit="USD", point=0.01,
                volume_min=0.01, volume_max=50.0, volume_step=0.01)

        monkeypatch.setattr(fake_mt5, "symbol_info", flaky_symbol_info)
        conn = MT5Connection()
        info = conn.symbol_info("XAUUSD")
        assert info["symbol"] == "XAUUSD"

    def test_raises_when_symbol_select_fails(self, fake_mt5, monkeypatch):
        monkeypatch.setattr(fake_mt5, "symbol_info", lambda symbol: None)
        monkeypatch.setattr(fake_mt5, "symbol_select", lambda symbol, enable: False)
        conn = MT5Connection()
        with pytest.raises(ValueError, match="not found on this broker"):
            conn.symbol_info("FAKESYM")

    def test_raises_when_still_unavailable_after_selection(self, fake_mt5, monkeypatch):
        monkeypatch.setattr(fake_mt5, "symbol_info", lambda symbol: None)
        monkeypatch.setattr(fake_mt5, "symbol_select", lambda symbol, enable: True)
        conn = MT5Connection()
        with pytest.raises(ValueError, match="unavailable after selection"):
            conn.symbol_info("FAKESYM")


# ─────────────────────────────────────────────────────────────────────────────
# account_info
# ─────────────────────────────────────────────────────────────────────────────

class TestAccountInfo:
    def test_returns_expected_fields(self, fake_mt5):
        conn = MT5Connection()
        info = conn.account_info()
        assert info["login"] == 12345
        assert info["balance"] == 10000.0
        assert info["free_margin"] == 10000.0

    def test_raises_when_account_info_is_none(self, fake_mt5, monkeypatch):
        monkeypatch.setattr(fake_mt5, "account_info", lambda: None)
        conn = MT5Connection()
        with pytest.raises(RuntimeError, match="Could not retrieve account info"):
            conn.account_info()


# ─────────────────────────────────────────────────────────────────────────────
# fetch_rates
# ─────────────────────────────────────────────────────────────────────────────

class TestFetchRates:
    def test_no_start_or_end_uses_copy_rates_from_pos(self, fake_mt5, monkeypatch):
        calls = []
        monkeypatch.setattr(fake_mt5, "copy_rates_from_pos",
                             lambda symbol, tf, start_pos, count:
                             calls.append("from_pos") or [{"time": 1, "open": 1, "high": 1,
                                                            "low": 1, "close": 1, "tick_volume": 1}])
        conn = MT5Connection()
        conn.fetch_rates("EURUSD", "H1", n_bars=10)
        assert calls == ["from_pos"]

    def test_start_only_uses_copy_rates_from(self, fake_mt5, monkeypatch):
        calls = []
        monkeypatch.setattr(fake_mt5, "copy_rates_from",
                             lambda symbol, tf, start, count:
                             calls.append("from") or [{"time": 1, "open": 1, "high": 1,
                                                        "low": 1, "close": 1, "tick_volume": 1}])
        conn = MT5Connection()
        conn.fetch_rates("EURUSD", "H1", start=datetime(2024, 1, 1))
        assert calls == ["from"]

    def test_start_and_end_uses_copy_rates_range(self, fake_mt5, monkeypatch):
        calls = []
        monkeypatch.setattr(fake_mt5, "copy_rates_range",
                             lambda symbol, tf, start, end:
                             calls.append("range") or [{"time": 1, "open": 1, "high": 1,
                                                         "low": 1, "close": 1, "tick_volume": 1}])
        conn = MT5Connection()
        conn.fetch_rates("EURUSD", "H1", start=datetime(2024, 1, 1), end=datetime(2024, 1, 2))
        assert calls == ["range"]

    def test_unselectable_symbol_raises(self, fake_mt5, monkeypatch):
        monkeypatch.setattr(fake_mt5, "symbol_select", lambda symbol, enable: False)
        conn = MT5Connection()
        with pytest.raises(ValueError, match="Cannot select symbol"):
            conn.fetch_rates("FAKESYM", "H1")

    def test_empty_result_raises(self, fake_mt5, monkeypatch):
        monkeypatch.setattr(fake_mt5, "copy_rates_from_pos",
                             lambda symbol, tf, start_pos, count: [])
        conn = MT5Connection()
        with pytest.raises(ValueError, match="no data"):
            conn.fetch_rates("EURUSD", "H1")

    def test_none_result_raises(self, fake_mt5, monkeypatch):
        monkeypatch.setattr(fake_mt5, "copy_rates_from_pos",
                             lambda symbol, tf, start_pos, count: None)
        conn = MT5Connection()
        with pytest.raises(ValueError, match="no data"):
            conn.fetch_rates("EURUSD", "H1")


# ─────────────────────────────────────────────────────────────────────────────
# symbol_info_tick / available_symbols / last_closed_bar
# ─────────────────────────────────────────────────────────────────────────────

class TestTickAndSymbolsAndLastBar:
    def test_symbol_info_tick_returns_dict(self, fake_mt5):
        conn = MT5Connection()
        tick = conn.symbol_info_tick("EURUSD")
        assert tick["bid"] == 1.1
        assert tick["ask"] == 1.1002

    def test_symbol_info_tick_returns_empty_dict_on_none(self, fake_mt5, monkeypatch):
        monkeypatch.setattr(fake_mt5, "symbol_info_tick", lambda symbol: None)
        conn = MT5Connection()
        assert conn.symbol_info_tick("FAKESYM") == {}

    def test_available_symbols_without_filter(self, fake_mt5):
        conn = MT5Connection()
        assert conn.available_symbols() == ["EURUSD", "XAUUSD"]

    def test_available_symbols_with_filter(self, fake_mt5, monkeypatch):
        monkeypatch.setattr(fake_mt5, "symbols_get",
                             lambda filter_str=None: [types.SimpleNamespace(name="XAUUSD")]
                             if filter_str else [])
        conn = MT5Connection()
        assert conn.available_symbols("XAU*") == ["XAUUSD"]

    def test_available_symbols_returns_empty_list_when_none(self, fake_mt5, monkeypatch):
        monkeypatch.setattr(fake_mt5, "symbols_get", lambda filter_str=None: None)
        conn = MT5Connection()
        assert conn.available_symbols() == []

    def test_last_closed_bar_returns_expected_shape(self, fake_mt5, monkeypatch):
        monkeypatch.setattr(fake_mt5, "copy_rates_from_pos",
                             lambda symbol, tf, start_pos, count: [
                                 {"time": 1704067200, "open": 1.0, "high": 1.5,
                                  "low": 0.5, "close": 1.2, "tick_volume": 100},
                                 {"time": 1704070800, "open": 1.2, "high": 1.6,
                                  "low": 1.0, "close": 1.4, "tick_volume": 110},
                             ])
        conn = MT5Connection()
        bar = conn.last_closed_bar("EURUSD", "H1")
        assert bar["open"] == 1.0
        assert bar["volume"] == 100.0
        assert isinstance(bar["time"], datetime)
