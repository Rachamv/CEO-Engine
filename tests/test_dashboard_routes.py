"""
Tests for dashboard.py routes not covered by test_dashboard_security.py
(which focuses on auth/rate-limiting/locking). These exercise the actual
business logic: state-reading endpoints, engine config persistence,
engine subprocess start/stop, Telegram test helpers, MT5 auto-detect,
and the trade-mutating routes' success paths.
"""

import importlib
import os
import sys
import types

import pytest

TEST_PASSWORD = "test-password-for-dashboard-routes-suite"


@pytest.fixture()
def dash(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("CEO_DASHBOARD_PASSWORD", TEST_PASSWORD)
    sys.modules.pop("ceo_engine_mt5.dashboard", None)
    mod = importlib.import_module("ceo_engine_mt5.dashboard")
    mod._rate_buckets.clear()
    mod._config_path = str(tmp_path / "ceo_engine_config.json")
    yield mod
    sys.modules.pop("ceo_engine_mt5.dashboard", None)


@pytest.fixture()
def client(dash):
    dash.app.testing = True
    return dash.app.test_client()


def _auth_headers():
    import base64
    token = base64.b64encode(f"admin:{TEST_PASSWORD}".encode()).decode()
    return {"Authorization": f"Basic {token}"}


# ─────────────────────────────────────────────────────────────────────────────
# Simple state-reading routes
# ─────────────────────────────────────────────────────────────────────────────

class TestStateReadingRoutes:
    def test_api_state_returns_snapshot(self, dash, client):
        r = client.get("/api/state", headers=_auth_headers())
        assert r.status_code == 200
        assert isinstance(r.get_json(), dict)

    def test_api_pnl_returns_trades_snapshot(self, dash, client):
        r = client.get("/api/pnl", headers=_auth_headers())
        assert r.status_code == 200

    def test_api_candles_returns_list(self, dash, client):
        r = client.get("/api/candles/EURUSD/H1", headers=_auth_headers())
        assert r.status_code == 200
        assert r.get_json() == []

    def test_api_candles_returns_pushed_bars(self, dash, client):
        dash.update_candles("EURUSD", "H1", [{"time": 1, "open": 1, "high": 1,
                                               "low": 1, "close": 1, "volume": 1}])
        r = client.get("/api/candles/EURUSD/H1", headers=_auth_headers())
        assert len(r.get_json()) == 1

    def test_api_chart_returns_empty_string_when_no_chart(self, dash, client):
        r = client.get("/api/chart/EURUSD/H1", headers=_auth_headers())
        assert r.status_code == 200
        assert r.data == b""

    def test_api_chart_returns_pushed_html(self, dash, client):
        dash.update_chart("EURUSD", "H1", "<div>chart</div>")
        r = client.get("/api/chart/EURUSD/H1", headers=_auth_headers())
        assert b"chart" in r.data

    def test_api_log_records_message(self, dash, client):
        r = client.post("/api/log", json={"message": "hello", "level": "warn"},
                         headers=_auth_headers())
        assert r.status_code == 200
        assert r.get_json()["ok"] is True

    def test_api_market_ticks_returns_dict(self, dash, client):
        r = client.get("/api/market_ticks", headers=_auth_headers())
        assert isinstance(r.get_json(), dict)

    def test_api_backtest_returns_dict(self, dash, client):
        r = client.get("/api/backtest", headers=_auth_headers())
        assert isinstance(r.get_json(), dict)

    def test_api_backtest_returns_pushed_results(self, dash, client):
        dash.update_backtest("EURUSD", "H1", [{"model": "LQ", "win_rate": 55.0}])
        r = client.get("/api/backtest", headers=_auth_headers())
        assert "EURUSD_H1" in r.get_json()

    def test_api_structure_returns_empty_shape_by_default(self, dash, client):
        r = client.get("/api/structure/EURUSD/H1", headers=_auth_headers())
        body = r.get_json()
        assert body == {"zoneLines": [], "priceLines": [], "markers": []}

    def test_api_structure_returns_pushed_overlay(self, dash, client):
        payload = {"zoneLines": [{"color": "#000", "data": []}],
                   "priceLines": [{"price": 1.5, "color": "#000", "title": "Fib 50%"}],
                   "markers": [{"time": 1, "text": "BOS"}]}
        dash.update_structure("EURUSD", "H1", payload)
        r = client.get("/api/structure/EURUSD/H1", headers=_auth_headers())
        assert r.get_json() == payload

    def test_structure_is_keyed_per_symbol_and_tf(self, dash, client):
        dash.update_structure("EURUSD", "H1", {"zoneLines": [], "priceLines": [], "markers": ["a"]})
        dash.update_structure("EURUSD", "M15", {"zoneLines": [], "priceLines": [], "markers": ["b"]})
        r1 = client.get("/api/structure/EURUSD/H1", headers=_auth_headers()).get_json()
        r2 = client.get("/api/structure/EURUSD/M15", headers=_auth_headers()).get_json()
        assert r1["markers"] == ["a"]
        assert r2["markers"] == ["b"]


# ─────────────────────────────────────────────────────────────────────────────
# api_history
# ─────────────────────────────────────────────────────────────────────────────

class TestApiHistory:
    def test_no_journal_returns_empty_shape(self, dash, client):
        r = client.get("/api/history", headers=_auth_headers())
        body = r.get_json()
        assert body["recent_trades"] == []
        assert body["stats"] == {}

    def test_journal_present_returns_full_shape(self, dash, client, tmp_path):
        from ceo_engine_mt5.journal import Journal
        journal_path = str(tmp_path / "journal.db")
        Journal(journal_path)  # creates the sqlite schema
        dash.set_journal(journal_path)

        r = client.get("/api/history", headers=_auth_headers())
        body = r.get_json()
        assert "recent_trades" in body
        assert "by_model" in body
        assert "by_session" in body

    def test_journal_error_returns_safe_error_response(self, dash, client, monkeypatch, tmp_path):
        bad_path = str(tmp_path / "journal.db")
        open(bad_path, "w").close()
        dash.set_journal(bad_path)

        import ceo_engine_mt5.journal as journal_mod
        def _raise(self, *a, **k):
            raise RuntimeError("corrupt journal")
        monkeypatch.setattr(journal_mod.Journal, "__init__", _raise)

        r = client.get("/api/history", headers=_auth_headers())
        body = r.get_json()
        assert "error" in body


# ─────────────────────────────────────────────────────────────────────────────
# api_config
# ─────────────────────────────────────────────────────────────────────────────

class TestApiConfig:
    def test_get_returns_defaults_when_no_file(self, dash, client):
        r = client.get("/api/config", headers=_auth_headers())
        body = r.get_json()
        assert body["symbols"] == ["XAUUSD"]
        assert body["dashboard_port"] == 5000

    def test_post_writes_config_and_get_reads_it_back(self, dash, client, tmp_path):
        cfg = {"symbols": ["EURUSD"], "tf": "M15", "risk_pct": 2.0}
        r = client.post("/api/config", json=cfg, headers=_auth_headers())
        assert r.get_json()["ok"] is True

        r2 = client.get("/api/config", headers=_auth_headers())
        assert r2.get_json()["symbols"] == ["EURUSD"]

    def test_post_sets_restrictive_permissions(self, dash, client):
        import os, stat
        client.post("/api/config", json={"symbols": ["EURUSD"]}, headers=_auth_headers())
        assert os.path.exists(dash._config_path)
        if os.name != "nt":
            mode = stat.S_IMODE(os.stat(dash._config_path).st_mode)
            assert mode == 0o600


# ─────────────────────────────────────────────────────────────────────────────
# Engine subprocess control: /api/start, /api/stop, /api/engine_status
# ─────────────────────────────────────────────────────────────────────────────

class _FakePopen:
    def __init__(self, *args, **kwargs):
        self.pid = 4242
        self._terminated = False
        self._poll_value = None  # None = still running

    def poll(self):
        return self._poll_value

    def terminate(self):
        self._terminated = True
        self._poll_value = 0


class TestEngineStartStop:
    def test_start_fails_without_saved_config(self, dash, client):
        r = client.post("/api/start", headers=_auth_headers())
        body = r.get_json()
        assert body["ok"] is False
        assert "No config saved" in body["error"]

    def test_start_launches_subprocess_with_config(self, dash, client, monkeypatch):
        client.post("/api/config", json={
            "symbols": ["EURUSD", "XAUUSD"], "tf": "H1", "auto_trade": True,
            "risk_pct": 1.5, "mt5_login": 123, "mt5_password": "pw",
            "mt5_server": "Broker-Demo", "telegram_token": "t", "telegram_chat": "c",
        }, headers=_auth_headers())

        captured = {}
        def _fake_popen(cmd, **kwargs):
            captured["cmd"] = cmd
            return _FakePopen()
        monkeypatch.setattr(dash._subprocess, "Popen", _fake_popen)

        r = client.post("/api/start", headers=_auth_headers())
        body = r.get_json()
        assert body["ok"] is True
        assert body["pid"] == 4242
        assert "--symbol" in captured["cmd"]
        assert "EURUSD" in captured["cmd"]
        assert "--auto-trade" in captured["cmd"]
        assert os.path.isabs(captured["cmd"][1])
        assert captured["cmd"][1].endswith("run.py")

    def test_start_rejects_when_already_running(self, dash, client, monkeypatch):
        client.post("/api/config", json={"symbols": ["EURUSD"]}, headers=_auth_headers())
        monkeypatch.setattr(dash._subprocess, "Popen", lambda cmd, **kwargs: _FakePopen())
        client.post("/api/start", headers=_auth_headers())

        r = client.post("/api/start", headers=_auth_headers())
        body = r.get_json()
        assert body["ok"] is False
        assert "already running" in body["error"]

    def test_start_returns_safe_error_on_popen_failure(self, dash, client, monkeypatch):
        client.post("/api/config", json={"symbols": ["EURUSD"]}, headers=_auth_headers())
        def _raise(*a, **k):
            raise OSError("no such file: run.py")
        monkeypatch.setattr(dash._subprocess, "Popen", _raise)

        r = client.post("/api/start", headers=_auth_headers())
        body = r.get_json()
        assert body["ok"] is False
        assert "no such file" not in body.get("error", "")  # safe_error_response scrubs raw text

    def test_stop_when_not_running(self, dash, client):
        r = client.post("/api/stop", headers=_auth_headers())
        body = r.get_json()
        assert body["ok"] is False
        assert "not running" in body["error"]

    def test_stop_terminates_running_engine(self, dash, client, monkeypatch):
        client.post("/api/config", json={"symbols": ["EURUSD"]}, headers=_auth_headers())
        fake = _FakePopen()
        monkeypatch.setattr(dash._subprocess, "Popen", lambda cmd, **kwargs: fake)
        client.post("/api/start", headers=_auth_headers())

        r = client.post("/api/stop", headers=_auth_headers())
        assert r.get_json()["ok"] is True
        assert fake._terminated is True

    def test_engine_status_reflects_running_state(self, dash, client, monkeypatch):
        r = client.get("/api/engine_status", headers=_auth_headers())
        assert r.get_json()["running"] is False

        client.post("/api/config", json={"symbols": ["EURUSD"]}, headers=_auth_headers())
        fake = _FakePopen()
        monkeypatch.setattr(dash._subprocess, "Popen", lambda cmd, **kwargs: fake)
        client.post("/api/start", headers=_auth_headers())

        r2 = client.get("/api/engine_status", headers=_auth_headers())
        body = r2.get_json()
        assert body["running"] is True
        assert body["pid"] == 4242


# ─────────────────────────────────────────────────────────────────────────────
# /api/test_telegram and /api/find_chat_id
# ─────────────────────────────────────────────────────────────────────────────

class _FakeResponse:
    def __init__(self, status_code=200, payload=None):
        self.status_code = status_code
        self._payload = payload or {}

    def json(self):
        return self._payload


class TestTelegramHelpers:
    def test_test_telegram_requires_token_and_chat_id(self, dash, client):
        r = client.post("/api/test_telegram", json={}, headers=_auth_headers())
        assert r.get_json()["ok"] is False

    def test_test_telegram_success(self, dash, client, monkeypatch):
        import requests as real_requests
        monkeypatch.setattr(real_requests, "post", lambda *a, **k: _FakeResponse(200))
        r = client.post("/api/test_telegram", json={"token": "t", "chat_id": "c"},
                         headers=_auth_headers())
        assert r.get_json()["ok"] is True

    def test_test_telegram_reports_telegram_error(self, dash, client, monkeypatch):
        import requests as real_requests
        monkeypatch.setattr(real_requests, "post",
                             lambda *a, **k: _FakeResponse(400, {"description": "bad token"}))
        r = client.post("/api/test_telegram", json={"token": "t", "chat_id": "c"},
                         headers=_auth_headers())
        body = r.get_json()
        assert body["ok"] is False
        assert body["error"] == "bad token"

    def test_test_telegram_network_error_is_caught(self, dash, client, monkeypatch):
        import requests as real_requests
        def _raise(*a, **k):
            raise real_requests.exceptions.ConnectionError("no network")
        monkeypatch.setattr(real_requests, "post", _raise)
        r = client.post("/api/test_telegram", json={"token": "t", "chat_id": "c"},
                         headers=_auth_headers())
        assert r.get_json()["ok"] is False

    def test_find_chat_id_requires_token(self, dash, client):
        r = client.post("/api/find_chat_id", json={}, headers=_auth_headers())
        assert r.get_json()["ok"] is False

    def test_find_chat_id_returns_most_recent_chat(self, dash, client, monkeypatch):
        import requests as real_requests
        payload = {"result": [
            {"message": {"chat": {"id": 111, "first_name": "Alice"}}},
            {"message": {"chat": {"id": 222, "first_name": "Bob"}}},
        ]}
        monkeypatch.setattr(real_requests, "get", lambda *a, **k: _FakeResponse(200, payload))
        r = client.post("/api/find_chat_id", json={"token": "t"}, headers=_auth_headers())
        body = r.get_json()
        assert body["ok"] is True
        assert body["chat_id"] == 222
        assert body["chat_name"] == "Bob"

    def test_find_chat_id_no_messages_yet(self, dash, client, monkeypatch):
        import requests as real_requests
        monkeypatch.setattr(real_requests, "get", lambda *a, **k: _FakeResponse(200, {"result": []}))
        r = client.post("/api/find_chat_id", json={"token": "t"}, headers=_auth_headers())
        body = r.get_json()
        assert body["ok"] is False
        assert "No messages found" in body["error"]

    def test_find_chat_id_rejected_token(self, dash, client, monkeypatch):
        import requests as real_requests
        monkeypatch.setattr(real_requests, "get",
                             lambda *a, **k: _FakeResponse(401, {"description": "Unauthorized"}))
        r = client.post("/api/find_chat_id", json={"token": "bad"}, headers=_auth_headers())
        body = r.get_json()
        assert body["ok"] is False
        assert "Unauthorized" in body["error"]


# ─────────────────────────────────────────────────────────────────────────────
# /api/mt5_detect
# ─────────────────────────────────────────────────────────────────────────────

class TestMt5Detect:
    def test_non_windows_returns_clear_error(self, dash, client, monkeypatch):
        monkeypatch.setattr(sys, "platform", "linux")
        r = client.get("/api/mt5_detect", headers=_auth_headers())
        body = r.get_json()
        assert body["ok"] is False
        assert "Windows" in body["error"]

    def test_windows_but_package_missing(self, dash, client, monkeypatch):
        monkeypatch.setattr(sys, "platform", "win32")
        monkeypatch.setitem(sys.modules, "MetaTrader5", None)
        r = client.get("/api/mt5_detect", headers=_auth_headers())
        body = r.get_json()
        assert body["ok"] is False
        assert "not installed" in body["error"]

    def test_windows_success_returns_account_and_symbols(self, dash, client, monkeypatch):
        monkeypatch.setattr(sys, "platform", "win32")
        fake_mt5 = types.SimpleNamespace(
            initialize=lambda: True,
            last_error=lambda: (0, "ok"),
            shutdown=lambda: None,
            account_info=lambda: types.SimpleNamespace(
                login=1, server="Broker-Demo", name="Test", currency="USD",
                balance=1000.0, equity=1000.0, leverage=100),
            terminal_info=lambda: types.SimpleNamespace(
                name="MT5", build=1, path="/opt/mt5"),
            symbols_get=lambda: [types.SimpleNamespace(name="XAUUSD"),
                                  types.SimpleNamespace(name="EURUSD")],
        )
        monkeypatch.setitem(sys.modules, "MetaTrader5", fake_mt5)
        r = client.get("/api/mt5_detect", headers=_auth_headers())
        body = r.get_json()
        assert body["ok"] is True
        assert body["account"]["server"] == "Broker-Demo"
        assert body["account"]["account_type"] == "demo"
        assert "XAUUSD" in body["broker_symbols"]

    def test_windows_not_logged_in(self, dash, client, monkeypatch):
        monkeypatch.setattr(sys, "platform", "win32")
        fake_mt5 = types.SimpleNamespace(
            initialize=lambda: False,
            last_error=lambda: (1, "not logged in"),
            shutdown=lambda: None,
        )
        monkeypatch.setitem(sys.modules, "MetaTrader5", fake_mt5)
        r = client.get("/api/mt5_detect", headers=_auth_headers())
        body = r.get_json()
        assert body["ok"] is False
        assert "not running or not logged in" in body["error"]


# ─────────────────────────────────────────────────────────────────────────────
# Trade route success paths (rate limit / lock already covered elsewhere)
# ─────────────────────────────────────────────────────────────────────────────

class _FakeExecutorForTrade:
    def __init__(self, trade_result=None, close_result=None,
                 modify_result=None, trail_result=True):
        self._trade_result = trade_result or {"ticket": 999, "lots": 0.1}
        self._close_result = close_result or {"ok": True, "pnl": 12.5}
        self._modify_result = modify_result or {"ok": True, "old_sl": 1.0, "new_sl": 1.1}
        self._trail_result = trail_result

    def place_trade(self, **kwargs):
        return self._trade_result

    def manual_close(self, ticket, reason="manual"):
        return self._close_result

    def manual_modify_sl(self, ticket, new_sl):
        return self._modify_result

    def set_trailing_sl(self, **kwargs):
        return self._trail_result


class _FakeConnForTrade:
    def account_info(self):
        return {"balance": 10_000.0}

    def symbol_info(self, symbol):
        return {"digits": 2, "tick_size": 0.01}


class TestTradeRoutesSuccessPaths:
    def test_trade_places_order_successfully(self, dash, client):
        dash.set_executor(_FakeExecutorForTrade(), _FakeConnForTrade())
        r = client.post("/api/trade", json={
            "symbol": "XAUUSD", "direction": "long", "entry": 2000.0, "sl": 1990.0,
        }, headers=_auth_headers())
        body = r.get_json()
        assert body["ok"] is True
        assert body["ticket"] == 999

    def test_trade_missing_symbol_or_direction(self, dash, client):
        dash.set_executor(_FakeExecutorForTrade(), _FakeConnForTrade())
        r = client.post("/api/trade", json={"symbol": "XAUUSD"}, headers=_auth_headers())
        assert r.get_json()["ok"] is False

    def test_trade_no_account_info_available(self, dash, client):
        class _NoInfoConn:
            def account_info(self): return None
            def symbol_info(self, s): return None
        dash.set_executor(_FakeExecutorForTrade(), _NoInfoConn())
        r = client.post("/api/trade", json={
            "symbol": "XAUUSD", "direction": "long", "entry": 2000.0, "sl": 1990.0,
        }, headers=_auth_headers())
        assert r.get_json()["ok"] is False

    def test_close_success(self, dash, client):
        dash.set_executor(_FakeExecutorForTrade(), _FakeConnForTrade())
        r = client.post("/api/close", json={"ticket": 5}, headers=_auth_headers())
        body = r.get_json()
        assert body["ok"] is True
        assert body["pnl"] == 12.5

    def test_close_missing_ticket(self, dash, client):
        dash.set_executor(_FakeExecutorForTrade(), _FakeConnForTrade())
        r = client.post("/api/close", json={}, headers=_auth_headers())
        assert r.get_json()["ok"] is False

    def test_modify_sl_success(self, dash, client):
        dash.set_executor(_FakeExecutorForTrade(), _FakeConnForTrade())
        r = client.post("/api/modify_sl", json={"ticket": 5, "new_sl": 1.23},
                         headers=_auth_headers())
        body = r.get_json()
        assert body["ok"] is True
        assert body["new_sl"] == 1.1

    def test_modify_sl_missing_fields(self, dash, client):
        dash.set_executor(_FakeExecutorForTrade(), _FakeConnForTrade())
        r = client.post("/api/modify_sl", json={"ticket": 5}, headers=_auth_headers())
        assert r.get_json()["ok"] is False

    def test_set_trail_success(self, dash, client):
        dash.set_executor(_FakeExecutorForTrade(), _FakeConnForTrade())
        r = client.post("/api/set_trail", json={"ticket": 5, "atr": 0.002},
                         headers=_auth_headers())
        body = r.get_json()
        assert body["ok"] is True

    def test_set_trail_ticket_not_found(self, dash, client):
        dash.set_executor(_FakeExecutorForTrade(trail_result=False), _FakeConnForTrade())
        r = client.post("/api/set_trail", json={"ticket": 999, "atr": 0.002},
                         headers=_auth_headers())
        body = r.get_json()
        assert body["ok"] is False
        assert "not found" in body["error"]

    def test_set_trail_missing_fields(self, dash, client):
        dash.set_executor(_FakeExecutorForTrade(), _FakeConnForTrade())
        r = client.post("/api/set_trail", json={"ticket": 5}, headers=_auth_headers())
        assert r.get_json()["ok"] is False
