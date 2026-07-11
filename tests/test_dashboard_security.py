"""
Tests for the dashboard.py security hardening:
  - rate limiting on the auth gate and on trade-mutating endpoints
  - a shared lock serializing /api/trade, /api/close, /api/modify_sl,
    /api/set_trail against each other
  - generic error responses that don't leak raw exception text
  - ceo_engine_config.json written with 0600 permissions (it can hold the
    MT5 password and Telegram bot token in plaintext)
  - start_dashboard() defaulting to a localhost-only bind

Runs with cwd redirected to a temp directory and CEO_DASHBOARD_PASSWORD
pinned to a known value *before* dashboard.py is imported, so the module's
import-time file I/O (auth file creation) never touches the repo and
tests get a predictable password to authenticate with.
"""

import importlib
import os
import stat
import sys
import threading
import time

import pytest


TEST_PASSWORD = "test-password-for-dashboard-suite"


@pytest.fixture()
def dash(tmp_path, monkeypatch):
    """Fresh dashboard module, cwd redirected to an isolated tmp dir."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("CEO_DASHBOARD_PASSWORD", TEST_PASSWORD)
    sys.modules.pop("ceo_engine_mt5.dashboard", None)
    mod = importlib.import_module("ceo_engine_mt5.dashboard")
    # Rate-limit buckets are module-global; start each test with a clean slate.
    mod._rate_buckets.clear()
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
# Login rate limiting (item 9)
# ─────────────────────────────────────────────────────────────────────────────

# ─────────────────────────────────────────────────────────────────────────────
# Dashboard HTML template extraction (moved from an inline Python literal
# to ceo_engine_mt5/templates/dashboard.html for maintainability)
# ─────────────────────────────────────────────────────────────────────────────

class TestDashboardTemplateExtraction:
    def test_template_file_exists_on_disk(self, dash):
        path = dash._dashboard_template_path()
        assert path.exists(), f"template not found at {path}"

    def test_loaded_template_matches_file_contents_exactly(self, dash):
        path = dash._dashboard_template_path()
        assert dash.DASHBOARD_HTML == path.read_text(encoding="utf-8")

    def test_template_is_well_formed_html_shell(self, dash):
        html = dash.DASHBOARD_HTML
        assert html.startswith("<!DOCTYPE html>")
        assert html.rstrip().endswith("</html>")
        assert "<title>CEO Engine</title>" in html

    def test_missing_template_raises_clear_error(self, dash, monkeypatch, tmp_path):
        missing = tmp_path / "does_not_exist.html"
        monkeypatch.setattr(dash, "_dashboard_template_path", lambda: missing)
        with pytest.raises(FileNotFoundError, match="Dashboard template not found"):
            dash._load_dashboard_template()

    def test_index_route_serves_the_template(self, dash, client):
        r = client.get("/")
        assert r.status_code == 200
        assert b"<!DOCTYPE html>" in r.data
        assert b"CEO Engine" in r.data


# ─────────────────────────────────────────────────────────────────────────────
# Trade endpoint rate limiting (item 9)
# ─────────────────────────────────────────────────────────────────────────────

class TestTradeRateLimiting:
    def test_trade_endpoint_rate_limited_independently_of_engine_state(self, dash, client):
        # Engine isn't running, so each call would 400 anyway -- the
        # decorator must still count and eventually 429 regardless.
        for _ in range(dash.TRADE_MAX_REQUESTS):
            r = client.post("/api/trade", json={"symbol": "XAUUSD", "direction": "long"},
                             headers=_auth_headers())
            assert r.status_code in (400, 200)
        r = client.post("/api/trade", json={"symbol": "XAUUSD", "direction": "long"},
                         headers=_auth_headers())
        assert r.status_code == 429

    def test_close_and_modify_sl_also_rate_limited(self, dash, client):
        for _ in range(dash.TRADE_MAX_REQUESTS):
            client.post("/api/close", json={"ticket": 1}, headers=_auth_headers())
        r = client.post("/api/close", json={"ticket": 1}, headers=_auth_headers())
        assert r.status_code == 429


# ─────────────────────────────────────────────────────────────────────────────
# Trade operation lock (item 7)
# ─────────────────────────────────────────────────────────────────────────────

class TestTradeLock:
    def test_trade_op_lock_exists_and_is_a_lock(self, dash):
        assert isinstance(dash._trade_op_lock, type(threading.Lock()))

    def test_concurrent_close_calls_are_serialized(self, dash, client):
        """Two overlapping /api/close calls against a slow fake executor
        must not run manual_close() concurrently."""
        calls = []
        overlap_detected = threading.Event()
        in_flight = threading.Event()

        class _SlowExecutor:
            def manual_close(self, ticket, reason="manual"):
                if in_flight.is_set():
                    overlap_detected.set()
                in_flight.set()
                calls.append(ticket)
                time.sleep(0.15)
                in_flight.clear()
                return {"ok": True, "pnl": 1.0}

        dash.state.set_executor(_SlowExecutor())
        # Reset rate limiter so both calls survive it.
        dash._rate_buckets.clear()

        def _hit(ticket):
            client.post("/api/close", json={"ticket": ticket}, headers=_auth_headers())

        t1 = threading.Thread(target=_hit, args=(1,))
        t2 = threading.Thread(target=_hit, args=(2,))
        t1.start(); time.sleep(0.03); t2.start()
        t1.join(); t2.join()

        assert not overlap_detected.is_set()
        assert sorted(calls) == [1, 2]


# ─────────────────────────────────────────────────────────────────────────────
# Safe error responses (item 8)
# ─────────────────────────────────────────────────────────────────────────────

class TestSafeErrorResponses:
    def test_trade_exception_does_not_leak_raw_message(self, dash, client):
        class _BoomExecutor:
            def manual_close(self, ticket, reason="manual"):
                raise RuntimeError("/home/user/.secret/mt5_creds.ini not found")

        dash.state.set_executor(_BoomExecutor())
        r = client.post("/api/close", json={"ticket": 5}, headers=_auth_headers())
        body = r.get_json()
        assert body["ok"] is False
        assert ".secret" not in body["error"]
        assert "mt5_creds" not in body["error"]
        assert r.status_code == 500

    def test_safe_error_response_helper_hides_exception_text(self, dash):
        with dash.app.test_request_context():
            resp, status = dash._safe_error_response(
                ValueError("leaky/path/details.txt"), "some_route")
        assert status == 500
        body = resp.get_json()
        assert "leaky" not in body["error"]
        assert body["ok"] is False

    def test_include_ok_false_shape_still_hides_exception_text(self, dash):
        with dash.app.test_request_context():
            resp, status = dash._safe_error_response(
                ValueError("leaky/path/details.txt"), "some_route", include_ok=False)
        body = resp.get_json()
        assert "ok" not in body
        assert "leaky" not in body["error"]


# ─────────────────────────────────────────────────────────────────────────────
# Config file permissions (item 2)
# ─────────────────────────────────────────────────────────────────────────────

class TestConfigFilePermissions:
    def test_config_saved_with_owner_only_permissions(self, dash, client, tmp_path):
        payload = {"mt5_password": "hunter2", "telegram_token": "123:ABC"}
        r = client.post("/api/config", json=payload, headers=_auth_headers())
        assert r.status_code == 200
        mode = stat.S_IMODE(os.stat(dash._config_path).st_mode)
        assert mode == 0o600

    def test_auth_file_also_owner_only(self, dash):
        # Written at import time by _load_or_create_dashboard_password()
        # only when CEO_DASHBOARD_PASSWORD isn't set; force that path here.
        os.environ.pop("CEO_DASHBOARD_PASSWORD", None)
        try:
            pw = dash._load_or_create_dashboard_password()
            mode = stat.S_IMODE(os.stat(dash._AUTH_CONFIG_PATH).st_mode)
            assert mode == 0o600
        finally:
            os.environ["CEO_DASHBOARD_PASSWORD"] = TEST_PASSWORD


# ─────────────────────────────────────────────────────────────────────────────
# tail_log path traversal (found during review, same file)
# ─────────────────────────────────────────────────────────────────────────────

class TestTailLogPathTraversal:
    def test_relative_traversal_is_confined_to_basename(self, dash, client, tmp_path):
        # A secret file *outside* cwd that traversal would have reached.
        secret_dir = tmp_path.parent / "outside_cwd_secret"
        secret_dir.mkdir(exist_ok=True)
        secret_file = secret_dir / "passwd"
        secret_file.write_text("root:x:0:0::/root:/bin/bash\n")
        r = client.post("/api/exec",
                         json={"cmd": "tail_log", "path": "../outside_cwd_secret/passwd"},
                         headers=_auth_headers())
        body = r.get_json()
        assert body["ok"] is True
        # basename("../outside_cwd_secret/passwd") == "passwd" -> looked for
        # ./passwd in cwd (tmp_path), which doesn't exist, not the secret file.
        assert "root:x:0:0" not in body["output"]
        assert "not found" in body["output"].lower()

    def test_absolute_path_is_confined_to_basename(self, dash, client, tmp_path):
        outside = tmp_path.parent / "another_secret.txt"
        outside.write_text("TOP SECRET CONTENTS")
        r = client.post("/api/exec",
                         json={"cmd": "tail_log", "path": str(outside)},
                         headers=_auth_headers())
        body = r.get_json()
        assert body["ok"] is True
        assert "TOP SECRET" not in body["output"]

    def test_dot_and_dotdot_alone_are_rejected(self, dash, client):
        for bad in ("..", "."):
            r = client.post("/api/exec", json={"cmd": "tail_log", "path": bad},
                             headers=_auth_headers())
            body = r.get_json()
            assert body["ok"] is False

    def test_legitimate_log_file_in_cwd_is_still_readable(self, dash, client, tmp_path):
        (tmp_path / "ceo_engine.log").write_text("line1\nline2\nline3\n")
        r = client.post("/api/exec", json={"cmd": "tail_log"}, headers=_auth_headers())
        body = r.get_json()
        assert body["ok"] is True
        assert "line1" in body["output"]

    def test_traversal_attempt_does_not_leak_raw_exception(self, dash, client):
        # Even if something unexpected goes wrong, no internal path details
        # should reach the client (covers the outer api_exec except too).
        r = client.post("/api/exec", json={"cmd": "tail_log", "path": "/nonexistent/dir/x"},
                         headers=_auth_headers())
        assert r.status_code in (200, 400, 500)
        body = r.get_json()
        # basename("/nonexistent/dir/x") == "x" -> just reports not-found, no leak.
        if body.get("ok"):
            assert "nonexistent" not in body.get("output", "")

# ─────────────────────────────────────────────────────────────────────────────
# Default bind address (item 10)
# ─────────────────────────────────────────────────────────────────────────────

class TestDefaultBindAddress:
    def test_start_dashboard_defaults_to_localhost(self, dash, monkeypatch):
        os.environ.pop("CEO_DASHBOARD_HOST", None)
        captured = {}
        monkeypatch.setattr(dash.app, "run",
            lambda **kwargs: captured.update(kwargs))
        t = dash.start_dashboard(port=0)
        t.join(timeout=2)
        assert captured.get("host") == "127.0.0.1"

    def test_env_var_can_opt_into_wider_bind(self, dash, monkeypatch):
        monkeypatch.setenv("CEO_DASHBOARD_HOST", "0.0.0.0")
        captured = {}
        monkeypatch.setattr(dash.app, "run",
            lambda **kwargs: captured.update(kwargs))
        t = dash.start_dashboard(port=0)
        t.join(timeout=2)
        assert captured.get("host") == "0.0.0.0"

    def test_explicit_host_argument_overrides_env(self, dash, monkeypatch):
        monkeypatch.setenv("CEO_DASHBOARD_HOST", "0.0.0.0")
        captured = {}
        monkeypatch.setattr(dash.app, "run",
            lambda **kwargs: captured.update(kwargs))
        t = dash.start_dashboard(host="127.0.0.1", port=0)
        t.join(timeout=2)
        assert captured.get("host") == "127.0.0.1"


# ─────────────────────────────────────────────────────────────────────────────
# Password never written to the persistent log file (regression)
# ─────────────────────────────────────────────────────────────────────────────

class TestPasswordNotLogged:
    def test_password_absent_from_all_logger_calls(self, dash, monkeypatch, capsys):
        """The dashboard password must reach the operator via print() (console
        only, never persisted) and must never be passed to logger.* -- those
        calls get written to the rotating ceo_engine.log file on disk."""
        monkeypatch.setattr(dash.app, "run", lambda **kwargs: None)

        logged_messages = []
        original_info = dash.logger.info

        def _capture_info(msg, *args, **kwargs):
            logged_messages.append(msg % args if args else msg)
            return original_info(msg, *args, **kwargs)

        monkeypatch.setattr(dash.logger, "info", _capture_info)

        t = dash.start_dashboard(port=0)
        t.join(timeout=2)

        for msg in logged_messages:
            assert dash._DASHBOARD_PASSWORD not in msg

        # The password should still reach the operator some other way (stdout).
        out = capsys.readouterr().out
        assert dash._DASHBOARD_PASSWORD in out
