"""
Tests for alerts.py's AlertSystem -- the Telegram alert dispatcher.
Previously 0% covered, despite handling the same bot token that gets
chmod 600'd on disk (see dashboard.py's api_config).

No real network calls anywhere in this file: dry_run mode (the default
when no token/chat_id is configured) is used for most tests, and
requests.post is monkeypatched with a small fake response object for the
transport-layer tests that need to exercise retry/backoff/fallback logic.
"""

import time

import pytest

from ceo_engine_mt5.alerts import AlertSystem


class _FakeResponse:
    def __init__(self, status_code=200, json_data=None, text=""):
        self.status_code = status_code
        self._json_data = json_data or {}
        self.text = text

    def json(self):
        return self._json_data


def _configured_system(**overrides):
    """An AlertSystem with a token/chat_id set (so dry_run isn't
    auto-forced), but dry_run explicitly controlled per-test."""
    kwargs = dict(token="test-token", chat_id="12345", verbose=False, dry_run=True)
    kwargs.update(overrides)
    return AlertSystem(**kwargs)


# ─────────────────────────────────────────────────────────────────────────────
# Construction / configuration
# ─────────────────────────────────────────────────────────────────────────────

class TestConstruction:
    def test_missing_credentials_forces_dry_run(self, monkeypatch):
        monkeypatch.delenv("CEO_TELEGRAM_TOKEN", raising=False)
        monkeypatch.delenv("CEO_TELEGRAM_CHAT_ID", raising=False)
        a = AlertSystem(verbose=False)
        assert a.dry_run is True

    def test_credentials_present_respects_explicit_dry_run_false(self):
        a = AlertSystem(token="t", chat_id="c", verbose=False, dry_run=False)
        assert a.dry_run is False

    def test_env_var_credentials_used_when_not_passed_explicitly(self, monkeypatch):
        monkeypatch.setenv("CEO_TELEGRAM_TOKEN", "env-token")
        monkeypatch.setenv("CEO_TELEGRAM_CHAT_ID", "env-chat")
        a = AlertSystem(verbose=False, dry_run=False)
        assert a.token == "env-token"
        assert a.chat_id == "env-chat"
        assert a.dry_run is False

    def test_explicit_args_override_env_vars(self, monkeypatch):
        monkeypatch.setenv("CEO_TELEGRAM_TOKEN", "env-token")
        a = AlertSystem(token="explicit-token", chat_id="c", verbose=False)
        assert a.token == "explicit-token"

    def test_default_event_toggles_are_on_except_trail_moved(self):
        a = _configured_system()
        assert a.on_signal and a.on_trade_open and a.on_tp_hit and a.on_sl_hit
        assert a.on_daily_summary and a.on_system and a.on_guard_halt
        assert a.on_trail_moved is False   # documented as off-by-default (noisy)


class TestFromConfig:
    def test_builds_from_full_config_dict(self):
        cfg = {
            "telegram_token": "cfg-token", "telegram_chat": "cfg-chat",
            "verbose_alerts": False,
            "alerts": {"on_signal": False, "on_trail_moved": True},
        }
        a = AlertSystem.from_config(cfg)
        assert a.token == "cfg-token"
        assert a.chat_id == "cfg-chat"
        assert a.on_signal is False
        assert a.on_trail_moved is True

    def test_missing_keys_fall_back_to_defaults(self):
        a = AlertSystem.from_config({})
        assert a.token == ""
        assert a.on_signal is True   # default preserved despite empty config
        assert a.dry_run is True     # no credentials -> forced dry_run

    def test_missing_alerts_subdict_uses_all_defaults(self):
        a = AlertSystem.from_config({"telegram_token": "t", "telegram_chat": "c"})
        assert a.on_tp_hit is True
        assert a.on_trail_moved is False


# ─────────────────────────────────────────────────────────────────────────────
# Per-event toggles
# ─────────────────────────────────────────────────────────────────────────────

class TestEventToggles:
    def _sent_texts(self, a, monkeypatch):
        sent = []
        monkeypatch.setattr(a, "_send_message", lambda text, **kw: sent.append(text) or True)
        monkeypatch.setattr(a, "_send_photo", lambda path, caption="", **kw: sent.append(caption) or True)
        return sent

    def test_signal_suppressed_when_toggle_off(self, monkeypatch):
        a = _configured_system(on_signal=False)
        sent = self._sent_texts(a, monkeypatch)
        a.signal("XAUUSD", "M15", "long", 2350, 2345, 2352, 2359, 2366, 72, "LQ")
        assert sent == []

    def test_signal_sent_when_toggle_on(self, monkeypatch):
        a = _configured_system(on_signal=True)
        sent = self._sent_texts(a, monkeypatch)
        a.signal("XAUUSD", "M15", "long", 2350, 2345, 2352, 2359, 2366, 72, "LQ")
        assert len(sent) == 1

    def test_trade_opened_suppressed_when_toggle_off(self, monkeypatch):
        a = _configured_system(on_trade_open=False)
        sent = self._sent_texts(a, monkeypatch)
        a.trade_opened(1, "XAUUSD", "long", 0.1, 2350, 2345, 2366)
        assert sent == []

    def test_tp_hit_suppressed_when_toggle_off(self, monkeypatch):
        a = _configured_system(on_tp_hit=False)
        sent = self._sent_texts(a, monkeypatch)
        a.tp_hit(1, "XAUUSD", "long", 1, 2352, 15.0)
        assert sent == []

    def test_sl_hit_suppressed_when_toggle_off(self, monkeypatch):
        a = _configured_system(on_sl_hit=False)
        sent = self._sent_texts(a, monkeypatch)
        a.sl_hit(1, "XAUUSD", "long", 2345, -20.0)
        assert sent == []

    def test_daily_summary_suppressed_when_toggle_off(self, monkeypatch):
        a = _configured_system(on_daily_summary=False)
        sent = self._sent_texts(a, monkeypatch)
        a.daily_summary({"trades": 5, "wins": 3, "losses": 2, "total_pnl": 40.0})
        assert sent == []

    def test_trail_moved_suppressed_by_default(self, monkeypatch):
        a = _configured_system()   # on_trail_moved defaults False
        sent = self._sent_texts(a, monkeypatch)
        a.trail_moved(1, "XAUUSD", 2345, 2350, 2360)
        assert sent == []

    def test_trail_moved_sent_when_explicitly_enabled(self, monkeypatch):
        a = _configured_system(on_trail_moved=True)
        sent = self._sent_texts(a, monkeypatch)
        a.trail_moved(1, "XAUUSD", 2345, 2350, 2360)
        assert len(sent) == 1

    def test_system_alert_suppressed_when_toggle_off(self, monkeypatch):
        a = _configured_system(on_system=False)
        sent = self._sent_texts(a, monkeypatch)
        a.system_alert("test message")
        assert sent == []

    def test_guard_halt_ignores_its_own_toggle_by_design(self, monkeypatch):
        """guard_halt() is documented as 'always sent regardless of
        on_guard_halt (safety critical)' -- this pins that down so a
        future refactor can't silently make halts suppressible."""
        a = _configured_system(on_guard_halt=False)
        sent = self._sent_texts(a, monkeypatch)
        a.guard_halt("daily loss limit reached")
        assert len(sent) == 1

    def test_trade_closed_has_no_toggle_and_always_fires(self, monkeypatch):
        """Unlike every other event type, trade_closed() has no
        corresponding on_* flag at all and always sends. Documenting
        current behavior -- if a future change adds an on_trade_close
        toggle, this test should be updated alongside it."""
        a = _configured_system()
        sent = self._sent_texts(a, monkeypatch)
        a.trade_closed(1, "XAUUSD", "long", "tp3", 2366, 160.0, True, True)
        assert len(sent) == 1


# ─────────────────────────────────────────────────────────────────────────────
# Message content
# ─────────────────────────────────────────────────────────────────────────────

class TestMessageContent:
    def _last_sent(self, a, monkeypatch):
        sent = []
        monkeypatch.setattr(a, "_send_message", lambda text, **kw: sent.append(text) or True)
        return sent

    def test_signal_message_includes_key_levels(self, monkeypatch):
        a = _configured_system()
        sent = self._last_sent(a, monkeypatch)
        a.signal("XAUUSD", "M15", "long", 2350.0, 2345.0, 2352.0, 2359.0, 2366.0, 72.5, "LQ")
        text = sent[0]
        assert "XAUUSD" in text and "M15" in text
        assert "2350.00000" in text
        assert "72.5" in text
        assert "LQ" in text

    def test_signal_message_shows_long_vs_short_arrow(self, monkeypatch):
        a = _configured_system()
        sent = self._last_sent(a, monkeypatch)
        a.signal("XAUUSD", "M15", "short", 2350, 2355, 2348, 2341, 2334, 70, "LQ")
        assert "SHORT" in sent[0]

    def test_signal_message_lists_ceo_sequence_checks(self, monkeypatch):
        a = _configured_system()
        sent = self._last_sent(a, monkeypatch)
        a.signal("XAUUSD", "M15", "long", 2350, 2345, 2352, 2359, 2366, 72, "LQ",
                 bos=True, in_discount=True, ceo_valid=True, pat_name="Double Bottom")
        text = sent[0]
        assert "BOS confirmed" in text
        assert "Discount zone" in text
        assert "CEO sequence valid" in text
        assert "Double Bottom" in text

    def test_signal_message_without_checks_shows_base_sweep_fallback(self, monkeypatch):
        a = _configured_system()
        sent = self._last_sent(a, monkeypatch)
        a.signal("XAUUSD", "M15", "long", 2350, 2345, 2352, 2359, 2366, 72, "LQ")
        assert "Base sweep signal" in sent[0]

    def test_trade_closed_message_shows_pnl_sign(self, monkeypatch):
        a = _configured_system()
        sent = self._last_sent(a, monkeypatch)
        a.trade_closed(1, "XAUUSD", "long", "sl", 2345, -25.5, False, False)
        assert "-$25.50" in sent[0] or "$-25.50" in sent[0] or "-25.50" in sent[0]

    def test_trade_closed_message_shows_tp_flags(self, monkeypatch):
        a = _configured_system()
        sent = self._last_sent(a, monkeypatch)
        a.trade_closed(1, "XAUUSD", "long", "tp3", 2366, 100.0, True, False)
        assert "TP1✓" in sent[0]
        assert "TP2✗" in sent[0]

    def test_daily_summary_includes_guard_status_when_provided(self, monkeypatch):
        a = _configured_system()
        sent = self._last_sent(a, monkeypatch)
        a.daily_summary(
            {"trades": 10, "wins": 6, "losses": 4, "total_pnl": 55.0, "win_rate": 60.0, "avg_r": 0.8},
            guard_status={"drawdown_used_pct": 3.2, "daily_loss_used_pct": 1.1,
                          "trading_days_completed": 4, "trading_days_required": 10},
        )
        text = sent[0]
        assert "Account Guard" in text
        assert "4/10" in text

    def test_daily_summary_omits_guard_section_when_not_provided(self, monkeypatch):
        a = _configured_system()
        sent = self._last_sent(a, monkeypatch)
        a.daily_summary({"trades": 1, "wins": 1, "losses": 0, "total_pnl": 10.0})
        assert "Account Guard" not in sent[0]

    def test_signal_uses_chart_path_when_it_exists(self, monkeypatch, tmp_path):
        a = _configured_system()
        photo_calls = []
        msg_calls = []
        monkeypatch.setattr(a, "_send_photo", lambda path, caption="", **kw: photo_calls.append((path, caption)))
        monkeypatch.setattr(a, "_send_message", lambda text, **kw: msg_calls.append(text))
        chart = tmp_path / "chart.png"
        chart.write_bytes(b"fake png bytes")
        a.signal("XAUUSD", "M15", "long", 2350, 2345, 2352, 2359, 2366, 72, "LQ",
                 chart_path=str(chart))
        assert len(photo_calls) == 1
        assert msg_calls == []

    def test_signal_falls_back_to_text_when_chart_path_missing(self, monkeypatch):
        a = _configured_system()
        photo_calls = []
        msg_calls = []
        monkeypatch.setattr(a, "_send_photo", lambda path, caption="", **kw: photo_calls.append(path))
        monkeypatch.setattr(a, "_send_message", lambda text, **kw: msg_calls.append(text))
        a.signal("XAUUSD", "M15", "long", 2350, 2345, 2352, 2359, 2366, 72, "LQ",
                 chart_path="/nonexistent/path.png")
        assert photo_calls == []
        assert len(msg_calls) == 1


# ─────────────────────────────────────────────────────────────────────────────
# dry_run mode never touches the network
# ─────────────────────────────────────────────────────────────────────────────

class TestDryRunNoNetworkCalls:
    def test_send_message_dry_run_does_not_call_requests(self, monkeypatch, capsys):
        a = _configured_system(dry_run=True)
        called = []
        monkeypatch.setattr("ceo_engine_mt5.alerts.requests.post", lambda *a, **kw: called.append(1))
        result = a._send_message("hello")
        assert result is True
        assert called == []

    def test_send_photo_dry_run_does_not_call_requests(self, monkeypatch):
        a = _configured_system(dry_run=True)
        called = []
        monkeypatch.setattr("ceo_engine_mt5.alerts.requests.post", lambda *a, **kw: called.append(1))
        result = a._send_photo("/some/path.png", caption="hi")
        assert result is True
        assert called == []


# ─────────────────────────────────────────────────────────────────────────────
# Transport layer: retries, backoff, rate limiting, fallback
# ─────────────────────────────────────────────────────────────────────────────

class TestSendMessageTransport:
    def test_success_returns_true_and_updates_last_sent(self, monkeypatch):
        a = _configured_system(dry_run=False)
        monkeypatch.setattr("ceo_engine_mt5.alerts.requests.post",
                            lambda *a, **kw: _FakeResponse(200))
        monkeypatch.setattr(time, "sleep", lambda s: None)
        before = a._last_sent
        result = a._send_message("hello")
        assert result is True
        assert a._last_sent > before

    def test_non_200_non_429_retries_then_fails(self, monkeypatch):
        a = _configured_system(dry_run=False)
        calls = []
        monkeypatch.setattr("ceo_engine_mt5.alerts.requests.post",
                            lambda *a, **kw: calls.append(1) or _FakeResponse(500, text="server error"))
        monkeypatch.setattr(time, "sleep", lambda s: None)
        result = a._send_message("hello", retries=2)
        assert result is False
        assert len(calls) == 3   # initial attempt + 2 retries

    def test_429_backs_off_using_retry_after_then_succeeds(self, monkeypatch):
        a = _configured_system(dry_run=False)
        responses = [
            _FakeResponse(429, json_data={"parameters": {"retry_after": 3}}),
            _FakeResponse(200),
        ]
        monkeypatch.setattr("ceo_engine_mt5.alerts.requests.post",
                            lambda *a, **kw: responses.pop(0))
        slept = []
        monkeypatch.setattr(time, "sleep", lambda s: slept.append(s))
        result = a._send_message("hello", retries=2)
        assert result is True
        assert 3 in slept   # honored Telegram's retry_after value

    def test_exception_during_post_retries_then_fails(self, monkeypatch):
        a = _configured_system(dry_run=False)
        def _raise(*args, **kwargs):
            raise ConnectionError("network down")
        monkeypatch.setattr("ceo_engine_mt5.alerts.requests.post", _raise)
        monkeypatch.setattr(time, "sleep", lambda s: None)
        result = a._send_message("hello", retries=1)
        assert result is False

    def test_min_interval_enforced_between_sends(self, monkeypatch):
        a = _configured_system(dry_run=False)
        monkeypatch.setattr("ceo_engine_mt5.alerts.requests.post",
                            lambda *a, **kw: _FakeResponse(200))
        slept = []
        monkeypatch.setattr(time, "sleep", lambda s: slept.append(s))
        a._last_sent = time.time()   # simulate a message just sent
        a._send_message("second message")
        assert len(slept) == 1
        assert 0 < slept[0] <= a._MIN_INTERVAL


class TestSendPhotoTransport:
    def test_success_returns_true(self, monkeypatch, tmp_path):
        a = _configured_system(dry_run=False)
        img = tmp_path / "chart.png"
        img.write_bytes(b"fake")
        monkeypatch.setattr("ceo_engine_mt5.alerts.requests.post",
                            lambda *a, **kw: _FakeResponse(200))
        result = a._send_photo(str(img), caption="test")
        assert result is True

    def test_failure_falls_back_to_text_message(self, monkeypatch, tmp_path):
        a = _configured_system(dry_run=False)
        img = tmp_path / "chart.png"
        img.write_bytes(b"fake")
        monkeypatch.setattr("ceo_engine_mt5.alerts.requests.post",
                            lambda *a, **kw: _FakeResponse(500))
        fallback_calls = []
        monkeypatch.setattr(a, "_send_message", lambda text, **kw: fallback_calls.append(text) or True)
        result = a._send_photo(str(img), caption="test caption")
        assert result is True
        assert fallback_calls == ["test caption"]

    def test_exception_falls_back_to_text_message(self, monkeypatch, tmp_path):
        a = _configured_system(dry_run=False)
        img = tmp_path / "chart.png"
        img.write_bytes(b"fake")
        def _raise(*args, **kwargs):
            raise ConnectionError("network down")
        monkeypatch.setattr("ceo_engine_mt5.alerts.requests.post", _raise)
        monkeypatch.setattr(time, "sleep", lambda s: None)
        fallback_calls = []
        monkeypatch.setattr(a, "_send_message", lambda text, **kw: fallback_calls.append(text) or True)
        result = a._send_photo(str(img), caption="test caption", retries=1)
        assert result is True
        assert fallback_calls == ["test caption"]


class TestConnection:
    def test_test_connection_sends_expected_message(self, monkeypatch):
        a = _configured_system()
        sent = []
        monkeypatch.setattr(a, "_send_message", lambda text, **kw: sent.append(text) or True)
        result = a.test_connection()
        assert result is True
        assert "connected" in sent[0].lower()
