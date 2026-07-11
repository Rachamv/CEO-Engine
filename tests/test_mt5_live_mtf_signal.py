"""
Tests for _handle_mtf_signal() in mt5_live_session.py -- the MTF-mode
counterpart to check_symbol()'s single-TF signal routing. Fakes
mtf_stack.check() to return a real MTFResult (so .summary() and field
access behave exactly as production code expects) and fakes the
downstream collaborators (conn, risk_engine, guard, journal, alerts,
executor) the same way test_mt5_live_signals_main_loop.py does for the
single-TF path.
"""

from datetime import datetime, timezone

import pandas as pd
import pytest

from ceo_engine_mt5 import mt5_live_session as mls
from ceo_engine_mt5.mt5_live_session import _handle_mtf_signal
from ceo_engine_mt5.multi_tf import MTFResult


def _entry_df():
    idx = pd.date_range("2024-01-01", periods=3, freq="15min", tz="UTC")
    return pd.DataFrame({
        "open":  [1.10, 1.11, 1.12],
        "high":  [1.12, 1.13, 1.14],
        "low":   [1.09, 1.10, 1.11],
        "close": [1.11, 1.12, 1.13],
        "volume":[100, 110, 120],
        "atr":   [0.002, 0.002, 0.002],
    }, index=idx)


def _valid_result(direction="LONG", score=80.0, tfs=None):
    edf = _entry_df()
    return MTFResult(
        valid=True, direction=direction, symbol="EURUSD",
        tfs=tfs or ["H4", "H1", "M15"], score=score, entry_tf="M15",
        entry_last=edf.iloc[-1], entry_df=edf,
        bar_time=edf.index[-1].to_pydatetime(), mode="bias",
    )


def _invalid_result():
    return MTFResult(valid=False, direction="", symbol="EURUSD",
                      tfs=["H4", "H1"], score=0.0, entry_tf="M15")


class _FakeMtfStack:
    def __init__(self, result):
        self._result = result
        self.checked = []

    def check(self, symbol, conn_local, verbose=False):
        self.checked.append(symbol)
        return self._result


class _FakeConnLocal:
    def __init__(self, account=None, sym_info=None, raise_on_account=False):
        self._account = account if account is not None else {"balance": 10_000.0}
        self._sym_info = sym_info or {"digits": 5, "spread": 2}
        self._raise_on_account = raise_on_account

    def account_info(self):
        if self._raise_on_account:
            raise RuntimeError("account_info boom")
        return self._account

    def symbol_info(self, symbol):
        return self._sym_info


class _FakeRiskEngine:
    def __init__(self, lots=0.10):
        self._lots = lots
        self.calls = []

    def evaluate(self, **kwargs):
        self.calls.append(kwargs)
        return self._lots, {"risk_amount": 25.0}


class _FakeGuard:
    def __init__(self):
        self.status_calls = []

    def status(self, account_info):
        self.status_calls.append(account_info)
        return {"blocked": False}


class _FakeJournal:
    def __init__(self, raise_error=False):
        self.logged = []
        self._raise = raise_error

    def log_signal(self, **kwargs):
        if self._raise:
            raise RuntimeError("journal boom")
        self.logged.append(kwargs)


class _FakeAlerts:
    def __init__(self, raise_error=False):
        self.sent = []
        self._raise = raise_error

    def signal(self, **kwargs):
        if self._raise:
            raise RuntimeError("alert boom")
        self.sent.append(kwargs)


class _FakeExecutor:
    def __init__(self, raise_error=False):
        self.trades = []
        self._raise = raise_error

    def place_trade(self, **kwargs):
        if self._raise:
            raise RuntimeError("executor boom")
        self.trades.append(kwargs)


class _FakeNewsFilter:
    def __init__(self, blocked=False, reason="High-impact news"):
        self._blocked = blocked
        self._reason = reason

    def is_blocked(self, bar_time, symbol):
        return self._blocked, self._reason


def _call(monkeypatch, mtf_result, **kwargs):
    stack = _FakeMtfStack(mtf_result)
    dash_calls = {"candles": [], "signal": [], "account": [], "guard": [], "structure": []}
    import ceo_engine_mt5.dashboard as dash
    monkeypatch.setattr(dash, "update_candles", lambda *a, **k: dash_calls["candles"].append((a, k)))
    monkeypatch.setattr(dash, "update_structure", lambda *a, **k: dash_calls["structure"].append((a, k)))
    monkeypatch.setattr(dash, "update_signal", lambda payload: dash_calls["signal"].append(payload))
    monkeypatch.setattr(dash, "update_account", lambda a: dash_calls["account"].append(a))
    monkeypatch.setattr(dash, "update_guard", lambda g: dash_calls["guard"].append(g))

    defaults = dict(
        mtf_stack=stack, symbol="EURUSD", conn_local=_FakeConnLocal(),
        tf="M15", bt_params={}, cur_bar=datetime(2024, 1, 1, 0, 45, tzinfo=timezone.utc),
        seen_ids=set(),
        risk_engine=None, guard=None, journal=None, alerts_obj=None,
        executor=None, risk_pct=1.0, news_filter=None,
    )
    defaults.update(kwargs)
    fired = _handle_mtf_signal(**defaults)
    return fired, stack, dash_calls


class TestHandleMtfSignalInvalid:
    def test_invalid_result_fires_nothing(self, monkeypatch):
        fired, stack, dash_calls = _call(monkeypatch, _invalid_result())
        assert fired == []
        assert stack.checked == ["EURUSD"]

    def test_candles_still_pushed_when_entry_df_present_but_invalid(self, monkeypatch):
        result = _invalid_result()
        result.entry_df = _entry_df()
        _, _, dash_calls = _call(monkeypatch, result)
        assert len(dash_calls["candles"]) == 1

    def test_no_candle_push_when_entry_df_missing(self, monkeypatch):
        _, _, dash_calls = _call(monkeypatch, _invalid_result())
        assert dash_calls["candles"] == []

    def test_candle_push_failure_does_not_propagate(self, monkeypatch):
        result = _invalid_result()
        result.entry_df = _entry_df()
        import ceo_engine_mt5.dashboard as dash
        monkeypatch.setattr(dash, "update_candles",
                             lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom")))
        fired, _, _ = _call(monkeypatch, result)
        assert fired == []  # doesn't crash despite the dashboard failure

    def test_structure_pushed_alongside_candles(self, monkeypatch):
        result = _invalid_result()
        result.entry_df = _entry_df()
        _, _, dash_calls = _call(monkeypatch, result)
        assert len(dash_calls["structure"]) == 1
        (args, _kwargs) = dash_calls["structure"][0]
        payload = args[2]
        assert set(payload.keys()) == {"zoneLines", "priceLines", "markers"}

    def test_structure_push_failure_does_not_propagate(self, monkeypatch):
        result = _invalid_result()
        result.entry_df = _entry_df()
        import ceo_engine_mt5.dashboard as dash
        monkeypatch.setattr(dash, "update_structure",
                             lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom")))
        fired, _, _ = _call(monkeypatch, result)
        assert fired == []


class TestHandleMtfSignalNewsGate:
    def test_blocked_news_prevents_firing(self, monkeypatch, capsys):
        fired, _, dash_calls = _call(monkeypatch, _valid_result(),
                                      news_filter=_FakeNewsFilter(blocked=True))
        assert fired == []
        assert "🚫" in capsys.readouterr().out

    def test_allowed_news_lets_it_proceed(self, monkeypatch):
        fired, _, _ = _call(monkeypatch, _valid_result(),
                             news_filter=_FakeNewsFilter(blocked=False))
        assert len(fired) == 1


class TestHandleMtfSignalDedup:
    def test_same_bar_id_is_not_refired(self, monkeypatch):
        from ceo_engine_mt5.mt5_live_utils import _bar_id
        result = _valid_result()
        cur_bar = datetime(2024, 1, 1, 0, 45, tzinfo=timezone.utc)
        bid = _bar_id("EURUSD", "M15", bar_time=cur_bar, direction=result.direction)
        seen = {bid}
        fired, _, _ = _call(monkeypatch, result, cur_bar=cur_bar, seen_ids=seen)
        assert fired == []

    def test_different_bar_fires_normally(self, monkeypatch):
        result = _valid_result()
        cur_bar = datetime(2024, 1, 1, 1, 0, tzinfo=timezone.utc)
        fired, _, _ = _call(monkeypatch, result, cur_bar=cur_bar, seen_ids=set())
        assert len(fired) == 1


class TestHandleMtfSignalFullRoute:
    def test_fires_and_routes_through_all_collaborators(self, monkeypatch):
        risk_engine = _FakeRiskEngine(lots=0.20)
        guard = _FakeGuard()
        journal = _FakeJournal()
        alerts = _FakeAlerts()
        executor = _FakeExecutor()
        conn = _FakeConnLocal(account={"balance": 10_000.0})

        fired, stack, dash_calls = _call(
            monkeypatch, _valid_result(direction="LONG", score=88.0),
            conn_local=conn, risk_engine=risk_engine, guard=guard,
            journal=journal, alerts_obj=alerts, executor=executor,
        )

        assert len(fired) == 1
        assert risk_engine.calls  # evaluate() was called
        assert journal.logged and journal.logged[0]["traded"] is True
        assert alerts.sent
        assert executor.trades
        assert dash_calls["signal"]
        assert dash_calls["account"]
        assert dash_calls["guard"]

    def test_zero_lots_skips_executor_but_not_journal(self, monkeypatch):
        risk_engine = _FakeRiskEngine(lots=0.0)
        journal = _FakeJournal()
        executor = _FakeExecutor()

        fired, _, _ = _call(
            monkeypatch, _valid_result(),
            risk_engine=risk_engine, journal=journal, executor=executor,
        )
        assert len(fired) == 1
        assert journal.logged  # still logs the signal even if not traded
        assert journal.logged[0]["traded"] is False
        assert executor.trades == []

    def test_journal_failure_does_not_propagate(self, monkeypatch):
        journal = _FakeJournal(raise_error=True)
        fired, _, _ = _call(monkeypatch, _valid_result(), journal=journal)
        assert len(fired) == 1

    def test_alerts_failure_does_not_propagate(self, monkeypatch):
        alerts = _FakeAlerts(raise_error=True)
        risk_engine = _FakeRiskEngine(lots=0.1)
        fired, _, _ = _call(monkeypatch, _valid_result(), risk_engine=risk_engine, alerts_obj=alerts)
        assert len(fired) == 1

    def test_executor_failure_does_not_propagate(self, monkeypatch):
        executor = _FakeExecutor(raise_error=True)
        risk_engine = _FakeRiskEngine(lots=0.1)
        fired, _, _ = _call(monkeypatch, _valid_result(), risk_engine=risk_engine, executor=executor)
        assert len(fired) == 1

    def test_account_info_failure_is_tolerated(self, monkeypatch):
        conn = _FakeConnLocal(raise_on_account=True)
        risk_engine = _FakeRiskEngine(lots=0.1)
        fired, _, _ = _call(monkeypatch, _valid_result(), conn_local=conn, risk_engine=risk_engine)
        # account_info() failing shouldn't crash the whole handler; risk_engine
        # just won't get to size a trade since account_info is falsy ({}).
        assert len(fired) == 1

    def test_alerts_fire_even_at_zero_lots_when_risk_pct_zero(self, monkeypatch):
        """alerts_obj.signal() should still fire when risk_pct==0 (alert-only
        mode) even though lots is 0 -- matches the single-TF path's semantics."""
        alerts = _FakeAlerts()
        fired, _, _ = _call(monkeypatch, _valid_result(), alerts_obj=alerts, risk_pct=0)
        assert alerts.sent

    def test_dashboard_push_failure_does_not_propagate(self, monkeypatch):
        import ceo_engine_mt5.dashboard as dash
        stack = _FakeMtfStack(_valid_result())
        monkeypatch.setattr(dash, "update_candles", lambda *a, **k: None)
        monkeypatch.setattr(dash, "update_signal",
                             lambda payload: (_ for _ in ()).throw(RuntimeError("boom")))
        fired = _handle_mtf_signal(
            mtf_stack=stack, symbol="EURUSD", conn_local=_FakeConnLocal(),
            tf="M15", bt_params={},
            cur_bar=datetime(2024, 1, 1, 0, 45, tzinfo=timezone.utc), seen_ids=set(),
            risk_engine=None, guard=None, journal=None, alerts_obj=None,
            executor=None, risk_pct=1.0, news_filter=None,
        )
        assert len(fired) == 1
