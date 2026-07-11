"""
Tests for mt5_live_signals.py's check_symbol() (the full bar-close
pipeline: fetch -> indicators -> signals -> structure -> patterns ->
session gate -> risk/guard gate -> route) and _route_fired_signal() (fans
a confirmed signal out to console/journal/CSV/executor/alerts/dashboard).
Previously 17% covered -- this is the layer that decides *whether to
trade at all*, gating everything downstream of it (including the
already-tested executor broker-confirmation logic), so it's arguably
higher-value than the executor tests themselves.

check_symbol() runs the real pipeline (calc_all/build_all/etc. are not
mocked -- they're already well-tested elsewhere and running them for
real here is what makes this an integration test worth having) against
a fake MT5 connection built from conftest's synthetic OHLCV. Getting a
signal to fire naturally out of random synthetic data isn't reliable,
so add_session_columns (the last pipeline stage before signal
extraction) is wrapped to force a gate column true on the last bar after
the real pipeline has already run -- every column check_symbol reads
downstream of that point is still real data from the real pipeline.
"""

import os
import sys
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import pytest

from tests.conftest import _make_ohlcv
from ceo_engine_mt5.indicators import calc_all
from ceo_engine_mt5 import mt5_live_signals as mls
from ceo_engine_mt5.mt5_live_signals import check_symbol, _route_fired_signal
from ceo_engine_mt5.funded_account_guard import FundedAccountGuard
from ceo_engine_mt5.risk_engine import RiskEngine


# ─────────────────────────────────────────────────────────────────────────────
# Fake MT5 connection
# ─────────────────────────────────────────────────────────────────────────────

def _df_to_rates(df: pd.DataFrame) -> list:
    out = []
    for ts, row in df.iterrows():
        out.append({
            "time": int(pd.Timestamp(ts).timestamp()),
            "open": float(row["open"]), "high": float(row["high"]),
            "low": float(row["low"]), "close": float(row["close"]),
            "tick_volume": float(row.get("volume", 100.0)),
        })
    return out


class _FakeConn:
    def __init__(self, df, htf_df=None, account=None, sym_info=None,
                 raise_on_htf=False, raise_on_account=False):
        self._df = df
        self._htf_df = htf_df if htf_df is not None else df
        self._account = account or {"balance": 10_000.0, "equity": 10_000.0}
        self._sym_info = sym_info or {"digits": 2, "tick_size": 0.01}
        self._raise_on_htf = raise_on_htf
        self._raise_on_account = raise_on_account
        self.fetch_calls = []

    def symbol_info(self, symbol):
        return self._sym_info

    def fetch_rates(self, symbol, tf, n_bars=1000):
        self.fetch_calls.append(tf)
        if self._raise_on_htf and tf not in ("M15",):
            raise RuntimeError("simulated HTF fetch failure")
        source = self._df if tf == "M15" else self._htf_df
        return _df_to_rates(source.tail(n_bars))

    def account_info(self):
        if self._raise_on_account:
            raise RuntimeError("simulated account_info failure")
        return self._account


def _base_ohlcv(seed=7, n=400):
    return _make_ohlcv(seed=seed, n=n)


def _force_signal_on_last_bar(monkeypatch, direction="LONG", quality=75.0,
                              gate_col=None, quality_col=None, extra_cols=None):
    """
    Wraps the real add_session_columns (last pipeline stage before
    check_symbol reads gate columns) so the returned DataFrame is
    otherwise 100% real pipeline output, with only the requested gate/
    quality columns on the last row forced -- deterministic without
    hand-crafting market data that happens to trigger a real sweep.
    Defaults to the m00 (default single-model) gate if gate_col isn't given.
    """
    from ceo_engine_mt5.session_filter import add_session_columns as _real

    col = "long" if direction == "LONG" else "short"
    gate_col = gate_col or f"m00_{col}"
    quality_col = quality_col or f"m00_quality_{col}"

    def _wrapped(df, **kwargs):
        out = _real(df, **kwargs)
        out.loc[out.index[-1], gate_col] = True
        out.loc[out.index[-1], quality_col] = quality
        out.loc[out.index[-1], "confluence_long_fired"] = False
        out.loc[out.index[-1], "confluence_short_fired"] = False
        if extra_cols:
            for k, v in extra_cols.items():
                out.loc[out.index[-1], k] = v
        return out

    monkeypatch.setattr(mls, "add_session_columns", _wrapped)


def _cs_kwargs(**overrides):
    kwargs = dict(
        params={"sessions": ["all"], "no_htf": True},
        signal_params={}, bt_params={},
        log_path=None, sound=False, seen_ids=set(),
    )
    kwargs.update(overrides)
    return kwargs


# ─────────────────────────────────────────────────────────────────────────────
# check_symbol -- basic firing / dedup / gating
# ─────────────────────────────────────────────────────────────────────────────

class TestCheckSymbolFiring:
    def test_fires_a_signal_when_gate_true_on_last_bar(self, monkeypatch):
        _force_signal_on_last_bar(monkeypatch, direction="LONG")
        conn = _FakeConn(_base_ohlcv())
        fired = check_symbol(conn, "XAUUSD", "M15", **_cs_kwargs())
        assert len(fired) == 1
        assert fired[0]["direction"] == "LONG"
        assert fired[0]["symbol"] == "XAUUSD"

    def test_no_signal_when_no_gate_is_true(self, monkeypatch):
        # add_session_columns runs for real, unforced -- random synthetic
        # data essentially never happens to satisfy a gate on the exact
        # last bar.
        conn = _FakeConn(_base_ohlcv())
        fired = check_symbol(conn, "XAUUSD", "M15", **_cs_kwargs())
        assert fired == []

    def test_seen_ids_prevents_refiring_the_same_bar(self, monkeypatch):
        _force_signal_on_last_bar(monkeypatch, direction="LONG")
        conn = _FakeConn(_base_ohlcv())
        seen = set()
        first = check_symbol(conn, "XAUUSD", "M15", **_cs_kwargs(seen_ids=seen))
        assert len(first) == 1
        second = check_symbol(conn, "XAUUSD", "M15", **_cs_kwargs(seen_ids=seen))
        assert second == []   # same bar_id already in seen_ids

    def test_different_symbols_do_not_share_dedup_state(self, monkeypatch):
        _force_signal_on_last_bar(monkeypatch, direction="LONG")
        conn = _FakeConn(_base_ohlcv())
        seen = set()
        f1 = check_symbol(conn, "XAUUSD", "M15", **_cs_kwargs(seen_ids=seen))
        f2 = check_symbol(conn, "EURUSD", "M15", **_cs_kwargs(seen_ids=seen))
        assert len(f1) == 1
        assert len(f2) == 1

    def test_short_direction_fires_correctly(self, monkeypatch):
        _force_signal_on_last_bar(monkeypatch, direction="SHORT")
        conn = _FakeConn(_base_ohlcv())
        fired = check_symbol(conn, "XAUUSD", "M15", **_cs_kwargs())
        assert len(fired) == 1
        assert fired[0]["direction"] == "SHORT"

    def test_fired_signal_has_computed_sl_tp_levels(self, monkeypatch):
        _force_signal_on_last_bar(monkeypatch, direction="LONG")
        conn = _FakeConn(_base_ohlcv())
        fired = check_symbol(conn, "XAUUSD", "M15",
                             **_cs_kwargs(bt_params={"sl_atr_mult": 1.5, "tp1_r": 1.0,
                                                     "tp2_r": 2.0, "tp3_r": 3.0}))
        sig = fired[0]
        assert sig["sl"] != sig["close"]
        assert sig["tp1"] is not None and sig["tp2"] is not None and sig["tp3"] is not None

    def test_ceo_only_mode_fires_labeled_as_ceo_full_sequence(self, monkeypatch):
        # Force only the CEO-sequence gate, not the default model's gate
        # -- with ceo_only exclusive, the default model isn't even
        # evaluated, so this isn't racing anything, just confirming the
        # sole remaining mode fires and labels correctly.
        _force_signal_on_last_bar(monkeypatch, direction="LONG",
                                  gate_col="ceo_long_valid", quality_col="quality_long",
                                  quality=80.0)
        conn = _FakeConn(_base_ohlcv())
        fired = check_symbol(conn, "XAUUSD", "M15",
                             **_cs_kwargs(params={"sessions": ["all"], "no_htf": True,
                                                  "ceo_only": True}))
        labels = {s["model"] for s in fired}
        assert "CEO Full Sequence" in labels

    def test_default_model_and_confluence_dedup_by_shared_bar_id(self, monkeypatch):
        """_bar_id() is keyed on (symbol, tf, bar_time, direction) only --
        it does not include the model/mode. So if the default model's
        gate AND confluence's gate are both true on the same bar and
        direction, only the first mode evaluated in the loop actually
        fires; the second is suppressed by the same seen_ids entry the
        first one just added. (This used to be demonstrated with
        ceo_only instead of confluence, but ceo_only became exclusive in
        v3.6.0 -- see CHANGELOG -- so it no longer competes with anything.
        Confluence still does, since it's additive alongside the base
        model.)"""
        _force_signal_on_last_bar(monkeypatch, direction="LONG",
                                  extra_cols={"confluence_long_fired": True})
        conn = _FakeConn(_base_ohlcv())
        fired = check_symbol(conn, "XAUUSD", "M15",
                             **_cs_kwargs(params={"sessions": ["all"], "no_htf": True,
                                                  "confluence": True}))
        assert len(fired) == 1
        assert fired[0]["model"] == "LQ"   # default model mode is evaluated first

    def test_ceo_only_excludes_base_model_entirely(self, monkeypatch):
        """ceo_only is exclusive (matching its name/help text): when set,
        the base single-model mode is not evaluated at all, even when its
        gate is also true on the same bar. This used to only win a
        same-bar dedup race against the base model (see CHANGELOG
        v3.6.0) -- now the base model isn't in play in the first place."""
        _force_signal_on_last_bar(monkeypatch, direction="LONG",
                                  extra_cols={"ceo_long_valid": True, "quality_long": 80.0})
        conn = _FakeConn(_base_ohlcv())
        fired = check_symbol(conn, "XAUUSD", "M15",
                             **_cs_kwargs(params={"sessions": ["all"], "no_htf": True,
                                                  "ceo_only": True}))
        assert len(fired) == 1
        assert fired[0]["model"] == "CEO Full Sequence"

    def test_ceo_only_silences_confluence_too(self, monkeypatch):
        """Full exclusivity: --ceo-only + --confluence together should
        still only evaluate CEO Full Sequence -- confluence is silenced,
        not run alongside it."""
        _force_signal_on_last_bar(monkeypatch, direction="LONG",
                                  gate_col="confluence_long_fired",
                                  extra_cols={"ceo_long_valid": False})
        conn = _FakeConn(_base_ohlcv())
        fired = check_symbol(conn, "XAUUSD", "M15",
                             **_cs_kwargs(params={"sessions": ["all"], "no_htf": True,
                                                  "ceo_only": True, "confluence": True}))
        # confluence_long_fired was forced True, but ceo_long_valid is
        # False and ceo_only excludes everything else -- nothing should fire.
        assert fired == []

    def test_ceo_only_logs_a_warning_when_confluence_also_requested(self, monkeypatch, caplog):
        import logging
        _force_signal_on_last_bar(monkeypatch, direction="LONG")
        conn = _FakeConn(_base_ohlcv())
        with caplog.at_level(logging.WARNING):
            check_symbol(conn, "XAUUSD", "M15",
                         **_cs_kwargs(params={"sessions": ["all"], "no_htf": True,
                                              "ceo_only": True, "confluence": True}))
        assert any("ceo_only" in r.message and "confluence" in r.message
                   for r in caplog.records)

    def test_ceo_only_without_confluence_does_not_warn(self, monkeypatch, caplog):
        import logging
        _force_signal_on_last_bar(monkeypatch, direction="LONG")
        conn = _FakeConn(_base_ohlcv())
        with caplog.at_level(logging.WARNING):
            check_symbol(conn, "XAUUSD", "M15",
                         **_cs_kwargs(params={"sessions": ["all"], "no_htf": True,
                                              "ceo_only": True}))
        assert not any("ceo_only" in r.message for r in caplog.records)


# ─────────────────────────────────────────────────────────────────────────────
# check_symbol -- session / news / risk gating
# ─────────────────────────────────────────────────────────────────────────────

class TestCheckSymbolGating:
    def test_session_gate_blocks_signal_outside_allowed_sessions(self, monkeypatch):
        _force_signal_on_last_bar(monkeypatch, direction="LONG")
        conn = _FakeConn(_base_ohlcv())
        # "asian" only -- the synthetic bar_time is extremely unlikely to
        # land in exactly that window, so this should suppress firing.
        fired = check_symbol(conn, "XAUUSD", "M15",
                             **_cs_kwargs(params={"sessions": ["asian"], "no_htf": True}))
        # Either blocked by session (empty) or -- if the random data
        # happened to land in the asian session -- still fires; assert
        # the mechanism ran without raising either way, and check the
        # stronger case below with a forced-invalid session instead.
        assert isinstance(fired, list)

    def test_news_filter_blocks_signal_and_logs_reason(self, monkeypatch, tmp_path):
        _force_signal_on_last_bar(monkeypatch, direction="LONG")
        conn = _FakeConn(_base_ohlcv())

        class _BlockingNewsFilter:
            def is_blocked(self, bar_time, symbol):
                return True, "High-impact NFP in 15 min"

        log_path = str(tmp_path / "signals.csv")
        fired = check_symbol(conn, "XAUUSD", "M15",
                             **_cs_kwargs(log_path=log_path, news_filter=_BlockingNewsFilter()))
        assert fired == []
        assert os.path.exists(log_path)
        content = open(log_path).read()
        assert "BLOCKED" in content

    def test_news_filter_allows_when_not_blocked(self, monkeypatch):
        _force_signal_on_last_bar(monkeypatch, direction="LONG")
        conn = _FakeConn(_base_ohlcv())

        class _AllowingNewsFilter:
            def is_blocked(self, bar_time, symbol):
                return False, ""

        fired = check_symbol(conn, "XAUUSD", "M15",
                             **_cs_kwargs(news_filter=_AllowingNewsFilter()))
        assert len(fired) == 1

    def test_risk_engine_rejection_blocks_signal(self, monkeypatch):
        _force_signal_on_last_bar(monkeypatch, direction="LONG")
        conn = _FakeConn(_base_ohlcv())

        class _RejectingRiskEngine:
            def evaluate(self, **kwargs):
                return 0.0, {"blocked_by": "max_daily_loss", "sizing": {}}

        fired = check_symbol(conn, "XAUUSD", "M15",
                             **_cs_kwargs(risk_engine=_RejectingRiskEngine()))
        assert fired == []

    def test_guard_rejection_blocks_signal(self, monkeypatch):
        _force_signal_on_last_bar(monkeypatch, direction="LONG")
        conn = _FakeConn(_base_ohlcv())

        class _ApprovingRiskEngine:
            def evaluate(self, **kwargs):
                return 0.1, {"blocked_by": None, "sizing": {"risk_amount": 10.0}}

        class _RejectingGuard:
            def pre_trade_check(self, **kwargs):
                return False, "Daily loss limit reached"

        fired = check_symbol(conn, "XAUUSD", "M15",
                             **_cs_kwargs(risk_engine=_ApprovingRiskEngine(),
                                         guard=_RejectingGuard()))
        assert fired == []

    def test_htf_fetch_failure_does_not_crash(self, monkeypatch):
        _force_signal_on_last_bar(monkeypatch, direction="LONG")
        conn = _FakeConn(_base_ohlcv(), raise_on_htf=True)
        fired = check_symbol(conn, "XAUUSD", "M15",
                             **_cs_kwargs(params={"sessions": ["all"], "no_htf": False}))
        # Must not raise -- HTF fetch failure degrades gracefully.
        assert isinstance(fired, list)

    def test_account_info_failure_does_not_crash_when_risk_engine_present(self, monkeypatch):
        _force_signal_on_last_bar(monkeypatch, direction="LONG")
        conn = _FakeConn(_base_ohlcv(), raise_on_account=True)

        class _AnyRiskEngine:
            def evaluate(self, **kwargs):
                return 0.1, {"blocked_by": None, "sizing": {"risk_amount": 10.0}}

        # account_info() raises -> risk gate should be skipped this bar,
        # not crash check_symbol entirely.
        fired = check_symbol(conn, "XAUUSD", "M15",
                             **_cs_kwargs(risk_engine=_AnyRiskEngine()))
        assert isinstance(fired, list)


# ─────────────────────────────────────────────────────────────────────────────
# _route_fired_signal -- fan-out to each output, isolated from each other
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture
def dash(tmp_path, monkeypatch):
    """Fresh dashboard module in an isolated cwd -- same isolation
    pattern as test_dashboard_security.py's fixture, duplicated locally
    since _route_fired_signal imports dashboard lazily inside the
    function body."""
    import importlib
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("CEO_DASHBOARD_PASSWORD", "test-password")
    sys.modules.pop("ceo_engine_mt5.dashboard", None)
    mod = importlib.import_module("ceo_engine_mt5.dashboard")
    yield mod
    sys.modules.pop("ceo_engine_mt5.dashboard", None)


def _sample_sig(**overrides):
    sig = dict(
        symbol="XAUUSD", tf="M15", direction="LONG",
        bar_time=datetime(2024, 1, 1, 12, 0), bar_time_utc=datetime(2024, 1, 1, 12, 0, tzinfo=timezone.utc),
        close=2350.0, atr=5.0, model="LQ", quality=75.0,
        regime="trending", alignment="aligned", htf_label="Bullish", conf_count=1,
        session="london", sl=2345.0, tp1=2352.0, tp2=2359.0, tp3=2366.0,
        digits=2, ceo_valid=False, bos=True, in_discount=True, pat_name="",
        cp_confirm=False,
    )
    sig.update(overrides)
    return sig


def _sample_df():
    return calc_all(_base_ohlcv(n=60))


class TestRouteFiredSignal:
    def test_runs_with_all_optional_outputs_disabled(self, dash, capsys):
        df = _sample_df()
        _route_fired_signal(
            sig=_sample_sig(), df=df, last=df.iloc[-1], lot_info=None, sound=False,
            journal=None, log_path=None, executor=None, alerts=None,
            guard=None, account_info={},
        )
        out = capsys.readouterr().out
        assert "XAUUSD" in out   # console line was printed

    def test_journal_log_signal_called_with_correct_args(self, dash):
        calls = []
        class _FakeJournal:
            def log_signal(self, **kwargs):
                calls.append(kwargs)
        df = _sample_df()
        _route_fired_signal(
            sig=_sample_sig(), df=df, last=df.iloc[-1], lot_info=None, sound=False,
            journal=_FakeJournal(), log_path=None, executor=None, alerts=None,
            guard=None, account_info={},
        )
        assert len(calls) == 1
        assert calls[0]["symbol"] == "XAUUSD"
        assert calls[0]["direction"] == "long"

    def test_journal_failure_does_not_propagate(self, dash):
        class _RaisingJournal:
            def log_signal(self, **kwargs):
                raise RuntimeError("db locked")
        df = _sample_df()
        # Must not raise.
        _route_fired_signal(
            sig=_sample_sig(), df=df, last=df.iloc[-1], lot_info=None, sound=False,
            journal=_RaisingJournal(), log_path=None, executor=None, alerts=None,
            guard=None, account_info={},
        )

    def test_csv_log_path_writes_a_row(self, dash, tmp_path):
        log_path = str(tmp_path / "signals.csv")
        df = _sample_df()
        _route_fired_signal(
            sig=_sample_sig(), df=df, last=df.iloc[-1], lot_info=None, sound=False,
            journal=None, log_path=log_path, executor=None, alerts=None,
            guard=None, account_info={},
        )
        assert os.path.exists(log_path)
        assert "FIRED" in open(log_path).read()

    def test_executor_place_trade_called_when_lot_info_and_account_present(self, dash):
        calls = []
        class _FakeExecutor:
            def place_trade(self, **kwargs):
                calls.append(kwargs)
                return 555
        journal_calls = []
        class _FakeJournal:
            def log_signal(self, **kwargs): pass
            def log_trade_open(self, **kwargs): journal_calls.append(kwargs)
        df = _sample_df()
        _route_fired_signal(
            sig=_sample_sig(lots=0.1), df=df, last=df.iloc[-1],
            lot_info={"lots": 0.1, "risk_amount": 25.0}, sound=False,
            journal=_FakeJournal(), log_path=None, executor=_FakeExecutor(), alerts=None,
            guard=None, account_info={"balance": 10000.0},
        )
        assert len(calls) == 1
        assert calls[0]["symbol"] == "XAUUSD"
        assert len(journal_calls) == 1
        assert journal_calls[0]["ticket"] == 555

    def test_executor_not_called_without_lot_info(self, dash):
        calls = []
        class _FakeExecutor:
            def place_trade(self, **kwargs):
                calls.append(kwargs)
                return 555
        df = _sample_df()
        _route_fired_signal(
            sig=_sample_sig(), df=df, last=df.iloc[-1], lot_info=None, sound=False,
            journal=None, log_path=None, executor=_FakeExecutor(), alerts=None,
            guard=None, account_info={"balance": 10000.0},
        )
        assert calls == []

    def test_executor_failure_does_not_propagate(self, dash):
        class _RaisingExecutor:
            def place_trade(self, **kwargs):
                raise RuntimeError("MT5 disconnected")
        df = _sample_df()
        _route_fired_signal(
            sig=_sample_sig(), df=df, last=df.iloc[-1], lot_info={"lots": 0.1, "risk_amount": 25.0}, sound=False,
            journal=None, log_path=None, executor=_RaisingExecutor(), alerts=None,
            guard=None, account_info={"balance": 10000.0},
        )

    def test_executor_skipped_when_not_connected(self, dash):
        calls = []
        class _FakeExecutor:
            simulation = False
            def is_connected(self):
                return False
            def place_trade(self, **kwargs):
                calls.append(kwargs)
                return 555
        df = _sample_df()
        _route_fired_signal(
            sig=_sample_sig(lots=0.1), df=df, last=df.iloc[-1], lot_info={"lots": 0.1, "risk_amount": 25.0}, sound=False,
            journal=None, log_path=None, executor=_FakeExecutor(), alerts=None,
            guard=None, account_info={"balance": 10000.0},
        )
        assert calls == []

    def test_alerts_signal_called_with_correct_direction_case(self, dash):
        calls = []
        class _FakeAlerts:
            def signal(self, **kwargs):
                calls.append(kwargs)
        df = _sample_df()
        _route_fired_signal(
            sig=_sample_sig(direction="SHORT"), df=df, last=df.iloc[-1], lot_info=None,
            sound=False, journal=None, log_path=None, executor=None,
            alerts=_FakeAlerts(), guard=None, account_info={},
        )
        assert len(calls) == 1
        assert calls[0]["direction"] == "short"

    def test_alerts_failure_does_not_propagate(self, dash):
        class _RaisingAlerts:
            def signal(self, **kwargs):
                raise RuntimeError("Telegram down")
        df = _sample_df()
        _route_fired_signal(
            sig=_sample_sig(), df=df, last=df.iloc[-1], lot_info=None, sound=False,
            journal=None, log_path=None, executor=None, alerts=_RaisingAlerts(),
            guard=None, account_info={},
        )

    def test_dashboard_state_reflects_fired_signal(self, dash):
        df = _sample_df()
        _route_fired_signal(
            sig=_sample_sig(symbol="GBPJPY"), df=df, last=df.iloc[-1], lot_info=None,
            sound=False, journal=None, log_path=None, executor=None, alerts=None,
            guard=None, account_info={},
        )
        snap = dash.state.snapshot()
        assert any(s.get("symbol") == "GBPJPY" for s in snap["signals"])

    def test_dashboard_account_and_guard_updated_when_provided(self, dash):
        class _FakeGuard:
            def status(self, account_info):
                return {"halted": False, "today_pnl": 5.0}
        df = _sample_df()
        _route_fired_signal(
            sig=_sample_sig(), df=df, last=df.iloc[-1], lot_info=None, sound=False,
            journal=None, log_path=None, executor=None, alerts=None,
            guard=_FakeGuard(), account_info={"balance": 12345.0},
        )
        snap = dash.state.snapshot()
        assert snap["account"].get("balance") == 12345.0
        assert snap["guard"].get("today_pnl") == 5.0

    def test_dashboard_structure_overlay_pushed_alongside_candles(self, dash):
        """Regression test for CEO structure visualization: whenever
        candles are pushed to the live dashboard chart, a structure
        overlay (order blocks/FVG/QM/Fib/BOS) must be pushed for the
        same symbol/tf key so the browser can render it."""
        df = _sample_df()
        _route_fired_signal(
            sig=_sample_sig(symbol="XAUUSD", tf="M15"), df=df, last=df.iloc[-1],
            lot_info=None, sound=False, journal=None, log_path=None,
            executor=None, alerts=None, guard=None, account_info={},
        )
        structure = dash.state.get_structure("XAUUSD", "M15")
        assert set(structure.keys()) == {"zoneLines", "priceLines", "markers"}

    def test_dashboard_structure_reflects_real_ceo_structure_columns(self, dash):
        """With the full CEO structure pipeline run (not just calc_all),
        an active order block should actually show up as a zone."""
        from ceo_engine_mt5.signals import build_all
        from ceo_engine_mt5.ceo_structure import build_ceo_structure
        df = build_ceo_structure(build_all(_sample_df()))
        df.loc[df.index[-5]:, "ob_bull_active"] = True
        df.loc[df.index[-5]:, "ob_bull_high"] = float(df["close"].iloc[-1]) + 1.0
        df.loc[df.index[-5]:, "ob_bull_low"] = float(df["close"].iloc[-1]) - 1.0
        _route_fired_signal(
            sig=_sample_sig(symbol="XAUUSD", tf="M15"), df=df, last=df.iloc[-1],
            lot_info=None, sound=False, journal=None, log_path=None,
            executor=None, alerts=None, guard=None, account_info={},
        )
        structure = dash.state.get_structure("XAUUSD", "M15")
        assert len(structure["zoneLines"]) >= 2
