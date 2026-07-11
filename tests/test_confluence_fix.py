"""
Regression tests for the confluence-quality-gate ordering fix.

Root cause: confluence_signals()'s quality gate reads m00_quality_long/
short (the bare LQ model -- no filters, so its score is just a
25-point base +/- a 15-point regime swing, capped at 40). ceo_structure.
validate_ceo_sequence() adds a further 0-30 structural bonus to that same
column, but only later in the pipeline. Calling confluence_signals()
before that bonus existed (previously bundled inside signals.build_all())
meant the quality gate (45 internal default, 50-60 in every real
run.py/mt5_live.py/launcher.py config) could never be satisfied by a
value capped at 40 -- confirmed via backtest: "Confluence" always showed
exactly 0 trades, in every config, on every dataset.

Fix: confluence_signals() (now build_confluence(), a thin params-dict
wrapper) is no longer called inside build_all(). Every full-pipeline
call site now calls it explicitly, positioned after build_ceo_structure()
so m00_quality_long/short already reflects the structural bonus.

These tests exist specifically because there was zero prior coverage of
confluence_signals() at all -- that's exactly how this went unnoticed.
"""

import sys

import numpy as np
import pandas as pd
import pytest

from tests.conftest import _make_ohlcv
from tests.test_mt5_live_signals_main_loop import _FakeConn, _base_ohlcv, _cs_kwargs
from ceo_engine_mt5.indicators import calc_all
from ceo_engine_mt5.signals import build_all, build_confluence, confluence_signals
from ceo_engine_mt5.candle_patterns import build_candle_patterns
from ceo_engine_mt5.ceo_structure import build_ceo_structure
from ceo_engine_mt5.backtest import run_backtest, results_table
from ceo_engine_mt5.mt5_live_signals import check_symbol


# ─────────────────────────────────────────────────────────────────────────────
# API shape: build_all() no longer produces confluence columns by itself
# ─────────────────────────────────────────────────────────────────────────────

class TestBuildAllNoLongerIncludesConfluence:
    def test_build_all_alone_has_no_confluence_columns(self, synthetic_ohlcv):
        df = calc_all(synthetic_ohlcv)
        df = build_all(df)
        assert "confluence_long_fired" not in df.columns
        assert "confluence_short_fired" not in df.columns
        assert "confluence_long_count" not in df.columns

    def test_m00_quality_capped_at_40_before_ceo_structure_runs(self, synthetic_ohlcv):
        """Pins down the root cause directly: m00's quality score has no
        filter components, so before the CEO structural bonus exists,
        its range is exactly [10, 40] -- below every real quality-gate
        threshold in the codebase (45/50/60)."""
        df = calc_all(synthetic_ohlcv)
        df = build_all(df)
        assert df["m00_quality_long"].min() >= 10.0
        assert df["m00_quality_long"].max() <= 40.0
        assert df["m00_quality_short"].max() <= 40.0


# ─────────────────────────────────────────────────────────────────────────────
# The fix: confluence can now actually fire, with the CEO bonus applied
# ─────────────────────────────────────────────────────────────────────────────

class TestBuildConfluenceAfterCeoStructure:
    def test_m00_quality_can_exceed_40_after_ceo_structure(self, synthetic_ohlcv_large):
        df = calc_all(synthetic_ohlcv_large)
        df = build_all(df)
        df = build_candle_patterns(df)
        df = build_ceo_structure(df)
        # The whole point of the fix: this must now be possible.
        assert df["m00_quality_long"].max() > 40.0

    def test_confluence_can_fire_end_to_end(self, synthetic_ohlcv_large):
        df = calc_all(synthetic_ohlcv_large)
        df = build_all(df)
        df = build_candle_patterns(df)
        df = build_ceo_structure(df)
        df = build_confluence(df)
        # Not just "doesn't crash" -- must actually be able to fire.
        assert df["confluence_long_fired"].sum() + df["confluence_short_fired"].sum() > 0

    def test_backtest_confluence_row_shows_nonzero_trades(self, synthetic_ohlcv_large):
        """The exact check that would have caught this originally: run a
        real backtest and look at the Confluence row, the same way a user
        looking at their own backtest report would."""
        df = calc_all(synthetic_ohlcv_large)
        df = build_all(df)
        df = build_candle_patterns(df)
        df = build_ceo_structure(df)
        df = build_confluence(df)
        bt = run_backtest(df)
        tbl = results_table(bt)
        assert "Confluence" in tbl.index
        assert tbl.loc["Confluence", "Trades"] > 0

    def test_build_confluence_returns_dataframe_unchanged_in_shape(self, synthetic_ohlcv):
        df = calc_all(synthetic_ohlcv)
        df = build_all(df)
        df = build_candle_patterns(df)
        df = build_ceo_structure(df)
        before_len = len(df)
        out = build_confluence(df)
        assert len(out) == before_len

    def test_build_confluence_respects_params_dict(self, synthetic_ohlcv_large):
        """build_confluence() is a thin wrapper over confluence_signals()
        reading the same param key names build_all() used to consume
        internally -- confirm those keys actually thread through."""
        df = calc_all(synthetic_ohlcv_large)
        df = build_all(df)
        df = build_candle_patterns(df)
        df = build_ceo_structure(df)

        lenient = build_confluence(df, params={"confluence_min_count": 1, "min_quality_score": 20.0})
        strict  = build_confluence(df, params={"confluence_min_count": 10, "min_quality_score": 95.0})
        assert lenient["confluence_long_fired"].sum() >= strict["confluence_long_fired"].sum()


# ─────────────────────────────────────────────────────────────────────────────
# Regression guard: calling it in the WRONG order reproduces the bug
# ─────────────────────────────────────────────────────────────────────────────

class TestWrongOrderingStillBroken:
    """Not testing that the bug is impossible to reintroduce -- it's a
    two-line ordering choice at each call site, so it's entirely possible
    to get this wrong again. This documents precisely what "wrong" looks
    like, so a future change that moves build_confluence() back before
    build_ceo_structure() has a test explaining why that's a regression,
    not just a silent behavior change."""

    def test_confluence_before_ceo_structure_reproduces_the_original_bug(self, synthetic_ohlcv_large):
        df = calc_all(synthetic_ohlcv_large)
        df = build_all(df)
        # Calling it here (the old, buggy position) instead of after
        # build_ceo_structure() -- m00_quality_long is still capped at 40.
        df = build_confluence(df)   # default min_quality=45.0
        assert not df["confluence_long_fired"].any()
        assert not df["confluence_short_fired"].any()


class TestConfluenceModeSelection:
    """Tests for the confluence_mode parameter: 'sweep' (default,
    N filter-models agree), 'ceo_structure' (the full CEO sequence
    validates instead of a model count), 'full' (both required)."""

    def test_invalid_mode_raises(self, synthetic_ohlcv):
        df = calc_all(synthetic_ohlcv)
        df = build_all(df)
        df = build_candle_patterns(df)
        df = build_ceo_structure(df)
        with pytest.raises(ValueError, match="confluence mode must be one of"):
            confluence_signals(df, mode="not_a_real_mode")

    def test_ceo_structure_mode_requires_ceo_columns(self, synthetic_ohlcv):
        """Calling with mode='ceo_structure' before build_ceo_structure()
        has run must raise a clear error, not silently return an
        always-False gate -- a silently dead signal is exactly the bug
        this mode system exists to not repeat."""
        df = calc_all(synthetic_ohlcv)
        df = build_all(df)   # no build_ceo_structure() call
        with pytest.raises(ValueError, match="ceo_long_valid"):
            confluence_signals(df, mode="ceo_structure")

    def test_full_mode_requires_ceo_columns(self, synthetic_ohlcv):
        df = calc_all(synthetic_ohlcv)
        df = build_all(df)
        with pytest.raises(ValueError, match="ceo_long_valid"):
            confluence_signals(df, mode="full")

    def test_sweep_mode_does_not_require_ceo_columns(self, synthetic_ohlcv):
        """The default mode must keep working even without
        build_ceo_structure() having run -- e.g. walkforward.py's
        per-window rebuild depends on exactly this."""
        df = calc_all(synthetic_ohlcv)
        df = build_all(df)
        out = confluence_signals(df, mode="sweep")   # must not raise
        assert "confluence_long_fired" in out.columns

    def test_ceo_structure_mode_gate_matches_ceo_long_valid(self, synthetic_ohlcv_large):
        """In 'ceo_structure' mode, firing should track ceo_long_valid
        directly (subject to the same quality/alignment/htf gates that
        every mode applies) -- not the sweep model count."""
        df = calc_all(synthetic_ohlcv_large)
        df = build_all(df)
        df = build_candle_patterns(df)
        df = build_ceo_structure(df)
        out = confluence_signals(df, min_quality=0, require_align=False, mode="ceo_structure")
        # With quality/alignment/htf gates neutralized, firing should be
        # an exact match for ceo_long_valid (modulo any htf_bullish gate,
        # which defaults to all-True when absent).
        assert (out["confluence_long_fired"] == out["ceo_long_valid"]).all()

    def test_full_mode_is_a_subset_of_both_individual_modes(self, synthetic_ohlcv_large):
        df = calc_all(synthetic_ohlcv_large)
        df = build_all(df)
        df = build_candle_patterns(df)
        df = build_ceo_structure(df)
        sweep_only = confluence_signals(df, mode="sweep")
        ceo_only   = confluence_signals(df, mode="ceo_structure")
        full       = confluence_signals(df, mode="full")
        # full firing must never fire where either individual mode didn't.
        assert (full["confluence_long_fired"] <= sweep_only["confluence_long_fired"]).all()
        assert (full["confluence_long_fired"] <= ceo_only["confluence_long_fired"]).all()

    def test_full_mode_fires_less_or_equal_often_than_either_alone(self, synthetic_ohlcv_large):
        df = calc_all(synthetic_ohlcv_large)
        df = build_all(df)
        df = build_candle_patterns(df)
        df = build_ceo_structure(df)
        sweep_count = confluence_signals(df, mode="sweep")["confluence_long_fired"].sum()
        ceo_count   = confluence_signals(df, mode="ceo_structure")["confluence_long_fired"].sum()
        full_count  = confluence_signals(df, mode="full")["confluence_long_fired"].sum()
        assert full_count <= sweep_count
        assert full_count <= ceo_count

    def test_build_confluence_reads_mode_from_params(self, synthetic_ohlcv):
        df = calc_all(synthetic_ohlcv)
        df = build_all(df)
        with pytest.raises(ValueError, match="ceo_long_valid"):
            # Confirms the params dict key actually reaches mode= --
            # build_ceo_structure() deliberately not called, so this
            # should raise the same way a direct mode="ceo_structure"
            # call would.
            build_confluence(df, params={"confluence_mode": "ceo_structure"})

    def test_build_confluence_defaults_to_sweep_mode(self, synthetic_ohlcv):
        df = calc_all(synthetic_ohlcv)
        df = build_all(df)
        out = build_confluence(df, params={})   # no confluence_mode key at all
        assert "confluence_long_fired" in out.columns   # sweep mode: no exception


class TestConfluenceModeCli:
    def test_confluence_mode_flag_parses(self):
        from ceo_engine_mt5.run import _build_parser
        p = _build_parser()
        args = p.parse_args(["--symbol", "XAUUSD", "--confluence-mode", "full"])
        assert args.confluence_mode == "full"

    def test_confluence_mode_defaults_to_sweep(self):
        from ceo_engine_mt5.run import _build_parser
        p = _build_parser()
        args = p.parse_args(["--symbol", "XAUUSD"])
        assert args.confluence_mode == "sweep"

    def test_invalid_confluence_mode_rejected_by_argparse(self):
        from ceo_engine_mt5.run import _build_parser
        p = _build_parser()
        with pytest.raises(SystemExit):
            p.parse_args(["--symbol", "XAUUSD", "--confluence-mode", "bogus"])


class TestWalkforwardConfluenceModeOverride:
    def test_non_sweep_mode_forced_back_to_sweep_and_warns(self, synthetic_ohlcv_large, caplog):
        """walkforward.py's per-window rebuild never runs
        build_ceo_structure(), so ceo_long_valid never exists there --
        requesting 'ceo_structure'/'full' globally must not crash walk-forward,
        it should fall back to 'sweep' for that path specifically, with a
        warning explaining why."""
        from ceo_engine_mt5.walkforward import walk_forward
        import logging
        with caplog.at_level(logging.WARNING):
            result = walk_forward(
                synthetic_ohlcv_large, n_windows=2, min_trades=1,
                signal_params={"confluence_mode": "ceo_structure"},
                verbose=False,
            )
        assert result is not None
        assert any("confluence_mode" in r.message for r in caplog.records)

    def test_sweep_mode_does_not_warn(self, synthetic_ohlcv_large, caplog):
        from ceo_engine_mt5.walkforward import walk_forward
        import logging
        with caplog.at_level(logging.WARNING):
            walk_forward(
                synthetic_ohlcv_large, n_windows=2, min_trades=1,
                signal_params={"confluence_mode": "sweep"},
                verbose=False,
            )
        assert not any("confluence_mode" in r.message for r in caplog.records)


class TestConfluenceGatedBehindFlag:
    """The --confluence CLI flag existed but was never actually read
    anywhere -- Confluence was unconditionally evaluated in check_symbol()
    regardless of it. Harmless while Confluence could never fire at all,
    but once fixed (this file's other tests), that became a real,
    unrequested default: every live/backtest run started firing on
    Confluence whether the user wanted it or not, with no way to opt out.
    Now gated behind params["confluence"], matching how ceo_only already
    works -- off unless explicitly requested."""

    def _fired_labels(self, monkeypatch, confluence_flag, gate_col="confluence_long_fired"):
        from ceo_engine_mt5 import mt5_live_signals as mls
        from ceo_engine_mt5.session_filter import add_session_columns as _real

        def _force(df, **kwargs):
            out = _real(df, **kwargs)
            out.loc[out.index[-1], gate_col] = True
            out.loc[out.index[-1], "m00_quality_long"] = 55.0
            return out
        monkeypatch.setattr(mls, "add_session_columns", _force)

        conn = _FakeConn(_base_ohlcv())
        params = {"sessions": ["all"], "no_htf": True}
        if confluence_flag is not None:
            params["confluence"] = confluence_flag
        fired = check_symbol(conn, "XAUUSD", "M15", **_cs_kwargs(params=params))
        return {s["model"] for s in fired}

    def test_confluence_absent_from_modes_by_default(self, monkeypatch):
        labels = self._fired_labels(monkeypatch, confluence_flag=None)
        assert not any("Confluence" in l for l in labels)

    def test_confluence_absent_when_flag_explicitly_false(self, monkeypatch):
        labels = self._fired_labels(monkeypatch, confluence_flag=False)
        assert not any("Confluence" in l for l in labels)

    def test_confluence_present_when_flag_true(self, monkeypatch):
        # Force ONLY the confluence gate (not m00_long too), so LQ's
        # dedup-by-bar_id can't mask whether Confluence mode is even in
        # play -- see test_mt5_live_signals_main_loop.py's
        # test_default_model_and_confluence_dedup_by_shared_bar_id for
        # why that matters.
        from ceo_engine_mt5 import mt5_live_signals as mls
        from ceo_engine_mt5.session_filter import add_session_columns as _real

        def _force(df, **kwargs):
            out = _real(df, **kwargs)
            out.loc[out.index[-1], "confluence_long_fired"] = True
            out.loc[out.index[-1], "m00_quality_long"] = 55.0
            out.loc[out.index[-1], "m00_long"] = False   # ensure LQ itself doesn't also fire
            return out
        monkeypatch.setattr(mls, "add_session_columns", _force)

        conn = _FakeConn(_base_ohlcv())
        fired = check_symbol(conn, "XAUUSD", "M15",
                             **_cs_kwargs(params={"sessions": ["all"], "no_htf": True,
                                                  "confluence": True}))
        labels = {s["model"] for s in fired}
        assert any("Confluence" in l for l in labels)


# ─────────────────────────────────────────────────────────────────────────────
# Live bar-close pipeline (mt5_live_signals.check_symbol) can fire Confluence
# ─────────────────────────────────────────────────────────────────────────────

class TestCheckSymbolCanFireConfluence:
    def test_confluence_mode_can_actually_fire_through_check_symbol(self, monkeypatch):
        """check_symbol() already had "Confluence" wired into its modes
        list before this fix (gate=confluence_long_fired, quality=
        m00_quality_long) -- it just never had real data to work with.
        Forces the gate directly (same technique as
        test_mt5_live_signals_main_loop.py) to confirm the full live
        pipeline now produces a usable value end-to-end."""
        from ceo_engine_mt5 import mt5_live_signals as mls
        from ceo_engine_mt5.session_filter import add_session_columns as _real

        def _force_confluence(df, **kwargs):
            out = _real(df, **kwargs)
            out.loc[out.index[-1], "confluence_long_fired"] = True
            out.loc[out.index[-1], "m00_quality_long"] = 55.0
            return out
        monkeypatch.setattr(mls, "add_session_columns", _force_confluence)

        class _FakeConn:
            def symbol_info(self, symbol):
                return {"digits": 2, "tick_size": 0.01}
            def fetch_rates(self, symbol, tf, n_bars=1000):
                df = _make_ohlcv(seed=11, n=400)
                return [{"time": int(pd.Timestamp(ts).timestamp()),
                         "open": float(r["open"]), "high": float(r["high"]),
                         "low": float(r["low"]), "close": float(r["close"]),
                         "tick_volume": float(r.get("volume", 100.0))}
                        for ts, r in df.iterrows()]

        fired = mls.check_symbol(
            _FakeConn(), "XAUUSD", "M15",
            params={"sessions": ["all"], "no_htf": True, "confluence": True},
            signal_params={}, bt_params={}, log_path=None, sound=False, seen_ids=set(),
        )
        labels = {s["model"] for s in fired}
        assert any("Confluence" in label for label in labels)
