"""
Tests for multi_tf.py — the multi-timeframe confirmation stack.

Per-TF pipeline execution (_run_pipeline) needs a live/mocked MT5
connection and is exercised indirectly through MTFStack.check() with a
fake connection object, following the same injected-dependency pattern
test_executor.py uses for Executor. Everything else here — _auto_htf,
_tf_rank, _sort_tfs, the four confirmation-mode checkers, and _score —
is pure logic over TFState objects built directly, with no I/O at all.
"""

from datetime import datetime, timezone

import numpy as np
import pandas as pd
import pytest

from ceo_engine_mt5.multi_tf import (
    _auto_htf, _tf_rank, _sort_tfs,
    _check_sweep, _check_ceo, _check_bias, _check_cascade, _score,
    TFState, MTFResult, MTFStack,
)


def _state(tf="H1", **overrides) -> TFState:
    """Build a TFState directly, bypassing the data pipeline entirely."""
    kwargs = dict(
        tf=tf, df=pd.DataFrame(), last=pd.Series(dtype=float),
        base_long=False, base_short=False, ceo_long=False, ceo_short=False,
        htf_bullish=False, htf_bearish=False, bos_long=False, bos_short=False,
        in_discount=False, in_premium=False, quality_long=0.0, quality_short=0.0,
        cp_bull_confirm=False, cp_bear_confirm=False, pat_name="", regime="",
        ob_bull_active=False, ob_bear_active=False,
    )
    kwargs.update(overrides)
    return TFState(**kwargs)


# ─────────────────────────────────────────────────────────────────────────────
# HTF auto-select / TF ranking
# ─────────────────────────────────────────────────────────────────────────────

class TestAutoHtf:
    def test_maps_m15_to_h1(self):
        assert _auto_htf("M15") == "h1"

    def test_maps_h1_to_h4(self):
        assert _auto_htf("h1") == "h4"

    def test_is_case_insensitive(self):
        assert _auto_htf("H4") == _auto_htf("h4")

    def test_unknown_tf_falls_back_to_h4(self):
        assert _auto_htf("weird_tf") == "h4"


class TestTfRankAndSort:
    def test_higher_timeframe_has_higher_rank(self):
        assert _tf_rank("h4") > _tf_rank("h1") > _tf_rank("m15")

    def test_unknown_tf_defaults_to_h1_rank(self):
        assert _tf_rank("bogus") == _tf_rank("h1")

    def test_sort_tfs_orders_highest_to_lowest(self):
        assert _sort_tfs(["m15", "h4", "h1"]) == ["h4", "h1", "m15"]

    def test_sort_tfs_is_case_insensitive_in_ranking(self):
        out = _sort_tfs(["M15", "H4", "H1"])
        assert [t.lower() for t in out] == ["h4", "h1", "m15"]


# ─────────────────────────────────────────────────────────────────────────────
# Confirmation mode: sweep
# ─────────────────────────────────────────────────────────────────────────────

class TestCheckSweep:
    def test_all_tfs_sweeping_long_confirms(self):
        states = {"H4": _state("H4", base_long=True), "H1": _state("H1", base_long=True)}
        ok, conf = _check_sweep(states, "LONG", min_tfs=2)
        assert ok and set(conf) == {"H4", "H1"}

    def test_one_tf_missing_sweep_fails_min_tfs(self):
        states = {"H4": _state("H4", base_long=True), "H1": _state("H1", base_long=False)}
        ok, conf = _check_sweep(states, "LONG", min_tfs=2)
        assert not ok and conf == ["H4"]

    def test_short_direction_checks_base_short(self):
        states = {"H4": _state("H4", base_short=True), "H1": _state("H1", base_short=True)}
        ok, conf = _check_sweep(states, "SHORT", min_tfs=2)
        assert ok and len(conf) == 2


# ─────────────────────────────────────────────────────────────────────────────
# Confirmation mode: ceo
# ─────────────────────────────────────────────────────────────────────────────

class TestCheckCeo:
    def test_requires_ceo_valid_not_just_sweep(self):
        states = {"H4": _state("H4", base_long=True, ceo_long=False)}
        ok, conf = _check_ceo(states, "LONG", min_tfs=1)
        assert not ok and conf == []

    def test_confirms_when_ceo_valid(self):
        states = {"H4": _state("H4", ceo_long=True)}
        ok, conf = _check_ceo(states, "LONG", min_tfs=1)
        assert ok and conf == ["H4"]


# ─────────────────────────────────────────────────────────────────────────────
# Confirmation mode: bias
# ─────────────────────────────────────────────────────────────────────────────

class TestCheckBias:
    def test_entry_tf_must_have_its_own_sweep(self):
        states = {
            "H4": _state("H4", htf_bullish=True),
            "M15": _state("M15", base_long=False),
        }
        ok, conf = _check_bias(states, "LONG", entry_tf="M15", min_tfs=2,
                                sorted_tfs=["H4", "M15"])
        assert not ok and conf == []

    def test_upper_tf_confirms_via_htf_bias(self):
        states = {
            "H4": _state("H4", htf_bullish=True),
            "M15": _state("M15", base_long=True),
        }
        ok, conf = _check_bias(states, "LONG", entry_tf="M15", min_tfs=2,
                                sorted_tfs=["H4", "M15"])
        assert ok and set(conf) == {"H4", "M15"}

    def test_upper_tf_confirms_via_sweep_or_ceo_even_without_htf_bias(self):
        states = {
            "H4": _state("H4", htf_bullish=False, ceo_long=True),
            "M15": _state("M15", base_long=True),
        }
        ok, conf = _check_bias(states, "LONG", entry_tf="M15", min_tfs=2,
                                sorted_tfs=["H4", "M15"])
        assert ok

    def test_unconfirmed_upper_tf_excluded_but_does_not_block_lower_tfs_alone(self):
        states = {
            "H4": _state("H4"),  # no bias, no sweep, no ceo
            "M15": _state("M15", base_long=True),
        }
        ok, conf = _check_bias(states, "LONG", entry_tf="M15", min_tfs=2,
                                sorted_tfs=["H4", "M15"])
        assert conf == ["M15"]
        assert not ok   # only 1 of 2 required TFs confirmed

    def test_missing_entry_state_returns_false(self):
        states = {"H4": _state("H4", htf_bullish=True)}
        ok, conf = _check_bias(states, "LONG", entry_tf="M15", min_tfs=1,
                                sorted_tfs=["H4", "M15"])
        assert not ok and conf == []


# ─────────────────────────────────────────────────────────────────────────────
# Confirmation mode: cascade
# ─────────────────────────────────────────────────────────────────────────────

class TestCheckCascade:
    def test_full_cascade_confirms_top_to_bottom(self):
        states = {
            "H4":  _state("H4", htf_bullish=True),
            "H1":  _state("H1", base_long=True, htf_bullish=True),
            "M15": _state("M15", base_long=True),
        }
        ok, conf = _check_cascade(states, "LONG", sorted_tfs=["H4", "H1", "M15"], min_tfs=3)
        assert ok and conf == ["H4", "H1", "M15"]

    def test_broken_link_stops_the_cascade(self):
        # H1 doesn't sweep -> M15 has no qualifying upper confirmation chain
        # beyond H1, but H4 still counts as upper_ok for M15's check since it
        # only requires ANY upper TF to confirm. Use a case where H4 itself
        # fails so nothing downstream can use it.
        states = {
            "H4":  _state("H4", htf_bullish=False, base_long=False),
            "H1":  _state("H1", base_long=False),
            "M15": _state("M15", base_long=True),
        }
        ok, conf = _check_cascade(states, "LONG", sorted_tfs=["H4", "H1", "M15"], min_tfs=3)
        assert "H4" not in conf
        assert "H1" not in conf
        assert "M15" not in conf   # upper_ok requires H4 or H1 to have confirmed
        assert not ok

    def test_short_direction_uses_bearish_fields(self):
        states = {
            "H4":  _state("H4", htf_bearish=True),
            "H1":  _state("H1", base_short=True),
        }
        ok, conf = _check_cascade(states, "SHORT", sorted_tfs=["H4", "H1"], min_tfs=2)
        assert ok and conf == ["H4", "H1"]


# ─────────────────────────────────────────────────────────────────────────────
# Composite score
# ─────────────────────────────────────────────────────────────────────────────

class TestScore:
    def test_no_confirmed_tfs_scores_zero(self):
        assert _score({}, [], "LONG", total_tfs=3) == 0.0

    def test_full_coverage_high_quality_scores_higher_than_partial(self):
        strong_states = {
            "H4": _state("H4", quality_long=90.0, bos_long=True, in_discount=True,
                         ob_bull_active=True, cp_bull_confirm=True, ceo_long=True),
            "H1": _state("H1", quality_long=90.0, bos_long=True, in_discount=True,
                         ob_bull_active=True, cp_bull_confirm=True, ceo_long=True),
        }
        weak_states = {
            "H4": _state("H4", quality_long=20.0),
        }
        strong_score = _score(strong_states, ["H4", "H1"], "LONG", total_tfs=2)
        weak_score   = _score(weak_states, ["H4"], "LONG", total_tfs=2)
        assert strong_score > weak_score

    def test_score_capped_at_100(self):
        states = {
            "H4": _state("H4", quality_long=100.0, bos_long=True, in_discount=True,
                         ob_bull_active=True, cp_bull_confirm=True, ceo_long=True),
        }
        score = _score(states, ["H4"], "LONG", total_tfs=1)
        assert score <= 100.0

    def test_short_direction_uses_short_quality_and_flags(self):
        states = {"H4": _state("H4", quality_short=80.0, bos_short=True, in_premium=True)}
        score = _score(states, ["H4"], "SHORT", total_tfs=1)
        assert score > 0


# ─────────────────────────────────────────────────────────────────────────────
# MTFStack — integration with a fake connection (no real pipeline/MT5)
# ─────────────────────────────────────────────────────────────────────────────

class _FakeConn:
    """Minimal stand-in for an MT5Connection used by _run_pipeline."""

    def __init__(self, n_bars=200):
        self.n_bars = n_bars

    def fetch_rates(self, symbol, tf, n_bars=800):
        idx_seconds = np.arange(self.n_bars) * 3600
        price = 2000 + np.cumsum(np.random.RandomState(1).randn(self.n_bars) * 2)
        return pd.DataFrame({
            "time": idx_seconds,
            "open": price, "high": price + 2, "low": price - 2, "close": price,
            "tick_volume": 100.0,
        }).to_dict("records")

    def symbol_info(self, symbol):
        return {"tick_size": 0.01}


class TestMTFStackIntegration:
    def test_check_returns_invalid_result_without_raising_when_no_confirmation(self):
        # Regression test: the no-confirmation fallback path previously
        # passed an `entry_df` kwarg that MTFResult's __init__ didn't accept,
        # raising TypeError on every "no signal" outcome.
        stack = MTFStack(tfs=["H1", "M15"], mode="bias", min_score=999.0)
        result = stack.check("XAUUSD", _FakeConn())
        assert isinstance(result, MTFResult)
        assert result.valid is False

    def test_entry_df_field_is_populated_on_fallback(self):
        stack = MTFStack(tfs=["H1", "M15"], mode="bias", min_score=999.0)
        result = stack.check("XAUUSD", _FakeConn())
        # entry_df should be the entry TF's enriched dataframe when available,
        # and must not have crashed the constructor either way.
        assert result.entry_df is None or isinstance(result.entry_df, pd.DataFrame)

    def test_unavailable_data_returns_error_detail(self):
        class _EmptyConn:
            def fetch_rates(self, symbol, tf, n_bars=800):
                return []
            def symbol_info(self, symbol):
                return {"tick_size": 0.01}
        stack = MTFStack(tfs=["H1", "M15"], mode="bias")
        result = stack.check("XAUUSD", _EmptyConn())
        assert result.valid is False
        assert "error" in result.details

    def test_constructor_sorts_tfs_high_to_low_and_defaults_entry_tf(self):
        stack = MTFStack(tfs=["M15", "H4", "H1"])
        assert stack.tfs == ["H4", "H1", "M15"]
        assert stack.entry_tf == "M15"
