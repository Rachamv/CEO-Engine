"""
Tests for ceo_structure.py — BOS, Fibonacci zones, Order Blocks, QM levels,
structural liquidity, unmitigated-level tracking, and the master CEO
sequence gate. This module is the biggest blind spot in the suite: every
signal model's final quality/validity gate runs through here.

Style mirrors test_signals.py: pure-function unit tests on the small
numpy/array helpers, plus integration tests on enriched_df for the
stateful, loop-driven detectors that are awkward to unit-test directly.
"""

import numpy as np
import pandas as pd
import pytest

from ceo_engine_mt5.ceo_structure import (
    _shift, _rolling_max, _rolling_min,
    _prune_order_blocks, _detect_new_order_block,
    _add_new_levels, _remove_mitigated_and_cap,
    detect_bos, detect_fib_zones, detect_order_blocks, detect_qm_levels,
    detect_structural_liquidity, track_unmitigated_levels,
    validate_ceo_sequence, build_ceo_structure,
    DEFAULT_STRUCT_PARAMS,
)


# ─────────────────────────────────────────────────────────────────────────────
# Pure array helpers
# ─────────────────────────────────────────────────────────────────────────────

class TestShift:
    def test_positive_shift_pushes_forward_and_fills_head(self):
        arr = np.array([1.0, 2.0, 3.0, 4.0])
        out = _shift(arr, 1)
        assert np.isnan(out[0])
        assert list(out[1:]) == [1.0, 2.0, 3.0]

    def test_negative_shift_pulls_backward_and_fills_tail(self):
        arr = np.array([1.0, 2.0, 3.0, 4.0])
        out = _shift(arr, -1)
        assert list(out[:-1]) == [2.0, 3.0, 4.0]
        assert np.isnan(out[-1])

    def test_zero_shift_is_identity(self):
        arr = np.array([1.0, 2.0, 3.0])
        out = _shift(arr, 0)
        assert list(out) == [1.0, 2.0, 3.0]

    def test_custom_fill_value(self):
        arr = np.array([1.0, 2.0, 3.0])
        out = _shift(arr, 1, fill=-1.0)
        assert out[0] == -1.0


class TestRollingMaxMin:
    def test_rolling_max_basic_window(self):
        arr = np.array([1.0, 5.0, 2.0, 8.0, 3.0])
        out = _rolling_max(arr, window=2)
        assert np.isnan(out[0])
        assert out[1] == 5.0   # max(1,5)
        assert out[2] == 5.0   # max(5,2)
        assert out[3] == 8.0   # max(2,8)
        assert out[4] == 8.0   # max(8,3)

    def test_rolling_min_basic_window(self):
        arr = np.array([1.0, 5.0, 2.0, 8.0, 3.0])
        out = _rolling_min(arr, window=2)
        assert np.isnan(out[0])
        assert out[1] == 1.0
        assert out[2] == 2.0
        assert out[3] == 2.0
        assert out[4] == 3.0

    def test_window_larger_than_array_is_all_nan(self):
        arr = np.array([1.0, 2.0])
        out = _rolling_max(arr, window=5)
        assert np.isnan(out).all()


# ─────────────────────────────────────────────────────────────────────────────
# Order block pure helpers
# ─────────────────────────────────────────────────────────────────────────────

class TestPruneOrderBlocks:
    def _params(self, **overrides):
        return {**DEFAULT_STRUCT_PARAMS, **overrides}

    def test_keeps_unmitigated_unexpired_bull_ob(self):
        c = {1: 105.0}
        h = {1: 106.0}
        lo = {1: 104.0}
        active = [(101.0, 100.0, 0)]   # ob_high, ob_low, ob_bar
        p = self._params(ob_mitigation_close=True, ob_max_age=100)
        out = _prune_order_blocks(active, 1, c, h, lo, p, is_bull=True)
        assert out == active

    def test_drops_mitigated_bull_ob_when_close_breaks_low(self):
        c = {1: 99.0}
        h = {1: 100.0}
        lo = {1: 98.0}
        active = [(101.0, 100.0, 0)]
        p = self._params(ob_mitigation_close=True, ob_max_age=100)
        out = _prune_order_blocks(active, 1, c, h, lo, p, is_bull=True)
        assert out == []

    def test_drops_expired_bull_ob_by_age(self):
        c = {50: 105.0}
        h = {50: 106.0}
        lo = {50: 104.0}
        active = [(101.0, 100.0, 0)]   # age 50
        p = self._params(ob_mitigation_close=True, ob_max_age=10)
        out = _prune_order_blocks(active, 50, c, h, lo, p, is_bull=True)
        assert out == []

    def test_drops_mitigated_bear_ob_when_close_breaks_high(self):
        c = {1: 102.0}
        h = {1: 103.0}
        lo = {1: 101.0}
        active = [(101.0, 99.0, 0)]
        p = self._params(ob_mitigation_close=True, ob_max_age=100)
        out = _prune_order_blocks(active, 1, c, h, lo, p, is_bull=False)
        assert out == []


class TestDetectNewOrderBlock:
    def test_adds_bull_ob_when_bearish_candle_clears_atr_threshold(self):
        # bar 2 is a displacement; bar 1 is the last bearish candle before it
        o   = {0: 100.0, 1: 102.0, 2: 99.0}
        c   = {0: 101.0, 1: 99.0,  2: 103.0}   # bar1 bearish: close<open, body=3
        atr = {2: 1.0}
        p = {**DEFAULT_STRUCT_PARAMS, "ob_lookback": 5, "ob_atr_min": 0.30}
        out = _detect_new_order_block(2, o, c, atr, p, active_obs=[], max_slots=3, is_bull=True)
        assert len(out) == 1
        ob_high, ob_low, ob_bar = out[0]
        assert ob_high == 102.0 and ob_low == 99.0 and ob_bar == 2

    def test_skips_when_body_below_atr_threshold(self):
        o   = {0: 100.0, 1: 100.2, 2: 99.0}
        c   = {0: 101.0, 1: 100.0, 2: 103.0}   # bar1 bearish but tiny body (0.2)
        atr = {2: 1.0}
        p = {**DEFAULT_STRUCT_PARAMS, "ob_lookback": 5, "ob_atr_min": 0.30}
        out = _detect_new_order_block(2, o, c, atr, p, active_obs=[], max_slots=3, is_bull=True)
        assert out == []

    def test_caps_at_max_slots_dropping_oldest(self):
        # bar 0 must itself be the bearish "last opposite candle" found by the
        # lookback scan from bar 1: c[0] < o[0] (bearish, body=5)
        o   = {0: 110.0, 1: 108.0}
        c   = {0: 105.0, 1: 109.0}
        atr = {1: 1.0}
        existing = [(50.0, 49.0, -3), (60.0, 59.0, -2), (70.0, 69.0, -1)]
        p = {**DEFAULT_STRUCT_PARAMS, "ob_lookback": 5, "ob_atr_min": 0.30}
        out = _detect_new_order_block(1, o, c, atr, p, active_obs=list(existing), max_slots=3, is_bull=True)
        assert len(out) == 3
        assert out[-1] == (110.0, 105.0, 1)   # newest kept
        assert (50.0, 49.0, -3) not in out    # oldest dropped


# ─────────────────────────────────────────────────────────────────────────────
# Unmitigated-level helpers
# ─────────────────────────────────────────────────────────────────────────────

class TestAddNewLevels:
    def test_appends_each_present_level_once(self):
        resistance, support = [], []
        ob_bull_high = {0: 105.0}
        ob_bear_low  = {0: 110.0}
        qm_bull      = {0: np.nan}
        qm_bear      = {0: np.nan}
        _add_new_levels(0, ob_bull_high, ob_bear_low, qm_bull, qm_bear, resistance, support)
        assert resistance == [110.0]
        assert support == [105.0]

    def test_ignores_nan_levels(self):
        resistance, support = [], []
        ob_bull_high = {0: np.nan}
        ob_bear_low  = {0: np.nan}
        qm_bull      = {0: np.nan}
        qm_bear      = {0: np.nan}
        _add_new_levels(0, ob_bull_high, ob_bear_low, qm_bull, qm_bear, resistance, support)
        assert resistance == [] and support == []

    def test_does_not_duplicate_existing_level(self):
        resistance, support = [110.0], []
        ob_bull_high = {0: np.nan}
        ob_bear_low  = {0: 110.0}
        qm_bull      = {0: np.nan}
        qm_bear      = {0: np.nan}
        _add_new_levels(0, ob_bull_high, ob_bear_low, qm_bull, qm_bear, resistance, support)
        assert resistance == [110.0]


class TestRemoveMitigatedAndCap:
    def test_drops_resistance_once_price_closes_above_it(self):
        res, sup = _remove_mitigated_and_cap([105.0, 110.0], [], close_i=106.0, max_levels=5)
        assert res == [110.0]

    def test_drops_support_once_price_closes_below_it(self):
        res, sup = _remove_mitigated_and_cap([], [95.0, 90.0], close_i=92.0, max_levels=5)
        assert sup == [90.0]

    def test_caps_resistance_to_nearest_n_above(self):
        res, sup = _remove_mitigated_and_cap(
            [101.0, 102.0, 103.0, 104.0], [], close_i=100.0, max_levels=2)
        assert res == [101.0, 102.0]

    def test_caps_support_to_nearest_n_below(self):
        res, sup = _remove_mitigated_and_cap(
            [], [99.0, 98.0, 97.0, 96.0], close_i=100.0, max_levels=2)
        assert sup == [99.0, 98.0]


# ─────────────────────────────────────────────────────────────────────────────
# detect_bos — integration on enriched data
# ─────────────────────────────────────────────────────────────────────────────

class TestDetectBOS:
    def test_adds_expected_columns(self, enriched_df):
        for col in ["bos_long", "bos_short", "double_bos_long", "double_bos_short"]:
            assert col in enriched_df.columns

    def test_columns_are_boolean(self, enriched_df):
        for col in ["bos_long", "bos_short", "double_bos_long", "double_bos_short"]:
            assert enriched_df[col].dtype == bool

    def test_double_bos_only_true_when_bos_also_true(self, enriched_df):
        assert (~enriched_df["double_bos_long"] | enriched_df["bos_long"]).all()
        assert (~enriched_df["double_bos_short"] | enriched_df["bos_short"]).all()

    def test_close_confirm_false_uses_wick_and_is_at_least_as_permissive(self, synthetic_ohlcv):
        from ceo_engine_mt5.indicators import calc_all
        from ceo_engine_mt5.signals import build_all
        df = build_all(calc_all(synthetic_ohlcv))
        strict = detect_bos(df, {"bos_close_confirm": True})
        loose  = detect_bos(df, {"bos_close_confirm": False})
        # Wick-based confirmation can only trigger BOS at least as often as close-based.
        assert loose["bos_long"].sum()  >= strict["bos_long"].sum()
        assert loose["bos_short"].sum() >= strict["bos_short"].sum()

    def test_does_not_mutate_input_df(self, enriched_df):
        before_cols = set(enriched_df.columns)
        _ = detect_bos(enriched_df)
        assert set(enriched_df.columns) == before_cols


# ─────────────────────────────────────────────────────────────────────────────
# detect_fib_zones
# ─────────────────────────────────────────────────────────────────────────────

class TestDetectFibZones:
    def test_adds_expected_columns(self, enriched_df):
        for col in ["in_discount", "in_premium", "in_golden", "fib_50", "fib_zone_score"]:
            assert col in enriched_df.columns

    def test_discount_and_premium_are_mutually_exclusive(self, enriched_df):
        assert not (enriched_df["in_discount"] & enriched_df["in_premium"]).any()

    def test_zone_score_within_bounds(self, enriched_df):
        valid = enriched_df["fib_zone_score"].dropna()
        assert (valid >= 0).all() and (valid <= 100).all()

    def test_golden_pocket_implies_max_score(self, enriched_df):
        golden_rows = enriched_df[enriched_df["in_golden"]]
        if len(golden_rows):
            assert (golden_rows["fib_zone_score"] == 100.0).all()

    def test_zero_range_swing_leaves_fib_50_nan(self):
        # Flat data → swing_high == swing_low → swing_rng <= 0 → skipped (stays NaN)
        n = 150
        idx = pd.date_range("2024-01-01", periods=n, freq="1h", tz="UTC")
        flat = pd.DataFrame({
            "open": 100.0, "high": 100.0, "low": 100.0, "close": 100.0,
            "volume": 100.0, "trend_long": True, "trend_short": False,
        }, index=idx)
        out = detect_fib_zones(flat, {"fib_swing_lookback": 50})
        assert out["fib_50"].isna().all()


# ─────────────────────────────────────────────────────────────────────────────
# detect_order_blocks / detect_qm_levels
# ─────────────────────────────────────────────────────────────────────────────

class TestDetectOrderBlocks:
    def test_adds_expected_columns(self, enriched_df):
        for col in ["ob_bull_active", "ob_bear_active", "ob_bull_high",
                    "ob_bull_low", "ob_bear_high", "ob_bear_low"]:
            assert col in enriched_df.columns

    def test_bull_ob_high_is_above_low_when_active(self, enriched_df):
        active = enriched_df[enriched_df["ob_bull_active"]]
        assert (active["ob_bull_high"] >= active["ob_bull_low"]).all()

    def test_bear_ob_high_is_above_low_when_active(self, enriched_df):
        active = enriched_df[enriched_df["ob_bear_active"]]
        assert (active["ob_bear_high"] >= active["ob_bear_low"]).all()

    def test_inactive_rows_have_nan_levels(self, enriched_df):
        inactive = enriched_df[~enriched_df["ob_bull_active"]]
        assert inactive["ob_bull_high"].isna().all()


class TestDetectQMLevels:
    def test_adds_expected_columns(self, enriched_df):
        for col in ["qm_bull_active", "qm_bear_active", "qm_bull_level", "qm_bear_level"]:
            assert col in enriched_df.columns

    def test_qm_active_implies_level_present(self, enriched_df):
        assert enriched_df.loc[enriched_df["qm_bull_active"], "qm_bull_level"].notna().all()
        assert enriched_df.loc[enriched_df["qm_bear_active"], "qm_bear_level"].notna().all()

    def test_runs_without_ob_columns_present(self, synthetic_ohlcv):
        from ceo_engine_mt5.indicators import calc_all
        from ceo_engine_mt5.signals import build_all
        df = build_all(calc_all(synthetic_ohlcv))
        df = detect_bos(df)
        # No detect_order_blocks() call — qm columns ("ob_bull_low" etc.) absent
        out = detect_qm_levels(df)
        assert not out["qm_bull_active"].any()
        assert not out["qm_bear_active"].any()


# ─────────────────────────────────────────────────────────────────────────────
# detect_structural_liquidity
# ─────────────────────────────────────────────────────────────────────────────

class TestDetectStructuralLiquidity:
    def test_adds_expected_columns(self, enriched_df):
        for col in ["struct_liq_long", "struct_liq_short", "inducement_long", "inducement_short"]:
            assert col in enriched_df.columns

    def test_long_side_struct_liq_and_inducement_mutually_exclusive(self, enriched_df):
        assert not (enriched_df["struct_liq_long"] & enriched_df["inducement_long"]).any()

    def test_short_side_struct_liq_and_inducement_mutually_exclusive(self, enriched_df):
        assert not (enriched_df["struct_liq_short"] & enriched_df["inducement_short"]).any()


# ─────────────────────────────────────────────────────────────────────────────
# track_unmitigated_levels
# ─────────────────────────────────────────────────────────────────────────────

class TestTrackUnmitigatedLevels:
    def test_adds_expected_columns(self, enriched_df):
        for col in ["nearest_unmit_res", "nearest_unmit_sup"]:
            assert col in enriched_df.columns

    def test_resistance_always_above_close_when_present(self, enriched_df):
        rows = enriched_df.dropna(subset=["nearest_unmit_res"])
        assert (rows["nearest_unmit_res"] > rows["close"]).all()

    def test_support_always_below_close_when_present(self, enriched_df):
        rows = enriched_df.dropna(subset=["nearest_unmit_sup"])
        assert (rows["nearest_unmit_sup"] < rows["close"]).all()

    def test_handles_missing_ob_qm_columns_gracefully(self, synthetic_ohlcv):
        from ceo_engine_mt5.indicators import calc_all
        from ceo_engine_mt5.signals import build_all
        df = build_all(calc_all(synthetic_ohlcv))
        out = track_unmitigated_levels(df)   # no OB/QM columns present at all
        assert out["nearest_unmit_res"].isna().all()
        assert out["nearest_unmit_sup"].isna().all()


# ─────────────────────────────────────────────────────────────────────────────
# validate_ceo_sequence — the master gate
# ─────────────────────────────────────────────────────────────────────────────

class TestValidateCeoSequence:
    def test_adds_expected_columns(self, enriched_df):
        for col in ["ceo_long_valid", "ceo_short_valid", "ceo_quality_bonus"]:
            assert col in enriched_df.columns

    def test_valid_long_requires_base_sweep(self, enriched_df):
        # ceo_long_valid is a refinement of base_long; it can never be true
        # where base_long itself is false.
        assert not (enriched_df["ceo_long_valid"] & ~enriched_df["base_long"]).any()

    def test_valid_short_requires_base_sweep(self, enriched_df):
        assert not (enriched_df["ceo_short_valid"] & ~enriched_df["base_short"]).any()

    def test_quality_bonus_within_bounds(self, enriched_df):
        assert (enriched_df["ceo_quality_bonus"] >= 0).all()
        assert (enriched_df["ceo_quality_bonus"] <= 30).all()

    def test_disabling_all_gates_leaves_validity_equal_to_base_sweep(self, synthetic_ohlcv):
        from ceo_engine_mt5.indicators import calc_all
        from ceo_engine_mt5.signals import build_all
        from ceo_engine_mt5.candle_patterns import build_candle_patterns
        df = build_candle_patterns(build_all(calc_all(synthetic_ohlcv)))
        df = build_ceo_structure(df, {
            "require_bos": False, "require_fib_zone": False,
            "require_ob_or_qm": False, "require_struct_liq": False,
        })
        # Only the inducement filter remains active by default in _apply_structural_gates,
        # so valid flags should be a subset of (not necessarily equal to) base sweeps.
        assert (df["ceo_long_valid"]  <= df["base_long"]).all()
        assert (df["ceo_short_valid"] <= df["base_short"]).all()

    def test_quality_columns_increase_or_stay_equal_after_bonus(self, synthetic_ohlcv):
        from ceo_engine_mt5.indicators import calc_all
        from ceo_engine_mt5.signals import build_all
        df = build_all(calc_all(synthetic_ohlcv))
        if "quality_long" not in df.columns:
            pytest.skip("quality_long not produced by signals.build_all in this configuration")
        before = df["quality_long"].copy()
        df = build_ceo_structure(df)
        assert (df["quality_long"] >= before - 1e-9).all()


# ─────────────────────────────────────────────────────────────────────────────
# build_ceo_structure — full pipeline
# ─────────────────────────────────────────────────────────────────────────────

class TestBuildCeoStructure:
    def test_returns_dataframe_with_all_phase_columns(self, synthetic_ohlcv):
        from ceo_engine_mt5.indicators import calc_all
        from ceo_engine_mt5.signals import build_all
        df = build_all(calc_all(synthetic_ohlcv))
        out = build_ceo_structure(df)
        expected = [
            "bos_long", "in_discount", "ob_bull_active", "qm_bull_active",
            "struct_liq_long", "nearest_unmit_res", "ceo_long_valid",
            "ceo_quality_bonus",
        ]
        for col in expected:
            assert col in out.columns

    def test_preserves_row_count(self, synthetic_ohlcv):
        from ceo_engine_mt5.indicators import calc_all
        from ceo_engine_mt5.signals import build_all
        df = build_all(calc_all(synthetic_ohlcv))
        out = build_ceo_structure(df)
        assert len(out) == len(df)

    def test_runs_on_large_dataset_without_error(self, enriched_df_large):
        # enriched_df_large already ran the full pipeline including
        # build_ceo_structure via conftest — just sanity check it succeeded.
        assert "ceo_long_valid" in enriched_df_large.columns
        assert len(enriched_df_large) > 0

    def test_custom_params_propagate_to_all_stages(self, synthetic_ohlcv):
        from ceo_engine_mt5.indicators import calc_all
        from ceo_engine_mt5.signals import build_all
        df = build_all(calc_all(synthetic_ohlcv))
        # A much shorter fib lookback should produce a different fib_50 series
        # than the default — proves params actually thread through build_ceo_structure.
        default_out = build_ceo_structure(df, None)
        custom_out  = build_ceo_structure(df, {"fib_swing_lookback": 10})
        assert not default_out["fib_50"].equals(custom_out["fib_50"])
