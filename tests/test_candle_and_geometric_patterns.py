"""
Tests for candle_patterns.py (30+ candlestick patterns) and patterns.py
(geometric chart patterns: H&S, double/triple tops, wedges, triangles,
flags, pennants, rectangles). Both feed directly into the signal quality
layer and are pure DataFrame-in/out, so they reuse the same
`synthetic_ohlcv` / `enriched_df` fixtures as the rest of the suite.

Exhaustively re-deriving every one of the 30+ candle shapes by hand isn't
practical here, so this suite combines:
  - unit tests on the small pure helpers in both modules,
  - hand-crafted OHLC cases for the simplest/most load-bearing patterns
    (doji, hammer, bullish/bearish engulfing) to catch shape-logic
    regressions directly, and
  - integration/invariant tests over enriched_df for everything else
    (columns present, correct dtypes, composite flags consistent with
    their components, quality bonuses stay within bounds, pipeline
    runs end-to-end without raising).
"""

import numpy as np
import pandas as pd
import pytest

from ceo_engine_mt5.candle_patterns import (
    _is_bull, _is_bear, _body_top, _body_bot, _shift, _local_trend,
    detect_candle_patterns, pattern_summary, apply_pattern_quality_bonus,
    build_candle_patterns, DEFAULT_CP_PARAMS,
)
from ceo_engine_mt5.patterns import (
    _build_pivot_index, _window_pivots, _atr_at, _near, _slope,
    detect_patterns, build_patterns, DEFAULT_PAT_PARAMS,
)


# ─────────────────────────────────────────────────────────────────────────────
# candle_patterns.py — pure helpers
# ─────────────────────────────────────────────────────────────────────────────

class TestCandleHelpers:
    def test_is_bull_and_is_bear_are_mutually_exclusive_on_nonequal_oc(self):
        o = np.array([1.0, 2.0, 3.0])
        c = np.array([2.0, 1.0, 3.0])   # up, down, flat
        bull = _is_bull(o, c)
        bear = _is_bear(o, c)
        assert list(bull) == [True, False, False]
        assert list(bear) == [False, True, False]

    def test_body_top_and_bot(self):
        o = np.array([1.0, 5.0])
        c = np.array([3.0, 2.0])
        assert list(_body_top(o, c)) == [3.0, 5.0]
        assert list(_body_bot(o, c)) == [1.0, 2.0]

    def test_shift_forward_fills_head_with_provided_value(self):
        arr = np.array([10.0, 20.0, 30.0])
        out = _shift(arr, 1, fill=arr[0])
        assert out[0] == 10.0
        assert list(out[1:]) == [10.0, 20.0]

    def test_local_trend_up_down_neutral(self):
        close = np.array([100.0, 100.0, 100.0, 105.0, 95.0, 100.0])
        trend = _local_trend(close, lookback=3)
        # bar 3: close=105 > close[0]=100 -> uptrend
        assert trend[3] == 1
        # bar 4: close=95 < close[1]=100 -> downtrend
        assert trend[4] == -1
        # bar 5: close=100 == close[2]=100 -> neutral
        assert trend[5] == 0


# ─────────────────────────────────────────────────────────────────────────────
# candle_patterns.py — hand-crafted canonical shapes
# ─────────────────────────────────────────────────────────────────────────────

def _make_indicator_ready_df(rows):
    """Build a minimal DataFrame with the columns detect_candle_patterns needs,
    computing body_ratio/upper_rejection/lower_rejection/body_size/candle_range
    the same way indicators.calc_all() does."""
    df = pd.DataFrame(rows)
    rng = (df["high"] - df["low"]).replace(0, np.nan)
    df["body_size"] = (df["close"] - df["open"]).abs()
    df["candle_range"] = df["high"] - df["low"]
    df["body_ratio"] = (df["body_size"] / rng).fillna(0)
    df["upper_rejection"] = ((df["high"] - df[["open", "close"]].max(axis=1)) / rng).fillna(0)
    df["lower_rejection"] = ((df[["open", "close"]].min(axis=1) - df["low"]) / rng).fillna(0)
    df["atr"] = 1.0
    return df


class TestDojiShape:
    def test_open_equals_close_with_lopsided_wicks_is_plain_doji(self):
        # With open==close, upper_rejection + lower_rejection always sum to 1,
        # so a "plain" (non-dragonfly/gravestone/long-legged) doji needs one
        # wick under 0.3 and the other under 0.6 -> e.g. 0.15 / 0.85.
        rows = [{"open": 100.0, "high": 101.5, "low": 91.5, "close": 100.0}] * 3
        df = _make_indicator_ready_df(rows)
        out = detect_candle_patterns(df)
        assert out["cp_doji"].iloc[-1] == True  # noqa: E712
        assert out["cp_long_legged_doji"].iloc[-1] == False  # noqa: E712
        assert out["cp_dragonfly_doji"].iloc[-1] == False  # noqa: E712

    def test_open_equals_close_with_balanced_wicks_is_long_legged_doji(self):
        rows = [{"open": 100.0, "high": 101.0, "low": 99.0, "close": 100.0}] * 3
        df = _make_indicator_ready_df(rows)
        out = detect_candle_patterns(df)
        assert out["cp_long_legged_doji"].iloc[-1] == True  # noqa: E712
        assert out["cp_doji"].iloc[-1] == False  # noqa: E712

    def test_large_body_is_not_doji(self):
        rows = [{"open": 100.0, "high": 110.0, "low": 99.0, "close": 109.0}] * 3
        df = _make_indicator_ready_df(rows)
        out = detect_candle_patterns(df)
        assert out["cp_doji"].iloc[-1] == False  # noqa: E712


class TestHammerShape:
    def test_long_lower_wick_small_body_in_downtrend_is_hammer(self):
        # Downtrend context: close well below close 10 bars ago.
        rows = [{"open": 120.0, "high": 121.0, "low": 110.0, "close": 119.0}] * 11
        rows += [{"open": 100.2, "high": 100.5, "low": 90.0, "close": 100.0}]
        df = _make_indicator_ready_df(rows)
        out = detect_candle_patterns(df, {"trend_lookback": 10})
        assert out["cp_hammer"].iloc[-1] == True  # noqa: E712

    def test_same_shape_in_uptrend_is_hanging_man_not_hammer(self):
        rows = [{"open": 90.0, "high": 91.0, "low": 80.0, "close": 91.0}] * 11
        rows += [{"open": 100.2, "high": 100.5, "low": 90.0, "close": 100.0}]
        df = _make_indicator_ready_df(rows)
        out = detect_candle_patterns(df, {"trend_lookback": 10})
        assert out["cp_hanging_man"].iloc[-1] == True  # noqa: E712
        assert out["cp_hammer"].iloc[-1] == False  # noqa: E712


class TestEngulfingShape:
    def test_bullish_engulfing_detected(self):
        rows = [
            {"open": 105.0, "high": 106.0, "low": 99.0, "close": 100.0},   # bearish prior
            {"open": 99.0,  "high": 107.0, "low": 98.0, "close": 106.0},   # bull engulfs prior body
        ]
        df = _make_indicator_ready_df(rows)
        out = detect_candle_patterns(df)
        assert out["cp_bull_engulfing"].iloc[-1] == True  # noqa: E712
        assert out["cp_bear_engulfing"].iloc[-1] == False  # noqa: E712

    def test_bearish_engulfing_detected(self):
        rows = [
            {"open": 100.0, "high": 106.0, "low": 99.0, "close": 105.0},   # bullish prior
            {"open": 106.0, "high": 107.0, "low": 98.0, "close": 99.0},    # bear engulfs prior body
        ]
        df = _make_indicator_ready_df(rows)
        out = detect_candle_patterns(df)
        assert out["cp_bear_engulfing"].iloc[-1] == True  # noqa: E712
        assert out["cp_bull_engulfing"].iloc[-1] == False  # noqa: E712

    def test_small_body_does_not_engulf(self):
        rows = [
            {"open": 105.0, "high": 106.0, "low": 99.0, "close": 100.0},
            {"open": 102.0, "high": 103.0, "low": 101.0, "close": 102.5},  # tiny body, doesn't engulf
        ]
        df = _make_indicator_ready_df(rows)
        out = detect_candle_patterns(df)
        assert out["cp_bull_engulfing"].iloc[-1] == False  # noqa: E712


# ─────────────────────────────────────────────────────────────────────────────
# candle_patterns.py — integration over enriched_df
# ─────────────────────────────────────────────────────────────────────────────

ALL_CP_PATTERN_COLS = [
    "cp_hammer", "cp_hanging_man", "cp_shooting_star", "cp_inverted_hammer",
    "cp_bull_marubozu", "cp_bear_marubozu", "cp_doji", "cp_dragonfly_doji",
    "cp_gravestone_doji", "cp_long_legged_doji", "cp_spinning_top",
    "cp_bull_engulfing", "cp_bear_engulfing", "cp_piercing_line",
    "cp_dark_cloud_cover", "cp_bull_harami", "cp_bear_harami",
    "cp_tweezers_top", "cp_tweezers_bottom", "cp_bull_meeting_lines",
    "cp_bear_meeting_lines", "cp_bull_belt_hold", "cp_bear_belt_hold",
    "cp_morning_star", "cp_evening_star", "cp_bull_harami_cross",
    "cp_bear_harami_cross", "cp_three_white_soldiers", "cp_three_black_crows",
    "cp_three_inside_up", "cp_three_inside_down", "cp_three_outside_up",
    "cp_three_outside_down", "cp_morning_doji_star", "cp_evening_doji_star",
]


class TestDetectCandlePatternsIntegration:
    def test_all_documented_columns_present(self, enriched_df):
        for col in ALL_CP_PATTERN_COLS:
            assert col in enriched_df.columns, f"missing {col}"

    def test_all_pattern_columns_are_boolean(self, enriched_df):
        for col in ALL_CP_PATTERN_COLS:
            assert enriched_df[col].dtype == bool

    def test_doji_and_dragonfly_and_gravestone_are_mutually_exclusive(self, enriched_df):
        assert not (enriched_df["cp_doji"] & enriched_df["cp_dragonfly_doji"]).any()
        assert not (enriched_df["cp_doji"] & enriched_df["cp_gravestone_doji"]).any()

    def test_cp_bull_any_is_union_of_bull_patterns(self, enriched_df):
        # cp_bull_any must be true whenever any contributing bull pattern fires
        bull_cols = ["cp_hammer", "cp_inverted_hammer", "cp_dragonfly_doji",
                     "cp_bull_marubozu", "cp_bull_engulfing", "cp_piercing_line",
                     "cp_bull_harami", "cp_bull_harami_cross", "cp_tweezers_bottom",
                     "cp_bull_meeting_lines", "cp_bull_belt_hold",
                     "cp_morning_star", "cp_morning_doji_star",
                     "cp_three_white_soldiers", "cp_three_inside_up", "cp_three_outside_up"]
        union = np.zeros(len(enriched_df), dtype=bool)
        for c in bull_cols:
            union |= enriched_df[c].values
        assert (enriched_df["cp_bull_any"].values >= union).all()

    def test_confirmation_requires_base_sweep_at_the_point_it_is_computed(self, synthetic_ohlcv):
        # cp_bull_confirmation is computed from base_long *before* the
        # session filter stage narrows base_long further, so the invariant
        # must be checked at that point in the pipeline, not on the fully
        # enriched_df (where session_filter can since shrink base_long).
        from ceo_engine_mt5.indicators import calc_all
        from ceo_engine_mt5.signals import build_all
        df = build_all(calc_all(synthetic_ohlcv))
        df = build_candle_patterns(df)
        assert not (df["cp_bull_confirmation"] & ~df["base_long"]).any()
        assert not (df["cp_bear_confirmation"] & ~df["base_short"]).any()

    def test_pattern_summary_returns_sorted_counts(self, enriched_df):
        summary = pattern_summary(enriched_df)
        assert "count" in summary.columns
        assert (summary["count"].diff().dropna() <= 0).all()  # descending

    def test_apply_pattern_quality_bonus_keeps_scores_in_bounds(self, synthetic_ohlcv):
        from ceo_engine_mt5.indicators import calc_all
        from ceo_engine_mt5.signals import build_all
        df = build_all(calc_all(synthetic_ohlcv))
        if "quality_long" not in df.columns:
            pytest.skip("quality_long not produced in this configuration")
        df = detect_candle_patterns(df)
        df = apply_pattern_quality_bonus(df)
        assert (df["quality_long"] >= 0).all() and (df["quality_long"] <= 100).all()
        assert (df["quality_short"] >= 0).all() and (df["quality_short"] <= 100).all()

    def test_build_candle_patterns_pipeline_runs_end_to_end(self, synthetic_ohlcv):
        from ceo_engine_mt5.indicators import calc_all
        from ceo_engine_mt5.signals import build_all
        df = build_all(calc_all(synthetic_ohlcv))
        out = build_candle_patterns(df)
        assert len(out) == len(df)
        for col in ALL_CP_PATTERN_COLS:
            assert col in out.columns

    def test_does_not_mutate_input_df(self, enriched_df):
        before_cols = set(enriched_df.columns)
        _ = detect_candle_patterns(enriched_df)
        assert set(enriched_df.columns) == before_cols


# ─────────────────────────────────────────────────────────────────────────────
# patterns.py — pure helpers
# ─────────────────────────────────────────────────────────────────────────────

class TestPatternsPureHelpers:
    def test_near_true_within_tolerance(self):
        assert _near(100.0, 100.5, tol=1.0) is True
        assert _near(100.0, 102.0, tol=1.0) is False

    def test_slope_basic(self):
        assert _slope(0, 100.0, 10, 110.0) == 1.0
        assert _slope(0, 100.0, 10, 90.0) == -1.0

    def test_slope_zero_when_x_equal(self):
        assert _slope(5, 100.0, 5, 200.0) == 0.0

    def test_atr_at_returns_value_when_present(self):
        arr = np.array([1.0, 2.0, np.nan])
        assert _atr_at(arr, 0) == 1.0
        assert _atr_at(arr, 1) == 2.0

    def test_atr_at_falls_back_to_1_on_nan(self):
        arr = np.array([np.nan])
        assert _atr_at(arr, 0) == 1.0

    def test_build_pivot_index_extracts_only_non_nan(self):
        df = pd.DataFrame({
            "pivot_high": [np.nan, 105.0, np.nan, 107.0],
            "pivot_low":  [95.0, np.nan, 93.0, np.nan],
        })
        ph_idx, ph_val, pl_idx, pl_val = _build_pivot_index(df)
        assert list(ph_idx) == [1, 3]
        assert list(ph_val) == [105.0, 107.0]
        assert list(pl_idx) == [0, 2]
        assert list(pl_val) == [95.0, 93.0]

    def test_window_pivots_filters_to_range(self):
        idx_arr = np.array([0, 5, 10, 50, 100])
        val_arr = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        out = _window_pivots(bar_idx=12, lookback=10, idx_arr=idx_arr, val_arr=val_arr)
        # range is [2, 12] -> only idx 5 and 10 qualify
        assert out == [(5, 2.0), (10, 3.0)]

    def test_window_pivots_empty_when_no_pivots(self):
        out = _window_pivots(bar_idx=5, lookback=5, idx_arr=np.array([]), val_arr=np.array([]))
        assert out == []


# ─────────────────────────────────────────────────────────────────────────────
# patterns.py — integration over enriched_df
# ─────────────────────────────────────────────────────────────────────────────

ALL_PAT_COLS = [
    "pat_hs", "pat_ihs", "pat_double_top", "pat_double_bottom",
    "pat_triple_top", "pat_triple_bottom", "pat_rising_wedge", "pat_falling_wedge",
    "pat_bull_flag", "pat_bear_flag", "pat_asc_triangle", "pat_desc_triangle",
    "pat_sym_triangle", "pat_bull_pennant", "pat_bear_pennant", "pat_rectangle",
    "pat_bull_any", "pat_bear_any", "pat_neutral_any", "pat_name", "pat_quality",
]


class TestDetectPatternsIntegration:
    def test_all_documented_columns_present(self, enriched_df):
        for col in ALL_PAT_COLS:
            assert col in enriched_df.columns, f"missing {col}"

    def test_bool_columns_are_boolean(self, enriched_df):
        bool_cols = [c for c in ALL_PAT_COLS if c not in ("pat_name", "pat_quality")]
        for col in bool_cols:
            assert enriched_df[col].dtype == bool

    def test_pat_quality_within_bounds(self, enriched_df):
        assert (enriched_df["pat_quality"] >= 0).all()
        assert (enriched_df["pat_quality"] <= 100).all()

    def test_pat_bull_any_is_union_of_bull_patterns(self, enriched_df):
        bull_cols = ["pat_ihs", "pat_double_bottom", "pat_triple_bottom",
                     "pat_falling_wedge", "pat_bull_flag", "pat_bull_pennant",
                     "pat_asc_triangle"]
        union = np.zeros(len(enriched_df), dtype=bool)
        for c in bull_cols:
            union |= enriched_df[c].values
        assert (enriched_df["pat_bull_any"].values == union).all()

    def test_pat_bear_any_is_union_of_bear_patterns(self, enriched_df):
        bear_cols = ["pat_hs", "pat_double_top", "pat_triple_top",
                     "pat_rising_wedge", "pat_bear_flag", "pat_bear_pennant",
                     "pat_desc_triangle"]
        union = np.zeros(len(enriched_df), dtype=bool)
        for c in bear_cols:
            union |= enriched_df[c].values
        assert (enriched_df["pat_bear_any"].values == union).all()

    def test_apply_pattern_quality_bonus_keeps_scores_in_bounds(self, synthetic_ohlcv):
        from ceo_engine_mt5.indicators import calc_all
        from ceo_engine_mt5.signals import build_all
        df = build_all(calc_all(synthetic_ohlcv))
        if "quality_long" not in df.columns:
            pytest.skip("quality_long not produced in this configuration")
        out = build_patterns(df)
        assert (out["quality_long"] >= 0).all() and (out["quality_long"] <= 100).all()
        assert (out["quality_short"] >= 0).all() and (out["quality_short"] <= 100).all()

    def test_build_patterns_pipeline_runs_end_to_end(self, synthetic_ohlcv):
        from ceo_engine_mt5.indicators import calc_all
        from ceo_engine_mt5.signals import build_all
        df = build_all(calc_all(synthetic_ohlcv))
        out = build_patterns(df)
        assert len(out) == len(df)
        for col in ALL_PAT_COLS:
            assert col in out.columns

    def test_handles_dataset_with_no_pivots_gracefully(self):
        # Flat data -> indicators may produce no pivot highs/lows at all;
        # detect_patterns must not raise and should report nothing detected.
        n = 150
        idx = pd.date_range("2024-01-01", periods=n, freq="1h", tz="UTC")
        df = pd.DataFrame({
            "open": 100.0, "high": 100.0, "low": 100.0, "close": 100.0,
            "atr": 1.0, "pivot_high": np.nan, "pivot_low": np.nan,
        }, index=idx)
        out = detect_patterns(df)
        assert not out["pat_bull_any"].any()
        assert not out["pat_bear_any"].any()

    def test_does_not_mutate_input_df(self, enriched_df):
        before_cols = set(enriched_df.columns)
        _ = detect_patterns(enriched_df)
        assert set(enriched_df.columns) == before_cols
