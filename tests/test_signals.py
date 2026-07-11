"""
Tests for signals.py — the 16 liquidity-sweep models. Focused on
detect_sweeps() and its helpers (extracted in the v2.2.0 complexity pass),
since this is the literal entry signal logic everything else builds on.
"""

import numpy as np
import pandas as pd
import pytest

from ceo_engine_mt5.signals import (
    detect_sweeps, build_all, MODEL_NAMES, NUM_MODELS,
    _update_swing_pool, _check_sweep_long, _check_sweep_short,
)


class TestSwingPool:
    """_update_swing_pool — the capped FIFO pivot pool maintained per bar."""

    def test_appends_valid_value(self):
        pool = [1.0, 2.0]
        _update_swing_pool(pool, 3.0, pool_size=5)
        assert pool == [1.0, 2.0, 3.0]

    def test_ignores_nan(self):
        pool = [1.0, 2.0]
        _update_swing_pool(pool, np.nan, pool_size=5)
        assert pool == [1.0, 2.0]

    def test_caps_at_pool_size_dropping_oldest(self):
        pool = [1.0, 2.0, 3.0]
        _update_swing_pool(pool, 4.0, pool_size=3)
        assert pool == [2.0, 3.0, 4.0]


class TestSweepChecks:
    """_check_sweep_long / _check_sweep_short — pure boolean+level logic."""

    def test_long_sweep_detected_when_depth_and_rejection_ok(self):
        # low dips 0.5 below a 100.0 pivot (0.5 ATR depth), closes back above,
        # strong lower rejection — should register as a sweep of that level.
        level = _check_sweep_long(
            pool_lows=[100.0], low_i=99.5, close_i=100.2, atr_i=1.0,
            max_depth_atr=0.80, min_rejection=0.20, lower_rej_i=0.50,
        )
        assert level == 100.0

    def test_long_sweep_rejected_when_too_deep(self):
        # 2.0 ATR depth, far beyond the 0.80 max — not a sweep, a breakdown.
        level = _check_sweep_long(
            pool_lows=[100.0], low_i=98.0, close_i=100.2, atr_i=1.0,
            max_depth_atr=0.80, min_rejection=0.20, lower_rej_i=0.50,
        )
        assert np.isnan(level)

    def test_long_sweep_rejected_when_rejection_too_weak(self):
        level = _check_sweep_long(
            pool_lows=[100.0], low_i=99.5, close_i=100.2, atr_i=1.0,
            max_depth_atr=0.80, min_rejection=0.20, lower_rej_i=0.05,
        )
        assert np.isnan(level)

    def test_long_sweep_rejected_when_close_does_not_reclaim_level(self):
        level = _check_sweep_long(
            pool_lows=[100.0], low_i=99.5, close_i=99.8, atr_i=1.0,
            max_depth_atr=0.80, min_rejection=0.20, lower_rej_i=0.50,
        )
        assert np.isnan(level)

    def test_long_sweep_checks_most_recent_level_first(self):
        """Pool is ordered oldest->newest; the most recently-confirmed pivot
        should be checked first (reversed iteration)."""
        level = _check_sweep_long(
            pool_lows=[105.0, 100.0], low_i=99.5, close_i=100.2, atr_i=1.0,
            max_depth_atr=0.80, min_rejection=0.20, lower_rej_i=0.50,
        )
        assert level == 100.0  # only 100.0 was actually swept; 105.0 wasn't touched

    def test_short_sweep_mirrors_long_sweep_logic(self):
        level = _check_sweep_short(
            pool_highs=[100.0], high_i=100.5, close_i=99.8, atr_i=1.0,
            max_depth_atr=0.80, min_rejection=0.20, upper_rej_i=0.50,
        )
        assert level == 100.0

    def test_no_levels_in_pool_returns_nan(self):
        assert np.isnan(_check_sweep_long([], 99.5, 100.2, 1.0, 0.8, 0.2, 0.5))
        assert np.isnan(_check_sweep_short([], 100.5, 99.8, 1.0, 0.8, 0.2, 0.5))


class TestDetectSweepsIntegration:
    """detect_sweeps() end-to-end on real indicator-enriched data."""

    def test_produces_expected_columns(self, enriched_df):
        for col in ["last_swing_high", "last_swing_low", "swept_level_high",
                    "swept_level_low", "base_long", "base_short"]:
            assert col in enriched_df.columns

    def test_base_long_and_short_are_boolean(self, enriched_df):
        assert enriched_df["base_long"].dtype == bool
        assert enriched_df["base_short"].dtype == bool

    def test_swept_level_only_set_when_base_signal_fires(self, synthetic_ohlcv):
        """
        detect_sweeps()'s own contract: swept_level_low/high are set if and
        only if base_long/base_short fired *at that stage*. Tested directly
        against detect_sweeps' output rather than the full enriched_df fixture,
        since session_filter.py legitimately ANDs base_long with session
        validity later in the pipeline — by design, swept_level_low isn't
        re-narrowed to match (it answers "was a sweep detected", not "is
        this bar actionable after all gating"), so the two can legitimately
        diverge downstream. That's not a bug — see session_filter.py:287-288.
        """
        from ceo_engine_mt5.indicators import calc_all
        from ceo_engine_mt5.signals import detect_sweeps
        df = detect_sweeps(calc_all(synthetic_ohlcv))
        long_fired = df["base_long"]
        short_fired = df["base_short"]
        assert df.loc[long_fired, "swept_level_low"].notna().all()
        assert df.loc[~long_fired, "swept_level_low"].isna().all()
        assert df.loc[short_fired, "swept_level_high"].notna().all()
        assert df.loc[~short_fired, "swept_level_high"].isna().all()

    def test_deterministic_given_same_seed(self, synthetic_ohlcv):
        """Running detect_sweeps twice on identical input must give identical output —
        this is the literal entry signal, must have zero hidden randomness/state leakage."""
        from ceo_engine_mt5.indicators import calc_all
        df1 = calc_all(synthetic_ohlcv.copy())
        df2 = calc_all(synthetic_ohlcv.copy())
        out1 = detect_sweeps(df1)
        out2 = detect_sweeps(df2)
        pd.testing.assert_series_equal(out1["base_long"], out2["base_long"])
        pd.testing.assert_series_equal(out1["base_short"], out2["base_short"])


class TestModelRegistry:
    """The 16-model name/filter registry that confluence mode depends on."""

    def test_exactly_sixteen_models(self):
        assert NUM_MODELS == 16
        assert len(MODEL_NAMES) == 16

    def test_model_0_is_raw_sweep(self):
        assert MODEL_NAMES[0] == "LQ"

    def test_model_15_is_all_filters(self):
        assert "All" in MODEL_NAMES[15]

    def test_build_all_produces_quality_columns_for_every_model(self, synthetic_ohlcv):
        from ceo_engine_mt5.indicators import calc_all
        df = calc_all(synthetic_ohlcv)
        df = build_all(df)
        for i in range(NUM_MODELS):
            assert f"m{i:02d}_long" in df.columns
            assert f"m{i:02d}_short" in df.columns
            assert f"m{i:02d}_quality_long" in df.columns


# ─────────────────────────────────────────────────────────────────────────────
# _track_fvg_fills
# ─────────────────────────────────────────────────────────────────────────────

def _make_fvg_df(high, low, close, bull_fvg=None, bear_fvg=None):
    """Minimal DataFrame for _track_fvg_fills testing."""
    n = len(high)
    idx = pd.date_range("2024-01-01", periods=n, freq="1h", tz="UTC")
    df = pd.DataFrame({
        "open":     close,
        "high":     high,
        "low":      low,
        "close":    close,
        "volume":   [100.0] * n,
        "bull_fvg": bull_fvg if bull_fvg is not None else [False] * n,
        "bear_fvg": bear_fvg if bear_fvg is not None else [False] * n,
    }, index=idx)
    df.index.name = "datetime"
    return df


class TestTrackFVGFills:
    """_track_fvg_fills — gap formation, expiry, and fill detection."""

    from ceo_engine_mt5.signals import _track_fvg_fills  # imported at class level

    def test_unfilled_gap_marks_bars_true_within_age_limit(self):
        from ceo_engine_mt5.signals import _track_fvg_fills
        # Bar 2: bull_fvg formed (high[0]=100.0 → gap_bottom; low[2]=102.0 → gap_top)
        # Bars 3-5: price stays above gap_top (low > 102) → gap lives
        high  = [100.0, 101.0, 103.0, 104.0, 105.0, 106.0]
        low   = [ 99.0,  99.5, 102.0, 102.5, 103.0, 103.5]
        close = [100.0, 100.5, 102.5, 103.0, 104.0, 105.0]
        fvg   = [False, False, True,  False, False, False]
        df    = _make_fvg_df(high, low, close, bull_fvg=fvg)
        result = _track_fvg_fills(df, recent_bars=6)
        # Bar 2 is the formation bar: low[2]==gap_top (102.0). The gap survives
        # (low >= gt holds at the boundary), so bar 2 is marked unfilled.
        # Bars 3-5: price strictly above gap_top → gap continues to survive.
        assert result["bull_fvg_recent_unfilled"].iloc[2] is np.bool_(True)
        assert result["bull_fvg_recent_unfilled"].iloc[3] is np.bool_(True)
        assert result["bull_fvg_recent_unfilled"].iloc[4] is np.bool_(True)

    def test_gap_filled_when_price_enters_zone(self):
        from ceo_engine_mt5.signals import _track_fvg_fills
        # Bull gap: gap_bottom=high[0]=100, gap_top=low[2]=102
        # Bar 4: low=101 which is inside the gap (100 < 101 < 102) → filled
        high  = [100.0, 101.0, 103.0, 104.0, 104.0]
        low   = [ 99.0,  99.5, 102.0, 102.5, 101.0]  # bar 4 enters gap
        close = [100.0, 100.5, 102.5, 103.5, 102.0]
        fvg   = [False, False, True,  False, False]
        df    = _make_fvg_df(high, low, close, bull_fvg=fvg)
        result = _track_fvg_fills(df, recent_bars=6)
        assert result["bull_fvg_recent_unfilled"].iloc[4] is np.bool_(False)

    def test_gap_filled_when_price_blows_straight_through(self):
        from ceo_engine_mt5.signals import _track_fvg_fills
        # Bull gap: gap_bottom=high[0]=100, gap_top=low[2]=102
        # Bar 3: low=99.5 — price blew THROUGH the entire gap bottom → filled
        high  = [100.0, 101.0, 103.0,  102.0]
        low   = [ 99.0,  99.5, 102.0,   99.5]   # bar 3: below gap_bottom
        close = [100.0, 100.5, 102.5,  100.0]
        fvg   = [False, False, True,   False]
        df    = _make_fvg_df(high, low, close, bull_fvg=fvg)
        result = _track_fvg_fills(df, recent_bars=6)
        # Gap must be marked filled on the bar where price passed through
        assert result["bull_fvg_recent_unfilled"].iloc[3] is np.bool_(False)

    def test_gap_expires_after_recent_bars(self):
        from ceo_engine_mt5.signals import _track_fvg_fills
        # Gap formed at bar 2; with recent_bars=2, it expires after bar 4
        # Price stays above gap all the way through → expiry is the killer
        high  = [100.0, 101.0, 103.0, 104.0, 105.0, 106.0]
        low   = [ 99.0,  99.5, 102.0, 102.5, 103.0, 103.5]
        close = [100.0, 100.5, 102.5, 103.5, 104.0, 105.0]
        fvg   = [False, False, True,  False, False, False]
        df    = _make_fvg_df(high, low, close, bull_fvg=fvg)
        result = _track_fvg_fills(df, recent_bars=2)
        # Gap formed at bar 2. recent_bars=2 means i-b <= 2 survives.
        # Bar 2 (i-b=0): alive; bar 3 (i-b=1): alive; bar 4 (i-b=2): alive (at limit).
        # Bar 5 (i-b=3): beyond recent_bars=2 → must expire → False
        assert result["bull_fvg_recent_unfilled"].iloc[4] is np.bool_(True)
        assert result["bull_fvg_recent_unfilled"].iloc[5] is np.bool_(False)

    def test_explicit_recent_bars_overrides_attrs(self):
        from ceo_engine_mt5.signals import _track_fvg_fills
        high  = [100.0, 101.0, 103.0, 104.0, 105.0]
        low   = [ 99.0,  99.5, 102.0, 102.5, 103.0]
        close = [100.0, 100.5, 102.5, 103.5, 104.0]
        fvg   = [False, False, True,  False, False]
        df    = _make_fvg_df(high, low, close, bull_fvg=fvg)
        df.attrs["fvg_recent_bars"] = 10    # attrs says 10...
        result = _track_fvg_fills(df, recent_bars=1)  # but param says 1
        # With recent_bars=1, gap expires when i-b > 1 i.e. bar 4+ (i-b=2).
        # Bar 3 (i-b=1): at the limit → still alive (True).
        # Bar 4 (i-b=2): beyond recent_bars=1 → False, regardless of attrs.
        assert result["bull_fvg_recent_unfilled"].iloc[3] is np.bool_(True)
        assert result["bull_fvg_recent_unfilled"].iloc[4] is np.bool_(False)
