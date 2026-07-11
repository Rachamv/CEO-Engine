"""
Tests for the three pure pattern-detection layers: candle_patterns.py,
ceo_structure.py, patterns.py. These are detection-only (no trade signal
logic of their own — see signals.py for that), so the bar here is
"produces well-formed, internally-consistent output on real pipeline
data" rather than exhaustively verifying all 30+ individual candlestick
patterns or every geometric shape by hand.
"""

import numpy as np
import pandas as pd

from ceo_engine_mt5.candle_patterns import build_candle_patterns
from ceo_engine_mt5.ceo_structure import build_ceo_structure, _prune_order_blocks, _detect_new_order_block
from ceo_engine_mt5.patterns import build_patterns


class TestCandlePatterns:
    def test_runs_without_error_on_real_pipeline_data(self, enriched_df):
        assert "cp_bull_any" in enriched_df.columns
        assert "cp_bear_any" in enriched_df.columns

    def test_bull_any_is_or_of_individual_bull_patterns(self, enriched_df):
        # Mirrors candle_patterns.py's own BULL_PATTERNS list exactly — this
        # is deliberately a literal copy, not introspected from column names,
        # because cp_bull_strong/cp_bull_confirmation are *derived from*
        # cp_bull_any (not inputs to it) and would corrupt a naive
        # "every cp_bull_* column" reconstruction.
        bull_patterns = [
            "cp_hammer", "cp_inverted_hammer", "cp_dragonfly_doji",
            "cp_bull_marubozu", "cp_bull_engulfing", "cp_piercing_line",
            "cp_bull_harami", "cp_bull_harami_cross", "cp_tweezers_bottom",
            "cp_bull_meeting_lines", "cp_bull_belt_hold",
            "cp_morning_star", "cp_morning_doji_star",
            "cp_three_white_soldiers", "cp_three_inside_up", "cp_three_outside_up",
        ]
        manual_any = enriched_df[bull_patterns].any(axis=1)
        pd.testing.assert_series_equal(
            enriched_df["cp_bull_any"], manual_any, check_names=False)

    def test_confirmation_implies_bull_any_and_base_long(self, synthetic_ohlcv):
        """
        cp_bull_confirmation = cp_bull_any & base_long by construction —
        checked right after build_candle_patterns() runs, not on the full
        pipeline output. session_filter.py legitimately ANDs base_long
        with session validity *later* in the pipeline without retroactively
        updating cp_bull_confirmation (same pattern as swept_level_low in
        test_signals.py — see that test's docstring for the full rationale).
        """
        from ceo_engine_mt5.indicators import calc_all
        from ceo_engine_mt5.signals import build_all
        df = build_all(calc_all(synthetic_ohlcv))
        df = build_candle_patterns(df)
        confirmed = df["cp_bull_confirmation"]
        assert df.loc[confirmed, "cp_bull_any"].all()
        assert df.loc[confirmed, "base_long"].all()

    def test_doji_and_marubozu_are_mutually_exclusive(self, enriched_df):
        """A doji (near-zero body) and a marubozu (near-zero wicks, large
        body) describe opposite candle shapes — never both on the same bar."""
        both = enriched_df["cp_doji"] & (enriched_df["cp_bull_marubozu"] |
                                          enriched_df["cp_bear_marubozu"])
        assert not both.any()


class TestCeoStructure:
    def test_runs_without_error_on_real_pipeline_data(self, enriched_df):
        for col in ["ob_bull_active", "ob_bear_active", "ceo_long_valid",
                    "ceo_short_valid", "nearest_unmit_res", "nearest_unmit_sup"]:
            assert col in enriched_df.columns

    def test_ob_high_always_above_ob_low_when_active(self, enriched_df):
        active = enriched_df["ob_bull_active"]
        assert (enriched_df.loc[active, "ob_bull_high"] >=
                enriched_df.loc[active, "ob_bull_low"]).all()

    def test_quality_bonus_is_bounded(self, enriched_df):
        assert (enriched_df["ceo_quality_bonus"] >= 0).all()
        assert (enriched_df["ceo_quality_bonus"] <= 30).all()

    def test_ceo_valid_implies_base_signal_fired(self, synthetic_ohlcv):
        """
        ceo_long_valid is a *gate* on top of base_long — it can never be
        True when the underlying sweep signal didn't fire. Checked right
        after build_ceo_structure() runs (same staging rationale as the
        candle-pattern confirmation test above — session gating runs later
        and can veto base_long without ceo_long_valid being recomputed).
        """
        from ceo_engine_mt5.indicators import calc_all
        from ceo_engine_mt5.signals import build_all
        df = build_all(calc_all(synthetic_ohlcv))
        df = build_ceo_structure(df)
        assert (df.loc[df["ceo_long_valid"], "base_long"]).all()
        assert (df.loc[df["ceo_short_valid"], "base_short"]).all()


class TestPruneOrderBlocks:
    """Direct unit tests for the extracted OB-pruning helper."""

    def _params(self):
        return {"ob_mitigation_close": True, "ob_max_age": 50}

    def test_keeps_unmitigated_recent_ob(self):
        active = [(105.0, 100.0, 0)]  # (high, low, bar_idx)
        c = np.array([102.0] * 10)
        h = np.array([106.0] * 10)
        lo = np.array([101.0] * 10)
        result = _prune_order_blocks(active, i=5, c=c, h=h, lo=lo,
                                      p=self._params(), is_bull=True)
        assert result == active

    def test_drops_mitigated_bull_ob(self):
        """Bull OB mitigated when close drops below the OB low."""
        active = [(105.0, 100.0, 0)]
        c = np.array([99.0] * 10)   # closed below ob_low=100 -> mitigated
        h = np.array([106.0] * 10)
        lo = np.array([98.0] * 10)
        result = _prune_order_blocks(active, i=5, c=c, h=h, lo=lo,
                                      p=self._params(), is_bull=True)
        assert result == []

    def test_drops_aged_out_ob(self):
        active = [(105.0, 100.0, 0)]
        c = np.array([102.0] * 200)
        h = np.array([106.0] * 200)
        lo = np.array([101.0] * 200)
        result = _prune_order_blocks(active, i=100, c=c, h=h, lo=lo,
                                      p={"ob_mitigation_close": True, "ob_max_age": 50},
                                      is_bull=True)
        assert result == []


class TestGeometricPatterns:
    def test_runs_without_error_on_real_pipeline_data(self, enriched_df):
        for col in ["pat_hs", "pat_double_top", "pat_rising_wedge",
                    "pat_bull_flag", "pat_rectangle", "pat_quality", "pat_name"]:
            assert col in enriched_df.columns

    def test_quality_is_zero_when_no_pattern_fired(self, enriched_df):
        no_pattern = ~(
            enriched_df["pat_hs"] | enriched_df["pat_ihs"] |
            enriched_df["pat_double_top"] | enriched_df["pat_double_bottom"]
        )
        # Where literally nothing in our checked subset fired and pat_name
        # is empty, quality should be 0 (never a phantom positive score).
        none_at_all = no_pattern & (enriched_df["pat_name"] == "")
        assert (enriched_df.loc[none_at_all, "pat_quality"] == 0).all()

    def test_double_and_triple_top_are_mutually_exclusive(self, enriched_df):
        """detect_patterns()'s elif chain — triple takes priority over
        double on the same bar, never both flagged."""
        both = enriched_df["pat_double_top"] & enriched_df["pat_triple_top"]
        assert not both.any()
        both_bot = enriched_df["pat_double_bottom"] & enriched_df["pat_triple_bottom"]
        assert not both_bot.any()
