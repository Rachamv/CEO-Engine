"""
Tests for backtest.py — the core P&L simulation engine. This is the most
safety-critical file in the codebase (it decides what counts as a win or
loss), so every helper extracted during the v2.2.0 complexity pass gets
direct unit tests here, on top of the full-pipeline integration test.
"""

import numpy as np
import pandas as pd
import pytest

from ceo_engine_mt5.backtest import (
    _check_exit_hits, _compute_r_result, _try_enter_trade, _calc_sl,
    _simulate_model, run_backtest, build_session_mask, DEFAULT_BT_PARAMS,
    TRADE_ANYTIME,
)


class TestCheckExitHits:
    """_check_exit_hits — pure TP1/TP2/TP3/SL boolean logic, both directions."""

    def test_long_tp_hits(self):
        hit_tp1, hit_tp2, hit_tp3, hit_sl = _check_exit_hits(
            direction=1, h=110, lo=99, sl=95, tp1=105, tp2=108, tp3=112)
        assert hit_tp1 and hit_tp2 and not hit_tp3 and not hit_sl

    def test_long_sl_hit(self):
        hit_tp1, hit_tp2, hit_tp3, hit_sl = _check_exit_hits(
            direction=1, h=101, lo=94, sl=95, tp1=105, tp2=108, tp3=112)
        assert not hit_tp1 and hit_sl

    def test_short_mirrors_long(self):
        hit_tp1, hit_tp2, hit_tp3, hit_sl = _check_exit_hits(
            direction=-1, h=101, lo=90, sl=105, tp1=95, tp2=92, tp3=88)
        assert hit_tp1 and hit_tp2 and not hit_tp3 and not hit_sl

    def test_neither_hit_mid_range(self):
        hit_tp1, hit_tp2, hit_tp3, hit_sl = _check_exit_hits(
            direction=1, h=102, lo=99, sl=95, tp1=105, tp2=108, tp3=112)
        assert not any([hit_tp1, hit_tp2, hit_tp3, hit_sl])


class TestComputeRResult:
    """
    _compute_r_result — the actual win/loss math. Every branch matches
    the original inline logic's priority order exactly (see CHANGELOG
    v2.2.0): TP3+SL-same-bar > TP3 > SL > time-expired.
    """

    def test_tp3_hit_gives_full_weighted_r(self):
        r = _compute_r_result(
            hit_tp3=True, hit_sl=False, conservative=False,
            tp1_hit=True, tp2_hit=True,
            tp1_r=1.0, tp2_r=2.0, tp3_r=3.0,
            tp1_w=0.33, tp2_w=0.33, tp3_w=0.34,
            close_price=0, entry_price=0, direction=1, risk=1.0,
        )
        expected = 1.0*0.33 + 2.0*0.33 + 3.0*0.34
        assert r == pytest.approx(expected)

    def test_sl_hit_with_no_prior_tp_is_full_loss(self):
        r = _compute_r_result(
            hit_tp3=False, hit_sl=True, conservative=False,
            tp1_hit=False, tp2_hit=False,
            tp1_r=1.0, tp2_r=2.0, tp3_r=3.0,
            tp1_w=0.33, tp2_w=0.33, tp3_w=0.34,
            close_price=0, entry_price=0, direction=1, risk=1.0,
        )
        assert r == pytest.approx(-1.0)

    def test_sl_hit_after_tp1_is_partial_loss(self):
        r = _compute_r_result(
            hit_tp3=False, hit_sl=True, conservative=False,
            tp1_hit=True, tp2_hit=False,
            tp1_r=1.0, tp2_r=2.0, tp3_r=3.0,
            tp1_w=0.33, tp2_w=0.33, tp3_w=0.34,
            close_price=0, entry_price=0, direction=1, risk=1.0,
        )
        # banked tp1 leg + lost the remaining (1 - tp1_w) of size at SL
        expected = 1.0*0.33 - (1.0 - 0.33)
        assert r == pytest.approx(expected)

    def test_conservative_mode_treats_same_bar_tp3_and_sl_as_full_loss(self):
        """
        Gap-open SL fix territory: if both TP3 and SL could have been hit
        on the same bar, conservative mode assumes the worst or der (SL
        first) rather than crediting the win — this is what 'conservative_both'
        actually controls.
        """
        r = _compute_r_result(
            hit_tp3=True, hit_sl=True, conservative=True,
            tp1_hit=True, tp2_hit=True,
            tp1_r=1.0, tp2_r=2.0, tp3_r=3.0,
            tp1_w=0.33, tp2_w=0.33, tp3_w=0.34,
            close_price=0, entry_price=0, direction=1, risk=1.0,
        )
        assert r == -1.0

    def test_non_conservative_mode_credits_tp3_even_if_sl_also_touched(self):
        r = _compute_r_result(
            hit_tp3=True, hit_sl=True, conservative=False,
            tp1_hit=True, tp2_hit=True,
            tp1_r=1.0, tp2_r=2.0, tp3_r=3.0,
            tp1_w=0.33, tp2_w=0.33, tp3_w=0.34,
            close_price=0, entry_price=0, direction=1, risk=1.0,
        )
        expected = 1.0*0.33 + 2.0*0.33 + 3.0*0.34
        assert r == pytest.approx(expected)

    def test_time_expired_marks_to_close_on_remaining_size(self):
        # Long entry @100, risk=5 (SL would be @95), closes @106 with no TPs hit
        r = _compute_r_result(
            hit_tp3=False, hit_sl=False, conservative=False,
            tp1_hit=False, tp2_hit=False,
            tp1_r=1.0, tp2_r=2.0, tp3_r=3.0,
            tp1_w=0.33, tp2_w=0.33, tp3_w=0.34,
            close_price=106, entry_price=100, direction=1, risk=5.0,
        )
        # open_r = (106-100)/5 = 1.2R on 100% remaining size (no partials banked)
        assert r == pytest.approx(1.2)

    def test_time_expired_with_zero_risk_does_not_divide_by_zero(self):
        r = _compute_r_result(
            hit_tp3=False, hit_sl=False, conservative=False,
            tp1_hit=False, tp2_hit=False,
            tp1_r=1.0, tp2_r=2.0, tp3_r=3.0,
            tp1_w=0.33, tp2_w=0.33, tp3_w=0.34,
            close_price=106, entry_price=100, direction=1, risk=0.0,
        )
        assert r == 0.0  # open_r falls back to 0 when risk<=0, no exception


class TestTryEnterTrade:
    """_try_enter_trade — entry gating (ATR validity, session mask, signal, dedup)."""

    def _base_kwargs(self, **overrides):
        n = 5
        kwargs = dict(
            i=2, sig_bar=1,
            atrs=np.array([1.0, 1.0, 1.0, 1.0, 1.0]),
            session_mask=np.ones(n, dtype=bool),
            long_sig=np.array([False, True, False, False, False]),
            short_sig=np.zeros(n, dtype=bool),
            last_dir=0,
            opens=np.array([100., 100., 101., 102., 103.]),
            lows=np.array([99., 98., 100., 101., 102.]),
            highs=np.array([101., 101., 102., 103., 104.]),
            tp1_r=1.0, tp2_r=2.0, tp3_r=3.0,
            p={"sl_atr_mult": 1.5, "sl_buffer": 0.1, "sl_mode": "atr",
               "sl_min_atr_mult": 0.5, "sl_max_atr_mult": 3.0},
        )
        kwargs.update(overrides)
        return kwargs

    def test_enters_long_on_valid_signal(self):
        entry = _try_enter_trade(**self._base_kwargs())
        assert entry is not None
        assert entry["direction"] == 1
        assert entry["entry_price"] == 101.0  # opens[i=2]
        assert entry["entry_bar"] == 2

    def test_no_entry_when_atr_is_nan(self):
        kwargs = self._base_kwargs()
        kwargs["atrs"] = np.array([1.0, np.nan, 1.0, 1.0, 1.0])
        assert _try_enter_trade(**kwargs) is None

    def test_no_entry_when_atr_is_zero_or_negative(self):
        kwargs = self._base_kwargs()
        kwargs["atrs"] = np.array([1.0, 0.0, 1.0, 1.0, 1.0])
        assert _try_enter_trade(**kwargs) is None

    def test_no_entry_when_session_blocked(self):
        kwargs = self._base_kwargs()
        kwargs["session_mask"] = np.array([True, True, False, True, True])
        assert _try_enter_trade(**kwargs) is None

    def test_no_entry_when_no_signal(self):
        kwargs = self._base_kwargs()
        kwargs["long_sig"] = np.zeros(5, dtype=bool)
        assert _try_enter_trade(**kwargs) is None

    def test_same_direction_reentry_blocked(self):
        """The last_dir dedup guard: don't re-enter the same direction
        back-to-back (must close oppositely or flip first)."""
        kwargs = self._base_kwargs(last_dir=1)
        assert _try_enter_trade(**kwargs) is None

    def test_opposite_direction_reentry_allowed(self):
        kwargs = self._base_kwargs(last_dir=-1)
        assert _try_enter_trade(**kwargs) is not None


class TestSimulateModelIntegration:
    """End-to-end checks on the assembled simulation loop."""

    def test_produces_a_dataframe_with_expected_columns(self, enriched_df):
        df = enriched_df
        results = run_backtest(df, DEFAULT_BT_PARAMS)
        assert isinstance(results, dict)
        sample = next(iter(results.values()))
        for col in ["direction", "entry", "sl", "tp1", "tp2", "tp3",
                    "r_result", "win", "tp1_hit", "tp2_hit"]:
            assert col in sample.columns

    def test_every_trade_has_a_finite_r_result(self, enriched_df):
        results = run_backtest(enriched_df, DEFAULT_BT_PARAMS)
        for name, trades in results.items():
            if len(trades):
                assert np.isfinite(trades["r_result"]).all(), f"non-finite R in {name}"

    def test_win_flag_matches_positive_r_result(self, enriched_df):
        results = run_backtest(enriched_df, DEFAULT_BT_PARAMS)
        for name, trades in results.items():
            if len(trades):
                assert (trades["win"] == (trades["r_result"] > 0)).all()

    def test_deterministic_across_runs(self, enriched_df):
        """Same input must give byte-identical trades every time — no
        hidden randomness anywhere in the simulation loop."""
        r1 = run_backtest(enriched_df, DEFAULT_BT_PARAMS)
        r2 = run_backtest(enriched_df, DEFAULT_BT_PARAMS)
        for name in r1:
            pd.testing.assert_frame_equal(r1[name], r2[name])


class TestSessionMaskTradeAnytime:
    """build_session_mask — the 'all' bypass added in v2.1.0."""

    def test_disabled_filter_allows_everything(self, enriched_df):
        params = {**DEFAULT_BT_PARAMS, "session_filter": False}
        mask = build_session_mask(enriched_df, params)
        assert mask.all()

    def test_all_sentinel_allows_everything(self, enriched_df):
        params = {**DEFAULT_BT_PARAMS, "session_filter": True,
                  "active_sessions": [TRADE_ANYTIME]}
        mask = build_session_mask(enriched_df, params)
        assert mask.all()

    def test_restricted_to_one_session_is_a_strict_subset(self, enriched_df):
        params_all = {**DEFAULT_BT_PARAMS, "session_filter": True,
                      "active_sessions": [TRADE_ANYTIME]}
        params_ldn = {**DEFAULT_BT_PARAMS, "session_filter": True,
                      "active_sessions": ["london"]}
        mask_all = build_session_mask(enriched_df, params_all)
        mask_ldn = build_session_mask(enriched_df, params_ldn)
        assert mask_ldn.sum() < mask_all.sum()
        assert (mask_ldn <= mask_all).all()  # subset relationship holds bar-for-bar
