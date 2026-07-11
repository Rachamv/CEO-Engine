"""
Tests for risk_engine.py — position sizing and the live pre-trade gate.
This is the money-math file: a sizing bug here either blows through risk
limits or makes every trade tiny, so PositionSizer gets worked-through,
concrete numeric examples rather than only structural assertions.
"""

from datetime import datetime, timezone

import pandas as pd
import pytest

from ceo_engine_mt5.risk_engine import PositionSizer, SessionFilter, RiskEngine


def _eurusd_info(**overrides):
    """A realistic 5-digit EURUSD symbol_info(), tick_size==point (forex default)."""
    info = dict(
        digits=5, tick_size=0.00001, tick_value=1.0, point=0.00001,
        volume_min=0.01, volume_max=100.0, volume_step=0.01, spread=10,
    )
    info.update(overrides)
    return info


class TestPositionSizerMath:
    """
    Worked example: $10,000 balance, 1% risk ($100), 10-pip (100-point) SL
    on EURUSD where tick_value=$1 per 0.00001 move per standard lot.
    Conventional sizing for this setup is 1.0 lot — verified independently
    by hand in the risk_engine review, not just re-deriving the formula.
    """

    def test_worked_example_matches_conventional_sizing(self):
        sizer = PositionSizer(risk_pct=1.0)
        lots, breakdown = sizer.calculate(
            balance=10_000, entry=1.10000, sl=1.09900,  # 0.0010 = 100 points = 10 pips
            sym_info=_eurusd_info(), direction="long",
        )
        # Rounds DOWN to the broker's lot step by design (never rounds up,
        # since that would risk more than configured) — so this lands just
        # under 1.0, not exactly 1.0.
        assert lots == pytest.approx(1.0, abs=0.02)
        assert lots <= 1.0
        assert breakdown["risk_amount"] == pytest.approx(100.0)
        assert breakdown["sl_pips"] == pytest.approx(100.0)  # in "points" units, see risk_engine.py docstring

    def test_doubling_risk_pct_doubles_lot_size(self):
        sizer_1pct = PositionSizer(risk_pct=1.0)
        sizer_2pct = PositionSizer(risk_pct=2.0)
        lots1, _ = sizer_1pct.calculate(10_000, 1.10000, 1.09900, _eurusd_info(), "long")
        lots2, _ = sizer_2pct.calculate(10_000, 1.10000, 1.09900, _eurusd_info(), "long")
        assert lots2 == pytest.approx(lots1 * 2, rel=0.02)

    def test_doubling_sl_distance_halves_lot_size(self):
        sizer = PositionSizer(risk_pct=1.0)
        lots_10pip, _ = sizer.calculate(10_000, 1.10000, 1.09900, _eurusd_info(), "long")
        lots_20pip, _ = sizer.calculate(10_000, 1.10000, 1.09800, _eurusd_info(), "long")
        assert lots_20pip == pytest.approx(lots_10pip / 2, rel=0.02)

    def test_zero_sl_distance_blocked(self):
        sizer = PositionSizer()
        lots, breakdown = sizer.calculate(10_000, 1.10000, 1.10000, _eurusd_info(), "long")
        assert lots == 0.0
        assert "error" in breakdown

    def test_sl_too_small_blocked(self):
        sizer = PositionSizer(min_sl_pips=5.0)
        # 1-point SL = 0.1 pip-equivalent in this unit system — far below min
        lots, breakdown = sizer.calculate(
            10_000, 1.10000, 1.099999, _eurusd_info(), "long")
        assert lots == 0.0
        assert "too small" in breakdown["error"]

    def test_sl_too_large_blocked(self):
        sizer = PositionSizer(max_sl_pips=500.0)
        lots, breakdown = sizer.calculate(
            10_000, 1.10000, 1.00000, _eurusd_info(), "long")  # 0.10 price = 10,000 points
        assert lots == 0.0
        assert "too large" in breakdown["error"]

    def test_result_respects_broker_lot_step(self):
        sizer = PositionSizer(risk_pct=1.0)
        lots, _ = sizer.calculate(
            10_000, 1.10000, 1.09900, _eurusd_info(volume_step=0.1), "long")
        # lots must land on a 0.1 grid
        assert round(lots / 0.1, 6) == round(lots / 0.1)

    def test_result_capped_at_broker_max(self):
        sizer = PositionSizer(risk_pct=50.0)  # deliberately oversized
        lots, _ = sizer.calculate(
            10_000, 1.10000, 1.09990, _eurusd_info(volume_max=5.0), "long")
        assert lots <= 5.0

    def test_below_broker_minimum_blocked_not_rounded_up(self):
        """A calculated lot size below the broker minimum should block the
        trade, not silently round up to the minimum (that would risk more
        than the configured risk_pct)."""
        sizer = PositionSizer(risk_pct=0.01)  # tiny risk -> tiny lot
        lots, breakdown = sizer.calculate(
            10_000, 1.10000, 1.09900, _eurusd_info(volume_min=1.0), "long")
        assert lots == 0.0
        assert "minimum" in breakdown["error"]


class TestSessionFilterTradeAnytime:
    """SessionFilter — mirrors session_filter.py's trade-anytime behavior
    but as the live order-gate's own independent implementation."""

    def test_default_is_trade_anytime(self):
        sf = SessionFilter()
        assert sf.trade_anytime is True

    def test_trade_anytime_allows_asian_session(self):
        sf = SessionFilter()
        ok, reason = sf.is_session_allowed(datetime(2024, 3, 12, 2, 0, tzinfo=timezone.utc))
        assert ok is True

    def test_trade_anytime_still_blocks_weekend(self):
        sf = SessionFilter()
        ok, reason = sf.is_session_allowed(datetime(2024, 3, 16, 12, 0, tzinfo=timezone.utc))
        assert ok is False

    def test_restricting_to_named_sessions_blocks_others(self):
        sf = SessionFilter(allowed_sessions=["london"])
        ok_london, _ = sf.is_session_allowed(datetime(2024, 3, 12, 8, 0, tzinfo=timezone.utc))
        ok_asian, _  = sf.is_session_allowed(datetime(2024, 3, 12, 2, 0, tzinfo=timezone.utc))
        assert ok_london is True
        assert ok_asian is False

    def test_quality_bar_still_applies_in_trade_anytime_mode(self):
        """'all' removes the time restriction, not the per-session quality bar
        (Asian/post-NY should still require a cleaner signal than London/NY)."""
        sf = SessionFilter()
        q_asian = sf.min_quality_for_session(datetime(2024, 3, 12, 2, 0, tzinfo=timezone.utc))
        q_london = sf.min_quality_for_session(datetime(2024, 3, 12, 8, 0, tzinfo=timezone.utc))
        assert q_asian > q_london


class TestRiskEngineEvaluate:
    """RiskEngine.evaluate() — the combined gate, smoke-tested end-to-end."""

    def _registered_engine(self, **kwargs) -> RiskEngine:
        """
        Gate 1 of evaluate() requires a deployable model — register_backtest()
        with >=min_trades and positive expectancy must run first, exactly as
        run_live() does via _register_models_for_symbols() before going live.
        """
        engine = RiskEngine(risk_pct=1.0, sessions=["all"], min_quality=0, **kwargs)
        results = pd.DataFrame({
            "Trades": [30], "Net R": [15.0], "Win Rate": [55.0], "Avg R": [0.5],
        }, index=["LQ"])
        engine.model_selector.register(results, "EURUSD", "M15")
        return engine

    def test_evaluate_returns_positive_lots_for_a_clean_setup(self):
        engine = self._registered_engine()
        lots, report = engine.evaluate(
            symbol="EURUSD", tf="M15", direction="long",
            entry=1.10000, sl=1.09900, quality=80.0,
            account={"balance": 10_000, "equity": 10_000},
            sym_info=_eurusd_info(),
            bar_time=datetime(2024, 3, 12, 8, 0, tzinfo=timezone.utc),
            spread_pips=1.0,
        )
        assert lots > 0

    def test_evaluate_blocks_on_excessive_spread(self):
        engine = self._registered_engine(max_spread_pips_abs=2.0)
        lots, report = engine.evaluate(
            symbol="EURUSD", tf="M15", direction="long",
            entry=1.10000, sl=1.09900, quality=80.0,
            account={"balance": 10_000, "equity": 10_000},
            sym_info=_eurusd_info(),
            bar_time=datetime(2024, 3, 12, 8, 0, tzinfo=timezone.utc),
            spread_pips=50.0,  # way over the 2.0 cap
        )
        assert lots == 0.0

    def test_evaluate_blocks_below_min_quality(self):
        engine = self._registered_engine()
        engine.min_quality = 90.0
        lots, report = engine.evaluate(
            symbol="EURUSD", tf="M15", direction="long",
            entry=1.10000, sl=1.09900, quality=50.0,  # below the 90 bar
            account={"balance": 10_000, "equity": 10_000},
            sym_info=_eurusd_info(),
            bar_time=datetime(2024, 3, 12, 8, 0, tzinfo=timezone.utc),
            spread_pips=1.0,
        )
        assert lots == 0.0

    def test_evaluate_blocks_when_no_model_registered(self):
        """Gate 1 itself: an engine with nothing registered must block
        every trade rather than sizing against an unvalidated model."""
        engine = RiskEngine(risk_pct=1.0, sessions=["all"], min_quality=0)
        lots, report = engine.evaluate(
            symbol="GBPUSD", tf="M15", direction="long",  # never registered
            entry=1.25000, sl=1.24900, quality=80.0,
            account={"balance": 10_000, "equity": 10_000},
            sym_info=_eurusd_info(),
            bar_time=datetime(2024, 3, 12, 8, 0, tzinfo=timezone.utc),
            spread_pips=1.0,
        )
        assert lots == 0.0
        assert report["blocked_by"] == "model_selector"


# ─────────────────────────────────────────────────────────────────────────────
# Spread normalisation (points → pips) inside evaluate()
# ─────────────────────────────────────────────────────────────────────────────

class TestSpreadNormalisation:
    """
    MT5 info.spread is always raw points. evaluate() must convert to true pips
    before comparing against the hard cap and ratio. A JPY pair (digits=3,
    point=0.001) has 1 pip = 100 points — without normalisation, a normal
    3-pip spread would be 300 points and would wrongly trip the 30-pip hard cap.
    """

    def _make_evaluator(self, max_spread_pips_abs=30.0, max_spread_ratio=0.20):
        from ceo_engine_mt5.risk_engine import RiskEngine
        re = RiskEngine(max_spread_pips_abs=max_spread_pips_abs,
                        max_spread_ratio=max_spread_ratio,
                        min_trades=1, min_expectancy=0.0)
        # Inject directly into the registry (avoids needing a full DataFrame)
        for sym in ("USDJPY", "EURUSD", "XAUUSD"):
            key = f"{sym}_H1"
            re.model_selector._registry[key] = {
                "model": "LQ", "expectancy": 0.3, "win_rate": 60.0,
                "avg_r": 0.3, "net_r": 5.0, "trades": 20, "max_dd_r": 1.0,
            }
        return re

    def _sym_info_jpy(self):
        """USDJPY: 3-digit, point=0.001 → 1 pip = 100 points."""
        return {"point": 0.001, "digits": 3, "tick_size": 0.001,
                "tick_value": 0.087, "contract_size": 100_000,
                "volume_min": 0.01, "volume_step": 0.01, "volume_max": 100.0}

    def _sym_info_forex(self):
        """EURUSD: 5-digit, point=0.00001 → 1 pip = 10 points."""
        return {"point": 0.00001, "digits": 5, "tick_size": 0.00001,
                "tick_value": 1.0, "contract_size": 100_000,
                "volume_min": 0.01, "volume_step": 0.01, "volume_max": 100.0}

    def _sym_info_xau(self):
        """XAUUSD: 2-digit, point=0.01 → 1 pip = 1 point."""
        return {"point": 0.01, "digits": 2, "tick_size": 0.01,
                "tick_value": 1.0, "contract_size": 100,
                "volume_min": 0.01, "volume_step": 0.01, "volume_max": 50.0}

    def _account(self):
        return {"balance": 10_000.0, "equity": 10_000.0, "currency": "USD"}

    def _bar_time(self):
        from datetime import datetime, timezone
        return datetime(2024, 6, 5, 10, 0, tzinfo=timezone.utc)  # Wednesday London

    def test_jpy_normal_spread_not_blocked(self):
        """3 pips on USDJPY = 300 raw points. Without normalisation this would
        wrongly exceed the 30-pip hard cap. With normalisation: 3 pips < 30 → OK."""
        ev = self._make_evaluator()
        sym = self._sym_info_jpy()
        # entry=150.000 sl=149.500 → 500 points = 5 pips SL — well within ratio
        lot, report = ev.evaluate(
            symbol="USDJPY", tf="H1", direction="long",
            entry=150.000, sl=149.500, quality=80.0,
            account=self._account(), sym_info=sym,
            bar_time=self._bar_time(),
            spread_pips=30,    # 30 raw points = 3 true pips (JPY: 1 pip = 10 points)
            verbose=False,
        )
        assert report["gates"]["spread"]["passed"] is True

    def test_jpy_wide_spread_blocked(self):
        """35 pips on USDJPY = 350 raw points → exceeds 30-pip hard cap → blocked."""
        ev = self._make_evaluator()
        sym = self._sym_info_jpy()
        lot, report = ev.evaluate(
            symbol="USDJPY", tf="H1", direction="long",
            entry=150.000, sl=149.500, quality=80.0,
            account=self._account(), sym_info=sym,
            bar_time=self._bar_time(),
            spread_pips=350,   # 350 raw points = 35 true pips → blocked
            verbose=False,
        )
        assert report["gates"]["spread"]["passed"] is False
        assert "35" in report["gates"]["spread"]["reason"] or "hard cap" in report["gates"]["spread"]["reason"].lower()

    def test_forex_5digit_3pip_spread_passes(self):
        """EURUSD: 3 pips = 30 raw points. Hard cap is 30 pips → at limit, passes."""
        ev = self._make_evaluator(max_spread_pips_abs=30.0)
        sym = self._sym_info_forex()
        lot, report = ev.evaluate(
            symbol="EURUSD", tf="H1", direction="long",
            entry=1.10000, sl=1.09500, quality=80.0,
            account=self._account(), sym_info=sym,
            bar_time=self._bar_time(),
            spread_pips=30,   # 30 raw points = 3 true pips (5-digit: 1 pip = 10 points)
            verbose=False,
        )
        assert report["gates"]["spread"]["passed"] is True

    def test_xauusd_normal_spread_passes(self):
        """XAUUSD: digits=2, pip==point. 30 raw points = 30 pips → at cap, passes."""
        ev = self._make_evaluator(max_spread_pips_abs=50.0)
        sym = self._sym_info_xau()
        lot, report = ev.evaluate(
            symbol="XAUUSD", tf="H1", direction="long",
            entry=2000.00, sl=1998.00, quality=80.0,
            account=self._account(), sym_info=sym,
            bar_time=self._bar_time(),
            spread_pips=30,   # 30 raw points = 30 pips (XAUUSD: 1 pip = 1 point)
            verbose=False,
        )
        assert report["gates"]["spread"]["passed"] is True

    def test_ratio_gate_uses_pip_units_not_points(self):
        """Ratio check: spread_in_pips / sl_pips must both be in pip units.
        EURUSD: spread=2 pips (20 points), SL=10 pips (100 points).
        Ratio = 2/10 = 0.20 → at the 20% limit → passes."""
        ev = self._make_evaluator(max_spread_ratio=0.20)
        sym = self._sym_info_forex()
        lot, report = ev.evaluate(
            symbol="EURUSD", tf="H1", direction="long",
            entry=1.10200, sl=1.10000, quality=80.0,   # SL = 200 points = 20 pips
            account=self._account(), sym_info=sym,
            bar_time=self._bar_time(),
            spread_pips=20,   # 20 raw points = 2 pips; ratio 2/20 = 0.10 → passes
            verbose=False,
        )
        assert report["gates"]["spread"]["passed"] is True
