"""
Tests for session_filter.py — session classification, the trade-anytime
default (v2.1.0), and real-market-closure handling.
"""

from datetime import datetime, timezone

import numpy as np
import pandas as pd
import pytest

from ceo_engine_mt5.session_filter import (
    _classify_bar, is_valid_session, add_session_columns,
    SESSION_WINDOWS, ALL_SESSIONS, TRADE_ANYTIME, DEFAULT_ALLOWED,
)


def _utc(*args) -> datetime:
    return datetime(*args, tzinfo=timezone.utc)


class TestClassifyBar:
    """_classify_bar: every weekday/weekend boundary, named session priority."""

    @pytest.mark.parametrize("dt,expected_primary", [
        (_utc(2024, 3, 12, 2, 0),  "asian"),       # Tuesday 02:00 UTC
        (_utc(2024, 3, 12, 6, 0),  "pre_london"),  # Tuesday 06:00 UTC
        (_utc(2024, 3, 12, 8, 0),  "london"),      # Tuesday 08:00 UTC
        (_utc(2024, 3, 12, 13, 0), "overlap"),     # Tuesday 13:00 UTC (London+NY)
        (_utc(2024, 3, 12, 18, 0), "new_york"),    # Tuesday 18:00 UTC
        (_utc(2024, 3, 12, 22, 0), "post_ny"),     # Tuesday 22:00 UTC
        (_utc(2024, 3, 16, 12, 0), "weekend"),     # Saturday
        (_utc(2024, 3, 17, 2, 0),  "weekend"),     # Sunday before 05:00
        (_utc(2024, 3, 17, 6, 0),  "pre_london"),  # Sunday after 05:00 — market open
        (_utc(2024, 3, 15, 21, 0), "post_ny"),     # Friday 21:00 — post-close
    ])
    def test_primary_session(self, dt, expected_primary):
        _, primary, _, _ = _classify_bar(dt)
        assert primary == expected_primary

    def test_weekend_has_zero_quality_multiplier_sessions(self):
        sessions, primary, mult, risk = _classify_bar(_utc(2024, 3, 16, 12, 0))
        assert primary == "weekend"
        assert all(v is False for v in sessions.values())

    def test_naive_datetime_treated_as_utc(self):
        """A tz-naive datetime should classify identically to its UTC-aware equivalent."""
        naive = datetime(2024, 3, 12, 13, 0)
        aware = _utc(2024, 3, 12, 13, 0)
        _, p1, _, _ = _classify_bar(naive)
        _, p2, _, _ = _classify_bar(aware)
        assert p1 == p2 == "overlap"


class TestTradeAnytime:
    """
    Regression tests for the v2.1.0 fix: the default session config used
    to silently block 21:00-07:00 UTC every day. 'all' must mean every
    hour the market is genuinely open, with real closure (weekend) still
    enforced. See CHANGELOG v2.1.0.
    """

    @pytest.mark.parametrize("dt", [
        _utc(2024, 3, 12, 2, 0),   # Asian
        _utc(2024, 3, 12, 13, 0),  # overlap
        _utc(2024, 3, 12, 22, 0),  # post_ny — this is the hour that used to be blocked
        _utc(2024, 3, 17, 6, 0),   # Sunday morning, market just opened
    ])
    def test_all_sessions_allowed_when_market_open(self, dt):
        valid, _, _ = is_valid_session(dt, allowed=["all"])
        assert valid is True

    @pytest.mark.parametrize("dt", [
        _utc(2024, 3, 16, 12, 0),  # Saturday
        _utc(2024, 3, 17, 2, 0),   # Sunday before 05:00
    ])
    def test_real_closure_still_blocked_even_with_all(self, dt):
        valid, primary, _ = is_valid_session(dt, allowed=["all"])
        assert valid is False
        assert primary == "weekend"

    def test_restricting_to_named_sessions_still_works(self):
        """'all' is the default, but explicit restriction must still work."""
        valid_london, _, _ = is_valid_session(_utc(2024, 3, 12, 8, 0), allowed=["london"])
        valid_asian_during_london, _, _ = is_valid_session(
            _utc(2024, 3, 12, 8, 0), allowed=["asian"])
        assert valid_london is True
        assert valid_asian_during_london is False

    def test_default_allowed_is_trade_anytime(self):
        """DEFAULT_ALLOWED itself must be the trade-anytime sentinel, not a
        restrictive subset — this is the actual default-safety regression."""
        assert DEFAULT_ALLOWED == [TRADE_ANYTIME]

    def test_all_sessions_constant_covers_every_window(self):
        assert set(ALL_SESSIONS) == set(SESSION_WINDOWS.keys())


@pytest.mark.filterwarnings("ignore:add_session_columns.*called before expected pipeline stages")
class TestAddSessionColumns:
    """
    add_session_columns() — the DataFrame-level version used by backtest/live.
    Tested standalone with a minimal synthetic DataFrame (not the full
    pipeline), which intentionally triggers the module's own "called out
    of order" warning — expected here, not a real concern.
    """

    def _hourly_week(self) -> pd.DataFrame:
        idx = pd.date_range("2024-03-11", periods=24 * 8, freq="1h", tz="UTC")  # Mon-Mon
        return pd.DataFrame({
            "close": 1.0, "base_long": True, "base_short": True,
        }, index=idx)

    def test_all_excludes_only_weekend_hours(self):
        df = self._hourly_week()
        out = add_session_columns(df, allowed=["all"])
        weekend_count = (out["sess_name"] == "weekend").sum()
        # Saturday (24h) + Sunday before 05:00 (5h) = 29 weekend hours in the week
        assert weekend_count == 29
        assert out["base_long"].sum() == len(df) - weekend_count

    def test_restricted_to_london_only_keeps_london_hours(self):
        df = self._hourly_week()
        out = add_session_columns(df, allowed=["london"])
        assert out["base_long"].sum() == (out["sess_name"] == "london").sum()
        assert (out["sess_name"] == "london").sum() > 0

    def test_quality_multiplier_present_for_every_row(self):
        df = self._hourly_week()
        out = add_session_columns(df, allowed=["all"])
        assert "sess_quality_mult" in out.columns
        assert out["sess_quality_mult"].notna().all()
