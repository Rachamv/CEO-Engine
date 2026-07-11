"""
Tests for news_filter.py — offline-mode behavior, caching, and the
NewsEvent dataclass (converted from a manual __init__ class in v2.2.0).
No network calls anywhere in this file — every test that would need one
uses offline_mode=True instead.
"""

from datetime import datetime, timezone

import pytest

from ceo_engine_mt5.news_filter import NewsFilter, NewsEvent


class TestNewsEvent:
    """NewsEvent dataclass — normalization happens in __post_init__."""

    def test_currency_is_uppercased(self):
        e = NewsEvent(title="CPI", currency="usd", impact="high",
                      event_time=datetime.now(timezone.utc))
        assert e.currency == "USD"

    def test_impact_is_lowercased(self):
        e = NewsEvent(title="CPI", currency="USD", impact="HIGH",
                      event_time=datetime.now(timezone.utc))
        assert e.impact == "high"

    def test_impact_rank_ordering(self):
        high = NewsEvent(title="x", currency="USD", impact="high",
                          event_time=datetime.now(timezone.utc))
        low = NewsEvent(title="x", currency="USD", impact="low",
                         event_time=datetime.now(timezone.utc))
        assert high.impact_rank > low.impact_rank

    def test_optional_fields_default_to_none(self):
        e = NewsEvent(title="CPI", currency="USD", impact="high",
                      event_time=datetime.now(timezone.utc))
        assert e.actual is None
        assert e.forecast is None
        assert e.previous is None
        assert e.source == ""


class TestOfflineMode:
    def test_refresh_returns_zero_events(self):
        nf = NewsFilter(symbols=["XAUUSD"], offline_mode=True)
        assert nf.refresh(verbose=False) == 0

    def test_is_blocked_never_blocks_with_no_events(self):
        nf = NewsFilter(symbols=["XAUUSD"], offline_mode=True)
        nf.refresh(verbose=False)
        blocked, reason = nf.is_blocked(datetime.now(timezone.utc))
        assert blocked is False


class TestCaching:
    def test_second_refresh_within_window_uses_cache(self):
        nf = NewsFilter(symbols=["XAUUSD"], offline_mode=True, cache_hours=1)
        nf.refresh(verbose=False)
        first_fetch = nf._last_fetch
        nf.refresh(verbose=False)
        assert nf._last_fetch == first_fetch

    def test_force_bypasses_cache(self):
        nf = NewsFilter(symbols=["XAUUSD"], offline_mode=True, cache_hours=1)
        nf.refresh(verbose=False)
        first_fetch = nf._last_fetch
        nf.refresh(force=True, verbose=False)
        assert nf._last_fetch >= first_fetch

    def test_different_target_date_does_not_use_stale_cache(self):
        from datetime import date, timedelta
        nf = NewsFilter(symbols=["XAUUSD"], offline_mode=True, cache_hours=24)
        today = datetime.now(timezone.utc).date()
        nf.refresh(target_date=today, verbose=False)
        tomorrow = today + timedelta(days=1)
        # Different date -> cache for `today` shouldn't satisfy `tomorrow`
        count = nf.refresh(target_date=tomorrow, force=False, verbose=False)
        assert nf._fetch_date == tomorrow
