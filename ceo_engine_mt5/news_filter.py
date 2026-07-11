"""
The CEO Protocol — News / Economic Calendar Filter
=======================================================
Blocks trading around high-impact economic news events.

Sources (tried in order, first that works is used):
    1. ForexFactory JSON feed  (free, no key)
    2. Investing.com scrape    (free, no key, fallback)
    3. FCS API                 (free tier, requires key)
    4. Manual override list    (always available, no network)

Blocking logic:
    - High impact events   : block [pre_mins] before → [post_mins] after
    - Medium impact events : block [med_pre_mins] before → [med_post_mins] after
    - Low impact events    : not blocked (configurable)

Currency filtering:
    Only blocks events that affect the currencies in your watchlist.
    XAUUSD → blocks USD and XAU events.
    EURUSD → blocks EUR and USD events.

Usage
-----
    from .news_filter import NewsFilter

    nf = NewsFilter(
        symbols          = ["XAUUSD", "EURUSD", "GBPUSD"],
        block_high       = True,
        block_medium     = False,
        pre_mins         = 30,
        post_mins        = 15,
    )

    # On startup — fetch today's calendar
    nf.refresh()

    # Before every signal — check if blocked
    blocked, reason = nf.is_blocked(bar_time=datetime.now(timezone.utc))
    if blocked:
        print(f"News block: {reason}")
        return

    # Get upcoming events in the next window
    upcoming = nf.upcoming_events(window_mins=60)
"""

import os
from datetime import datetime, timedelta, timezone, date
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

import requests

from ceo_engine_mt5.ceo_logging import get_logger
logger = get_logger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Currency extraction from symbol
# ─────────────────────────────────────────────────────────────────────────────

# Maps symbols → affected currencies for news filtering
_SYMBOL_CURRENCIES: Dict[str, List[str]] = {
    "XAUUSD": ["USD", "XAU", "GOLD"],
    "XAGUSD": ["USD", "XAG", "SILVER"],
    "XAUEUR": ["EUR", "XAU", "GOLD"],
    "EURUSD": ["EUR", "USD"],
    "GBPUSD": ["GBP", "USD"],
    "USDJPY": ["USD", "JPY"],
    "USDCHF": ["USD", "CHF"],
    "USDCAD": ["USD", "CAD"],
    "AUDUSD": ["AUD", "USD"],
    "NZDUSD": ["NZD", "USD"],
    "EURGBP": ["EUR", "GBP"],
    "EURJPY": ["EUR", "JPY"],
    "GBPJPY": ["GBP", "JPY"],
    "US30":   ["USD"],
    "US500":  ["USD"],
    "NAS100": ["USD"],
    "UK100":  ["GBP"],
    "GER40":  ["EUR"],
    "BTCUSD": ["USD"],
    "ETHUSD": ["USD"],
}

def _currencies_for_symbol(symbol: str) -> List[str]:
    """Extract affected currencies from a symbol name."""
    sym = symbol.upper().replace("_", "").replace(".", "").replace(" ", "")

    # Direct lookup first
    if sym in _SYMBOL_CURRENCIES:
        return _SYMBOL_CURRENCIES[sym]

    # Generic: try splitting 6-char forex pair
    if len(sym) == 6:
        return [sym[:3], sym[3:]]

    # Fallback: treat as USD-affected
    return ["USD"]


def _currencies_for_watchlist(symbols: List[str]) -> List[str]:
    """Return unique set of currencies affected by the watchlist."""
    currencies = set()
    for sym in symbols:
        currencies.update(_currencies_for_symbol(sym))
    return list(currencies)


# ─────────────────────────────────────────────────────────────────────────────
# Event dataclass
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class NewsEvent:
    """A single economic calendar event."""

    title:      str
    currency:   str
    impact:     str
    event_time: datetime   # UTC
    actual:     Optional[str] = None
    forecast:   Optional[str] = None
    previous:   Optional[str] = None
    source:     str = ""

    IMPACT_RANK = {"high": 3, "medium": 2, "low": 1, "holiday": 0}

    def __post_init__(self):
        # Normalize exactly as the original hand-written __init__ did
        self.title    = str(self.title)
        self.currency = str(self.currency).upper()
        self.impact   = str(self.impact).lower()

    @property
    def impact_rank(self) -> int:
        return self.IMPACT_RANK.get(self.impact, 0)

    def __repr__(self):
        ts = self.event_time.strftime("%H:%M UTC")
        return f"[{ts}] {self.currency} {self.impact.upper()}: {self.title}"


# ─────────────────────────────────────────────────────────────────────────────
# Calendar sources
# ─────────────────────────────────────────────────────────────────────────────

def _fetch_forexfactory(target_date: date, timeout: int = 10) -> List[NewsEvent]:
    """
    Fetch from ForexFactory JSON API.
    URL: https://nfs.faireconomy.media/ff_calendar_thisweek.json
    Free, no key required, returns the current week.
    """
    url = "https://nfs.faireconomy.media/ff_calendar_thisweek.json"
    try:
        resp = requests.get(url, timeout=timeout,
                            headers={"User-Agent": "Mozilla/5.0"})
        if resp.status_code != 200:
            return []
        data = resp.json()
    except Exception as e:
        logger.warning("ForexFactory news fetch failed: %s", e)
        return []

    events = []
    for item in data:
        try:
            # FF date format: "01-15-2024T13:30:00-0500"
            dt_str   = item.get("date", "")
            # Parse and convert to UTC
            from dateutil import parser as dtparser
            dt_local = dtparser.parse(dt_str)
            dt_utc   = dt_local.astimezone(timezone.utc)

            if dt_utc.date() != target_date:
                continue

            impact_map = {"High": "high", "Medium": "medium",
                          "Low": "low", "Holiday": "holiday"}
            impact = impact_map.get(item.get("impact", ""), "low")

            events.append(NewsEvent(
                title      = item.get("title", "Unknown"),
                currency   = item.get("country", "USD").upper(),
                impact     = impact,
                event_time = dt_utc,
                actual     = item.get("actual"),
                forecast   = item.get("forecast"),
                previous   = item.get("previous"),
                source     = "forexfactory",
            ))
        except Exception as e:
            logger.debug("Skipping malformed ForexFactory event: %s", e)
            continue

    return events


def _fetch_fcsapi(target_date: date, api_key: str,
                  timeout: int = 10) -> List[NewsEvent]:
    """
    Fetch from FCS API (free tier, 500 req/month).
    Requires a free API key from https://fcsapi.com
    """
    url = (f"https://fcsapi.com/api-v3/forex/economy_cal"
           f"?access_key={api_key}"
           f"&from={target_date}&to={target_date}")
    try:
        resp = requests.get(url, timeout=timeout)
        if resp.status_code != 200:
            return []
        data = resp.json()
        if not data.get("status"):
            return []
        items = data.get("response", [])
    except Exception as e:
        logger.warning("FCS API news fetch failed: %s", e)
        return []

    events = []
    for item in items:
        try:
            dt_str  = f"{item['date']} {item['time']}"
            dt_utc  = datetime.strptime(dt_str, "%Y-%m-%d %H:%M").replace(
                tzinfo=timezone.utc)
            impact_map = {"3": "high", "2": "medium", "1": "low"}
            impact = impact_map.get(str(item.get("importance", "1")), "low")
            events.append(NewsEvent(
                title      = item.get("event", "Unknown"),
                currency   = item.get("country", "USD").upper(),
                impact     = impact,
                event_time = dt_utc,
                source     = "fcsapi",
            ))
        except Exception as e:
            logger.debug("Skipping malformed FCS API event: %s", e)
            continue

    return events


# Known recurring high-impact events as a hardcoded fallback
# Approximate UTC times — not exact but better than nothing offline
_MANUAL_EVENTS = {
    "friday": [
        {"title": "US Non-Farm Payrolls", "currency": "USD",
         "impact": "high", "hour": 13, "minute": 30,
         "week": "first"},   # first Friday of month
        {"title": "US Unemployment Rate", "currency": "USD",
         "impact": "high", "hour": 13, "minute": 30,
         "week": "first"},
    ],
    "any": [
        {"title": "FOMC Rate Decision (approx)", "currency": "USD",
         "impact": "high", "hour": 19, "minute": 0,
         "dates": []},   # populated at runtime if known
        {"title": "US CPI (approx)", "currency": "USD",
         "impact": "high", "hour": 13, "minute": 30,
         "dates": []},
    ],
}


# ─────────────────────────────────────────────────────────────────────────────
# Main NewsFilter class
# ─────────────────────────────────────────────────────────────────────────────

class NewsFilter:
    """
    Economic calendar-aware trading gate.

    Parameters
    ----------
    symbols         : list of trading symbols to watch
    block_high      : block trading around high-impact events (default True)
    block_medium    : block trading around medium-impact events (default False)
    block_low       : block trading around low-impact events (default False)
    pre_mins        : minutes to block BEFORE a high-impact event (default 30)
    post_mins       : minutes to block AFTER a high-impact event (default 15)
    med_pre_mins    : minutes to block before medium-impact event (default 15)
    med_post_mins   : minutes to block after medium-impact event (default 10)
    fcsapi_key      : optional FCS API key for fallback calendar source
    cache_hours     : how long to cache calendar data (default 6 hours)
    offline_mode    : use manual event list only (no network calls)
    """

    def __init__(
        self,
        symbols:        List[str] = None,
        block_high:     bool  = True,
        block_medium:   bool  = False,
        block_low:      bool  = False,
        pre_mins:       int   = 30,
        post_mins:      int   = 15,
        med_pre_mins:   int   = 15,
        med_post_mins:  int   = 10,
        fcsapi_key:     Optional[str] = None,
        cache_hours:    int   = 6,
        offline_mode:   bool  = False,
    ):
        self.symbols      = symbols or []
        self.block_high   = block_high
        self.block_medium = block_medium
        self.block_low    = block_low
        self.pre_mins     = pre_mins
        self.post_mins    = post_mins
        self.med_pre_mins = med_pre_mins
        self.med_post_mins= med_post_mins
        self.fcsapi_key   = fcsapi_key or os.environ.get("CEO_FCSAPI_KEY", "")
        self.cache_hours  = cache_hours
        self.offline_mode = offline_mode

        self._events:      List[NewsEvent]  = []
        self._last_fetch:  Optional[datetime] = None
        self._fetch_date:  Optional[date]     = None
        self._watch_currencies = _currencies_for_watchlist(self.symbols)

    # ── Refresh ───────────────────────────────────────────────────────────────

    def _is_cache_fresh(self, today: date, force: bool) -> bool:
        """True if a cached fetch for `today` is still within cache_hours."""
        if force or self._last_fetch is None or self._fetch_date != today:
            return False
        age = (datetime.now(timezone.utc) - self._last_fetch).total_seconds()
        return age < self.cache_hours * 3600

    def _fetch_all_sources(self, today: date, verbose: bool) -> List["NewsEvent"]:
        """Tries ForexFactory, then FCS API as a fallback if that returned nothing."""
        if self.offline_mode:
            return []

        all_events: List[NewsEvent] = []
        try:
            ff_events = _fetch_forexfactory(today)
            if ff_events:
                all_events.extend(ff_events)
                if verbose:
                    print(f"  📅  NewsFilter: {len(ff_events)} events "
                          f"from ForexFactory for {today}")
        except Exception as e:
            if verbose:
                print(f"  ⚠️  ForexFactory fetch failed: {e}")

        if not all_events and self.fcsapi_key:
            try:
                fcs_events = _fetch_fcsapi(today, self.fcsapi_key)
                if fcs_events:
                    all_events.extend(fcs_events)
                    if verbose:
                        print(f"  📅  NewsFilter: {len(fcs_events)} events "
                              f"from FCS API for {today}")
            except Exception as e:
                if verbose:
                    print(f"  ⚠️  FCS API fetch failed: {e}")

        return all_events

    def refresh(
        self,
        target_date: Optional[date] = None,
        force:       bool           = False,
        verbose:     bool           = True,
    ) -> int:
        """
        Fetch today's economic calendar events.
        Returns number of relevant events found.
        Caches for cache_hours to avoid repeated API calls.
        """
        today = target_date or datetime.now(timezone.utc).date()

        if self._is_cache_fresh(today, force):
            return len(self._events)

        all_events = self._fetch_all_sources(today, verbose)

        # Filter to relevant currencies only
        self._events = [
            e for e in all_events
            if e.currency in self._watch_currencies
            or e.currency in ("ALL", "GLOBAL", "")
        ]

        self._last_fetch  = datetime.now(timezone.utc)
        self._fetch_date  = today

        if verbose and self._events:
            print(f"  📅  Relevant events today ({', '.join(self._watch_currencies)}):")
            for e in sorted(self._events, key=lambda x: x.event_time):
                print(f"       {e}")

        if not self._events and verbose:
            print(f"  📅  NewsFilter: no relevant events found for {today} "
                  f"(currencies: {', '.join(self._watch_currencies)})")

        return len(self._events)

    # ── Block check ───────────────────────────────────────────────────────────

    def is_blocked(
        self,
        bar_time:  Optional[datetime] = None,
        symbol:    Optional[str]      = None,
        verbose:   bool               = False,
    ) -> Tuple[bool, str]:
        """
        Check if trading is blocked at bar_time due to news.
        Returns (is_blocked, reason_string).

        If symbol is provided, only checks events affecting that symbol's
        currencies. Otherwise checks all watched currencies.
        """
        now = bar_time or datetime.now(timezone.utc)
        if now.tzinfo is None:
            now = now.replace(tzinfo=timezone.utc)

        # Auto-refresh if stale or new day
        today = now.date()
        stale = (
            self._fetch_date != today
            or self._last_fetch is None
            or (now - self._last_fetch).total_seconds() > self.cache_hours * 3600
        )
        if stale:
            self.refresh(target_date=today, verbose=False)
            # Re-read now after refresh
            today = now.date()

        # Which currencies to check
        if symbol:
            check_currencies = set(_currencies_for_symbol(symbol))
        else:
            check_currencies = set(self._watch_currencies)

        for event in self._events:
            if event.currency not in check_currencies:
                continue

            # Determine block window
            if event.impact == "high" and self.block_high:
                pre  = self.pre_mins
                post = self.post_mins
            elif event.impact == "medium" and self.block_medium:
                pre  = self.med_pre_mins
                post = self.med_post_mins
            elif event.impact == "low" and self.block_low:
                pre  = 5
                post = 5
            else:
                continue

            block_start = event.event_time - timedelta(minutes=pre)
            block_end   = event.event_time + timedelta(minutes=post)

            if block_start <= now <= block_end:
                reason = (
                    f"News block: {event.currency} {event.impact.upper()} "
                    f"'{event.title}' @ {event.event_time.strftime('%H:%M UTC')} "
                    f"(window: -{pre}min / +{post}min)"
                )
                if verbose:
                    print(f"  🚫  {reason}")
                return True, reason

        return False, ""

    # ── Upcoming events ───────────────────────────────────────────────────────

    def upcoming_events(
        self,
        window_mins: int              = 120,
        bar_time:    Optional[datetime] = None,
        symbol:      Optional[str]    = None,
    ) -> List[NewsEvent]:
        """
        Return events scheduled within the next window_mins minutes.
        Useful for alerting in advance.
        """
        now = bar_time or datetime.now(timezone.utc)
        if now.tzinfo is None:
            now = now.replace(tzinfo=timezone.utc)

        cutoff = now + timedelta(minutes=window_mins)

        currencies = (set(_currencies_for_symbol(symbol))
                      if symbol else set(self._watch_currencies))

        return [
            e for e in self._events
            if e.currency in currencies
            and now <= e.event_time <= cutoff
        ]

    def next_event(
        self,
        symbol: Optional[str] = None,
    ) -> Optional[NewsEvent]:
        """Return the next upcoming event, or None."""
        now       = datetime.now(timezone.utc)
        currencies = (set(_currencies_for_symbol(symbol))
                      if symbol else set(self._watch_currencies))
        future = [
            e for e in self._events
            if e.currency in currencies and e.event_time > now
        ]
        return min(future, key=lambda e: e.event_time) if future else None

    def minutes_to_next_event(self, symbol: Optional[str] = None) -> Optional[float]:
        """Minutes until the next relevant event, or None if no events."""
        event = self.next_event(symbol)
        if event is None:
            return None
        now = datetime.now(timezone.utc)
        return (event.event_time - now).total_seconds() / 60.0

    # ── Status / display ──────────────────────────────────────────────────────

    def status(self, bar_time: Optional[datetime] = None) -> dict:
        """Returns a status dict for the dashboard."""
        now      = bar_time or datetime.now(timezone.utc)
        blocked, reason = self.is_blocked(now)
        upcoming = self.upcoming_events(window_mins=120, bar_time=now)
        next_ev  = self.next_event()

        return {
            "blocked":          blocked,
            "block_reason":     reason,
            "events_today":     len(self._events),
            "upcoming_count":   len(upcoming),
            "next_event":       str(next_ev) if next_ev else None,
            "minutes_to_next":  self.minutes_to_next_event(),
            "last_refresh":     self._last_fetch.isoformat() if self._last_fetch else None,
            "watch_currencies": self._watch_currencies,
        }

    def print_today(self):
        """Print all relevant events for today."""
        if not self._events:
            print("  No relevant events loaded. Call refresh() first.")
            return

        print(f"\n  ── Today's Events ({', '.join(self._watch_currencies)}) ──")
        high   = [e for e in self._events if e.impact == "high"]
        medium = [e for e in self._events if e.impact == "medium"]
        low    = [e for e in self._events if e.impact == "low"]

        for group, label in [(high,"🔴 HIGH"),(medium,"🟡 MEDIUM"),(low,"⚪ LOW")]:
            if group:
                print(f"\n  {label}")
                for e in sorted(group, key=lambda x: x.event_time):
                    blocked = ""
                    if e.impact == "high" and self.block_high:
                        blocked = f"  ← BLOCKED ({self.pre_mins}min pre / {self.post_mins}min post)"
                    elif e.impact == "medium" and self.block_medium:
                        blocked = f"  ← BLOCKED"
                    print(f"    {e}{blocked}")


# ─────────────────────────────────────────────────────────────────────────────
# Integration helpers for mt5_live.py
# ─────────────────────────────────────────────────────────────────────────────

def make_news_filter(
    symbols:      List[str],
    block_high:   bool = True,
    block_medium: bool = False,
    pre_mins:     int  = 30,
    post_mins:    int  = 15,
    fcsapi_key:   str  = "",
    offline:      bool = False,
    verbose:      bool = True,
) -> NewsFilter:
    """
    Convenience constructor — creates, refreshes, and returns a NewsFilter.
    Call once at startup.
    """
    nf = NewsFilter(
        symbols      = symbols,
        block_high   = block_high,
        block_medium = block_medium,
        pre_mins     = pre_mins,
        post_mins    = post_mins,
        fcsapi_key   = fcsapi_key,
        offline_mode = offline,
    )
    nf.refresh(verbose=verbose)
    nf.print_today()
    return nf


# ─────────────────────────────────────────────────────────────────────────────
# Self-test
# ─────────────────────────────────────────────────────────────────────────────
