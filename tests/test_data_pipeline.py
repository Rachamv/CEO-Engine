"""
Additional tests for data.py, covering the parts test_data.py doesn't:
  - _normalise_tf: timeframe alias mapping + unsupported-tf errors
  - _to_utc: tz-naive vs tz-aware index handling
  - _fetch_csv: datetime column auto-detection
  - resample_ohlcv: OHLCV aggregation + unsupported-tf error
  - fetch_ohlcv: source dispatch (yfinance/ccxt/csv/mt5/unknown), with the
    actual network/broker fetchers monkeypatched out
"""

import pandas as pd
import pytest

from ceo_engine_mt5 import data as data_mod
from ceo_engine_mt5.data import _normalise_tf, _to_utc, _fetch_csv, resample_ohlcv, fetch_ohlcv


# ─────────────────────────────────────────────────────────────────────────────
# _normalise_tf
# ─────────────────────────────────────────────────────────────────────────────

class TestNormaliseTf:
    def test_yfinance_alias_maps_to_canonical_form(self):
        assert _normalise_tf("60m", "yfinance") == "1h"

    def test_ccxt_alias_maps_to_canonical_form(self):
        assert _normalise_tf("60m", "ccxt") == "1h"

    def test_case_and_whitespace_insensitive(self):
        assert _normalise_tf("  1H ".lower().strip(), "yfinance") == "1h"
        assert _normalise_tf("1H", "yfinance") == "1h"

    def test_unsupported_timeframe_raises(self):
        with pytest.raises(ValueError, match="Unsupported timeframe"):
            _normalise_tf("7m", "yfinance")

    def test_ccxt_and_yfinance_maps_can_differ(self):
        # "1w" -> "1wk" for yfinance, "1w" for ccxt
        assert _normalise_tf("1w", "yfinance") == "1wk"
        assert _normalise_tf("1w", "ccxt") == "1w"


# ─────────────────────────────────────────────────────────────────────────────
# _to_utc
# ─────────────────────────────────────────────────────────────────────────────

class TestToUtc:
    def test_naive_index_gets_localized(self):
        idx = pd.date_range("2024-01-01", periods=3, freq="1h")
        out = _to_utc(idx)
        assert str(out.tz) == "UTC"

    def test_aware_index_gets_converted(self):
        idx = pd.date_range("2024-01-01", periods=3, freq="1h", tz="America/New_York")
        out = _to_utc(idx)
        assert str(out.tz) == "UTC"
        # Conversion preserves the instant, not the wall clock time.
        assert out[0] == idx[0].tz_convert("UTC")


# ─────────────────────────────────────────────────────────────────────────────
# _fetch_csv
# ─────────────────────────────────────────────────────────────────────────────

class TestFetchCsv:
    def test_uses_explicit_datetime_column(self, tmp_path):
        p = tmp_path / "bars.csv"
        p.write_text("datetime,open,high,low,close,volume\n"
                      "2024-01-01 00:00:00,1,2,0.5,1.5,100\n"
                      "2024-01-01 01:00:00,1.5,2.5,1,2,110\n")
        df = _fetch_csv(str(p))
        assert df.index.name == "datetime"
        assert len(df) == 2

    def test_autodetects_a_date_like_column_when_no_datetime_col(self, tmp_path):
        p = tmp_path / "bars.csv"
        p.write_text("timestamp,open,high,low,close,volume\n"
                      "2024-01-01 00:00:00,1,2,0.5,1.5,100\n")
        df = _fetch_csv(str(p))
        assert df.index.name == "datetime"

    def test_falls_back_to_first_column_when_nothing_date_like(self, tmp_path):
        p = tmp_path / "bars.csv"
        p.write_text("2024-01-01,1,2,0.5,1.5,100\n2024-01-02,1.5,2.5,1,2,110\n"
                      )
        # Give it a header with no date/time/timestamp substring at all.
        p.write_text("idx,open,high,low,close,volume\n"
                      "2024-01-01 00:00:00,1,2,0.5,1.5,100\n")
        df = _fetch_csv(str(p))
        assert df.index.name == "datetime"
        assert len(df) == 1


# ─────────────────────────────────────────────────────────────────────────────
# resample_ohlcv
# ─────────────────────────────────────────────────────────────────────────────

class TestResampleOhlcv:
    def _hourly_df(self):
        idx = pd.date_range("2024-01-01", periods=4, freq="1h", tz="UTC")
        df = pd.DataFrame({
            "open":  [1.0, 2.0, 3.0, 4.0],
            "high":  [1.5, 2.5, 3.5, 4.5],
            "low":   [0.5, 1.5, 2.5, 3.5],
            "close": [1.2, 2.2, 3.2, 4.2],
            "volume":[10, 20, 30, 40],
        }, index=idx)
        df.attrs["tick_size"] = 0.01
        return df

    def test_aggregates_ohlcv_correctly(self):
        resampled = resample_ohlcv(self._hourly_df(), "4h")
        assert len(resampled) == 1
        row = resampled.iloc[0]
        assert row["open"] == 1.0
        assert row["high"] == 4.5
        assert row["low"] == 0.5
        assert row["close"] == 4.2
        assert row["volume"] == 100

    def test_preserves_attrs(self):
        resampled = resample_ohlcv(self._hourly_df(), "4h")
        assert resampled.attrs["tick_size"] == 0.01

    def test_unsupported_target_tf_raises(self):
        with pytest.raises(ValueError, match="Unsupported resample timeframe"):
            resample_ohlcv(self._hourly_df(), "7m")


# ─────────────────────────────────────────────────────────────────────────────
# fetch_ohlcv — source dispatch
# ─────────────────────────────────────────────────────────────────────────────

def _stub_raw_df():
    idx = pd.date_range("2024-01-01", periods=2, freq="1h")
    return pd.DataFrame({
        "open": [1.0, 2.0], "high": [1.5, 2.5], "low": [0.5, 1.5],
        "close": [1.2, 2.2], "volume": [10, 20],
    }, index=idx)


class TestFetchOhlcvDispatch:
    def test_yfinance_source_calls_yfinance_fetcher(self, monkeypatch):
        calls = []
        monkeypatch.setattr(data_mod, "_fetch_yfinance",
                             lambda symbol, timeframe, start=None, end=None:
                             calls.append(("yfinance", symbol, timeframe)) or _stub_raw_df())
        df = fetch_ohlcv("EURUSD", timeframe="1h", source="yfinance")
        assert calls == [("yfinance", "EURUSD", "1h")]
        assert len(df) == 2

    def test_ccxt_source_calls_ccxt_fetcher(self, monkeypatch):
        calls = []
        monkeypatch.setattr(data_mod, "_fetch_ccxt",
                             lambda symbol, timeframe, start=None, end=None, exchange="binance":
                             calls.append(("ccxt", symbol, exchange)) or _stub_raw_df())
        df = fetch_ohlcv("BTC/USDT", source="ccxt", exchange="kraken")
        assert calls == [("ccxt", "BTC/USDT", "kraken")]
        assert len(df) == 2

    def test_csv_source_requires_filepath(self):
        with pytest.raises(ValueError, match="filepath required"):
            fetch_ohlcv("EURUSD", source="csv")

    def test_csv_source_calls_csv_fetcher(self, monkeypatch):
        monkeypatch.setattr(data_mod, "_fetch_csv", lambda filepath: _stub_raw_df())
        df = fetch_ohlcv("EURUSD", source="csv", filepath="whatever.csv")
        assert len(df) == 2

    def test_mt5_source_calls_mt5_fetcher(self, monkeypatch):
        calls = []
        monkeypatch.setattr(data_mod, "_fetch_mt5",
                             lambda symbol, timeframe, start=None, end=None, n_bars=5000:
                             calls.append(symbol) or _stub_raw_df())
        df = fetch_ohlcv("XAUUSD", source="mt5")
        assert calls == ["XAUUSD"]
        assert len(df) == 2

    def test_unknown_source_raises(self):
        with pytest.raises(ValueError, match="Unknown source"):
            fetch_ohlcv("EURUSD", source="carrier_pigeon")

    def test_source_is_case_and_whitespace_insensitive(self, monkeypatch):
        monkeypatch.setattr(data_mod, "_fetch_yfinance",
                             lambda symbol, timeframe, start=None, end=None: _stub_raw_df())
        df = fetch_ohlcv("EURUSD", source="  YFinance ")
        assert len(df) == 2


# ─────────────────────────────────────────────────────────────────────────────
# _clean — edge cases beyond tick_size (covered in test_data.py)
# ─────────────────────────────────────────────────────────────────────────────

from ceo_engine_mt5.data import _clean


class TestCleanEdgeCases:
    def test_price_at_max_bucket_returns_ten(self):
        from ceo_engine_mt5.data import _estimate_tick_size
        series = pd.Series([50000.0] * 10)
        assert _estimate_tick_size(series) == 10.0

    def test_adj_close_renamed_to_close(self):
        idx = pd.date_range("2024-01-01", periods=2, freq="1h")
        df = pd.DataFrame({
            "Open": [1.0, 2.0], "High": [1.5, 2.5], "Low": [0.5, 1.5],
            "Adj Close": [1.2, 2.2],
        }, index=idx)
        cleaned = _clean(df, "AAPL")
        assert "close" in cleaned.columns
        assert list(cleaned["close"]) == [1.2, 2.2]

    def test_missing_required_column_raises(self):
        idx = pd.date_range("2024-01-01", periods=2, freq="1h")
        df = pd.DataFrame({"open": [1.0, 2.0], "high": [1.5, 2.5]}, index=idx)
        with pytest.raises(ValueError, match="Missing columns"):
            _clean(df, "EURUSD")

    def test_missing_volume_defaults_to_zero(self):
        idx = pd.date_range("2024-01-01", periods=2, freq="1h")
        df = pd.DataFrame({
            "open": [1.0, 2.0], "high": [1.5, 2.5],
            "low": [0.5, 1.5], "close": [1.2, 2.2],
        }, index=idx)
        cleaned = _clean(df, "EURUSD")
        assert list(cleaned["volume"]) == [0.0, 0.0]

    def test_non_positive_close_rows_are_dropped(self):
        idx = pd.date_range("2024-01-01", periods=3, freq="1h")
        df = pd.DataFrame({
            "open": [1.0, 2.0, 3.0], "high": [1.5, 2.5, 3.5],
            "low": [0.5, 1.5, 2.5], "close": [1.2, 0.0, 3.2],
            "volume": [10, 20, 30],
        }, index=idx)
        cleaned = _clean(df, "EURUSD")
        assert len(cleaned) == 2
        assert (cleaned["close"] > 0).all()


# ─────────────────────────────────────────────────────────────────────────────
# _fetch_yfinance / _fetch_ccxt / _fetch_mt5 — network/broker fetchers,
# exercised with the external library or connection object mocked out.
# ─────────────────────────────────────────────────────────────────────────────

class _FakeYfTicker:
    def __init__(self, df):
        self._df = df

    def history(self, **kwargs):
        return self._df


class TestFetchYfinance:
    def test_returns_history_dataframe(self, monkeypatch):
        import sys, types
        fake_yf = types.SimpleNamespace(Ticker=lambda symbol: _FakeYfTicker(_stub_raw_df()))
        monkeypatch.setitem(sys.modules, "yfinance", fake_yf)
        df = data_mod._fetch_yfinance("EURUSD=X", "1h")
        assert len(df) == 2

    def test_empty_result_raises(self, monkeypatch):
        import sys, types
        fake_yf = types.SimpleNamespace(
            Ticker=lambda symbol: _FakeYfTicker(pd.DataFrame()))
        monkeypatch.setitem(sys.modules, "yfinance", fake_yf)
        with pytest.raises(ValueError, match="no data"):
            data_mod._fetch_yfinance("BADTICKER", "1h")

    def test_missing_library_raises_helpful_error(self, monkeypatch):
        import sys
        monkeypatch.setitem(sys.modules, "yfinance", None)
        with pytest.raises(ImportError, match="pip install yfinance"):
            data_mod._fetch_yfinance("EURUSD=X", "1h")


class _FakeExchange:
    def __init__(self, pages):
        self._pages = list(pages)
        self.loaded = False

    def load_markets(self):
        self.loaded = True

    def fetch_ohlcv(self, symbol, timeframe=None, since=None, limit=None):
        if self._pages:
            return self._pages.pop(0)
        return []


class TestFetchCcxt:
    def test_paginates_until_short_page(self, monkeypatch):
        import sys, types
        page1 = [[1704067200000 + i * 60000, 1, 2, 0.5, 1.5, 10] for i in range(3)]
        fake_exchange_cls = lambda cfg: _FakeExchange([page1])
        fake_ccxt = types.SimpleNamespace(binance=fake_exchange_cls)
        monkeypatch.setitem(sys.modules, "ccxt", fake_ccxt)
        df = data_mod._fetch_ccxt("BTC/USDT", "1m", limit_per_call=1000)
        assert len(df) == 3

    def test_unknown_exchange_raises(self, monkeypatch):
        import sys, types
        fake_ccxt = types.SimpleNamespace()  # no attribute for any exchange
        monkeypatch.setitem(sys.modules, "ccxt", fake_ccxt)
        with pytest.raises(ValueError, match="Unknown ccxt exchange"):
            data_mod._fetch_ccxt("BTC/USDT", "1m", exchange="not_real")

    def test_no_data_raises(self, monkeypatch):
        import sys, types
        fake_exchange_cls = lambda cfg: _FakeExchange([])
        fake_ccxt = types.SimpleNamespace(binance=fake_exchange_cls)
        monkeypatch.setitem(sys.modules, "ccxt", fake_ccxt)
        with pytest.raises(ValueError, match="no data"):
            data_mod._fetch_ccxt("BTC/USDT", "1m")

    def test_missing_library_raises_helpful_error(self, monkeypatch):
        import sys
        monkeypatch.setitem(sys.modules, "ccxt", None)
        with pytest.raises(ImportError, match="pip install ccxt"):
            data_mod._fetch_ccxt("BTC/USDT", "1m")


class _FakeMT5Conn:
    def __init__(self, rates):
        self._rates = rates
        self.disconnected = False

    def connect(self):
        pass

    def symbol_info(self, symbol):
        return {"tick_size": 0.01}

    def fetch_rates(self, symbol, timeframe, start=None, end=None, n_bars=None):
        return self._rates

    def disconnect(self):
        self.disconnected = True


class TestFetchMt5:
    def _rates(self):
        return [
            {"time": 1704067200, "open": 1.0, "high": 1.5, "low": 0.5,
             "close": 1.2, "tick_volume": 100},
            {"time": 1704070800, "open": 1.2, "high": 1.6, "low": 1.0,
             "close": 1.4, "tick_volume": 110},
        ]

    def test_returns_ohlcv_with_broker_tick_size(self, monkeypatch):
        conn = _FakeMT5Conn(self._rates())
        monkeypatch.setattr(data_mod, "MT5Connection", lambda: conn, raising=False)
        import ceo_engine_mt5.mt5_connect as mt5_connect_mod
        monkeypatch.setattr(mt5_connect_mod, "MT5Connection", lambda: conn)
        monkeypatch.setattr(mt5_connect_mod, "_require_mt5", lambda: None)

        df = data_mod._fetch_mt5("XAUUSD", "H1")
        assert list(df.columns) == ["open", "high", "low", "close", "volume"]
        assert df.attrs["tick_size"] == 0.01
        assert conn.disconnected is True

    def test_disconnects_even_if_processing_fails(self, monkeypatch):
        conn = _FakeMT5Conn(rates=None)  # will blow up building the DataFrame
        import ceo_engine_mt5.mt5_connect as mt5_connect_mod
        monkeypatch.setattr(mt5_connect_mod, "MT5Connection", lambda: conn)
        monkeypatch.setattr(mt5_connect_mod, "_require_mt5", lambda: None)

        with pytest.raises(Exception):
            data_mod._fetch_mt5("XAUUSD", "H1")
        assert conn.disconnected is True
