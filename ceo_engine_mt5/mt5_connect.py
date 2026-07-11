"""
The CEO Protocol MT5 — Connection Manager
==============================================
Handles MT5 terminal initialisation, symbol info, account info.
Fails cleanly on non-Windows or missing MetaTrader5 package.
"""

import sys
from typing import Dict, Any
from ceo_engine_mt5.ceo_logging import get_logger, log_to_dashboard

logger = get_logger(__name__)
from datetime import datetime


def _check_mt5_available():
    if sys.platform != "win32":
        return False
    try:
        return True
    except ImportError:
        return False

MT5_AVAILABLE = _check_mt5_available()


def _require_mt5():
    if not MT5_AVAILABLE:
        if sys.platform != "win32":
            raise RuntimeError(
                "MT5 Python API is only available on Windows.\n"
                "Use --source yfinance or --source ccxt on Mac/Linux."
            )
        raise RuntimeError(
            "MetaTrader5 package not installed.\n"
            "Run: pip install MetaTrader5"
        )


def get_mt5_timeframe(tf: str):
    _require_mt5()
    import MetaTrader5 as mt5
    TF_MAP = {
        "m1":mt5.TIMEFRAME_M1,   "1m":mt5.TIMEFRAME_M1,
        "m2":mt5.TIMEFRAME_M2,   "2m":mt5.TIMEFRAME_M2,
        "m3":mt5.TIMEFRAME_M3,   "3m":mt5.TIMEFRAME_M3,
        "m4":mt5.TIMEFRAME_M4,   "4m":mt5.TIMEFRAME_M4,
        "m5":mt5.TIMEFRAME_M5,   "5m":mt5.TIMEFRAME_M5,
        "m6":mt5.TIMEFRAME_M6,   "6m":mt5.TIMEFRAME_M6,
        "m10":mt5.TIMEFRAME_M10, "10m":mt5.TIMEFRAME_M10,
        "m12":mt5.TIMEFRAME_M12, "12m":mt5.TIMEFRAME_M12,
        "m15":mt5.TIMEFRAME_M15, "15m":mt5.TIMEFRAME_M15,
        "m20":mt5.TIMEFRAME_M20, "20m":mt5.TIMEFRAME_M20,
        "m30":mt5.TIMEFRAME_M30, "30m":mt5.TIMEFRAME_M30,
        "h1":mt5.TIMEFRAME_H1,   "1h":mt5.TIMEFRAME_H1,
        "h2":mt5.TIMEFRAME_H2,   "2h":mt5.TIMEFRAME_H2,
        "h3":mt5.TIMEFRAME_H3,   "3h":mt5.TIMEFRAME_H3,
        "h4":mt5.TIMEFRAME_H4,   "4h":mt5.TIMEFRAME_H4,
        "h6":mt5.TIMEFRAME_H6,   "6h":mt5.TIMEFRAME_H6,
        "h8":mt5.TIMEFRAME_H8,   "8h":mt5.TIMEFRAME_H8,
        "h12":mt5.TIMEFRAME_H12, "12h":mt5.TIMEFRAME_H12,
        "d1":mt5.TIMEFRAME_D1,   "1d":mt5.TIMEFRAME_D1,
        "w1":mt5.TIMEFRAME_W1,   "1w":mt5.TIMEFRAME_W1,
        "mn1":mt5.TIMEFRAME_MN1, "1mo":mt5.TIMEFRAME_MN1,
    }
    key = tf.lower().strip()
    if key not in TF_MAP:
        raise ValueError(f"Unknown MT5 timeframe '{tf}'. Valid: {sorted(TF_MAP.keys())}")
    return TF_MAP[key]


class MT5Connection:
    """
    Context manager for MT5 terminal connection.

    Usage
    -----
    with MT5Connection() as conn:
        df = conn.fetch_rates("EURUSD", "H1", n_bars=5000)

    Parameters
    ----------
    login    : MT5 account number (optional — uses terminal active account)
    password : account password (optional)
    server   : broker server name (optional)
    path     : path to MT5 terminal64.exe (optional)
    timeout  : connection timeout ms (default 10000)
    """

    def __init__(self, login=None, password=None, server=None,
                 path=None, timeout=10000):
        _require_mt5()
        self.login    = login
        self.password = password
        self.server   = server
        self.path     = path
        self.timeout  = timeout
        self._connected = False

    def connect(self):
        import MetaTrader5 as mt5
        kwargs = {"timeout": self.timeout}
        if self.path:
            kwargs["path"] = self.path
        if self.login and self.password and self.server:
            kwargs.update({"login":self.login,
                           "password":self.password,
                           "server":self.server})
        if not mt5.initialize(**kwargs):
            err = mt5.last_error()
            raise ConnectionError(
                f"MT5 initialisation failed: {err}\n"
                "Make sure MetaTrader5 terminal is running and logged in."
            )
        self._connected = True
        info = mt5.terminal_info()
        acct = mt5.account_info()
        msg = f"✅  MT5 Connected — {acct.login} @ {acct.server}"
        logger.info(msg)
        try:
            log_to_dashboard(msg, level="info")
        except Exception:
            pass
        logger.info("Terminal: %s  build %s", info.name, info.build)
        logger.info("Balance: %s %.2f", acct.currency, acct.balance)
        return True

    def disconnect(self):
        if self._connected:
            import MetaTrader5 as mt5
            mt5.shutdown()
            self._connected = False

    def is_connected(self) -> bool:
        if not self._connected:
            return False
        try:
            import MetaTrader5 as mt5
            if mt5.terminal_info() is None:
                self._connected = False
                return False
            return True
        except Exception as e:
            self._connected = False
            logger.warning("MT5 health check failed: %s", e)
            return False

    def __enter__(self):
        self.connect()
        return self

    def __exit__(self, *args):
        self.disconnect()

    def symbol_info(self, symbol: str) -> Dict[str, Any]:
        import MetaTrader5 as mt5
        info = mt5.symbol_info(symbol)
        if info is None:
            if not mt5.symbol_select(symbol, True):
                raise ValueError(
                    f"Symbol '{symbol}' not found on this broker.\n"
                    "Check exact symbol name in MT5 MarketWatch."
                )
            info = mt5.symbol_info(symbol)
            if info is None:
                raise ValueError(f"Symbol '{symbol}' unavailable after selection.")
        return {
            "symbol":          info.name,
            "description":     info.description,
            "digits":          info.digits,
            "tick_size":       info.trade_tick_size,
            "tick_value":      info.trade_tick_value,
            "contract_size":   info.trade_contract_size,
            "spread":          info.spread,
            "currency_base":   info.currency_base,
            "currency_profit": info.currency_profit,
            "point":           info.point,
            "volume_min":      info.volume_min,
            "volume_max":      info.volume_max,
            "volume_step":     info.volume_step,
        }

    def account_info(self) -> Dict[str, Any]:
        import MetaTrader5 as mt5
        acct = mt5.account_info()
        if acct is None:
            raise RuntimeError("Could not retrieve account info.")
        return {
            "login":       acct.login,
            "server":      acct.server,
            "name":        acct.name,
            "currency":    acct.currency,
            "balance":     acct.balance,
            "equity":      acct.equity,
            "margin":      acct.margin,
            "free_margin": acct.margin_free,
            "leverage":    acct.leverage,
            "profit":      acct.profit,
        }

    def fetch_rates(self, symbol, timeframe, n_bars=5000,
                    start=None, end=None):
        import MetaTrader5 as mt5
        tf_const = get_mt5_timeframe(timeframe)
        if not mt5.symbol_select(symbol, True):
            raise ValueError(f"Cannot select symbol '{symbol}'.")
        if start and end:
            rates = mt5.copy_rates_range(symbol, tf_const, start, end)
        elif start:
            rates = mt5.copy_rates_from(symbol, tf_const, start, n_bars)
        else:
            rates = mt5.copy_rates_from_pos(symbol, tf_const, 0, n_bars)
        if rates is None or len(rates) == 0:
            raise ValueError(
                f"MT5 returned no data for '{symbol}' [{timeframe}]. "
                f"Error: {mt5.last_error()}"
            )
        return rates

    def symbol_info_tick(self, symbol: str) -> dict:
        """
        Returns the latest bid/ask tick for a symbol.
        Used by the dashboard market-watch live tick feed.
        Returns {} on failure (caller should handle gracefully).
        """
        import MetaTrader5 as mt5
        tick = mt5.symbol_info_tick(symbol)
        if tick is None:
            return {}
        return {
            "symbol": symbol,
            "bid":    tick.bid,
            "ask":    tick.ask,
            "last":   tick.last,
            "volume": tick.volume,
            "time":   tick.time,
        }

    def available_symbols(self, filter_str: str = "") -> list:
        import MetaTrader5 as mt5
        symbols = mt5.symbols_get(filter_str) if filter_str else mt5.symbols_get()
        return [s.name for s in symbols] if symbols else []

    def last_closed_bar(self, symbol: str, timeframe: str) -> dict:
        rates = self.fetch_rates(symbol, timeframe, n_bars=2)
        bar   = rates[0]
        return {
            "time":   datetime.utcfromtimestamp(bar["time"]),
            "open":   float(bar["open"]),
            "high":   float(bar["high"]),
            "low":    float(bar["low"]),
            "close":  float(bar["close"]),
            "volume": float(bar["tick_volume"]),
        }
