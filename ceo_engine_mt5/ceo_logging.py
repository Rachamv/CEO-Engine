"""
The CEO Protocol — Centralized Logging Setup
=================================================
Single place that configures Python's `logging` module for the whole
package. Every other module gets a logger the same way:

    from ceo_engine_mt5.ceo_logging import get_logger
    logger = get_logger(__name__)

    logger.info("RiskEngine initialised")
    logger.warning("Journal write failed: %s", e)
    logger.error("Order send failed", exc_info=True)

Why this exists
----------------
Before this module, the live trading loop (`mt5_live.py`) had ~24
`except Exception: pass` blocks with no record of what went wrong, and
no file the operator could check after the fact — only whatever scrolled
past in the terminal. `print()` is still used throughout the codebase for
direct, intentional console UI (backtest result tables, startup banners,
status lines) — that's a deliberate choice and stays. This module is for
*diagnostics*: anything that represents a failure, a fallback being
taken, or state worth being able to reconstruct after the process exits.

Usage
-----
- `get_logger(name)` — standard logger, writes to console (WARNING+) and
  to `ceo_engine.log` (INFO+) in the working directory by default.
- `configure(level=..., log_file=..., console_level=...)` — call once at
  the top of `run.py` / `mt5_live.py` entry points to customize before
  any `get_logger()` calls happen. Safe to call multiple times; only the
  first call takes effect unless `force=True`.
"""

import logging
import sys
from logging.handlers import RotatingFileHandler
from typing import Optional

_CONFIGURED = False
DEFAULT_LOG_FILE = "ceo_engine.log"
_FORMAT = "%(asctime)s  %(levelname)-8s  %(name)-22s  %(message)s"
_DATEFMT = "%Y-%m-%d %H:%M:%S"


def configure(
    level: int = logging.INFO,
    console_level: int = logging.WARNING,
    log_file: Optional[str] = DEFAULT_LOG_FILE,
    force: bool = False,
) -> None:
    """
    Configure the root 'ceo_engine' logger tree once.

    level         : minimum level written to the log file
    console_level : minimum level echoed to stderr (kept high by default
                    so console output isn't duplicated with the many
                    intentional print()-based status lines elsewhere)
    log_file      : path to the rotating log file, or None to disable
                    file logging entirely (console only)
    force         : reconfigure even if already configured
    """
    global _CONFIGURED
    if _CONFIGURED and not force:
        return

    root = logging.getLogger("ceo_engine")
    root.setLevel(level)
    root.handlers.clear()

    formatter = logging.Formatter(_FORMAT, datefmt=_DATEFMT)

    console = logging.StreamHandler(sys.stderr)
    console.setLevel(console_level)
    console.setFormatter(formatter)
    root.addHandler(console)

    if log_file:
        try:
            file_handler = RotatingFileHandler(
                log_file, maxBytes=5_000_000, backupCount=3, encoding="utf-8"
            )
            file_handler.setLevel(level)
            file_handler.setFormatter(formatter)
            root.addHandler(file_handler)
        except OSError as e:
            # Can't write a log file (read-only fs, permissions, etc.) —
            # fall back to console-only rather than crashing on startup.
            console.setLevel(min(console_level, level))
            root.warning("Could not open log file %r (%s) — logging to console only.",
                         log_file, e)

    root.propagate = False
    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    """
    Return a logger namespaced under 'ceo_engine.<name>'. Configures
    sane defaults automatically on first use if configure() was never
    called explicitly (e.g. when a module is used as a library, not via
    run.py/mt5_live.py).
    """
    if not _CONFIGURED:
        configure()
    short_name = name.rsplit(".", 1)[-1]
    return logging.getLogger(f"ceo_engine.{short_name}")


def log_to_dashboard(message: str, level: str = "info") -> None:
    """
    Convenience helper: emit to the normal logger and also forward
    a short message to the running dashboard (if available).

    This is intentionally best-effort: if the dashboard isn't running
    or the import fails, the function silently falls back to logging
    only so it never raises during normal execution.
    """
    # Normal logging first
    try:
        logger = get_logger(__name__)
        lvl = level.lower()
        if lvl == "debug":
            logger.debug(message)
        elif lvl == "warning":
            logger.warning(message)
        elif lvl == "error":
            logger.error(message)
        elif lvl == "critical":
            logger.critical(message)
        else:
            logger.info(message)
    except Exception:
        # If logging itself fails for any reason, silently continue.
        pass

    # Try to forward to the dashboard UI (best-effort)
    try:
        from ceo_engine_mt5.dashboard import add_log

        add_log(message, level=lvl if 'lvl' in locals() else level)
    except Exception:
        # Dashboard not running or import error — ignore.
        pass
