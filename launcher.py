"""
CEO Engine Launcher
===================
Windows system tray application.
- Starts the Flask dashboard server on port 5000
- Opens the browser to http://localhost:5000 automatically
- Sits in the system tray with right-click menu
- Handles clean shutdown

Built with pystray + threading so it works as a PyInstaller .exe
"""

import os
import sys
import threading
import webbrowser
import time
import subprocess
import logging
from pathlib import Path

# ── Resolve paths relative to the .exe (PyInstaller) or script ──────────────
if getattr(sys, 'frozen', False):
    # Running as compiled .exe — PyInstaller sets sys._MEIPASS
    BASE_DIR = Path(sys.executable).parent
    INTERNAL = Path(sys._MEIPASS)
else:
    BASE_DIR  = Path(__file__).parent
    INTERNAL  = BASE_DIR

ICON_PATH    = INTERNAL / "installer" / "assets" / "icon.ico"
CONFIG_PATH  = BASE_DIR / "ceo_engine_config.json"
LOG_PATH     = BASE_DIR / "ceo_launcher.log"
PORT         = 5000
DASHBOARD_URL = f"http://localhost:{PORT}"

logging.basicConfig(
    filename=str(LOG_PATH),
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)


# ── Dashboard server thread ──────────────────────────────────────────────────

_dashboard_started = threading.Event()
_dashboard_error: str = ""


def _run_dashboard():
    global _dashboard_error
    try:
        # Add project root to path so imports resolve
        sys.path.insert(0, str(INTERNAL))
        os.chdir(str(BASE_DIR))   # working dir = install folder (config files live here)

        from ceo_engine_mt5.dashboard import app, state, set_journal
        import ceo_engine_mt5.dashboard as dash

        # Auto-connect journal if config exists
        try:
            import json
            with open(CONFIG_PATH) as f:
                cfg = json.load(f)
            journal = cfg.get("journal", "ceo_journal.db")
            set_journal(str(BASE_DIR / journal))
        except Exception:
            pass

        # Try MT5 auto-detect at startup — pre-populate config if not set
        _try_mt5_predetect(CONFIG_PATH)

        state.add_log("CEO Engine dashboard started", level="info")
        _dashboard_started.set()
        logger.info("Dashboard starting on port %d", PORT)

        app.run(host="127.0.0.1", port=PORT, debug=False, use_reloader=False)

    except Exception as e:
        _dashboard_error = str(e)
        _dashboard_started.set()   # unblock main thread
        logger.exception("Dashboard failed to start")


def start_dashboard_thread():
    t = threading.Thread(target=_run_dashboard, daemon=True, name="dashboard")
    t.start()
    _dashboard_started.wait(timeout=15)
    if _dashboard_error:
        raise RuntimeError(f"Dashboard failed: {_dashboard_error}")


# ── System tray ──────────────────────────────────────────────────────────────

def build_tray():
    try:
        import pystray
        from PIL import Image as PILImage
    except ImportError:
        # pystray not available — just open browser and block
        logger.warning("pystray not installed — running headless")
        webbrowser.open(DASHBOARD_URL)
        threading.Event().wait()   # block forever (daemon threads keep running)
        return

    # Load icon
    try:
        icon_img = PILImage.open(str(ICON_PATH))
    except Exception:
        # Fallback: generate a minimal orange square
        icon_img = PILImage.new("RGBA", (64, 64), "#f59e0b")

    def on_open(icon, item):
        webbrowser.open(DASHBOARD_URL)

    def on_quit(icon, item):
        logger.info("User quit from tray")
        icon.stop()
        # Give dashboard thread a moment to clean up
        time.sleep(0.5)
        os._exit(0)

    def on_backtest(icon, item):
        """Open backtest tab directly."""
        webbrowser.open(DASHBOARD_URL + "/#backtest")

    menu = pystray.Menu(
        pystray.MenuItem("Open CEO Engine",   on_open,  default=True),
        pystray.MenuItem("Backtest Tab",       on_backtest),
        pystray.Menu.SEPARATOR,
        pystray.MenuItem("Quit",               on_quit),
    )

    tray = pystray.Icon(
        name="CEO Engine",
        icon=icon_img,
        title="CEO Engine — running",
        menu=menu,
    )
    return tray


# ── Entry point ──────────────────────────────────────────────────────────────

def _try_mt5_predetect(config_path: Path):
    """
    On Windows, attempt a silent MT5 connection at startup.
    If successful and no config exists yet, write a minimal config
    pre-populated with the active account's login, server, and balance.
    This means the user can launch the app, MT5 already open, and the
    wizard Step 0 is already filled — just click through to confirm.
    """
    import sys
    if sys.platform != "win32":
        return
    try:
        import MetaTrader5 as mt5, json
        if not mt5.initialize():
            mt5.shutdown()
            return
        acct = mt5.account_info()
        if acct is None:
            mt5.shutdown()
            return
        mt5.shutdown()

        # Only write if no config exists yet (don't overwrite user settings)
        if config_path.exists():
            return

        is_demo = "demo" in acct.server.lower()
        cfg = {
            "mt5_login":      acct.login,
            "mt5_server":     acct.server,
            "mt5_password":   None,           # never stored — connect by active session
            "account_type":   "demo" if is_demo else "personal",
            "account_size":   round(acct.balance, 2),
            "symbols":        ["XAUUSD", "GBPUSD"],
            "tf":             "H1",
            "risk_pct":       1.0,
            "min_quality":    60,
            "confluence_mode": "sweep",
            "auto_trade":     False,
            "sessions":       ["london", "new_york"],
            "journal":        "ceo_journal.db",
            "dashboard_port": 5000,
            "daily_loss_pct": 5.0,
            "max_dd_pct":     10.0,
        }
        with open(config_path, "w") as f:
            json.dump(cfg, f, indent=2)
        try:
            os.chmod(config_path, 0o600)
        except Exception:
            pass
        logger.info("MT5 pre-detected: account %s @ %s — config written", acct.login, acct.server)
    except Exception as e:
        logger.debug("MT5 pre-detect skipped: %s", e)


def main():
    logger.info("CEO Engine launcher starting — BASE_DIR=%s", BASE_DIR)

    # Start dashboard server in background thread
    try:
        start_dashboard_thread()
        logger.info("Dashboard ready at %s", DASHBOARD_URL)
    except RuntimeError as e:
        # Show error dialog on Windows
        try:
            import ctypes
            ctypes.windll.user32.MessageBoxW(
                0,
                f"CEO Engine failed to start:\n\n{e}\n\nCheck ceo_launcher.log for details.",
                "CEO Engine — Error",
                0x10   # MB_ICONERROR
            )
        except Exception:
            print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    # Small delay then open browser
    def open_browser():
        time.sleep(1.2)
        webbrowser.open(DASHBOARD_URL)
        logger.info("Browser opened")

    threading.Thread(target=open_browser, daemon=True).start()

    # Start tray (blocks until quit)
    tray = build_tray()
    if tray:
        logger.info("Tray icon active")
        tray.run()
    else:
        threading.Event().wait()


if __name__ == "__main__":
    main()
