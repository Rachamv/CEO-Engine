# -*- mode: python ; coding: utf-8 -*-
"""
CEO Engine — PyInstaller build spec
=====================================
Produces a single-folder distribution (onedir) with:
  - launcher.exe          (system tray entry point — what the user launches)
  - run.exe               (CLI backtest / live runner)
  - All Python deps bundled (pandas, numpy, flask, etc.)
  - MT5 bindings included but only active on Windows
  - Static assets (icon, templates) bundled

Build command (run from project root on Windows):
    pip install pyinstaller pystray pillow
    pyinstaller ceo_engine.spec

Output: dist/CEOEngine/
"""

import sys
from pathlib import Path

ROOT = Path(SPECPATH)

# ── Shared analysis ──────────────────────────────────────────────────────────

common_hiddenimports = [
    # Flask internals
    "flask", "flask.json", "flask.templating",
    "werkzeug", "werkzeug.serving", "werkzeug.routing",
    "jinja2", "jinja2.ext",
    # Data
    "pandas", "pandas.core.arrays", "pandas.io.formats",
    "numpy", "numpy.core", "numpy.lib",
    # MT5 (Windows only — import guarded in code)
    "MetaTrader5",
    # CEO Engine modules
    "ceo_engine_mt5",
    "ceo_engine_mt5.run",
    "ceo_engine_mt5.dashboard",
    "ceo_engine_mt5.backtest",
    "ceo_engine_mt5.signals",
    "ceo_engine_mt5.risk_engine",
    "ceo_engine_mt5.executor",
    "ceo_engine_mt5.funded_account_guard",
    "ceo_engine_mt5.journal",
    "ceo_engine_mt5.alerts",
    "ceo_engine_mt5.mt5_connect",
    "ceo_engine_mt5.mt5_live",
    "ceo_engine_mt5.mt5_live_session",
    "ceo_engine_mt5.mt5_live_signals",
    "ceo_engine_mt5.walkforward",
    "ceo_engine_mt5.ceo_logging",
    "ceo_engine_mt5.session_filter",
    "ceo_engine_mt5.indicators",
    "ceo_engine_mt5.ceo_structure",
    "ceo_engine_mt5.multi_tf",
    "ceo_engine_mt5.patterns",
    "ceo_engine_mt5.chart",
    "ceo_engine_mt5.journal",
    "ceo_engine_mt5.performance_monitor",
    # System tray
    "pystray", "pystray._win32",
    "PIL", "PIL.Image", "PIL.ImageDraw",
    # Networking / alerts
    "requests", "urllib3",
    # Crypto / ccxt optional
    "sqlite3",
    # Optional deps
    "plotly", "matplotlib",
]

# Data files to bundle: (source_path, dest_folder_in_bundle)
datas = [
    # Icon and installer assets
    (str(ROOT / "installer" / "assets"), "installer/assets"),
    # README and quickstart (shown in installer)
    (str(ROOT / "README.md"),            "."),
    (str(ROOT / "QUICKSTART.md"),        "."),
    # Dashboard HTML/CSS/JS template -- dashboard.py reads this from disk
    # at runtime (via sys._MEIPASS when frozen), it's not compiled into
    # the PYZ archive like regular .py source, so it must be bundled
    # explicitly here or the dashboard fails to start with a clear
    # FileNotFoundError pointing back at this comment.
    (str(ROOT / "ceo_engine_mt5" / "templates"), "ceo_engine_mt5/templates"),
]

common_excludes = [
    "tkinter", "test", "unittest",
    "email", "html", "http",   # not http — flask needs it, only exclude client-side
    "xmlrpc", "curses",
    "IPython", "ipykernel", "jupyter",
]

# ── Launcher (tray app) ──────────────────────────────────────────────────────

launcher_analysis = Analysis(
    [str(ROOT / "launcher.py")],
    pathex=[str(ROOT)],
    binaries=[],
    datas=datas,
    hiddenimports=common_hiddenimports,
    hookspath=[],
    runtime_hooks=[],
    excludes=common_excludes,
    noarchive=False,
)

launcher_pyz = PYZ(launcher_analysis.pure)

launcher_exe = EXE(
    launcher_pyz,
    launcher_analysis.scripts,
    [],
    exclude_binaries=True,
    name="CEOEngine",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,           # no terminal window — tray-only app
    icon=str(ROOT / "installer" / "assets" / "icon.ico"),
    version=str(ROOT / "installer" / "assets" / "version_info.txt") if
            (ROOT / "installer" / "assets" / "version_info.txt").exists() else None,
)

# ── CLI runner (run.exe) ─────────────────────────────────────────────────────

runner_analysis = Analysis(
    [str(ROOT / "run.py")],
    pathex=[str(ROOT)],
    binaries=[],
    datas=datas,
    hiddenimports=common_hiddenimports,
    hookspath=[],
    runtime_hooks=[],
    excludes=common_excludes,
    noarchive=False,
)

runner_pyz = PYZ(runner_analysis.pure)

runner_exe = EXE(
    runner_pyz,
    runner_analysis.scripts,
    [],
    exclude_binaries=True,
    name="ceo-run",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=True,            # keep terminal for CLI backtest output
    icon=str(ROOT / "installer" / "assets" / "icon.ico"),
)

# ── Collect everything into one folder ──────────────────────────────────────

coll = COLLECT(
    # Launcher
    launcher_exe,
    launcher_analysis.binaries,
    launcher_analysis.zipfiles,
    launcher_analysis.datas,
    # CLI runner
    runner_exe,
    runner_analysis.binaries,
    runner_analysis.zipfiles,
    runner_analysis.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name="CEOEngine",        # dist/CEOEngine/
)
