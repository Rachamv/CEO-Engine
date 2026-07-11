# The CEO Protocol — Python / MT5 Edition

A multi-model liquidity-sweep trading system: 16 signal models, a full
vectorised backtester, walk-forward validation, an MT5 live monitor, and
an optional auto-trading layer with prop-firm rule enforcement — all
sharing the same signal/indicator code so backtest and live can never
quietly diverge.

This README documents the **full** package (24 modules). Earlier versions
of this file only covered the Phase 1 backtester — that gap has been
closed below.

> **Status note:** this is trading software that can place real orders on
> a real account when `--auto-trade` is used. Read [Safety & Auto-Trading](#safety--auto-trading)
> before pointing it at anything other than a demo account.

---

## Table of contents

1. [Architecture at a glance](#architecture-at-a-glance)
2. [Installation](#installation)
3. [Quickstart](#quickstart)
4. [The 16 signal models](#the-16-signal-models)
5. [Sessions & "trade anytime"](#sessions--trade-anytime)
6. [Risk Engine & position sizing](#risk-engine--position-sizing)
7. [Funded Account Guard](#funded-account-guard)
8. [Safety & Auto-Trading](#safety--auto-trading)
9. [Pattern detection layers](#pattern-detection-layers)
10. [News filter](#news-filter)
11. [Multi-timeframe stack](#multi-timeframe-stack)
12. [Performance feedback loop](#performance-feedback-loop)
13. [Walk-forward validation](#walk-forward-validation)
14. [Output: journal, alerts, dashboard, HTML report](#output-journal-alerts-dashboard-html-report)
15. [Full CLI reference](#full-cli-reference)
16. [Programmatic use (as a library)](#programmatic-use-as-a-library)
17. [File manifest](#file-manifest)
18. [Known limitations](#known-limitations)

---

## Architecture at a glance

```
                       ┌────────────────────────────────────────┐
                       │              data.py                   │
                       │  yfinance / ccxt / csv / MT5  → OHLCV  │
                       └───────────────────┬────────────────────┘
                                           ▼
                       ┌────────────────────────────────────────┐
                       │            indicators.py               │
                       │  ATR, EMA, RSI, pivots, candle parts,  │
                       │  FVGs, displacement                    │
                       └───────────────────┬────────────────────┘
                                           ▼
        ┌───────────────────┬──────────────┴───────────────┬───────────────────┐
        ▼                   ▼                              ▼                   ▼
 candle_patterns.py    ceo_structure.py                patterns.py        signals.py
 (30+ candlestick      (BOS, double-BOS,            (H&S, triangles,    (16 sweep-based
  patterns)             order blocks, Fib)           flags, wedges...)   liquidity models)
        └───────────────────┴──────────────┬───────────────┴───────────────────┘
                                           ▼
                       ┌────────────────────────────────────────┐
                       │     session_filter.py / multi_tf.py    │
                       │   session gating, quality, MTF stack   │
                       └───────────────────┬────────────────────┘
                                           ▼
                ┌──────────────────────────┴──────────────────────────┐
                ▼                                                     ▼
        backtest.py                                          mt5_live.py
   (vectorised, walk-forward                         (live polling, MT5 candles,
    via walkforward.py)                               news_filter.py, perf feedback)
                ▼                                                     ▼
        visualise.py / report.py                             risk_engine.py
        (equity curves, HTML report)                                  ▼
                                                          funded_account_guard.py
                                                                      ▼
                                                            executor.py
                                                     (order placement / lifecycle)
                                                                      ▼
                                          journal.py · alerts.py · dashboard.py
```

The package is organized in four loosely-coupled phases. **Phases are
additive** — Phase 1 alone gives you a complete backtester; Phases 2-4
are opt-in via CLI flags and only get imported when you actually use them.

| Phase | What it adds | Files |
|---|---|---|
| **1 — Core** | OHLCV ingestion, indicators, the 16 sweep models, vectorised backtest, equity-curve charts | `data.py`, `indicators.py`, `signals.py`, `backtest.py`, `visualise.py` |
| **1.5 / 2.7** | Pure pattern-recognition layers (no signal logic — additive confluence) | `candle_patterns.py`, `ceo_structure.py`, `patterns.py` |
| **2** | Position sizing, the live order gate, prop-firm rule enforcement, MT5 order execution | `risk_engine.py`, `funded_account_guard.py`, `executor.py` |
| **3** | Session/news gating, multi-timeframe agreement | `session_filter.py`, `news_filter.py`, `multi_tf.py` |
| **4** | Trade journal, Telegram alerts, web dashboard, HTML reports, rolling performance feedback, walk-forward | `journal.py`, `alerts.py`, `dashboard.py`, `report.py`, `performance_monitor.py`, `walkforward.py` |

`mt5_connect.py` and `mt5_live.py` are the MT5-specific connection and
live-polling layers used by Phase 2-4 when running against a real terminal.

---

## Installation

```bash
pip install pandas numpy
```

Or, to install as an editable package (enables `pip install -e ".[dev]"`
for running the test suite, and makes the modules importable from
anywhere without manually adding the directory to `sys.path`):

```bash
pip install -e .                # core only
pip install -e ".[all]"         # every optional feature
pip install -e ".[data,charts]" # pick specific feature groups
```

That's the only hard requirement — Phase 1 backtesting against a CSV
works with nothing else. Everything below is opt-in, matched to the
feature you actually use:

| Feature | Extra package | Flag that triggers it |
|---|---|---|
| Yahoo Finance data | `yfinance` | `--source yfinance` (default) |
| Crypto exchange data | `ccxt` | `--source ccxt` |
| MT5 data / live monitor / auto-trade | `MetaTrader5` (Windows only) | `--source mt5`, `--live`, `--auto-trade` |
| Static chart PNGs | `matplotlib` | `--ceo-chart` |
| Interactive equity curve / HTML report | none — CDN-loaded JS (Chart.js) | `--html-report` |
| Web dashboard | `flask` | `--dashboard-port` |
| Telegram alerts, news calendar fetch | `requests` | `--telegram-token`, news filter (on by default in live mode) |

`MetaTrader5` only loads on Windows — on macOS/Linux, `executor.py` and
`mt5_connect.py` detect this and fall back to **simulation mode**
automatically (see [Safety & Auto-Trading](#safety--auto-trading)).

---

## Quickstart

**Backtest XAUUSD H1, all 16 models, trade anytime the market is open:**

```bash
python run.py --symbol XAUUSD --tf H1 --source yfinance \
    --start 2023-01-01 --out results/
```

**Backtest with confluence mode (only fire when ≥3 models agree) and an HTML report:**

```bash
python run.py --symbol XAUUSD EURUSD --tf H1 \
    --confluence --min-models 3 --min-quality 55 \
    --html-report --out results/
```

**Walk-forward validate the best model over 6 windows before trusting it:**

```bash
python run.py --symbol XAUUSD --tf H1 --walkforward --windows 6
```

**Live monitor on MT5 (signal alerts only, no orders placed):**

```bash
python run.py --symbol XAUUSD --tf M15 --source mt5 --live \
    --mt5-login 12345678 --mt5-password "***" --mt5-server "Broker-Live" \
    --sound --log signals.csv
```

**Live monitor with auto-trading, risk-based sizing, and a funded-account guard:**

```bash
python run.py --symbol XAUUSD --tf M15 --source mt5 --live \
    --auto-trade --risk-pct 0.5 --account-size 100000 \
    --daily-loss-pct 5 --max-dd-pct 10 \
    --journal ceo_journal.db --telegram-token "$CEO_TELEGRAM_TOKEN" --telegram-chat "$CEO_TELEGRAM_CHAT_ID"
```

---

## The 16 signal models

Every model is a "sweep" (liquidity grab past a confirmed swing point,
then rejection) combined with zero or more confluence filters. Model 0
(`LQ`) is the raw sweep — everything else adds a filter on top of it.

| # | Model | Trend | FVG | Volume | RSI | Displacement |
|---|---|---|---|---|---|---|
| 0 | LQ | | | | | |
| 1 | LQ + Trend | ✓ | | | | |
| 2 | LQ + FVG | | ✓ | | | |
| 3 | LQ + Volume | | | ✓ | | |
| 4 | LQ + RSI | | | | ✓ | |
| 5 | LQ + Displacement | | | | | ✓ |
| 6 | LQ + Trend + FVG | ✓ | ✓ | | | |
| 7 | LQ + Trend + Volume | ✓ | | ✓ | | |
| 8 | LQ + Trend + RSI | ✓ | | | ✓ | |
| 9 | LQ + Trend + Displacement | ✓ | | | | ✓ |
| 10 | LQ + FVG + Volume | | ✓ | ✓ | | |
| 11 | LQ + FVG + RSI | | ✓ | | ✓ | |
| 12 | LQ + FVG + Displacement | | ✓ | | | ✓ |
| 13 | LQ + Volume + RSI | | | ✓ | ✓ | |
| 14 | LQ + Trend + FVG + Volume | ✓ | ✓ | ✓ | | |
| 15 | LQ + All Filters | ✓ | ✓ | ✓ | ✓ | ✓ |

`--model N` runs one model. `--confluence --min-models K` fires only when
at least `K` of the 16 agree on direction in the same bar. `--ceo-only`
additionally requires the full CEO structure sequence (BOS → OB/Fib zone
→ candle confirmation) from `ceo_structure.py`, the strictest gate
available.

---

## Sessions & "trade anytime"

`--sessions` controls which hours of the day a signal is allowed to fire.
**The default is `all`** — trade any hour the market is actually open.
Real market closure (Saturday, and Sunday before 05:00 UTC) is still
blocked regardless of this setting; that's the market being shut, not a
session preference.

```bash
--sessions all                      # default — every hour the market is open
--sessions london new_york          # restrict to specific named windows
```

Named windows (UTC), defined once in `session_filter.py` and imported by
every other module so they can't drift apart:

| Session | Window (UTC) | Min. signal quality required |
|---|---|---|
| `asian` | 00:00 – 09:00 | 70 |
| `pre_london` | 05:00 – 07:00 | 55 |
| `london` | 07:00 – 16:00 | 50 |
| `overlap` (London/NY) | 12:00 – 16:00 | 45 |
| `new_york` | 12:00 – 21:00 | 50 |
| `post_ny` | 21:00 – 23:59 | 70 |

The quality bar is *always* applied per-session — `--sessions all` removes
the time-of-day restriction, it does not remove the quality filter. Thin
sessions (Asian, post-NY) still require a cleaner signal (quality ≥ 70)
than London/NY (≥ 50).

---

## Risk Engine & position sizing

`risk_engine.py`'s `RiskEngine.evaluate()` is the single gate every live
signal passes through before a lot size is returned (or the trade is
blocked). It checks, in order:

1. **Model gate** — has this model's live/backtested win rate degraded
   below a usable expectancy (set via `--min-consistency`)?
2. **Spread gate** — is the current spread too wide relative to the SL
   distance (`max_spread_ratio`) or in absolute pips (`max_spread_pips_abs`)?
3. **Session gate** — `SessionFilter.is_session_allowed()` (see above).
4. **Position sizing** — `PositionSizer.calculate()` converts
   `risk_pct × account balance` and the SL distance into a lot size using
   the broker's actual `tick_value` / `tick_size` / `point` from
   `mt5.symbol_info()` — never an estimate, when MT5 is the source.

`--risk-pct 0` (the default) disables position sizing — the engine still
runs the other gates and reports what *would* have been sized, useful for
dry-running the risk logic before committing real capital.

---

## Funded Account Guard

`funded_account_guard.py`'s `FundedAccountGuard` is a general-purpose
prop-firm rule enforcer, checked via `pre_trade_check()` before every
order and `record_closed_trade()` after every close. It tracks, per day
and account-wide:

- Daily loss limit (hard stop before the limit, with a configurable buffer)
- Maximum trailing drawdown
- Consistency score (warns when one day's profit approaches X% of total —
  **advisory only, does not block the trade**, since you can't know future
  days' profit share in advance)
- Minimum trading days / minimum hold time / max concurrent open trades
- Correlated-symbol exposure (`CORRELATION_MAP` — e.g. EURUSD+GBPUSD count
  as doubled USD-negative exposure)

**Honesty note:** the module's docstring references named presets
("Blue Guardian, FTMO, The5ers, etc.") but as shipped, `PROP_FIRM_PRESETS`
only defines `"custom"`. There is no built-in FTMO/Blue Guardian preset
yet — configure the six parameters directly to match your specific
firm's rule sheet, or add named presets to that dict yourself.

```python
guard = FundedAccountGuard(preset="custom", **{
    "daily_loss_limit_pct": 5.0,
    "max_drawdown_pct": 10.0,
    "consistency_pct": 0.0,
    "min_hold_minutes": 0,
    "min_trading_days": 0,
    "max_daily_trades": 0,
    "require_sl": True,
    "max_open_trades": 0,
    "buffer_pct": 0.5,
})
```

---

## Safety & Auto-Trading

`executor.py`'s `Executor` is the only module that places real orders.
Lifecycle:

```
signal fires (bar close)
   → RiskEngine.evaluate()              lot size, or blocked
   → FundedAccountGuard.pre_trade_check()  allowed, or blocked
   → Executor.place_trade()             sends the order
   → Executor.manage_open_trades()      runs every bar — TP1/TP2 partials, SL trail
   → FundedAccountGuard.record_closed_trade()
```

**Simulation mode is automatic, not opt-in.** `Executor` checks
`sys.platform == "win32"` and whether `MetaTrader5` actually imports; on
any other platform (or if the package is missing) it forces
`self.simulation = True` and prints a notice — no orders are sent, P&L is
estimated locally instead. You can also force it explicitly:

```python
executor = Executor(connection=conn, risk_engine=risk_engine,
                     guard=guard, magic_number=20250101, simulation=True)
```

To actually place live orders you need all of: Windows, `MetaTrader5`
installed and logged in, `simulation=False` (or omitted), and `--auto-trade`
on the CLI. Every order carries a unique `magic_number` so the engine's
trades are identifiable and won't interfere with manual trades on the
same account.

**Run on a demo account first.** Nothing in this codebase enforces that
for you — it's a deliberate choice left to the operator.

---

## Pattern detection layers

These three layers are pure detection — they don't generate trade signals
on their own, they feed confluence/context into the 16 sweep models and
the live alert payload.

**`candle_patterns.py`** — 30+ vectorised candlestick patterns from the
enriched OHLC (no loops): dragonfly/gravestone/long-legged doji, hammer,
inverted hammer, hanging man, shooting star, spinning top, bullish/bearish
marubozu, engulfing, harami, harami cross, belt hold, meeting lines,
tweezers top/bottom, piercing line, dark cloud cover, morning/evening
star (+ doji variants), three white soldiers, three black crows, three
inside/outside up/down — plus `cp_bull_any` / `cp_bear_any` rollups.

**`ceo_structure.py`** — structural concepts that can't be replicated in
Pine Script's compute/state limits: Break of Structure (BOS) after a
sweep, double-BOS validation (two sequential breaks to confirm a
protected swing leg), order blocks, and Fibonacci zones anchored to
validated legs.

**`patterns.py`** — classical chart patterns from confirmed pivot points
(geometric, not curve-fit): Head & Shoulders / Inverse H&S, Double/Triple
Top/Bottom, Rising/Falling Wedge (reversal); Bull/Bear Flag, Ascending/
Descending/Symmetrical Triangle, Bull/Bear Pennant, Rectangle (continuation).

---

## News filter

`news_filter.py` blocks trading around high-impact economic releases in
live mode (on by default — disable with `--no-news-filter`, "not
recommended for funded accounts" per the module's own warning). Sources
are tried in order, first that works wins: ForexFactory JSON feed →
Investing.com scrape → FCS API (needs `--fcsapi-key`) → a manual override
list that always works offline. Default blocking window: 30 min before /
15 min after high-impact events (`--news-pre-mins`, `--news-post-mins`);
medium-impact events are off by default (`--news-medium` to enable).

---

## Multi-timeframe stack

`multi_tf.py` requires agreement across timeframes before a signal is
treated as high-quality, following the CEO method's top-down structure:
H4 bias → H1 intermediate structure → M15 entry timing → M5 fine-tuning.
Enable with `--mtf H4 H1 M15` (any subset/order). `--mtf-mode` controls
how agreement is scored: `bias` (HTF trend direction only), `sweep`
(sweep confirmed on multiple TFs), `ceo` (full structure sequence on
multiple TFs), or `cascade` (each TF must confirm in sequence,
highest bar).

---

## Performance feedback loop

`performance_monitor.py` is the live-vs-backtest reality check: it reads
the trade journal, computes rolling win rate over the last N trades
(`--perf-window`, default 10), and shrinks `RiskEngine.risk_pct` /
raises `RiskEngine.min_quality` when live performance falls below what
the backtest predicted — e.g. a model that backtested at 55% WR but is
running 30% WR over the last 10 live trades gets risk reduced before a
real drawdown compounds, not after. Triggers on a loss streak
(`--perf-loss-streak`, default 3) or a win-rate floor breach
(`--perf-wr-floor`, default 35%). Requires `--journal` and `--risk-pct` > 0.

---

## Walk-forward validation

`walkforward.py` splits the historical range into N equal windows
(`--windows`, default 6) and reruns the full backtest independently on
each one — answering "is the best model genuinely robust, or did it just
curve-fit the full sample?" `--wf-metric` picks what's compared across
windows (`Net R`, `Profit Factor`, `Win Rate`, `Avg R`); `--wf-min-trades`
excludes windows too thin to trust.

```bash
python run.py --symbol XAUUSD --tf H1 --walkforward --windows 6
```

---

## Output: journal, alerts, dashboard, HTML report

- **`journal.py`** — SQLite (`--journal path.db`) trade log with `trades`,
  `signals` (every signal fired, traded or not), and `daily_summary`
  tables. CSV export available on demand.
- **`alerts.py`** — Telegram alerts on signal, trade open, TP hits, SL hit,
  daily summary; attaches a chart PNG to signal alerts. Needs a bot via
  `@BotFather` — set `CEO_TELEGRAM_TOKEN` / `CEO_TELEGRAM_CHAT_ID` or pass
  `--telegram-token` / `--telegram-chat`.
- **`dashboard.py`** — local web dashboard (`--dashboard-port 8080`),
  open-trade status and floating P&L as the hero element, 4-second
  refresh. Preview with mock data: `python run_dashboard_preview.py`.
- **`report.py`** — self-contained HTML backtest report (`--html-report`):
  equity curve, full model comparison table, R-multiple distribution,
  win-rate-vs-profit-factor scatter, session breakdown, drawdown chart,
  annotated CEO candlestick chart, and walk-forward consistency if available.

---

## Full CLI reference

Run `python run.py --help` for the live, authoritative list — this is a
grouped summary.

**Data**
| Flag | Default | Notes |
|---|---|---|
| `--symbol` | *required* | One or more, e.g. `XAUUSD EURUSD GBPUSD` |
| `--tf` | `1h` | `M5 M15 M30 H1 H4 D1` |
| `--source` | `mt5` | MT5 only |
| `--exchange` | `binance` | (unused) |
| `--file` | – | (unused) |
| `--start` / `--end` | `2022-01-01` / today | |

**MT5 / Live**
| Flag | Default | Notes |
|---|---|---|
| `--live` | off | requires `--source mt5` |
| `--mt5-login/-password/-server` | – | |
| `--reconnect-wait` | `30` s | |
| `--bars` | `1000` | history bars pulled per poll |
| `--sound` | off | audible alert on signal |
| `--log` | – | CSV signal log path |
| `--poll` | `5` s | |
| `--model` | `0` | which of the 16 models to run live |

**Signal mode**
| Flag | Default |
|---|---|
| `--confluence` | off |
| `--ceo-only` | off — full CEO structure gate |
| `--min-models` | `3` |
| `--min-quality` | `50.0` |
| `--no-align-block` | off |

**HTF Bias**
`--htf`, `--no-htf`, `--htf-ema-fast 50`, `--htf-ema-slow 200`

**Indicators**
`--atr-len 14`, `--ema-fast 50`, `--ema-slow 200`, `--rsi-len 14`,
`--rsi-os 35`, `--rsi-ob 65`, `--pivot-len 5`, `--vol-mult 1.30`, `--pool-size 3`

**Sweep Detection**
`--max-depth 0.80`, `--min-rej 0.20`

**Session**
`--sessions all` (choices: `all london new_york asian overlap pre_london post_ny`) — see [Sessions](#sessions--trade-anytime)

**Risk / Backtest**
| Flag | Default |
|---|---|
| `--sl-mode` | `sweep` (`sweep \| atr`) |
| `--sl` / `--sl-buffer` / `--sl-max` | `1.50` / `0.10` / `3.00` |
| `--tp1` / `--tp2` / `--tp3` | `1.00` / `2.00` / `3.00` |
| `--maxbars` | `30` |
| `--commission` | `0.05` |
| `--spread-mode` | `fixed_r` (`fixed_r \| price`) |
| `--spread-points` | `0.0` |

**Phase 2 — Auto-trade + Risk**
| Flag | Default | Notes |
|---|---|---|
| `--auto-trade` | off | live mode only |
| `--risk-pct` | `0.0` | 0 = sizing disabled |
| `--account-size` | `10000.0` | |
| `--max-sl-pips` | `2000.0` | |
| `--min-consistency` | `0.0` | min walk-forward consistency fraction |
| `--daily-loss-pct` | `5.0` | |
| `--max-dd-pct` | `10.0` | |
| `--consistency-pct` | `0.0` | max single-day % of total profits, 0 = off |

**Phase 4 — Output**
`--journal path.db`, `--telegram-token`, `--telegram-chat`,
`--dashboard-port`, `--out dir`, `--no-charts`, `--no-csv`,
`--min-trades 10`, `--ceo-chart`, `--html-report`

**News Filter (live mode)**
`--no-news-filter`, `--news-pre-mins 30`, `--news-post-mins 15`,
`--news-medium`, `--fcsapi-key`

**Multi-Timeframe Stack (live mode)**
`--mtf H4 H1 M15`, `--mtf-mode bias` (`bias \| sweep \| ceo \| cascade`),
`--mtf-min-tfs`, `--mtf-min-score 40.0`

**Performance Feedback Loop (live mode)**
`--perf-feedback`, `--perf-window 10`, `--perf-loss-streak 3`, `--perf-wr-floor 35.0`

**Walk-Forward**
`--walkforward`, `--windows 6`, `--wf-metric "Net R"`
(`Net R \| Profit Factor \| Win Rate \| Avg R`), `--wf-min-trades 5`

---

## Programmatic use (as a library)

Every CLI flag maps 1:1 to a module-level function/class parameter — the
CLI is a thin wrapper, not a separate code path:

```python
from data import fetch_ohlcv
from indicators import calc_all
from signals import build_signals
from backtest import run_backtest, DEFAULT_BT_PARAMS

df = fetch_ohlcv("XAUUSD", tf="H1", source="yfinance", start="2023-01-01")
df = calc_all(df)
df = build_signals(df)

params = {**DEFAULT_BT_PARAMS, "session_filter": True, "active_sessions": ["all"]}
results = run_backtest(df, params)
```

```python
from walkforward import walk_forward, print_wf_report

wf = walk_forward(df, n_windows=6, min_trades=5)
print_wf_report(wf)
```

---

## File manifest

| File | Phase | Purpose |
|---|---|---|
| `data.py` | 1 | OHLCV fetch (yfinance/ccxt/csv/MT5) + cleaning |
| `indicators.py` | 1 | ATR, EMA, RSI, pivots, candle parts, FVGs, displacement |
| `signals.py` | 1 | The 16 sweep-based liquidity models, HTF bias |
| `backtest.py` | 1 | Vectorised backtest engine, session mask |
| `visualise.py` | 1 | Equity curve / drawdown charts |
| `candle_patterns.py` | 1.5 | 30+ candlestick pattern detectors |
| `ceo_structure.py` | 2.7 | BOS, double-BOS, order blocks, Fib zones |
| `patterns.py` | 3 | Classical geometric chart patterns |
| `session_filter.py` | 3 | Canonical session windows, gating, quality multipliers |
| `news_filter.py` | 3 | Economic calendar blocking |
| `multi_tf.py` | 3 | Multi-timeframe signal agreement |
| `risk_engine.py` | 2 | Position sizing, live order gate |
| `funded_account_guard.py` | 2 | Prop-firm rule enforcement |
| `executor.py` | 2 | MT5 order placement & lifecycle |
| `mt5_connect.py` | — | MT5 terminal connection manager |
| `mt5_live.py` | — | CLI entry point + `run_live()` orchestrator |
| `mt5_live_utils.py` | — | Shared live-monitor helpers (lot sizing, dedup, SL/TP calc) |
| `mt5_live_signals.py` | — | `check_symbol()` + risk/guard gates + signal routing |
| `mt5_live_session.py` | — | Component init, model registration, MTF handling, shutdown |
| `performance_monitor.py` | 4 | Rolling live performance → dynamic risk adjustment |
| `walkforward.py` | 4 | Walk-forward window validation |
| `journal.py` | 4 | SQLite trade/signal journal |
| `alerts.py` | 4 | Telegram alerts |
| `dashboard.py` | 4 | Local web dashboard |
| `report.py` | 4 | Self-contained HTML backtest report |
| `chart.py` | 4 | Public chart API (re-exports `chart_png`/`chart_lwc`) |
| `chart_theme.py` | 4 | Shared color theme + DataFrame-slicing helpers |
| `chart_png.py` | 4 | Static matplotlib PNG renderer |
| `chart_lwc.py` | 4 | Interactive Lightweight Charts HTML renderer (CDN-loaded JS, no Python extra needed) |
| `run.py` | — | CLI entry point, wires every phase together |
| `ceo_logging.py` | — | Centralized logging (`get_logger`/`configure`) |
| `tests/` | — | pytest suite (568 tests, 67% coverage) — see Testing below |

`mt5_live.py` and `chart.py` were each one large file until v2.3.0; both
are now thin public-facing entry points over their respective
implementation modules, split purely for maintainability (see CHANGELOG).

---

## Testing

```bash
pip install -e ".[dev,live,charts]"
pytest                       # 568 tests, ~25s
pytest --cov=ceo_engine_mt5 --cov-report=term-missing   # coverage breakdown
```

Coverage is concentrated on the trading logic — `signals.py` (97%),
`indicators.py` (98%), `backtest.py` (90%), `patterns.py` (83%),
`ceo_structure.py` (91%), `candle_patterns.py` (93%), `risk_engine.py`
(75%), `funded_account_guard.py` (73%), `alerts.py` (88%),
`mt5_live_signals.py` (89%) — including a pinned golden-reference trade
count for the core P&L simulation (`backtest.py::_simulate_model`), so
an accidental behavior change there fails a test immediately instead of
needing a manual diff.

`multi_tf.py` (81%) and the broker-confirmation paths in `executor.py`
(the retcode-checking around order placement/modification/close) are
tested by injecting a fake `mt5` module at import time and forcing
`MT5_AVAILABLE=True` (see `test_executor_broker_confirmation.py`) --
MetaTrader5 only ships wheels for Windows, so this is the only way to
exercise that code without a real MT5 terminal, on any platform.
`mt5_connect.py`, `mt5_live.py`, and `mt5_live_session.py`'s component
init/shutdown paths don't have that treatment yet and remain at 0-25%.
A CI workflow (`.github/workflows/tests.yml`) runs the full suite across
Python 3.9-3.13 on every push/PR.

---

## Known limitations

- `PROP_FIRM_PRESETS` currently only defines `"custom"` — no built-in
  FTMO/Blue Guardian/The5ers presets despite the module docstring
  referencing them. Configure the parameters directly for now.
- The consistency-rule check in `FundedAccountGuard` is advisory
  (warns) rather than a hard block — by design, since a single day's
  share of total profit can't be known with certainty until later days
  play out.
- `executor.py`'s simulation-mode P&L estimate is a local approximation,
  not a broker fill — treat simulation-mode numbers as directional, not
  exact.
- No tests cover MT5-connection-dependent code paths (live order
  placement, live polling) — see Testing above.
- Tick size is pulled directly from `mt5.symbol_info()` when the source
  is MT5; non-MT5 sources (yfinance/ccxt/csv) fall back to a
  price-magnitude estimate in `data.py`, which is approximate by nature.

---

*Questions, bugs, or a feature you want documented that isn't above?
Open an issue against this repo or just ask.*
# CEO-Engine
