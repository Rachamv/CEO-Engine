# CEO Engine MT5 Edition — Quick Start

## Setup (one time)

```bash
python -m pip install -e .
```

Make sure MetaTrader5 terminal is **open and logged in** before running.

---

## Backtest using MT5 data

```bash
# EURUSD H1 — last 5000 bars from your broker
python run.py --symbol EURUSD --tf H1 --source mt5

# XAUUSD M15 from 2023
python run.py --symbol XAUUSD --tf M15 --source mt5 --start 2023-01-01

# US30 H4
python run.py --symbol US30 --tf H4 --source mt5 --start 2022-01-01

# NAS100 with confluence mode
python run.py --symbol NAS100 --tf H1 --source mt5 --confluence --min-models 4

# GBPUSD with tighter risk
python run.py --symbol GBPUSD --tf H1 --source mt5 --sl 1.2 --tp3 2.5
```

---

## Live signal monitor

```bash
# Monitor EURUSD on H1 — prints signal on every new bar close
python run.py --symbol EURUSD --tf H1 --source mt5 --live

# Monitor multiple symbols
python mt5_live.py --symbol EURUSD GBPUSD XAUUSD --tf H1

# With sound + CSV log
python mt5_live.py --symbol EURUSD --tf M15 --sound --log signals.csv

# Confluence mode (fire when 3+ models agree)
python mt5_live.py --symbol XAUUSD --tf H1 --confluence --min-models 3

# No HTF filter (raw signals only)
python mt5_live.py --symbol EURUSD --tf M15 --no-htf
```

---

## What live output looks like

```
════════════════════════════════════════════════════════════
[2024-01-15 14:00:00 UTC]  EURUSD H1  ▲ LONG
════════════════════════════════════════════════════════════
  Model      : LQ + Trend
  Quality    : 72.5 / 100
  Regime     : Trend Up
  Alignment  : Aligned
  HTF Bias   : Bullish (H4)
  Confluence : 5 / 16 models
  ─────────────────────────────
  Entry ref  : 1.09245
  SL    ref  : 1.08980  (-26.5 pips)
  TP1   ref  : 1.09510  (+26.5 pips)
  TP2   ref  : 1.09775  (+53.0 pips)
  TP3   ref  : 1.10040  (+79.5 pips)
════════════════════════════════════════════════════════════
```

Entry/SL/TP are **reference levels** — not live orders.
Place trades manually in MT5 using these as your guide.

---

## Symbol name formats by broker

Symbol names vary by broker. Check your MT5 MarketWatch for exact names.

| Instrument | Common formats |
|------------|---------------|
| EURUSD | `EURUSD` `EURUSD.m` `EURUSDm` |
| Gold | `XAUUSD` `GOLD` `XAUUSD.m` |
| US30 (Dow) | `US30` `DJ30` `DJIA` `US30Cash` |
| NAS100 | `NAS100` `USTEC` `NAS100m` |
| Brent Oil | `XBRUSD` `OILm` `BRENT` |
| Bitcoin | `BTCUSD` `Bitcoin` `BTCUSD.m` |

---

## MT5 connection options

```bash
# Auto-connect (uses terminal's active login)
python run.py --symbol EURUSD --tf H1 --source mt5

# Explicit login (if terminal has multiple accounts)
python run.py --symbol EURUSD --tf H1 --source mt5 \
  --mt5-login 12345678 --mt5-password MyPass --mt5-server BrokerName-Live
```

---

## Backtest output folder

```
results_EURUSD_H1/
├── results_summary.csv       Model performance table
├── all_trades.csv            Full trade log (all 16 models)
├── chart_dashboard.png       Full overview
├── chart_comparison.png      Net R + Win Rate
├── chart_equity.png          Equity curves
├── chart_wr_pf.png           Win Rate vs PF scatter
├── chart_drawdown.png        Drawdown chart
└── chart_distribution.png   R histogram
```

---

## Programmatic API

```python
from run import run

# MT5 historical backtest
bt, tbl = run(
    symbol = "EURUSD",
    tf     = "H1",
    source = "mt5",
    start  = "2022-01-01",
)
print(tbl[["Trades", "Win Rate", "Net R"]].sort_values("Net R", ascending=False))

# Direct MT5 connection
from mt5_connect import MT5Connection

with MT5Connection() as conn:
    info  = conn.symbol_info("XAUUSD")
    acct  = conn.account_info()
    syms  = conn.available_symbols("XAU")
    print(info, acct, syms)
```
