# nsequant

Research, backtest and signal framework for Indian equities. **Signal-only by default.**
No order is sent anywhere. `PaperBroker` only records. Live ordering is a later stage and needs your
broker's SEBI algo setup (exchange-approved, static IP, algo ID) before you enable it.

## Stage status

| Stage | Status |
|---|---|
| 1. Foundation (config, broker adapters, storage layout) | done (adapter interfaces + paper broker; no vendor SDK wired yet) |
| 2. Stocks daily/weekly/BTST research engine | core built and tested on synthetic data; **not yet run on real data** |
| 3. Intraday | not started |
| 4. Index options | not started (needs recorded option-chain history) |
| 5. Execution + risk layer | not started |

## What is in stage 2

- `costs.py`, `tax.py`, `config/costs.yaml` — brokerage, STT, exchange, SEBI, stamp, GST, DP charges,
  margin, and tax by income head. Every rate is data with a source and a `verify` / `assumption` flag.
  `python -m nsequant.config` lists what is not verified.
- `backtest.py` — daily engine: stop / target / time exit, same-bar = stop first, gap-through-stop fills at
  open, slippage, capital and free-margin limits, net reward:risk gate, mark-to-market equity, walk-forward
  splits with embargo.
- `stats.py` — trial registry, Deflated Sharpe, bootstrap probability of NET profit after costs and tax,
  drawdown probability.
- `features.py` — standard indicators plus structural features (gap, close location, range compression,
  trend efficiency, variance ratio, volatility regime, volume anomalies, residual return, relative strength,
  global context with explicit lags). A test proves no feature looks at future bars.
- `news.py` — ingest, dedupe/novelty, entity linking, event type, pluggable sentiment (FinBERT on CPU),
  **measured** impact via event study, point-in-time alignment (publish time + ingest delay vs decision cutoff).
- `models.py` — meta-labelling on triple-barrier outcomes (same exit logic as the backtester).

## Rules this project follows

1. Every strategy variant you try goes in the `TrialRegistry`. One registry per idea + dataset.
2. Only out-of-sample walk-forward results count.
3. A trade must clear costs: net reward:risk and minimum net profit are checked after charges.
4. Profit probability is quoted after costs AND tax, never before.
5. News only becomes a feature after it could have been seen. Sentiment without measured impact is ignored.

## Run

    pip install -r requirements.txt
    pytest -q
    python scripts/e2e_synthetic.py 1.0     # no-edge control vs injected edge (synthetic, ~1 min)

Synthetic data tests plumbing only. They say nothing about real markets.

## Cloud deployment (all free tiers)
* **Jobs:** GitHub Actions (`.github/workflows`): monthly universe, weekday ingest + research. Run manually from the Actions tab.
* **Database:** any free Postgres (Neon or a dedicated Supabase project) via the `DATABASE_URL` Actions secret.
* **Dashboard:** `render.yaml` (Render free web service; sleeps when idle, wakes on visit).
* Secrets live only in GitHub/Render secret settings, never in the repo. First run: `doctor` -> `all`.
* Research only; no orders are placed. Changing model code? Bump the trial scope so the trial count stays honest.
