"""Pick the tradable universe: the most liquid NSE equities, plus the Nifty 50 index as the market proxy.

Run monthly. Liquidity = median traded value (close x volume) over the last ~60 sessions.
KNOWN LIMIT: this starts from stocks listed TODAY, so delisted stocks are missing (survivorship bias).
Backtests on this universe overstate results until a survivorship-free source is added.
"""
from __future__ import annotations

import os
from datetime import date, timedelta

import pandas as pd

from .. import store
from ..upstox import UpstoxClient

INDEX_KEY = "NSE_INDEX|Nifty 50"


def run(eng, client: UpstoxClient, top_n: int = 150, lookback_days: int = 130, min_price: float = 50.0,
        max_candidates: int | None = None, today: date | None = None) -> dict:
    today = today or date.today()
    inst = client.nse_equity_instruments()
    if max_candidates:
        inst = inst.head(max_candidates)
    scores = []
    for r in inst.itertuples():
        try:
            df = client.daily_candles(r.instrument_key, today - timedelta(days=lookback_days), today)
        except Exception:
            continue
        tail = df.tail(60)
        if len(tail) < 40 or tail["close"].iloc[-1] < min_price:
            continue
        scores.append((r.trading_symbol, r.instrument_key, getattr(r, "name", None), getattr(r, "isin", None),
                       float((tail["close"] * tail["volume"]).median())))
    scores.sort(key=lambda x: -x[4])
    chosen = scores[:top_n]
    store.upsert_universe(eng, [dict(symbol=s, instrument_key=k, name=n, isin=i, role="stock")
                                for s, k, n, i, _ in chosen])
    store.set_active(eng, [c[0] for c in chosen])
    store.upsert_universe(eng, [dict(symbol="NIFTY50", instrument_key=INDEX_KEY, role="index")])
    summary = dict(candidates=len(inst), eligible=len(scores), selected=len(chosen))
    store.log_job(eng, "build_universe", "ok", str(summary))
    return summary


def main() -> None:
    eng = store.get_engine()
    store.init_db(eng)
    print(run(eng, UpstoxClient(os.environ["UPSTOX_ANALYTICS_TOKEN"])))


if __name__ == "__main__":
    main()
