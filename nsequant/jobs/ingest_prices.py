"""Incremental daily-price ingest. Run after the market close (about 16:30 IST) on trading days.

For every active universe row it fetches only new bars, plus a short overlap. If the overlap disagrees with
what is stored (a split/bonus restated history), that symbol is re-downloaded in full, so stored history never
silently goes stale. Afterwards each symbol is checked for data-quality problems.
"""
from __future__ import annotations

import os
from datetime import date, datetime, timedelta
from zoneinfo import ZoneInfo

import pandas as pd

from .. import store
from ..data_quality import blocking, find_issues
from ..upstox import UpstoxClient

HISTORY_START = os.environ.get("HISTORY_START", "2019-01-01")
RESTATE_TOLERANCE = 0.005          # 0.5% close mismatch on overlapping days means history was restated


def _today_ist(now: datetime | None = None) -> tuple[date, bool]:
    now = now or datetime.now(ZoneInfo("Asia/Kolkata"))
    return now.date(), now.hour >= 16          # today's daily bar is final only after the close


def run(eng, client: UpstoxClient, now: datetime | None = None, overlap_days: int = 7) -> dict:
    today, final_today = _today_ist(now)
    end = today if final_today else today - timedelta(days=1)
    uni = store.get_universe(eng)
    stats = dict(symbols=len(uni), new_rows=0, restated=[], failed=[], flagged=[])
    for r in uni.itertuples():
        try:
            last = store.last_price_date(eng, r.symbol)
            start = date.fromisoformat(HISTORY_START) if last is None else \
                date.fromisoformat(last) - timedelta(days=overlap_days)
            fresh = client.daily_candles(r.instrument_key, start, end)
            if last is not None and not fresh.empty:
                old = store.load_prices(eng, [r.symbol]).get(r.symbol)
                both = old["close"].reindex(fresh.index).dropna() if old is not None else pd.Series(dtype=float)
                if len(both) and ((both / fresh["close"].reindex(both.index) - 1).abs() > RESTATE_TOLERANCE).any():
                    stats["restated"].append(r.symbol)
                    store.delete_prices(eng, r.symbol)
                    fresh = client.daily_candles(r.instrument_key, date.fromisoformat(HISTORY_START), end)
            stats["new_rows"] += store.upsert_prices(eng, r.symbol, fresh)
            full = store.load_prices(eng, [r.symbol]).get(r.symbol)
            iss = find_issues(full) if full is not None else [("no_data", "empty")]
            for kind in ("suspected_unadjusted_corporate_action", "bad_ohlc", "stale_prices", "duplicate_dates",
                         "no_data", "short_history"):
                store.clear_issues(eng, r.symbol, kind)
            for kind, detail in iss:
                store.set_issue(eng, r.symbol, kind, detail)
            if blocking(iss):
                stats["flagged"].append(r.symbol)
        except Exception as e:                     # one bad symbol must not stop the run
            stats["failed"].append(f"{r.symbol}: {e}"[:200])
    store.log_job(eng, "ingest_prices", "ok" if not stats["failed"] else "partial", str(stats)[:1500])
    return stats


def main() -> None:
    eng = store.get_engine()
    store.init_db(eng)
    print(run(eng, UpstoxClient(os.environ["UPSTOX_ANALYTICS_TOKEN"])))


if __name__ == "__main__":
    main()
