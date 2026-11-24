"""Data-quality checks. Bad data is the most common reason a backtest lies, so symbols with blocking
issues are excluded from research until a human clears them.

Upstox does not document whether daily candles are adjusted for splits/bonuses, so unadjusted corporate
actions are detected from the price series itself: a 1:1 bonus or 1:2 split halves the price overnight.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

BLOCKING = {"suspected_unadjusted_corporate_action", "bad_ohlc", "stale_prices", "duplicate_dates"}
DROP_RATIO, JUMP_RATIO = 0.65, 1.80


def find_issues(df: pd.DataFrame) -> list[tuple[str, str]]:
    issues: list[tuple[str, str]] = []
    if df is None or df.empty:
        return [("no_data", "no rows")]
    if df.index.duplicated().any():
        issues.append(("duplicate_dates", f"{int(df.index.duplicated().sum())} duplicate dates"))
    c = df["close"]
    ratio = c / c.shift(1)
    bad = ratio[(ratio <= DROP_RATIO) | (ratio >= JUMP_RATIO)]
    if len(bad):
        d = bad.index[0]
        issues.append(("suspected_unadjusted_corporate_action",
                       f"{len(bad)} jump(s); first {pd.Timestamp(d).date()} close ratio {bad.iloc[0]:.2f}"))
    tol = 1e-6 * c.abs().clip(lower=1)
    broken = (df["high"] < df["low"] - tol) | (df["close"] > df["high"] + tol) | (df["close"] < df["low"] - tol) \
        | (df[["open", "high", "low", "close"]] <= 0).any(axis=1)
    if broken.any():
        issues.append(("bad_ohlc", f"{int(broken.sum())} inconsistent bars; first {broken.idxmax().date()}"))
    flat = (c.diff() == 0) & (df["volume"] == 0)
    run = flat.astype(int).groupby((~flat).cumsum()).cumsum()
    if run.max() >= 5:
        issues.append(("stale_prices", f"{int(run.max())} consecutive zero-volume unchanged days"))
    if len(df) < 250:
        issues.append(("short_history", f"only {len(df)} bars"))
    return issues


def blocking(issues: list[tuple[str, str]]) -> list[tuple[str, str]]:
    return [i for i in issues if i[0] in BLOCKING]
