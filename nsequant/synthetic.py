"""Synthetic market for plumbing tests. NOT evidence about real markets.

`edge_drift` injects a small, known, observable pattern: after a down-gap that closes in the upper part of
its range, the next 3 days drift upward. With edge_drift=0 the market has no exploitable structure and any
pipeline that reports profit on it has a leak or a cost-model bug.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def make_market(n_symbols: int = 30, n_days: int = 1800, edge_drift: float = 0.0, seed: int = 11):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2019-01-01", periods=n_days)
    mkt = rng.normal(0.0003, 0.009, n_days)
    index_close = pd.Series(1000 * np.exp(np.cumsum(mkt)), index=idx, name="index")
    bars = {}
    for k in range(n_symbols):
        beta = rng.uniform(0.6, 1.3)
        idio = 0.011 + 0.004 * rng.random()
        c = np.empty(n_days); o = np.empty(n_days); h = np.empty(n_days); l = np.empty(n_days)
        v = rng.integers(2_000_000, 8_000_000, n_days).astype(float)
        pending = np.zeros(n_days + 4)
        c_prev = 200.0 * (1 + rng.random())
        for t in range(n_days):
            gap = rng.normal(0, 0.006)
            ret = beta * mkt[t] + rng.normal(0, idio) + pending[t]
            o[t] = c_prev * (1 + gap + 0.3 * pending[t])
            c[t] = c_prev * (1 + ret)
            top, bot = max(o[t], c[t]), min(o[t], c[t])
            h[t] = top * (1 + abs(rng.normal(0, 0.004)))
            l[t] = bot * (1 - abs(rng.normal(0, 0.004)))
            if edge_drift:
                clv = ((c[t] - l[t]) - (h[t] - c[t])) / max(h[t] - l[t], 1e-9)
                g = o[t] / c_prev - 1
                if g <= -0.007 and clv > 0.3:
                    pending[t + 1:t + 4] += edge_drift / 3
            c_prev = c[t]
        bars[f"SYM{k:02d}"] = pd.DataFrame(dict(open=o, high=h, low=l, close=c, volume=v), index=idx)
    return bars, index_close
