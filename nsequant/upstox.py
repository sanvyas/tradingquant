"""Read-only Upstox client (historical daily candles, instrument master) for the Analytics Token.

The Analytics Token is read-only, valid for a year, and its docs list Historical Data, Option Chain and the
market-data websocket as covered with no static IP. It cannot place orders. Official limits for standard
APIs are 50/s, 500/min and 2000 per 30 minutes; this client stays well under them.
"""
from __future__ import annotations

import gzip
import io
import json
import time
from collections import deque
from datetime import date, timedelta
from urllib.parse import quote

import pandas as pd
import requests

BASE = "https://api.upstox.com"
NSE_INSTRUMENTS_URL = "https://assets.upstox.com/market-quote/instruments/exchange/NSE.json.gz"
MAX_DAYS_PER_REQUEST = 3650          # docs: one decade of daily data per request


class AuthError(RuntimeError):
    pass


class RateLimiter:
    """Sliding windows kept under Upstox's published limits (50/s, 500/min, 2000/30min)."""

    def __init__(self, limits=((1, 40), (60, 400), (1800, 1700)), clock=time.monotonic, sleep=time.sleep):
        self.limits, self.clock, self.sleep = limits, clock, sleep
        self.hits = {w: deque() for w, _ in limits}

    def wait(self) -> None:
        while True:
            now = self.clock()
            delay = 0.0
            for w, n in self.limits:
                q = self.hits[w]
                while q and now - q[0] >= w:
                    q.popleft()
                if len(q) >= n:
                    delay = max(delay, w - (now - q[0]))
            if delay <= 0:
                break
            self.sleep(delay + 0.01)
        now = self.clock()
        for w, _ in self.limits:
            self.hits[w].append(now)


class UpstoxClient:
    def __init__(self, token: str, session=None, limiter: RateLimiter | None = None, base: str = BASE,
                 sleep=time.sleep):
        if not token:
            raise AuthError("UPSTOX_ANALYTICS_TOKEN is empty")
        self.token, self.base, self.sleep = token, base, sleep
        self.session = session or requests.Session()
        self.limiter = limiter or RateLimiter()

    def _get(self, path: str, params: dict | None = None, retries: int = 5):
        for attempt in range(retries):
            self.limiter.wait()
            r = self.session.get(self.base + path, params=params, timeout=30, headers={
                "Accept": "application/json", "Authorization": f"Bearer {self.token}"})
            if r.status_code == 401:
                raise AuthError("Upstox rejected the token (401). Regenerate the Analytics Token.")
            if r.status_code == 429 or r.status_code >= 500:
                self.sleep(min(2 ** attempt, 30))
                continue
            if r.status_code != 200:
                raise RuntimeError(f"Upstox {r.status_code}: {r.text[:200]}")
            return r.json()
        raise RuntimeError(f"Upstox request failed after {retries} tries: {path}")

    def daily_candles(self, instrument_key: str, start: date, end: date) -> pd.DataFrame:
        """Daily OHLCV, ascending, index = trading date (IST). Corporate-action adjustment is NOT
        documented by Upstox, so callers must run data-quality checks (see data_quality.py)."""
        frames, hi = [], end
        while hi >= start:
            lo = max(start, hi - timedelta(days=MAX_DAYS_PER_REQUEST))
            path = f"/v3/historical-candle/{quote(instrument_key, safe='')}/days/1/{hi:%Y-%m-%d}/{lo:%Y-%m-%d}"
            data = self._get(path)
            frames.append(parse_candles(data))
            hi = lo - timedelta(days=1)
        out = pd.concat(frames) if frames else parse_candles({})
        return out[~out.index.duplicated(keep="last")].sort_index()

    def nse_equity_instruments(self) -> pd.DataFrame:
        r = self.session.get(NSE_INSTRUMENTS_URL, timeout=120)
        if r.status_code != 200:
            raise RuntimeError(f"instrument master download failed: {r.status_code}")
        rows = json.loads(gzip.GzipFile(fileobj=io.BytesIO(r.content)).read().decode("utf-8"))
        return equity_only(pd.DataFrame(rows))


def parse_candles(payload: dict) -> pd.DataFrame:
    cols = ["open", "high", "low", "close", "volume"]
    candles = (payload or {}).get("data", {}).get("candles", [])
    if not candles:
        return pd.DataFrame(columns=cols, index=pd.DatetimeIndex([]))
    ts = pd.to_datetime([c[0] for c in candles], utc=True).tz_convert("Asia/Kolkata").tz_localize(None).normalize()
    df = pd.DataFrame([c[1:6] for c in candles], columns=cols, index=ts).astype(float)
    return df.sort_index()


def equity_only(inst: pd.DataFrame) -> pd.DataFrame:
    """Plain NSE equity (series EQ), one row per trading symbol."""
    df = inst.copy()
    if "segment" in df:
        df = df[df["segment"] == "NSE_EQ"]
    if "instrument_type" in df:
        df = df[df["instrument_type"] == "EQ"]
    if "isin" in df:
        df = df[df["isin"].astype(str).str.startswith("INE")]
    keep = [c for c in ("instrument_key", "trading_symbol", "name", "isin") if c in df.columns]
    return df[keep].drop_duplicates("trading_symbol").reset_index(drop=True)
