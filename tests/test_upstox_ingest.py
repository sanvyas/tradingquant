import gzip
import json
from datetime import date, datetime
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import pytest

from nsequant import store
from nsequant.data_quality import blocking, find_issues
from nsequant.jobs import build_universe, ingest_prices
from nsequant.upstox import AuthError, RateLimiter, UpstoxClient, equity_only, parse_candles


class Resp:
    def __init__(self, code=200, payload=None, content=b""):
        self.status_code, self._p, self.content, self.text = code, payload, content, str(payload)[:100]

    def json(self):
        return self._p


def candles_payload(start, n, base=100.0, scale=1.0):
    idx = pd.bdate_range(start, periods=n)
    rows = [[f"{d:%Y-%m-%d}T00:00:00+05:30", base * scale, base * scale + 1, base * scale - 1,
             (base + i % 3) * scale, 1_000_000, 0] for i, d in enumerate(idx)]
    return {"status": "success", "data": {"candles": rows[::-1]}}          # Upstox returns newest first


class FakeSession:
    def __init__(self, handler):
        self.handler, self.calls = handler, []

    def get(self, url, params=None, timeout=None, headers=None):
        self.calls.append((url, headers))
        return self.handler(url)


def client(handler, **kw):
    return UpstoxClient("tok", session=FakeSession(handler), limiter=RateLimiter(clock=lambda: 0.0, sleep=lambda s: None),
                        sleep=lambda s: None, **kw)


def test_parse_candles_sorts_and_normalises_to_ist_dates():
    df = parse_candles(candles_payload("2026-01-05", 3))
    assert list(df.index) == list(pd.bdate_range("2026-01-05", periods=3))
    assert df.columns.tolist() == ["open", "high", "low", "close", "volume"]


def test_daily_candles_encodes_key_and_sends_bearer():
    c = client(lambda url: Resp(200, candles_payload("2026-01-05", 3)))
    c.daily_candles("NSE_EQ|INE848E01016", date(2026, 1, 1), date(2026, 1, 9))
    url, headers = c.session.calls[0]
    assert "NSE_EQ%7CINE848E01016/days/1/2026-01-09/2026-01-01" in url
    assert headers["Authorization"] == "Bearer tok"


def test_long_history_is_chunked_under_the_decade_limit():
    c = client(lambda url: Resp(200, candles_payload("2020-01-06", 3)))
    c.daily_candles("NSE_EQ|X", date(2005, 1, 1), date(2026, 1, 1))
    assert len(c.session.calls) >= 3


def test_401_raises_auth_error_and_429_is_retried():
    with pytest.raises(AuthError):
        client(lambda url: Resp(401)).daily_candles("k", date(2026, 1, 1), date(2026, 1, 5))
    state = {"n": 0}

    def flaky(url):
        state["n"] += 1
        return Resp(429) if state["n"] < 3 else Resp(200, candles_payload("2026-01-05", 2))
    assert len(client(flaky).daily_candles("k", date(2026, 1, 1), date(2026, 1, 9))) == 2


def test_rate_limiter_never_exceeds_window_limit():
    t = {"now": 0.0}
    sleeps = []

    def sleep(s):
        sleeps.append(s)
        t["now"] += s
    rl = RateLimiter(limits=((1, 5),), clock=lambda: t["now"], sleep=sleep)
    for _ in range(12):
        rl.wait()
    assert sleeps and t["now"] >= 1.0               # had to wait for the window to clear


def test_data_quality_flags_unadjusted_split_and_bad_bars():
    idx = pd.bdate_range("2024-01-01", periods=300)
    close = np.r_[np.full(150, 200.0), np.full(150, 100.0)]            # 1:1 bonus, unadjusted
    df = pd.DataFrame(dict(open=close, high=close + 1, low=close - 1, close=close, volume=1e6), index=idx)
    kinds = [k for k, _ in find_issues(df)]
    assert "suspected_unadjusted_corporate_action" in kinds
    bad = df.copy()
    bad.iloc[10, bad.columns.get_loc("high")] = 50.0
    assert "bad_ohlc" in [k for k, _ in find_issues(bad)]
    clean = df.copy()
    clean["close"] = 100.0
    clean["open"], clean["high"], clean["low"] = 100.0, 101.0, 99.0
    assert not blocking(find_issues(clean))


def _setup(tmp_path):
    e = store.get_engine(f"sqlite:///{tmp_path}/t.db")
    store.init_db(e)
    store.upsert_universe(e, [dict(symbol="AAA", instrument_key="NSE_EQ|A")])
    return e


NOW = datetime(2026, 11, 25, 17, 0, tzinfo=ZoneInfo("Asia/Kolkata"))


def test_ingest_first_load_then_incremental(tmp_path, monkeypatch):
    monkeypatch.setattr(ingest_prices, "HISTORY_START", "2025-01-01")
    e = _setup(tmp_path)
    c = client(lambda url: Resp(200, candles_payload("2025-01-01", 300)))
    s1 = ingest_prices.run(e, c, now=NOW)
    assert s1["new_rows"] == 300 and not s1["failed"]
    s2 = ingest_prices.run(e, c, now=NOW)                    # same data again: idempotent
    assert len(store.load_prices(e)["AAA"]) == 300 and not s2["restated"]


def test_ingest_detects_restated_history_and_redownloads(tmp_path, monkeypatch):
    monkeypatch.setattr(ingest_prices, "HISTORY_START", "2025-01-01")
    e = _setup(tmp_path)
    ingest_prices.run(e, client(lambda url: Resp(200, candles_payload("2025-01-01", 300))), now=NOW)
    old_first = store.load_prices(e)["AAA"]["close"].iloc[0]
    halved = client(lambda url: Resp(200, candles_payload("2025-01-01", 300, scale=0.5)))   # provider restated
    s = ingest_prices.run(e, halved, now=NOW)
    assert s["restated"] == ["AAA"]
    assert store.load_prices(e)["AAA"]["close"].iloc[0] == pytest.approx(old_first * 0.5)


def test_ingest_one_failing_symbol_does_not_stop_the_rest(tmp_path, monkeypatch):
    monkeypatch.setattr(ingest_prices, "HISTORY_START", "2025-01-01")
    e = _setup(tmp_path)
    store.upsert_universe(e, [dict(symbol="BBB", instrument_key="NSE_EQ|B")])

    def handler(url):
        return Resp(500) if "NSE_EQ%7CA/" in url else Resp(200, candles_payload("2025-01-01", 300))
    s = ingest_prices.run(e, client(handler), now=NOW)
    assert len(s["failed"]) == 1 and "AAA" in s["failed"][0]
    assert "BBB" in store.load_prices(e)


def test_build_universe_selects_most_liquid_and_adds_index(tmp_path):
    e = store.get_engine(f"sqlite:///{tmp_path}/u.db")
    store.init_db(e)
    inst = [dict(instrument_key=f"NSE_EQ|K{i}", trading_symbol=f"S{i}", name=f"Co {i}", isin=f"INE00{i}",
                 segment="NSE_EQ", instrument_type="EQ") for i in range(5)]
    inst.append(dict(instrument_key="NSE_EQ|BOND", trading_symbol="BND", name="Bond", isin="INF001",
                     segment="NSE_EQ", instrument_type="EQ"))            # not INE -> excluded
    raw = gzip.compress(json.dumps(inst).encode())

    def handler(url):
        if url.endswith("NSE.json.gz"):
            return Resp(200, content=raw)
        k = int(url.split("K")[1].split("/")[0])
        p = candles_payload("2026-05-01", 100, base=100.0)
        for row in p["data"]["candles"]:
            row[5] = 1000 * (k + 1)                                   # S4 is the most liquid
        return Resp(200, p)
    c = client(handler)
    out = build_universe.run(e, c, top_n=2, today=date(2026, 9, 25))
    assert out["candidates"] == 5 and out["selected"] == 2
    assert set(store.get_universe(e, role="stock")["symbol"]) == {"S4", "S3"}
    assert len(store.get_universe(e, role="index")) == 1


def test_equity_only_filters_non_equity():
    df = pd.DataFrame([dict(instrument_key="a", trading_symbol="A", isin="INE1", segment="NSE_EQ", instrument_type="EQ"),
                       dict(instrument_key="b", trading_symbol="B", isin="INF1", segment="NSE_EQ", instrument_type="EQ"),
                       dict(instrument_key="c", trading_symbol="C", isin="INE3", segment="NSE_FO", instrument_type="FUT")])
    assert list(equity_only(df)["trading_symbol"]) == ["A"]


def test_doctor_reports_each_check_without_leaking_secrets(tmp_path):
    from nsequant.jobs import doctor
    e = store.get_engine(f"sqlite:///{tmp_path}/d.db")
    ok = doctor.check({}, client=client(lambda url: Resp(200, candles_payload("2026-01-05", 3))), engine=e)
    assert all(x[1] for x in ok)
    missing = doctor.check({"DATABASE_URL": "sqlite:///" + str(tmp_path) + "/e.db"})
    assert missing[1][1] is False and "not set" in missing[1][2]
    bad = doctor.check({}, client=client(lambda url: Resp(401)), engine=e)
    assert bad[1][1] is False
