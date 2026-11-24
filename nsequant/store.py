"""Storage. One code path for SQLite (tests, local) and Postgres (Render). Dates are ISO text."""
from __future__ import annotations

import json
import os
from datetime import datetime, timezone

import pandas as pd
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine

SCHEMA = [
    """CREATE TABLE IF NOT EXISTS universe (
        symbol TEXT PRIMARY KEY, instrument_key TEXT NOT NULL, name TEXT, isin TEXT,
        role TEXT NOT NULL DEFAULT 'stock', active INTEGER NOT NULL DEFAULT 1, updated_at TEXT)""",
    """CREATE TABLE IF NOT EXISTS prices_daily (
        symbol TEXT NOT NULL, date TEXT NOT NULL, open DOUBLE PRECISION, high DOUBLE PRECISION,
        low DOUBLE PRECISION, close DOUBLE PRECISION, volume DOUBLE PRECISION,
        PRIMARY KEY (symbol, date))""",
    """CREATE TABLE IF NOT EXISTS trials (
        scope TEXT NOT NULL, key TEXT NOT NULL, name TEXT, params TEXT, sr DOUBLE PRECISION,
        n_obs INTEGER, created_at TEXT, PRIMARY KEY (scope, key))""",
    """CREATE TABLE IF NOT EXISTS runs (
        id TEXT PRIMARY KEY, ts TEXT, scope TEXT, variant TEXT, metrics TEXT, promoted INTEGER)""",
    """CREATE TABLE IF NOT EXISTS signals (
        date TEXT NOT NULL, symbol TEXT NOT NULL, side INTEGER, stop_pct DOUBLE PRECISION,
        target_pct DOUBLE PRECISION, score DOUBLE PRECISION, status TEXT, scope TEXT,
        PRIMARY KEY (date, symbol, scope))""",
    """CREATE TABLE IF NOT EXISTS data_issues (
        symbol TEXT NOT NULL, kind TEXT NOT NULL, detail TEXT, found_at TEXT,
        PRIMARY KEY (symbol, kind))""",
    """CREATE TABLE IF NOT EXISTS job_log (
        ts TEXT, job TEXT, status TEXT, detail TEXT)""",
]


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def get_engine(url: str | None = None) -> Engine:
    url = url or os.environ.get("DATABASE_URL", "sqlite:///data/tq.db")
    if url.startswith("postgres://"):
        url = "postgresql+psycopg2://" + url[len("postgres://"):]
    elif url.startswith("postgresql://"):
        url = "postgresql+psycopg2://" + url[len("postgresql://"):]
    if url.startswith("sqlite:///") and not url.endswith(":memory:"):
        os.makedirs(os.path.dirname(url[len("sqlite:///"):]) or ".", exist_ok=True)
    return create_engine(url, future=True)


def init_db(eng: Engine) -> None:
    with eng.begin() as c:
        for ddl in SCHEMA:
            c.execute(text(ddl))


def log_job(eng: Engine, job: str, status: str, detail: str = "") -> None:
    with eng.begin() as c:
        c.execute(text("INSERT INTO job_log(ts, job, status, detail) VALUES (:t,:j,:s,:d)"),
                  dict(t=now_iso(), j=job, s=status, d=detail[:2000]))


# ---------------------------------------------------------------- universe
def upsert_universe(eng: Engine, rows: list[dict]) -> None:
    if not rows:
        return
    sql = text("""INSERT INTO universe(symbol, instrument_key, name, isin, role, active, updated_at)
        VALUES (:symbol,:instrument_key,:name,:isin,:role,1,:updated_at)
        ON CONFLICT (symbol) DO UPDATE SET instrument_key=excluded.instrument_key, name=excluded.name,
        isin=excluded.isin, role=excluded.role, active=1, updated_at=excluded.updated_at""")
    ts = now_iso()
    with eng.begin() as c:
        c.execute(sql, [{**dict(name=None, isin=None, role="stock"), **r, "updated_at": ts} for r in rows])


def get_universe(eng: Engine, role: str | None = None, active_only: bool = True) -> pd.DataFrame:
    q = "SELECT * FROM universe WHERE 1=1"
    p: dict = {}
    if role:
        q += " AND role=:r"
        p["r"] = role
    if active_only:
        q += " AND active=1"
    return pd.read_sql(text(q + " ORDER BY symbol"), eng, params=p)


def set_active(eng: Engine, symbols: list[str]) -> None:
    """Mark exactly these stocks active (index rows are untouched)."""
    with eng.begin() as c:
        c.execute(text("UPDATE universe SET active=0 WHERE role='stock'"))
        for s in symbols:
            c.execute(text("UPDATE universe SET active=1 WHERE symbol=:s"), dict(s=s))


# ---------------------------------------------------------------- prices
def upsert_prices(eng: Engine, symbol: str, df: pd.DataFrame) -> int:
    if df is None or df.empty:
        return 0
    rows = [dict(symbol=symbol, date=pd.Timestamp(i).strftime("%Y-%m-%d"), open=float(r.open), high=float(r.high),
                 low=float(r.low), close=float(r.close), volume=float(r.volume)) for i, r in df.iterrows()]
    sql = text("""INSERT INTO prices_daily(symbol,date,open,high,low,close,volume)
        VALUES (:symbol,:date,:open,:high,:low,:close,:volume)
        ON CONFLICT (symbol,date) DO UPDATE SET open=excluded.open, high=excluded.high, low=excluded.low,
        close=excluded.close, volume=excluded.volume""")
    with eng.begin() as c:
        c.execute(sql, rows)
    return len(rows)


def delete_prices(eng: Engine, symbol: str) -> None:
    with eng.begin() as c:
        c.execute(text("DELETE FROM prices_daily WHERE symbol=:s"), dict(s=symbol))


def last_price_date(eng: Engine, symbol: str) -> str | None:
    with eng.connect() as c:
        return c.execute(text("SELECT MAX(date) FROM prices_daily WHERE symbol=:s"), dict(s=symbol)).scalar()


def load_prices(eng: Engine, symbols: list[str] | None = None, start: str | None = None) -> dict[str, pd.DataFrame]:
    q, p = "SELECT * FROM prices_daily WHERE 1=1", {}
    if start:
        q += " AND date >= :start"
        p["start"] = start
    df = pd.read_sql(text(q + " ORDER BY symbol, date"), eng, params=p)
    out = {}
    for sym, g in df.groupby("symbol"):
        if symbols is not None and sym not in symbols:
            continue
        g = g.drop(columns="symbol").set_index(pd.to_datetime(g["date"])).drop(columns="date")
        g.index.name = None
        out[sym] = g
    return out


# ---------------------------------------------------------------- issues, runs, signals
def set_issue(eng: Engine, symbol: str, kind: str, detail: str) -> None:
    with eng.begin() as c:
        c.execute(text("""INSERT INTO data_issues(symbol,kind,detail,found_at) VALUES (:s,:k,:d,:t)
            ON CONFLICT (symbol,kind) DO UPDATE SET detail=excluded.detail, found_at=excluded.found_at"""),
                  dict(s=symbol, k=kind, d=detail[:500], t=now_iso()))


def clear_issues(eng: Engine, symbol: str, kind: str) -> None:
    with eng.begin() as c:
        c.execute(text("DELETE FROM data_issues WHERE symbol=:s AND kind=:k"), dict(s=symbol, k=kind))


def flagged_symbols(eng: Engine, kinds: set[str] | None = None) -> set[str]:
    df = pd.read_sql(text("SELECT symbol, kind FROM data_issues"), eng)
    if kinds is not None:
        df = df[df["kind"].isin(kinds)]
    return set(df["symbol"])


def save_run(eng: Engine, run_id: str, scope: str, variant: str, metrics: dict, promoted: bool) -> None:
    with eng.begin() as c:
        c.execute(text("""INSERT INTO runs(id,ts,scope,variant,metrics,promoted) VALUES (:i,:t,:s,:v,:m,:p)
            ON CONFLICT (id) DO UPDATE SET metrics=excluded.metrics, promoted=excluded.promoted"""),
                  dict(i=run_id, t=now_iso(), s=scope, v=variant, m=json.dumps(metrics, default=float), p=int(promoted)))


def save_signals(eng: Engine, rows: list[dict]) -> None:
    if not rows:
        return
    with eng.begin() as c:
        c.execute(text("""INSERT INTO signals(date,symbol,side,stop_pct,target_pct,score,status,scope)
            VALUES (:date,:symbol,:side,:stop_pct,:target_pct,:score,:status,:scope)
            ON CONFLICT (date,symbol,scope) DO UPDATE SET score=excluded.score, status=excluded.status,
            stop_pct=excluded.stop_pct, target_pct=excluded.target_pct"""), rows)
