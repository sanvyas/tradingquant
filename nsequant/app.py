"""Read-only dashboard. Password protected (HTTP Basic, DASH_PASSWORD env); locked if no password is set."""
from __future__ import annotations

import html
import json
import os
import secrets

import pandas as pd
from fastapi import Depends, FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from sqlalchemy import text

from . import store

app = FastAPI(title="tradingquant")
basic = HTTPBasic(auto_error=False)
_eng = None


def engine():
    global _eng
    if _eng is None:
        _eng = store.get_engine()
        store.init_db(_eng)
    return _eng


def auth(cred: HTTPBasicCredentials | None = Depends(basic)):
    pw = os.environ.get("DASH_PASSWORD")
    if not pw:
        raise HTTPException(503, "DASH_PASSWORD is not set; dashboard locked")
    if cred is None or not secrets.compare_digest(cred.password.encode(), pw.encode()):
        raise HTTPException(401, "auth required", headers={"WWW-Authenticate": "Basic"})


def _table(df: pd.DataFrame) -> str:
    if df is None or df.empty:
        return "<p><i>none yet</i></p>"
    return df.to_html(index=False, escape=True, border=0, classes="t", float_format=lambda x: f"{x:,.3f}")


def _q(sql: str, **p) -> pd.DataFrame:
    return pd.read_sql(text(sql), engine(), params=p)


def runs_frame() -> pd.DataFrame:
    df = _q("SELECT id, ts, scope, variant, metrics, promoted FROM runs ORDER BY ts DESC LIMIT 30")
    rows = []
    for r in df.itertuples():
        m = json.loads(r.metrics)
        rows.append(dict(ts=r.ts[:16], scope=r.scope, variant=r.variant, trades=m.get("n_trades"),
                         net_after_tax=m.get("net_pnl_after_tax"), dsr=m.get("dsr"), p_profit=m.get("prob_net_profit"),
                         max_dd=m.get("max_drawdown_pct"), trials=m.get("n_trials"),
                         gate="PASS" if r.promoted else "no"))
    return pd.DataFrame(rows)


@app.get("/healthz")
def healthz():
    return {"ok": True}


@app.get("/api/summary", dependencies=[Depends(auth)])
def summary():
    return dict(
        symbols=int(_q("SELECT COUNT(*) n FROM universe WHERE active=1")["n"][0]),
        price_rows=int(_q("SELECT COUNT(*) n FROM prices_daily")["n"][0]),
        last_price_date=_q("SELECT MAX(date) d FROM prices_daily")["d"][0],
        trials=_q("SELECT scope, COUNT(*) n FROM trials GROUP BY scope").to_dict("records"),
        open_data_issues=int(_q("SELECT COUNT(*) n FROM data_issues")["n"][0]))


@app.get("/", response_class=HTMLResponse, dependencies=[Depends(auth)])
def home():
    s = summary()
    sig = _q("SELECT * FROM signals WHERE date=(SELECT MAX(date) FROM signals) ORDER BY score DESC LIMIT 50")
    issues = _q("SELECT symbol, kind, detail FROM data_issues ORDER BY kind, symbol LIMIT 100")
    jobs = _q("SELECT ts, job, status, detail FROM job_log ORDER BY ts DESC LIMIT 15")
    jobs["detail"] = jobs["detail"].str.slice(0, 120)
    body = f"""<h1>tradingquant</h1>
<p><b>RESEARCH ONLY. No orders are placed.</b> A gate pass means 'worth paper trading', not 'safe to trade'.
Backtests are on today's listed stocks (survivorship bias) and costs/taxes are partly assumptions.</p>
<p>{s['symbols']} symbols &middot; {s['price_rows']:,} price rows &middot; latest bar {html.escape(str(s['last_price_date']))}
&middot; {s['open_data_issues']} data issues &middot; trials: {html.escape(str(s['trials']))}</p>
<h2>Latest research runs</h2>{_table(runs_frame())}
<h2>Latest signals (research only)</h2>{_table(sig)}
<h2>Data issues</h2>{_table(issues)}
<h2>Job log</h2>{_table(jobs)}"""
    return f"""<!doctype html><meta charset=utf-8><meta name=viewport content="width=device-width,initial-scale=1">
<style>body{{font:14px system-ui;margin:16px;max-width:1100px}}.t{{border-collapse:collapse;font-size:12px}}
.t td,.t th{{padding:3px 8px;border-bottom:1px solid #ddd;text-align:left}}</style>{body}"""
