"""Control panel + results dashboard.

Password protected (HTTP Basic, DASH_PASSWORD); locked if no password is set. Actions start GitHub Actions
workflows (needs GITHUB_TOKEN with Actions read/write on the repo); the heavy work never runs in this web service.
"""
from __future__ import annotations

import json
import math
import os
import secrets

import pandas as pd
import requests
from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from pydantic import BaseModel
from sqlalchemy import text

from . import store
from .config import load_settings
from .research import GATE, gate_reasons

app = FastAPI(title="tradingquant")
basic = HTTPBasic(auto_error=False)
_eng = None

ACTIONS = {
    "doctor": "Check setup",
    "universe": "Refresh stock list",
    "ingest": "Pull latest prices",
    "research": "Run research",
    "all": "Run everything",
    "holdout": "Final one-time test",
}


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


def clean(o):
    """JSON-safe: NaN/inf -> None."""
    if isinstance(o, float):
        return o if math.isfinite(o) else None
    if isinstance(o, dict):
        return {k: clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [clean(v) for v in o]
    return o


def _q(sql: str, **p) -> pd.DataFrame:
    return pd.read_sql(text(sql), engine(), params=p)


# ---------- GitHub Actions bridge ----------
def gh_request(method: str, path: str, **kw):
    token = os.environ.get("GITHUB_TOKEN")
    if not token:
        return 0, {"message": "GITHUB_TOKEN is not set on this service"}
    repo = os.environ.get("GITHUB_REPO", "sanvyas/tradingquant")
    r = requests.request(method, f"https://api.github.com/repos/{repo}{path}", timeout=15, headers={
        "Authorization": f"Bearer {token}", "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28"}, **kw)
    try:
        body = r.json() if r.content else {}
    except ValueError:
        body = {}
    return r.status_code, body


def github_runs() -> dict:
    try:
        code, body = gh_request("GET", "/actions/workflows/daily.yml/runs?per_page=6")
    except Exception as e:
        return dict(connected=False, error=f"{type(e).__name__}", runs=[])
    if code == 0:
        return dict(connected=False, error=body["message"], runs=[])
    if code != 200:
        return dict(connected=False, error=f"GitHub said {code}: {body.get('message', '')}"[:200], runs=[])
    runs = [dict(id=r["id"], status=r["status"], conclusion=r.get("conclusion"), created=r["created_at"],
                 url=r["html_url"], title=r.get("display_title") or "", event=r["event"])
            for r in body.get("workflow_runs", [])]
    return dict(connected=True, error=None, runs=runs)


class ActionIn(BaseModel):
    action: str


@app.post("/api/action", dependencies=[Depends(auth)])
def run_action(a: ActionIn, x_tq: str | None = Header(default=None)):
    if x_tq != "1":                                   # blocks cross-site form posts (CSRF)
        raise HTTPException(403, "missing header")
    if a.action not in ACTIONS:
        raise HTTPException(400, "unknown action")
    code, body = gh_request("POST", "/actions/workflows/daily.yml/dispatches",
                            json={"ref": "main", "inputs": {"job": a.action}})
    if code == 204:
        return dict(ok=True, message=f"Started: {ACTIONS[a.action]}. Progress shows under 'Jobs'.")
    if code == 0:
        raise HTTPException(503, body["message"])
    raise HTTPException(502, f"GitHub refused ({code}): {body.get('message', '')}"[:200])


# ---------- data for the page ----------
def _runs(limit=40) -> list[dict]:
    df = _q("SELECT id, ts, scope, variant, metrics, promoted FROM runs ORDER BY ts DESC LIMIT :n", n=limit)
    out = []
    for r in df.itertuples():
        m = json.loads(r.metrics)
        out.append(dict(id=r.id, ts=r.ts, scope=r.scope, variant=r.variant, passed=bool(r.promoted),
                        holdout=r.variant.startswith("HOLDOUT"), m=m))
    return out


def _signals(capital: float, risk: dict) -> list[dict]:
    sig = _q("SELECT * FROM signals WHERE date=(SELECT MAX(date) FROM signals) ORDER BY score DESC LIMIT 25")
    if sig.empty:
        return []
    px = store.load_prices(engine(), list(sig["symbol"]), start=str((pd.Timestamp.now() - pd.Timedelta(days=15)).date()))
    rows = []
    for r in sig.itertuples():
        b = px.get(r.symbol)
        last = float(b["close"].iloc[-1]) if b is not None and len(b) else None
        qty = None
        if last and r.stop_pct:
            by_risk = capital * risk["risk_per_trade_pct"] / (last * r.stop_pct)
            qty = int(max(0, min(by_risk, capital * risk["max_position_pct"] / last)))
        rows.append(dict(date=r.date, symbol=r.symbol, score=r.score, status=r.status, ref_price=last,
                         stop=round(last * (1 - r.stop_pct), 2) if last else None,
                         target=round(last * (1 + r.target_pct), 2) if last else None,
                         stop_pct=r.stop_pct, target_pct=r.target_pct, qty=qty))
    return rows


_FMT = {"Enough trades": "int", "Deflated Sharpe (luck-adjusted)": "num", "Profit after costs and tax": "inr",
        "Chance of profit after tax": "pct", "Worst drawdown": "pct"}


def reasons_for(m: dict) -> list[dict]:
    return [dict(r, fmt=_FMT.get(r["name"], "num")) for r in gate_reasons(m)]


def state() -> dict:
    settings = load_settings()
    runs = _runs()
    research = [r for r in runs if not r["holdout"]]
    scored = [r for r in research if r["m"].get("n_trades")]
    best = max(scored, key=lambda r: (r["m"].get("dsr") is not None, r["m"].get("dsr") or -1), default=None)
    holdout = next((r for r in runs if r["holdout"]), None)
    if best is None:
        verdict = dict(level="none", text="No results yet. Run 'Check setup', then 'Run everything'.")
    elif best["passed"]:
        verdict = dict(level="pass", text="Passed the research gate. Next step is paper trading, not real money.")
    else:
        verdict = dict(level="fail", text="Not good enough yet. Do not trade this with real money.")
    summary = dict(
        symbols=int(_q("SELECT COUNT(*) n FROM universe WHERE active=1")["n"][0]),
        price_rows=int(_q("SELECT COUNT(*) n FROM prices_daily")["n"][0]),
        last_price_date=_q("SELECT MAX(date) d FROM prices_daily")["d"][0],
        trials=int(_q("SELECT COUNT(*) n FROM trials")["n"][0]),
        open_data_issues=int(_q("SELECT COUNT(*) n FROM data_issues")["n"][0]),
        capital=settings["capital"])
    for r in runs:
        r["m"] = {k: v for k, v in r["m"].items() if k != "equity"} if r is not best and r is not holdout else r["m"]
    return clean(dict(
        summary=summary, verdict=verdict, actions=ACTIONS,
        best=dict(variant=best["variant"], ts=best["ts"], m=best["m"], reasons=reasons_for(best["m"])) if best else None,
        holdout=dict(ts=holdout["ts"], m=holdout["m"]) if holdout else None,
        runs=[dict(ts=r["ts"], variant=r["variant"], passed=r["passed"], holdout=r["holdout"],
                   m={k: r["m"].get(k) for k in ("n_trades", "net_pnl_after_tax", "dsr", "prob_net_profit",
                                                 "max_drawdown_pct", "n_trials")}) for r in runs[:15]],
        signals=_signals(settings["capital"], settings["risk"]),
        issues=_q("SELECT symbol, kind, detail FROM data_issues ORDER BY kind, symbol LIMIT 100").to_dict("records"),
        jobs=_q("SELECT ts, job, status, detail FROM job_log ORDER BY ts DESC LIMIT 12").to_dict("records"),
        github=github_runs()))


@app.get("/healthz")
def healthz():
    return {"ok": True}


@app.get("/api/state", dependencies=[Depends(auth)])
def api_state():
    return state()


@app.get("/", response_class=HTMLResponse, dependencies=[Depends(auth)])
def home():
    return PAGE


PAGE = r"""<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>tradingquant</title>
<style>
:root{--bg:#f6f7f9;--card:#fff;--ink:#14171c;--mute:#667085;--line:#e4e7ec;--ok:#067647;--okbg:#e7f6ee;--bad:#b42318;--badbg:#fdecea;--warn:#93370d;--warnbg:#fff4e5;--acc:#2447d6}
@media(prefers-color-scheme:dark){:root{--bg:#0e1116;--card:#171b22;--ink:#e7eaf0;--mute:#98a2b3;--line:#2a313c;--ok:#4ade80;--okbg:#10291c;--bad:#f87171;--badbg:#341616;--warn:#fbbf24;--warnbg:#33260c;--acc:#7c9bff}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:14px/1.45 system-ui,-apple-system,Segoe UI,Roboto,sans-serif}
main{max-width:1080px;margin:0 auto;padding:20px 16px 60px}h1{font-size:20px;margin:0}h2{font-size:15px;margin:0 0 10px}
.top{display:flex;justify-content:space-between;align-items:center;margin-bottom:14px;gap:12px;flex-wrap:wrap}
.card{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:16px;margin-bottom:14px}
.banner{background:var(--warnbg);color:var(--warn);border-radius:10px;padding:10px 14px;margin-bottom:14px;font-size:13px}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:10px}
.stat{border:1px solid var(--line);border-radius:10px;padding:10px 12px}.stat b{display:block;font-size:18px}.stat span{color:var(--mute);font-size:12px}
.verdict{border-radius:10px;padding:12px 14px;font-weight:600}.v-pass{background:var(--okbg);color:var(--ok)}.v-fail{background:var(--badbg);color:var(--bad)}.v-none{background:var(--bg);color:var(--mute)}
.btns{display:flex;gap:8px;flex-wrap:wrap}button{font:inherit;border:1px solid var(--line);background:var(--card);color:var(--ink);padding:8px 12px;border-radius:8px;cursor:pointer}
button.p{background:var(--acc);color:#fff;border-color:var(--acc)}button:disabled{opacity:.5;cursor:default}
table{width:100%;border-collapse:collapse;font-size:13px}th,td{text-align:left;padding:6px 8px;border-bottom:1px solid var(--line);vertical-align:top}th{color:var(--mute);font-weight:500;font-size:12px}
.tag{display:inline-block;padding:1px 8px;border-radius:99px;font-size:12px}.t-ok{background:var(--okbg);color:var(--ok)}.t-bad{background:var(--badbg);color:var(--bad)}.t-run{background:var(--warnbg);color:var(--warn)}
.mute{color:var(--mute)}.msg{margin-top:10px;font-size:13px}.scroll{overflow-x:auto}svg{width:100%;height:auto;display:block}
</style>
<main>
<div class="top"><h1>tradingquant</h1><span class="mute" id="asof"></span></div>
<div class="banner"><b>Research only. No orders are ever placed.</b> Backtests use today's listed stocks (survivorship bias flatters results), and some cost/tax rates are assumptions. A pass means "worth paper trading", never "safe to trade". No strategy is guaranteed to make money.</div>

<div class="card"><h2>Actions</h2><div class="btns" id="btns"></div><div class="msg" id="msg"></div>
<p class="mute" style="margin:10px 0 0">Order: Check setup, then Run everything (first time takes about an hour). After that it runs by itself every weekday at 4:45 pm IST.</p></div>

<div class="card"><h2>Verdict</h2><div id="verdict" class="verdict v-none">Loading…</div><div class="grid" id="stats" style="margin-top:12px"></div></div>

<div class="card"><h2>Why it passed or failed</h2><div class="scroll"><table id="reasons"></table></div>
<h2 style="margin-top:16px">Pretend account (₹ after all costs and tax)</h2><div id="chart" class="mute">No curve yet.</div></div>

<div class="card"><h2>Final one-time test</h2><div id="holdout" class="mute">Not run. This tests the best variant on the last year of data that research never touched. It can be run once only, so run it when research looks good.</div></div>

<div class="card"><h2>Latest signals <span class="mute">(research only)</span></h2><div class="scroll"><table id="signals"></table></div>
<p class="mute" style="margin:8px 0 0">Reference price is the last close; real fills differ. Quantity risks 1% of capital at the stop, capped by position limit. Delivery, long-only.</p></div>

<div class="card"><h2>Jobs</h2><div class="scroll"><table id="gh"></table></div><div class="scroll" style="margin-top:10px"><table id="jobs"></table></div></div>
<div class="card"><h2>Research history</h2><div class="scroll"><table id="runs"></table></div></div>
<div class="card"><h2>Data problems</h2><div class="scroll"><table id="issues"></table></div></div>
</main>
<script>
const $=id=>document.getElementById(id);
const esc=s=>String(s??"").replace(/[&<>"']/g,c=>({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}[c]));
const inr=v=>v==null?"–":"₹"+Math.round(v).toLocaleString("en-IN");
const pct=(v,d=0)=>v==null?"–":(v*100).toFixed(d)+"%";
const num=(v,d=2)=>v==null?"–":Number(v).toFixed(d);
function rows(el,head,body,empty){el.innerHTML=body.length?"<tr>"+head.map(h=>"<th>"+esc(h)+"</th>").join("")+"</tr>"+body.map(r=>"<tr>"+r.map(c=>"<td>"+c+"</td>").join("")+"</tr>").join(""):"<tr><td class='mute'>"+esc(empty)+"</td></tr>"}
function chart(pts){
  if(!pts||pts.length<2){$("chart").textContent="No curve yet.";return}
  const W=900,H=240,p=36,ys=pts.map(x=>x[1]),mn=Math.min(...ys),mx=Math.max(...ys),sp=(mx-mn)||1;
  const X=i=>p+i*(W-p-8)/(pts.length-1),Y=v=>H-p+4-(v-mn)*(H-p-14)/sp;
  const d=pts.map((q,i)=>(i?"L":"M")+X(i).toFixed(1)+" "+Y(q[1]).toFixed(1)).join("");
  $("chart").innerHTML=`<svg viewBox="0 0 ${W} ${H}" role="img" aria-label="Equity curve"><path d="${d}" fill="none" stroke="var(--acc)" stroke-width="2"/>
  <text x="${p}" y="14" fill="var(--mute)" font-size="12">${inr(mx)}</text><text x="${p}" y="${H-6}" fill="var(--mute)" font-size="12">${inr(mn)} · ${esc(pts[0][0])} to ${esc(pts[pts.length-1][0])}</text></svg>`}
function tagFor(s,c){return s=="completed"?(c=="success"?"<span class='tag t-ok'>done</span>":"<span class='tag t-bad'>"+esc(c||"failed")+"</span>"):"<span class='tag t-run'>"+esc(s)+"</span>"}
async function api(path,opt){const r=await fetch(path,opt);if(!r.ok){let t="";try{t=(await r.json()).detail}catch(e){}throw new Error(t||r.statusText)}return r.json()}
async function act(a,label){
  if(a=="holdout"&&!confirm("The final test can only be run once. Run it now?"))return;
  $("msg").textContent="Starting…";
  try{const r=await api("/api/action",{method:"POST",headers:{"Content-Type":"application/json","X-TQ":"1"},body:JSON.stringify({action:a})});$("msg").textContent=r.message;setTimeout(load,4000)}
  catch(e){$("msg").textContent="Could not start: "+e.message}}
async function load(){
  let s;try{s=await api("/api/state")}catch(e){$("verdict").textContent="Could not load: "+e.message;return}
  $("asof").textContent="Updated "+new Date().toLocaleTimeString();
  $("btns").innerHTML="";
  Object.entries(s.actions).forEach(([k,v])=>{const b=document.createElement("button");b.textContent=v;if(k=="all")b.className="p";b.onclick=()=>act(k,v);$("btns").appendChild(b)});
  const v=$("verdict");v.className="verdict v-"+s.verdict.level;v.textContent=s.verdict.text;
  const m=s.best?s.best.m:{},sm=s.summary;
  $("stats").innerHTML=[["Profit after tax",inr(m.net_pnl_after_tax)],["Chance of profit",pct(m.prob_net_profit)],["Trades",m.n_trades??"–"],["Worst drawdown",pct(m.max_drawdown_pct,1)],
   ["Luck-adjusted score (DSR)",num(m.dsr)],["Variants tried",sm.trials],["Stocks tracked",sm.symbols],["Latest price date",sm.last_price_date||"–"]]
   .map(([a,b])=>`<div class="stat"><b>${esc(b)}</b><span>${esc(a)}</span></div>`).join("");
  rows($("reasons"),["Test","Result","Needed",""],(s.best?s.best.reasons:[]).map(r=>[esc(r.name),esc(r.value==null?"–":r.fmt=="pct"?pct(r.value,1):r.fmt=="inr"?inr(r.value):r.fmt=="int"?r.value:num(r.value,3)),esc(r.need),r.ok?"<span class='tag t-ok'>pass</span>":"<span class='tag t-bad'>fail</span>"]),"No results yet.");
  chart(s.best&&s.best.m.equity);
  if(s.holdout){const h=s.holdout.m;$("holdout").innerHTML=`Run ${esc(s.holdout.ts.slice(0,16))}: ${esc(h.n_trades)} trades, profit after tax ${inr(h.net_pnl_after_tax)}, worst drawdown ${pct(h.max_drawdown_pct,1)}. This number is the honest one; do not re-run hoping for better.`}
  rows($("signals"),["Date","Stock","Score","Ref. price","Stop","Target","Qty","Status"],s.signals.map(x=>[esc(x.date),"<b>"+esc(x.symbol)+"</b>",num(x.score,3),num(x.ref_price),num(x.stop),num(x.target),esc(x.qty??"–"),esc(x.status.replace(/_/g," "))]),"No signals yet.");
  const g=s.github;
  rows($("gh"),["Started","Event","Status","Link"],g.runs.map(r=>[esc(r.created.replace("T"," ").slice(0,16)),esc(r.event),tagFor(r.status,r.conclusion),"<a href='"+esc(r.url)+"' target=_blank rel=noopener>open</a>"]),g.connected?"No runs yet.":("Not connected: "+(g.error||"")));
  rows($("jobs"),["Time","Job","Status","Detail"],s.jobs.map(j=>[esc(j.ts.slice(0,16)),esc(j.job),esc(j.status),esc((j.detail||"").slice(0,140))]),"Nothing logged yet.");
  rows($("runs"),["Time","Variant","Trades","Profit after tax","Chance","DSR","Gate"],s.runs.map(r=>[esc(r.ts.slice(0,16)),esc(r.variant),esc(r.m.n_trades??"–"),inr(r.m.net_pnl_after_tax),pct(r.m.prob_net_profit),num(r.m.dsr),r.holdout?"final test":(r.passed?"<span class='tag t-ok'>pass</span>":"<span class='tag t-bad'>no</span>")]),"No runs yet.");
  rows($("issues"),["Stock","Problem","Detail"],s.issues.map(i=>[esc(i.symbol),esc(i.kind.replace(/_/g," ")),esc(i.detail)]),"None.");
}
load();setInterval(load,30000);
</script></html>"""
