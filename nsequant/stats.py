"""Honest statistics: count every trial, deflate the Sharpe ratio, and estimate profit probability.

Why this exists: if you try 500 strategy variants, the best one looks great by luck alone.
The TrialRegistry records every variant you evaluate, and `deflated_sharpe` asks:
"how likely is this Sharpe to beat what the best of N lucky random strategies would show?"
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as sps

from .tax import tax_for_trades

EULER_GAMMA = 0.5772156649


def sharpe(returns: pd.Series | np.ndarray) -> float:
    r = np.asarray(returns, dtype=float)
    r = r[~np.isnan(r)]
    if len(r) < 3 or r.std(ddof=1) == 0:
        return 0.0
    return float(r.mean() / r.std(ddof=1))        # per-period, not annualised


def annualised_sharpe(returns, periods: int = 252) -> float:
    return sharpe(returns) * np.sqrt(periods)


def expected_max_sharpe(n_trials: int, sr_variance: float) -> float:
    """Expected best Sharpe among n_trials strategies with NO real edge (Bailey & Lopez de Prado)."""
    if n_trials <= 1:
        return 0.0
    z1 = sps.norm.ppf(1 - 1.0 / n_trials)
    z2 = sps.norm.ppf(1 - 1.0 / (n_trials * np.e))
    return float(np.sqrt(max(sr_variance, 0.0)) * ((1 - EULER_GAMMA) * z1 + EULER_GAMMA * z2))


def deflated_sharpe(returns, n_trials: int, sr_variance: float) -> dict:
    """Probability the observed Sharpe is genuine after accounting for n_trials tried and fat tails."""
    r = np.asarray(returns, dtype=float)
    r = r[~np.isnan(r)]
    n = len(r)
    if n < 30:
        return {"dsr": float("nan"), "sr": sharpe(r), "sr0": float("nan"), "n_obs": n}
    sr = sharpe(r)
    sr0 = expected_max_sharpe(n_trials, sr_variance)
    skew, kurt = sps.skew(r), sps.kurtosis(r, fisher=False)
    denom = np.sqrt(max(1 - skew * sr + (kurt - 1) / 4 * sr ** 2, 1e-12))
    z = (sr - sr0) * np.sqrt(n - 1) / denom
    return {"dsr": float(sps.norm.cdf(z)), "sr": sr, "sr0": sr0, "n_obs": n}


class TrialRegistry:
    """Append-only log of every strategy variant evaluated. Never skip logging a failed variant."""

    def __init__(self, path: str | Path = "data/trials.jsonl"):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def log(self, name: str, params: dict, returns) -> dict:
        key = hashlib.sha1(json.dumps([name, params], sort_keys=True, default=str).encode()).hexdigest()[:12]
        rec = {"key": key, "name": name, "params": params, "sr": sharpe(returns),
               "n_obs": int(np.sum(~np.isnan(np.asarray(returns, dtype=float))))}
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, default=str) + "\n")
        return rec

    def records(self) -> list[dict]:
        if not self.path.exists():
            return []
        recs = [json.loads(x) for x in self.path.read_text().splitlines() if x.strip()]
        seen = {}
        for r in recs:                      # re-running the identical variant is not a new trial
            seen[r["key"]] = r
        return list(seen.values())

    def n_trials(self) -> int:
        return len(self.records())

    def sr_variance(self) -> float:
        srs = [r["sr"] for r in self.records()]
        return float(np.var(srs, ddof=1)) if len(srs) > 1 else 0.0

    def deflate(self, returns) -> dict:
        n, v = max(self.n_trials(), 1), self.sr_variance()
        out = deflated_sharpe(returns, n, v)
        out["n_trials"] = n
        return out


def drawdown_pct(equity: pd.Series) -> float:
    return float((equity / equity.cummax() - 1).min()) if len(equity) else 0.0


@dataclass
class ProfitProbability:
    prob_net_profit_after_tax: float
    prob_drawdown_exceeds: float
    drawdown_threshold: float
    ev_per_trade_after_costs: float
    after_tax_pnl_p05: float
    after_tax_pnl_median: float
    after_tax_pnl_p95: float
    n_trades_per_horizon: int
    horizon_years: float


def profit_probability(trades: pd.DataFrame, tax_cfg: dict, capital: float, horizon_years: float = 1.0,
                       n_boot: int = 1000, block: int = 5, dd_threshold: float = 0.25,
                       seed: int = 7) -> ProfitProbability:
    """Bootstrap the *trade list* to estimate the chance of net profit AFTER costs and tax.

    Trades are resampled in short consecutive blocks (keeps some streak/clustering), laid out as
    one financial year at a time, and taxed with the same function the backtest uses.
    Caveat: this reflects the past trade distribution only; a regime change breaks it.
    """
    if trades is None or len(trades) < 20:
        return ProfitProbability(float("nan"), float("nan"), dd_threshold, float("nan"),
                                 float("nan"), float("nan"), float("nan"), 0, horizon_years)
    t = trades.sort_values("entry_date").reset_index(drop=True)
    years = max((t["exit_date"].max() - t["entry_date"].min()).days / 365.25, 0.25)
    m = max(int(round(len(t) / years * horizon_years)), 1)
    rng = np.random.default_rng(seed)
    cols = ["segment", "holding_days", "gross_pnl", "costs", "stt", "net_pnl"]
    arr = t[cols].to_dict("records")
    n = len(arr)
    outs, ruin = [], 0
    for _ in range(n_boot):
        picks = []
        while len(picks) < m:
            s = int(rng.integers(0, n))
            picks.extend(range(s, min(s + block, n)))
        picks = picks[:m]
        samp = pd.DataFrame([arr[i] for i in picks])
        samp["fy"] = "sim"
        samp = samp.rename(columns={"gross_pnl": "pnl_gross"})
        tax = float(tax_for_trades(samp, tax_cfg)["tax"].sum())
        path = capital + samp["net_pnl"].cumsum().to_numpy()
        peak = np.maximum.accumulate(np.concatenate([[capital], path]))[1:]
        if (path / peak - 1).min() < -dd_threshold:
            ruin += 1
        outs.append(float(samp["net_pnl"].sum()) - tax)
    o = np.array(outs)
    return ProfitProbability(
        prob_net_profit_after_tax=float((o > 0).mean()), prob_drawdown_exceeds=ruin / n_boot,
        drawdown_threshold=dd_threshold, ev_per_trade_after_costs=float(t["net_pnl"].mean()),
        after_tax_pnl_p05=float(np.percentile(o, 5)), after_tax_pnl_median=float(np.median(o)),
        after_tax_pnl_p95=float(np.percentile(o, 95)), n_trades_per_horizon=m, horizon_years=horizon_years)


class DbTrialRegistry(TrialRegistry):
    """Same interface as TrialRegistry, stored in the database. One scope per idea + dataset."""

    def __init__(self, engine, scope: str):
        self.engine, self.scope = engine, scope

    def log(self, name: str, params: dict, returns) -> dict:
        from sqlalchemy import text
        key = hashlib.sha1(json.dumps([name, params], sort_keys=True, default=str).encode()).hexdigest()[:12]
        rec = {"key": key, "name": name, "params": params, "sr": sharpe(returns),
               "n_obs": int(np.sum(~np.isnan(np.asarray(returns, dtype=float))))}
        with self.engine.begin() as c:
            c.execute(text("""INSERT INTO trials(scope,key,name,params,sr,n_obs,created_at)
                VALUES (:sc,:k,:n,:p,:sr,:no,:t)
                ON CONFLICT (scope,key) DO UPDATE SET sr=excluded.sr, n_obs=excluded.n_obs"""),
                      dict(sc=self.scope, k=key, n=name, p=json.dumps(params, default=str), sr=rec["sr"],
                           no=rec["n_obs"], t=pd.Timestamp.now(tz="UTC").isoformat()))
        return rec

    def records(self) -> list[dict]:
        from sqlalchemy import text
        with self.engine.connect() as c:
            rows = c.execute(text("SELECT key,name,params,sr,n_obs FROM trials WHERE scope=:s"),
                             dict(s=self.scope)).fetchall()
        return [dict(key=r[0], name=r[1], params=json.loads(r[2] or "{}"), sr=r[3], n_obs=r[4]) for r in rows]
