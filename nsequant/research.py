"""Automated research: walk-forward evaluation of the meta-labelling strategy on stored prices.

Guard rails built in:
  * The last `holdout_days` sessions are excluded from ALL research (features still see only the past;
    labels are built from research-period bars only). Look at them once, with holdout_check.
  * Every variant is logged in the database trial registry; the Deflated Sharpe uses the final trial count.
  * Symbols with blocking data-quality issues are excluded.
  * Passing the research gate is NOT permission to trade: it only means "worth paper trading".
"""
from __future__ import annotations

import uuid
from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import store
from .backtest import run_backtest, walk_forward_splits
from .data_quality import BLOCKING
from .features import build_features
from .models import fit_meta, make_dataset, signals_from_model
from .stats import DbTrialRegistry, profit_probability

GATE = dict(dsr=0.95, prob_profit=0.80, min_trades=100, max_drawdown=-0.30)


@dataclass
class Prepared:
    bars_full: dict
    bars_r: dict
    feats: dict
    cal_full: pd.DatetimeIndex
    cal_r: pd.DatetimeIndex
    holdout_dates: pd.DatetimeIndex
    cols: list
    dataset: pd.DataFrame
    market_proxy: str


def prepare(eng, holdout_days: int, wf: dict, max_hold: int = 5, min_symbols: int = 8):
    uni = store.get_universe(eng, role="stock")
    bad = store.flagged_symbols(eng, BLOCKING)
    symbols = [s for s in uni["symbol"] if s not in bad]
    bars = store.load_prices(eng, symbols)
    bars = {s: b for s, b in bars.items() if len(b) >= 300}
    index_df = store.load_prices(eng, ["NIFTY50"]).get("NIFTY50")
    need = wf["train_days"] + wf["embargo_days"] + wf["test_days"] + holdout_days
    if len(bars) < min_symbols:
        return dict(status="insufficient_data", reason=f"only {len(bars)} usable symbols (need {min_symbols})")
    cal_full = pd.DatetimeIndex(sorted(set().union(*[set(b.index) for b in bars.values()])))
    if len(cal_full) < need + 60:
        return dict(status="insufficient_data", reason=f"{len(cal_full)} sessions, need about {need + 60}")
    if index_df is not None:
        index_close, proxy = index_df["close"], "NIFTY50"
    else:                                                     # fall back to an equal-weight market proxy
        norm = pd.concat([b["close"] / b["close"].iloc[0] for b in bars.values()], axis=1).mean(axis=1)
        index_close, proxy = norm, "equal_weight_proxy"
    cal_r, holdout = cal_full[:-holdout_days], cal_full[-holdout_days:]
    bars_r = {s: b.loc[:cal_r[-1]] for s, b in bars.items()}
    feats = {s: build_features(b, index_close) for s, b in bars.items()}
    cols = [c for c in next(iter(feats.values())).columns]
    full = make_dataset(bars_r, feats, cal_r, max_hold=max_hold)
    cols = [c for c in cols if c in full.columns]
    return Prepared(bars, bars_r, feats, cal_full, cal_r, holdout, cols, full, proxy)


def fit_folds(prep: Prepared, wf: dict, kind: str = "hgb") -> list:
    folds = []
    for tr, te in walk_forward_splits(prep.cal_r, wf["train_days"], wf["test_days"], wf["embargo_days"]):
        train = prep.dataset[prep.dataset["date"].isin(tr)]
        m = fit_meta(train, prep.cols, kind=kind)
        scores = m.predict_proba(train[prep.cols].to_numpy())[:, 1] if m is not None else None
        folds.append((m, scores, te))
    return folds


def oos_signals(prep: Prepared, folds: list, q: float, max_hold: int = 5) -> pd.DataFrame:
    parts = []
    for m, scores, te in folds:
        if m is None:
            continue
        parts.append(signals_from_model(m, prep.feats, te, prep.cols, float(np.quantile(scores, q)), max_hold))
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def passes_gate(m: dict) -> bool:
    return bool(m.get("n_trades", 0) >= GATE["min_trades"] and m.get("dsr", 0) >= GATE["dsr"]
                and m.get("net_pnl_after_tax", -1) > 0 and m.get("prob_net_profit", 0) >= GATE["prob_profit"]
                and m.get("max_drawdown_pct", -1) >= GATE["max_drawdown"])


def metrics_for(res, rets, reg, costs, capital) -> dict:
    s = dict(res.summary)
    if s.get("n_trades", 0) == 0:
        return dict(n_trades=0, rejections=res.rejections, dsr=float("nan"), n_trials=reg.n_trials())
    pp = profit_probability(res.trades, costs["tax"], capital, n_boot=300)
    d = reg.deflate(rets)
    s.update(prob_net_profit=pp.prob_net_profit_after_tax, prob_dd_gt_25pct=pp.prob_drawdown_exceeds,
             after_tax_p05=pp.after_tax_pnl_p05, after_tax_median=pp.after_tax_pnl_median,
             dsr=d["dsr"], n_trials=d["n_trials"], rejections=res.rejections)
    return s


def run_research(eng, settings: dict, costs: dict, scope: str = "meta_hgb_v1", quantiles=(0.85, 0.90, 0.95),
                 holdout_days: int = 252, wf: dict | None = None, kind: str = "hgb") -> dict:
    wf = wf or settings["backtest"]["walk_forward"]
    prep = prepare(eng, holdout_days, wf)
    if isinstance(prep, dict):
        store.log_job(eng, "research_nightly", "skipped", prep["reason"])
        return prep
    folds = fit_folds(prep, wf, kind)
    reg = DbTrialRegistry(eng, scope)
    outs = {}
    for q in quantiles:
        sig = oos_signals(prep, folds, q)
        res = run_backtest(prep.bars_r, sig, costs, settings)
        rets = res.equity.pct_change().dropna() if len(res.equity) else pd.Series(dtype=float)
        reg.log(scope, dict(quantile=q, kind=kind), rets)
        outs[q] = (res, rets)
    summary, best = {}, None
    for q, (res, rets) in outs.items():
        m = metrics_for(res, rets, reg, costs, settings["capital"])
        ok = passes_gate(m)
        m["market_proxy"], m["holdout_days"] = prep.market_proxy, holdout_days
        store.save_run(eng, f"{scope}-{pd.Timestamp.now().strftime('%Y%m%d%H%M%S')}-q{q}-{uuid.uuid4().hex[:4]}",
                       scope, f"quantile={q}", m, ok)
        summary[q] = dict(passed_gate=ok, **{k: m.get(k) for k in
                          ("n_trades", "net_pnl_after_tax", "dsr", "prob_net_profit", "max_drawdown_pct")})
        if m.get("n_trades", 0) and (best is None or (m.get("dsr") or 0) > (best[1].get("dsr") or 0)):
            best = (q, m, ok)
    if best:
        _write_signals(eng, prep, folds, best, scope, wf)
    store.log_job(eng, "research_nightly", "ok", str(summary)[:1500])
    return dict(status="ok", variants=summary, n_trials=reg.n_trials(), market_proxy=prep.market_proxy)


def _write_signals(eng, prep: Prepared, folds: list, best, scope: str, wf: dict) -> None:
    """Signals for the latest session from a model fitted on the most recent window. Never marked live."""
    q, _, ok = best
    last = prep.cal_full[-1]
    recent = prep.cal_full[-(wf["train_days"] + wf["embargo_days"] + 5):-(wf["embargo_days"] + 5)]
    full = make_dataset(prep.bars_full, prep.feats, recent, max_hold=5)
    m = fit_meta(full, prep.cols, kind="hgb")
    if m is None:
        return
    thr = float(np.quantile(m.predict_proba(full[prep.cols].to_numpy())[:, 1], q))
    sig = signals_from_model(m, prep.feats, pd.DatetimeIndex([last]), prep.cols, thr, 5)
    status = "candidate_research_gate_passed" if ok else "research_only"
    store.save_signals(eng, [dict(date=last.strftime("%Y-%m-%d"), symbol=r.symbol, side=int(r.side),
                                  stop_pct=float(r.stop_pct), target_pct=float(r.target_pct), score=float(r.score),
                                  status=status, scope=scope) for r in sig.itertuples()])


def holdout_check(eng, settings: dict, costs: dict, scope: str, quantile: float, holdout_days: int = 252,
                  wf: dict | None = None, kind: str = "hgb", force: bool = False) -> dict:
    """Evaluate ONE chosen variant on the untouched holdout. Allowed once per scope, because every extra look
    turns the holdout into more training data."""
    wf = wf or settings["backtest"]["walk_forward"]
    with eng.connect() as c:
        from sqlalchemy import text
        done = c.execute(text("SELECT COUNT(*) FROM runs WHERE scope=:s AND variant LIKE 'HOLDOUT%'"),
                         dict(s=scope)).scalar()
    if done and not force:
        return dict(status="refused", reason="holdout already used for this scope")
    prep = prepare(eng, holdout_days, wf)
    if isinstance(prep, dict):
        return prep
    train = prep.dataset[prep.dataset["date"].isin(prep.cal_r[-wf["train_days"]:])]
    m = fit_meta(train, prep.cols, kind=kind)
    if m is None:
        return dict(status="no_model")
    thr = float(np.quantile(m.predict_proba(train[prep.cols].to_numpy())[:, 1], quantile))
    sig = signals_from_model(m, prep.feats, prep.holdout_dates, prep.cols, thr, 5)
    res = run_backtest(prep.bars_full, sig, costs, settings)
    out = dict(res.summary, rejections=res.rejections)
    store.save_run(eng, f"{scope}-HOLDOUT-{uuid.uuid4().hex[:6]}", scope, f"HOLDOUT quantile={quantile}", out, False)
    return dict(status="ok", **out)
