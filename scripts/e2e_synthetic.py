"""End-to-end plumbing check on synthetic data: no-edge control vs injected edge.

Run:  python scripts/e2e_synthetic.py
Expectation: control must NOT look profitable after costs+tax; the injected edge should be detected.
"""
import copy
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from nsequant.backtest import run_backtest, walk_forward_splits
from nsequant.config import load_costs, load_settings
from nsequant.features import build_features
from nsequant.models import fit_meta, make_dataset, signals_from_model
from nsequant.stats import TrialRegistry, profit_probability
from nsequant.synthetic import make_market

COSTS, SET = load_costs(), load_settings()
SET = copy.deepcopy(SET)
SET["backtest"]["walk_forward"] = dict(train_days=756, test_days=126, embargo_days=10)
QUANTILES = (0.85, 0.90, 0.95)
GATE = float(sys.argv[1]) if len(sys.argv) > 1 else SET["risk"]["min_net_reward_risk"]
SET["risk"]["min_net_reward_risk"] = GATE
print(f"net reward:risk gate = {GATE}")


def scenario(name, edge, reg):
    t0 = time.time()
    bars, idxc = make_market(n_symbols=30, n_days=1800, edge_drift=edge)
    feats = {s: build_features(b, idxc) for s, b in bars.items()}
    cal = next(iter(bars.values())).index
    cols = [c for c in next(iter(feats.values())).columns]
    full = make_dataset(bars, feats, cal, max_hold=5)
    cols = [c for c in cols if c in full.columns]
    print(f"[{name}] dataset rows={len(full)} base target-hit rate={full['y'].mean():.3f} ({time.time()-t0:.0f}s)")
    folds = []
    for tr, te in walk_forward_splits(cal, 756, 126, 10):
        train = full[full["date"].isin(tr)]
        m = fit_meta(train, cols, kind="hgb")
        scores = m.predict_proba(train[cols].to_numpy())[:, 1] if m is not None else None
        folds.append((m, scores, te))
    out = {}
    for q in QUANTILES:
        parts = []
        for m, scores, te in folds:
            if m is None:
                continue
            thr = float(np.quantile(scores, q))
            parts.append(signals_from_model(m, feats, te, cols, thr, max_hold=5))
        sig = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
        res = run_backtest(bars, sig, COSTS, SET)
        rets = res.equity.pct_change().dropna() if len(res.equity) else pd.Series(dtype=float)
        reg.log(f"{name}", {"quantile": q, "model": "hgb"}, rets)
        out[q] = (res, rets)
    return out


def report(name, out, reg):
    print(f"\n=== {name} ===  trials logged so far: {reg.n_trials()}")
    for q, (res, rets) in out.items():
        s = res.summary
        if s.get("n_trades", 0) == 0:
            print(f" q={q}: no trades. rejections={res.rejections}")
            continue
        pp = profit_probability(res.trades, COSTS["tax"], SET["capital"], n_boot=300)
        d = reg.deflate(rets)
        print(f" q={q}: trades={s['n_trades']:4d} win={s['win_rate']:.2f} gross={s['gross_pnl']:9.0f} "
              f"costs={s['costs']:8.0f} net_pre_tax={s['net_pnl_pre_tax']:9.0f} tax={s['tax']:7.0f} "
              f"AFTER_TAX={s['net_pnl_after_tax']:9.0f} maxDD={s['max_drawdown_pct']:.1%} | "
              f"P(profit/yr,after tax)={pp.prob_net_profit_after_tax:.2f} DSR={d['dsr']:.2f}")
    r = next(iter(out.values()))[0].rejections
    print(" rejections (first variant):", r)


if __name__ == "__main__":
    with tempfile.TemporaryDirectory() as td:
        for name, edge in (("NO-EDGE CONTROL", 0.0), ("INJECTED EDGE (+1.5%/3d)", 0.015)):
            # One registry per research question + dataset. Pooling unrelated experiments distorts DSR.
            reg = TrialRegistry(Path(td) / f"{abs(hash(name))}.jsonl")
            out = scenario(name, edge, reg)
            report(name, out, reg)
