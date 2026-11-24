"""Meta-labelling model: predicts whether a candidate trade will reach its TARGET before its STOP.

Barriers are volatility-scaled (ATR multiples) and the label comes from the same exit logic the
backtester uses, so training and testing agree on what a "win" means. Output probabilities are
only used as a ranking score and a threshold; never treat them as calibrated until checked.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .backtest import triple_barrier_label


def barriers(atr_pct: float, stop_k: float = 1.0, target_k: float = 2.0,
             lo: float = 0.01, hi: float = 0.06) -> tuple[float, float]:
    s = float(np.clip(stop_k * atr_pct, lo, hi))
    return s, float(np.clip(target_k * atr_pct, lo, hi * 2))


def make_dataset(bars: dict[str, pd.DataFrame], feats: dict[str, pd.DataFrame], dates: pd.DatetimeIndex,
                 max_hold: int = 5, entry_mode: str = "next_open", stop_k: float = 1.0,
                 target_k: float = 2.0) -> pd.DataFrame:
    rows = []
    dset = set(dates)
    for sym, f in feats.items():
        b = bars[sym]
        for d in f.index:
            if d not in dset:
                continue
            x = f.loc[d]
            if x.isna().any():
                continue
            s, t = barriers(float(x["atr14_pct"]), stop_k, target_k)
            lab = triple_barrier_label(b, d, 1, s, t, max_hold, entry_mode)
            if lab is None:
                continue
            rows.append(dict(date=d, symbol=sym, y=int(lab["label"] == 1), ret=lab["ret"],
                             stop_pct=s, target_pct=t, **x.to_dict()))
    return pd.DataFrame(rows)


def fit_meta(train: pd.DataFrame, feature_cols: list[str], C: float = 0.05, kind: str = "logit"):
    """kind='logit' is linear and hard to overfit; kind='hgb' captures interactions (needs more care)."""
    if train["y"].nunique() < 2 or len(train) < 200:
        return None
    if kind == "hgb":
        m = HistGradientBoostingClassifier(max_depth=3, learning_rate=0.05, max_iter=120,
                                           l2_regularization=5.0, min_samples_leaf=100, random_state=0)
    else:
        m = make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=500))
    m.fit(train[feature_cols].to_numpy(), train["y"].to_numpy())
    return m


def signals_from_model(model, feats: dict[str, pd.DataFrame], dates: pd.DatetimeIndex, feature_cols: list[str],
                       threshold: float, max_hold: int = 5, stop_k: float = 1.0, target_k: float = 2.0,
                       top_n_per_day: int = 3) -> pd.DataFrame:
    cols = ["date", "symbol", "side", "stop_pct", "target_pct", "max_hold", "score"]
    if model is None:
        return pd.DataFrame(columns=cols)
    rows = []
    dset = set(dates)
    for sym, f in feats.items():
        g = f[f.index.isin(dset)].dropna(subset=feature_cols + ["atr14_pct"])
        if g.empty:
            continue
        p = model.predict_proba(g[feature_cols].to_numpy())[:, 1]
        for d, pr, atr in zip(g.index, p, g["atr14_pct"]):
            if pr >= threshold:
                s, t = barriers(float(atr), stop_k, target_k)
                rows.append(dict(date=d, symbol=sym, side=1, stop_pct=s, target_pct=t,
                                 max_hold=max_hold, score=float(pr)))
    if not rows:
        return pd.DataFrame(columns=cols)
    out = pd.DataFrame(rows)
    out = out.sort_values(["date", "score"], ascending=[True, False]).groupby("date").head(top_n_per_day)
    return out.reset_index(drop=True)
