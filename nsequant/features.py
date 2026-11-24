"""Feature library. Point-in-time safe: the row for date t uses only information known at the close of t.

Two groups on purpose:
  * standard    - RSI, MACD, ATR, Bollinger, ADX, stochastic. Everyone has these; treat them as
                  raw material, not as an edge.
  * structural  - gap behaviour, close location, range compression, trend efficiency, variance ratio,
                  volatility regime, volume anomalies, residual (beta-adjusted) returns, relative
                  strength, distance to highs, global/cross-asset context, delivery % when supplied.
Feature value is decided by out-of-sample tests, not by how well known an indicator is.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def _rsi(c: pd.Series, n: int = 14) -> pd.Series:
    d = c.diff()
    up = d.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    return 100 - 100 / (1 + up / dn.replace(0, np.nan))


def _atr(df: pd.DataFrame, n: int = 14) -> pd.Series:
    pc = df["close"].shift(1)
    tr = pd.concat([df["high"] - df["low"], (df["high"] - pc).abs(), (df["low"] - pc).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / n, adjust=False).mean()


def _adx(df: pd.DataFrame, n: int = 14) -> pd.Series:
    up, dn = df["high"].diff(), -df["low"].diff()
    plus = np.where((up > dn) & (up > 0), up, 0.0)
    minus = np.where((dn > up) & (dn > 0), dn, 0.0)
    atr = _atr(df, n)
    pdi = 100 * pd.Series(plus, index=df.index).ewm(alpha=1 / n, adjust=False).mean() / atr
    mdi = 100 * pd.Series(minus, index=df.index).ewm(alpha=1 / n, adjust=False).mean() / atr
    dx = 100 * (pdi - mdi).abs() / (pdi + mdi).replace(0, np.nan)
    return dx.ewm(alpha=1 / n, adjust=False).mean()


def standard_features(df: pd.DataFrame) -> pd.DataFrame:
    c = df["close"]
    f = pd.DataFrame(index=df.index)
    for n in (1, 5, 20):
        f[f"ret_{n}"] = c.pct_change(n)
    for n in (20, 50):
        f[f"sma_dist_{n}"] = c / c.rolling(n).mean() - 1
    f["rsi14"] = _rsi(c)
    macd = c.ewm(span=12, adjust=False).mean() - c.ewm(span=26, adjust=False).mean()
    f["macd_hist"] = (macd - macd.ewm(span=9, adjust=False).mean()) / c
    f["atr14_pct"] = _atr(df) / c
    ma, sd = c.rolling(20).mean(), c.rolling(20).std()
    f["bb_pctb"] = (c - (ma - 2 * sd)) / (4 * sd).replace(0, np.nan)
    f["adx14"] = _adx(df)
    lo, hi = df["low"].rolling(14).min(), df["high"].rolling(14).max()
    f["stoch_k"] = 100 * (c - lo) / (hi - lo).replace(0, np.nan)
    return f


def structural_features(df: pd.DataFrame, index_close: pd.Series | None = None) -> pd.DataFrame:
    c, o, h, l, v = (df[k] for k in ("close", "open", "high", "low", "volume"))
    r1 = c.pct_change()
    f = pd.DataFrame(index=df.index)
    atr_pct = _atr(df) / c
    f["gap_pct"] = o / c.shift(1) - 1
    f["gap_z"] = f["gap_pct"] / atr_pct.shift(1)
    rng = (h - l).replace(0, np.nan)
    clv = ((c - l) - (h - c)) / rng
    f["clv"] = clv
    f["clv_ema5"] = clv.ewm(span=5, adjust=False).mean()
    f["range_ratio"] = (h - l) / (h - l).rolling(20).mean()
    f["nr7"] = ((h - l) <= (h - l).rolling(7).min()).astype(float)
    net = (c - c.shift(20)).abs()
    path = c.diff().abs().rolling(20).sum()
    f["efficiency_20"] = net / path.replace(0, np.nan)
    var1 = r1.rolling(60).var()
    r5 = c.pct_change(5)
    f["variance_ratio_5"] = r5.rolling(60).var() / (5 * var1)
    rv = r1.rolling(20).std()
    f["vol_regime"] = rv.rolling(252, min_periods=60).rank(pct=True)
    f["vol_of_vol"] = rv.rolling(20).std() / rv
    f["vol_z"] = (v - v.rolling(20).mean()) / v.rolling(20).std().replace(0, np.nan)
    up_v = v.where(r1 > 0, 0.0).rolling(20).sum()
    f["up_volume_ratio"] = up_v / v.rolling(20).sum().replace(0, np.nan)
    f["median_value_traded_20"] = (c * v).rolling(20).median()
    hi252 = h.rolling(252, min_periods=60).max()
    f["dist_52w_high"] = c / hi252 - 1
    if "delivery_pct" in df.columns:
        d = df["delivery_pct"]
        f["delivery_z"] = (d - d.rolling(60).mean()) / d.rolling(60).std().replace(0, np.nan)
    if index_close is not None:
        ic = index_close.reindex(df.index).ffill()
        ir1 = ic.pct_change()
        beta = r1.rolling(60).cov(ir1) / ir1.rolling(60).var()
        f["beta_60"] = beta
        f["resid_ret_1"] = r1 - beta.shift(1) * ir1             # beta known BEFORE today's move
        f["rel_strength_20"] = c.pct_change(20) - ic.pct_change(20)
    return f


def external_features(index: pd.DatetimeIndex, ext: pd.DataFrame | None, lags: dict[str, int] | None = None) -> pd.DataFrame:
    """Cross-asset context aligned so only information known at the Indian close of t is used.

    `lags[name]` is in India trading days. A US/Europe series needs lag 1: the US session of date t
    ends after the Indian close of t. India-session series (India VIX, USDINR at 15:30) can use 0.
    Default is lag 1, the safe choice.
    """
    if ext is None or ext.empty:
        return pd.DataFrame(index=index)
    lags = lags or {}
    out = pd.DataFrame(index=index)
    for name in ext.columns:
        s = ext[name].reindex(index.union(ext.index)).ffill().reindex(index)
        s = s.shift(lags.get(name, 1))
        out[f"ext_{name}_ret_1"] = s.pct_change()
        out[f"ext_{name}_ret_5"] = s.pct_change(5)
    return out


def calendar_features(index: pd.DatetimeIndex) -> pd.DataFrame:
    f = pd.DataFrame(index=index)
    f["dow"] = index.dayofweek
    f["days_to_month_end"] = (index + pd.offsets.MonthEnd(0) - index).days
    return f


def build_features(df: pd.DataFrame, index_close: pd.Series | None = None, ext: pd.DataFrame | None = None,
                   ext_lags: dict[str, int] | None = None) -> pd.DataFrame:
    parts = [standard_features(df), structural_features(df, index_close),
             external_features(df.index, ext, ext_lags), calendar_features(df.index)]
    return pd.concat(parts, axis=1).replace([np.inf, -np.inf], np.nan)
