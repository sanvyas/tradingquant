import numpy as np
import pandas as pd

from nsequant.features import build_features, external_features


def _ohlcv(n=400, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2022-01-03", periods=n)
    r = rng.normal(0.0003, 0.012, n)
    c = 100 * np.exp(np.cumsum(r))
    o = np.r_[c[0], c[:-1]] * (1 + rng.normal(0, 0.003, n))
    h = np.maximum(o, c) * (1 + np.abs(rng.normal(0, 0.004, n)))
    l = np.minimum(o, c) * (1 - np.abs(rng.normal(0, 0.004, n)))
    v = rng.integers(500_000, 2_000_000, n).astype(float)
    return pd.DataFrame(dict(open=o, high=h, low=l, close=c, volume=v), index=idx)


def test_features_have_no_lookahead():
    """Changing future bars must not change any feature value on earlier dates."""
    df = _ohlcv()
    idxc = df["close"] * 1.01
    full = build_features(df, idxc)
    cut = 300
    df2 = df.copy()
    rng = np.random.default_rng(9)
    for col in ("open", "high", "low", "close", "volume"):
        df2.iloc[cut:, df2.columns.get_loc(col)] *= rng.uniform(0.5, 1.5, len(df2) - cut)
    changed = build_features(df2, idxc.where(df.index < df.index[cut], idxc * 1.7))
    a, b = full.iloc[:cut], changed.iloc[:cut]
    pd.testing.assert_frame_equal(a, b, check_exact=False, rtol=1e-9, atol=1e-12)


def test_feature_columns_present_and_finite_after_warmup():
    df = _ohlcv()
    f = build_features(df, df["close"] * 1.0)
    for col in ("rsi14", "gap_z", "efficiency_20", "variance_ratio_5", "vol_regime", "resid_ret_1", "atr14_pct"):
        assert col in f.columns
        assert np.isfinite(f[col].iloc[-1])
    assert (f["rsi14"].dropna().between(0, 100)).all()


def test_foreign_series_default_lag_is_one_day():
    idx = pd.bdate_range("2024-01-01", periods=10)
    ext = pd.DataFrame({"sp500": np.arange(100.0, 110.0)}, index=idx)
    safe = external_features(idx, ext)                       # default lag 1
    raw = external_features(idx, ext, {"sp500": 0})
    # at day 5, the lag-1 feature must reflect day 4's move, not day 5's
    assert safe["ext_sp500_ret_1"].iloc[5] == raw["ext_sp500_ret_1"].iloc[4]
