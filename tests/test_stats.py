import numpy as np
import pandas as pd
import pytest

from nsequant.config import load_costs
from nsequant.stats import (TrialRegistry, deflated_sharpe, expected_max_sharpe, profit_probability, sharpe)

TAX = load_costs()["tax"]


def test_expected_max_sharpe_grows_with_trials():
    v = 0.0001
    assert expected_max_sharpe(1, v) == 0
    assert expected_max_sharpe(10, v) < expected_max_sharpe(100, v) < expected_max_sharpe(1000, v)


def test_noise_sharpe_is_not_significant_after_many_trials():
    rng = np.random.default_rng(0)
    # best of 200 pure-noise strategies
    rets = [rng.normal(0, 0.01, 500) for _ in range(200)]
    srs = [sharpe(r) for r in rets]
    best = rets[int(np.argmax(srs))]
    res = deflated_sharpe(best, n_trials=200, sr_variance=float(np.var(srs, ddof=1)))
    assert sharpe(best) > 0.05                    # looks decent on its own
    assert res["dsr"] < 0.95                      # but is NOT significant once the 200 trials are counted


def test_real_edge_survives_deflation():
    rng = np.random.default_rng(1)
    r = rng.normal(0.002, 0.01, 1500)            # strong, persistent edge
    res = deflated_sharpe(r, n_trials=20, sr_variance=0.0004)
    assert res["dsr"] > 0.95


def test_registry_counts_unique_trials(tmp_path):
    reg = TrialRegistry(tmp_path / "t.jsonl")
    rng = np.random.default_rng(2)
    reg.log("rsi", {"p": 14}, rng.normal(0, 0.01, 100))
    reg.log("rsi", {"p": 14}, rng.normal(0, 0.01, 100))     # identical variant re-run
    reg.log("rsi", {"p": 7}, rng.normal(0, 0.01, 100))
    assert reg.n_trials() == 2


def _trades(n, mean, sd, seed=3):
    rng = np.random.default_rng(seed)
    gross = rng.normal(mean, sd, n)
    costs = np.full(n, 100.0)
    d0 = pd.Timestamp("2025-01-01")
    return pd.DataFrame({
        "segment": "equity_delivery", "holding_days": 3, "gross_pnl": gross, "costs": costs, "stt": 40.0,
        "net_pnl": gross - costs,
        "entry_date": [d0 + pd.Timedelta(days=int(i * 365 / n)) for i in range(n)],
        "exit_date": [d0 + pd.Timedelta(days=int(i * 365 / n) + 3) for i in range(n)]})


def test_no_edge_after_costs_has_low_profit_probability():
    pp = profit_probability(_trades(200, 0.0, 800.0), TAX, 100000, n_boot=300)
    assert pp.prob_net_profit_after_tax < 0.2         # zero gross edge minus costs minus tax -> mostly losing


def test_real_edge_has_high_profit_probability():
    pp = profit_probability(_trades(200, 600.0, 800.0), TAX, 100000, n_boot=300)
    assert pp.prob_net_profit_after_tax > 0.9


def test_too_few_trades_returns_nan():
    pp = profit_probability(_trades(10, 100.0, 50.0), TAX, 100000)
    assert np.isnan(pp.prob_net_profit_after_tax)
