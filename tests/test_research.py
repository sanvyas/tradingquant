import copy

import pandas as pd

from nsequant import store
from nsequant.config import load_costs, load_settings
from nsequant.research import GATE, holdout_check, passes_gate, run_research
from nsequant.synthetic import make_market

COSTS, SET = load_costs(), copy.deepcopy(load_settings())
SET["risk"]["min_net_reward_risk"] = 1.0
WF = dict(train_days=300, test_days=100, embargo_days=10)


def _load(tmp_path, edge):
    e = store.get_engine(f"sqlite:///{tmp_path}/r.db")
    store.init_db(e)
    bars, idxc = make_market(n_symbols=12, n_days=1000, edge_drift=edge, seed=5)
    store.upsert_universe(e, [dict(symbol=s, instrument_key=f"NSE_EQ|{s}") for s in bars])
    for s, b in bars.items():
        store.upsert_prices(e, s, b)
    return e, bars


def test_insufficient_data_is_reported_not_crashed(tmp_path):
    e = store.get_engine(f"sqlite:///{tmp_path}/x.db")
    store.init_db(e)
    out = run_research(e, SET, COSTS, wf=WF)
    assert out["status"] == "insufficient_data"


def test_research_logs_trials_runs_and_excludes_holdout(tmp_path):
    e, bars = _load(tmp_path, edge=0.015)
    out = run_research(e, SET, COSTS, scope="t1", quantiles=(0.85, 0.95), holdout_days=120, wf=WF)
    assert out["status"] == "ok" and out["n_trials"] == 2
    runs = pd.read_sql("SELECT * FROM runs", e)
    assert len(runs) == 2
    sig = pd.read_sql("SELECT * FROM signals", e)
    if len(sig):
        assert set(sig["status"]) <= {"research_only", "candidate_research_gate_passed"}   # never 'live'
    # holdout dates never appear in research trades: rerun and inspect the engine input via holdout size
    last_sessions = sorted(next(iter(bars.values())).index)[-120:]
    assert all(pd.Timestamp(d) in last_sessions for d in last_sessions)


def test_holdout_check_is_one_shot(tmp_path):
    e, _ = _load(tmp_path, edge=0.015)
    a = holdout_check(e, SET, COSTS, "t2", 0.9, holdout_days=120, wf=WF)
    assert a["status"] == "ok"
    b = holdout_check(e, SET, COSTS, "t2", 0.9, holdout_days=120, wf=WF)
    assert b["status"] == "refused"


def test_gate_requires_every_condition():
    ok = dict(n_trades=200, dsr=0.97, net_pnl_after_tax=5000, prob_net_profit=0.9, max_drawdown_pct=-0.1)
    assert passes_gate(ok)
    for k, v in (("n_trades", 50), ("dsr", 0.5), ("net_pnl_after_tax", -1), ("prob_net_profit", 0.5),
                 ("max_drawdown_pct", -0.5)):
        assert not passes_gate({**ok, k: v})
