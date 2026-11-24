import numpy as np
import pandas as pd

from nsequant import store
from nsequant.stats import DbTrialRegistry


def _eng(tmp_path):
    e = store.get_engine(f"sqlite:///{tmp_path}/t.db")
    store.init_db(e)
    return e


def _df(n=5, start="2026-01-01", base=100.0):
    idx = pd.bdate_range(start, periods=n)
    return pd.DataFrame(dict(open=base, high=base + 1, low=base - 1, close=base + 0.5, volume=1000.0), index=idx)


def test_prices_upsert_is_idempotent_and_updates(tmp_path):
    e = _eng(tmp_path)
    store.upsert_prices(e, "AAA", _df())
    store.upsert_prices(e, "AAA", _df())
    assert len(store.load_prices(e)["AAA"]) == 5
    store.upsert_prices(e, "AAA", _df(base=200.0))
    assert store.load_prices(e)["AAA"]["close"].iloc[0] == 200.5
    assert store.last_price_date(e, "AAA") == "2026-01-07"


def test_universe_roles_and_active(tmp_path):
    e = _eng(tmp_path)
    store.upsert_universe(e, [dict(symbol="AAA", instrument_key="NSE_EQ|X1"),
                              dict(symbol="NIFTY50", instrument_key="NSE_INDEX|Nifty 50", role="index")])
    assert list(store.get_universe(e, role="stock")["symbol"]) == ["AAA"]
    store.set_active(e, [])
    assert store.get_universe(e, role="stock").empty
    assert len(store.get_universe(e, role="index")) == 1          # index untouched


def test_postgres_url_is_normalised():
    e = store.get_engine("postgres://u:p@h:5432/db")
    assert e.url.drivername == "postgresql+psycopg2"


def test_db_trial_registry_scopes_are_separate(tmp_path):
    e = _eng(tmp_path)
    rng = np.random.default_rng(0)
    a, b = DbTrialRegistry(e, "idea_a"), DbTrialRegistry(e, "idea_b")
    a.log("x", {"q": 1}, rng.normal(0, 0.01, 100))
    a.log("x", {"q": 2}, rng.normal(0, 0.01, 100))
    a.log("x", {"q": 2}, rng.normal(0, 0.01, 100))               # duplicate variant
    b.log("y", {"q": 1}, rng.normal(0, 0.01, 100))
    assert a.n_trials() == 2 and b.n_trials() == 1
