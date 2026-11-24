"""One-time final test of the best research variant on the never-touched last ~252 sessions."""
import json

import pandas as pd
from sqlalchemy import text

from .. import store
from ..config import load_costs, load_settings
from ..research import holdout_check

SCOPE = "meta_hgb_v1"


def main() -> None:
    eng = store.get_engine()
    store.init_db(eng)
    runs = pd.read_sql(text("SELECT variant, metrics FROM runs WHERE scope=:s AND variant NOT LIKE 'HOLDOUT%'"),
                       eng, params=dict(s=SCOPE))
    best = None
    for r in runs.itertuples():
        m = json.loads(r.metrics)
        d = m.get("dsr")
        if m.get("n_trades") and d == d and d is not None and (best is None or d > best[0]):
            best = (d, float(r.variant.split("=")[1]))
    if best is None:
        print("No research variant with trades yet; run research first.")
        store.log_job(eng, "holdout", "skipped", "no research variant with trades")
        return
    out = holdout_check(eng, load_settings(), load_costs(), SCOPE, best[1])
    store.log_job(eng, "holdout", out.get("status", "?"), str({k: v for k, v in out.items() if k != "equity"})[:800])
    print({k: v for k, v in out.items() if k != "equity"})


if __name__ == "__main__":
    main()
