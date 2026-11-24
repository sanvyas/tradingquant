"""Pre-flight check: is the database reachable and does the Upstox token work? Never prints secrets."""
from __future__ import annotations

import os
import sys
from datetime import date, timedelta

from .. import store
from ..upstox import AuthError, UpstoxClient


def check(env: dict | None = None, client=None, engine=None) -> list[tuple[str, bool, str]]:
    env = os.environ if env is None else env
    out = []
    try:
        eng = engine or store.get_engine(env.get("DATABASE_URL"))
        store.init_db(eng)
        out.append(("database", True, "reachable, tables ready"))
    except Exception as e:
        out.append(("database", False, f"{type(e).__name__}: {str(e)[:120]}".replace(env.get("DATABASE_URL") or "\0", "***")))
    tok = env.get("UPSTOX_ANALYTICS_TOKEN")
    if not tok and client is None:
        out.append(("upstox_token", False, "UPSTOX_ANALYTICS_TOKEN is not set"))
    else:
        try:
            c = client or UpstoxClient(tok)
            df = c.daily_candles("NSE_INDEX|Nifty 50", date.today() - timedelta(days=10), date.today())
            out.append(("upstox_token", len(df) > 0, f"got {len(df)} Nifty bars" if len(df) else "no data returned"))
        except AuthError as e:
            out.append(("upstox_token", False, str(e)))
        except Exception as e:
            out.append(("upstox_token", False, f"{type(e).__name__}: {str(e)[:120]}"))
    return out


def main() -> None:
    res = check()
    for name, ok, msg in res:
        print(f"{'OK  ' if ok else 'FAIL'} {name}: {msg}")
    sys.exit(0 if all(ok for _, ok, _ in res) else 1)


if __name__ == "__main__":
    main()
