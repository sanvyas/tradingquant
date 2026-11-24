"""Nightly research run (after ingest). Writes runs, trials and research-only signals to the database."""
from .. import store
from ..config import load_costs, load_settings
from ..research import run_research


def main() -> None:
    eng = store.get_engine()
    store.init_db(eng)
    print(run_research(eng, load_settings(), load_costs()))


if __name__ == "__main__":
    main()
