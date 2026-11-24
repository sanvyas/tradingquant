"""Config loading. Also: `python -m nsequant.config` lists every rate not marked verified."""
from __future__ import annotations

from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


def load_yaml(name: str) -> dict:
    with open(ROOT / "config" / name, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_costs() -> dict:
    return load_yaml("costs.yaml")


def load_settings() -> dict:
    return load_yaml("settings.yaml")


def unverified_notes() -> list[str]:
    """Text lines in costs.yaml that say verify/assumption, so nothing hides silently."""
    out = []
    for i, line in enumerate((ROOT / "config" / "costs.yaml").read_text().splitlines(), 1):
        low = line.lower()
        if "status verify" in low or "status assumption" in low or "status: assumption" in low:
            out.append(f"costs.yaml:{i}: {line.strip()}")
        elif "status: verify" in low:
            out.append(f"costs.yaml:{i}: {line.strip()}")
    return out


if __name__ == "__main__":
    for n in unverified_notes():
        print(n)
