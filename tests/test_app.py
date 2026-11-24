import json

import numpy as np
import pandas as pd
from fastapi.testclient import TestClient

from nsequant import app as appmod
from nsequant import store

AUTH = ("u", "secret")


def _client(tmp_path, monkeypatch, pw="secret", token=None):
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{tmp_path}/a.db")
    for k, v in (("DASH_PASSWORD", pw), ("GITHUB_TOKEN", token)):
        monkeypatch.setenv(k, v) if v else monkeypatch.delenv(k, raising=False)
    appmod._eng = None
    return TestClient(appmod.app)


def test_locked_without_password_and_with_wrong_password(tmp_path, monkeypatch):
    c = _client(tmp_path, monkeypatch, pw=None)
    assert c.get("/").status_code == 503 and c.get("/healthz").status_code == 200
    c = _client(tmp_path, monkeypatch)
    assert c.get("/").status_code == 401 and c.get("/api/state", auth=("u", "bad")).status_code == 401
    assert c.post("/api/action", json={"action": "doctor"}).status_code == 401


def test_empty_state_renders_without_errors(tmp_path, monkeypatch):
    c = _client(tmp_path, monkeypatch)
    s = c.get("/api/state", auth=AUTH).json()
    assert s["verdict"]["level"] == "none" and s["best"] is None and s["github"]["connected"] is False
    assert "Research only" in c.get("/", auth=AUTH).text


def test_state_shows_best_run_reasons_equity_and_sized_signals(tmp_path, monkeypatch):
    c = _client(tmp_path, monkeypatch)
    e = appmod.engine()
    good = dict(n_trades=150, net_pnl_after_tax=5000.0, dsr=0.97, prob_net_profit=0.9, max_drawdown_pct=-0.1,
                n_trials=3, equity=[["2026-01-01", 100000.0], ["2026-02-01", 105000.0]])
    bad = dict(n_trades=20, net_pnl_after_tax=-100.0, dsr=float("nan"), prob_net_profit=0.4, max_drawdown_pct=-0.4)
    store.save_run(e, "a", "s", "quantile=0.9", good, True)
    store.save_run(e, "b", "s", "quantile=0.85", bad, False)
    idx = pd.bdate_range("2026-11-10", periods=5)
    store.upsert_prices(e, "AAA", pd.DataFrame(dict(open=500.0, high=505.0, low=495.0, close=500.0, volume=1e6), index=idx))
    store.save_signals(e, [dict(date="2026-11-13", symbol="AAA", side=1, stop_pct=0.04, target_pct=0.08, score=0.7,
                                status="research_only", scope="s")])
    s = c.get("/api/state", auth=AUTH).json()                      # NaN in 'bad' must not break JSON
    assert s["verdict"]["level"] == "pass" and s["best"]["variant"] == "quantile=0.9"
    assert all(r["ok"] for r in s["best"]["reasons"]) and len(s["best"]["m"]["equity"]) == 2
    sig = s["signals"][0]
    assert sig["stop"] == 480.0 and sig["target"] == 540.0
    assert sig["qty"] == 50                       # 1% of 1L = 1000 risk / (500*0.04=20) = 50 shares (25k < 30% cap)


def test_action_requires_header_valid_name_and_token(tmp_path, monkeypatch):
    c = _client(tmp_path, monkeypatch)
    assert c.post("/api/action", auth=AUTH, json={"action": "doctor"}).status_code == 403     # no CSRF header
    h = {"X-TQ": "1"}
    assert c.post("/api/action", auth=AUTH, headers=h, json={"action": "rm -rf"}).status_code == 400
    assert c.post("/api/action", auth=AUTH, headers=h, json={"action": "doctor"}).status_code == 503  # no token
    calls = []
    monkeypatch.setattr(appmod, "gh_request", lambda m, p, **kw: (calls.append((m, p, kw)) or (204, {})))
    r = c.post("/api/action", auth=AUTH, headers=h, json={"action": "ingest"})
    assert r.status_code == 200 and calls[0][2]["json"]["inputs"] == {"job": "ingest"}
    monkeypatch.setattr(appmod, "gh_request", lambda m, p, **kw: (404, {"message": "Not Found"}))
    assert c.post("/api/action", auth=AUTH, headers=h, json={"action": "ingest"}).status_code == 502


def test_github_runs_are_summarised(tmp_path, monkeypatch):
    c = _client(tmp_path, monkeypatch)
    run = dict(id=1, status="completed", conclusion="success", created_at="2026-11-25T10:00:00Z", html_url="http://x",
               event="workflow_dispatch", display_title="daily")
    monkeypatch.setattr(appmod, "gh_request", lambda m, p, **kw: (200, {"workflow_runs": [run]}))
    g = c.get("/api/state", auth=AUTH).json()["github"]
    assert g["connected"] and g["runs"][0]["conclusion"] == "success"
