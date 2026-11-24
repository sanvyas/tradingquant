import json

from fastapi.testclient import TestClient

from nsequant import app as appmod
from nsequant import store


def _client(tmp_path, monkeypatch, pw="secret"):
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{tmp_path}/a.db")
    if pw:
        monkeypatch.setenv("DASH_PASSWORD", pw)
    else:
        monkeypatch.delenv("DASH_PASSWORD", raising=False)
    appmod._eng = None
    return TestClient(appmod.app)


def test_locked_without_password_and_with_wrong_password(tmp_path, monkeypatch):
    c = _client(tmp_path, monkeypatch, pw=None)
    assert c.get("/").status_code == 503
    assert c.get("/healthz").status_code == 200
    c = _client(tmp_path, monkeypatch)
    assert c.get("/").status_code == 401
    assert c.get("/", auth=("u", "bad")).status_code == 401


def test_dashboard_renders_runs_and_escapes_html(tmp_path, monkeypatch):
    c = _client(tmp_path, monkeypatch)
    e = appmod.engine()
    store.save_run(e, "r1", "s", "quantile=0.9", dict(n_trades=120, net_pnl_after_tax=1234.5, dsr=0.97,
                                                    prob_net_profit=0.9, max_drawdown_pct=-0.1, n_trials=3), True)
    store.set_issue(e, "X<script>", "bad_ohlc", "<b>boom</b>")
    r = c.get("/", auth=("u", "secret"))
    assert r.status_code == 200 and "PASS" in r.text and "RESEARCH ONLY" in r.text
    assert "<script>" not in r.text and "&lt;script&gt;" in r.text
    assert c.get("/api/summary", auth=("u", "secret")).json()["open_data_issues"] == 1
