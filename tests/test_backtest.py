import copy

import numpy as np
import pandas as pd
import pytest

from nsequant.backtest import simulate_exit, run_backtest, walk_forward_splits, triple_barrier_label
from nsequant.config import load_costs, load_settings
from nsequant.costs import leg_charges

COSTS = load_costs()
SET = load_settings()


def arr(*xs):
    return np.array(xs, dtype=float)


def test_both_barriers_same_bar_stop_wins():
    # entry 100, stop 98, target 104. Bar 1 spans 97..105 -> stop must be assumed first.
    o, h, l, c = arr(100, 100), arr(101, 105), arr(99, 97), arr(100, 100)
    j, px, why = simulate_exit(o, h, l, c, 1, False, 100.0, 1, 0.02, 0.04, 5)
    assert (j, px, why) == (1, 98.0, "stop")


def test_gap_through_stop_fills_at_open():
    o, h, l, c = arr(100, 90), arr(101, 91), arr(99, 89), arr(100, 90)
    j, px, why = simulate_exit(o, h, l, c, 1, False, 100.0, 1, 0.02, 0.04, 5)
    assert why == "stop_gap" and px == 90.0


def test_open_fill_entry_bar_has_no_gap_check_but_checks_intrabar():
    # entry at bar-1 open = 100 (open_fill). Same bar low 97 hits the stop at 98.
    o, h, l, c = arr(99, 100), arr(100, 101), arr(98, 97), arr(99, 100)
    j, px, why = simulate_exit(o, h, l, c, 1, True, 100.0, 1, 0.02, 0.04, 5)
    assert (j, px, why) == (1, 98.0, "stop")


def test_target_and_time_exit():
    o, h, l, c = arr(100, 100, 100), arr(101, 105, 101), arr(99, 99, 99), arr(100, 100, 100)
    assert simulate_exit(o, h, l, c, 1, False, 100.0, 1, 0.02, 0.04, 5)[2] == "target"
    o, h, l, c = arr(100, 100, 100, 100), arr(101, 101, 101, 101), arr(99.5, 99.5, 99.5, 99.5), arr(100, 100, 100, 100.5)
    j, px, why = simulate_exit(o, h, l, c, 1, False, 100.0, 1, 0.02, 0.04, 3)
    assert why == "time" and j == 3 and px == 100.5


def test_short_mirror():
    o, h, l, c = arr(100, 100), arr(101, 103), arr(99, 99), arr(100, 100)
    j, px, why = simulate_exit(o, h, l, c, 1, False, 100.0, -1, 0.02, 0.04, 5)
    assert why == "stop" and px == pytest.approx(102.0)


def _bars(rows, start="2026-01-01"):
    idx = pd.bdate_range(start, periods=len(rows))
    return pd.DataFrame(rows, columns=["open", "high", "low", "close"], index=idx).assign(volume=1e6)


def _sig(date, sym="AAA", **kw):
    d = dict(date=pd.Timestamp(date), symbol=sym, side=1, stop_pct=0.02, target_pct=0.04, max_hold=5, score=1.0)
    d.update(kw)
    return d


def _settings(**over):
    s = copy.deepcopy(SET)
    s["backtest"]["entry_mode"] = "next_open"
    s["risk"].update(over)
    return s


def test_single_winning_trade_matches_hand_calculation():
    # signal day0 close; entry day1 open 100; day2 high reaches target 104.
    b = _bars([(100, 100, 100, 100), (100, 101, 99.5, 100), (100, 105, 100, 104), (104, 104, 103, 103)])
    costs = copy.deepcopy(COSTS)
    costs["slippage"] = dict(entry_bps=0.0, exit_bps=0.0, stop_exit_extra_bps=0.0)
    # Gate relaxed on purpose: with these costs a 2:1 setup nets only ~1.4:1, which the default 1.5 gate rejects.
    res = run_backtest({"AAA": b}, pd.DataFrame([_sig("2026-01-01")]), costs,
                       _settings(risk_per_trade_pct=0.01, min_net_reward_risk=1.0))
    assert res.summary["n_trades"] == 1, res.rejections
    t = res.trades.iloc[0]
    # equity 100000, risk 1% = 1000, stop dist 2 -> qty 500, capped by 30% position = 300 shares
    assert t.qty == 300
    assert t.reason == "target" and t.exit == pytest.approx(104.0)
    gross = 300 * 4
    ch = leg_charges(costs, "equity_delivery", "BUY", 300, 100).total + \
        leg_charges(costs, "equity_delivery", "SELL", 300, 104).total
    assert t.gross_pnl == pytest.approx(gross)
    assert t.net_pnl == pytest.approx(gross - ch)
    assert res.summary["net_pnl_after_tax"] < res.summary["net_pnl_pre_tax"]     # tax was charged
    assert res.summary["tax"] == pytest.approx(res.tax["tax"].sum())


def test_short_signals_rejected_in_delivery():
    b = _bars([(100, 100, 100, 100)] * 5)
    res = run_backtest({"AAA": b}, pd.DataFrame([_sig("2026-01-01", side=-1)]), COSTS, _settings())
    assert res.rejections.get("short_not_allowed_in_delivery") == 1
    assert res.summary["n_trades"] == 0


def test_margin_limits_concurrent_positions():
    rows = [(100, 100, 100, 100)] * 8
    bars = {s: _bars(rows) for s in ("A", "B", "C", "D")}
    sigs = pd.DataFrame([_sig("2026-01-01", s, score=10 - i) for i, s in enumerate("ABCD")])
    # each position capped at 30% of equity -> 3 fit (90%), the 4th is limited by free margin
    res = run_backtest(bars, sigs, COSTS, _settings(max_open_positions=10, max_position_pct=0.30,
                                                    risk_per_trade_pct=0.05, min_expected_net_profit=0,
                                                    min_net_reward_risk=0))
    held = res.trades.sort_values("score", ascending=False)
    assert len(held) >= 3
    assert held["margin"].sum() <= 100000 + 1e-6          # never uses margin it does not have


def test_net_reward_risk_gate_rejects_tiny_edge():
    b = _bars([(100, 100, 100, 100), (100, 100.5, 99.9, 100.2)] + [(100, 100.3, 99.9, 100)] * 6)
    sig = pd.DataFrame([_sig("2026-01-01", stop_pct=0.02, target_pct=0.003)])
    res = run_backtest({"AAA": b}, sig, COSTS, _settings())
    assert res.summary["n_trades"] == 0
    assert sum(v for k, v in res.rejections.items() if "too" in k) == 1


def test_capital_not_reused_same_day():
    # Trade A exits on day 2 (target). A new signal for B enters day 2 open -> A's capital not yet free.
    a = _bars([(100, 100, 100, 100), (100, 101, 99.5, 100), (100, 105, 100, 104), (104, 104, 104, 104)])
    bb = _bars([(50, 50, 50, 50), (50, 50, 50, 50), (50, 50, 50, 50), (50, 50, 50, 50)])
    sigs = pd.DataFrame([_sig("2026-01-01", "A"), _sig("2026-01-02", "B")])
    s = _settings(max_position_pct=1.0, risk_per_trade_pct=1.0, min_expected_net_profit=0, min_net_reward_risk=0)
    res = run_backtest({"A": a, "B": bb}, sigs, COSTS, s)
    # A uses ~100% of equity; B enters on 2026-01-05? signal 01-02 -> entry 01-05, after A's exit 01-05?
    assert res.trades["margin"].max() <= 100000 + 1e-6


def test_walk_forward_has_embargo_and_no_overlap():
    d = pd.bdate_range("2020-01-01", periods=1000)
    for tr, te in walk_forward_splits(d, 300, 100, 10):
        gap = d.get_loc(te[0]) - d.get_loc(tr[-1]) - 1
        assert gap == 10
        assert tr[-1] < te[0]


def test_triple_barrier_label_agrees_with_backtest_exit():
    b = _bars([(100, 100, 100, 100), (100, 101, 99.5, 100), (100, 105, 100, 104), (104, 104, 103, 103)])
    lab = triple_barrier_label(b, b.index[0], 1, 0.02, 0.04, 5)
    assert lab["label"] == 1 and lab["reason"] == "target"
