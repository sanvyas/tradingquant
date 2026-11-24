import pandas as pd
import pytest

from nsequant.config import load_costs
from nsequant.costs import leg_charges, round_trip_charges, margin_required, breakeven_move_pct
from nsequant.tax import tax_for_trades, financial_year

CFG = load_costs()


def test_delivery_buy_leg_hand_computed():
    c = leg_charges(CFG, "equity_delivery", "BUY", 100, 1000.0)
    gst = 0.18 * (20 + 100000 * 0.0000297 + 0.1)
    assert c.brokerage == 20
    assert c.stt == pytest.approx(100.0)
    assert c.stamp == pytest.approx(15.0)
    assert c.dp == 0
    assert c.total == pytest.approx(20 + 100 + 2.97 + 0.1 + 15 + gst)


def test_delivery_sell_leg_has_dp_and_no_stamp():
    c = leg_charges(CFG, "equity_delivery", "SELL", 100, 1010.0)
    assert c.stamp == 0
    assert c.dp == 20
    assert c.stt == pytest.approx(101.0)
    gst = 0.18 * (20 + 101000 * 0.0000297 + 0.101 + 20)
    assert c.gst == pytest.approx(gst)


def test_intraday_brokerage_is_lower_of_flat_or_pct():
    small = leg_charges(CFG, "equity_intraday", "BUY", 1, 1000.0)       # 0.1% of 1000 = 1
    assert small.brokerage == pytest.approx(1.0)
    big = leg_charges(CFG, "equity_intraday", "BUY", 1000, 1000.0)      # 0.1% = 1000 -> flat 20
    assert big.brokerage == 20


def test_options_stt_is_on_premium_sell_only_at_new_rate():
    buy = leg_charges(CFG, "equity_options", "BUY", 75, 100.0)
    sell = leg_charges(CFG, "equity_options", "SELL", 75, 100.0)
    assert buy.stt == 0
    assert sell.stt == pytest.approx(75 * 100 * 0.0015)


def test_futures_stt_new_rate():
    sell = leg_charges(CFG, "equity_futures", "SELL", 1, 2_000_000.0)
    assert sell.stt == pytest.approx(2_000_000 * 0.0005)


def test_short_delivery_rejected():
    with pytest.raises(ValueError):
        round_trip_charges(CFG, "equity_delivery", 10, 100, 99, long=False)


def test_margin_delivery_is_full_value():
    assert margin_required(CFG, "equity_delivery", 10, 500) == 5000


def test_breakeven_is_positive_and_bigger_for_small_orders():
    small = breakeven_move_pct(CFG, "equity_delivery", 5, 1000)     # fixed Rs 40 brokerage on 5k
    large = breakeven_move_pct(CFG, "equity_delivery", 50, 1000)
    assert small > large > 0


def test_financial_year_labels():
    assert financial_year(pd.Timestamp("2026-03-31")) == "2025-26"
    assert financial_year(pd.Timestamp("2026-04-01")) == "2026-27"


def _row(fy, seg, days, pnl, costs=0.0, stt=0.0):
    return dict(fy=fy, segment=seg, holding_days=days, pnl_gross=pnl, costs=costs, stt=stt)


def test_stcg_tax_with_cess():
    t = pd.DataFrame([_row("2026-27", "equity_delivery", 5, 10000.0)])
    out = tax_for_trades(t, CFG["tax"])
    assert out["tax"].iloc[0] == pytest.approx(10000 * 0.20 * 1.04)


def test_loss_year_pays_no_tax_and_no_carry_credit():
    t = pd.DataFrame([_row("2026-27", "equity_futures", 1, -5000.0),
                      _row("2027-28", "equity_futures", 1, 5000.0)])
    out = tax_for_trades(t, CFG["tax"]).set_index("fy")["tax"]
    assert out["2026-27"] == 0
    assert out["2027-28"] == pytest.approx(5000 * 0.30 * 1.04)   # conservative: no carry-forward


def test_stt_deductible_for_business_not_for_capital_gains():
    biz = tax_for_trades(pd.DataFrame([_row("2026-27", "equity_futures", 1, 1000.0, costs=100.0, stt=40.0)]), CFG["tax"])
    cg = tax_for_trades(pd.DataFrame([_row("2026-27", "equity_delivery", 1, 1000.0, costs=100.0, stt=40.0)]), CFG["tax"])
    assert biz["taxable"].iloc[0] == pytest.approx(900.0)      # all costs deducted
    assert cg["taxable"].iloc[0] == pytest.approx(940.0)       # STT not deducted


def test_short_term_loss_offsets_long_term_gain():
    t = pd.DataFrame([_row("2026-27", "equity_delivery", 10, -50000.0),
                      _row("2026-27", "equity_delivery", 400, 300000.0)])
    out = tax_for_trades(t, CFG["tax"]).iloc[0]
    # LT gain 300k - 50k offset = 250k; minus 125k exemption = 125k taxed at 12.5% (+cess)
    assert out["tax"] == pytest.approx(125000 * 0.125 * 1.04)
