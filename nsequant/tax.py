"""Income-tax estimate on a set of closed trades, per financial year and income head.

This is an ESTIMATE for ranking strategies after tax, not tax advice. Simplifications are
conservative and listed in config/costs.yaml. Confirm treatment with a chartered accountant.
"""
from __future__ import annotations

import pandas as pd

HEADS = {
    "equity_delivery": "capital_gains",
    "equity_intraday": "speculative",
    "equity_futures": "non_speculative",
    "equity_options": "non_speculative",
}


def financial_year(ts: pd.Timestamp) -> str:
    """Indian FY runs 1 Apr - 31 Mar. Label by starting year: '2026-27'."""
    y = ts.year if ts.month >= 4 else ts.year - 1
    return f"{y}-{str(y + 1)[-2:]}"


def tax_for_trades(trades: pd.DataFrame, tax_cfg: dict) -> pd.DataFrame:
    """trades columns: fy, segment, holding_days, pnl_gross, costs, stt.

    Returns one row per (fy, head) with taxable income and tax. Empty input -> empty frame.
    """
    cols = ["fy", "head", "taxable", "tax"]
    if trades is None or len(trades) == 0:
        return pd.DataFrame(columns=cols)
    t = trades.copy()
    t["head"] = t["segment"].map(HEADS)
    if t["head"].isna().any():
        raise ValueError("trade with unknown segment")
    # STT is deductible against business income, not against capital gains.
    is_cg = t["head"] == "capital_gains"
    t["taxable"] = t["pnl_gross"] - t["costs"] + is_cg * t["stt"]

    cess = 1.0 + tax_cfg["cess"]
    rows = []
    for fy, g in t.groupby("fy"):
        cg = g[g["head"] == "capital_gains"]
        if len(cg):
            lt = cg["holding_days"] > tax_cfg["long_term_days"]
            s = float(cg.loc[~lt, "taxable"].sum())
            l = float(cg.loc[lt, "taxable"].sum())
            if s < 0 < l:                       # short-term loss may offset long-term gain
                off = min(-s, l)
                s, l = s + off, l - off
            tax = max(s, 0.0) * tax_cfg["stcg_rate"] + \
                max(l - tax_cfg["ltcg_exemption"], 0.0) * tax_cfg["ltcg_rate"]
            rows.append((fy, "capital_gains", s + l, tax * cess))
        for head in ("speculative", "non_speculative"):
            h = g[g["head"] == head]
            if len(h):
                inc = float(h["taxable"].sum())
                tax = max(inc, 0.0) * tax_cfg["business_income_marginal_rate"]
                rows.append((fy, head, inc, tax * cess))
    return pd.DataFrame(rows, columns=cols)
