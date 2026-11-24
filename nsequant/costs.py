"""Transaction costs and margin, driven entirely by config/costs.yaml (no rates in code)."""
from __future__ import annotations

from dataclasses import dataclass

SEGMENTS = ("equity_delivery", "equity_intraday", "equity_futures", "equity_options")


@dataclass
class Charges:
    brokerage: float = 0.0
    stt: float = 0.0
    exchange: float = 0.0
    sebi: float = 0.0
    stamp: float = 0.0
    dp: float = 0.0
    gst: float = 0.0

    @property
    def total(self) -> float:
        return self.brokerage + self.stt + self.exchange + self.sebi + self.stamp + self.dp + self.gst

    def __add__(self, o: "Charges") -> "Charges":
        return Charges(*(getattr(self, f) + getattr(o, f) for f in
                         ("brokerage", "stt", "exchange", "sebi", "stamp", "dp", "gst")))


def leg_charges(cfg: dict, segment: str, side: str, qty: int, price: float,
                charge_dp: bool = True) -> Charges:
    """Charges for ONE executed leg. side is 'BUY' or 'SELL'.

    For options, `price` is the premium and `qty` is units (lots x lot size).
    `charge_dp` should be True only for the first delivery sell of a scrip on a given day.
    """
    if segment not in SEGMENTS:
        raise ValueError(f"unknown segment {segment}")
    if side not in ("BUY", "SELL"):
        raise ValueError("side must be BUY or SELL")
    seg = cfg["segments"][segment]
    turnover = qty * price
    b = seg["brokerage"]
    brokerage = b["flat"]
    if b.get("pct_cap"):
        brokerage = min(brokerage, b["pct_cap"] * turnover)
    stt = seg["stt"]["buy" if side == "BUY" else "sell"] * turnover
    exchange = seg["exchange_txn_pct"] * turnover
    sebi = cfg["sebi_turnover_pct"] * turnover
    ipft = cfg.get("ipft_pct", 0.0) * turnover
    stamp = seg["stamp_buy_pct"] * turnover if side == "BUY" else 0.0
    dp = seg["dp_charge_sell_flat"] if (side == "SELL" and charge_dp) else 0.0
    gst = cfg["gst_rate"] * (brokerage + exchange + sebi + ipft + dp)
    return Charges(brokerage, stt, exchange + ipft, sebi, stamp, dp, gst)


def round_trip_charges(cfg: dict, segment: str, qty: int, entry: float, exit_: float,
                       long: bool = True) -> Charges:
    """Entry leg plus exit leg. Short positions sell first (intraday / futures only)."""
    if not long and segment in ("equity_delivery", "equity_options"):
        raise ValueError(f"cannot open a short in {segment} via cash/option buying")
    if long:
        return leg_charges(cfg, segment, "BUY", qty, entry) + leg_charges(cfg, segment, "SELL", qty, exit_)
    return leg_charges(cfg, segment, "SELL", qty, entry, charge_dp=False) + \
        leg_charges(cfg, segment, "BUY", qty, exit_)


def margin_required(cfg: dict, segment: str, qty: int, price: float) -> float:
    return cfg["segments"][segment]["margin_pct"] * qty * price


def breakeven_move_pct(cfg: dict, segment: str, qty: int, price: float) -> float:
    """Price move (fraction of entry) a long needs just to cover round-trip charges at flat price,
    including entry+exit slippage. Used to reject trades that cannot clear friction."""
    c = round_trip_charges(cfg, segment, qty, price, price, long=True).total
    slip = (cfg["slippage"]["entry_bps"] + cfg["slippage"]["exit_bps"]) / 1e4
    return c / (qty * price) + slip
