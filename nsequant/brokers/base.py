"""Broker adapter interfaces. Research code depends only on these, never on a vendor SDK.

Two roles are kept separate on purpose: data and execution can come from different brokers.
`PaperBroker` is the default execution adapter: it records orders, never sends them.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Protocol

import pandas as pd


@dataclass
class OrderRequest:
    symbol: str
    side: str                 # "BUY" | "SELL"
    qty: int
    product: str              # equity_delivery | equity_intraday | equity_futures | equity_options
    order_type: str = "LIMIT"
    price: float | None = None
    stop_price: float | None = None
    tag: str = ""             # idempotency key: same tag must never create two orders


@dataclass
class OrderResult:
    order_id: str
    status: str               # "PAPER_ACCEPTED" | "REJECTED" | ...
    message: str = ""


class MarketDataProvider(Protocol):
    def historical_candles(self, symbol: str, interval: str, start: datetime, end: datetime) -> pd.DataFrame:
        """Columns: open, high, low, close, volume; index = tz-aware timestamps (Asia/Kolkata)."""

    def option_chain(self, underlying: str, expiry: str) -> pd.DataFrame:
        """Strike-wise call/put snapshot. Snapshots should be stored by the caller for later backtests."""


class ExecutionProvider(Protocol):
    def place_order(self, req: OrderRequest) -> OrderResult: ...
    def margin_required(self, req: OrderRequest) -> float: ...


@dataclass
class PaperBroker:
    """Records orders only. Enforces tag uniqueness so retries cannot double-order."""
    log: list = field(default_factory=list)
    _tags: set = field(default_factory=set)
    live_enabled: bool = False

    def place_order(self, req: OrderRequest) -> OrderResult:
        if req.tag and req.tag in self._tags:
            return OrderResult("", "REJECTED", f"duplicate tag {req.tag}")
        if req.tag:
            self._tags.add(req.tag)
        oid = f"PAPER-{len(self.log) + 1}"
        self.log.append((datetime.now(), oid, req))
        return OrderResult(oid, "PAPER_ACCEPTED")

    def margin_required(self, req: OrderRequest) -> float:
        raise NotImplementedError("Use the config margin model or the broker margin API")
