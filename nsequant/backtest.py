"""Daily-bar backtest with stops, targets, time exits, capital/margin limits, costs and tax.

Design rules (each one removes a common source of fake profit):
  * Signal at the close of day t. Entry is the NEXT open ('next_open') or that close ('close', BTST style).
  * Same bar touches both stop and target -> the STOP is assumed to hit first.
  * Gap through a stop fills at the (worse) open, not at the stop price.
  * Slippage on every fill; stops get extra slippage.
  * Delivery is long-only. Shorting needs intraday/futures, which this daily engine does not simulate.
  * Position size comes from risk-at-stop, then is capped by position limit and FREE MARGIN.
  * A trade is rejected unless its NET reward:risk (after charges) and net profit at target clear thresholds.
  * Capital freed by an exit is reusable only from the NEXT day (conservative settlement).
  * Equity curve is mark-to-market, so drawdowns are not hidden by unrealised losses.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Iterator

import numpy as np
import pandas as pd

from .costs import leg_charges, margin_required
from .tax import financial_year, tax_for_trades

REQUIRED_SIGNAL_COLS = ["date", "symbol", "side", "stop_pct", "target_pct", "max_hold", "score"]


def simulate_exit(o: np.ndarray, h: np.ndarray, l: np.ndarray, c: np.ndarray, first_bar: int,
                  entry_bar_is_open_fill: bool, entry: float, side: int, stop_pct: float,
                  target_pct: float, max_hold: int) -> tuple[int, float, str]:
    """Walk bars forward from `first_bar` and return (exit_index, raw_exit_price, reason).

    Shared by the backtester and by triple-barrier labelling so they can never disagree.
    `entry_bar_is_open_fill` True means entry happened at first_bar's open (no gap check there).
    """
    n = len(c)
    if side == 1:
        stop, target = entry * (1 - stop_pct), entry * (1 + target_pct)
    else:
        stop, target = entry * (1 + stop_pct), entry * (1 - target_pct)
    last = min(first_bar + max_hold - 1, n - 1)
    for j in range(first_bar, last + 1):
        gap_check = not (j == first_bar and entry_bar_is_open_fill)
        if side == 1:
            if gap_check and o[j] <= stop:
                return j, o[j], "stop_gap"
            if gap_check and o[j] >= target:
                return j, o[j], "target_gap"
            stop_hit, tgt_hit = l[j] <= stop, h[j] >= target
        else:
            if gap_check and o[j] >= stop:
                return j, o[j], "stop_gap"
            if gap_check and o[j] <= target:
                return j, o[j], "target_gap"
            stop_hit, tgt_hit = h[j] >= stop, l[j] <= target
        if stop_hit:                      # includes the both-hit case: stop wins
            return j, stop, "stop"
        if tgt_hit:
            return j, target, "target"
    reason = "time" if last == first_bar + max_hold - 1 else "data_end"
    return last, c[last], reason


def triple_barrier_label(bars: pd.DataFrame, signal_date, side: int, stop_pct: float, target_pct: float,
                         max_hold: int, entry_mode: str = "next_open") -> dict | None:
    """Label for ML / meta-labelling: +1 target, -1 stop, 0 time exit (by sign of return)."""
    idx = bars.index.get_loc(signal_date)
    first, open_fill, entry = _entry(bars, idx, entry_mode)
    if first is None:
        return None
    o, h, l, c = (bars[k].to_numpy() for k in ("open", "high", "low", "close"))
    j, px, why = simulate_exit(o, h, l, c, first, open_fill, entry, side, stop_pct, target_pct, max_hold)
    ret = side * (px / entry - 1)
    lab = 1 if why.startswith("target") else (-1 if why.startswith("stop") else int(np.sign(ret)))
    return {"label": lab, "ret": ret, "exit_idx": j, "reason": why, "bars_held": j - first + 1}


def _entry(bars: pd.DataFrame, idx: int, mode: str):
    n = len(bars)
    if mode == "next_open":
        if idx + 1 >= n:
            return None, None, None
        return idx + 1, True, float(bars["open"].iloc[idx + 1])
    if mode == "close":
        if idx + 1 >= n:
            return None, None, None
        return idx + 1, False, float(bars["close"].iloc[idx])   # barriers start the following bar
    raise ValueError("entry_mode must be next_open or close")


@dataclass
class BacktestResult:
    trades: pd.DataFrame
    equity: pd.Series                       # mark-to-market, pre-tax, net of costs
    tax: pd.DataFrame
    rejections: dict = field(default_factory=dict)
    summary: dict = field(default_factory=dict)


def run_backtest(bars: dict[str, pd.DataFrame], signals: pd.DataFrame, costs_cfg: dict, settings: dict,
                 segment: str = "equity_delivery") -> BacktestResult:
    if segment != "equity_delivery":
        raise NotImplementedError("Daily engine supports equity_delivery only; intraday/F&O come in later stages")
    miss = [c for c in REQUIRED_SIGNAL_COLS if c not in signals.columns]
    if miss:
        raise ValueError(f"signals missing columns: {miss}")
    risk, entry_mode = settings["risk"], settings["backtest"]["entry_mode"]
    slip = costs_cfg["slippage"]
    cap0 = float(settings["capital"])
    rej: dict[str, int] = {}

    def reject(why):
        rej[why] = rej.get(why, 0) + 1

    # 1. Resolve each signal to a path-dependent exit (independent of capital).
    cands = []
    for r in signals.itertuples(index=False):
        b = bars.get(r.symbol)
        if b is None or pd.Timestamp(r.date) not in b.index:
            reject("no_data")
            continue
        if r.side == -1:
            reject("short_not_allowed_in_delivery")
            continue
        idx = b.index.get_loc(pd.Timestamp(r.date))
        first, open_fill, raw_entry = _entry(b, idx, entry_mode)
        if first is None:
            reject("no_next_bar")
            continue
        entry = raw_entry * (1 + slip["entry_bps"] / 1e4)
        o, h, l, c = (b[k].to_numpy() for k in ("open", "high", "low", "close"))
        j, raw_exit, why = simulate_exit(o, h, l, c, first if open_fill else first, open_fill,
                                         raw_entry, 1, r.stop_pct, r.target_pct, int(r.max_hold))
        extra = slip["stop_exit_extra_bps"] if why.startswith("stop") else 0.0
        exit_px = raw_exit * (1 - (slip["exit_bps"] + extra) / 1e4)
        entry_date = b.index[first] if open_fill else b.index[idx]
        cands.append(dict(symbol=r.symbol, score=float(r.score), entry_date=entry_date,
                          exit_date=b.index[j], entry=entry, exit=exit_px, reason=why,
                          stop_pct=r.stop_pct, target_pct=r.target_pct, entry_idx=first, exit_idx=j))
    cands.sort(key=lambda d: (d["entry_date"], -d["score"]))

    # 2. Allocate capital chronologically.
    taken, open_pos = [], []
    realized = 0.0
    by_date: dict = {}
    for d in cands:
        by_date.setdefault(d["entry_date"], []).append(d)
    for day in sorted(by_date):
        still = []
        for p in open_pos:
            if p["exit_date"] < day:
                realized += p["net_pnl"]
            else:
                still.append(p)
        open_pos = still
        equity = cap0 + realized
        blocked = sum(p["margin"] for p in open_pos)
        for d in by_date[day]:
            if len(open_pos) >= risk["max_open_positions"]:
                reject("max_positions")
                continue
            if any(p["symbol"] == d["symbol"] for p in open_pos):
                reject("already_in_symbol")
                continue
            stop_dist = d["entry"] * d["stop_pct"]
            qty = int(min(equity * risk["risk_per_trade_pct"] / stop_dist,
                          equity * risk["max_position_pct"] / d["entry"]))
            if qty < 1:
                reject("size_below_one_share")
                continue
            margin = margin_required(costs_cfg, segment, qty, d["entry"])
            free = equity - blocked
            if margin > free:
                qty = int(free / (margin / qty)) if margin > 0 else 0
                if qty < 1:
                    reject("insufficient_margin")
                    continue
                margin = margin_required(costs_cfg, segment, qty, d["entry"])
            # Net reward:risk gate, after charges, at the planned stop and target prices.
            tgt_px = d["entry"] * (1 + d["target_pct"]) * (1 - slip["exit_bps"] / 1e4)
            stp_px = d["entry"] * (1 - d["stop_pct"]) * (1 - (slip["exit_bps"] + slip["stop_exit_extra_bps"]) / 1e4)
            buy = leg_charges(costs_cfg, segment, "BUY", qty, d["entry"]).total
            gain = qty * (tgt_px - d["entry"]) - buy - leg_charges(costs_cfg, segment, "SELL", qty, tgt_px).total
            loss = qty * (d["entry"] - stp_px) + buy + leg_charges(costs_cfg, segment, "SELL", qty, stp_px).total
            if gain < risk["min_expected_net_profit"]:
                reject("net_profit_at_target_too_small")
                continue
            if gain / loss < risk["min_net_reward_risk"]:
                reject("net_reward_risk_too_low")
                continue
            ch = leg_charges(costs_cfg, segment, "BUY", qty, d["entry"]) + \
                leg_charges(costs_cfg, segment, "SELL", qty, d["exit"])
            gross = qty * (d["exit"] - d["entry"])
            rec = dict(d, qty=qty, margin=margin, gross_pnl=gross, costs=ch.total, stt=ch.stt,
                       net_pnl=gross - ch.total, segment=segment)
            open_pos.append(rec)
            blocked += margin
            taken.append(rec)

    trades = pd.DataFrame(taken)
    if trades.empty:
        return BacktestResult(trades, pd.Series(dtype=float), pd.DataFrame(), rej, {"n_trades": 0})
    trades["holding_days"] = (trades["exit_date"] - trades["entry_date"]).dt.days
    trades["fy"] = trades["exit_date"].map(financial_year)

    # 3. Mark-to-market equity.
    all_dates = sorted(set().union(*[set(b.index) for b in bars.values()]))
    all_dates = pd.DatetimeIndex([d for d in all_dates if trades["entry_date"].min() <= d <= trades["exit_date"].max()])
    realized_s = pd.Series(0.0, index=all_dates)
    unreal = pd.Series(0.0, index=all_dates)
    for t in trades.itertuples():
        b = bars[t.symbol]
        realized_s.loc[t.exit_date] += t.net_pnl
        path = b["close"].loc[t.entry_date:t.exit_date]
        path = path[path.index < t.exit_date]
        unreal.loc[path.index] += (path - t.entry) * t.qty
    equity = cap0 + realized_s.cumsum() + unreal
    equity.name = "equity"

    tax = tax_for_trades(trades.rename(columns={"gross_pnl": "pnl_gross"}), costs_cfg["tax"])
    total_tax = float(tax["tax"].sum()) if len(tax) else 0.0
    net_pre = float(trades["net_pnl"].sum())
    summary = dict(
        n_trades=len(trades), gross_pnl=float(trades["gross_pnl"].sum()), costs=float(trades["costs"].sum()),
        net_pnl_pre_tax=net_pre, tax=total_tax, net_pnl_after_tax=net_pre - total_tax,
        win_rate=float((trades["net_pnl"] > 0).mean()),
        avg_win=float(trades.loc[trades["net_pnl"] > 0, "net_pnl"].mean()) if (trades["net_pnl"] > 0).any() else 0.0,
        avg_loss=float(trades.loc[trades["net_pnl"] <= 0, "net_pnl"].mean()) if (trades["net_pnl"] <= 0).any() else 0.0,
        costs_pct_of_gross_abs=float(trades["costs"].sum() / max(abs(trades["gross_pnl"]).sum(), 1e-9)),
        max_drawdown_pct=float((equity / equity.cummax() - 1).min()),
    )
    return BacktestResult(trades, equity, tax, rej, summary)


def walk_forward_splits(dates: pd.DatetimeIndex, train_days: int, test_days: int,
                        embargo_days: int) -> Iterator[tuple[pd.DatetimeIndex, pd.DatetimeIndex]]:
    """Rolling train/test windows in trading days with an embargo gap between them.

    The embargo should be at least the longest holding period, so labels built from the
    train window cannot peek into the test window.
    """
    n = len(dates)
    start = 0
    while start + train_days + embargo_days + test_days <= n:
        tr = dates[start:start + train_days]
        te = dates[start + train_days + embargo_days:start + train_days + embargo_days + test_days]
        yield tr, te
        start += test_days


def run_walk_forward(bars: dict[str, pd.DataFrame], fit_fn: Callable, signal_fn: Callable,
                     costs_cfg: dict, settings: dict, calendar: pd.DatetimeIndex) -> BacktestResult:
    """fit_fn(train_dates) -> model; signal_fn(model, test_dates) -> signals DataFrame.

    Only out-of-sample test windows produce trades, so reported results never include
    data the model was fitted on.
    """
    wf = settings["backtest"]["walk_forward"]
    parts = []
    for tr, te in walk_forward_splits(calendar, wf["train_days"], wf["test_days"], wf["embargo_days"]):
        model = fit_fn(tr)
        s = signal_fn(model, te)
        if len(s):
            parts.append(s)
    sigs = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=REQUIRED_SIGNAL_COLS)
    return run_backtest(bars, sigs, costs_cfg, settings)
