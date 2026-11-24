"""News pipeline: ingest -> dedupe -> link entities -> classify event -> sentiment -> measured impact.

The key idea is that sentiment is NOT impact. "Positive headline" only matters if, historically,
that kind of event moved that kind of stock after the market could have traded on it. So impact is
*measured* with an event study (abnormal return vs beta-adjusted index) and shrunk toward zero when
evidence is thin. Everything is point-in-time: an item only enters a feature row after it could have
been seen (publication time + ingest delay) before that row's decision time.

Store headlines, links, timestamps and derived scores. Do not redistribute full article text; respect
each source's terms. Historical Indian news is hard to get free: start recording now.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import time
from typing import Iterable, Protocol

import numpy as np
import pandas as pd

IST = "Asia/Kolkata"


@dataclass
class NewsItem:
    id: str
    source: str
    region: str                      # "IN" | "GLOBAL"
    published_at: pd.Timestamp       # tz-aware
    title: str
    url: str = ""
    summary: str = ""


# ---------------------------------------------------------------- dedupe / novelty
_STOP = {"the", "a", "an", "of", "to", "in", "on", "for", "and", "at", "as", "by", "with", "is", "are"}


def title_tokens(title: str) -> frozenset:
    t = re.sub(r"[^a-z0-9% ]+", " ", title.lower())
    return frozenset(w for w in t.split() if w not in _STOP and len(w) > 1)


def jaccard(a: frozenset, b: frozenset) -> float:
    return len(a & b) / len(a | b) if a and b else 0.0


class Deduper:
    """Clusters near-duplicate headlines inside a time window. Novelty falls as a story repeats."""

    def __init__(self, window_hours: float = 24, threshold: float = 0.8):
        self.window = pd.Timedelta(hours=window_hours)
        self.thr = threshold
        self._clusters: list[dict] = []

    def add(self, item: NewsItem) -> tuple[int, float]:
        toks = title_tokens(item.title)
        now = item.published_at
        self._clusters = [c for c in self._clusters if now - c["last"] <= self.window]
        for i, c in enumerate(self._clusters):
            if jaccard(toks, c["tokens"]) >= self.thr:
                c["n"] += 1
                c["last"] = now
                return c["id"], 1.0 / c["n"]
        cid = hash((item.id, toks)) & 0x7FFFFFFF
        self._clusters.append({"id": cid, "tokens": toks, "n": 1, "last": now})
        return cid, 1.0


# ---------------------------------------------------------------- entity linking
MACRO_KEYWORDS = {
    "RATES_IN": ["rbi", "repo rate", "mpc", "reserve bank of india"],
    "RATES_US": ["fed", "fomc", "federal reserve", "treasury yield"],
    "CRUDE": ["crude", "brent", "opec"],
    "USDINR": ["rupee", "usd/inr", "dollar index"],
    "TRADE_POLICY": ["tariff", "trade deal", "export curbs"],
    "BUDGET_POLICY": ["budget", "gst council", "finance ministry"],
    "GEOPOLITICS": ["war", "sanction", "ceasefire", "missile"],
    "MONSOON": ["monsoon", "imd forecast"],
}

_SUFFIX = re.compile(r"\b(limited|ltd|corporation|corp|inc|industries)\b\.?", re.I)


def aliases_from_instruments(instruments: pd.DataFrame, min_symbol_len: int = 4) -> dict[str, str]:
    """instruments needs columns: symbol, name. Builds lowercase alias -> symbol.

    Short tickers (e.g. 'ITC') are ambiguous as plain words, so only company-name aliases are used for them.
    """
    out: dict[str, str] = {}
    for r in instruments.itertuples():
        full = re.sub(r"\s+", " ", str(r.name).replace(".", " ").replace(",", " ")).strip().lower()
        if " " in full:                     # "itc ltd" is unambiguous even though "itc" alone is not
            out[full] = r.symbol
        name = re.sub(r"\s+", " ", _SUFFIX.sub("", full)).strip()
        if len(name) >= 4:
            out[name] = r.symbol
        if len(r.symbol) >= min_symbol_len:
            out[r.symbol.lower()] = r.symbol
    return out


class EntityLinker:
    def __init__(self, aliases: dict[str, str], sector_map: dict[str, str] | None = None):
        keys = sorted(aliases, key=len, reverse=True)           # longest alias wins
        self._rx = re.compile(r"\b(" + "|".join(re.escape(k) for k in keys) + r")\b", re.I) if keys else None
        self._aliases = aliases
        self.sector_map = sector_map or {}

    def link(self, text: str) -> tuple[list[str], list[str]]:
        t = text.lower()
        syms = []
        if self._rx:
            for m in self._rx.finditer(text):
                s = self._aliases[m.group(1).lower()]
                if s not in syms:
                    syms.append(s)
        tags = [tag for tag, kws in MACRO_KEYWORDS.items() if any(re.search(rf"\b{re.escape(k)}\b", t) for k in kws)]
        return syms, tags


# ---------------------------------------------------------------- event type
EVENT_RULES: list[tuple[str, str]] = [
    ("fundraise", r"\b(qip|rights issue|ipo|fund ?raise|raises? rs|preferential allotment|ncd)\b"),
    ("mna", r"\b(acquir|acquisition|merger|demerger|buyout|stake sale|open offer)\w*"),
    ("order_win", r"\b(bags?|wins?|secures?|lands?) .*\b(order|contract|deal)\b"),
    ("rating", r"\b(upgrade[sd]?|downgrade[sd]?|target price|initiates coverage|rating)\b"),
    ("regulatory_legal", r"\b(sebi|nclt|ed raid|penalty|fine[sd]?|probe|court|fraud|ban)\b"),
    ("management", r"\b(resigns?|appoints?|ceo|cfo|managing director|promoter)\b"),
    ("corporate_action", r"\b(dividend|bonus|stock split|buyback|record date)\b"),
    ("guidance", r"\b(guidance|outlook|forecast|margin expansion)\b"),
    ("earnings", r"\b(q[1-4]|results?|profit|net loss|revenue|ebitda|earnings)\b"),
]
_EVENT_RX = [(n, re.compile(p, re.I)) for n, p in EVENT_RULES]


def classify_event(title: str, macro_tags: list[str]) -> str:
    for name, rx in _EVENT_RX:
        if rx.search(title):
            return name
    if macro_tags:
        return "macro_" + macro_tags[0].lower()
    return "other"


# ---------------------------------------------------------------- sentiment (pluggable)
class SentimentModel(Protocol):
    def score(self, text: str) -> tuple[float, float]:
        """Return (sentiment in [-1, 1], confidence in [0, 1])."""


_POS = {"surge", "jump", "rally", "beat", "beats", "record", "upgrade", "upgraded", "wins", "wins", "bags", "gain",
        "gains", "strong", "growth", "profit", "rise", "rises", "soar", "outperform", "approval", "approved", "raises"}
_NEG = {"fall", "falls", "plunge", "slump", "miss", "misses", "downgrade", "downgraded", "loss", "losses", "weak",
        "probe", "fraud", "penalty", "fine", "ban", "cut", "cuts", "decline", "drops", "default", "resigns",
        "crash", "sell-off", "selloff", "warning", "slashes"}
_NEGATORS = {"not", "no", "never", "without", "fails", "failed"}


class LexiconSentiment:
    """Cheap baseline so the pipeline runs anywhere. Replace with FinBertSentiment for real use."""

    def score(self, text: str) -> tuple[float, float]:
        words = re.findall(r"[a-z\-]+", text.lower())
        s = 0
        hits = 0
        for i, w in enumerate(words):
            v = 1 if w in _POS else (-1 if w in _NEG else 0)
            if v:
                if i > 0 and words[i - 1] in _NEGATORS:
                    v = -v
                s += v
                hits += 1
        if hits == 0:
            return 0.0, 0.0
        return float(np.tanh(s / 2)), min(1.0, hits / 3)


class FinBertSentiment:
    """Runs on CPU. Needs `pip install transformers torch`. Not used in tests (no model download)."""

    def __init__(self, model_name: str = "ProsusAI/finbert"):
        from transformers import pipeline          # lazy: heavy import
        self._pipe = pipeline("text-classification", model=model_name, top_k=None)

    def score(self, text: str) -> tuple[float, float]:
        res = {d["label"].lower(): d["score"] for d in self._pipe(text[:512])[0]}
        return float(res.get("positive", 0) - res.get("negative", 0)), float(1 - res.get("neutral", 0))


SOURCE_CREDIBILITY = {"exchange_filing": 1.0, "regulator": 1.0, "reuters": 0.9, "bloomberg": 0.9}
DEFAULT_CREDIBILITY = 0.6


@dataclass
class ScoredNews:
    item: NewsItem
    cluster_id: int
    novelty: float
    symbols: list[str]
    macro_tags: list[str]
    event_type: str
    sentiment: float
    confidence: float
    credibility: float
    feature_date: pd.Timestamp | None = None

    @property
    def weight(self) -> float:
        return self.novelty * self.credibility * self.confidence


class NewsPipeline:
    def __init__(self, linker: EntityLinker, sentiment: SentimentModel | None = None,
                 dedupe: Deduper | None = None):
        self.linker, self.sentiment, self.dedupe = linker, sentiment or LexiconSentiment(), dedupe or Deduper()

    def process(self, item: NewsItem) -> ScoredNews:
        cid, novelty = self.dedupe.add(item)
        syms, tags = self.linker.link(item.title + " " + item.summary)
        ev = classify_event(item.title, tags)
        s, conf = self.sentiment.score(item.title)
        return ScoredNews(item, cid, novelty, syms, tags, ev, s, conf,
                          SOURCE_CREDIBILITY.get(item.source, DEFAULT_CREDIBILITY))


# ---------------------------------------------------------------- point-in-time alignment
def feature_date(published_at: pd.Timestamp, calendar: pd.DatetimeIndex, ingest_delay_min: float = 10,
                 decision_time: time = time(15, 20)) -> pd.Timestamp | None:
    """First trading date whose decision point is AFTER we could have seen the item.

    known = published + ingest delay (IST). The item belongs to the first trading day d with
    known <= d at `decision_time`. Weekend/holiday news rolls to the next trading day.
    """
    known = (published_at + pd.Timedelta(minutes=ingest_delay_min)).tz_convert(IST)
    for d in calendar[calendar >= known.normalize().tz_localize(None)]:
        cutoff = pd.Timestamp.combine(d.date(), decision_time).tz_localize(IST)
        if known <= cutoff:
            return d
    return None


# ---------------------------------------------------------------- measured impact
def event_study(events: pd.DataFrame, bars: dict[str, pd.DataFrame], index_close: pd.Series,
                horizons: Iterable[int] = (1, 3, 5), est_window: int = 120, est_gap: int = 5) -> pd.DataFrame:
    """events columns: symbol, event_type, sign (-1/0/+1), feature_date.

    Abnormal return over h days from the close of feature_date: stock return minus beta * index return,
    with beta estimated on `est_window` days that end `est_gap` days BEFORE the event (no contamination).
    """
    rows = []
    horizons = list(horizons)
    for e in events.itertuples():
        b = bars.get(e.symbol)
        if b is None or e.feature_date not in b.index:
            continue
        c = b["close"]
        ic = index_close.reindex(c.index).ffill()
        i = c.index.get_loc(e.feature_date)
        lo, hi = i - est_gap - est_window, i - est_gap
        if lo < 1 or i + max(horizons) >= len(c):
            continue
        r, ir = c.pct_change().iloc[lo:hi], ic.pct_change().iloc[lo:hi]
        var = ir.var()
        beta = float(r.cov(ir) / var) if var and not np.isnan(var) else 1.0
        row = dict(symbol=e.symbol, event_type=e.event_type, sign=e.sign, feature_date=e.feature_date, beta=beta)
        for h in horizons:
            row[f"ar_{h}"] = float((c.iloc[i + h] / c.iloc[i] - 1) - beta * (ic.iloc[i + h] / ic.iloc[i] - 1))
        rows.append(row)
    return pd.DataFrame(rows)


@dataclass
class ImpactTable:
    """Expected abnormal return per (event_type, sign), shrunk toward zero when evidence is thin."""
    table: pd.DataFrame
    shrink_k: float = 20.0
    min_n: int = 30

    @classmethod
    def fit(cls, studied: pd.DataFrame, horizon: int = 3, shrink_k: float = 20.0, min_n: int = 30) -> "ImpactTable":
        col = f"ar_{horizon}"
        rows = []
        for (ev, sg), g in studied.groupby(["event_type", "sign"]):
            x = g[col].dropna()
            n = len(x)
            if n == 0:
                continue
            mean, sd = float(x.mean()), float(x.std(ddof=1)) if n > 1 else float("nan")
            t = mean / (sd / np.sqrt(n)) if n > 1 and sd > 0 else 0.0
            rows.append(dict(event_type=ev, sign=sg, n=n, mean=mean, shrunk_mean=mean * n / (n + shrink_k),
                             t_stat=t, hit_rate=float((np.sign(x) == sg).mean()) if sg != 0 else np.nan))
        return cls(pd.DataFrame(rows), shrink_k, min_n)

    def expected(self, event_type: str, sign: int) -> dict:
        t = self.table
        r = t[(t.event_type == event_type) & (t.sign == sign)] if len(t) else t
        if len(r) == 0:
            return dict(expected_ar=0.0, n=0, t_stat=0.0, evidence="none")
        r = r.iloc[0]
        ok = r.n >= self.min_n and abs(r.t_stat) >= 2.0
        return dict(expected_ar=float(r.shrunk_mean) if ok else 0.0, n=int(r.n), t_stat=float(r.t_stat),
                    evidence="measured" if ok else "insufficient")


def news_feature_frame(scored: list[ScoredNews], calendar: pd.DatetimeIndex, symbols: list[str],
                       impact: ImpactTable | None = None, half_life_days: float = 1.5) -> dict[str, pd.DataFrame]:
    """Per-symbol daily news features plus a market-wide macro frame, decayed and point-in-time safe.

    Items must already have `feature_date` set. Returns {symbol: DataFrame, "_macro": DataFrame}.
    """
    decay = 0.5 ** (1.0 / half_life_days)
    per_sym = {s: pd.DataFrame(0.0, index=calendar, columns=["news_sent", "news_count", "news_impact"]) for s in symbols}
    tags = sorted(MACRO_KEYWORDS)
    macro = pd.DataFrame(0.0, index=calendar, columns=[f"macro_{t.lower()}" for t in tags])
    for n in scored:
        d = n.feature_date
        if d is None or d not in calendar:
            continue
        sign = int(np.sign(n.sentiment)) if abs(n.sentiment) > 0.05 else 0
        for s in n.symbols:
            if s in per_sym:
                f = per_sym[s]
                f.loc[d, "news_sent"] += n.sentiment * n.weight
                f.loc[d, "news_count"] += n.novelty
                if impact is not None and sign != 0:
                    f.loc[d, "news_impact"] += impact.expected(n.event_type, sign)["expected_ar"] * n.novelty
        for t in n.macro_tags:
            macro.loc[d, f"macro_{t.lower()}"] += n.sentiment * n.weight
    for f in list(per_sym.values()) + [macro]:
        for col in f.columns:
            v = f[col].to_numpy()
            out = np.zeros_like(v)
            acc = 0.0
            for i in range(len(v)):
                acc = acc * decay + v[i]
                out[i] = acc
            f[col] = out
    per_sym["_macro"] = macro
    return per_sym
