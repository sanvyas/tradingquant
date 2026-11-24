import numpy as np
import pandas as pd

from nsequant.news import (Deduper, EntityLinker, ImpactTable, LexiconSentiment, NewsItem, NewsPipeline,
                           aliases_from_instruments, classify_event, event_study, feature_date,
                           news_feature_frame)

IST = "Asia/Kolkata"
INSTR = pd.DataFrame({"symbol": ["RELIANCE", "TCS", "ITC", "HDFCBANK"],
                      "name": ["Reliance Industries Limited", "Tata Consultancy Services Ltd", "ITC Ltd.",
                               "HDFC Bank Limited"]})


def item(i, title, ts="2026-03-10 10:00", src="reuters", region="IN"):
    return NewsItem(str(i), src, region, pd.Timestamp(ts, tz=IST), title)


def test_dedupe_repeats_lower_novelty():
    d = Deduper(window_hours=24, threshold=0.7)
    _, n1 = d.add(item(1, "Reliance Industries wins Rs 5000 crore order from government"))
    _, n2 = d.add(item(2, "Reliance Industries wins Rs 5000 crore order from govt", "2026-03-10 10:30"))
    _, n3 = d.add(item(3, "RBI keeps repo rate unchanged", "2026-03-10 10:40"))
    assert n1 == 1.0 and n2 == 0.5 and n3 == 1.0


def test_dedupe_window_expires():
    d = Deduper(window_hours=2, threshold=0.7)
    d.add(item(1, "Tata Consultancy Services Q3 profit rises"))
    _, n = d.add(item(2, "Tata Consultancy Services Q3 profit rises", "2026-03-11 10:00"))
    assert n == 1.0


def test_entity_linking_and_short_ticker_ambiguity():
    link = EntityLinker(aliases_from_instruments(INSTR))
    syms, tags = link.link("Reliance Industries and Tata Consultancy Services gain as RBI holds repo rate")
    assert "RELIANCE" in syms and "TCS" in syms and "RATES_IN" in tags
    syms, _ = link.link("Itc of the matter is unclear")           # 3-letter ticker alone must not match
    assert "ITC" not in syms
    syms, _ = link.link("ITC Ltd declares dividend")                 # but full company name does
    assert "ITC" in syms


def test_event_classification():
    assert classify_event("HDFC Bank Q3 profit beats estimates", []) == "earnings"
    assert classify_event("Brokerage upgrades TCS, raises target price", []) == "rating"
    assert classify_event("SEBI slaps penalty on promoter", []) == "regulatory_legal"
    assert classify_event("Brent surges", ["CRUDE"]) == "macro_crude"


def test_sentiment_negation():
    m = LexiconSentiment()
    assert m.score("Company profit surges, beats estimates")[0] > 0
    assert m.score("Company posts loss, shares plunge")[0] < 0
    assert m.score("Profit not strong this quarter")[0] < m.score("Profit strong this quarter")[0]


def _cal():
    return pd.bdate_range("2026-03-02", "2026-03-20")


def test_feature_date_respects_decision_cutoff_and_weekends():
    cal = _cal()
    # 10:00 Tuesday -> same day
    assert feature_date(pd.Timestamp("2026-03-10 10:00", tz=IST), cal) == pd.Timestamp("2026-03-10")
    # 15:15 + 10 min delay = 15:25 > 15:20 cutoff -> next trading day
    assert feature_date(pd.Timestamp("2026-03-10 15:15", tz=IST), cal) == pd.Timestamp("2026-03-11")
    # Saturday news -> Monday
    assert feature_date(pd.Timestamp("2026-03-07 11:00", tz=IST), cal) == pd.Timestamp("2026-03-09")
    # UTC timestamp converts correctly: 04:30 UTC == 10:00 IST
    assert feature_date(pd.Timestamp("2026-03-10 04:30", tz="UTC"), cal) == pd.Timestamp("2026-03-10")


def test_no_future_news_in_feature_row():
    cal = _cal()
    link = EntityLinker(aliases_from_instruments(INSTR))
    p = NewsPipeline(link)
    sn = p.process(item(1, "Tata Consultancy Services profit surges, beats estimates", "2026-03-10 16:30"))
    sn.feature_date = feature_date(sn.item.published_at, cal)
    frames = news_feature_frame([sn], cal, ["TCS"])
    f = frames["TCS"]
    assert f.loc["2026-03-10", "news_sent"] == 0                       # after cutoff: must not leak into 10 March
    assert f.loc["2026-03-11", "news_sent"] > 0
    assert f.loc["2026-03-12", "news_sent"] < f.loc["2026-03-11", "news_sent"]    # decays


def _market(n=700, seed=4):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2022-01-03", periods=n)
    mkt = rng.normal(0.0003, 0.009, n)
    return idx, mkt, rng


def test_event_study_recovers_real_impact_and_rejects_null():
    idx, mkt, rng = _market()
    index_close = pd.Series(100 * np.exp(np.cumsum(mkt)), index=idx)
    bars, ev_rows = {}, []
    for k in range(40):
        sym = f"S{k}"
        r = 1.0 * mkt + rng.normal(0, 0.01, len(idx))
        picks = rng.choice(np.arange(150, len(idx) - 10, 12), size=12, replace=False)
        for i in picks:
            real_type = k % 2 == 0
            if real_type:
                r[i + 1:i + 4] += 0.01                                  # +1%/day for 3 days after 'order_win'
            ev_rows.append(dict(symbol=sym, event_type="order_win" if real_type else "management",
                                sign=1, feature_date=idx[i]))
        c = 100 * np.exp(np.cumsum(r))
        bars[sym] = pd.DataFrame({"close": c}, index=idx)
    studied = event_study(pd.DataFrame(ev_rows), bars, index_close)
    tbl = ImpactTable.fit(studied, horizon=3)
    real = tbl.expected("order_win", 1)
    null = tbl.expected("management", 1)
    assert real["evidence"] == "measured" and real["expected_ar"] > 0.015
    assert null["evidence"] == "insufficient" and null["expected_ar"] == 0.0     # noise is NOT turned into signal


def test_unknown_event_has_no_evidence():
    tbl = ImpactTable.fit(pd.DataFrame({"event_type": ["a"], "sign": [1], "ar_3": [0.05]}), horizon=3)
    assert tbl.expected("a", 1)["evidence"] == "insufficient"                    # n=1 is not evidence
    assert tbl.expected("zzz", 1)["evidence"] == "none"
