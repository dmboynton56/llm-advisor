#!/usr/bin/env python3
"""
Multi-source market & watchlist news scraper.

Sources:
- Alpaca News API (if ALPACA_API_KEY/SECRET found)
- Google News RSS (macro, per-symbol, per-industry)
- Finviz ticker pages (per-symbol headlines + Sector/Industry discovery)

Window:
- By default (for the morning run): previous ET day 16:00 -> today ET 09:30
- Override via env:
  NEWS_WINDOW_START_ET="YYYY-MM-DDTHH:MM"  (ET)
  NEWS_WINDOW_END_ET="YYYY-MM-DDTHH:MM"    (ET)

Output:
  data/daily_news/YYYY-MM-DD/raw/news.json
Schema (top-level):
{
  "date_et": "YYYY-MM-DD",
  "window_et": {"start": "...", "end": "..."},
  "sources_used": ["alpaca","google_news","finviz"],
  "universe": ["SPY","QQQ",...],
  "industries": {"NVDA": {"sector":"Technology","industry":"Semiconductors"}, ...},
  "macro": [Article, ...],
  "symbols": {"SPY": [Article, ...], "NVDA": [...], ...}
}

Article fields:
{
  "headline": str,
  "source": str,
  "published_utc": ISO-8601 UTC with tz,
  "url": str,
  "summary": str,
  "tags": [str],
  "tickers": [str],
  "keywords": [str],
  "sentiment": {"compound": float, "label": "positive|neutral|negative", "method": "vader|lexicon"}
}

Dependencies:
  pip install requests beautifulsoup4 python-dotenv pandas
  (optional) pip install vaderSentiment
"""

import os, sys, json, time, re, html, urllib.parse
from pathlib import Path
from datetime import datetime, timezone
import requests
import pandas as pd
from bs4 import BeautifulSoup
import time

# Project root (this file lives in src/data_processing/)
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from dotenv import load_dotenv
from config.settings import WATCHLIST  # list[str]

ET_TZ = "US/Eastern"
UA = {"User-Agent": "Mozilla/5.0 (compatible; LLM-Advisor/1.0)"}

# ---------------------- Time helpers ---------------------- #
def _today_et_date():
    return pd.Timestamp.now(tz=ET_TZ).date()

def _parse_env_window_or_default():
    """Prev ET 16:00 -> Today ET 09:30, unless NEWS_WINDOW_* override."""
    start_env = os.getenv("NEWS_WINDOW_START_ET")
    end_env = os.getenv("NEWS_WINDOW_END_ET")
    if start_env and end_env:
        start = pd.Timestamp(start_env).tz_localize(ET_TZ) if pd.Timestamp(start_env).tzinfo is None else pd.Timestamp(start_env).tz_convert(ET_TZ)
        end   = pd.Timestamp(end_env).tz_localize(ET_TZ) if pd.Timestamp(end_env).tzinfo is None else pd.Timestamp(end_env).tz_convert(ET_TZ)
        return start, end
    now_et = pd.Timestamp.now(tz=ET_TZ)
    prev = (now_et - pd.Timedelta(days=1)).date()
    start = pd.Timestamp.combine(pd.Timestamp(prev), pd.Timestamp("16:00").time()).tz_localize(ET_TZ)
    end   = pd.Timestamp.combine(pd.Timestamp(now_et.date()), pd.Timestamp("09:30").time()).tz_localize(ET_TZ)
    return start, end

# ---------------------- Tagging / keyword / sentiment ---------------------- #
_TAG_RULES = [
    ("earnings", ["earnings","eps","revenue","guidance","outlook","beat","miss"]),
    ("upgrade", ["upgrade","raised to","overweight","outperform","buy rating","upgrade to"]),
    ("downgrade", ["downgrade","underweight","underperform","sell rating","cut to"]),
    ("price_target", ["price target","pt raised","pt cut"]),
    ("mna", ["merger","acquisition","acquires","combine","takeover","buyout","deal"]),
    ("sec", ["sec","investigation","subpoena","settlement","charges"]),
    ("macro_cpi", ["cpi","inflation","consumer price"]),
    ("macro_ppi", ["ppi","producer price"]),
    ("macro_jobs", ["nonfarm","payrolls","jobs report","initial claims","unemployment"]),
    ("macro_fomc", ["fed","fomc","powell","rate hike","rate cut","dot plot","minutes"]),
    ("macro_gdp", ["gdp","growth"]),
    ("macro_isms", ["ism","pmi","manufacturing index"]),
    ("yields", ["treasury","yield","10-year","2-year","curve","term premium"]),
    ("strike", ["strike","walkout","union"]),
    ("legal", ["lawsuit","litigation","court","injunction"]),
    ("guidance", ["guidance","forecast","outlook"]),
    ("ai", ["ai","artificial intelligence","genai","chatbot","llm"]),
    ("chips", ["chip","semiconductor","gpu","foundry","node"]),
    ("ev", ["ev","electric vehicle","battery","gigafactory"]),
    ("cloud", ["cloud","saas","hyperscaler"]),
    ("cyber", ["cyber","ransomware","breach","vulnerability"]),
    ("oil", ["oil","crude","opec","barrel"]),
    ("energy", ["energy","refinery","gas","lng"]),
    ("regulatory", ["regulator","antitrust","doj","ftc"])
]

_STOPWORDS = set("""
a an and are as at be by for from has have if in into is it its of on or than that the their there they this to was were will with over under up down ahead amid
""".split())

def _tag_text(text: str) -> list:
    t = (text or "").lower()
    tags = [name for name, keys in _TAG_RULES if any(k in t for k in keys)]
    # rollups
    if any(x in tags for x in ["macro_cpi","macro_ppi","macro_jobs","macro_fomc","macro_gdp","macro_isms","yields"]):
        tags.append("macro")
    if "downgrade" in tags or "sec" in tags or "legal" in tags: tags.append("risk")
    if "upgrade" in tags or "price_target" in tags: tags.append("potential_tailwind")
    return sorted(set(tags))

def _extract_keywords(text: str, k: int = 6) -> list:
    """Very light keyword pick: top non-stopword tokens (letters/#/dash), length>=3."""
    if not text: return []
    text = re.sub(r"[^A-Za-z0-9\- ]+", " ", text)
    tokens = [w.lower() for w in text.split() if len(w) >= 3 and w.lower() not in _STOPWORDS]
    if not tokens: return []
    freq = {}
    for w in tokens:
        freq[w] = freq.get(w, 0) + 1
    ranked = sorted(freq.items(), key=lambda x: (-x[1], x[0]))
    return [w for w, _ in ranked[:k]]

def _sentiment(text: str):
    """Try VADER, fallback to a tiny lexicon."""
    t = (text or "").strip()
    if not t:
        return {"compound": 0.0, "label": "neutral", "method": "lexicon"}
    try:
        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
        vs = SentimentIntensityAnalyzer().polarity_scores(t)
        comp = float(vs.get("compound", 0.0))
        label = "positive" if comp >= 0.05 else ("negative" if comp <= -0.05 else "neutral")
        return {"compound": comp, "label": label, "method": "vader"}
    except Exception:
        pos_words = {"beat","up","surge","record","strong","expand","growth","accelerate","raise","upgrade","buy"}
        neg_words = {"miss","down","fall","drop","weak","slow","cut","downgrade","lawsuit","investigation","risk","layoff"}
        score = 0
        tl = t.lower()
        for w in pos_words:
            if w in tl: score += 1
        for w in neg_words:
            if w in tl: score -= 1
        comp = max(-1.0, min(1.0, score / 5.0))
        label = "positive" if comp > 0.2 else ("negative" if comp < -0.2 else "neutral")
        return {"compound": comp, "label": label, "method": "lexicon"}

# ---------------------- Normalize article ---------------------- #
def _to_utc_iso(ts) -> str:
    try:
        p = pd.to_datetime(ts)
    except Exception:
        p = pd.Timestamp.utcnow()
    if p.tzinfo is None:
        try:
            p = p.tz_localize("UTC")
        except Exception:
            p = pd.Timestamp.utcnow().tz_localize("UTC")
    else:
        p = p.tz_convert("UTC")
    return p.isoformat()

def _norm(headline, source, published, url, summary="", tickers=None):
    text_for_tags = f"{headline or ''}. {summary or ''}"
    return {
        "headline": (headline or "").strip(),
        "source": (source or "").strip(),
        "published_utc": _to_utc_iso(published),
        "url": url,
        "summary": (summary or "").strip(),
        "tags": _tag_text(text_for_tags),
        "tickers": sorted(set(tickers or [])),
        "keywords": _extract_keywords(text_for_tags),
        "sentiment": _sentiment(text_for_tags),
    }

# ---------------------- Finviz: Sector/Industry + per-symbol news ---------------------- #
def _finviz_sector_industry(symbol: str):
    url = f"https://finviz.com/quote.ashx?t={urllib.parse.quote(symbol)}"
    try:
        r = requests.get(url, headers=UA, timeout=20)
        if r.status_code != 200: return None, None
        soup = BeautifulSoup(r.text, "html.parser")
        # Sector/Industry typically in snapshot table
        meta = {}
        for td in soup.select("table.snapshot-table2 td"):
            txt = td.get_text(" ", strip=True)
            if txt.lower() in ("sector", "industry"):
                nxt = td.find_next_sibling("td")
                if nxt:
                    meta[txt.lower()] = nxt.get_text(" ", strip=True)
        return meta.get("sector"), meta.get("industry")
    except Exception:
        return None, None

def _finviz_symbol_news(symbol: str, start_et, end_et):
    out = []
    url = f"https://finviz.com/quote.ashx?t={urllib.parse.quote(symbol)}"
    try:
        r = requests.get(url, headers=UA, timeout=20)
        if r.status_code != 200: return out
        soup = BeautifulSoup(r.text, "html.parser")
        tbl = soup.find("table", id="news-table")
        if not tbl: return out
        rows = tbl.find_all("tr")
        # Finviz mixes date-only and time-only rows; assume today if time only
        cur_date = None
        for tr in rows[:60]:
            date_td = tr.find("td", class_="nn-date")
            a = tr.find("a")
            if not a: continue
            headline = a.get_text(" ", strip=True)
            link = a.get("href")
            stamp = date_td.get_text(" ", strip=True) if date_td else ""
            # parse possible patterns
            ts = pd.to_datetime(stamp, errors="coerce")
            if ts is None or pd.isna(ts):
                # if only time, attach today's ET date
                if re.search(r"\d{1,2}:\d{2}\s*(am|pm|AM|PM)", stamp):
                    ts = pd.to_datetime(f"{_today_et_date()} {stamp}", errors="coerce")
            if ts is None or pd.isna(ts):
                continue
            ts = ts.tz_localize(ET_TZ) if ts.tzinfo is None else ts.tz_convert(ET_TZ)
            if not (start_et <= ts <= end_et):
                continue
            out.append(_norm(headline, "Finviz", ts, link, "", [symbol]))
            time.sleep(0.02)
    except Exception:
        return out
    return out

# ---------------------- Google News RSS ---------------------- #
def _google_rss(query: str, start_et, end_et, max_items=30):
    url = f"https://news.google.com/rss/search?q={urllib.parse.quote(query)}&hl=en-US&gl=US&ceid=US:en"
    out = []
    try:
        r = requests.get(url, headers=UA, timeout=20)
        if r.status_code != 200: return out
        import xml.etree.ElementTree as ET
        root = ET.fromstring(r.content)
        items = root.findall(".//item")
        for it in items[:max_items]:
            title = (it.findtext("title") or "").strip()
            link = (it.findtext("link") or "").strip()
            pub = it.findtext("{http://purl.org/dc/elements/1.1/}date") or it.findtext("pubDate") or ""
            src_node = it.find("{http://news.google.com}source")
            source = (src_node.text or "").strip() if src_node is not None else "GoogleNews"
            ts = pd.to_datetime(pub, errors="coerce")
            if ts is None or pd.isna(ts): ts = pd.Timestamp.utcnow()
            ts = ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert("UTC")
            # ET window filter
            ts_et = ts.tz_convert(ET_TZ)
            if not (start_et <= ts_et <= end_et): continue
            out.append(_norm(title, source, ts, link, "", []))
    except Exception:
        return out
    return out

# ---------------------- Alpaca News ---------------------- #
def _alpaca_available():
    return bool(os.getenv("ALPACA_API_KEY") or os.getenv("ALPACA_API_KEY_ID"))

def _fetch_alpaca(symbols, start_et, end_et, per_page=50, max_pages=10):
    api_key = os.getenv("ALPACA_API_KEY") or os.getenv("ALPACA_API_KEY_ID")
    api_secret = os.getenv("ALPACA_SECRET_KEY") or os.getenv("ALPACA_API_SECRET_KEY")
    if not api_key or not api_secret:
        return []
    headers = {
        "Apca-Api-Key-Id": api_key,
        "Apca-Api-Secret-Key": api_secret,
        "Accept": "application/json",
    }
    params = {
        "symbols": ",".join(symbols),
        "start": start_et.tz_convert("UTC").isoformat(),
        "end": end_et.tz_convert("UTC").isoformat(),
        "include_content": "false",
        "limit": str(per_page),
        "sort": "desc",
    }
    base = "https://data.alpaca.markets/v1beta1/news"
    all_items, tok, pages = [], None, 0
    while pages < max_pages:
        q = params.copy()
        if tok: q["page_token"] = tok
        resp = requests.get(base, headers=headers, params=q, timeout=20)
        if resp.status_code != 200: break
        js = resp.json()
        items = js.get("news") or js.get("data") or []
        all_items.extend(items)
        tok = js.get("next_page_token")
        if not tok: break
        pages += 1
        time.sleep(0.25)

    out = []
    for it in all_items:
        syms = it.get("symbols") or []
        headline = it.get("headline") or it.get("title") or ""
        source = (it.get("source") or {}).get("name") if isinstance(it.get("source"), dict) else it.get("source") or ""
        published = it.get("created_at") or it.get("updated_at") or it.get("published_at") or it.get("time_published")
        url = it.get("url") or it.get("link")
        summary = it.get("summary") or ""
        out.append(_norm(headline, source, published, url, summary, syms))
    return out

# ---------------------- Build industry map via Finviz ---------------------- #
def _build_industry_map(symbols):
    ind = {}
    for s in symbols:
        sec, indus = _finviz_sector_industry(s)
        ind[s] = {"sector": sec, "industry": indus}
        time.sleep(0.2)
    return ind

# ---------------------- De-duplication ---------------------- #
def _dedupe_sort(items):
    seen, out = set(), []
    for a in items:
        key = (a.get("url") or "") + "||" + (a.get("headline") or "")
        if key in seen: continue
        seen.add(key); out.append(a)
    return sorted(out, key=lambda x: x.get("published_utc",""), reverse=True)

# ---------------------- MAIN ---------------------- #
def main():
    start_time = time.time()
    # env
    dotenv_path = PROJECT_ROOT / ".env"
    load_dotenv(dotenv_path if dotenv_path.exists() else None)

    et_date = str(_today_et_date())
    start_et, end_et = _parse_env_window_or_default()

    watchlist = list(dict.fromkeys(WATCHLIST))  # preserve order; dedupe
    sources_used = []
    macro_items = []
    symbol_items = {s: [] for s in watchlist}

    # 1) Industry discovery (Finviz)
    industries = _build_industry_map(watchlist)

    # 2) Alpaca news (watchlist + macro proxies)
    alpaca_symbols = watchlist + ["SPY","QQQ","IWM","DIA","TLT","XLF","XLE","XLK","XLY","XLP","XLV"]
    alpaca_items = _fetch_alpaca(alpaca_symbols, start_et, end_et) if _alpaca_available() else []
    if alpaca_items:
        sources_used.append("alpaca")
        # Split into macro vs symbol: anything with broad ETFs or no direct WL symbol → macro
        wl_set = set(watchlist)
        for a in alpaca_items:
            syms = set(a.get("tickers", []))
            if "SPY" in syms or "QQQ" in syms or "IWM" in syms or ("XL" in "".join(syms)):  # sector ETFs heuristic
                macro_items.append(a)
            added_to_symbol = False
            for s in wl_set.intersection(syms):
                symbol_items.setdefault(s, []).append(a)
                added_to_symbol = True
            if not added_to_symbol and not syms:
                macro_items.append(a)

    # 3) Google News RSS — Macro queries (always)
    sources_used.append("google_news")
    macro_queries = [
        "US stocks OR equities premarket OR Wall Street",
        "Federal Reserve OR FOMC OR interest rates",
        "CPI inflation OR PPI inflation",
        "jobs report OR nonfarm payrolls OR unemployment claims",
        "Treasury yields OR 10-year yield OR 2-year yield",
        "VIX volatility OR risk-off OR risk-on",
        "GDP growth OR recession",
        "ISM PMI manufacturing services",
        "oil prices OR OPEC OR crude",
    ]
    for q in macro_queries:
        macro_items.extend(_google_rss(q, start_et, end_et, max_items=25))
        time.sleep(0.15)

    # 4) Google News RSS — Industry queries
    for s, meta in industries.items():
        ind = meta.get("industry")
        sec = meta.get("sector")
        if ind:
            macro_items.extend(_google_rss(f'{ind} industry', start_et, end_et, max_items=15))
            time.sleep(0.1)
        if sec:
            macro_items.extend(_google_rss(f'{sec} sector stocks', start_et, end_et, max_items=10))
            time.sleep(0.1)

    # 5) Google News RSS — Per-symbol queries
    for s in watchlist:
        items = _google_rss(f'{s} stock', start_et, end_et, max_items=25)
        if items:
            symbol_items.setdefault(s, []).extend(items)
        time.sleep(0.1)

    # 6) Finviz per-symbol headlines
    sources_used.append("finviz")
    for s in watchlist:
        items = _finviz_symbol_news(s, start_et, end_et)
        if items:
            symbol_items.setdefault(s, []).extend(items)

    # 7) Dedupe + sort
    macro_items = _dedupe_sort(macro_items)
    for s in watchlist:
        symbol_items[s] = _dedupe_sort(symbol_items.get(s, []))

    # 8) Write out
    out_dir = PROJECT_ROOT / "data" / "daily_news" / et_date / "raw"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "news.json"
    payload = {
        "date_et": et_date,
        "window_et": {"start": start_et.isoformat(), "end": end_et.isoformat()},
        "sources_used": sources_used,
        "universe": watchlist,
        "industries": industries,
        "macro": macro_items,
        "symbols": symbol_items,
    }
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Wrote news file: {out_path}")
    end_time = time.time()
    elapsed = end_time - start_time
    print(f"Time taken: {elapsed:.2f} seconds")

if __name__ == "__main__":
    main()
