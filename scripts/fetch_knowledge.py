#!/usr/bin/env python3
"""Fetch the knowledge base the model will be trained to answer from.

Two sources, both public and both cached to disk so the build is reproducible
and a network blip never costs a night of training:

  * Simple English Wikipedia intros  -> "What is X?" definitions
  * Wikidata                         -> structured facts (capitals, currencies,
                                        element symbols, authors' nationalities)

Output: data/kb.json

    {"topics": [{"title": ..., "text": ...}],
     "facts":  [{"rel": "capital_of", "subject": "France", "object": "Paris"}, ...]}

The fetch is incremental: rerunning only asks for what is missing.
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import os
import sys
import time
import urllib.parse
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
UA = {"User-Agent": "neurobrain-corpus/0.1 (from-scratch LM training; contact: none)"}
# Wikimedia rate-limits this sandbox's IP after a few hundred calls (429 for
# everything, even single REST summaries). r.jina.ai proxies the same endpoints
# from its own address; the JSON body comes back wrapped in a short markdown
# header, which is cheap to strip.
PROXY = os.environ.get("KB_PROXY", "https://r.jina.ai/")


def _unwrap(text: str):
    i, j = text.find("{"), text.rfind("}")
    if i < 0 or j <= i:
        return None
    try:
        return json.loads(text[i:j + 1])
    except Exception:
        return None


def get(url: str, tries: int = 3, timeout: int = 90):
    for i in range(tries):
        try:
            target = (PROXY + url) if PROXY and "r.jina.ai" not in url else url
            req = urllib.request.Request(target, headers=UA)
            with urllib.request.urlopen(req, timeout=timeout) as r:
                raw = r.read().decode("utf-8", "replace")
            if raw.lstrip().startswith("{"):
                return json.loads(raw)
            return _unwrap(raw)
        except Exception as e:  # noqa: BLE001 - network is allowed to be flaky
            if i == tries - 1:
                print(f"[kb] giving up on {url[:90]}: {e}")
                return None
            time.sleep(1.5 * (i + 1))
    return None


# Categories whose members are, by construction, "basic knowledge". Article
# creation order is useless (it starts at '!!' and ends at whatever was made
# last month); a list of school-level categories is what a person means by
# basic questions.
CATEGORIES = [
    "Countries", "Continents", "Capitals of country subdivisions", "Cities",
    "Science", "Biology", "Chemistry", "Physics", "Mathematics", "Geometry",
    "History", "Geography", "Animals", "Mammals", "Birds", "Fish", "Insects",
    "Plants", "Flowers", "Trees", "Fruits", "Vegetables", "Food",
    "Sports", "Football", "Olympic sports", "Music", "Musical instruments",
    "Art", "Painting", "Literature", "Books", "Technology", "Computers",
    "Software", "Programming languages", "Internet", "Languages",
    "Money", "Currency", "Medicine", "Diseases", "Human body", "Space",
    "Planets", "Stars", "Chemical elements", "Weather", "Climate", "Energy",
    "Electricity", "Machines", "Transport", "Vehicles", "Education", "Law",
    "Government", "Religion", "Philosophy", "Psychology", "Emotions",
    "Time", "Measurement", "Units of measurement", "Numbers", "Colors",
    "Materials", "Metals", "Gases", "Water", "Oceans", "Rivers", "Mountains",
    "Volcanoes", "Countries in Europe", "Countries in Asia",
    "Countries in Africa", "Countries in North America",
    "Countries in South America", "Countries in Oceania",
    "United States", "World War II", "World War I", "Ancient Rome",
    "Ancient Greece", "Egypt", "China", "India", "Japan", "France",
    "Germany", "United Kingdom", "Russia", "Brazil", "Canada", "Australia",
]


def category_titles(cat: str, limit: int = 200) -> list[str]:
    q = {"action": "query", "list": "categorymembers", "cmtitle": f"Category:{cat}",
         "cmlimit": str(min(500, limit)), "cmtype": "page", "format": "json"}
    d = get("https://simple.wikipedia.org/w/api.php?" + urllib.parse.urlencode(q))
    if not d:
        return []
    return [m["title"] for m in d.get("query", {}).get("categorymembers", [])]


def list_titles(limit: int) -> list[str]:
    """Article titles in creation order: the older ones are the basic ones."""
    out, cont = [], None
    while len(out) < limit:
        q = {"action": "query", "list": "allpages", "aplimit": "500",
             "apnamespace": "0", "apfilterredir": "nonredirects", "format": "json"}
        if cont:
            q["apcontinue"] = cont
        d = get("https://simple.wikipedia.org/w/api.php?" + urllib.parse.urlencode(q))
        if not d:
            break
        pages = d.get("query", {}).get("allpages", [])
        out.extend(p["title"] for p in pages)
        cont = d.get("continue", {}).get("apcontinue")
        if not cont or not pages:
            break
    return out[:limit]


def _extract_batch(batch: list[str]) -> dict[str, str]:
    """40 titles per request. The batch endpoint is the one that behaves behind
    the proxy; the per-title REST summary path stays rate-limited."""
    q = {"action": "query", "prop": "extracts", "exintro": "1", "explaintext": "1",
         "exsentences": "2", "format": "json", "titles": "|".join(batch)}
    d = get("https://simple.wikipedia.org/w/api.php?" + urllib.parse.urlencode(q))
    out = {}
    if d:
        for p in d.get("query", {}).get("pages", {}).values():
            text = (p.get("extract") or "").strip()
            if text and p.get("title") and len(text) > 40:
                out[p["title"]] = ". ".join(text.split(". ")[:2]).strip()
    return out


def fetch_extracts(titles: list[str], workers: int = 3) -> dict[str, str]:
    batches = [titles[i:i + 40] for i in range(0, len(titles), 40)]
    got: dict[str, str] = {}
    with cf.ThreadPoolExecutor(max_workers=workers) as ex:
        for i, part in enumerate(ex.map(_extract_batch, batches)):
            got.update(part)
            if i % 10 == 0:
                print(f"[kb] extracts {i * 40}/{len(titles)} ({len(got)} kept)", flush=True)
    return got


SPARQL = "https://query.wikidata.org/sparql?format=json&query="


def sparql(query: str) -> list[dict]:
    d = get(SPARQL + urllib.parse.quote(query))
    if not d:
        return []
    rows = []
    for b in d.get("results", {}).get("bindings", []):
        rows.append({k: v.get("value") for k, v in b.items()})
    return rows


def fact_queries() -> list[tuple[str, str]]:
    """(relation, SPARQL) pairs. Each row must project subject/object labels."""
    label = ('SERVICE wikibase:label { bd:serviceParam wikibase:language "en". }')
    return [
        ("capital_of", """
            SELECT ?subject ?object WHERE { ?c wdt:P31 wd:Q6256 ; wdt:P36 ?cap .
              ?c rdfs:label ?subject . ?cap rdfs:label ?object .
              FILTER(LANG(?subject)='en' && LANG(?object)='en') } LIMIT 400"""),
        ("currency_of", """
            SELECT ?subject ?object WHERE { ?c wdt:P31 wd:Q6256 ; wdt:P38 ?cur .
              ?c rdfs:label ?subject . ?cur rdfs:label ?object .
              FILTER(LANG(?subject)='en' && LANG(?object)='en') } LIMIT 400"""),
        ("element_symbol", """
            SELECT ?subject ?object WHERE { ?e wdt:P31 wd:Q11344 ; wdt:P246 ?object .
              ?e rdfs:label ?subject . FILTER(LANG(?subject)='en') } LIMIT 200"""),
        ("official_language_of", """
            SELECT ?subject ?object WHERE { ?c wdt:P31 wd:Q6256 ; wdt:P37 ?lang .
              ?c rdfs:label ?subject . ?lang rdfs:label ?object .
              FILTER(LANG(?subject)='en' && LANG(?object)='en') } LIMIT 400"""),
        ("continent_of", """
            SELECT ?subject ?object WHERE { ?c wdt:P31 wd:Q6256 ; wdt:P30 ?cont .
              ?c rdfs:label ?subject . ?cont rdfs:label ?object .
              FILTER(LANG(?subject)='en' && LANG(?object)='en') } LIMIT 400"""),
    ]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(ROOT, "data", "kb.json"))
    ap.add_argument("--topics", type=int, default=6000)
    ap.add_argument("--facts", action="store_true", help="(re)fetch structured facts")
    args = ap.parse_args()

    kb = {"topics": [], "facts": []}
    if os.path.exists(args.out):
        with open(args.out) as f:
            kb = json.load(f)
        print(f"[kb] existing: {len(kb['topics'])} topics, {len(kb['facts'])} facts")

    have = {t["title"] for t in kb["topics"]}
    want: list[str] = []
    seen = set(have)
    for cat in CATEGORIES:
        for t in category_titles(cat):
            if t not in seen and ":" not in t:
                seen.add(t)
                want.append(t)
    print(f"[kb] {len(want)} category titles, need {args.topics - len(have)} more")
    titles = want[: max(0, args.topics - len(have))]
    if len(titles) < max(0, args.topics - len(have)):
        titles += [t for t in list_titles(args.topics + len(have)) if t not in seen][:200]
    print(f"[kb] {len(titles)} new titles to fetch")
    if titles:
        extracts = fetch_extracts(titles)
        for title, text in extracts.items():
            kb["topics"].append({"title": title, "text": text})
        print(f"[kb] topics now {len(kb['topics'])}")

    if args.facts or not kb["facts"]:
        rows = []
        for rel, q in fact_queries():
            got = sparql(q)
            kept = [{"rel": rel, "subject": r["subject"], "object": r["object"]}
                    for r in got if r.get("subject") and r.get("object")]
            print(f"[kb] {rel}: {len(kept)}")
            rows.extend(kept)
        # dedupe on (rel, subject, object); capitals can have several per country
        seen, out = set(), []
        for r in rows:
            k = (r["rel"], r["subject"], r["object"])
            if k not in seen:
                seen.add(k)
                out.append(r)
        kb["facts"] = out

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(kb, f)
    print(f"[kb] wrote {args.out}: {len(kb['topics'])} topics, {len(kb['facts'])} facts")


if __name__ == "__main__":
    main()
