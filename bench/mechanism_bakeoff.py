"""Mechanism bake-off: which stdlib-only retrieval/scoring methods survive at
scale? Same paraphrased-query corpus as scale_bench, but here we hold the
CORPUS fixed and swap the SCORER, so the number reflects the mechanism, not the
provider. Answers the operator's "next work": pick mechanisms to fold into the
one unified layer.

All mechanisms are zero-dependency (no embeddings). The honest question is
whether any of them beats the pure-lexical ceiling on PARAPHRASED queries
(synonym-shifted, not keyword-identical) — if none do, the finding is
"semantic needs an optional embedding flag"; if one does, fold it in.

    python bench/mechanism_bakeoff.py --scales 1000 2000
"""
import argparse
import re
import sqlite3
import statistics
import time
from collections import Counter

from scale_bench import TARGETS, distractors  # reuse the fixture

_TOK = re.compile(r"[a-z0-9]{3,}")
_STOP = set("the and for with that this から have has are was were will not you your "
            "any per via from into out off its it's a an of to in on at by is be as or "
            "does do done when where what which who how why then than only also".split())


def toks(s):
    return [t for t in _TOK.findall(s.lower()) if t not in _STOP]


def trigrams(s):
    s = re.sub(r"\s+", " ", s.lower()).strip()
    return {s[i:i + 3] for i in range(len(s) - 2)} if len(s) > 2 else set()


def jac(a, b):
    return len(a & b) / len(a | b) if a and b else 0.0


def build(scale):
    corpus = [(t, b) for t, b, _ in TARGETS] + distractors(max(0, scale - len(TARGETS)))
    c = sqlite3.connect(":memory:")
    c.execute("CREATE TABLE m (rid INTEGER PRIMARY KEY, title TEXT, body TEXT)")
    try:
        c.execute("CREATE VIRTUAL TABLE fts USING fts5(title, body, content='m', content_rowid='rid', tokenize='porter unicode61')")
        fts = True
    except sqlite3.OperationalError:
        fts = False
    for i, (t, b) in enumerate(corpus, 1):
        c.execute("INSERT INTO m (rid, title, body) VALUES (?,?,?)", (i, t, b))
        if fts:
            c.execute("INSERT INTO fts (rowid, title, body) VALUES (?,?,?)", (i, t, b))
    c.commit()
    # document frequency for a deterministic co-occurrence expansion table
    docs = [toks(t + " " + b) for t, b in corpus]
    return c, corpus, docs, fts


def cooccurrence(docs, topn=3):
    """Deterministic pseudo-semantic: for each term, the terms that most often
    co-occur with it in the corpus. Expands a query toward its neighbors —
    a zero-LLM stand-in for 'semantic depth'."""
    co = {}
    for d in docs:
        u = set(d)
        for a in u:
            c = co.setdefault(a, Counter())
            for b in u:
                if a != b:
                    c[b] += 1
    return {a: [w for w, _ in c.most_common(topn)] for a, c in co.items()}


def fts_candidates(conn, query, limit=40):
    qt = toks(query)
    if not qt:
        return []
    match = " OR ".join(f'"{t}"' for t in qt)
    try:
        rows = conn.execute(
            "SELECT m.rid, m.title, m.body, bm25(fts) FROM fts JOIN m ON m.rid=fts.rowid "
            "WHERE fts MATCH ? ORDER BY bm25(fts) LIMIT ?", (match, limit)).fetchall()
    except sqlite3.OperationalError:
        return []
    if not rows:
        return []
    vals = [r[3] for r in rows]
    lo, hi = min(vals), max(vals)
    span = (hi - lo) or 1.0
    return [(r[0], r[1], r[2], (hi - r[3]) / span) for r in rows]  # (rid,title,body,lex0..1)


# --- mechanisms: query -> ranked [rid] ----------------------------------------
def m_bm25(conn, docs, co, query):
    return [r[0] for r in fts_candidates(conn, query)]


def m_bm25_jaccard(conn, docs, co, query):
    qg = trigrams(query)
    scored = [(0.65 * lex + 0.35 * jac(qg, trigrams(t + " " + kw_body(b))), rid)
              for rid, t, b, lex in fts_candidates(conn, query) for kw_body in [lambda x: x[:200]]]
    scored.sort(reverse=True)
    return [rid for _, rid in scored]


def m_jaccard_only(conn, docs, co, query):
    qg = trigrams(query)
    scored = sorted(((jac(qg, trigrams(t + " " + b)), i + 1)
                     for i, (t, b) in enumerate(corpus_ref[0])), reverse=True)
    return [rid for _, rid in scored[:40]]


def m_bm25_expand(conn, docs, co, query):
    """BM25 over the query expanded with corpus co-occurrence neighbors."""
    qt = toks(query)
    expanded = list(qt)
    for t in qt:
        expanded += co.get(t, [])
    match = " OR ".join(f'"{t}"' for t in dict.fromkeys(expanded))
    if not match:
        return []
    try:
        rows = conn.execute(
            "SELECT m.rid FROM fts JOIN m ON m.rid=fts.rowid WHERE fts MATCH ? "
            "ORDER BY bm25(fts) LIMIT 40", (match,)).fetchall()
    except sqlite3.OperationalError:
        return []
    return [r[0] for r in rows]


def m_bm25_kwboost(conn, docs, co, query):
    """BM25 + a boost when query tokens land in a title (entity-lite: titles are
    the operator's own keyword-dense summary)."""
    qt = set(toks(query))
    scored = []
    for rid, t, b, lex in fts_candidates(conn, query):
        tt = set(toks(t))
        boost = len(qt & tt) / (len(qt) or 1)
        scored.append((0.7 * lex + 0.3 * boost, rid))
    scored.sort(reverse=True)
    return [rid for _, rid in scored]


MECHANISMS = {
    "bm25": m_bm25,
    "bm25+jaccard": m_bm25_jaccard,
    "bm25+kwboost": m_bm25_kwboost,
    "bm25+expand": m_bm25_expand,
    "jaccard_only": m_jaccard_only,
}

corpus_ref = [None]  # module handle for jaccard_only


def run(scale):
    conn, corpus, docs, fts = build(scale)
    corpus_ref[0] = corpus
    co = cooccurrence(docs)
    results = {}
    for name, fn in MECHANISMS.items():
        hits, lats = 0, []
        for idx, (title, _body, query) in enumerate(TARGETS):
            t0 = time.perf_counter()
            ranked = fn(conn, docs, co, query)[:3]
            lats.append((time.perf_counter() - t0) * 1000)
            target_rid = idx + 1  # TARGETS are seeded first, rid 1..len
            if target_rid in ranked:
                hits += 1
        results[name] = (round(hits / len(TARGETS), 3),
                         round(statistics.median(lats), 3))
    conn.close()
    return results, fts


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--scales", nargs="*", type=int, default=[1000, 2000])
    args = ap.parse_args()
    print(f"{len(TARGETS)} paraphrased target queries · precision@3 (p) + median latency ms (t)")
    print("mechanism".ljust(16) + "".join(f"{s:>16}" for s in args.scales))
    grids = {s: run(s) for s in args.scales}
    fts_ok = all(g[1] for g in grids.values())
    for name in MECHANISMS:
        row = name.ljust(16)
        for s in args.scales:
            p, t = grids[s][0][name]
            row += f"{f'p={p} t={t}':>16}"
        print(row)
    print(f"\nFTS5 available: {fts_ok}")
    print("Read: a mechanism that beats bm25's p on PARAPHRASED queries is worth "
          "folding in; a tie means it adds cost without recall — leave it out.")
