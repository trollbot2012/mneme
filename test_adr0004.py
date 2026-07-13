"""ADR-0004 gate tests (optional embedding rerank, UNION-then-rerank).

Section 1 (gate 1, fixture integrity) — runs WHEEL-ABSENT, no onnxruntime:

    uv run --python 3.11 --with pytest pytest test_adr0004.py -k "fixture or zero_overlap" -q

Later sections (gates 2-6: union divergence, uplift, encode guard, zero-encode
reindex, off-path byte-identity) land with the engine work and extend the run
matrix documented here:

    uv run --python 3.11 --with pytest pytest -q                       # gate 6 env
    uv run --python 3.14 --with pytest --with onnxruntime pytest test_adr0004.py -q
"""
import sqlite3
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent / "bench"))

from mneme import _FTS_SCHEMA, _tokens, slugify  # noqa: E402
from scale_bench import (  # noqa: E402
    KEYWORD_QUERIES,
    TARGETS,
    ZERO_OVERLAP_TARGETS,
    _PARAPHRASE_TARGETS,
    is_hit,
)


# --- Section 1: fixture invariants (ADR gate 1) -------------------------------

def test_fixture_size_and_class_ordering():
    assert len(TARGETS) >= 100
    assert len(ZERO_OVERLAP_TARGETS) >= 25
    # Composition is load-bearing: zero-overlap LAST keeps mechanism_bakeoff's
    # rid=idx+1 rule and enables per-class index splits in engine_gate.
    assert TARGETS == _PARAPHRASE_TARGETS + ZERO_OVERLAP_TARGETS
    for entry in TARGETS:
        assert isinstance(entry, tuple) and len(entry) == 3
        title, body, query = entry
        assert title.strip() and body.strip() and query.strip()


def test_fixture_uniqueness_invariants():
    titles = [t for t, _, _ in TARGETS]
    # exact titles unique (is_hit's ruler must be unambiguous)
    assert len(set(titles)) == len(titles)
    # first-40-lowercase unique: the retired substring rule must never have
    # been ambiguous either, and no two notes can shadow each other's prefix
    prefixes = [t.lower()[:40] for t in titles]
    assert len(set(prefixes)) == len(prefixes), sorted(
        p for p in prefixes if prefixes.count(p) > 1)
    # slug unique: MnemeProvider.seed calls add_note(title[:120]) with
    # overwrite=True — a slug collision silently REPLACES an earlier note file
    slugs = [slugify(t[:120]) for t in titles]
    assert all(slugs)
    assert len(set(slugs)) == len(slugs), sorted(
        s for s in slugs if slugs.count(s) > 1)


def test_fixture_keyword_queries_derived():
    # keyword-identical class is DERIVED (query == title), never hand-authored
    assert KEYWORD_QUERIES == [(t, b, t) for t, b, _ in TARGETS]


def test_fixture_match_rule_is_exact_title_equality():
    assert is_hit("Alpha beta", ["other", "Alpha beta"])
    assert not is_hit("Alpha beta", [])
    assert not is_hit("Alpha beta", ["alpha beta"])       # not case-folded
    assert not is_hit("Alpha beta", ["Alpha beta gamma"])  # not substring
    assert not is_hit("Alpha beta", ["Alpha"])             # not prefix


def test_zero_overlap_queries_share_no_stem_with_their_target():
    """Each ZERO_OVERLAP_TARGETS query must MATCH zero rows against a scratch
    FTS table holding only its own target, built from the production
    _FTS_SCHEMA (fts5, tokenize='porter unicode61') — exactly the tokenizer
    the engine indexes with, and the OR-of-quoted-tokens MATCH string exactly
    as _candidates builds it."""
    con = sqlite3.connect(":memory:")
    try:
        con.executescript(_FTS_SCHEMA)
    except sqlite3.OperationalError:
        pytest.skip("sqlite build lacks FTS5; zero-overlap check needs the production tokenizer")
    try:
        for title, body, query in ZERO_OVERLAP_TARGETS:
            toks = _tokens(query, cap=256)  # engine regex, uncapped in practice
            assert toks, f"query tokenizes to nothing: {query!r}"
            con.execute("DELETE FROM mem_fts")
            con.execute(
                "INSERT INTO mem_fts (dedupe_key, title, body, keywords, tags)"
                " VALUES ('k', ?, ?, '', '')", (title, body))
            match = " OR ".join(f'"{t}"' for t in toks)
            n = con.execute("SELECT COUNT(*) FROM mem_fts WHERE mem_fts MATCH ?",
                            (match,)).fetchone()[0]
            offenders = [t for t in toks if con.execute(
                "SELECT COUNT(*) FROM mem_fts WHERE mem_fts MATCH ?",
                (f'"{t}"',)).fetchone()[0]] if n else []
            assert n == 0, (f"lexical overlap: query {query!r} reaches target "
                            f"{title!r} via tokens {offenders}")
    finally:
        con.close()
