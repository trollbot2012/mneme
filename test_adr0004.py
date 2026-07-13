"""ADR-0004 gate tests (optional embedding rerank, UNION-then-rerank).

RUN MATRIX — which interpreter proves what:

  1. Wheel-absent, THE gate-6 environment (onnxruntime physically absent;
     embedding tests skip, everything else — fixture, golden parity,
     import-blocker, off-path identity, cooc rescue — must be green):

        uv run --python 3.11 --with pytest pytest -q

  2. Embeddings-on (gates 2-5 active; needs models/ provisioned via
     scripts/provision_embeddings.{ps1,sh} plus the optional wheel):

        uv run --python 3.14 --with pytest --with onnxruntime pytest test_adr0004.py -q

  3. Deployment-interpreter gate-4 battery (bench, not pytest — exits nonzero
     on any latency/encode-guard breach, fails LOUD if models are absent):

        C:\\Python314\\python.exe bench/scale_bench.py --assert --model-dir models

Per-gate -k selectors: "fixture or zero_overlap" (gate 1), union_diverges
(gate 2), uplift (gate 3), "encode_guard or latency" (gate 4), zero_encodes
(gate 5), golden / import_block (gate 6).

NOTE on gate 3: DEFAULTS['w_sem'] ships 0.0 (neutral placeholder) until WP7
commits the grid-fitted value; the fitted-default uplift test SKIPS with an
explicit message until then, and the divergence-direction probe (fixed probe
w_sem) proves the mechanism moves p@3 the right way TODAY.
"""
import contextlib
import json
import os
import statistics
import sqlite3
import subprocess
import sys
import time
from importlib.util import find_spec
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent / "bench"))

from mneme import (  # noqa: E402
    DEFAULTS,
    EMBED_MODEL_FILES,
    ENCODER_ID,
    JAC_WEIGHT,
    LEX_WEIGHT,
    Mneme,
    _FTS_SCHEMA,
    _tokens,
    slugify,
)
from scale_bench import (  # noqa: E402
    BASELINE_P3_KEYWORD,
    BASELINE_P3_OVERALL,
    BASELINE_P3_ZERO_OVERLAP,
    KEYWORD_QUERIES,
    TARGETS,
    ZERO_OVERLAP_TARGETS,
    _PARAPHRASE_TARGETS,
    distractors,
    is_hit,
)

REPO = Path(__file__).resolve().parent
MODELS_DIR = REPO / "models"
HAVE_MODELS = all((MODELS_DIR / f).is_file() for f in EMBED_MODEL_FILES)
HAVE_ORT = find_spec("onnxruntime") is not None
HAVE_NUMPY = find_spec("numpy") is not None

EMBED = pytest.mark.skipif(
    not (HAVE_MODELS and HAVE_ORT),
    reason="embeddings tests need models/ (scripts/provision_embeddings.*) "
           "and the optional onnxruntime wheel")

SEED_TS = 1_700_000_000.0
EMBED_SCALE = 1000       # mirrors capture_golden / scale_bench --assert
# Divergence-direction probe weight (gates 2/3 mechanism checks). The SHIPPED
# default is fitted by WP7's grid; this constant only has to be a weight at
# which the union mechanism visibly moves rankings, and its assertions carry
# wide margins under the measured values (zero-overlap top-3: 9/30 at 0.5).
PROBE_W_SEM = 0.5


def _calls(mem) -> int:
    """PER-TEXT encode counter (0 when the encoder was never built)."""
    return int(getattr(getattr(mem, "_encoder", None), "calls", 0) or 0)


def _drain(mem) -> int:
    total = 0
    while True:
        n = mem._sync_vectors()
        if not n:
            return total
        total += n


def _seed_direct(mem, entries, *, bank="global", credibility="operator",
                 prefix="note") -> None:
    """Deterministic direct seed (capture_golden precedent): synthetic keys +
    fixed timestamps, no note files — never call reindex() on such a store."""
    with mem._lock:
        for i, (title, body) in enumerate(entries):
            key = f"file:/bench/{prefix}_{i}.md"
            mem._upsert_locked(key, "fact", bank, title, body, "", "",
                               key, False, SEED_TS, SEED_TS,
                               credibility=credibility)
        mem._conn.commit()


@contextlib.contextmanager
def _open(tmp: Path, **cfg):
    cfg.setdefault("embed_model_dir", str(MODELS_DIR))
    mem = Mneme(tmp / "m.db", tmp / "notes", config=cfg)
    try:
        yield mem
    finally:
        mem.close()


@pytest.fixture(scope="module")
def embed_store(tmp_path_factory):
    """One seeded-and-vectorized 1000-row store shared by the embedding tests
    (vectors persist in mem_vec; each test opens its own instance/config)."""
    if not (HAVE_MODELS and HAVE_ORT):
        pytest.skip("embeddings unavailable")
    tmp = tmp_path_factory.mktemp("adr4-embed")
    corpus = [(t, b) for t, b, _ in TARGETS] + distractors(
        EMBED_SCALE - len(TARGETS))
    with _open(tmp) as mem:
        assert mem.embeddings_available, "models present but probe failed"
        _seed_direct(mem, corpus)
        assert _drain(mem) == EMBED_SCALE
    return tmp


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


# --- Section 2: union diverges from rerank-only (ADR gate 2) ------------------

@EMBED
def test_union_diverges_from_rerank_only(embed_store):
    # (name deliberately avoids the substring 'zero_overlap' so gate 1's
    # -k "fixture or zero_overlap" selector never pulls this embed test in)
    """embed_top_n=16 (UNION) vs embed_top_n=0 (rerank-of-BM25-only, the ADR's
    rejected variant) as plain config rows. Zero-overlap targets share no
    porter stem with their query, so BM25 can never nominate them: rerank-only
    must score 0.0 on the class STRUCTURALLY (target absent from the entire
    retrieve output, not merely out of top-3), while the union leg recovers
    them. Cooc rescue is gated off whenever qvec is present, so this
    divergence cannot pass for the wrong reason."""
    with _open(embed_store, w_sem=PROBE_W_SEM, embed_top_n=16) as m:
        union_top3, union_present = 0, 0
        for title, _body, query in ZERO_OVERLAP_TARGETS:
            out = m.retrieve(query, None)
            if is_hit(title, [c["title"] for c in out[:3]]):
                union_top3 += 1
            if is_hit(title, [c["title"] for c in out]):
                union_present += 1
    with _open(embed_store, w_sem=PROBE_W_SEM, embed_top_n=0) as m:
        rerank_top3 = 0
        for title, _body, query in ZERO_OVERLAP_TARGETS:
            out = m.retrieve(query, None)
            # lexically-absent target recovered ONLY by union: rerank-only
            # cannot even NOMINATE it, at any rank
            assert not is_hit(title, [c["title"] for c in out]), (
                f"zero-overlap target {title!r} surfaced without the union leg")
            if is_hit(title, [c["title"] for c in out[:3]]):
                rerank_top3 += 1
    assert rerank_top3 == 0
    n = len(ZERO_OVERLAP_TARGETS)
    assert union_top3 / n > rerank_top3 / n, "no strict p@3 divergence"
    # measured 9/30 top-3 and 27/30 present at PROBE_W_SEM on the deployment
    # runtime; asserted with wide margins so a different CPU cannot flake this
    assert union_top3 >= 3, f"union recovered only {union_top3}/{n} in top-3"
    assert union_present >= 15, f"union nominated only {union_present}/{n}"


# --- Section 3: p@3 uplift vs the commit-A baseline (ADR gate 3) --------------

def _p3_by_class(mem):
    n_para = len(_PARAPHRASE_TARGETS)
    hits = [0, 0]  # [paraphrase, zero-overlap] by fixture index split
    for idx, (title, _body, query) in enumerate(TARGETS):
        top3 = [h["title"] for h in mem.recall(query, None, top_k=3)]
        if is_hit(title, top3):
            hits[0 if idx < n_para else 1] += 1
    kw = sum(1 for title, _body, query in KEYWORD_QUERIES
             if is_hit(title, [h["title"] for h in mem.recall(query, None, top_k=3)]))
    return {"overall": (hits[0] + hits[1]) / len(TARGETS),
            "paraphrase": hits[0] / n_para,
            "zero_overlap": hits[1] / len(ZERO_OVERLAP_TARGETS),
            "keyword": kw / len(KEYWORD_QUERIES)}


@EMBED
def test_uplift_gate3_fitted_default_vs_baseline(embed_store):
    """THE gate-3 assertion: fitted-default p@3 minus the commit-A baseline
    constants >= +0.10, keyword-identical class never regresses. Activates
    when WP7 lands the grid-fitted w_sem; arithmetically vacuous before."""
    if float(DEFAULTS["w_sem"]) == 0.0:
        pytest.skip("DEFAULTS['w_sem'] is the neutral 0.0 placeholder — "
                    "gate 3 activates when WP7 commits the grid-fitted value")
    with _open(embed_store) as m:  # shipped defaults, no probe overrides
        p3 = _p3_by_class(m)
    assert p3["overall"] >= BASELINE_P3_OVERALL + 0.10 - 1e-9, p3
    assert p3["keyword"] >= BASELINE_P3_KEYWORD - 1e-9, p3


@EMBED
def test_uplift_divergence_direction_probe(embed_store):
    """Mechanism probe that runs TODAY (w_sem fit pending): at a fixed probe
    weight the semantic term must move p@3 in the claimed direction — overall
    up vs baseline, zero-overlap strictly above its 0.0 lexical ceiling — with
    the keyword-identical class not regressing. Measured on the deployment
    runtime at PROBE_W_SEM: overall 0.527 (baseline 0.400), zero 0.30,
    keyword 1.000."""
    with _open(embed_store, w_sem=PROBE_W_SEM) as m:
        p3 = _p3_by_class(m)
    assert p3["zero_overlap"] > BASELINE_P3_ZERO_OVERLAP, p3
    assert p3["overall"] > BASELINE_P3_OVERALL, p3
    assert p3["keyword"] >= BASELINE_P3_KEYWORD - 1e-9, p3


# --- Section 4: encode guard + latency budgets (ADR gate 4 mirror) -------------

@EMBED
def test_encode_guard_per_text_counter(embed_store):
    """<=1 text encoded per recall across all 110 fixture queries (a
    per-candidate encode smuggled into one batch call moves the PER-TEXT
    counter by the candidate count and fails here); ==0 on an immediately
    repeated query (query-vec LRU proof); counter must MOVE overall (a dead
    counter would make both guards vacuous — scale_bench --assert mirrors
    this vacuity check)."""
    with _open(embed_store) as m:
        for _title, _body, query in TARGETS:
            c0 = _calls(m)
            m.recall(query, None, top_k=3)
            c1 = _calls(m)
            assert c1 - c0 <= 1, f"{c1 - c0} texts encoded in one recall ({query!r})"
            m.recall(query, None, top_k=3)
            assert _calls(m) == c1, f"repeated query {query!r} re-encoded"
        assert _calls(m) > 0, "per-text counter never moved: guards were vacuous"


@EMBED
def test_latency_budgets_mirror_gate4(embed_store):
    """Pytest mirror of `scale_bench.py --assert` (the authoritative gate-4
    home): p50 <= 2ms embeddings-off, p50 <= 20ms embeddings-on at scale 1000,
    cold encoder load excluded via one warmup recall."""
    def p50_ms(mem):
        lats = []
        for _title, _body, query in TARGETS:
            t0 = time.perf_counter()
            mem.recall(query, None, top_k=3)
            lats.append((time.perf_counter() - t0) * 1000)
        return statistics.median(lats)

    with _open(embed_store, embeddings="off") as m:
        assert not m.embeddings_available
        off = p50_ms(m)
    assert off <= 2.0, f"embeddings-off p50 {off:.2f}ms > 2ms"
    with _open(embed_store) as m:
        m.recall(TARGETS[0][2], None, top_k=3)  # warmup: cold load paid here
        on = p50_ms(m)
    assert on <= 20.0, f"embeddings-on p50 {on:.2f}ms > 20ms"


# --- Section 5: zero encodes on unchanged reindex (ADR gate 5) -----------------

# One distinctive word per note: single digits are dropped by mneme's token
# regex, so "Gauge 7" alone cannot discriminate — a per-note word can.
_GAUGE_WORDS = ("amber", "birch", "cedar", "delta", "ember", "fjord",
                "garnet", "hazel", "iris", "jasper", "kelp", "lotus")


def _seed_files(tmp: Path, n: int):
    """File-backed store (reindex-able): returns (mem, [note paths])."""
    assert n <= len(_GAUGE_WORDS)
    mem = Mneme(tmp / "m.db", tmp / "notes",
                config={"embed_model_dir": str(MODELS_DIR)})
    paths = [mem.add_note("fact", f"Gauge {i} calibration note",
                          f"The gauge {i} calibrates against the "
                          f"{_GAUGE_WORDS[i]} reference cell unique-{i}.")
             for i in range(n)]
    return mem, paths


@EMBED
def test_zero_encodes_on_unchanged_touched_and_forced_reindex(tmp_path):
    mem, paths = _seed_files(tmp_path, 12)
    try:
        assert _calls(mem) == 12  # one encode per add_note, at write time
        res = mem.reindex()
        assert res["embedded"] == 0 and res["embeddings"] is True
        assert _calls(mem) == 12, "warm reindex encoded"
        # mtime touch: row re-upserts, content hash matches -> vector kept
        os.utime(paths[3], (SEED_TS, SEED_TS))
        res = mem.reindex()
        assert res["updated"] >= 1 and res["embedded"] == 0
        assert _calls(mem) == 12, "mtime-touched reindex encoded"
        res = mem.reindex(force=True)
        assert res["embedded"] == 0
        assert _calls(mem) == 12, "reindex(force=True) encoded"
        # one REAL edit -> exactly one encode
        p = paths[5]
        p.write_text(p.read_text(encoding="utf-8").replace(
            "unique-5", "unique-5-edited"), encoding="utf-8")
        os.utime(p, (SEED_TS + 60, SEED_TS + 60))
        res = mem.reindex()
        assert res["embedded"] == 1
        assert _calls(mem) == 13, "edit must cost exactly one encode"
    finally:
        mem.close()


@EMBED
def test_zero_encodes_heal_after_edit_while_embeddings_off(tmp_path):
    """The judged wheel-absent-writer scenario: a note edited while embeddings
    are OFF must still invalidate its vector (unconditional upsert hash-delete)
    so the next embeddings-ON reindex re-encodes exactly that note — even
    though the file itself is unchanged since the off-writer's reindex (the
    COUNT fast path must see the gap)."""
    mem, paths = _seed_files(tmp_path, 10)
    mem.close()
    off = Mneme(tmp_path / "m.db", tmp_path / "notes",
                config={"embeddings": "off", "embed_model_dir": str(MODELS_DIR)})
    try:
        assert off.embeddings_available is False
        p = paths[4]
        p.write_text(p.read_text(encoding="utf-8").replace(
            "unique-4", "unique-4-rewritten"), encoding="utf-8")
        os.utime(p, (SEED_TS + 60, SEED_TS + 60))
        res = off.reindex()
        assert res["embedded"] == 0 and res["embeddings"] is False
        with off._lock:
            left = off._conn.execute(
                "SELECT COUNT(*) FROM mem_vec").fetchone()[0]
        assert left == 9, "stale vector survived an embeddings-off edit"
    finally:
        off.close()
    m2 = Mneme(tmp_path / "m.db", tmp_path / "notes",
               config={"embed_model_dir": str(MODELS_DIR)})
    try:
        assert _calls(m2) == 0
        res = m2.reindex()  # files unchanged since off's reindex: dirty=False
        assert res["embedded"] == 1, "heal pass must re-encode the edited note"
        assert _calls(m2) == 1, "exactly the edited note re-encodes"
    finally:
        m2.close()


@EMBED
def test_sync_fast_path_not_masked_by_orphan_vectors(tmp_path):
    """Adversarial-verify finding (2026-07-13): an orphan vector (left by
    supersession, or by a pre-ADR writer in a mixed-version window) must not
    mask a live row that lacks a vector. With a raw mem_vec count the totals
    tie (live=10, vectors=9+1 orphan), the fast path reports embedded=0
    forever, and stats coverage claims 1.0 in exactly the state where the
    drift-visibility promise matters. The live-row join unmasks both."""
    mem, _paths = _seed_files(tmp_path, 9)
    try:
        assert _calls(mem) == 9
    finally:
        mem.close()
    off = Mneme(tmp_path / "m.db", tmp_path / "notes",
                config={"embeddings": "off", "embed_model_dir": str(MODELS_DIR)})
    try:
        # Orphan: invalidate one row the way supersession does — mem row
        # goes invalid, its vector stays behind until a sweep.
        with off._lock:
            victim = off._conn.execute(
                "SELECT dedupe_key FROM mem WHERE invalid_at IS NULL"
                " LIMIT 1").fetchone()[0]
            off._conn.execute(
                "UPDATE mem SET invalid_at=? WHERE dedupe_key=?",
                (time.time(), victim))
            off._conn.commit()
        # Missing: a note authored while embeddings are off — live row,
        # no vector. Totals now tie: 9 live rows, 9 vectors (8 live + 1 orphan).
        off.add_note("fact", "Gauge 11 calibration note",
                     f"The gauge 11 calibrates against the "
                     f"{_GAUGE_WORDS[11]} reference cell unique-11.")
    finally:
        off.close()
    m2 = Mneme(tmp_path / "m.db", tmp_path / "notes",
               config={"embed_model_dir": str(MODELS_DIR)})
    try:
        assert _calls(m2) == 0
        res = m2.reindex()  # files unchanged: COUNT fast path
        assert res["embedded"] == 1, \
            "orphan vector masked the missing one — fast path unsound"
        assert _calls(m2) == 1
        emb = m2.stats()["embeddings"]
        assert emb["vectors"] == 9, "orphan still counted after sweep"
        assert emb["coverage"] == 1.0
        with m2._lock:
            assert m2._conn.execute(
                "SELECT COUNT(*) FROM mem_vec WHERE dedupe_key=?",
                (victim,)).fetchone()[0] == 0, "orphan vector not swept"
    finally:
        m2.close()


@EMBED
def test_zero_encodes_disposability_delete_db_rebuilds_vectors(tmp_path):
    """Delete-the-db roundtrip: mem_vec is as disposable as the rest of the
    index — vectors rebuild from canon files alone."""
    mem, _paths = _seed_files(tmp_path, 10)
    mem.close()
    for suffix in ("", "-wal", "-shm"):
        f = Path(str(tmp_path / "m.db") + suffix)
        if f.exists():
            f.unlink()
    m2 = Mneme(tmp_path / "m.db", tmp_path / "notes",
               config={"embed_model_dir": str(MODELS_DIR)})
    try:
        res = m2.reindex()
        assert res["added"] == 10
        assert res["embedded"] == 10, "vectors must rebuild from files"
        st = m2.stats()["embeddings"]
        assert st["vectors"] == 10 and st["coverage"] == 1.0
        out = m2.retrieve("jasper reference cell", None)  # word of note 9
        assert out and out[0]["title"] == "Gauge 9 calibration note"
    finally:
        m2.close()


# --- Section 6: embeddings-off byte-identity (ADR gate 6) ----------------------

def test_golden_off_path_parity():
    """Replay the golden off-path pin captured from the PRE-ADR engine at
    commit A: same deterministic corpus, exact (key, title, score-to-9-places)
    equality for every pinned recall, exact index_block text. The engine
    sha256 provenance field is the only excluded key — it names the file that
    changed; everything BEHAVIORAL must not have."""
    golden_path = REPO / "bench" / "golden_off_path.json"
    want = json.loads(golden_path.read_text(encoding="utf-8"))
    import capture_golden
    try:
        got = json.loads(capture_golden.capture())
    except SystemExit as e:  # FTS5-less build: the pin targets the FTS path
        pytest.skip(str(e))
    got.pop("engine_mneme_sha256", None)
    want.pop("engine_mneme_sha256", None)
    assert got == want, "embeddings-off read path diverged from the pre-ADR pin"


def test_import_block_full_exercise_keeps_optional_wheels_out():
    """Strictest gate-6 import hygiene, order-independent via a subprocess:
    a sys.meta_path blocker makes onnxruntime AND numpy unimportable, then a
    FULL seed/recall/index_block/explain/reindex exercise must succeed
    lexical-only (with model files present the availability probe starts True
    and the first encode attempt degrades it — the wheel-absent-host case),
    and neither module may appear in sys.modules afterwards."""
    models = str(MODELS_DIR) if HAVE_MODELS else ""
    script = f'''
import sys
BLOCKED = {{"onnxruntime", "numpy"}}
class _Blocker:
    def find_spec(self, name, path=None, target=None):
        if name.split(".")[0] in BLOCKED:
            raise ImportError("blocked optional wheel: " + name)
        return None
sys.meta_path.insert(0, _Blocker())
import shutil, tempfile
from pathlib import Path
sys.path.insert(0, {str(REPO)!r})
import mneme
tmp = Path(tempfile.mkdtemp(prefix="mneme-impblock-"))
try:
    models = {models!r}
    cfg = {{"embed_model_dir": models}} if models else None
    mem = mneme.Mneme(tmp / "m.db", tmp / "notes", config=cfg)
    had_files = bool(models)
    assert mem.embeddings_available == had_files, "files-only probe broke"
    for i in range(8):
        mem.add_note("fact", f"Blocked wheel exercise note {{i}}",
                     f"body text {{i}} alpha")
    assert mem.embeddings_available is False, "degrade-to-lexical never happened"
    hits = mem.recall("alpha exercise note", None, top_k=3)
    assert hits, "recall returned nothing"
    assert all("sem" not in h for h in hits), "sem key on the off path"
    ib = mem.index_block(None, "alpha exercise note")
    assert ib.text and ib.keys, "index_block failed"
    exp = mem.explain_recall("alpha exercise note")
    assert exp and all("semantic" not in e for e in exp), "semantic key off-path"
    r = mem.reindex()
    assert r["embedded"] == 0 and r["embeddings"] is False
    st = mem.stats()["embeddings"]
    assert st["available"] is False and st["vectors"] == 0
    assert "onnxruntime" not in sys.modules, "onnxruntime entered sys.modules"
    assert "numpy" not in sys.modules, "numpy entered sys.modules"
    mem.close()
    print("IMPORT_BLOCK_OK")
finally:
    shutil.rmtree(tmp, ignore_errors=True)
'''
    proc = subprocess.run([sys.executable, "-c", script],
                          capture_output=True, text=True, timeout=300)
    assert proc.returncode == 0, f"stdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
    assert "IMPORT_BLOCK_OK" in proc.stdout


def test_off_path_dicts_carry_no_sem_or_semantic_keys(tmp_path):
    """Field-identity on the lexical path: candidate dicts and explain_recall
    entries gain NOTHING when embeddings are unavailable, and the sem weight
    is IEEE-exact 0.0."""
    mem = Mneme(tmp_path / "m.db", tmp_path / "notes")  # no models next to db
    try:
        assert mem.embeddings_available is False
        _seed_direct(mem, [(f"Topic {i} handbook entry",
                            f"handbook body {i} about topic {i}")
                           for i in range(6)])
        out = mem.retrieve("topic handbook", None)
        assert out, "lexical retrieve found nothing"
        assert all("sem" not in c for c in out)
        exp = mem.explain_recall("topic handbook")
        assert exp and all("semantic" not in e for e in exp)
        if mem.fts_available:
            assert mem._weights() == (LEX_WEIGHT, JAC_WEIGHT, 0.0)
        st = mem.stats()["embeddings"]
        assert st["available"] is False and st["vectors"] == 0
        assert st["coverage"] == 0.0 or st["coverage"] == 0
        assert st["encoder"] == ENCODER_ID
    finally:
        mem.close()


def _zebra_store(tmp_path):
    """Cooc-rescue construction: 'zebra' matches ZERO rows in bank 'global'
    (its only occurrence lives in bank 'projx'), but the cooc table — built
    across banks — pairs it with 'yak', which DOES match a global row."""
    mem = Mneme(tmp_path / "m.db", tmp_path / "notes")
    with mem._lock:
        mem._upsert_locked("file:/t/pair.md", "fact", "projx",
                           "zebra yak", "zebra yak", "", "",
                           "s", False, SEED_TS, SEED_TS)
        mem._upsert_locked("file:/t/global.md", "fact", "global",
                           "yak care", "brush the yak", "", "",
                           "s", False, SEED_TS, SEED_TS)
        mem._conn.commit()
    return mem


def test_cooc_rescue_still_fires_when_qvec_is_none(tmp_path):
    """The empty-result cooc rescue must survive the ADR restructure on the
    lexical path (its gate is 'if not rows and qvec is None')."""
    mem = _zebra_store(tmp_path)
    try:
        if not mem.fts_available:
            pytest.skip("cooc rescue is an FTS-path feature")
        out = mem._candidates("zebra", ("global",))
        assert [c["title"] for c in out] == ["yak care"], (
            "cooc rescue did not fire on the qvec=None path")
    finally:
        mem.close()


@pytest.mark.skipif(not HAVE_NUMPY, reason="needs numpy to fabricate a qvec")
def test_cooc_rescue_gated_off_on_qvec_path(tmp_path):
    """With a query vector present the union leg IS the rescue: the cooc
    expansion must not run (stacking both would dilute cosine-ranked
    additions — settled in the ADR), and an empty lexical+vector result
    returns [] cleanly."""
    import numpy as np
    mem = _zebra_store(tmp_path)
    try:
        if not mem.fts_available:
            pytest.skip("cooc rescue is an FTS-path feature")
        called = []
        mem._expand_tokens = lambda toks: called.append(toks) or toks
        qvec = np.zeros(4, dtype="<f4")  # mem_vec is empty: union adds nothing
        out = mem._candidates("zebra", ("global",), qvec=qvec)
        assert out == [] and not called, (
            "cooc rescue ran on the qvec path (or ghost candidates appeared)")
    finally:
        mem.close()


# --- Tier sort: cosine can never launder repo notes past canon -----------------

@EMBED
def test_tier_sort_high_cosine_repo_note_never_outranks_operator_canon(tmp_path):
    """A repo-credibility note engineered to win the raw blended score (exact
    lexical AND near-perfect cosine match on the query) must still sort after
    every operator row — the credibility-tier-first sort is structural, not a
    down-weight hope. The test self-checks its premise: if the repo row does
    NOT out-score the operator rows, the construction went soft and the
    assertion would be vacuous."""
    query = "database migration rollback procedure"
    with _open(tmp_path, w_sem=0.7) as mem:
        _seed_direct(mem, [
            ("Catering order procedure alpha",
             "weekly lunch orders for the office kitchen"),
            ("Plant watering procedure beta",
             "indoor greenery watering rota for the office"),
        ], credibility="operator", prefix="op")
        _seed_direct(mem, [
            (query, "how to roll back a database migration safely"),
        ], credibility="repo", prefix="repo")
        assert _drain(mem) == 3
        out = mem.retrieve(query, None)
        ops = [c for c in out if c["credibility"] == "operator"]
        repos = [c for c in out if c["credibility"] == "repo"]
        assert ops and repos, f"expected both tiers in output, got {out}"
        assert max(r["score"] for r in repos) > max(o["score"] for o in ops), (
            "premise went soft: repo row no longer wins the raw score, "
            "re-engineer the fixture")
        first_repo = next(i for i, c in enumerate(out)
                          if c["credibility"] == "repo")
        assert all(c["credibility"] == "operator"
                   for c in out[:first_repo]), out
        assert out[0]["credibility"] == "operator"


# --- retrieve vs explain_recall consistency -------------------------------------

@EMBED
def test_retrieve_explain_consistency_with_embeddings(embed_store):
    """explain_recall's shown components come from the SAME _score_parts call
    retrieve() ranks with: per-title rounded scores must match exactly, and
    every on-path explain entry carries the 'semantic' component."""
    with _open(embed_store, w_sem=PROBE_W_SEM) as m:
        query = TARGETS[2][2]
        exp = m.explain_recall(query, None, top_k=5)
        assert exp
        ret = m.retrieve(query, None, limit=32)  # same candidate pool width
        by_title = {c["title"]: c for c in ret}
        for e in exp:
            assert "semantic" in e, "on-path explain entry lacks 'semantic'"
            assert e["title"] in by_title
            assert e["score"] == round(by_title[e["title"]]["score"], 4), (
                f"explain score drifted from retrieve for {e['title']!r}")


@EMBED
def test_stats_embeddings_block_reports_coverage(embed_store):
    with _open(embed_store) as m:
        st = m.stats()["embeddings"]
        assert st["available"] is True
        assert st["encoder"] == ENCODER_ID
        assert st["vectors"] == EMBED_SCALE
        assert st["coverage"] == 1.0
        assert st["w_sem"] == float(DEFAULTS["w_sem"])
