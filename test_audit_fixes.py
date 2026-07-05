"""Regression tests for the 2026-07 adversarial audit findings. Each test is
named for the finding it locks down; each fails against the pre-fix code."""
import sqlite3

from mneme import Mneme


def mk(tmp, **cfg):
    return Mneme(tmp / "m.db", tmp / "notes", config=cfg)


# -- C1: cross-bank supersession isolation ------------------------------------

def test_supersession_cannot_cross_banks(tmp_path):
    mem = mk(tmp_path)
    mem.add_note("fact", "Retry policy", "global truth")           # global bank
    repo = tmp_path / "repoA"
    (repo / ".ktisis" / "memory").mkdir(parents=True)
    (repo / ".ktisis" / "memory" / "retry-policy.md").write_text(
        "---\nkind: fact\nsupersedes: retry-policy\n---\n# Retry policy\nrepo truth\n",
        encoding="utf-8")
    mem.reindex(repo)
    # the repo note must NOT have invalidated the global note
    glob_hits = [h for h in mem.retrieve("retry policy") if h["bank"] == "global"]
    assert glob_hits and "global truth" in glob_hits[0]["body"], "global note was cross-bank superseded"
    with sqlite3.connect(str(mem.db_path)) as c:
        inv = c.execute("SELECT invalid_at FROM mem WHERE bank='global' AND title='Retry policy'").fetchone()
    assert inv[0] is None


def test_supersession_still_works_within_a_bank(tmp_path):
    mem = mk(tmp_path)
    mem.add_note("preference", "Indent style", "four spaces")
    mem.add_note("preference", "Indent style tabs", "tabs now", supersedes="indent-style")
    text = mem.index_block(None, "indent style").text
    assert "tabs now" in text and "four spaces" not in text


# -- C4: reindex prune LIKE metacharacter over-match --------------------------

def test_reindex_prune_underscore_does_not_evict_sibling_repo(tmp_path):
    mem = mk(tmp_path)
    a = tmp_path / "rax"; b = tmp_path / "r_x"          # 'r_x' LIKE-matches 'rax'
    for r in (a, b):
        (r / ".ktisis" / "memory").mkdir(parents=True)
        (r / ".ktisis" / "memory" / "note.md").write_text(
            f"---\nkind: fact\n---\n# note in {r.name}\nbody {r.name}\n", encoding="utf-8")
    mem.reindex(a)
    mem.reindex(b)  # b's prune prefix 'r_x/...' must NOT match a's 'rax/...' key
    assert any("rax" in h["title"] for h in mem.retrieve("note", a)), "sibling repo's note was wrongly pruned"


# -- valid_at overflow must not crash reindex ---------------------------------

def test_far_future_valid_at_does_not_crash_reindex(tmp_path):
    mem = mk(tmp_path)
    (mem.global_dir / "lessons").mkdir(parents=True, exist_ok=True)
    (mem.global_dir / "lessons" / "bad-date.md").write_text(
        "---\nkind: lesson\nvalid_at: 9999-12-31\n---\n# Bad date lesson\nsurvives\n", encoding="utf-8")
    r = mem.reindex()  # must not raise OverflowError/OSError from time.mktime
    assert any("Bad date" in h["title"] for h in mem.retrieve("bad date lesson"))


# -- C2/C3: export/import round-trip ------------------------------------------

def test_import_preserves_body_with_markdown_headers(tmp_path):
    mem = mk(tmp_path)
    body = "First line.\n## A heading inside the body\n### Another\nLast line."
    mem.add_note("lesson", "Body with headers", body)
    snap = mem.export_memory()
    mem2 = mk(tmp_path / "second")
    assert mem2.import_memory(snap) == 1
    got = [h for h in mem2.retrieve("body with headers")][0]
    # the header-looking body lines survived (not parsed as structure)
    assert "A heading inside the body" in got["body"] and "Last line." in got["body"]


def test_export_import_roundtrips_metadata(tmp_path):
    mem = mk(tmp_path)
    mem.add_note("fact", "Pinned fact", "durable", keywords="alpha beta", tags="infra,prod", pinned=True)
    snap = mem.export_memory()
    mem2 = mk(tmp_path / "second")
    mem2.import_memory(snap)
    with sqlite3.connect(str(mem2.db_path)) as c:
        row = c.execute("SELECT keywords, tags, pinned FROM mem WHERE title='Pinned fact'").fetchone()
    assert row == ("alpha beta", "infra,prod", 1), f"metadata lost on round-trip: {row}"


def test_import_short_snapshot_not_misread_as_path(tmp_path):
    mem = mk(tmp_path)
    # a multi-line snapshot under any length must never be treated as a file path
    snap = "## lessons\n\n### Short note\n    body text\n"
    assert mem.import_memory(snap) == 1
    assert mem.retrieve("short note")


# -- M1: quarantine escaping + total veto -------------------------------------

def test_quarantine_underscore_slug_hits_right_note(tmp_path):
    mem = mk(tmp_path)
    mem.add_note("lesson", "db pool tuning", "correct one")   # slug: db-pool-tuning
    mem.add_note("lesson", "dbXpool tuning", "decoy")         # slug: dbxpool-tuning
    assert mem.quarantine("db-pool-tuning")
    titles = [h["title"] for h in mem.retrieve("pool tuning")]
    assert "db pool tuning" not in titles       # the intended one is vetoed
    assert "dbXpool tuning" in titles           # the decoy is NOT (underscore not a wildcard)


# -- M2: per-bank episode compaction ------------------------------------------

def test_compact_is_per_bank_not_global(tmp_path):
    mem = mk(tmp_path, max_episodes=3, auto_compact=False)
    repo = tmp_path / "busy"
    (repo / ".ktisis" / "memory").mkdir(parents=True)
    for i in range(8):  # busy bank: 8 episodes, cap 3 -> 5 evicted
        mem.add_episode(f"busy{i}", f"busy goal {i}", "done", "d", repo)
    mem.add_episode("quiet0", "quiet goal", "done", "d", None)  # global bank: 1 episode
    mem.compact()
    with sqlite3.connect(str(mem.db_path)) as c:
        glob = c.execute("SELECT COUNT(*) FROM mem WHERE kind='episode' AND bank='global'").fetchone()[0]
    assert glob == 1, "quiet bank's episode was evicted by the busy bank's overflow"


def test_compact_conserves_stats_in_archive(tmp_path):
    mem = mk(tmp_path, max_episodes=1, auto_compact=False)
    mem.add_episode("r0", "goal zero", "done", "d", None)
    mem.add_episode("r1", "goal one", "done", "d", None)
    mem.record_served("x", ["run:r0"], "index")  # give r0 a served count
    r = mem.compact()
    assert r["archived"] >= 1
    with sqlite3.connect(r["archive"]) as arc:
        tables = {t[0] for t in arc.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    assert "mem_stats" in tables  # archive conserves the stats table


# -- M3: apply_outcome idempotency --------------------------------------------

def test_apply_outcome_is_idempotent_per_run(tmp_path):
    mem = mk(tmp_path)
    mem.add_note("lesson", "Trusted lesson", "x")
    key = mem.retrieve("trusted lesson")[0]["key"]
    mem.record_served("run7", [key], "index")
    assert mem.apply_outcome("run7", "done", used_keys=[key]) == 1
    assert mem.apply_outcome("run7", "done", used_keys=[key]) == 0        # repeat is a no-op
    assert mem.apply_outcome("run7", "rolled_back", used_keys=[key]) == 0  # even w/ new status
    # trust moved exactly once: 0.5 -> 0.667, not further
    assert abs(mem.retrieve("trusted lesson")[0]["trust"] - (2 / 3)) < 1e-9


# -- M4: graveyard_ratio denominator ------------------------------------------

def test_graveyard_ratio_excludes_episodes_and_superseded(tmp_path):
    mem = mk(tmp_path)
    mem.add_note("lesson", "Old rule", "v1")
    mem.add_note("lesson", "New rule", "v2", supersedes="old-rule")  # invalidates Old rule
    mem.add_episode("e0", "some run", "done", "d", None)              # episode
    key = mem.retrieve("new rule")[0]["key"]
    mem.record_served("r", [key], "index")
    a = mem.audit()
    # only 1 live non-episode note (New rule), and it WAS served -> graveyard 0.0
    assert a["live_notes"] == 1 and a["graveyard_ratio"] == 0.0


# -- L1: quarantined top row must not hide a lower-ranked legit row -----------

def test_quarantine_does_not_shrink_visible_set(tmp_path):
    mem = mk(tmp_path)
    # many matches; quarantine the would-be top hit, ensure others still surface
    for i in range(20):
        mem.add_note("lesson", f"widget lesson {i}", "widget " * (20 - i))
    top_key = mem.retrieve("widget", limit=1)[0]["key"]
    slug = top_key.rsplit("/", 1)[-1].removesuffix(".md")
    mem.quarantine(slug)
    hits = mem.retrieve("widget", limit=3)
    assert hits and all(h["key"] != top_key for h in hits)  # quarantined gone, set not empty


# -- bank isolation on the retrieve/recall path (not just index_block) --------

def test_retrieve_respects_bank_scoping(tmp_path):
    mem = mk(tmp_path)
    repo = tmp_path / "proj"
    (repo / ".ktisis" / "memory").mkdir(parents=True)
    (repo / ".ktisis" / "memory" / "secret.md").write_text(
        "---\nkind: fact\n---\n# Project secret\nrepo only\n", encoding="utf-8")
    mem.reindex(repo)
    assert not any("Project secret" in h["title"] for h in mem.retrieve("project secret"))       # global view
    assert any("Project secret" in h["title"] for h in mem.retrieve("project secret", repo))     # repo view


# -- FTS-absent fallback parity -----------------------------------------------

def test_like_fallback_paths_work_without_fts(tmp_path):
    mem = mk(tmp_path)
    mem.fts_available = False
    mem.add_note("lesson", "Windows path slashes in git", "backslashes break")
    mem.add_note("lesson", "Decoy note", "unrelated")
    hits = mem.retrieve("windows git path")
    assert hits and "Windows path" in hits[0]["title"]
    # quarantine still works on the LIKE path
    key = hits[0]["key"]
    mem.quarantine(key.rsplit("/", 1)[-1].removesuffix(".md"))
    assert not any("Windows path" in h["title"] for h in mem.retrieve("windows git path"))


# -- add_note ADD-only mode ---------------------------------------------------

def test_overwrite_false_returns_none_for_existing(tmp_path):
    mem = mk(tmp_path)
    p1 = mem.add_note("lesson", "Once", "first")
    assert p1 is not None
    assert mem.add_note("lesson", "Once", "second", overwrite=False) is None
    assert "first" in (mem.global_dir / "lessons" / "once.md").read_text(encoding="utf-8")


# -- index_block never exceeds budget; served keys == emitted lines -----------

def test_index_block_within_budget_and_keys_match_text(tmp_path):
    mem = mk(tmp_path, index_budget_chars=600)
    for kind in ("lesson", "fact", "preference"):
        for i in range(15):
            mem.add_note(kind, f"{kind} item number {i} about widgets", "body " * 10)
    ib = mem.index_block(None, "widgets item", run_id="runB")
    assert len(ib.text) <= 600
    # every served key's title must actually appear in the emitted text
    import sqlite3 as _s
    with _s.connect(str(mem.db_path)) as c:
        for k in ib.keys:
            title = c.execute("SELECT title FROM mem WHERE dedupe_key=?", (k,)).fetchone()[0]
            assert title[:40] in ib.text, f"served key {k} not in emitted block"
