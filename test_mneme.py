"""Mneme store: files-as-canon indexing, retrieval, supersession, trust,
banks, budget, fallback, episodes, compaction. No LLM, no network."""

import sqlite3

import pytest

from mneme import Mneme
from mneme import parse_note, render_note


def mk(tmp_path, **cfg):
    gdir = tmp_path / "mneme"
    for k in ("lessons", "facts", "preferences"):
        (gdir / k).mkdir(parents=True, exist_ok=True)
    return Mneme(tmp_path / "mneme.db", gdir, config=cfg)


def note(mem, kind, title, body="", **kw):
    return mem.add_note(kind, title, body, **kw)


# -- notes.py -----------------------------------------------------------------

def test_note_parse_frontmatter_and_legacy(tmp_path):
    f = tmp_path / "a.md"
    f.write_text(render_note("preference", "Prefer tabs", "Always.", tags="style",
                             pinned=True), encoding="utf-8")
    n = parse_note(f)
    assert n.kind == "preference" and n.title == "Prefer tabs" and n.pinned
    assert n.tags == "style" and "Always." in n.body
    # legacy lesson file (no frontmatter): kind from directory name
    lessons = tmp_path / "lessons"
    lessons.mkdir()
    g = lessons / "old-lesson.md"
    g.write_text("# Old lesson summary\nDetails here.\n", encoding="utf-8")
    m = parse_note(g)
    assert m.kind == "lesson" and m.title == "Old lesson summary" and m.body == "Details here."


# -- indexing + retrieval -------------------------------------------------------

def test_reindex_and_porter_stemmed_retrieval(tmp_path):
    mem = mk(tmp_path)
    note(mem, "lesson", "Route health probes before trusting a provider",
         "A route that answers /health may still return empty completions.")
    hits = mem.retrieve("routing probe trust")
    assert hits and "Route health probes" in hits[0]["title"]  # routing->rout matches route


def test_legacy_lessons_dir_rides_as_extra_dir(tmp_path):
    legacy = tmp_path / "legacy-lessons"
    legacy.mkdir()
    (legacy / "use-branches.md").write_text("# Always work on branches\nGit is rollback.\n",
                                            encoding="utf-8")
    gdir = tmp_path / "mneme"
    (gdir / "lessons").mkdir(parents=True)
    mem = Mneme(tmp_path / "m.db", gdir, extra_dirs=[legacy])
    mem.reindex()
    assert any("branches" in h["title"] for h in mem.retrieve("branch git"))


def test_bank_isolation(tmp_path):
    mem = mk(tmp_path)
    repo_a = tmp_path / "repo_a"
    repo_b = tmp_path / "repo_b"
    for r in (repo_a, repo_b):
        (r / ".ktisis" / "memory").mkdir(parents=True)
    (repo_a / ".ktisis" / "memory" / "fact.md").write_text(
        "---\nkind: fact\n---\n# Tests need the fake redis fixture\nUse conftest.\n",
        encoding="utf-8")
    mem.reindex(repo_a)
    in_a = mem.index_block(repo_a, "redis fixture tests").text
    in_b = mem.index_block(repo_b, "redis fixture tests").text
    assert "fake redis" in in_a
    assert "fake redis" not in in_b  # repo A's fact must never leak into repo B


def test_supersession_hides_old_row(tmp_path):
    mem = mk(tmp_path)
    note(mem, "preference", "Use four spaces for indentation", "Old rule.")
    note(mem, "preference", "Use tabs for indentation", "New rule.",
         supersedes="use-four-spaces-for-indentation")
    text = mem.index_block(None, "indentation").text
    assert "tabs" in text and "four spaces" not in text
    with sqlite3.connect(str(mem.db_path)) as c:
        row = c.execute("SELECT invalid_at, superseded_by FROM mem WHERE title LIKE ?",
                        ("%four spaces%",)).fetchone()
    assert row[0] is not None and row[1] is not None  # invalidated, never deleted


def test_index_budget_is_enforced(tmp_path):
    mem = mk(tmp_path, index_budget_chars=800)
    for i in range(60):
        note(mem, "lesson", f"Lesson number {i} about widget frobnication",
             "Body " * 40)
    block = mem.index_block(None, "widget frobnication lesson")
    assert 0 < len(block.text) <= 800


def test_pinned_always_present_even_off_topic(tmp_path):
    mem = mk(tmp_path)
    note(mem, "preference", "Never push to main without asking", "Hard rule.", pinned=True)
    block = mem.index_block(None, "completely unrelated cheese soufflé recipe")
    assert "Never push to main" in block.text


def test_reindex_idempotent_and_db_rebuild(tmp_path):
    mem = mk(tmp_path)
    note(mem, "lesson", "Alpha lesson", "Body a.")
    note(mem, "fact", "Beta fact", "Body b.")
    r1 = mem.reindex()
    assert r1["added"] == 0 and r1["updated"] == 0  # add_note already indexed
    text1 = mem.index_block(None, "alpha beta").text
    mem.close()
    (tmp_path / "mneme.db").unlink()  # the DB is disposable
    mem2 = Mneme(tmp_path / "mneme.db", tmp_path / "mneme")
    mem2.reindex()
    assert mem2.index_block(None, "alpha beta").text == text1


def test_pruned_when_file_deleted(tmp_path):
    mem = mk(tmp_path)
    p = note(mem, "lesson", "Ephemeral lesson", "Gone soon.")
    assert mem.retrieve("ephemeral")
    p.unlink()
    r = mem.reindex()
    assert r["pruned"] == 1 and not mem.retrieve("ephemeral")


def test_like_fallback_without_fts(tmp_path):
    mem = mk(tmp_path)
    mem.fts_available = False  # simulate an SQLite build without FTS5
    note(mem, "lesson", "Windows paths need forward slashes in git",
         "Backslashes break exclude patterns.")
    hits = mem.retrieve("windows git paths")
    assert hits and "Windows paths" in hits[0]["title"]
    assert mem.stats()["fts"] is False


def test_quarantine_excludes_everywhere(tmp_path):
    mem = mk(tmp_path)
    note(mem, "lesson", "Bad advice to be vetoed", "Wrong.", pinned=True)
    assert "Bad advice" in mem.index_block(None, "vetoed advice").text
    assert mem.quarantine("bad-advice-to-be-vetoed")
    assert "Bad advice" not in mem.index_block(None, "vetoed advice").text
    assert not any("Bad advice" in h["title"] for h in mem.retrieve("vetoed advice"))
    assert mem.quarantine("bad-advice-to-be-vetoed", on=False)
    assert "Bad advice" in mem.index_block(None, "vetoed advice").text


# -- proof-coupled trust ---------------------------------------------------------

def test_trust_sinks_serially_failing_memory(tmp_path):
    mem = mk(tmp_path)
    note(mem, "lesson", "Alpha rule one", "Fine.")
    note(mem, "lesson", "Alpha rule two", "Rides failing runs.")
    key2 = [h["key"] for h in mem.retrieve("alpha rule") if "two" in h["title"]][0]
    for i in range(3):
        run = f"run{i}"
        mem.record_served(run, [key2], "index")
        assert mem.apply_outcome(run, "rolled_back") == 1
    hits = mem.retrieve("alpha rule")
    assert "one" in hits[0]["title"]  # untouched note (trust 0.5) outranks 0.2
    assert hits[-1]["trust"] == pytest.approx((0 + 1) / (3 + 2))


def test_neutral_outcomes_do_not_move_trust(tmp_path):
    mem = mk(tmp_path)
    note(mem, "lesson", "Neutral lesson", "x")
    key = mem.retrieve("neutral lesson")[0]["key"]
    mem.record_served("r1", [key], "index")
    assert mem.apply_outcome("r1", "denied") == 0
    assert mem.apply_outcome("r1", "vacuous_checks") == 0
    assert mem.retrieve("neutral lesson")[0]["trust"] == 0.5


# -- episodes + compaction --------------------------------------------------------

def test_episodes_idempotent_and_compact_archives(tmp_path):
    mem = mk(tmp_path, max_episodes=3, compact_after_days=90)
    for i in range(5):
        mem.add_episode(f"run{i}", f"goal number {i}", "done", f"message {i}", None)
        mem.add_episode(f"run{i}", "duplicate write", "done", "ignored", None)  # idempotent
    assert mem.stats()["by_kind"]["episode"] == 5
    r = mem.compact()
    assert r["archived"] == 2
    assert mem.stats()["by_kind"]["episode"] == 3
    with sqlite3.connect(r["archive"]) as arc:
        n = arc.execute("SELECT COUNT(*) FROM mem WHERE kind='episode'").fetchone()[0]
    assert n == 2  # conserved: archived, never silently deleted


def test_stats_shape(tmp_path):
    mem = mk(tmp_path)
    note(mem, "lesson", "Something", "x")
    s = mem.stats()
    assert s["rows"] == 1 and s["never_served"] == 1 and "fts" in s
