"""Regression tests for production memory gaps found in the Hermes adapter."""

import sqlite3

import mneme as mneme_module
from mneme import Mneme


def make_store(tmp_path, **config):
    return Mneme(tmp_path / "mneme.db", tmp_path / "notes", config=config)


def test_embedding_failure_is_visible_in_health(tmp_path, monkeypatch):
    model_dir = tmp_path / "models"
    model_dir.mkdir()
    for name in mneme_module.EMBED_MODEL_FILES:
        (model_dir / name).write_bytes(b"configured")

    class BrokenEncoder:
        def __init__(self, _model_dir):
            raise ImportError("onnx DLL missing")

    monkeypatch.setattr(mneme_module, "_MiniLMEncoder", BrokenEncoder)
    mem = make_store(tmp_path, embed_model_dir=str(model_dir))
    assert mem.embeddings_available is True

    assert mem._query_vec("force encoder load") is None
    health = mem.stats()["embeddings"]
    assert health["configured"] is True
    assert health["available"] is False
    assert "onnx DLL missing" in health["error"]
    assert health["model_dir"] == str(model_dir)


def test_checkpoint_episode_can_be_finalized(tmp_path):
    mem = make_store(tmp_path)
    mem.add_episode("run-1", "initial goal", "session_open", "first serve", None)

    assert mem.finalize_episode(
        "run-1",
        "initial goal",
        "session_end",
        "Decisions: use cobalt deployment.\nResult: verification passed.",
        None,
    ) is True

    with sqlite3.connect(mem.db_path) as conn:
        row = conn.execute(
            "SELECT title, body, tags FROM mem WHERE dedupe_key='run:run-1'"
        ).fetchone()
    assert row == (
        "[session_end] initial goal",
        "Decisions: use cobalt deployment.\nResult: verification passed.",
        "session_end",
    )
    assert mem.recall("cobalt deployment verification")[0]["kind"] == "episode"


def test_conflict_candidates_are_scoped_and_ignore_superseded_notes(tmp_path):
    mem = make_store(tmp_path)
    mem.add_note("preference", "Worker priority order", "Use Claude first.")
    mem.add_note(
        "preference",
        "Worker priority order updated",
        "Use Fable first.",
    )

    conflicts = mem.find_conflicts()
    assert any(
        {pair["left"]["title"], pair["right"]["title"]}
        == {"Worker priority order", "Worker priority order updated"}
        for pair in conflicts
    )

    mem.add_note(
        "preference",
        "Worker dispatch priority",
        "Use Fable first.",
        supersedes="worker-priority-order,worker-priority-order-updated",
    )
    active_titles = {
        side["title"]
        for pair in mem.find_conflicts()
        for side in (pair["left"], pair["right"])
    }
    assert "Worker priority order" not in active_titles
    assert "Worker priority order updated" not in active_titles


def test_stats_exposes_outcome_and_trust_health(tmp_path):
    mem = make_store(tmp_path)
    mem.add_note("lesson", "Probe before patching", "Reproduce the failure first.")
    key = mem.recall("probe patching")[0]["key"]
    mem.record_served("good", [key], "recall")
    mem.apply_outcome("good", "done", used_keys=[key])
    mem.record_served("bad", [key], "recall")
    mem.apply_outcome("bad", "rolled_back", used_keys=[key])

    health = mem.stats()["evidence"]
    assert health["outcomes"] == {"done": 1, "rolled_back": 1}
    assert health["trust"]["evaluated"] == 1
    assert health["trust"]["unevaluated"] == 0


def test_finalize_is_monotonic_on_richness(tmp_path):
    mem = make_store(tmp_path)
    mem.add_episode("run-m", "ship cobalt", "session_open",
                    "Checkpointed at first serve: x on cli", None)
    rich = (
        "Session: s1\nPlatform: cli\nTurns: 2\n"
        "User development: ship cobalt to staging\n"
        "Final response/result: deployed and verified green"
    )
    mem.finalize_episode("run-m", "ship cobalt", "session_progress", rich, None)
    mem.finalize_episode(
        "run-m", "ship cobalt", "session_end",
        "Session s1 on cli, 0 turns", None,
    )
    with sqlite3.connect(mem.db_path) as conn:
        body, tags = conn.execute(
            "SELECT body, tags FROM mem WHERE dedupe_key='run:run-m'"
        ).fetchone()
    assert "deployed and verified green" in body
    assert tags == "session_end"


def test_thin_checkpoint_episodes_are_demoted_in_recall(tmp_path):
    mem = make_store(tmp_path)
    mem.add_episode(
        "thin", "cobalt deploy", "session_open",
        "Checkpointed at first serve: thin on cli", None,
    )
    mem.add_episode(
        "rich", "cobalt deploy", "session_end",
        "Session: r\nPlatform: cli\nTurns: 3\n"
        "User development: fix cobalt deploy auth\n"
        "Final response/result: cobalt deploy auth fixed and verified",
        None,
    )
    hits = mem.recall("cobalt deploy auth verification", top_k=5)
    assert hits
    assert hits[0]["key"] == "run:rich"
    assert mem.stats()["episodes"]["thin"] >= 1
    assert mem.stats()["episodes"]["rich"] >= 1


def test_repair_removes_only_legacy_no_use_failure_debits(tmp_path):
    mem = make_store(tmp_path)
    mem.add_note("lesson", "Alpha guidance", "Verify alpha before acting.")
    mem.add_note("lesson", "Beta guidance", "Verify beta before acting.")
    hits = mem.recall("guidance")
    keys = [hit["key"] for hit in hits]

    mem.record_served("legacy", keys, "index")
    mem.apply_outcome("legacy", "rolled_back")  # historical weak bystander debit

    beta = next(hit["key"] for hit in hits if hit["title"] == "Beta guidance")
    mem.record_served("strong", keys, "index")
    mem.apply_outcome("strong", "rolled_back", used_keys=[beta])

    report = mem.repair_weak_bystander_trust()
    assert report["runs"] == 1
    assert report["notes"] == 2
    assert report["negative_points_removed"] == 0.2

    with sqlite3.connect(mem.db_path) as conn:
        counters = dict(conn.execute(
            "SELECT m.title, s.negative FROM mem m JOIN mem_stats s"
            " ON s.note_id=m.note_id WHERE m.kind='lesson'"
        ).fetchall())
    assert counters == {"Alpha guidance": 0.0, "Beta guidance": 1.0}
    assert mem.repair_weak_bystander_trust()["already_applied"] is True
