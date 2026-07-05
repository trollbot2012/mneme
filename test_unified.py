"""Unified agent-facing API: the 10 verbs an agent calls (one layer, one API)."""
from pathlib import Path

from mneme import Mneme


def mk(tmp):
    return Mneme(Path(tmp) / "m.db", Path(tmp) / "notes")


def test_remember_and_recall_roundtrip(tmp_path):
    m = mk(tmp_path)
    m.remember("Use tabs in Go files", "The formatter enforces tabs.",
               kind="preference", keywords="golang indentation")
    hits = m.recall("indentation style for go")
    assert hits and "tabs in Go" in hits[0]["title"]


def test_record_served_then_outcome_moves_trust(tmp_path):
    m = mk(tmp_path)
    m.remember("Route probe before trusting a provider", "health != working")
    key = m.recall("route probe")[0]["key"]
    m.record_served("run1", [key], "index")
    assert m.record_outcome("run1", "done") == 1
    # 0.5 -> 0.667 on one positive, exactly as the formula predicts
    assert abs(m.recall("route probe")[0]["trust"] - (2 / 3)) < 1e-9


def test_audit_reports_graveyard_and_trust(tmp_path):
    m = mk(tmp_path)
    m.remember("Served note", "x")
    m.remember("Never served note", "y")
    key = m.recall("served note")[0]["key"]
    m.record_served("r", [key], "index")
    m.record_outcome("r", "done")
    a = m.audit()
    assert a["rows"] == 2 and 0.0 <= a["graveyard_ratio"] <= 1.0
    assert a["trust_evaluated"] == 1 and a["trust_mean"] is not None


def test_explain_recall_exposes_components(tmp_path):
    m = mk(tmp_path)
    m.remember("Postgres pool exhausts under load", "100 concurrent workers")
    rows = m.explain_recall("postgres pool exhausts under concurrent load")
    assert rows
    top = rows[0]
    for field in ("lexical", "jaccard", "trust", "decay", "score"):
        assert field in top and isinstance(top[field], (int, float))


def test_summarize_user_model_from_preferences(tmp_path):
    m = mk(tmp_path)
    m.remember("Prefer stdlib over dependencies", "portable bundle breaks otherwise",
               kind="preference")
    m.remember("Always work on branches", "git is the rollback", kind="preference")
    m.remember("Tests need live redis", "not a preference", kind="fact")
    model = m.summarize_user_model()
    assert "Operator model" in model
    assert "stdlib" in model and "branches" in model
    assert "redis" not in model  # facts are not the user model


def test_export_import_roundtrip(tmp_path):
    m = mk(tmp_path)
    m.remember("Nightly backup at 03:15", "writes to the NAS", kind="fact",
               keywords="cron backup")
    m.remember("Prefer four-space python", "pep8", kind="preference")
    snapshot = m.export_memory()
    assert "## facts" in snapshot and "## preferences" in snapshot
    assert "Nightly backup at 03:15" in snapshot

    tmp2 = tmp_path / "second"
    tmp2.mkdir()
    m2 = mk(tmp2)
    n = m2.import_memory(snapshot)
    assert n == 2
    assert m2.recall("nightly backup nas")[0]["title"].startswith("Nightly backup")
    # ADD-only: re-import writes nothing new
    assert m2.import_memory(snapshot) == 0


def test_export_to_file_is_readable_markdown(tmp_path):
    m = mk(tmp_path)
    m.remember("Human readable", "and portable", kind="lesson")
    out = tmp_path / "export.md"
    m.export_memory(out)
    text = out.read_text(encoding="utf-8")
    assert text.startswith("# Mneme memory export") and "### Human readable" in text
