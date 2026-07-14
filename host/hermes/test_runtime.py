"""Production-runtime tests: project scope and evidence integrity."""

from pathlib import Path

from runtime import MnemeRuntime


class FakeEngine:
    def __init__(self):
        self.calls = []
        self.used = []

    def recall(self, query, repo=None, top_k=None):
        self.calls.append(("recall", query, repo, top_k))
        return [{"key": "k1", "title": "hit"}]

    def record_served(self, run_id, keys, tier):
        self.calls.append(("served", run_id, keys, tier))

    def index_block(self, repo, query, run_id=None, budget=None):
        self.calls.append(("index", query, repo, run_id, budget))
        return "block"

    def add_note(self, kind, title, body, **kwargs):
        self.calls.append(("add", kind, title, kwargs.get("repo")))
        return Path("note.md")

    def served_keys(self, run_id, tier=None):
        return list(self.used)

    def apply_outcome(self, run_id, status, used_keys=None):
        self.calls.append(("outcome", run_id, status, used_keys))
        return len(used_keys or [])


def make_repo(tmp_path):
    repo = tmp_path / "repo"
    (repo / ".git").mkdir(parents=True)
    nested = repo / "src" / "pkg"
    nested.mkdir(parents=True)
    return repo, nested


def test_reads_use_session_aware_repository_root(tmp_path):
    repo, nested = make_repo(tmp_path)
    engine = FakeEngine()
    runtime = MnemeRuntime(engine, cwd_resolver=lambda: nested)

    runtime.recall("redis fixture", run_id="r1", top_k=4)
    runtime.index_block("redis fixture", run_id="r1", budget=1000)

    assert engine.calls[0] == ("recall", "redis fixture", repo, 4)
    assert engine.calls[2] == ("index", "redis fixture", repo, "r1", 1000)


def test_auto_write_scope_keeps_preferences_global_and_project_facts_local(tmp_path):
    repo, nested = make_repo(tmp_path)
    engine = FakeEngine()
    runtime = MnemeRuntime(engine, cwd_resolver=lambda: nested)

    runtime.add_note("preference", "Prefer concise output", "Keep it short.")
    runtime.add_note("fact", "Redis fixture path", "Use conftest.")

    assert engine.calls[0][-1] is None
    assert engine.calls[1][-1] == repo


def test_outcome_without_explicitly_recalled_keys_is_neutral(tmp_path):
    engine = FakeEngine()
    runtime = MnemeRuntime(engine, cwd_resolver=lambda: tmp_path)

    assert runtime.apply_outcome("r1", "rolled_back") == 0
    assert not any(call[0] == "outcome" for call in engine.calls)

    engine.used = ["k1"]
    assert runtime.apply_outcome("r2", "done") == 1
    assert engine.calls[-1] == ("outcome", "r2", "done", ["k1"])

