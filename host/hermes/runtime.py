"""Deep production runtime for the Hermes Mneme adapter.

This module owns the policies that every hook must share: session-aware
project isolation, write scoping, served-key attribution, and trust movement.
The vendored engine remains host-agnostic; Hermes-specific context lives here.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable


def _default_cwd_resolver() -> Path:
    # Hermes's resolver honors the per-gateway-session ContextVar before the
    # process-wide TERMINAL_CWD. Import lazily so the runtime stays testable
    # outside a full Hermes process; fall back to process cwd.
    try:
        from agent.runtime_cwd import resolve_agent_cwd

        return Path(resolve_agent_cwd())
    except Exception:
        return Path.cwd()


def _repository_root(path: Path) -> Path | None:
    """Find a Git worktree root without spawning git on every memory call."""
    current = Path(path).expanduser()
    try:
        current = current.resolve()
    except OSError:
        return None
    if current.is_file():
        current = current.parent
    for candidate in (current, *current.parents):
        if (candidate / ".git").exists():
            return candidate
    return None


class MnemeRuntime:
    """Hermes policy module over one Mneme engine instance."""

    def __init__(self, engine, *, cwd_resolver: Callable[[], Path] | None = None):
        self.engine = engine
        self._cwd_resolver = cwd_resolver or _default_cwd_resolver
        self._repo_cache: dict[str, Path | None] = {}

    def current_project(self) -> Path | None:
        try:
            cwd = Path(self._cwd_resolver()).expanduser()
            key = str(cwd.resolve()).lower()
        except (OSError, TypeError, ValueError):
            return None
        if key not in self._repo_cache:
            self._repo_cache[key] = _repository_root(cwd)
        return self._repo_cache[key]

    def _write_project(self, scope: str, kind: str) -> Path | None:
        scope = str(scope or "auto").strip().lower()
        if scope not in {"auto", "global", "project"}:
            raise ValueError("scope must be auto, global, or project")
        if scope == "global" or (scope == "auto" and kind == "preference"):
            return None
        project = self.current_project()
        if scope == "project" and project is None:
            raise ValueError("project scope requested outside a Git repository")
        return project

    def recall(self, query: str, *, run_id: str = "", top_k: int = 8):
        hits = self.engine.recall(
            query, repo=self.current_project(), top_k=top_k)
        if run_id and hits:
            self.engine.record_served(
                run_id, [hit["key"] for hit in hits], "recall")
        return hits

    def index_block(self, query: str, *, run_id: str = "", budget: int = 4000):
        return self.engine.index_block(
            repo=self.current_project(), query=query, run_id=run_id, budget=budget)

    def reindex(self):
        return self.engine.reindex(self.current_project())

    def add_note(self, kind: str, title: str, body: str, *, scope: str = "auto", **kwargs):
        return self.engine.add_note(
            kind=kind, title=title, body=body,
            repo=self._write_project(scope, kind), **kwargs)

    def add_episode(self, run_id: str, goal: str, status: str, detail: str):
        return self.engine.add_episode(
            run_id, goal, status, detail, self.current_project())

    def finalize_episode(self, run_id: str, goal: str, status: str, detail: str):
        if hasattr(self.engine, "finalize_episode"):
            return self.engine.finalize_episode(
                run_id, goal, status, detail, self.current_project())
        return self.add_episode(run_id, goal, status, detail)

    def used_keys(self, run_id: str):
        if not hasattr(self.engine, "served_keys"):
            return []
        try:
            return self.engine.served_keys(run_id, tier="recall") or []
        except Exception:
            return []

    def apply_outcome(self, run_id: str, status: str) -> int:
        """Move trust only for memories with explicit usage evidence.

        Index injection proves retrieval, not use. In production, weakly
        debiting every injected memory on a failed job produced a large
        negative skew. No recalled keys now means a neutral outcome.
        """
        used = self.used_keys(run_id)
        if not used:
            return 0
        return self.engine.apply_outcome(run_id, status, used_keys=used)

    def stats(self):
        return self.engine.stats()

    def conflicts(self, *, limit: int = 20):
        if not hasattr(self.engine, "find_conflicts"):
            return []
        return self.engine.find_conflicts(self.current_project(), limit=limit)

