"""Mneme memory provider plugin for Hermes.

File-canon memory layer: markdown notes (operator-owned) + disposable SQLite
FTS5 index. Proof-coupled trust: memories served to a proven-successful run
gain rank; those served to a failure lose it. Per-project banks stop
cross-repo bleed.

This is a MemoryProvider ABC implementation — an adapter over the vendored
mneme.py. Production-polished with:

  - system_prompt_block: live stats banner (note counts, FTS status)
  - prefetch: query-ranked index block with periodic mtime-based reindex
  - get_tool_schemas: mneme_recall (mid-task search) + mneme_add_note
    (operator-writeable path for the model)
  - sync_turn: lightweight turn counter (no auto-retention of transcripts —
    Mneme's doctrine; but tracks turns for episode status)
  - on_session_end: episode recording with turn count + outcome
  - on_memory_write: bridge — mirrors built-in `memory` tool writes into
    Mneme canon so the transition off MEMORY.md is lossless
  - Auto-mounts $HERMES_HOME/memories/ as a read-only extra_dir so existing
    MEMORY.md content competes in ranking instead of being lost

Config ($HERMES_HOME/mneme/config.json):
  index_budget:  max chars for the prefetch index block (default 4000)
  extra_dirs:    additional read-only canon directories (default [])
  auto_mount_memory: mount $HERMES_HOME/memories/ (default true)
  reindex_interval: seconds between mtime sweeps in prefetch (default 60)
  embeddings:    ADR-0004 rerank mode: 'auto' (on iff model files exist) or
                 'off' (force lexical-only; the one-line rollback) (default auto)
  embed_model_dir: dir holding model_quint8_avx2.onnx + vocab.txt
                 ('' = $HERMES_HOME/mneme/models) (default '')

The whole config dict is passed through to the engine as config=; the
engine's DEFAULTS filter drops provider-only keys harmlessly, so one file
configures both layers. w_sem is deliberately NOT exposed here — it is a
fitted constant (bench/engine_gate.py --wsem-grid), not an operator knob.
"""

from __future__ import annotations

import json
import logging
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from agent.memory_provider import MemoryProvider
from tools.registry import tool_error

from .runtime import MnemeRuntime

logger = logging.getLogger(__name__)

# Lazy import of the vendored mneme.py (sibling file)
_mneme_cls = None


def _get_mneme_class():
    global _mneme_cls
    if _mneme_cls is not None:
        return _mneme_cls
    import sys
    from importlib.util import spec_from_file_location, module_from_spec

    mneme_path = Path(__file__).parent / "mneme.py"
    mod_name = "mneme_core"
    spec = spec_from_file_location(mod_name, str(mneme_path))
    mod = module_from_spec(spec)
    # Register BEFORE exec_module so @dataclass can find the module in
    # sys.modules (Python 3.14 dataclasses._is_type requires this).
    sys.modules[mod_name] = mod
    spec.loader.exec_module(mod)
    _mneme_cls = mod.Mneme
    return _mneme_cls


# ---------------------------------------------------------------------------
# Tool schemas
# ---------------------------------------------------------------------------

RECALL_SCHEMA = {
    "name": "mneme_recall",
    "description": (
        "Search Mneme's file-canon memory (lessons, facts, preferences, past "
        "episodes). Returns ranked results with trust scores — memories proven "
        "useful in prior successful runs rank higher. Use this to find relevant "
        "context from past sessions."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "What to search for.",
            },
            "top_k": {
                "type": "integer",
                "description": "Max results to return (default 8).",
            },
        },
        "required": ["query"],
    },
}

ADD_NOTE_SCHEMA = {
    "name": "mneme_add_note",
    "description": (
        "Write a durable memory note (operator-owned canon file). Use for "
        "lessons, facts, or preferences worth remembering across sessions. "
        "The note file is the source of truth — same title = same file = "
        "update in place. Prefer this over the built-in memory tool for "
        "facts that should survive context compression."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "kind": {
                "type": "string",
                "enum": ["lesson", "fact", "preference"],
                "description": "Note category.",
            },
            "title": {
                "type": "string",
                "description": "Short title (slugified to filename).",
            },
            "body": {
                "type": "string",
                "description": "The note content.",
            },
            "keywords": {
                "type": "string",
                "description": "Space-separated keywords for retrieval.",
            },
            "tags": {
                "type": "string",
                "description": "Comma-separated tags.",
            },
            "pinned": {
                "type": "boolean",
                "description": "If true, always included in context (default false).",
            },
            "scope": {
                "type": "string",
                "enum": ["auto", "global", "project"],
                "description": (
                    "Where to store the note. auto keeps preferences global and "
                    "scopes facts/lessons to the current Git project when possible."
                ),
            },
            "supersedes": {
                "type": "string",
                "description": "Slug/title of an older note this one explicitly replaces.",
            },
        },
        "required": ["kind", "title", "body"],
    },
}

FEEDBACK_SCHEMA = {
    "name": "mneme_feedback",
    "description": (
        "Record the OPERATOR'S explicit verdict on this session's outcome, "
        "moving proof-coupled trust for every memory served so far. Call this "
        "ONLY when the operator explicitly states the result in their own "
        "words (e.g. 'that worked', 'that fixed it', 'that was wrong') — "
        "NEVER from your own judgment of how the session went. Self-grading "
        "corrupts the trust system."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "verdict": {
                "type": "string",
                "enum": ["worked", "failed"],
                "description": "The operator's stated outcome.",
            },
            "operator_words": {
                "type": "string",
                "description": "Quote the operator's exact words stating the outcome.",
            },
            "capture_path": {
                "type": "string",
                "enum": ["gate", "session-learn", "manual"],
                "description": "Where the explicit operator verdict was captured.",
            },
            "log_only": {
                "type": "boolean",
                "description": "Log the event without moving trust (gate rollout mode).",
            },
        },
        "required": ["verdict", "operator_words"],
    },
}

STATS_SCHEMA = {
    "name": "mneme_stats",
    "description": (
        "Show Mneme memory statistics: note counts by kind, FTS status, "
        "most-served memories, quarantined count. Use to inspect memory health."
    ),
    "parameters": {
        "type": "object",
        "properties": {},
    },
}

CONFLICTS_SCHEMA = {
    "name": "mneme_conflicts",
    "description": (
        "Report likely competing active notes in the global/current-project "
        "banks. This is read-only; resolve a confirmed conflict by writing a "
        "successor note with the supersedes field."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "limit": {"type": "integer", "description": "Maximum pairs (default 20)."}
        },
    },
}


class MnemeProvider(MemoryProvider):
    """File-canon memory with proof-coupled trust and per-project banks."""

    def __init__(self):
        self._mneme = None
        self._runtime: Optional[MnemeRuntime] = None
        self._session_id: str = ""
        self._run_id: str = ""
        self._mneme_dir: Path = Path(".")
        self._extra_dirs: List[str] = []
        self._index_budget: int = 4000
        self._reindex_interval: float = 60.0
        self._last_reindex: float = 0.0
        self._turn_count: int = 0
        self._episode_checkpointed: bool = False
        self._episode_goal: str = ""
        self._platform: str = ""
        self._agent_identity: str = ""
        self._agent_context: str = ""
        self._hermes_home: str = ""
        self._config: Dict[str, Any] = {}
        self._recent_turns: List[tuple[str, str]] = []

    @property
    def name(self) -> str:
        return "mneme"

    # -- Lifecycle ------------------------------------------------------------

    def is_available(self) -> bool:
        """Check if mneme.py loads."""
        try:
            cls = _get_mneme_class()
            return cls is not None
        except Exception:
            return False

    def initialize(self, session_id: str, **kwargs) -> None:
        self._session_id = str(session_id or "").strip()
        self._run_id = str(uuid.uuid4().hex[:12])
        self._platform = str(kwargs.get("platform", "") or "")
        self._agent_identity = str(kwargs.get("agent_identity", "") or "")
        self._agent_context = str(kwargs.get("agent_context", "") or "")
        self._hermes_home = str(kwargs.get("hermes_home", "") or "")

        # Resolve mneme_dir from HERMES_HOME
        if self._hermes_home:
            self._mneme_dir = Path(self._hermes_home) / "mneme"
        else:
            self._mneme_dir = Path.home() / ".hermes" / "mneme"

        # Read config from $HERMES_HOME/mneme/config.json if present
        config_path = self._mneme_dir / "config.json"
        self._config = {}
        if config_path.exists():
            try:
                self._config = json.loads(
                    config_path.read_text(encoding="utf-8")
                )
            except Exception:
                pass

        self._index_budget = int(self._config.get("index_budget", 4000))
        self._reindex_interval = float(self._config.get("reindex_interval", 60))

        # Build extra_dirs: start with config-provided, auto-mount memories dir
        extra_dirs: List[str] = list(self._config.get("extra_dirs", []))
        auto_mount = self._config.get("auto_mount_memory", True)
        if auto_mount and self._hermes_home:
            mem_dir = str(Path(self._hermes_home) / "memories")
            if mem_dir not in extra_dirs:
                extra_dirs.append(mem_dir)
        self._extra_dirs = extra_dirs

        # Non-fatal guard — Mneme must never break a session (AGENT_SETUP.md)
        try:
            cls = _get_mneme_class()
            notes_dir = self._mneme_dir / "notes"
            db_path = self._mneme_dir / "mneme.db"
            # Filter extra_dirs to only existing paths
            valid_extras = [Path(d) for d in self._extra_dirs if Path(d).exists()]
            # ADR-0004: pass the config through wholesale — the engine's
            # DEFAULTS filter (mneme.py Mneme.__init__) drops provider-only
            # keys (index_budget, extra_dirs, ...) harmlessly. Default the
            # model dir to $HERMES_HOME/mneme/models so 'auto' finds the
            # staged encoder without any config edit.
            engine_cfg = dict(self._config)
            engine_cfg.setdefault(
                "embed_model_dir", str(self._mneme_dir / "models"))
            self._mneme = cls(
                db_path, notes_dir,
                extra_dirs=valid_extras,
                config=engine_cfg,
            )
            self._runtime = MnemeRuntime(self._mneme)
            self._runtime.reindex()
            self._last_reindex = time.time()
            stats = self._mneme.stats()
            emb = stats.get("embeddings", {}) or {}
            logger.info(
                "Mneme initialized: dir=%s, run_id=%s, extra_dirs=%d, "
                "notes=%d rows, fts=%s, embeddings=%s, vectors=%s, coverage=%s",
                self._mneme_dir, self._run_id, len(valid_extras),
                stats.get("rows", 0), stats.get("fts", False),
                emb.get("available", False), emb.get("vectors", 0),
                emb.get("coverage", 0.0),
            )
            # ADR-0004 warm-up: pay the ~100-300ms cold ONNX encoder load
            # HERE at session init — synchronously (no daemon thread, no
            # race) — instead of on the no-timeout prefetch path of the
            # first real turn. One recall = one query encode = model loaded.
            if emb.get("available"):
                self._mneme.recall("session warmup", top_k=1)
        except Exception as exc:
            logger.error("Mneme init failed (non-fatal, memory disabled): %s", exc)
            self._mneme = None
            self._runtime = None

    def shutdown(self) -> None:
        # Finalize the first-turn checkpoint instead of attempting a second
        # first-write-wins add (which used to discard all closing detail).
        if self._runtime is not None and (
                self._turn_count > 0 or self._episode_checkpointed):
            try:
                self._finalize_current("session_end")
            except Exception as exc:
                logger.debug("Mneme shutdown episode finalization failed: %s", exc)
        if self._mneme is not None:
            try:
                self._mneme.close()
            except Exception:
                pass

    def backup_paths(self) -> List[str]:
        """Mneme data lives under HERMES_HOME, so hermes backup already covers it."""
        return []

    # -- Session rotation -----------------------------------------------------

    def on_session_switch(
        self,
        new_session_id: str,
        *,
        parent_session_id: str = "",
        reset: bool = False,
        **kwargs,
    ) -> None:
        # On reset/new, record the old session's episode if it had turns
        if reset and self._runtime is not None and (
                self._turn_count > 0 or self._episode_checkpointed):
            try:
                self._finalize_current("session_reset")
            except Exception as exc:
                logger.debug("Mneme reset episode finalization failed: %s", exc)
        self._session_id = str(new_session_id or "").strip()
        self._run_id = str(uuid.uuid4().hex[:12])
        self._turn_count = 0
        self._episode_checkpointed = False
        self._episode_goal = ""
        self._recent_turns = []

    # -- System prompt (volatile tier) ----------------------------------------

    def system_prompt_block(self) -> str:
        if self._mneme is None:
            return ""
        try:
            stats = self._runtime.stats() if self._runtime else self._mneme.stats()
            by_kind = stats.get("by_kind", {})
            counts = " ".join(
                f"{k}={v}" for k, v in sorted(by_kind.items()) if k != "episode"
            )
            ep_count = by_kind.get("episode", 0)
            fts = "FTS5" if stats.get("fts") else "LIKE"
            emb = stats.get("embeddings", {}) or {}
            semantic = "semantic" if emb.get("available") else "lexical-only"
            degraded = ""
            if emb.get("configured") and not emb.get("available"):
                degraded = f" WARNING: semantic recall degraded ({emb.get('error','unknown error')})."
            project = self._runtime.current_project() if self._runtime else None
            scope = f" Project bank: {project}." if project else " Global bank."
            return (
                f"# Mneme Memory\n"
                f"Active. File-canon notes + proof-coupled trust ({fts}, {semantic}). "
                f"{counts} episodes={ep_count}. "
                f"Use mneme_recall to search, mneme_add_note to write durable notes."
                f"{scope}{degraded}"
            )
        except Exception:
            return (
                f"# Mneme Memory\n"
                f"Active. File-canon notes + proof-coupled trust. "
                f"Use mneme_recall to search, mneme_add_note to write durable notes."
            )

    # -- Prefetch (the guaranteed read path) ----------------------------------

    def _maybe_reindex(self) -> None:
        """Cheap periodic mtime sweep — picks up externally-edited notes."""
        if self._mneme is None:
            return
        now = time.time()
        if now - self._last_reindex < self._reindex_interval:
            return
        try:
            result = self._runtime.reindex() if self._runtime else self._mneme.reindex()
            if result.get("added", 0) or result.get("updated", 0) or result.get("pruned", 0):
                logger.debug("Mneme reindex: %s", result)
            self._last_reindex = now
        except Exception as exc:
            logger.debug("Mneme periodic reindex failed (non-fatal): %s", exc)

    def _checkpoint_episode(self, goal: str) -> None:
        """Record this segment's episode NOW (idempotent, first write wins).
        Called from prefetch (synchronous, in the request path) because
        sync_turn runs on the manager's background worker — a CLI one-shot
        exits before the queued write lands, which is the exact unclean-exit
        gap this checkpoint exists to close."""
        if self._runtime is None or self._episode_checkpointed:
            return
        try:
            goal_text = (goal or "").strip()[:200] or f"Session {self._session_id}"
            self._episode_goal = goal_text
            self._runtime.add_episode(
                self._run_id,
                goal_text,
                "session_open",
                f"Checkpointed at first serve: {self._session_id} on {self._platform}",
            )
            self._episode_checkpointed = True
        except Exception as exc:
            logger.debug("Mneme episode checkpoint failed (non-fatal): %s", exc)

    def _finalize_current(
        self,
        status: str,
        messages: Optional[List[Dict[str, Any]]] = None,
        *,
        run_id: str = "",
    ) -> None:
        """Upgrade the current (or specified) run segment with best-known detail."""
        if self._runtime is None:
            return
        rid = run_id or self._run_id
        if not rid:
            return
        goal, detail = self._episode_summary(messages or [])
        self._runtime.finalize_episode(rid, goal, status, detail)

    def prefetch(self, query: str, *, session_id: str = "") -> str:
        """Return the index block for the current turn's query."""
        if self._mneme is None:
            return ""
        try:
            self._maybe_reindex()
            self._checkpoint_episode(query)
            ib = self._runtime.index_block(
                query or "", run_id=self._run_id, budget=self._index_budget,
            )
            if not ib.text:
                return ""
            return (
                "# Mneme Memory — KNOWN CONSTRAINTS (file-canon persistent context)\n"
                "These lessons, facts, and past episodes are already established. "
                "Synthesize them BEFORE your first tool call: aim the first probe at "
                "what they leave unknown, verify their claims at the point of use, "
                "and never ignore them to probe blind.\n\n"
                f"{ib.text}"
            )
        except Exception as exc:
            logger.debug("Mneme prefetch failed (non-fatal): %s", exc)
            return ""

    def queue_prefetch(self, query: str, *, session_id: str = "") -> None:
        """No background prefetch — index_block is fast (SQLite, in-process)."""
        pass

    # -- Turn sync -----------------------------------------------------------

    def sync_turn(
        self,
        user_content: str,
        assistant_content: str,
        *,
        session_id: str = "",
        messages: Optional[List[Dict[str, Any]]] = None,
    ) -> None:
        """Track turn count, checkpoint, and progressively upgrade the episode.

        Does NOT auto-retain full chat transcripts — Mneme's canon is
        operator-authored notes (AGENT_SETUP.md failure mode: 'LLM-managed
        memory'). Progressive finalize means unclean exits still leave the
        latest turn summary instead of a permanent thin checkpoint.
        """
        self._turn_count += 1
        self._recent_turns.append((
            str(user_content or "")[:600], str(assistant_content or "")[:900]))
        self._recent_turns = self._recent_turns[-4:]
        # Fallback anchor: prefetch is the primary checkpoint site (synchronous);
        # this covers hosts that skip prefetch but still deliver turns.
        self._checkpoint_episode(user_content)
        # Progressive upgrade (ADR-0006): each turn rewrites the same run row
        # with growing detail. Engine richness is monotonic, so a later thin
        # close cannot erase this.
        if self._episode_checkpointed:
            try:
                self._finalize_current("session_progress")
            except Exception as exc:
                logger.debug("Mneme progressive episode upgrade failed: %s", exc)

    def _rotate_run_segment(self, reason: str) -> None:
        """Start a fresh evidence segment after an applied outcome.

        The engine is exactly-once per run_id, and gateway sessions can live
        for days — with a single session-lifetime run_id, only the FIRST
        operator verdict could ever move trust. Rotating the run id after
        each applied outcome turns a long session into a chain of segments:
        serves attribute to the current segment, each verdict closes one and
        opens the next. This is how trust accumulates DURING sessions instead
        of only at (unreliable) clean exits.

        The closing segment is finalized first so mid-session verdicts do not
        leave orphan session_open checkpoints with no summary.
        """
        old = self._run_id
        if self._episode_checkpointed or self._turn_count > 0:
            try:
                self._finalize_current("session_segment", run_id=old)
            except Exception as exc:
                logger.debug("Mneme segment close failed (%s): %s", old, exc)
        self._run_id = str(uuid.uuid4().hex[:12])
        self._episode_checkpointed = False
        # Keep goal/turns for continuity in the next segment summary; only
        # served evidence is segmented by run_id.
        logger.info("Mneme run segment rotated (%s): %s -> %s", reason, old, self._run_id)

    def on_turn_start(self, turn_number: int, message: str, **kwargs) -> None:
        """Light hook — no action needed (reindex handled in prefetch)."""
        pass

    @staticmethod
    def _message_text(message: Dict[str, Any]) -> str:
        content = message.get("content", "")
        if isinstance(content, str):
            return content.strip()
        if isinstance(content, list):
            parts = []
            for item in content:
                if isinstance(item, dict) and isinstance(item.get("text"), str):
                    parts.append(item["text"])
            return "\n".join(parts).strip()
        return str(content or "").strip()

    def _episode_summary(self, messages: List[Dict[str, Any]]) -> tuple[str, str]:
        """Build a compact deterministic episode, never a raw transcript dump."""
        users = [self._message_text(m) for m in messages
                 if m.get("role") == "user" and self._message_text(m)]
        assistants = [self._message_text(m) for m in messages
                      if m.get("role") == "assistant" and self._message_text(m)]
        if not users and self._recent_turns:
            users = [u for u, _ in self._recent_turns if u]
        if not assistants and self._recent_turns:
            assistants = [a for _, a in self._recent_turns if a]
        goal = (
            users[0] if users
            else (self._episode_goal or f"Session {self._session_id}")
        )[:200]
        lines = [
            f"Session: {self._session_id}",
            f"Platform: {self._platform or 'unknown'}",
            f"Turns: {self._turn_count}",
        ]
        project = self._runtime.current_project() if self._runtime else None
        if project:
            lines.append(f"Project: {project}")
        if goal and (not users or users[0][:200] != goal):
            lines.append(f"Goal: {goal}")
        for text in users[-3:]:
            lines.append(f"User development: {text[:300]}")
        if assistants:
            lines.append(f"Final response/result: {assistants[-1][:900]}")
        elif self._turn_count == 0 and not users:
            # Prefetch-only / smoke path — keep the body honest and short so
            # richness scoring demotes it instead of inventing a summary.
            lines.append(
                f"Checkpoint only: no completed turns recorded "
                f"({self._session_id} on {self._platform or 'unknown'})"
            )
        return goal, "\n".join(lines)[:2000]

    def on_session_end(self, messages: List[Dict[str, Any]]) -> None:
        """Record one episode per session. Trust is NOT credited here.

        The design law: trust moves only on PROVEN outcomes. A chat session
        ending is not evidence the served memories helped — crediting every
        session as 'done' inflates trust monotonically (attendance trophies).
        Episodes here use the neutral status 'session_end' (anything outside
        Mneme's POSITIVE/NEGATIVE_STATUSES is neutral by construction).

        Proven outcomes arrive via: (a) the mneme_feedback tool when the
        OPERATOR explicitly states an outcome, or (b) apply_job_outcome()
        called by gateway job/cron plumbing with a mechanical ok/error result.
        """
        if self._runtime is None:
            return
        try:
            self._finalize_current("session_end", messages)
        except Exception as exc:
            logger.debug("Mneme on_session_end failed (non-fatal): %s", exc)
        # FABLE.5 §6.7 auto-feedback: a TERMINAL receipt in this session IS a
        # proven outcome (the design law is satisfied — trust still moves only on
        # proof, now sourced from the Evidence Bus instead of only operator
        # feedback). Positive verdict -> credit served memories; negative ->
        # debit. Neutral/no-terminal-receipt sessions move nothing.
        try:
            from tools import receipts as _bus
            from tools import trust_writeback as _tw
            recs = [c.receipt for c in _bus.resolve({"session_id": self._session_id})]
            verdict = _tw.classify_session(recs)
            if verdict in ("positive", "negative"):
                self.apply_job_outcome(verdict == "positive", run_id=self._run_id)
        except Exception as exc:
            logger.debug("Mneme receipt auto-feedback skipped: %s", exc)

    # -- Proven-outcome inputs -------------------------------------------------

    def _used_keys_for(self, run_id: str):
        """Honest usage evidence for a run: memories the model explicitly
        fetched via mneme_recall (tier='recall'). Index-block serves are
        unknown, so they stay out — Phase 0 trust measures usefulness, not
        retrievability. None (not []) when there is no evidence."""
        if self._runtime is None:
            return None
        try:
            return self._runtime.used_keys(run_id) or None
        except Exception:
            return None

    def _apply_outcome(self, run_id: str, status: str) -> int:
        if self._runtime is None:
            return 0
        # No explicit recall/use evidence means neutral. This prevents failed
        # jobs from weakly debiting every memory injected into their prompt.
        return self._runtime.apply_outcome(run_id, status)

    def apply_job_outcome(self, ok: bool, *, run_id: str = "") -> int:
        """Mechanical outcome from gateway job/cron plumbing: ok -> 'done',
        error -> 'rolled_back'. Phase 0 semantics: memories the model RECALLED
        during the run move a full trust point; merely index-served memories
        stay neutral on wins (failures nudge them weakly). Returns notes moved.
        This is the hook the cron runner should call per job completion."""
        if self._mneme is None:
            return 0
        try:
            rid = run_id or self._run_id
            moved = self._apply_outcome(rid, "done" if ok else "rolled_back")
            if rid == self._run_id:
                self._rotate_run_segment("job outcome")
            return moved
        except Exception as exc:
            logger.debug("Mneme apply_job_outcome failed (non-fatal): %s", exc)
            return 0

    def on_pre_compress(self, messages: List[Dict[str, Any]]) -> str:
        """Before context compression, note what we're about to lose."""
        if self._turn_count < 5:
            return ""
        return (
            f"[Mneme: {self._turn_count} turns in this session — "
            f"key facts already in Mneme canon via mneme_add_note]"
        )

    # -- Bridge: mirror built-in memory tool writes ---------------------------

    def on_memory_write(
        self,
        action: str,
        target: str,
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Bridge from the built-in `memory` tool into Mneme canon.

        When the model uses the built-in memory tool (action=add/replace), the
        content is mirrored as a Mneme note so it enters the ranking pipeline
        and benefits from proof-coupled trust. This makes the transition from
        MEMORY.md to Mneme lossless — nothing written via the old tool is lost.
        """
        if self._mneme is None or not content:
            return
        if action not in ("add", "replace"):
            return
        try:
            # Determine kind from target
            kind = "preference" if target == "user" else "fact"
            # Use the first line as title, rest as body
            lines = content.strip().split("\n", 1)
            title = lines[0].lstrip("# ").strip()[:80] or "untitled"
            body = lines[1].strip() if len(lines) > 1 else ""
            # Add with overwrite=True for replace, False for add
            self._runtime.add_note(
                kind=kind,
                title=title,
                body=body or content[:500],
                keywords="",
                tags="bridged",
                scope="auto",
                overwrite=(action == "replace"),
            )
            logger.debug("Mneme bridged memory %s: %s", action, title[:50])
        except Exception as exc:
            logger.debug("Mneme on_memory_write bridge failed (non-fatal): %s", exc)

    # -- Tools ----------------------------------------------------------------

    def get_tool_schemas(self) -> List[Dict[str, Any]]:
        return [RECALL_SCHEMA, ADD_NOTE_SCHEMA, STATS_SCHEMA,
                FEEDBACK_SCHEMA, CONFLICTS_SCHEMA]

    def handle_tool_call(self, tool_name: str, args: Dict[str, Any], **kwargs) -> str:
        if tool_name == "mneme_recall":
            return self._handle_recall(args)
        if tool_name == "mneme_add_note":
            return self._handle_add_note(args)
        if tool_name == "mneme_stats":
            return self._handle_stats(args)
        if tool_name == "mneme_feedback":
            return self._handle_feedback(args)
        if tool_name == "mneme_conflicts":
            return self._handle_conflicts(args)
        return tool_error(f"Unknown tool: {tool_name}")

    def _handle_feedback(self, args: Dict[str, Any]) -> str:
        """Operator verdict -> proven outcome. The operator is the verifier of
        a chat system; their explicit statement is the proof signal."""
        if self._mneme is None:
            return tool_error("Mneme is not initialized")
        verdict = str(args.get("verdict", "")).strip().lower()
        words = str(args.get("operator_words", "")).strip()
        capture_path = str(args.get("capture_path", "manual")).strip()
        log_only = bool(args.get("log_only", False))
        if verdict not in ("worked", "failed"):
            return tool_error("verdict must be 'worked' or 'failed'")
        if not words:
            return tool_error("operator_words is required — quote the operator's "
                              "exact statement; never self-grade")
        try:
            status = "done" if verdict == "worked" else "rolled_back"
            run_segment = self._run_id
            used = self._used_keys_for(self._run_id)
            touched = 0 if log_only else self._apply_outcome(self._run_id, status)
            event = {
                "ts": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
                "session_id": self._session_id,
                "run_segment": run_segment,
                "verdict": verdict,
                "operator_quote": words[:300],
                "capture_path": capture_path if capture_path in ("gate", "session-learn", "manual") else "manual",
                "notes_touched": used or [],
                "trust_applied": bool(touched) and not log_only,
            }
            feedback_path = self._mneme_dir / "notes" / ".mneme" / "feedback_database.json"
            feedback_path.parent.mkdir(parents=True, exist_ok=True)
            with feedback_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(event, ensure_ascii=False) + "\n")
            logger.info("Mneme feedback (%s): %d memories %s (used=%s) — operator said: %s",
                        verdict, touched,
                        "credited" if verdict == "worked" else "debited",
                        len(used) if used else 0, words[:100])
            note = ""
            if verdict == "worked" and not used:
                note = ("no memories were recalled this segment, so a win moves "
                        "nothing (Phase 0: trust measures usefulness — only "
                        "recalled/used memories earn credit)")
            # Fresh evidence segment: the next verdict in this session can move
            # trust again instead of hitting the exactly-once run guard.
            if not log_only:
                self._rotate_run_segment(f"operator verdict: {verdict}")
            return json.dumps({"status": "ok", "verdict": verdict,
                               "memories_moved": touched,
                               "segment": "rotated — future recalls/verdicts in this "
                                          "session form a new evidence segment",
                               **({"note": note} if note else {})})
        except Exception as exc:
            logger.warning("mneme_feedback failed: %s", exc, exc_info=True)
            return tool_error(f"Feedback failed: {exc}")

    def _handle_recall(self, args: Dict[str, Any]) -> str:
        if self._mneme is None:
            return tool_error("Mneme is not initialized")
        query = str(args.get("query", "")).strip()
        if not query:
            return tool_error("Missing required parameter: query")
        try:
            top_k = int(args.get("top_k", 8))
        except (TypeError, ValueError):
            top_k = 8
        try:
            hits = self._runtime.recall(
                query, run_id=self._run_id, top_k=top_k)
            results = [
                {
                    "kind": h["kind"],
                    "title": h["title"],
                    "body": h["body"][:500],
                    "trust": round(h["trust"], 2),
                    "score": round(h["score"], 3),
                }
                for h in hits
            ]
            return json.dumps({"results": results})
        except Exception as exc:
            logger.warning("mneme_recall failed: %s", exc, exc_info=True)
            return tool_error(f"Search failed: {exc}")

    def _handle_add_note(self, args: Dict[str, Any]) -> str:
        if self._mneme is None:
            return tool_error("Mneme is not initialized")
        kind = str(args.get("kind", "")).strip()
        title = str(args.get("title", "")).strip()
        body = str(args.get("body", "")).strip()
        if not kind or not title or not body:
            return tool_error("Missing required parameter: kind, title, body")
        try:
            path = self._runtime.add_note(
                kind=kind,
                title=title,
                body=body,
                keywords=str(args.get("keywords", "")),
                tags=str(args.get("tags", "")),
                pinned=bool(args.get("pinned", False)),
                supersedes=str(args.get("supersedes", "")),
                scope=str(args.get("scope", "auto")),
            )
            return json.dumps({
                "status": "ok", "path": str(path) if path else "",
                "scope": str(args.get("scope", "auto")),
            })
        except ValueError as exc:
            return tool_error(str(exc))
        except Exception as exc:
            logger.warning("mneme_add_note failed: %s", exc, exc_info=True)
            return tool_error(f"Failed to write note: {exc}")

    def _handle_stats(self, args: Dict[str, Any]) -> str:
        if self._mneme is None:
            return tool_error("Mneme is not initialized")
        try:
            stats = self._runtime.stats()
            stats["project"] = (str(self._runtime.current_project())
                                if self._runtime.current_project() else None)
            stats["conflict_samples"] = self._runtime.conflicts(limit=5)
            return json.dumps(stats, indent=2, default=str)
        except Exception as exc:
            return tool_error(f"Stats failed: {exc}")

    def _handle_conflicts(self, args: Dict[str, Any]) -> str:
        if self._runtime is None:
            return tool_error("Mneme is not initialized")
        try:
            limit = max(1, min(100, int(args.get("limit", 20))))
            return json.dumps({"conflicts": self._runtime.conflicts(limit=limit)},
                              indent=2, default=str)
        except (TypeError, ValueError):
            return tool_error("limit must be an integer")
        except Exception as exc:
            return tool_error(f"Conflict scan failed: {exc}")

    # -- Config schema (for `hermes memory setup`) ----------------------------

    def get_config_schema(self) -> List[Dict[str, Any]]:
        return [
            {
                "key": "index_budget",
                "description": "Max characters for the prefetch index block",
                "default": 4000,
                "required": False,
            },
            {
                "key": "extra_dirs",
                "description": "Additional read-only canon note directories (JSON list)",
                "default": [],
                "required": False,
            },
            {
                "key": "auto_mount_memory",
                "description": "Auto-mount $HERMES_HOME/memories/ as read-only canon",
                "default": True,
                "required": False,
            },
            {
                "key": "reindex_interval",
                "description": "Seconds between mtime sweeps in prefetch (0 = init only)",
                "default": 60,
                "required": False,
            },
            {
                "key": "embeddings",
                "description": ("ADR-0004 embedding rerank: 'auto' (on iff model "
                                "files exist) or 'off' (force lexical-only)"),
                "default": "auto",
                "required": False,
            },
            {
                "key": "embed_model_dir",
                "description": ("Directory with model_quint8_avx2.onnx + vocab.txt "
                                "('' = $HERMES_HOME/mneme/models)"),
                "default": "",
                "required": False,
            },
        ]

    def save_config(self, values: Dict[str, Any], hermes_home: str) -> None:
        config_dir = Path(hermes_home) / "mneme"
        config_dir.mkdir(parents=True, exist_ok=True)
        config_path = config_dir / "config.json"
        existing = {}
        if config_path.exists():
            try:
                existing = json.loads(config_path.read_text(encoding="utf-8"))
            except Exception:
                pass
        existing.update(values)
        config_path.write_text(
            json.dumps(existing, indent=2), encoding="utf-8"
        )


# ---------------------------------------------------------------------------
# Plugin registration (register(ctx) pattern)
# ---------------------------------------------------------------------------

_provider_instance: Optional[MnemeProvider] = None


def register(ctx) -> None:
    global _provider_instance
    _provider_instance = MnemeProvider()
    ctx.register_memory_provider(_provider_instance)
    logger.info("Mneme memory provider registered (file-canon, proof-coupled trust)")
