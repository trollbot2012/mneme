"""Mneme — a persistent memory layer for AI agents. One file, stdlib only.

Files are canon, SQLite is the engine:
- Your durable memory is plain markdown notes (lessons / facts / preferences)
  in folders you own; git or backups version them.
- A DISPOSABLE SQLite FTS5 index does retrieval (BM25 + porter stemming,
  trigram re-rank, LIKE fallback on FTS-less builds). Delete the .db and
  reindex — nothing is lost.
- Episodes (one row per agent run) live in the DB, auto-compacted to an
  archive, never silently deleted.
- Bi-temporal supersession: updates invalidate, never destroy.
- Proof-coupled trust: a memory served into a run that provably SUCCEEDED
  gains rank; one served into a failure loses it. Model opinion never moves it.
- Banks: per-project scoping so one project's facts never leak into another.

Wire it into any agent with three calls:
    mem = Mneme(db_path, notes_dir)
    block = mem.index_block(project, task_text, run_id=...)   # inject at task start
    mem.record_served(run_id, keys, tier); mem.apply_outcome(run_id, status)
Born inside the Ktisis coding agent (proof-gated, ADR-driven); design brief and
research provenance: see PRD.md. Extracted unchanged — Ktisis runs this code.
"""

from __future__ import annotations

__version__ = "0.1.0"

import hashlib
import re
import time
from dataclasses import dataclass
from pathlib import Path

KINDS = ("lesson", "fact", "preference", "episode")

_BOOL_TRUE = ("1", "true", "yes", "on")


@dataclass
class Note:
    path: Path
    kind: str = "lesson"
    title: str = ""
    body: str = ""
    keywords: str = ""
    tags: str = ""
    pinned: bool = False
    supersedes: str = ""
    valid_at: float = 0.0
    mtime: float = 0.0
    note_id: str = ""   # stable identity from frontmatter `id:` (may be empty)
    raw_sha: str = ""   # sha256 of the exact file text (promotion integrity)


def _parse_frontmatter(text: str) -> tuple[dict, str]:
    """Split leading `---\\nkey: value\\n---` block from the body. Flat string
    values only — this is deliberately not YAML (stdlib bundle, ADR 0003)."""
    if not text.startswith("---"):
        return {}, text
    m = re.match(r"^---\s*\n(.*?)\n---\s*\n?", text, re.DOTALL)
    if not m:
        return {}, text
    meta = {}
    for line in m.group(1).splitlines():
        if ":" not in line:
            continue
        k, _, v = line.partition(":")
        meta[k.strip().lower()] = v.strip()
    return meta, text[m.end():]


def _parse_valid_at(raw: str, fallback: float) -> float:
    raw = raw.strip()
    if not raw:
        return fallback
    for fmt in ("%Y-%m-%d %H:%M", "%Y-%m-%d"):
        try:
            return time.mktime(time.strptime(raw, fmt))
        except (ValueError, OverflowError, OSError):
            # mktime overflows on far-future dates and raises OSError on
            # pre-1970 dates on Windows — a bad frontmatter date must never
            # crash the whole reindex (audit C: OverflowError escape).
            continue
    try:
        return float(raw)
    except (ValueError, OverflowError):
        return fallback


def _kind_from_dir(path: Path) -> str:
    name = path.parent.name.lower().rstrip("s")  # lessons -> lesson
    return name if name in KINDS else "lesson"


def parse_note(path: Path) -> Note | None:
    """Parse one note file. Returns None for empty files; a corrupt note must
    not kill the reindex (same doctrine as SkillStore.all)."""
    try:
        text = path.read_text(encoding="utf-8", errors="ignore").strip()
        mtime = path.stat().st_mtime
    except OSError:
        return None
    if not text:
        return None
    meta, rest = _parse_frontmatter(text)
    rest = rest.strip()
    first, _, body = rest.partition("\n")
    title = first.lstrip("# ").strip()
    kind = meta.get("kind", "").strip().lower() or _kind_from_dir(path)
    if kind not in KINDS:
        kind = "lesson"
    return Note(
        path=path,
        kind=kind,
        title=title or path.stem,
        body=body.strip(),
        keywords=meta.get("keywords", ""),
        tags=meta.get("tags", ""),
        pinned=meta.get("pinned", "").lower() in _BOOL_TRUE,
        supersedes=meta.get("supersedes", ""),
        valid_at=_parse_valid_at(meta.get("valid_at", ""), mtime),
        mtime=mtime,
        note_id=meta.get("id", "").strip(),
        raw_sha=hashlib.sha256(text.encode("utf-8")).hexdigest()[:16],
    )


def scan_notes(root: Path) -> list[Note]:
    """All notes under a canon directory (recursive), corrupt/empty skipped."""
    if not root.is_dir():
        return []
    out = []
    for f in sorted(root.rglob("*.md")):
        note = parse_note(f)
        if note is not None:
            out.append(note)
    return out


def slugify(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")[:60]


def _derived_note_id(bank: str, kind: str, title: str) -> str:
    """Deterministic stable id for notes with no frontmatter `id:` (hand-authored,
    repo-dropped). Derived from bank+kind+slug, NOT the filesystem path, so a
    file rename, repo move, or cross-host checkout keeps the same identity and
    its trust/quarantine state follows (design review: path-keyed stats made
    the 'disposable DB' promise false)."""
    basis = f"{bank}|{kind}|{slugify(title)}"
    return "n:" + hashlib.sha256(basis.encode("utf-8")).hexdigest()[:16]


def render_note(kind: str, title: str, body: str, *, keywords: str = "", tags: str = "",
                pinned: bool = False, supersedes: str = "", note_id: str = "") -> str:
    """Render a note file with frontmatter (for `ktisis mneme add` / distiller)."""
    lines = ["---", f"kind: {kind}"]
    if note_id:
        lines.append(f"id: {note_id}")
    if keywords:
        lines.append(f"keywords: {keywords}")
    if tags:
        lines.append(f"tags: {tags}")
    if pinned:
        lines.append("pinned: true")
    if supersedes:
        lines.append(f"supersedes: {supersedes}")
    lines += ["---", f"# {title}", "", body.strip(), ""]
    return "\n".join(lines)


import hashlib
import json
import os
import re
import sqlite3
import subprocess
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path


def _norm_path(path: Path) -> str:
    """Canonical posix string for a path. Case-fold ONLY on Windows — POSIX
    filesystems are case-sensitive, so lowercasing there collides distinct
    files (audit: Foo.md vs foo.md). _file_key and reindex-prune share this."""
    p = path.resolve().as_posix()
    return p.lower() if os.name == "nt" else p


def _like_escape(s: str) -> str:
    r"""Escape SQL LIKE metacharacters so filesystem paths/slugs match
    literally. `_` and `%` in a path (common in usernames/repo names) are
    wildcards otherwise — reindex-prune deleted sibling trees, quarantine
    vetoed the wrong row (audit C4/M1). Pair with ESCAPE '\'."""
    return s.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")


_SCHEMA = """
CREATE TABLE IF NOT EXISTS mem (
    id TEXT PRIMARY KEY,
    dedupe_key TEXT NOT NULL UNIQUE,
    note_id TEXT NOT NULL DEFAULT '',
    kind TEXT NOT NULL,
    bank TEXT NOT NULL,
    title TEXT NOT NULL,
    body TEXT NOT NULL,
    keywords TEXT NOT NULL DEFAULT '',
    tags TEXT NOT NULL DEFAULT '',
    source TEXT NOT NULL,
    pinned INTEGER NOT NULL DEFAULT 0,
    credibility TEXT NOT NULL DEFAULT 'operator',
    file_mtime REAL,
    created_at REAL NOT NULL,
    valid_at REAL NOT NULL,
    invalid_at REAL,
    superseded_by TEXT
);
CREATE INDEX IF NOT EXISTS mem_bank_kind ON mem (bank, kind);
CREATE INDEX IF NOT EXISTS mem_note_id ON mem (note_id);
CREATE TABLE IF NOT EXISTS mem_stats (
    note_id TEXT PRIMARY KEY,
    served INTEGER NOT NULL DEFAULT 0,
    positive REAL NOT NULL DEFAULT 0,
    negative REAL NOT NULL DEFAULT 0,
    quarantined INTEGER NOT NULL DEFAULT 0,
    last_positive REAL
);
CREATE TABLE IF NOT EXISTS mem_served (
    run_id TEXT NOT NULL,
    dedupe_key TEXT NOT NULL,
    note_id TEXT NOT NULL DEFAULT '',
    tier TEXT NOT NULL,
    ts REAL NOT NULL,
    used INTEGER NOT NULL DEFAULT 0
);
CREATE TABLE IF NOT EXISTS run_outcome (
    run_id TEXT PRIMARY KEY,
    status TEXT NOT NULL,
    ts REAL NOT NULL
);
"""

_FTS_SCHEMA = """
CREATE VIRTUAL TABLE IF NOT EXISTS mem_fts USING fts5(
    dedupe_key UNINDEXED, title, body, keywords, tags,
    tokenize='porter unicode61'
);
"""

# Per-kind share of the ranked (non-pinned) index budget. Lessons dominate:
# a coding agent's most valuable memory is what failed before.
_KIND_SHARE = {"lesson": 0.45, "fact": 0.25, "preference": 0.15, "episode": 0.15}
_SECTION_HEADERS = {
    "lesson": "Known lessons for this environment:",
    "fact": "Known facts about this repo:",
    "preference": "Operator preferences:",
    "episode": "Related past runs:",
}
_LINE_CAP = 220
_TOKEN_RE = re.compile(r"[A-Za-z0-9_]{2,}")
_NL = chr(10)

DEFAULTS = {
    "index_budget_chars": 4000,
    "recall_top_k": 8,
    "episode_half_life_days": 30,
    "max_episodes": 2000,          # per BANK, not global (audit M2)
    "compact_after_days": 90,
    "auto_compact": True,
    # Where project-scoped notes are written under a project dir. Default keeps
    # Ktisis parity (.ktisis/memory); standalone hosts can point it anywhere
    # (audit: host-generic). Also the dir reindex(project) scans.
    "project_subdir": ".ktisis/memory",
    # Row count below which df-aware query-term pruning stays off (benches can
    # lower this to prove p@3 is unchanged with pruning active). Mirrors
    # DF_PRUNE_MIN_ROWS below (defined after this dict).
    "df_prune_min_rows": 2000,
}

# Episodes may exceed max_episodes by this slack before an automatic compact
# fires (keeps the common add_episode path cheap; compaction stays amortized).
AUTO_COMPACT_SLACK = 10

# Outcomes that count as evidence about served memories. Everything else
# (denied, blocked, stopped, crashed, vacuous_checks, planned) is neutral:
# the work was never carried to a provable end, so memory takes no blame.
POSITIVE_STATUSES = ("done",)
NEGATIVE_STATUSES = ("rolled_back", "failed_verification")

# Trust semantics (design review: served != useful). A memory the host marks
# USED moves trust by a full point; one merely SERVED gets a weak negative —
# being retrieved a lot without ever being used is evidence AGAINST it, and
# crediting the whole served set was a rich-get-richer loop that collapsed
# trust toward the repo's success rate. Without used_keys all evidence is weak.
SERVED_UNUSED_NEG = 0.1

# Score multiplier for unverified (repo-credibility) notes: they may surface,
# clearly labelled, but must never outrank trusted canon by default.
UNVERIFIED_WEIGHT = 0.3

# Ranking blend — ONE source of truth (the 0.65/0.35 literal used to live in
# three places). Trigram-Jaccard is DROPPED when FTS5 ranks candidates: the
# bakeoff shows byte-identical p@3 at every scale while doubling per-candidate
# scoring cost (bm25 0.52/0.56 == bm25+jaccard 0.52/0.56 at 1000/2000). On the
# FTS-less LIKE fallback, hit-count is the only lexical signal, so Jaccard
# keeps its old weight there — it genuinely re-ranks that path.
LEX_WEIGHT, JAC_WEIGHT = 1.0, 0.0            # fts_available
FALLBACK_LEX_WEIGHT, FALLBACK_JAC_WEIGHT = 0.65, 0.35  # LIKE fallback

# df-aware query-term pruning (measured 3-64x recall speedup at 100k rows):
# a term matching more than DF_PRUNE_RATIO of the corpus forces FTS5 to
# bm25-score a huge candidate union; dropping it keeps the rare terms that
# carry the ranking anyway. Inactive below DF_PRUNE_MIN_ROWS (small stores
# are already sub-ms, and the bakeoff corpus proves p@3 is unchanged there).
DF_PRUNE_MIN_ROWS = 2000
DF_PRUNE_RATIO = 0.10
DF_PRUNE_KEEP_MIN = 3

_UNVERIFIED_HEADER = "Unverified notes suggested by this repo (verify before trusting):"
_UNVERIFIED_MAX_LINES = 3


def _tokens(text: str, cap: int = 24) -> list[str]:
    seen: dict[str, None] = {}
    for m in _TOKEN_RE.finditer(text.lower()):
        seen.setdefault(m.group(0))
        if len(seen) >= cap:
            break
    return list(seen)


def _trigrams(text: str) -> set[str]:
    t = re.sub(r"\s+", " ", text.lower()).strip()
    return {t[i:i + 3] for i in range(len(t) - 2)} if len(t) > 2 else {t} if t else set()


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


@dataclass
class IndexBlock:
    text: str = ""
    keys: list[str] = field(default_factory=list)


class Mneme:
    def __init__(self, db_path: Path, global_dir: Path, extra_dirs: list | None = None,
                 config: dict | None = None):
        self.db_path = Path(db_path)
        self.global_dir = Path(global_dir)
        self.extra_dirs = [Path(d) for d in (extra_dirs or [])]
        cfg = dict(DEFAULTS)
        cfg.update({k: v for k, v in (config or {}).items() if k in DEFAULTS})
        self.cfg = cfg
        self._bank_cache: dict[str, str] = {}
        self._cooc = None  # lazy co-occurrence expansion table
        self._df_cache: dict[str, int] = {}  # token -> document frequency
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        # RLock: _with_write_retry holds it across attempt AND rollback so a
        # sibling thread can never interleave onto a half-failed transaction
        # (verify F18); closures still take it themselves — re-entrant.
        self._lock = threading.RLock()
        with self._lock:
            try:
                self._conn.execute("PRAGMA journal_mode=WAL")
            except sqlite3.DatabaseError:
                pass  # network drives / old builds: default journal still works
            try:
                # Fleet-safe: a concurrent writer in ANOTHER PROCESS makes SQLite
                # wait instead of instantly raising 'database is locked' (design
                # review: default busy_timeout=0 crashed multi-agent hosts).
                self._conn.execute("PRAGMA busy_timeout=5000")
            except sqlite3.DatabaseError:
                pass
            self._with_write_retry(self._migrate_locked)
            self._conn.executescript(_SCHEMA)
            self.fts_available = True
            try:
                self._conn.executescript(_FTS_SCHEMA)
            except sqlite3.OperationalError:
                self.fts_available = False  # LIKE+Jaccard fallback takes over
            self._conn.commit()

    def close(self) -> None:
        self._conn.close()

    def _migrate_locked(self) -> None:
        """Upgrade a pre-stable-id database in place (schema v0 -> v1). v0 keyed
        mem_stats/mem_served by dedupe_key (a filesystem path), which orphaned
        trust on any rename/move and made 'delete the .db, lose nothing' false.

        ATOMIC and race-safe (verify F1): one BEGIN IMMEDIATE transaction with
        the idempotence guard re-checked inside it — a process kill or lock
        timeout rolls back to clean v0 instead of leaving a torn schema the
        column-presence guard would then skip forever. Orphaned v0 stats rows
        (a normal state: v0 kept stats across file deletes) are PRESERVED under
        their old key, never dropped (verify F0) — dropping them silently
        lifted quarantines."""
        cols = [r[1] for r in self._conn.execute("PRAGMA table_info(mem)").fetchall()]
        if not cols or "note_id" in cols:
            return
        self._conn.execute("BEGIN IMMEDIATE")
        try:
            cols = [r[1] for r in self._conn.execute("PRAGMA table_info(mem)").fetchall()]
            if not cols or "note_id" in cols:
                self._conn.rollback()
                return  # another process migrated while we waited for the lock
            self._conn.execute("ALTER TABLE mem ADD COLUMN note_id TEXT NOT NULL DEFAULT ''")
            self._conn.execute(
                "ALTER TABLE mem ADD COLUMN credibility TEXT NOT NULL DEFAULT 'operator'")
            rows = self._conn.execute("SELECT dedupe_key, kind, bank, title FROM mem").fetchall()
            for dk, kind, bank, title in rows:
                nid = dk if kind == "episode" else _derived_note_id(bank, kind, title)
                self._conn.execute("UPDATE mem SET note_id=? WHERE dedupe_key=?", (nid, dk))
            # Existing non-global notes predate provenance; mark them unverified —
            # safe-by-default (the operator promotes what they trust). Clear their
            # pinned flag too: a repo-credibility pinned row is excluded from the
            # pinned block AND from ranking, i.e. invisible (verify F4).
            self._conn.execute(
                "UPDATE mem SET credibility='repo', pinned=0"
                " WHERE bank != 'global' AND kind != 'episode'")
            self._conn.execute("""
                CREATE TABLE mem_stats_v1 (
                    note_id TEXT PRIMARY KEY,
                    served INTEGER NOT NULL DEFAULT 0,
                    positive REAL NOT NULL DEFAULT 0,
                    negative REAL NOT NULL DEFAULT 0,
                    quarantined INTEGER NOT NULL DEFAULT 0,
                    last_positive REAL
                )""")
            self._conn.execute("""
                INSERT INTO mem_stats_v1 (note_id, served, positive, negative, quarantined)
                SELECT COALESCE(m.note_id, s.dedupe_key), SUM(s.served), SUM(s.positive),
                       SUM(s.negative), MAX(s.quarantined)
                FROM mem_stats s LEFT JOIN mem m ON m.dedupe_key = s.dedupe_key
                GROUP BY COALESCE(m.note_id, s.dedupe_key)""")
            self._conn.execute("DROP TABLE mem_stats")
            self._conn.execute("ALTER TABLE mem_stats_v1 RENAME TO mem_stats")
            self._conn.execute("ALTER TABLE mem_served ADD COLUMN note_id TEXT NOT NULL DEFAULT ''")
            self._conn.execute("ALTER TABLE mem_served ADD COLUMN used INTEGER NOT NULL DEFAULT 0")
            self._conn.execute(
                "UPDATE mem_served SET note_id ="
                " COALESCE((SELECT m.note_id FROM mem m WHERE m.dedupe_key = mem_served.dedupe_key),"
                " mem_served.dedupe_key)")
            # v0 had no sidecars: export migrated quarantines to the canon
            # blocklist NOW, or a post-migration db delete still loses them.
            quarantined = [r[0] for r in self._conn.execute(
                "SELECT note_id FROM mem_stats WHERE quarantined=1").fetchall()]
            self._append_jsonl(self._blocklist_path(), [
                {"id": nid, "label": "migrated-v0", "on": True} for nid in quarantined])
            self._conn.commit()
        except BaseException:
            try:
                self._conn.rollback()
            except sqlite3.Error:
                pass
            raise

    def _with_write_retry(self, fn, attempts: int = 4):
        """Run a write closure, retrying on cross-process lock contention.
        busy_timeout makes SQLite wait 5s first; this catches the residual
        'database is locked' burst so fleet writers wait, never throw.
        EVERY failing attempt rolls back UNDER THE SAME LOCK HOLD (RLock) —
        a half-applied implicit transaction must never escape to be silently
        committed by the next unrelated writer (verify F17/F18)."""
        def _attempt():
            with self._lock:
                try:
                    return fn()
                except sqlite3.Error:
                    try:
                        if self._conn.in_transaction:
                            self._conn.rollback()
                    except sqlite3.Error:
                        pass
                    raise
        for i in range(attempts - 1):
            try:
                return _attempt()
            except sqlite3.OperationalError as e:
                if "locked" not in str(e).lower() and "busy" not in str(e).lower():
                    raise
                time.sleep(0.05 * (2 ** i))
        return _attempt()

    # -- canon sidecars ---------------------------------------------------------
    # The DB is disposable ONLY if operator state survives it. Trust counts,
    # quarantine, and repo-note promotion live in small files under the canon
    # root the operator owns (global_dir/.mneme/) — unreachable from any repo,
    # restored on reindex. Delete the .db and reindex: nothing is lost.
    def _sidecar_dir(self) -> Path:
        return self.global_dir / ".mneme"

    def _read_jsonl_last(self, path: Path) -> dict:
        """Last-entry-per-id view of a JSONL sidecar ({} if absent/corrupt)."""
        out: dict = {}
        try:
            for line in path.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue  # a torn concurrent append must not kill restore
                if isinstance(rec, dict) and rec.get("id"):
                    out[rec["id"]] = rec
        except OSError:
            pass
        return out

    def _append_jsonl(self, path: Path, recs: list) -> None:
        if not recs:
            return
        try:
            self._sidecar_dir().mkdir(parents=True, exist_ok=True)
            # One buffered write for the whole batch: concurrent appenders can
            # interleave BETWEEN lines but not tear one mid-line in practice
            # (and the reader skips undecodable lines regardless).
            payload = "".join(json.dumps(r, separators=(",", ":")) + "\n" for r in recs)
            with open(path, "a", encoding="utf-8") as f:
                f.write(payload)
        except OSError:
            pass  # sidecar is durability insurance; never fail the operation

    def _rewrite_atomic(self, path: Path, text: str) -> None:
        self._sidecar_dir().mkdir(parents=True, exist_ok=True)
        # Unique tmp name: two processes rewriting concurrently must not
        # clobber each other's tmp file mid-write (verify F19).
        tmp = path.with_suffix(f".{os.getpid()}.{uuid.uuid4().hex[:6]}.tmp")
        tmp.write_text(text, encoding="utf-8")
        os.replace(tmp, path)

    def _blocklist_path(self) -> Path:
        return self._sidecar_dir() / "blocklist.jsonl"

    def _trust_path(self) -> Path:
        return self._sidecar_dir() / "trust.jsonl"

    def _promoted_path(self) -> Path:
        return self._sidecar_dir() / "promoted.jsonl"

    def _append_trust_sidecar_locked(self, note_ids: list) -> None:
        """Snapshot the CURRENT cumulative counters for these ids to canon."""
        if not note_ids:
            return
        qs = ",".join("?" * len(note_ids))
        rows = self._conn.execute(
            f"SELECT note_id, served, positive, negative, last_positive"
            f" FROM mem_stats WHERE note_id IN ({qs})", note_ids).fetchall()
        self._append_jsonl(self._trust_path(), [
            {"id": r[0], "served": r[1], "pos": r[2], "neg": r[3], "lp": r[4]}
            for r in rows])

    def _restore_sidecars_locked(self) -> None:
        """Rebuild operator state after any DB loss. Trust restore is fill-only
        (a live DB's counters are always >= its own last sidecar snapshot);
        blocklist restore always ENFORCES quarantine and never silently lifts
        one (design review: a rebuild disarming a security blocklist was the
        single worst failure)."""
        trust = self._read_jsonl_last(self._trust_path())
        for nid, rec in trust.items():
            self._conn.execute(
                "INSERT OR IGNORE INTO mem_stats"
                " (note_id, served, positive, negative, last_positive)"
                " VALUES (?,?,?,?,?)",
                (nid, int(rec.get("served") or 0), float(rec.get("pos") or 0),
                 float(rec.get("neg") or 0), rec.get("lp")))
        block = self._read_jsonl_last(self._blocklist_path())
        for nid, rec in block.items():
            if rec.get("on", True):
                self._conn.execute(
                    "INSERT INTO mem_stats (note_id, quarantined) VALUES (?, 1)"
                    " ON CONFLICT(note_id) DO UPDATE SET quarantined = 1", (nid,))
        # Compact the append-only trust log to last-entry-per-id.
        if trust:
            lines = [json.dumps(r, separators=(",", ":")) for r in trust.values()]
            try:
                self._rewrite_atomic(self._trust_path(), "\n".join(lines) + "\n")
            except OSError:
                pass

    # -- banks ---------------------------------------------------------------
    def bank_of(self, repo: Path | None) -> str:
        """Stable per-repo memory bank: hash of the git remote URL, falling
        back to the resolved path. Banks stop cross-repo bleed — a fact about
        repo A must never surface in repo B's context."""
        if repo is None:
            return "global"
        key = str(Path(repo).resolve()).lower()
        if key in self._bank_cache:
            return self._bank_cache[key]
        ident = key
        try:
            proc = subprocess.run(["git", "remote", "get-url", "origin"], cwd=str(repo),
                                  capture_output=True, text=True, timeout=10)
            if proc.returncode == 0 and proc.stdout.strip():
                ident = proc.stdout.strip().lower()
        except (OSError, subprocess.SubprocessError):
            pass
        bank = hashlib.sha256(ident.encode("utf-8")).hexdigest()[:12]
        self._bank_cache[key] = bank
        return bank

    # -- reindex: files -> disposable mirror ----------------------------------
    @staticmethod
    def _file_key(path: Path) -> str:
        return "file:" + _norm_path(path)

    def _project_root(self, repo: Path) -> Path:
        return Path(repo) / self.cfg["project_subdir"]

    @staticmethod
    def _compose_note_id(bank: str, frontmatter_id: str) -> str:
        """Bank-scope a frontmatter id. Without this, a repo note declaring the
        same `id:` as an operator note shares its stats row — quarantining one
        silently vetoes the other, cross-bank (verify F8)."""
        return "f:" + hashlib.sha256(f"{bank}|{frontmatter_id}".encode("utf-8")).hexdigest()[:16]

    def _note_identity(self, n: Note, bank: str) -> str:
        if n.note_id:
            return self._compose_note_id(bank, n.note_id)
        return _derived_note_id(bank, n.kind, n.title)

    def _note_credibility(self, n: Note, bank: str, promoted: dict) -> str:
        """Provenance tier at ingest. Global-dir/extra-dir notes are operator
        canon. A note under a REPO's memory dir is 'repo' (unverified) unless
        its id is in the operator's promoted registry AND its content hash
        still matches — an attacker editing a promoted file demotes it back
        (design review: repo .md files were laundered into trusted canon)."""
        if bank == "global":
            return "operator"
        rec = promoted.get(self._note_identity(n, bank))
        if rec and rec.get("sha") == n.raw_sha:
            return "operator"
        return "repo"

    def reindex(self, repo: Path | None = None, force: bool = False) -> dict:
        """Mirror canon note files into the DB. mtime-based, idempotent, and
        prunes rows whose file vanished. Episodes (DB-canon) are untouched.
        Ends by restoring canon sidecars (trust/blocklist), so delete-the-db
        + reindex loses neither trust nor quarantine."""
        roots: list[tuple[Path, str]] = [(self.global_dir, "global")]
        roots += [(d, "global") for d in self.extra_dirs]
        if repo is not None:
            roots.append((self._project_root(repo), self.bank_of(repo)))
        promoted = self._read_jsonl_last(self._promoted_path())
        # _with_write_retry: a lock-timeout mid-scan must roll back rather than
        # leak a half-applied implicit transaction (verify F17); the mtime scan
        # is idempotent so the retry re-runs cleanly.
        return self._with_write_retry(lambda: self._reindex_locked(roots, force, promoted))

    def _reindex_locked(self, roots: list, force: bool, promoted: dict) -> dict:
        added = updated = pruned = 0
        # (new_key, superseded slug, bank, superseder pinned, superseder cred)
        supersedes: list[tuple[str, str, str, bool, str]] = []
        with self._lock:
            for root, bank in roots:
                # Warm-path economy (verify F12-scale): stat BEFORE read — an
                # unchanged corpus does zero full file reads — and ONE batched
                # mtime SELECT per root instead of one query per file
                # (measured: warm reindex of 20k files was 6.25s, mostly
                # read_text on files the mtime gate was about to skip).
                prefix = "file:" + _norm_path(root)
                known = dict(self._conn.execute(
                    "SELECT dedupe_key, file_mtime FROM mem WHERE dedupe_key LIKE ? ESCAPE '\\'",
                    (_like_escape(prefix) + "/%",)).fetchall())
                seen_keys = set()
                if root.is_dir():
                    files = sorted(root.rglob("*.md"))
                else:
                    files = []
                for f in files:
                    key = self._file_key(f)
                    try:
                        mtime = f.stat().st_mtime
                    except OSError:
                        continue
                    existed = key in known
                    if existed and not force and known[key] == mtime:
                        seen_keys.add(key)  # unchanged: proven parseable at index time
                        continue
                    n = parse_note(f)
                    if n is None:
                        continue  # empty/corrupt: NOT seen, so a stale row prunes
                    seen_keys.add(key)
                    cred = self._note_credibility(n, bank, promoted)
                    self._upsert_locked(key, n.kind, bank, n.title, n.body, n.keywords,
                                        n.tags, str(n.path), n.pinned, n.mtime, n.valid_at,
                                        note_id=self._note_identity(n, bank),
                                        credibility=cred)
                    if n.supersedes:
                        supersedes.append((key, n.supersedes, bank, n.pinned, cred))
                    added += 0 if existed else 1
                    updated += 1 if existed else 0
                # Prune rows whose file vanished from THIS root — `known` was
                # gathered with the escaped, '/'-bounded LIKE prefix (audit C4),
                # so it is exactly this root's rows.
                for key in known:
                    if key not in seen_keys:
                        self._delete_locked(key)
                        pruned += 1
            for new_key, target, bank, sup_pinned, sup_cred in supersedes:
                self._supersede_locked(new_key, target, bank,
                                       sup_pinned=sup_pinned, sup_cred=sup_cred)
            self._restore_sidecars_locked()
            self._conn.commit()
        return {"added": added, "updated": updated, "pruned": pruned,
                "fts": self.fts_available}

    def _upsert_locked(self, key: str, kind: str, bank: str, title: str, body: str,
                       keywords: str, tags: str, source: str, pinned: bool,
                       file_mtime: float | None, valid_at: float, *,
                       note_id: str = "", credibility: str = "operator") -> None:
        now = time.time()
        old = self._conn.execute("SELECT id, created_at FROM mem WHERE dedupe_key=?", (key,)).fetchone()
        self._invalidate_cooc()
        mem_id = old[0] if old else uuid.uuid4().hex[:12]
        created = old[1] if old else now
        note_id = note_id or _derived_note_id(bank, kind, title)
        if credibility != "operator":
            pinned = False  # pinning is operator authority; a repo can't self-pin
        self._conn.execute("DELETE FROM mem WHERE dedupe_key=?", (key,))
        self._conn.execute(
            "INSERT INTO mem (id, dedupe_key, note_id, kind, bank, title, body, keywords,"
            " tags, source, pinned, credibility, file_mtime, created_at, valid_at,"
            " invalid_at, superseded_by)"
            " VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,NULL,NULL)",
            (mem_id, key, note_id, kind, bank, title, body, keywords, tags, source,
             int(pinned), credibility, file_mtime, created, valid_at))
        if self.fts_available:
            self._conn.execute("DELETE FROM mem_fts WHERE dedupe_key=?", (key,))
            self._conn.execute(
                "INSERT INTO mem_fts (dedupe_key, title, body, keywords, tags) VALUES (?,?,?,?,?)",
                (key, title, body, keywords, tags))

    def _invalidate_cooc(self) -> None:
        self._cooc = None
        self._df_cache.clear()  # writes change document frequencies too

    def _delete_locked(self, key: str) -> None:
        # mem_stats is deliberately NOT cleaned here: stats are keyed by the
        # STABLE note_id, and a deleted file is usually a rename/move about to
        # reappear under a new path — its trust must survive the gap.
        self._invalidate_cooc()
        self._conn.execute("DELETE FROM mem WHERE dedupe_key=?", (key,))
        if self.fts_available:
            self._conn.execute("DELETE FROM mem_fts WHERE dedupe_key=?", (key,))

    def _supersede_locked(self, new_key: str, target: str, bank: str, *,
                          sup_pinned: bool = False, sup_cred: str = "operator") -> None:
        """Mark older rows invalid instead of deleting them (bi-temporal).
        Target matches a note filename stem or a slugified title — SCOPED TO
        THE SAME BANK (audit C1). Supersession needs AUTHORITY (design review):
        an unverified repo note can never invalidate operator canon, and only
        a pinned note may supersede a pinned one."""
        slug = slugify(target) or target.lower()
        now = time.time()
        rows = self._conn.execute(
            "SELECT dedupe_key, title, pinned, credibility FROM mem WHERE invalid_at IS NULL"
            " AND bank = ? AND dedupe_key != ?",
            (bank, new_key)).fetchall()
        for key, title, tgt_pinned, tgt_cred in rows:
            stem = key.rsplit("/", 1)[-1].removesuffix(".md")
            if stem != slug and slugify(title) != slug:
                continue
            if tgt_cred == "operator" and sup_cred != "operator":
                continue  # no authority: unverified cannot displace canon
            if tgt_pinned and not sup_pinned:
                continue  # pinned canon yields only to a pinned successor
            self._conn.execute(
                "UPDATE mem SET invalid_at=?, superseded_by=? WHERE dedupe_key=?",
                (now, new_key, key))

    # -- episodes: DB-canon, append-only --------------------------------------
    def add_episode(self, run_id: str, goal: str, status: str, detail: str,
                    repo: Path | None) -> None:
        key = "run:" + run_id

        def _do() -> int:
            with self._lock:
                exists = self._conn.execute("SELECT 1 FROM mem WHERE dedupe_key=?", (key,)).fetchone()
                if exists:
                    return -1  # idempotent: one episode per run, ever
                now = time.time()
                title = f"[{status}] {goal[:140]}"
                self._conn.execute(
                    "INSERT INTO mem (id, dedupe_key, note_id, kind, bank, title, body,"
                    " keywords, tags, source, pinned, credibility, file_mtime, created_at,"
                    " valid_at, invalid_at, superseded_by)"
                    " VALUES (?,?,?,?,?,?,?,?,?,?,0,'operator',NULL,?,?,NULL,NULL)",
                    (uuid.uuid4().hex[:12], key, key, "episode", self.bank_of(repo), title,
                     detail[:2000], "", status, "run:" + run_id, now, now))
                if self.fts_available:
                    self._conn.execute(
                        "INSERT INTO mem_fts (dedupe_key, title, body, keywords, tags) VALUES (?,?,?,?,?)",
                        (key, title, detail[:2000], "", status))
                self._conn.commit()
                # BANK-scoped trigger: max_episodes is per bank, and counting
                # globally made every add_episode past ~cap+slack TOTAL run a
                # full cross-bank compact scan (design review F: unit mismatch).
                return self._conn.execute(
                    "SELECT COUNT(*) FROM mem WHERE kind='episode' AND bank=?",
                    (self.bank_of(repo),)).fetchone()[0]

        episode_count = self._with_write_retry(_do)
        # Outside the lock (compact re-acquires it): amortized self-maintenance.
        if (episode_count > 0 and self.cfg.get("auto_compact", True)
                and episode_count > int(self.cfg["max_episodes"]) + AUTO_COMPACT_SLACK):
            self.compact()

    # -- retrieval -------------------------------------------------------------
    def _trust(self, stats_row) -> float:
        """Laplace-smoothed success ratio over proven outcomes. No outcomes yet
        -> 0.5. A memory that keeps riding failing runs sinks; one that keeps
        appearing in proven-done runs rises. Auditable: the counts are stored."""
        if stats_row is None:
            return 0.5
        _, positive, negative = stats_row
        return (positive + 1) / (positive + negative + 2)

    def _decay(self, kind: str, valid_at: float) -> float:
        if kind != "episode":
            return 1.0  # lessons/facts/preferences die by supersession, not by age
        age_days = max(0.0, (time.time() - valid_at) / 86400)
        return 0.5 ** (age_days / max(1, self.cfg["episode_half_life_days"]))

    def _stats_map(self, keys: list[str]) -> dict:
        """Stats rows for these mem rows, keyed by dedupe_key for the callers
        but JOINED through the stable note_id (stats survive renames/moves)."""
        if not keys:
            return {}
        qs = ",".join("?" * len(keys))
        rows = self._conn.execute(
            f"SELECT m.dedupe_key, s.served, s.positive, s.negative, s.quarantined"
            f" FROM mem m JOIN mem_stats s ON s.note_id = m.note_id"
            f" WHERE m.dedupe_key IN ({qs})", keys).fetchall()
        return {r[0]: r for r in rows}

    def _cooccurrence(self, topn: int = 3) -> dict:
        """Deterministic query expansion table (the stdlib stand-in for
        semantic depth): for each term, the terms that most often co-occur
        with it across live rows. Built lazily, invalidated by writes.
        Mechanism selected by the scale bake-off: bm25+expand was the only
        variant that held precision@3 at 1000 entries, at the lowest latency."""
        if self._cooc is not None:
            return self._cooc
        from collections import Counter
        # Trusted, non-vetoed rows only: a quarantined or unverified note must
        # not keep steering retrieval through its vocabulary (verify F9).
        rows = self._conn.execute(
            "SELECT m.title, m.keywords, m.tags, substr(m.body, 1, 400) FROM mem m"
            " LEFT JOIN mem_stats s ON s.note_id = m.note_id"
            " WHERE m.invalid_at IS NULL AND COALESCE(s.quarantined, 0) = 0"
            " AND m.credibility = 'operator'").fetchall()
        co: dict = {}
        for parts in rows:
            terms = set(_tokens(" ".join(x or "" for x in parts), cap=40))
            for a in terms:
                c = co.setdefault(a, Counter())
                for b in terms:
                    if a != b:
                        c[b] += 1
        self._cooc = {a: [w for w, _ in c.most_common(topn)] for a, c in co.items()}
        return self._cooc

    def _expand_tokens(self, toks: list) -> list:
        try:
            co = self._cooccurrence()
        except Exception:
            return toks
        out = list(toks)
        for t in toks:
            out += co.get(t, [])
        return list(dict.fromkeys(out))[:48]

    def _prune_common_terms(self, toks: list) -> list:
        """df-aware pruning (measured 3-64x recall speedup at 100k rows): FTS5
        evaluates OR by unioning every term's posting list and bm25-scoring the
        union, so ONE common query word ('note', 'file') makes recall O(corpus).
        Drop terms matching more than DF_PRUNE_RATIO of rows — the rare terms
        carry the ranking anyway. Off below df_prune_min_rows; keeps at least
        the DF_PRUNE_KEEP_MIN rarest terms so a query can never prune to less."""
        if not self.fts_available or len(toks) < 2:
            return toks
        total = self._conn.execute("SELECT COUNT(*) FROM mem").fetchone()[0]
        if total < int(self.cfg.get("df_prune_min_rows", DF_PRUNE_MIN_ROWS)):
            return toks
        cutoff = total * DF_PRUNE_RATIO
        dfs = []
        for t in toks:
            df = self._df_cache.get(t)
            if df is None:
                try:
                    df = self._conn.execute(
                        "SELECT COUNT(*) FROM mem_fts WHERE mem_fts MATCH ?",
                        (f'"{t}"',)).fetchone()[0]
                except sqlite3.OperationalError:
                    df = 0  # unparseable token: harmless, keep it
                self._df_cache[t] = df
            dfs.append((df, t))
        kept = [t for df, t in dfs if df <= cutoff]
        if kept:
            # Any rare term carries the query alone — padding back common terms
            # would reintroduce the O(corpus) union ('note 54321' must query
            # just '54321', the whole point of the prototype's 64x win).
            return kept
        dfs.sort()  # ALL terms common: fall back to the KEEP_MIN rarest
        return [t for _df, t in dfs[:DF_PRUNE_KEEP_MIN]]

    def _candidates(self, query: str, banks: tuple, limit: int = 64) -> list[dict]:
        """Lexical candidates: FTS5/BM25 (porter-stemmed) when available, else
        LIKE token matching. Returns rows with a normalized lexical score."""
        toks = _tokens(query)
        if toks and self.fts_available:
            toks = self._prune_common_terms(toks)
        qs_banks = ",".join("?" * len(banks))
        rows: list[tuple] = []
        # Quarantine is filtered INSIDE the candidate query (audit L1): filtering
        # it post-LIMIT let a top-ranked quarantined row eat a slot and hide a
        # legitimate row ranked past `limit`. LEFT JOIN + quarantined=0 so LIMIT
        # applies only to eligible rows.
        _FTS_SQL = (
            "SELECT m.dedupe_key, m.kind, m.bank, m.title, m.body, m.keywords, m.tags,"
            " m.pinned, m.valid_at, m.credibility, bm25(mem_fts) AS r"
            " FROM mem_fts JOIN mem m ON m.dedupe_key = mem_fts.dedupe_key"
            " LEFT JOIN mem_stats s ON s.note_id = m.note_id"
            f" WHERE mem_fts MATCH ? AND m.bank IN ({qs_banks}) AND m.invalid_at IS NULL"
            " AND COALESCE(s.quarantined, 0) = 0"
            " ORDER BY r LIMIT ?")
        if toks and self.fts_available:
            match = " OR ".join(f'"{t}"' for t in toks)
            rows = self._conn.execute(_FTS_SQL, (match, *banks, limit)).fetchall()
            if not rows:
                # Empty-result rescue (bake-off verdict): co-occurrence expansion
                # dilutes ranked results on the hot path, but when the strict
                # query finds NOTHING a wider net can only help.
                wide = self._expand_tokens(toks)
                if len(wide) > len(toks):
                    match = " OR ".join(f'"{t}"' for t in wide)
                    rows = self._conn.execute(_FTS_SQL, (match, *banks, limit)).fetchall()
        elif toks:
            like = " OR ".join(["m.title LIKE ? OR m.body LIKE ? OR m.keywords LIKE ? OR m.tags LIKE ?"] * min(len(toks), 8))
            params: list = []
            for t in toks[:8]:
                # tokens are [A-Za-z0-9_] — a stray '_' over-matches slightly but
                # hit-count scoring self-corrects, so no ESCAPE needed here (this
                # is a fuzzy candidate gather, not a destructive prefix match).
                params += [f"%{t}%"] * 4
            found = self._conn.execute(
                f"SELECT m.dedupe_key, m.kind, m.bank, m.title, m.body, m.keywords, m.tags,"
                f" m.pinned, m.valid_at, m.credibility"
                f" FROM mem m LEFT JOIN mem_stats s ON s.note_id = m.note_id"
                f" WHERE ({like}) AND m.bank IN ({qs_banks}) AND m.invalid_at IS NULL"
                f" AND COALESCE(s.quarantined, 0) = 0 LIMIT 400",
                (*params, *banks)).fetchall()
            scored = []
            tokset = set(toks)
            for f in found:
                # Score against the SAME full body the SQL LIKE matched (audit
                # L2): truncating to 2000 here made a token past char 2000 count
                # as a miss, diverging from what the query actually matched.
                text = (f[3] + " " + f[4] + " " + f[5] + " " + f[6]).lower()
                hits = sum(1 for t in tokset if t in text)
                scored.append((*f, -float(hits)))  # negative = better, like bm25
            scored.sort(key=lambda x: (x[-1], x[0]))
            rows = scored[:limit]
        if not rows:
            return []
        # normalize lexical score to 0..1 (bm25 returns negative-better)
        vals = [r[-1] for r in rows]
        lo, hi = min(vals), max(vals)
        span = (hi - lo) or 1.0
        out = []
        for r in rows:
            out.append({"key": r[0], "kind": r[1], "bank": r[2], "title": r[3], "body": r[4],
                        "keywords": r[5], "tags": r[6], "pinned": bool(r[7]), "valid_at": r[8],
                        "credibility": r[9], "lex": (hi - r[-1]) / span})
        return out

    def _weights(self) -> tuple[float, float]:
        """(lexical, jaccard) blend for this store's retrieval mode."""
        if self.fts_available:
            return LEX_WEIGHT, JAC_WEIGHT
        return FALLBACK_LEX_WEIGHT, FALLBACK_JAC_WEIGHT

    def retrieve(self, query: str, repo: Path | None = None, limit: int = 64) -> list[dict]:
        """Ranked memories: lexical(BM25) * trust * decay (Jaccard re-rank only
        on the FTS-less fallback — see the weight constants for the evidence).
        Quarantined and superseded rows never surface; unverified (repo-sourced,
        unpromoted) notes are down-weighted so they can't outrank trusted canon."""
        banks = ("global",) if repo is None else ("global", self.bank_of(repo))
        with self._lock:
            cands = self._candidates(query, banks, limit)
            stats = self._stats_map([c["key"] for c in cands])
        lex_w, jac_w = self._weights()
        qgrams = _trigrams(query) if jac_w else set()
        out = []
        for c in cands:
            s = stats.get(c["key"])
            if s is not None and s[4]:
                continue  # quarantined
            trust = self._trust((s[1], s[2], s[3]) if s else None)
            jac = _jaccard(qgrams, _trigrams(
                c["title"] + " " + c["keywords"] + " " + c["tags"])) if jac_w else 0.0
            cred_w = 1.0 if c.get("credibility", "operator") == "operator" else UNVERIFIED_WEIGHT
            c["trust"] = trust
            c["score"] = (lex_w * c["lex"] + jac_w * jac) * trust * cred_w \
                * self._decay(c["kind"], c["valid_at"])
            out.append(c)
        # Credibility tier FIRST, score second: a down-weight alone cannot
        # guarantee 'unverified never outranks canon' when the lexical gap is
        # large — the tier sort makes the promise structural.
        out.sort(key=lambda x: (0 if x.get("credibility", "operator") == "operator" else 1,
                                -x["score"], x["key"]))
        return out

    def recall(self, query: str, repo: Path | None = None, top_k: int | None = None) -> list[dict]:
        """Top-k retrieve(). May include credibility='repo' rows (always sorted
        AFTER operator canon): a host injecting recall output into a model
        context MUST check each hit's `credibility` field and frame unverified
        rows as verify-before-trusting, mirroring index_block's header."""
        k = top_k or self.cfg["recall_top_k"]
        return self.retrieve(query, repo, limit=max(32, k * 4))[:k]

    def _repo_hint(self, repo: Path | None) -> str:
        """Query boost from a host's repo snapshot (languages and frameworks):
        'python pytest' pulls framework-specific memories in even when the goal
        never names the stack. This is an OPTIONAL host integration — it reads
        <repo>/<project_subdir-parent>/repo_snapshot.json if present and
        fails safe (empty) everywhere else, so non-Ktisis hosts are unaffected."""
        if repo is None:
            return ""
        snap = Path(repo) / self.cfg["project_subdir"].split("/")[0] / "repo_snapshot.json"
        try:
            data = json.loads(snap.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError, ValueError):
            return ""
        parts = []
        for field_name in ("languages", "frameworks"):
            v = data.get(field_name) or []
            items = list(v.keys()) if isinstance(v, dict) else [str(x) for x in v]
            parts += items[:5]
        return " ".join(parts)

    # -- the guaranteed read path ----------------------------------------------
    def index_block(self, repo: Path | None, query: str, run_id: str | None = None,
                    budget: int | None = None) -> IndexBlock:
        """The always-injected memory block for task start. Pinned rows are
        unconditional; the rest is query-ranked with per-kind budget shares so
        episodes can never starve lessons. Deterministic ordering."""
        budget = budget or self.cfg["index_budget_chars"]
        query = (query + " " + self._repo_hint(repo)).strip()
        banks = ("global",) if repo is None else ("global", self.bank_of(repo))
        qs_banks = ",".join("?" * len(banks))
        with self._lock:
            # credibility='operator' is belt-and-braces: repo notes can't self-pin
            # (downgraded at ingest), so this only guards pre-migration rows.
            pinned_rows = self._conn.execute(
                f"SELECT dedupe_key, kind, title, body, valid_at FROM mem"
                f" WHERE pinned=1 AND invalid_at IS NULL AND bank IN ({qs_banks})"
                f" AND credibility='operator'"
                f" ORDER BY kind, title", banks).fetchall()
            pinned_stats = self._stats_map([r[0] for r in pinned_rows])
        pinned = [{"key": r[0], "kind": r[1], "title": r[2], "body": r[3], "valid_at": r[4]}
                  for r in pinned_rows if not (pinned_stats.get(r[0]) or (0, 0, 0, 0, 0))[4]]
        all_ranked = [c for c in self.retrieve(query, repo) if not c["pinned"]]
        ranked = [c for c in all_ranked if c.get("credibility", "operator") == "operator"]
        unverified = [c for c in all_ranked if c.get("credibility", "operator") != "operator"]
        if not ranked and not query.strip():
            ranked = self._recent_fallback(banks)

        sections: dict[str, list] = {k: [] for k in _KIND_SHARE}
        used = 0
        chosen: list[dict] = []
        for p in pinned:  # pinned first, exempt from ranking, still budgeted
            line = self._render_line(p)
            if used + len(line) + 1 > budget:
                break
            sections[p["kind"]].append(line)
            chosen.append(p)
            used += len(line) + 1
        remaining = budget - used
        shares = {k: int(remaining * v) for k, v in _KIND_SHARE.items()}
        leftovers: list[dict] = []
        for c in ranked:
            line = self._render_line(c)
            if len(line) + 1 <= shares.get(c["kind"], 0):
                sections[c["kind"]].append(line)
                chosen.append(c)
                shares[c["kind"]] -= len(line) + 1
                used += len(line) + 1
            else:
                leftovers.append(c)
        for c in leftovers:  # spill unused share to whatever ranked highest
            line = self._render_line(c)
            if used + len(line) + 1 > budget:
                continue
            sections[c["kind"]].append(line)
            chosen.append(c)
            used += len(line) + 1
        # Precise assembly (audit budget bug): the old code counted only line
        # lengths in `used` but the emitted text also carries section HEADERS and
        # "\n\n" separators, then hard-sliced with [:budget] — so trailing lines
        # were counted as served yet cut from the text, and trust later moved for
        # memories the agent never saw. Here we account for exact rendered length
        # and take served keys ONLY from lines that actually fit.
        picked: dict[str, list] = {k: [] for k in ("lesson", "fact", "preference", "episode")}
        for c in chosen:
            picked[c["kind"]].append((self._render_line(c), c["key"]))
        segments = [(_SECTION_HEADERS[k], picked[k])
                    for k in ("lesson", "fact", "preference", "episode") if picked[k]]
        if unverified:
            # Repo-suggested notes surface LAST, capped, and clearly labelled —
            # provenance must reach the model, or a repo-dropped .md is laundered
            # into the same authority as operator canon (design review).
            segments.append((_UNVERIFIED_HEADER,
                             [(self._render_line(c), c["key"])
                              for c in unverified[:_UNVERIFIED_MAX_LINES]]))
        parts, keys, total = [], [], 0
        for header, items in segments:
            sep = 2 if parts else 0  # "\n\n" before this section
            seg_lines, seg_len, seg_keys = [], len(header), []
            for line, key in items:
                if total + sep + seg_len + 1 + len(line) > budget:
                    break
                seg_lines.append(line)
                seg_len += 1 + len(line)  # "\n" + line
                seg_keys.append(key)
            if seg_lines:
                parts.append(header + "\n" + "\n".join(seg_lines))
                total += sep + seg_len
                keys.extend(seg_keys)
        text = "\n\n".join(parts)
        if run_id and keys:
            self.record_served(run_id, keys, "index")
        return IndexBlock(text=text, keys=keys)

    def _recent_fallback(self, banks: tuple) -> list[dict]:
        """No query signal: serve the most trusted, most recent memories.
        Trusted canon only — a blind fallback must never inject unverified."""
        qs = ",".join("?" * len(banks))
        with self._lock:
            rows = self._conn.execute(
                f"SELECT dedupe_key, kind, title, body, valid_at, pinned FROM mem"
                f" WHERE invalid_at IS NULL AND bank IN ({qs}) AND credibility='operator'"
                f" ORDER BY valid_at DESC LIMIT 64", banks).fetchall()
            stats = self._stats_map([r[0] for r in rows])
        out = []
        for r in rows:
            s = stats.get(r[0])
            if s is not None and s[4]:
                continue
            trust = self._trust((s[1], s[2], s[3]) if s else None)
            out.append({"key": r[0], "kind": r[1], "title": r[2], "body": r[3],
                        "valid_at": r[4], "pinned": bool(r[5]),
                        "score": trust * self._decay(r[1], r[4]), "trust": trust})
        out.sort(key=lambda x: (-x["score"], x["key"]))
        return out

    @staticmethod
    def _render_line(row: dict) -> str:
        title = row["title"].strip()
        first = (row.get("body") or "").strip().partition("\n")[0].strip()
        if row.get("kind") == "episode":
            day = time.strftime("%Y-%m-%d", time.localtime(row.get("valid_at") or 0))
            return f"- {title} ({day})"[:_LINE_CAP]
        line = f"- {title}"
        if first and first.lower() != title.lower() and len(line) < 120:
            line += f" — {first}"
        return line[:_LINE_CAP]

    # -- proof-coupled trust -----------------------------------------------------
    def record_served(self, run_id: str, keys: list[str], tier: str) -> None:
        def _do() -> None:
            now = time.time()
            with self._lock:
                qs = ",".join("?" * len(keys))
                nid_map = dict(self._conn.execute(
                    f"SELECT dedupe_key, note_id FROM mem WHERE dedupe_key IN ({qs})",
                    keys).fetchall()) if keys else {}
                counted: set[str] = set()
                for key in keys:
                    nid = nid_map.get(key)
                    self._conn.execute(
                        "INSERT INTO mem_served (run_id, dedupe_key, note_id, tier, ts)"
                        " VALUES (?,?,?,?,?)", (run_id, key, nid or key, tier, now))
                    # served counts move only for keys that resolve to a real
                    # mem row, once per distinct note id — an arbitrary string
                    # must not fabricate a stats row (verify F15/F13).
                    if nid and nid not in counted:
                        counted.add(nid)
                        self._conn.execute(
                            "INSERT INTO mem_stats (note_id, served) VALUES (?, 1)"
                            " ON CONFLICT(note_id) DO UPDATE SET served = served + 1", (nid,))
                self._conn.commit()
        self._with_write_retry(_do)

    def served_keys(self, run_id: str, tier: str | None = None) -> list[str]:
        """Keys served into a run, optionally filtered by tier. Hosts use this
        to build `used_keys` from honest evidence — e.g. tier='recall' rows
        are memories the model explicitly asked for."""
        with self._lock:
            if tier is None:
                rows = self._conn.execute(
                    "SELECT DISTINCT dedupe_key FROM mem_served WHERE run_id=?",
                    (run_id,)).fetchall()
            else:
                rows = self._conn.execute(
                    "SELECT DISTINCT dedupe_key FROM mem_served WHERE run_id=? AND tier=?",
                    (run_id, tier)).fetchall()
        return [r[0] for r in rows]

    def apply_outcome(self, run_id: str, status: str, used_keys: list[str] | None = None) -> int:
        """Move trust from the PROVEN outcome of a run. Neutral statuses change
        nothing. Trust measures USEFULNESS, not retrievability (design review:
        crediting the whole served set was a rich-get-richer loop):

        - a memory in `used_keys` (the host says it was actually consulted or
          acted on) moves a FULL point, positive or negative;
        - with `used_keys` given, a memory served-but-not-used takes a WEAK
          negative on a win (retrieved constantly without ever helping is
          evidence against) and nothing on a failure (it takes no blame);
        - without `used_keys` a WIN is NEUTRAL — no usage signal means a win
          says nothing about any served note (a weak negative here would erode
          all trust monotonically and invert ranking, verify F12) — while a
          FAILURE still nudges every served memory weakly down;
        - pinned notes never auto-credit (always-injected must not launder
          trust); unverified repo notes never move until promoted; quarantined
          notes are frozen (a security veto stops trust in both directions);
          keys with no live mem row have no authority and move nothing.

        EXACTLY-ONCE per run_id, cross-process: the run_outcome PK insert is
        the atomic guard inside BEGIN IMMEDIATE. Credit lands once per NOTE
        (stable id), however many paths served it. Returns notes touched."""
        if status in POSITIVE_STATUSES:
            positive = True
        elif status in NEGATIVE_STATUSES:
            positive = False
        else:
            return 0
        used = {k for k in (used_keys or [])}

        def _do() -> int:
            with self._lock:
                self._conn.execute("BEGIN IMMEDIATE")
                claim = self._conn.execute(
                    "INSERT OR IGNORE INTO run_outcome (run_id, status, ts) VALUES (?,?,?)",
                    (run_id, status, time.time()))
                if claim.rowcount == 0:
                    self._conn.rollback()
                    return 0  # another process/call already applied this run
                # Raw m.credibility, no COALESCE: a served key whose mem row is
                # gone (pruned, never existed, injected) must read as
                # no-authority, not default to trusted canon (verify F11).
                rows = self._conn.execute(
                    "SELECT sv.dedupe_key, sv.note_id, m.pinned, m.credibility,"
                    " COALESCE(st.quarantined, 0)"
                    " FROM mem_served sv LEFT JOIN mem m ON m.dedupe_key = sv.dedupe_key"
                    " LEFT JOIN mem_stats st ON st.note_id = sv.note_id"
                    " WHERE sv.run_id=?", (run_id,)).fetchall()
                # Aggregate per stable note id: one credit per NOTE per run,
                # however many files/serve events carried it (verify F13).
                groups: dict[str, dict] = {}
                for dk, nid, pinned, cred, quar in rows:
                    nid = nid or dk
                    g = groups.setdefault(nid, {"used_dks": [], "pinned": False,
                                                "authorized": True, "quar": False})
                    if cred != "operator":  # includes None: mem row missing
                        g["authorized"] = False
                    if pinned:
                        g["pinned"] = True
                    if quar:
                        g["quar"] = True
                    if dk in used or nid in used:
                        g["used_dks"].append(dk)
                now = time.time()
                touched = []
                for nid, g in groups.items():
                    is_used = bool(g["used_dks"])
                    if not g["authorized"]:
                        continue  # unverified / no live row: no trust authority
                    if g["quar"]:
                        continue  # security veto freezes trust both directions
                    if g["pinned"] and not is_used:
                        continue  # pinned = always eligible, never auto-credited
                    if is_used:
                        dpos, dneg = (1.0, 0.0) if positive else (0.0, 1.0)
                    elif used_keys is None:
                        if positive:
                            continue  # no usage signal: a win proves nothing
                        dpos, dneg = 0.0, SERVED_UNUSED_NEG
                    elif positive:
                        dpos, dneg = 0.0, SERVED_UNUSED_NEG
                    else:
                        continue  # failed run, host says this one wasn't used
                    self._conn.execute(
                        "INSERT INTO mem_stats (note_id, positive, negative, last_positive)"
                        " VALUES (?,?,?,?)"
                        " ON CONFLICT(note_id) DO UPDATE SET"
                        " positive = positive + ?, negative = negative + ?,"
                        " last_positive = CASE WHEN ? > 0 THEN ? ELSE last_positive END",
                        (nid, dpos, dneg, now if dpos > 0 else None,
                         dpos, dneg, dpos, now))
                    for dk in g["used_dks"]:
                        self._conn.execute(
                            "UPDATE mem_served SET used=1 WHERE run_id=? AND dedupe_key=?",
                            (run_id, dk))
                    touched.append(nid)
                # Sidecar BEFORE commit (verify F15): a crash between the two
                # leaves canon slightly ahead with the transaction rolled back
                # and the run unclaimed — retryable, instead of silent loss.
                self._append_trust_sidecar_locked(touched)
                self._conn.commit()
                return len(touched)
        return self._with_write_retry(_do)

    def _resolve_rows(self, key_or_slug: str) -> list[tuple]:
        """(dedupe_key, note_id) for an exact key, a note_id, a filename stem,
        or a TITLE slug (what the operator actually sees in the index block —
        repo files often carry a filename unrelated to their title). The slug
        LIKE is ESCAPE'd (audit M1) and matches ALL same-stem rows (across
        banks) so an operator action is total, not one arbitrary row."""
        rows = self._conn.execute(
            "SELECT dedupe_key, note_id FROM mem WHERE dedupe_key=? OR note_id=?"
            " OR dedupe_key LIKE '%/' || ? || '.md' ESCAPE '\\'",
            (key_or_slug, key_or_slug, _like_escape(key_or_slug))).fetchall()
        if rows:
            return rows
        want = slugify(key_or_slug) or key_or_slug.lower()
        return [(dk, nid) for dk, nid, title in self._conn.execute(
            "SELECT dedupe_key, note_id, title FROM mem WHERE kind != 'episode'"
        ).fetchall() if slugify(title) == want]

    def quarantine(self, key_or_slug: str, on: bool = True) -> bool:
        """Operator veto: a quarantined memory never surfaces anywhere except
        `mneme show`. The note file is untouched — this is a store-level gate.
        The veto is SECURITY state, so it is mirrored to the canon blocklist
        sidecar and survives any DB rebuild (design review: delete-and-reindex
        silently lifted every quarantine)."""
        def _do() -> bool:
            with self._lock:
                rows = self._resolve_rows(key_or_slug)
                if not rows:
                    return False
                for _dk, nid in rows:
                    self._conn.execute(
                        "INSERT INTO mem_stats (note_id, quarantined) VALUES (?, ?)"
                        " ON CONFLICT(note_id) DO UPDATE SET quarantined = ?",
                        (nid, int(on), int(on)))
                self._invalidate_cooc()  # vetoed vocabulary leaves the expansion table
                self._conn.commit()
                self._append_jsonl(self._blocklist_path(), [
                    {"id": nid, "label": key_or_slug, "on": bool(on)}
                    for _dk, nid in rows])
            return True
        return self._with_write_retry(_do)

    def promote(self, key_or_slug: str) -> int:
        """Operator promotion: mark a repo-suggested note as trusted canon.
        The promoted registry stores the note's CONTENT HASH — if the file is
        later edited in the repo, the hash no longer matches and the note
        demotes back to unverified on the next reindex (an attacker cannot
        ride an old promotion). Returns rows promoted."""
        def _do() -> int:
            promoted = 0
            recs = []
            with self._lock:
                rows = []
                for dk, nid in self._resolve_rows(key_or_slug):
                    r = self._conn.execute(
                        "SELECT source FROM mem WHERE dedupe_key=? AND credibility != 'operator'",
                        (dk,)).fetchone()
                    if r is not None:
                        rows.append((dk, nid, r[0]))
                for dk, nid, source in rows:
                    try:
                        sha = hashlib.sha256(
                            Path(source).read_text(encoding="utf-8", errors="ignore")
                            .strip().encode("utf-8")).hexdigest()[:16]
                    except OSError:
                        continue  # file gone: nothing to trust
                    self._conn.execute(
                        "UPDATE mem SET credibility='operator' WHERE dedupe_key=?", (dk,))
                    recs.append({"id": nid, "sha": sha, "label": key_or_slug})
                    promoted += 1
                if promoted:
                    self._invalidate_cooc()  # eligibility changed
                    self._append_jsonl(self._promoted_path(), recs)
                    self._conn.commit()
                elif self._conn.in_transaction:
                    # a 0-row UPDATE still opened the implicit transaction —
                    # leaving it open crashes the next BEGIN IMMEDIATE (F16)
                    self._conn.rollback()
            return promoted
        return self._with_write_retry(_do)

    def unpromote(self, key_or_slug: str) -> int:
        """Withdraw promotion: the note returns to unverified immediately and
        stays there on future reindexes (a blank sha can never match)."""
        def _do() -> int:
            demoted = 0
            with self._lock:
                rows = self._resolve_rows(key_or_slug)
                recs = []
                for dk, nid in rows:
                    cur = self._conn.execute(
                        "UPDATE mem SET credibility='repo' WHERE dedupe_key=? AND bank != 'global'"
                        " AND kind != 'episode'", (dk,))
                    if cur.rowcount:
                        recs.append({"id": nid, "sha": "", "label": key_or_slug})
                        demoted += 1
                if demoted:
                    self._invalidate_cooc()
                    self._append_jsonl(self._promoted_path(), recs)
                    self._conn.commit()
                elif self._conn.in_transaction:
                    self._conn.rollback()  # same open-transaction hazard (F16)
            return demoted
        return self._with_write_retry(_do)

    # -- lifecycle ------------------------------------------------------------
    def compact(self, archive_path: Path | None = None) -> dict:
        """Archive (never silently delete) episode overflow and long-invalidated
        rows. Episode overflow is computed PER BANK (audit M2 — a global cap let
        a busy repo evict a quiet repo's episodes), never-served/oldest first.
        The archive conserves mem_stats too (audit — trust/served counts were
        silently lost on archive). Canon note files are never touched."""
        from collections import defaultdict
        archive_path = archive_path or self.db_path.with_name("mneme_archive.db")
        cutoff = time.time() - self.cfg["compact_after_days"] * 86400
        cap = int(self.cfg["max_episodes"])
        return self._with_write_retry(lambda: self._compact_locked(
            archive_path, cutoff, cap, defaultdict))

    def _compact_locked(self, archive_path: Path, cutoff: float, cap: int,
                        defaultdict) -> dict:
        with self._lock:
            doomed = [r[0] for r in self._conn.execute(
                "SELECT dedupe_key FROM mem WHERE invalid_at IS NOT NULL AND invalid_at < ?",
                (cutoff,)).fetchall()]
            doomed_set = set(doomed)
            episodes = self._conn.execute(
                "SELECT m.dedupe_key, m.bank, COALESCE(s.served, 0) FROM mem m"
                " LEFT JOIN mem_stats s ON s.note_id = m.note_id"
                " WHERE m.kind='episode' ORDER BY COALESCE(s.served,0) ASC, m.valid_at ASC").fetchall()
            by_bank: dict = defaultdict(list)
            for key, bank, _served in episodes:
                by_bank[bank].append(key)
            for bank, keys_in_bank in by_bank.items():
                overflow = len(keys_in_bank) - cap
                if overflow > 0:
                    for k in keys_in_bank[:overflow]:
                        if k not in doomed_set:
                            doomed.append(k)
                            doomed_set.add(k)
            if not doomed:
                return {"archived": 0, "archive": str(archive_path)}
            arc = sqlite3.connect(str(archive_path))
            try:
                # A v0-era archive predates note_id/credibility; CREATE IF NOT
                # EXISTS would leave its old tables in place and every insert
                # below would raise forever — including from add_episode's
                # auto-compact (verify F2). Sideline old tables as *_v0 copies.
                arc_cols = {r[1] for r in arc.execute("PRAGMA table_info(mem)").fetchall()}
                if arc_cols and "note_id" not in arc_cols:
                    arc.execute("ALTER TABLE mem RENAME TO mem_v0")
                    arc.execute("ALTER TABLE mem_stats RENAME TO mem_stats_v0")
                arc.executescript(_SCHEMA)  # full schema so mem_stats is conserved too
                qs = ",".join("?" * len(doomed))
                # Explicit column lists both sides: a migrated main DB has
                # note_id/credibility ALTERed to the end, a fresh archive has
                # them mid-table — SELECT * + positional INSERT would scramble.
                _MEM_COLS = ("id, dedupe_key, note_id, kind, bank, title, body, keywords,"
                             " tags, source, pinned, credibility, file_mtime, created_at,"
                             " valid_at, invalid_at, superseded_by")
                mem_rows = self._conn.execute(
                    f"SELECT {_MEM_COLS} FROM mem WHERE dedupe_key IN ({qs})", doomed).fetchall()
                arc.executemany(
                    f"INSERT OR REPLACE INTO mem ({_MEM_COLS}) VALUES"
                    f" ({','.join('?' * 17)})", mem_rows)
                _STAT_COLS = "note_id, served, positive, negative, quarantined, last_positive"
                stat_rows = self._conn.execute(
                    f"SELECT {_STAT_COLS} FROM mem_stats WHERE note_id IN"
                    f" (SELECT note_id FROM mem WHERE dedupe_key IN ({qs}))", doomed).fetchall()
                arc.executemany(
                    f"INSERT OR REPLACE INTO mem_stats ({_STAT_COLS}) VALUES (?,?,?,?,?,?)",
                    stat_rows)
                arc.commit()
            finally:
                arc.close()
            # Episode stats never come back (run ids are unique) — clean them from
            # the live db. File-note stats stay: their note_id may reappear.
            self._conn.execute(
                f"DELETE FROM mem_stats WHERE note_id IN"
                f" (SELECT note_id FROM mem WHERE dedupe_key IN ({qs})"
                f"  AND kind='episode')", doomed)
            for key in doomed:
                self._delete_locked(key)
            self._conn.commit()
        return {"archived": len(doomed), "archive": str(archive_path)}

    def stats(self) -> dict:
        with self._lock:
            by_kind = dict(self._conn.execute(
                "SELECT kind, COUNT(*) FROM mem WHERE invalid_at IS NULL GROUP BY kind").fetchall())
            invalid = self._conn.execute(
                "SELECT COUNT(*) FROM mem WHERE invalid_at IS NOT NULL").fetchone()[0]
            served = self._conn.execute(
                "SELECT COUNT(*) FROM mem_stats WHERE served > 0").fetchone()[0]
            total = self._conn.execute("SELECT COUNT(*) FROM mem").fetchone()[0]
            quarantined = self._conn.execute(
                "SELECT COUNT(*) FROM mem_stats WHERE quarantined=1").fetchone()[0]
            top = self._conn.execute(
                "SELECT DISTINCT s.note_id, s.served, s.positive, s.negative FROM mem_stats s"
                " JOIN mem m ON m.note_id = s.note_id"
                " ORDER BY s.served DESC LIMIT 10").fetchall()
        return {"db": str(self.db_path), "fts": self.fts_available, "rows": total,
                "by_kind": by_kind, "superseded": invalid, "ever_served": served,
                "never_served": max(0, total - served), "quarantined": quarantined,
                "most_served": [{"key": t[0], "served": t[1], "positive": t[2],
                                 "negative": t[3]} for t in top]}

    # -- authoring -------------------------------------------------------------
    def add_note(self, kind: str, title: str, body: str = "", *, keywords: str = "",
                 tags: str = "", pinned: bool = False, supersedes: str = "",
                 repo: Path | None = None, overwrite: bool = True) -> Path | None:
        """Write a canon note file (global dir, or the repo's .ktisis/memory)
        and index it immediately. Same slug = same file = update in place —
        the file IS the identity; git is its history. overwrite=False is the
        ADD-only mode (distiller): an existing note is never touched; returns
        None so the caller knows nothing was written."""
        if kind not in KINDS or kind == "episode":
            raise ValueError(f"kind must be one of lesson/fact/preference, got {kind!r}")
        slug = slugify(title)
        if not slug:
            raise ValueError("title produced an empty slug")
        root = self._project_root(repo) if repo else (self.global_dir / f"{kind}s")
        root.mkdir(parents=True, exist_ok=True)
        path = root / f"{slug}.md"
        if not overwrite and path.exists():
            return None
        # Update-in-place keeps the note's stable identity: reuse the existing
        # frontmatter id; a hand-authored note without one has been living
        # under its DERIVED id — minting a fresh uuid here would sever its
        # trust and dodge its quarantine (verify F3), so write nothing and let
        # the derived identity keep applying.
        note_id = ""
        derived_identity = False
        if path.exists():
            old = parse_note(path)
            if old is not None:
                note_id = old.note_id
                derived_identity = not old.note_id
        if not note_id and not derived_identity:
            note_id = uuid.uuid4().hex[:16]
        text = render_note(kind, title, body, keywords=keywords, tags=tags,
                           pinned=pinned, supersedes=supersedes, note_id=note_id)
        path.write_text(text, encoding="utf-8")
        if repo is not None:
            # We authored this repo note ourselves — register it as promoted
            # (id + content hash) so reindex keeps trusting THIS content. If the
            # repo later edits the file, the hash breaks and it demotes. The
            # registry keys by the STORED id (bank-composed frontmatter id, or
            # the derived id for id-less files), matching what
            # _note_credibility looks up at reindex.
            bank = self.bank_of(repo)
            stored_id = (self._compose_note_id(bank, note_id) if note_id
                         else _derived_note_id(bank, kind, title))
            sha = hashlib.sha256(text.strip().encode("utf-8")).hexdigest()[:16]
            self._append_jsonl(self._promoted_path(), [
                {"id": stored_id, "sha": sha, "label": slug}])
        self._index_note_file(path, self.bank_of(repo), credibility="operator")
        return path

    def _index_note_file(self, path: Path, bank: str, credibility: str | None = None) -> None:
        """Index exactly ONE canon note file. add_note knows precisely what it
        wrote — a full reindex per add rescans every file and re-runs the
        supersession pass, which made bulk seeding quadratic (the 2000-entry
        scale bench timed out at 10 minutes). External edits still use
        reindex(); this is the O(1) path for programmatic writes."""
        n = parse_note(Path(path))
        if n is None:
            return
        if credibility is None:
            credibility = self._note_credibility(
                n, bank, self._read_jsonl_last(self._promoted_path()))
        key = self._file_key(n.path)

        def _do() -> None:
            with self._lock:
                self._upsert_locked(key, n.kind, bank, n.title, n.body, n.keywords,
                                    n.tags, str(n.path), n.pinned, n.mtime, n.valid_at,
                                    note_id=self._note_identity(n, bank),
                                    credibility=credibility)
                if n.supersedes:
                    self._supersede_locked(key, n.supersedes, bank,
                                           sup_pinned=n.pinned, sup_cred=credibility)
                self._conn.commit()
        self._with_write_retry(_do)

    # === Unified agent-facing API =========================================
    # One layer, one interface. The verbs below are what an agent calls; the
    # methods above are the engine. Research sources (Mneme/Holographic/
    # Hindsight/Honcho/MEMORY.md) are folded in here, not routed between.

    def remember(self, title, body="", *, kind="lesson", keywords="", tags="",
                 pinned=False, supersedes="", project=None):
        """Durable write. kind in lesson|fact|preference. Returns the note path."""
        return self.add_note(kind, title, body, keywords=keywords, tags=tags,
                             pinned=pinned, supersedes=supersedes, repo=project)

    def record_outcome(self, run_id, status, used_keys=None):
        """Close a run: credit/debit proof-coupled trust for served memories.
        status: done | rolled_back | failed_verification | (neutral otherwise).
        Pass `used_keys` (the served keys the agent actually consulted/acted
        on) for full-strength signal; without it all evidence is weak and
        nothing gains trust from mere retrieval."""
        return self.apply_outcome(run_id, status, used_keys=used_keys)

    def audit(self):
        """Whole-store health: counts, served-vs-never (graveyard detector),
        trust distribution, quarantine, fts mode. graveyard_ratio is over LIVE
        NON-EPISODE notes only (audit M4 — including superseded rows and episodes
        in the denominator trended the ratio to ~1.0 regardless of real usage)."""
        s = self.stats()
        with self._lock:
            live = self._conn.execute(
                "SELECT COUNT(*) FROM mem WHERE invalid_at IS NULL AND kind != 'episode'").fetchone()[0]
            served_live = self._conn.execute(
                "SELECT COUNT(*) FROM mem m JOIN mem_stats st ON st.note_id = m.note_id"
                " WHERE m.invalid_at IS NULL AND m.kind != 'episode' AND st.served > 0").fetchone()[0]
            dist = self._conn.execute(
                "SELECT positive, negative FROM mem_stats WHERE positive+negative > 0").fetchall()
            unverified = self._conn.execute(
                "SELECT COUNT(*) FROM mem WHERE invalid_at IS NULL"
                " AND credibility != 'operator'").fetchone()[0]
        trusts = [(pos + 1) / (pos + neg + 2) for pos, neg in dist]
        s["trust_evaluated"] = len(trusts)
        s["trust_mean"] = round(sum(trusts) / len(trusts), 3) if trusts else None
        s["live_notes"] = live
        s["unverified_live"] = unverified
        s["graveyard_ratio"] = round((live - served_live) / live, 3) if live else 0.0
        return s

    def explain_recall(self, query, project=None, top_k=5):
        """Every hit with its score COMPONENTS (lexical, jaccard, trust, decay)
        so an operator can see WHY a memory ranked where it did."""
        banks = ("global",) if project is None else ("global", self.bank_of(project))
        q = (query + " " + self._repo_hint(project)).strip()
        with self._lock:
            cands = self._candidates(q, banks, limit=max(32, top_k * 6))
            stats = self._stats_map([c["key"] for c in cands])
        qg = _trigrams(q)
        out = []
        lex_w, jac_w = self._weights()
        for c in cands:
            srow = stats.get(c["key"])
            if srow is not None and srow[4]:
                continue
            trust = self._trust((srow[1], srow[2], srow[3]) if srow else None)
            jacc = _jaccard(qg, _trigrams(
                c["title"] + " " + c["keywords"] + " " + c["tags"])) if jac_w else 0.0
            decay = self._decay(c["kind"], c["valid_at"])
            cred = c.get("credibility", "operator")
            cred_w = 1.0 if cred == "operator" else UNVERIFIED_WEIGHT
            out.append({"title": c["title"], "kind": c["kind"],
                        "lexical": round(c["lex"], 3), "jaccard": round(jacc, 3),
                        "trust": round(trust, 3), "decay": round(decay, 3),
                        "credibility": cred,
                        "score": round((lex_w * c["lex"] + jac_w * jacc)
                                       * trust * cred_w * decay, 4)})
        out.sort(key=lambda x: -x["score"])
        return out[:top_k]

    def summarize_user_model(self, project=None):
        """Honcho-style user model, deterministic (no cloud, no LLM in the core
        path): synthesize preference notes into one readable block. A dialectic
        LLM pass is an optional host layer, never the core dependency."""
        banks = ("global",) if project is None else ("global", self.bank_of(project))
        qs = ",".join("?" * len(banks))
        with self._lock:
            # Operator credibility ONLY: this block is titled "Operator model" —
            # a repo-dropped kind:preference file must never speak as the
            # operator (verify F6: the one model-facing path the gate missed).
            rows = self._conn.execute(
                "SELECT title, body FROM mem WHERE kind='preference' AND invalid_at IS NULL "
                "AND credibility='operator' "
                "AND bank IN (" + qs + ") ORDER BY pinned DESC, valid_at DESC", banks).fetchall()
        if not rows:
            return ""
        lines = ["Operator model (from stated preferences):"]
        for title, body in rows[:30]:
            first = (body or "").strip().partition(_NL)[0].strip()
            line = "- " + title.strip()
            if first and first.lower() != title.strip().lower():
                line += " — " + first
            lines.append(line[:220])
        return _NL.join(lines)

    def export_memory(self, path=None):
        """MEMORY.md-style portable snapshot: all live notes as ONE readable
        markdown doc (kind-sectioned). Round-trips losslessly via import_memory:
        body lines are INDENTED 4 spaces so a body line like '## Heading' can't
        be mistaken for structure (audit C2), and bank/keywords/tags/pinned ride
        in the title-line comment (audit C3). Note: `bank` is exported for
        reference but import re-scopes to the import-time project (add_note has
        no bank override) — see import_memory."""
        with self._lock:
            # Operator canon only: exporting unverified repo notes and importing
            # them back would launder them into operator credibility (verify F7).
            # Repo notes are the repo's canon — regenerated by reindexing it,
            # never ported through the operator's snapshot.
            rows = self._conn.execute(
                "SELECT kind, bank, title, body, keywords, tags, pinned FROM mem "
                "WHERE invalid_at IS NULL AND kind != 'episode'"
                " AND credibility='operator' ORDER BY kind, bank, title").fetchall()
        parts = ["# Mneme memory export", ""]
        cur = None
        for kind, bank, title, body, kw, tags, pinned in rows:
            if kind != cur:
                parts += ["", "## " + kind + "s", ""]
                cur = kind
            meta = []
            if bank != "global":
                meta.append("bank=" + bank)
            if kw:
                meta.append("keywords=" + kw)
            if tags:
                meta.append("tags=" + tags)
            if pinned:
                meta.append("pinned")
            head = "### " + title
            if meta:
                head += "  <!-- " + "; ".join(meta) + " -->"
            parts.append(head)
            if body.strip():
                parts += ["    " + ln if ln.strip() else "" for ln in body.splitlines()]
                parts.append("")
        text = _NL.join(parts) + _NL
        if path:
            Path(path).write_text(text, encoding="utf-8")
        return text

    @staticmethod
    def _parse_export_meta(comment: str) -> dict:
        """Parse a `bank=..; keywords=..; tags=..; pinned` export comment."""
        out = {"keywords": "", "tags": "", "pinned": False}
        for part in comment.split(";"):
            part = part.strip()
            if part == "pinned":
                out["pinned"] = True
            elif part.startswith("keywords="):
                out["keywords"] = part[len("keywords="):].strip()
            elif part.startswith("tags="):
                out["tags"] = part[len("tags="):].strip()
        return out

    def import_memory(self, path_or_text, project=None):
        """Parse an export_memory() snapshot back into notes. ADD-only: existing
        slugs untouched. Restores keywords/tags/pinned from the export comment
        (audit C3); notes re-scope to `project` (bank is not reconstructable via
        add_note). The path-vs-text heuristic only treats the input as a path if
        it is a single line with no newline AND that file exists. Returns count
        written."""
        import re as _re
        raw = path_or_text
        if "\n" not in path_or_text and len(path_or_text) < 4096:
            try:
                cand = Path(path_or_text)
                if cand.exists():
                    raw = cand.read_text(encoding="utf-8")
            except (OSError, ValueError):
                pass  # not a usable path — treat the string as literal snapshot
        state = {"kind": "lesson", "written": 0, "title": None, "meta": None, "buf": []}

        def flush():
            if state["title"]:
                body = _NL.join(state["buf"]).strip()
                m = state["meta"] or {"keywords": "", "tags": "", "pinned": False}
                try:
                    if self.add_note(state["kind"], state["title"], body,
                                     keywords=m["keywords"], tags=m["tags"],
                                     pinned=m["pinned"], repo=project,
                                     overwrite=False) is not None:
                        state["written"] += 1
                except ValueError:
                    pass
            state["title"], state["meta"], state["buf"] = None, None, []

        for line in raw.splitlines():
            if line.startswith("## ") and not line.startswith("### "):
                flush()
                k = line[3:].strip().lower().rstrip("s")
                state["kind"] = k if k in ("lesson", "fact", "preference") else "lesson"
            elif line.startswith("### "):
                flush()
                head = line[4:]
                cm = _re.search(r"<!--(.*?)-->", head)
                state["meta"] = self._parse_export_meta(cm.group(1)) if cm else None
                state["title"] = _re.sub(r"\s*<!--.*?-->\s*", "", head).strip()
            elif state["title"] is not None:
                # body lines are exported indented 4 spaces; strip exactly that
                state["buf"].append(line[4:] if line.startswith("    ") else line)
        flush()
        return state["written"]


# ---------------------------------------------------------------------------
# CLI: python mneme.py <cmd> [...] — MNEME_DIR env or --dir sets the home.
# ---------------------------------------------------------------------------

def _cli() -> int:
    import argparse
    import json as _json
    import os

    ap = argparse.ArgumentParser(prog="mneme", description="file-canon agent memory")
    ap.add_argument("--dir", default=os.environ.get("MNEME_DIR", "./mneme-data"),
                    help="memory home (notes + mneme.db); default ./mneme-data or MNEME_DIR")
    ap.add_argument("--project", default=None, help="project dir for bank scoping")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p_add = sub.add_parser("add"); p_add.add_argument("--kind", default="lesson",
        choices=["lesson", "fact", "preference"]); p_add.add_argument("--title", required=True)
    p_add.add_argument("--body", default=""); p_add.add_argument("--keywords", default="")
    p_add.add_argument("--tags", default=""); p_add.add_argument("--pin", action="store_true")
    p_add.add_argument("--supersedes", default="", help="slug of a note this one replaces")
    p_rec = sub.add_parser("recall"); p_rec.add_argument("query")
    p_show = sub.add_parser("show"); p_show.add_argument("query", nargs="?", default="")
    sub.add_parser("stats"); sub.add_parser("audit")
    sub.add_parser("reindex"); sub.add_parser("compact")
    p_q = sub.add_parser("quarantine"); p_q.add_argument("slug"); p_q.add_argument("--off", action="store_true")
    p_pro = sub.add_parser("promote", help="trust a repo-suggested note (content-hash pinned)")
    p_pro.add_argument("slug"); p_pro.add_argument("--off", action="store_true",
                                                   help="withdraw promotion instead")
    p_exp = sub.add_parser("export"); p_exp.add_argument("--out", default="", help="write to file (else stdout)")
    p_imp = sub.add_parser("import"); p_imp.add_argument("file", help="snapshot file to import")
    args = ap.parse_args()

    home = Path(args.dir); home.mkdir(parents=True, exist_ok=True)
    mem = Mneme(home / "mneme.db", home / "notes")
    project = Path(args.project) if args.project else None
    mem.reindex(project)
    if args.cmd == "add":
        path = mem.add_note(args.kind, args.title, args.body, keywords=args.keywords,
                            tags=args.tags, pinned=args.pin, supersedes=args.supersedes,
                            repo=project)
        print(f"note written: {path}")
    elif args.cmd == "recall":
        for h in mem.recall(args.query, project):
            tag = "" if h.get("credibility", "operator") == "operator" else " [UNVERIFIED:repo]"
            print(f"{h['score']:.3f} trust={h['trust']:.2f} [{h['kind']}]{tag} {h['title'][:80]}")
    elif args.cmd == "show":
        print(mem.index_block(project, args.query).text or "(empty index block)")
    elif args.cmd == "stats":
        print(_json.dumps(mem.stats(), indent=2))
    elif args.cmd == "audit":
        print(_json.dumps(mem.audit(), indent=2))
    elif args.cmd == "reindex":
        print(_json.dumps(mem.reindex(project, force=True)))
    elif args.cmd == "compact":
        print(_json.dumps(mem.compact()))
    elif args.cmd == "quarantine":
        print("ok" if mem.quarantine(args.slug, on=not args.off) else "no such memory")
    elif args.cmd == "promote":
        n = mem.unpromote(args.slug) if args.off else mem.promote(args.slug)
        print(f"{'demoted' if args.off else 'promoted'} {n} note(s)" if n else "no matching unverified note")
    elif args.cmd == "export":
        text = mem.export_memory(args.out or None)
        print(f"exported to {args.out}" if args.out else text, end="" if args.out else "\n")
    elif args.cmd == "import":
        print(f"imported {mem.import_memory(args.file, project)} note(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
