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
        except ValueError:
            continue
    try:
        return float(raw)
    except ValueError:
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


def render_note(kind: str, title: str, body: str, *, keywords: str = "", tags: str = "",
                pinned: bool = False, supersedes: str = "") -> str:
    """Render a note file with frontmatter (for `ktisis mneme add` / distiller)."""
    lines = ["---", f"kind: {kind}"]
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
import re
import sqlite3
import subprocess
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path


_SCHEMA = """
CREATE TABLE IF NOT EXISTS mem (
    id TEXT PRIMARY KEY,
    dedupe_key TEXT NOT NULL UNIQUE,
    kind TEXT NOT NULL,
    bank TEXT NOT NULL,
    title TEXT NOT NULL,
    body TEXT NOT NULL,
    keywords TEXT NOT NULL DEFAULT '',
    tags TEXT NOT NULL DEFAULT '',
    source TEXT NOT NULL,
    pinned INTEGER NOT NULL DEFAULT 0,
    file_mtime REAL,
    created_at REAL NOT NULL,
    valid_at REAL NOT NULL,
    invalid_at REAL,
    superseded_by TEXT
);
CREATE INDEX IF NOT EXISTS mem_bank_kind ON mem (bank, kind);
CREATE TABLE IF NOT EXISTS mem_stats (
    dedupe_key TEXT PRIMARY KEY,
    served INTEGER NOT NULL DEFAULT 0,
    positive INTEGER NOT NULL DEFAULT 0,
    negative INTEGER NOT NULL DEFAULT 0,
    quarantined INTEGER NOT NULL DEFAULT 0
);
CREATE TABLE IF NOT EXISTS mem_served (
    run_id TEXT NOT NULL,
    dedupe_key TEXT NOT NULL,
    tier TEXT NOT NULL,
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
    "max_episodes": 2000,
    "compact_after_days": 90,
    "auto_compact": True,
}

# Episodes may exceed max_episodes by this slack before an automatic compact
# fires (keeps the common add_episode path cheap; compaction stays amortized).
AUTO_COMPACT_SLACK = 10

# Outcomes that count as evidence about served memories. Everything else
# (denied, blocked, stopped, crashed, vacuous_checks, planned) is neutral:
# the work was never carried to a provable end, so memory takes no blame.
POSITIVE_STATUSES = ("done",)
NEGATIVE_STATUSES = ("rolled_back", "failed_verification")


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
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        self._lock = threading.Lock()
        with self._lock:
            try:
                self._conn.execute("PRAGMA journal_mode=WAL")
            except sqlite3.DatabaseError:
                pass  # network drives / old builds: default journal still works
            self._conn.executescript(_SCHEMA)
            self.fts_available = True
            try:
                self._conn.executescript(_FTS_SCHEMA)
            except sqlite3.OperationalError:
                self.fts_available = False  # LIKE+Jaccard fallback takes over
            self._conn.commit()

    def close(self) -> None:
        self._conn.close()

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
        return "file:" + path.resolve().as_posix().lower()

    def reindex(self, repo: Path | None = None, force: bool = False) -> dict:
        """Mirror canon note files into the DB. mtime-based, idempotent, and
        prunes rows whose file vanished. Episodes (DB-canon) are untouched."""
        roots: list[tuple[Path, str]] = [(self.global_dir, "global")]
        roots += [(d, "global") for d in self.extra_dirs]
        if repo is not None:
            roots.append((Path(repo) / ".ktisis" / "memory", self.bank_of(repo)))
        added = updated = pruned = 0
        supersedes: list[tuple[str, str]] = []  # (new_key, superseded slug)
        with self._lock:
            for root, bank in roots:
                notes = scan_notes(root)
                seen_keys = set()
                for n in notes:
                    key = self._file_key(n.path)
                    seen_keys.add(key)
                    row = self._conn.execute(
                        "SELECT file_mtime FROM mem WHERE dedupe_key=?", (key,)).fetchone()
                    if row is not None and not force and row[0] == n.mtime:
                        continue
                    self._upsert_locked(key, n.kind, bank, n.title, n.body, n.keywords,
                                        n.tags, str(n.path), n.pinned, n.mtime, n.valid_at)
                    if n.supersedes:
                        supersedes.append((key, n.supersedes))
                    added += 1 if row is None else 0
                    updated += 0 if row is None else 1
                prefix = "file:" + root.resolve().as_posix().lower()
                stale = self._conn.execute(
                    "SELECT dedupe_key FROM mem WHERE dedupe_key LIKE ? || '%'",
                    (prefix,)).fetchall()
                for (key,) in stale:
                    if key not in seen_keys:
                        self._delete_locked(key)
                        pruned += 1
            for new_key, target in supersedes:
                self._supersede_locked(new_key, target)
            self._conn.commit()
        return {"added": added, "updated": updated, "pruned": pruned,
                "fts": self.fts_available}

    def _upsert_locked(self, key: str, kind: str, bank: str, title: str, body: str,
                       keywords: str, tags: str, source: str, pinned: bool,
                       file_mtime: float | None, valid_at: float) -> None:
        now = time.time()
        old = self._conn.execute("SELECT id, created_at FROM mem WHERE dedupe_key=?", (key,)).fetchone()
        self._invalidate_cooc()
        mem_id = old[0] if old else uuid.uuid4().hex[:12]
        created = old[1] if old else now
        self._conn.execute("DELETE FROM mem WHERE dedupe_key=?", (key,))
        self._conn.execute(
            "INSERT INTO mem (id, dedupe_key, kind, bank, title, body, keywords, tags, source,"
            " pinned, file_mtime, created_at, valid_at, invalid_at, superseded_by)"
            " VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,NULL,NULL)",
            (mem_id, key, kind, bank, title, body, keywords, tags, source,
             int(pinned), file_mtime, created, valid_at))
        if self.fts_available:
            self._conn.execute("DELETE FROM mem_fts WHERE dedupe_key=?", (key,))
            self._conn.execute(
                "INSERT INTO mem_fts (dedupe_key, title, body, keywords, tags) VALUES (?,?,?,?,?)",
                (key, title, body, keywords, tags))

    def _invalidate_cooc(self) -> None:
        self._cooc = None

    def _delete_locked(self, key: str) -> None:
        self._invalidate_cooc()
        self._conn.execute("DELETE FROM mem WHERE dedupe_key=?", (key,))
        if self.fts_available:
            self._conn.execute("DELETE FROM mem_fts WHERE dedupe_key=?", (key,))

    def _supersede_locked(self, new_key: str, target: str) -> None:
        """Mark older rows invalid instead of deleting them (bi-temporal).
        Target matches a note filename stem or a slugified title."""
        slug = slugify(target) or target.lower()
        now = time.time()
        rows = self._conn.execute(
            "SELECT dedupe_key, title FROM mem WHERE invalid_at IS NULL AND dedupe_key != ?",
            (new_key,)).fetchall()
        for key, title in rows:
            stem = key.rsplit("/", 1)[-1].removesuffix(".md")
            if stem == slug or slugify(title) == slug:
                self._conn.execute(
                    "UPDATE mem SET invalid_at=?, superseded_by=? WHERE dedupe_key=?",
                    (now, new_key, key))

    # -- episodes: DB-canon, append-only --------------------------------------
    def add_episode(self, run_id: str, goal: str, status: str, detail: str,
                    repo: Path | None) -> None:
        key = "run:" + run_id
        episode_count = 0
        with self._lock:
            exists = self._conn.execute("SELECT 1 FROM mem WHERE dedupe_key=?", (key,)).fetchone()
            if exists:
                return  # idempotent: one episode per run, ever
            now = time.time()
            title = f"[{status}] {goal[:140]}"
            self._conn.execute(
                "INSERT INTO mem (id, dedupe_key, kind, bank, title, body, keywords, tags,"
                " source, pinned, file_mtime, created_at, valid_at, invalid_at, superseded_by)"
                " VALUES (?,?,?,?,?,?,?,?,?,0,NULL,?,?,NULL,NULL)",
                (uuid.uuid4().hex[:12], key, "episode", self.bank_of(repo), title,
                 detail[:2000], "", status, "run:" + run_id, now, now))
            if self.fts_available:
                self._conn.execute(
                    "INSERT INTO mem_fts (dedupe_key, title, body, keywords, tags) VALUES (?,?,?,?,?)",
                    (key, title, detail[:2000], "", status))
            self._conn.commit()
            episode_count = self._conn.execute(
                "SELECT COUNT(*) FROM mem WHERE kind='episode'").fetchone()[0]
        # Outside the lock (compact re-acquires it): amortized self-maintenance.
        if (self.cfg.get("auto_compact", True)
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
        if not keys:
            return {}
        qs = ",".join("?" * len(keys))
        rows = self._conn.execute(
            f"SELECT dedupe_key, served, positive, negative, quarantined FROM mem_stats"
            f" WHERE dedupe_key IN ({qs})", keys).fetchall()
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
        rows = self._conn.execute(
            "SELECT title, keywords, tags, substr(body, 1, 400) FROM mem"
            " WHERE invalid_at IS NULL").fetchall()
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

    def _candidates(self, query: str, banks: tuple, limit: int = 64) -> list[dict]:
        """Lexical candidates: FTS5/BM25 (porter-stemmed) when available, else
        LIKE token matching. Returns rows with a normalized lexical score."""
        toks = _tokens(query)
        qs_banks = ",".join("?" * len(banks))
        rows: list[tuple] = []
        if toks and self.fts_available:
            match = " OR ".join(f'"{t}"' for t in toks)
            rows = self._conn.execute(
                f"SELECT m.dedupe_key, m.kind, m.bank, m.title, m.body, m.keywords, m.tags,"
                f" m.pinned, m.valid_at, bm25(mem_fts) AS r"
                f" FROM mem_fts JOIN mem m ON m.dedupe_key = mem_fts.dedupe_key"
                f" WHERE mem_fts MATCH ? AND m.bank IN ({qs_banks}) AND m.invalid_at IS NULL"
                f" ORDER BY r LIMIT ?", (match, *banks, limit)).fetchall()
            if not rows:
                # Empty-result rescue (bake-off verdict): co-occurrence expansion
                # dilutes ranked results on the hot path, but when the strict
                # query finds NOTHING a wider net can only help.
                wide = self._expand_tokens(toks)
                if len(wide) > len(toks):
                    match = " OR ".join(f'"{t}"' for t in wide)
                    rows = self._conn.execute(
                        f"SELECT m.dedupe_key, m.kind, m.bank, m.title, m.body, m.keywords, m.tags,"
                        f" m.pinned, m.valid_at, bm25(mem_fts) AS r"
                        f" FROM mem_fts JOIN mem m ON m.dedupe_key = mem_fts.dedupe_key"
                        f" WHERE mem_fts MATCH ? AND m.bank IN ({qs_banks}) AND m.invalid_at IS NULL"
                        f" ORDER BY r LIMIT ?", (match, *banks, limit)).fetchall()
        elif toks:
            like = " OR ".join(["title LIKE ? OR body LIKE ? OR keywords LIKE ? OR tags LIKE ?"] * min(len(toks), 8))
            params: list = []
            for t in toks[:8]:
                params += [f"%{t}%"] * 4
            found = self._conn.execute(
                f"SELECT dedupe_key, kind, bank, title, body, keywords, tags, pinned, valid_at"
                f" FROM mem WHERE ({like}) AND bank IN ({qs_banks}) AND invalid_at IS NULL LIMIT 400",
                (*params, *banks)).fetchall()
            scored = []
            tokset = set(toks)
            for f in found:
                text = (f[3] + " " + f[4][:2000] + " " + f[5] + " " + f[6]).lower()
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
                        "lex": (hi - r[-1]) / span})
        return out

    def retrieve(self, query: str, repo: Path | None = None, limit: int = 64) -> list[dict]:
        """Ranked memories: (0.65*lexical + 0.35*trigram-Jaccard) * trust * decay.
        Quarantined and superseded rows never surface."""
        banks = ("global",) if repo is None else ("global", self.bank_of(repo))
        with self._lock:
            cands = self._candidates(query, banks, limit)
            stats = self._stats_map([c["key"] for c in cands])
        qgrams = _trigrams(query)
        out = []
        for c in cands:
            s = stats.get(c["key"])
            if s is not None and s[4]:
                continue  # quarantined
            trust = self._trust((s[1], s[2], s[3]) if s else None)
            jac = _jaccard(qgrams, _trigrams(c["title"] + " " + c["keywords"] + " " + c["tags"]))
            c["trust"] = trust
            c["score"] = (0.65 * c["lex"] + 0.35 * jac) * trust * self._decay(c["kind"], c["valid_at"])
            out.append(c)
        out.sort(key=lambda x: (-x["score"], x["key"]))
        return out

    def recall(self, query: str, repo: Path | None = None, top_k: int | None = None) -> list[dict]:
        k = top_k or self.cfg["recall_top_k"]
        return self.retrieve(query, repo, limit=max(32, k * 4))[:k]

    @staticmethod
    def _repo_hint(repo: Path | None) -> str:
        """Query boost from the deterministic repo snapshot (languages and
        frameworks): 'python pytest' pulls framework-specific memories in even
        when the goal sentence never names the stack."""
        if repo is None:
            return ""
        try:
            data = json.loads((Path(repo) / ".ktisis" / "repo_snapshot.json").read_text(encoding="utf-8"))
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
            pinned_rows = self._conn.execute(
                f"SELECT dedupe_key, kind, title, body, valid_at FROM mem"
                f" WHERE pinned=1 AND invalid_at IS NULL AND bank IN ({qs_banks})"
                f" ORDER BY kind, title", banks).fetchall()
            pinned_stats = self._stats_map([r[0] for r in pinned_rows])
        pinned = [{"key": r[0], "kind": r[1], "title": r[2], "body": r[3], "valid_at": r[4]}
                  for r in pinned_rows if not (pinned_stats.get(r[0]) or (0, 0, 0, 0, 0))[4]]
        ranked = [c for c in self.retrieve(query, repo) if not c["pinned"]]
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
        parts = []
        for kind in ("lesson", "fact", "preference", "episode"):
            if sections[kind]:
                parts.append(_SECTION_HEADERS[kind] + "\n" + "\n".join(sections[kind]))
        text = "\n\n".join(parts)[:budget]
        keys = [c["key"] for c in chosen]
        if run_id and keys:
            self.record_served(run_id, keys, "index")
        return IndexBlock(text=text, keys=keys)

    def _recent_fallback(self, banks: tuple) -> list[dict]:
        """No query signal: serve the most trusted, most recent memories."""
        qs = ",".join("?" * len(banks))
        with self._lock:
            rows = self._conn.execute(
                f"SELECT dedupe_key, kind, title, body, valid_at, pinned FROM mem"
                f" WHERE invalid_at IS NULL AND bank IN ({qs})"
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
        now = time.time()
        with self._lock:
            for key in keys:
                self._conn.execute(
                    "INSERT INTO mem_served (run_id, dedupe_key, tier, ts) VALUES (?,?,?,?)",
                    (run_id, key, tier, now))
                self._conn.execute(
                    "INSERT INTO mem_stats (dedupe_key, served) VALUES (?, 1)"
                    " ON CONFLICT(dedupe_key) DO UPDATE SET served = served + 1", (key,))
            self._conn.commit()

    def apply_outcome(self, run_id: str, status: str) -> int:
        """Credit or debit every memory served in this run, based on the PROVEN
        outcome. Neutral statuses change nothing. Returns rows touched."""
        if status in POSITIVE_STATUSES:
            col = "positive"
        elif status in NEGATIVE_STATUSES:
            col = "negative"
        else:
            return 0
        with self._lock:
            keys = [r[0] for r in self._conn.execute(
                "SELECT DISTINCT dedupe_key FROM mem_served WHERE run_id=?", (run_id,)).fetchall()]
            for key in keys:
                self._conn.execute(
                    f"INSERT INTO mem_stats (dedupe_key, {col}) VALUES (?, 1)"
                    f" ON CONFLICT(dedupe_key) DO UPDATE SET {col} = {col} + 1", (key,))
            self._conn.commit()
        return len(keys)

    def quarantine(self, key_or_slug: str, on: bool = True) -> bool:
        """Operator veto: a quarantined memory never surfaces anywhere except
        `mneme show`. The note file is untouched — this is a store-level gate."""
        with self._lock:
            row = self._conn.execute(
                "SELECT dedupe_key FROM mem WHERE dedupe_key=? OR dedupe_key LIKE '%/' || ? || '.md'",
                (key_or_slug, key_or_slug)).fetchone()
            if row is None:
                return False
            self._conn.execute(
                "INSERT INTO mem_stats (dedupe_key, quarantined) VALUES (?, ?)"
                " ON CONFLICT(dedupe_key) DO UPDATE SET quarantined = ?",
                (row[0], int(on), int(on)))
            self._conn.commit()
        return True

    # -- lifecycle ------------------------------------------------------------
    def compact(self, archive_path: Path | None = None) -> dict:
        """Archive (never silently delete) episode overflow and long-invalidated
        rows. Never-served episodes go first; canon files are never touched."""
        archive_path = archive_path or self.db_path.with_name("mneme_archive.db")
        cutoff = time.time() - self.cfg["compact_after_days"] * 86400
        with self._lock:
            doomed = [r[0] for r in self._conn.execute(
                "SELECT dedupe_key FROM mem WHERE invalid_at IS NOT NULL AND invalid_at < ?",
                (cutoff,)).fetchall()]
            episodes = self._conn.execute(
                "SELECT m.dedupe_key, COALESCE(s.served, 0) FROM mem m"
                " LEFT JOIN mem_stats s ON s.dedupe_key = m.dedupe_key"
                " WHERE m.kind='episode' ORDER BY COALESCE(s.served,0) ASC, m.valid_at ASC").fetchall()
            overflow = len(episodes) - int(self.cfg["max_episodes"])
            if overflow > 0:
                doomed += [k for k, _ in episodes[:overflow] if k not in doomed]
            if not doomed:
                return {"archived": 0, "archive": str(archive_path)}
            arc = sqlite3.connect(str(archive_path))
            arc.executescript(_SCHEMA.split("CREATE TABLE IF NOT EXISTS mem_stats")[0])
            qs = ",".join("?" * len(doomed))
            rows = self._conn.execute(f"SELECT * FROM mem WHERE dedupe_key IN ({qs})", doomed).fetchall()
            arc.executemany(
                "INSERT OR REPLACE INTO mem VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)", rows)
            arc.commit()
            arc.close()
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
                "SELECT s.dedupe_key, s.served, s.positive, s.negative FROM mem_stats s"
                " JOIN mem m ON m.dedupe_key = s.dedupe_key"
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
        root = (Path(repo) / ".ktisis" / "memory") if repo else (self.global_dir / f"{kind}s")
        root.mkdir(parents=True, exist_ok=True)
        path = root / f"{slug}.md"
        if not overwrite and path.exists():
            return None
        path.write_text(render_note(kind, title, body, keywords=keywords, tags=tags,
                                    pinned=pinned, supersedes=supersedes), encoding="utf-8")
        self._index_note_file(path, self.bank_of(repo))
        return path

    def _index_note_file(self, path: Path, bank: str) -> None:
        """Index exactly ONE canon note file. add_note knows precisely what it
        wrote — a full reindex per add rescans every file and re-runs the
        supersession pass, which made bulk seeding quadratic (the 2000-entry
        scale bench timed out at 10 minutes). External edits still use
        reindex(); this is the O(1) path for programmatic writes."""
        n = parse_note(Path(path))
        if n is None:
            return
        key = self._file_key(n.path)
        with self._lock:
            self._upsert_locked(key, n.kind, bank, n.title, n.body, n.keywords,
                                n.tags, str(n.path), n.pinned, n.mtime, n.valid_at)
            if n.supersedes:
                self._supersede_locked(key, n.supersedes)
            self._conn.commit()

    # === Unified agent-facing API =========================================
    # One layer, one interface. The verbs below are what an agent calls; the
    # methods above are the engine. Research sources (Mneme/Holographic/
    # Hindsight/Honcho/MEMORY.md) are folded in here, not routed between.

    def remember(self, title, body="", *, kind="lesson", keywords="", tags="",
                 pinned=False, supersedes="", project=None):
        """Durable write. kind in lesson|fact|preference. Returns the note path."""
        return self.add_note(kind, title, body, keywords=keywords, tags=tags,
                             pinned=pinned, supersedes=supersedes, repo=project)

    def record_outcome(self, run_id, status):
        """Close a run: credit/debit proof-coupled trust for served memories.
        status: done | rolled_back | failed_verification | (neutral otherwise)."""
        return self.apply_outcome(run_id, status)

    def audit(self):
        """Whole-store health: counts, served-vs-never (graveyard detector),
        trust distribution, quarantine, fts mode."""
        s = self.stats()
        with self._lock:
            dist = self._conn.execute(
                "SELECT positive, negative FROM mem_stats WHERE positive+negative > 0").fetchall()
        trusts = [(pos + 1) / (pos + neg + 2) for pos, neg in dist]
        s["trust_evaluated"] = len(trusts)
        s["trust_mean"] = round(sum(trusts) / len(trusts), 3) if trusts else None
        s["graveyard_ratio"] = round(s["never_served"] / s["rows"], 3) if s["rows"] else 0.0
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
        for c in cands:
            srow = stats.get(c["key"])
            if srow is not None and srow[4]:
                continue
            trust = self._trust((srow[1], srow[2], srow[3]) if srow else None)
            jacc = _jaccard(qg, _trigrams(c["title"] + " " + c["keywords"] + " " + c["tags"]))
            decay = self._decay(c["kind"], c["valid_at"])
            out.append({"title": c["title"], "kind": c["kind"],
                        "lexical": round(c["lex"], 3), "jaccard": round(jacc, 3),
                        "trust": round(trust, 3), "decay": round(decay, 3),
                        "score": round((0.65 * c["lex"] + 0.35 * jacc) * trust * decay, 4)})
        out.sort(key=lambda x: -x["score"])
        return out[:top_k]

    def summarize_user_model(self, project=None):
        """Honcho-style user model, deterministic (no cloud, no LLM in the core
        path): synthesize preference notes into one readable block. A dialectic
        LLM pass is an optional host layer, never the core dependency."""
        banks = ("global",) if project is None else ("global", self.bank_of(project))
        qs = ",".join("?" * len(banks))
        with self._lock:
            rows = self._conn.execute(
                "SELECT title, body FROM mem WHERE kind='preference' AND invalid_at IS NULL "
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
        markdown doc (kind-sectioned). Importable back via import_memory()."""
        with self._lock:
            rows = self._conn.execute(
                "SELECT kind, bank, title, body, keywords, tags, pinned FROM mem "
                "WHERE invalid_at IS NULL AND kind != 'episode' ORDER BY kind, bank, title").fetchall()
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
                parts += [body.strip(), ""]
        text = _NL.join(parts) + _NL
        if path:
            Path(path).write_text(text, encoding="utf-8")
        return text

    def import_memory(self, path_or_text, project=None):
        """Parse an export_memory() snapshot back into notes. ADD-only: existing
        slugs untouched. Returns count written."""
        import re as _re
        raw = path_or_text
        cand = Path(path_or_text) if len(path_or_text) < 400 else None
        if cand is not None and cand.exists():
            raw = cand.read_text(encoding="utf-8")
        state = {"kind": "lesson", "written": 0, "title": None, "buf": []}

        def flush():
            if state["title"]:
                body = _NL.join(state["buf"]).strip()
                try:
                    if self.add_note(state["kind"], state["title"], body, tags="imported",
                                     repo=project, overwrite=False) is not None:
                        state["written"] += 1
                except ValueError:
                    pass
            state["title"], state["buf"] = None, []

        for line in raw.splitlines():
            if line.startswith("## ") and not line.startswith("### "):
                flush()
                k = line[3:].strip().lower().rstrip("s")
                state["kind"] = k if k in ("lesson", "fact", "preference") else "lesson"
            elif line.startswith("### "):
                flush()
                state["title"] = _re.sub(r"\s*<!--.*?-->\s*", "", line[4:]).strip()
            elif state["title"] is not None:
                state["buf"].append(line)
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
    p_rec = sub.add_parser("recall"); p_rec.add_argument("query")
    p_show = sub.add_parser("show"); p_show.add_argument("query", nargs="?", default="")
    sub.add_parser("stats"); sub.add_parser("reindex"); sub.add_parser("compact")
    p_q = sub.add_parser("quarantine"); p_q.add_argument("slug"); p_q.add_argument("--off", action="store_true")
    args = ap.parse_args()

    home = Path(args.dir); home.mkdir(parents=True, exist_ok=True)
    mem = Mneme(home / "mneme.db", home / "notes")
    project = Path(args.project) if args.project else None
    mem.reindex(project)
    if args.cmd == "add":
        path = mem.add_note(args.kind, args.title, args.body, keywords=args.keywords,
                            tags=args.tags, pinned=args.pin, repo=project)
        print(f"note written: {path}")
    elif args.cmd == "recall":
        for h in mem.recall(args.query, project):
            print(f"{h['score']:.3f} trust={h['trust']:.2f} [{h['kind']}] {h['title'][:80]}")
    elif args.cmd == "show":
        print(mem.index_block(project, args.query).text or "(empty index block)")
    elif args.cmd == "stats":
        print(_json.dumps(mem.stats(), indent=2))
    elif args.cmd == "reindex":
        print(_json.dumps(mem.reindex(project, force=True)))
    elif args.cmd == "compact":
        print(_json.dumps(mem.compact()))
    elif args.cmd == "quarantine":
        print("ok" if mem.quarantine(args.slug, on=not args.off) else "no such memory")
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
