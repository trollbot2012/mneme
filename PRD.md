# PRD — Mneme: a file-canon, proof-coupled memory layer for AI agents

Version 0.1.0 · 2026-07-05 · Status: shipped inside Ktisis; standalone extraction released

## 1. Problem

Agent memory systems fail in a specific, repeated way: **they are written to and
never read from.** A production audit of a long-running agent (Hermes, 2026-07-04)
found 356KB of accumulated "lessons" no code path ever loaded, vector stores whose
only reader was never configured, and memory directories whose folder names implied
function they didn't have. Writes prove nothing; only injection into a live prompt
proves memory exists. Secondary failures: cross-project contamination (one bank
bleeding into another), silent destruction of old memory by LLM "consolidation",
and trust assigned by model vibes rather than outcomes.

## 2. Evidence base

A 106-agent adversarially-verified research pass (24 confirmed findings, 1 killed)
across Mem0, Zep/Graphiti, MemGPT/Letta, A-MEM, Claude Code file memory, and
zero-dependency SQLite/HRR stores. Key confirmed findings:

| Finding | Source | Consequence in Mneme |
|---|---|---|
| BM25/FTS lexical search is a first-class signal in every flagship (Zep co-equal w/ cosine; Mem0 V3 fuses it) | arXiv 2501.13956; mem0 repo | Embeddings-free retrieval is primary, not a fallback |
| Never destroy at write time: Mem0 V3 moved to ADD-only + read-time resolution; Zep invalidates bi-temporally | mem0 migration docs; Zep paper | Append-only supersession (`valid_at`/`invalid_at`), no UPDATE/DELETE of content |
| Extraction is a latency/token win, NOT an accuracy win (full context beat Mem0 on its own benchmark) | arXiv 2504.19413 | Markdown canon is kept whole; distillation optional and ADD-only |
| Graph memory adds ~2%, worse on multi-hop, per its own vendor | arXiv 2504.19413 | No graph engine; plain links table |
| Two-tier read path (always-loaded capped index + on-demand files) works in production | Claude Code docs (fetched live) | Index block + recall tier |
| Memory files are advisory, not enforcement | Anthropic docs (admission) | Anything that MUST happen belongs in host checks/hooks, not prose notes |
| Memory benchmarks are context-saturated; evaluate on cross-session reuse instead | arXiv 2602.19320 | Success metrics §7 |

## 3. Users

1. **Agent builders** who want durable memory without a server, a vector DB, an
   embedding model, or a SaaS dependency.
2. **Agent operators** who want to read, edit, diff, and version their agent's
   memory as plain files they own.
3. **Host systems** (reference: Ktisis coding agent; adoptable by any
   Hermes-class chat agent) that can report proven outcomes per run.

## 4. Requirements

### Functional
- F1. Four memory kinds: `lesson` (procedural), `fact` (semantic, per-project),
  `preference` (user model), `episode` (run history).
- F2. Canon = markdown files with flat frontmatter; DB is disposable and fully
  rebuildable from files (`reindex`); episodes are DB-canon with archive-only
  compaction.
- F3. Guaranteed read path: `index_block(project, task_text)` — budget-capped,
  pinned-first, per-kind shares, deterministic ordering; host injects it at
  every task start.
- F4. On-demand tier: `recall(query, project)` ranked by
  `(0.65·BM25 + 0.35·trigram-Jaccard) × trust × decay`.
- F5. Proof-coupled trust: `record_served(run_id, keys, tier)` +
  `apply_outcome(run_id, status)`; trust = (positive+1)/(positive+negative+2);
  only host-proven outcomes move counters; all counters auditable.
- F6. Bi-temporal supersession via `supersedes:` frontmatter; superseded rows
  never surface, never deleted.
- F7. Per-project banks (git-remote/path hash); global + project scoping on
  every read.
- F8. Operator veto: `quarantine(slug)` hides a memory everywhere, reversibly,
  without touching the file.
- F9. Lifecycle: auto-compaction of episode overflow to an archive DB
  (never-served-first), configurable caps, nothing silently deleted.
- F10. CLI for humans: add / recall / show / stats / reindex / compact / quarantine.

### Non-functional
- N1. **Zero third-party dependencies**; Python 3.11+ stdlib only; single file.
- N2. No server, no daemon, no network; safe under WAL for one writer + readers.
- N3. Graceful degradation: FTS5 missing → LIKE+Jaccard fallback, same API.
- N4. Deterministic given identical inputs (retrieval quality is regression-tested
  with a seeded precision@3 fixture in the host suite).
- N5. Memory must never break the host: all host integrations wrap Mneme calls
  in non-fatal guards (reference implementation does).

## 5. Architecture

```
files (canon, human-owned)          sqlite (engine, disposable)
  notes/lessons|facts|prefs/*.md --->  mem (mirror rows)  --+-- mem_fts (FTS5, porter)
       frontmatter + [[links]]        episodes (DB-canon) --+   LIKE fallback
                                       mem_stats (served/pos/neg/quarantine; SURVIVES reindex)
                                       mem_served (run_id, key, tier)
read paths: index_block (every task start) · recall (mid-task tool) · CLI show
write paths: add_note / drop a file / add_episode / host distiller (ADD-only)
```

Identity & versioning: file rows are identified by path and versioned by *your*
git; DB rows use bi-temporal supersession. One versioning authority per canon —
no duplicated history machinery.

## 6. What is deliberately NOT in scope

- Knowledge-graph store (evidence: ≈2% gain, real complexity cost).
- Required embeddings/vectors (optional future signal behind a flag; HRR
  similarity was prototyped-adjacent and deferred as unproven).
- LLM calls inside the library (the host may run an optional ADD-only distiller;
  the library never talks to a model).
- Cross-machine sync, multi-tenant auth, encryption-at-rest (host/OS concern).

## 7. Success metrics

Per the context-saturation finding, conversational-recall benchmarks are the wrong
test. Mneme is measured on:
1. **Read-path liveness**: smoke check (canary note → live prompt) passes; %
   of runs with a non-empty served index.
2. **Cross-session reuse**: fraction of successful runs that had ≥1 relevant
   prior-failure lesson in context.
3. **Served-vs-never-served ratio** per kind (graveyard detector).
4. **Trust calibration**: memories with negative-dominant counters exit the
   index organically (no manual pruning).
5. **Startup cost**: index block ≤ budget (default 4KB) regardless of store size.

### Field results (production host, first 36 hours)
- 24 active rows (11 lessons, 13 episodes), **223 serves**, outcomes +76/−104,
  under adversarial conditions (self-modifying agent, rollbacks, concurrent
  benchmark harness rewriting the repo).
- An unattended overnight run was served the index and correctly took a trust
  debit when it rolled back — proof-coupling worked before anyone watched it.
- A previously 3×-failed build succeeded with its three failed episodes served
  in context; the run's own memory of failure was part of the fix.

## 8. Distribution & compatibility

- Standalone: this repo — `mneme.py` (single file), README, PRD, portable tests.
- Reference host: Ktisis (integrated copy is the same code + engine hooks:
  ledger serve-metering, smoke command, optional distiller, chat/UI surfaces).
- Compatibility promise: the standalone core is extracted from the host sources;
  host-side improvements flow here by re-running the assembly. Adopting Mneme in
  other agents (Hermes-class chat agents, other frameworks) requires only the
  three-call wiring and MUST NOT require changes that degrade the reference host
  (operator constraint, 2026-07-05).

## 9. Risks

- FTS5 availability varies on exotic Python builds → N3 fallback (tested).
- A host that never reports outcomes gets flat trust (0.5) — retrieval still
  works; trust features are additive.
- Unbounded lesson growth (files) is owner-managed by design; the index budget
  caps the context cost regardless.
- Single-writer assumption: two processes writing simultaneously rely on SQLite
  WAL + short transactions; heavy multi-process write loads are out of scope.

## 10. Roadmap

- 0.1: this release (extraction parity with the production host).
- 0.2: retrieval-quality fixture shipped in-repo; optional embedding re-rank flag.
- 0.3: import adapters (Claude Code MEMORY.md, Obsidian vault mount as extra dir
  — already supported read-only via `extra_dirs`).
