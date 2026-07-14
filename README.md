# Mneme

**A persistent memory layer for AI agents. One file. Zero dependencies. Your notes stay yours.**

Mneme gives any LLM agent durable memory — lessons from failures, project facts,
user preferences, and run history — with a guarantee most memory systems can't make:
**every memory has a read path**. Memory that is never read back is not memory;
it's a write-only graveyard. Mneme's index block is generated at every task start,
every serve is metered, and one command proves the whole loop end to end.

```
pip? no. server? no. embeddings? no. just:  python mneme.py --dir ./m add --title "..."
```

## Hermes production layout

The engine stays one file. Hermes host policy lives under `host/hermes/` and is
deployed into a profile with:

```powershell
python scripts/sync_hermes_bundle.py --profile $env:LOCALAPPDATA/hermes/profiles/aletheon
python scripts/verify_hermes_bundle.py --profile $env:LOCALAPPDATA/hermes/profiles/aletheon
```

See [docs/LAYOUT.md](docs/LAYOUT.md) and [docs/ADR-0006-episode-quality-and-host-layout.md](docs/ADR-0006-episode-quality-and-host-layout.md).

## Install (one command)

macOS / Linux / Git Bash:

```sh
curl -fsSL https://raw.githubusercontent.com/trollbot2012/mneme/master/install.sh | sh
```

Windows PowerShell:

```powershell
iwr -useb https://raw.githubusercontent.com/trollbot2012/mneme/master/install.ps1 | iex
```

User-space only (no sudo/admin), self-tests on install, needs Python 3.11+
and nothing else. Then hand your AI agent `~/.mneme/HANDOFF.md` and it wires
itself in. Or skip the installer entirely: `mneme.py` is one file — copy it.

Standard `curl | sh` trust model applies: the script fetches over HTTPS from
this repo and runs a self-test. Setting `MNEME_REPO` points the installer at a
different origin and it will execute whatever that host serves — only override
it with a trusted HTTPS source. Prefer to read first? Same URL without `| sh`.

## Why another memory layer

Built after a deep research pass across Mem0, Zep/Graphiti, MemGPT/Letta, A-MEM,
Claude Code's file memory, and holographic SQLite stores (full provenance in
[PRD.md](PRD.md)). The verified findings that shaped it:

- **BM25 lexical search is load-bearing in every flagship system** — Zep runs it
  co-equal with vectors; Mem0 V3 fuses it in. You don't need an embedding model.
- **Never destroy at write time** — Mem0 and Zep independently converged on
  append-only writes with read-time conflict resolution. Mneme supersedes
  (bi-temporal `valid_at`/`invalid_at`), never deletes.
- **Extraction is a cost optimization, not an accuracy win** — full context beats
  extracted memory when it fits. So Mneme keeps raw canon (your markdown files)
  and distills only to fit a startup budget.
- **Graph memory buys ~2%** on its vendors' own benchmarks. Skipped. A links
  table gives Zettelkasten linking without a graph engine.

And one mechanism the researched systems don't have: **proof-coupled trust**.
A memory's rank moves only on *proven outcomes* of runs where it was actually
**used** — pass `used_keys` at `record_outcome` and a consulted memory rises on
verified success, sinks on failure (Laplace-smoothed ratio, counters stored and
auditable). Merely being retrieved earns nothing: a win with no usage signal is
neutral, so heavily-served-but-useless notes can't ride other work's success.
The model's opinion of a memory never moves it.

## How it works

```
your-agent/
  mneme-data/
    notes/                 <- CANON: plain markdown, yours, git-friendly
      lessons/*.md            frontmatter: kind, tags, keywords, pinned, supersedes
      facts/*.md
      preferences/*.md
    notes/.mneme/          <- CANON SIDECARS: trust counters, quarantine
                              blocklist, repo-note promotions (yours, tiny)
    mneme.db               <- DISPOSABLE: SQLite FTS5 index + episode rows
                              delete it, reindex, nothing is lost — trust and
                              quarantine restore from the canon sidecars
```

- **Identity**: every note has a stable id (frontmatter `id:`, or derived from
  its title) — rename or move a file, move the whole repo, sync across hosts:
  its trust and quarantine state follow it.
- **Retrieval**: FTS5/BM25 plus optional semantic union/rerank × trust × recency-decay (episodes
  only), with df-aware query pruning (a query term matching >10% of the corpus
  is dropped — measured up to 100× faster recall at 100k rows, with equal-or-
  better precision). Lexical precision holds flat under distractor noise from
  100 to 2000+ entries (see `bench/`); synonym-only queries benefit from
  author-supplied `keywords:`. Pure-Python LIKE+Jaccard fallback if your
  SQLite lacks FTS5.
- **Provenance**: notes ingested from a *repo's* memory dir are UNVERIFIED —
  rendered only under a labelled "Unverified" section, never able to outrank
  your own canon, never earning trust, never self-pinning, until you
  `promote` them (promotion is bound to the note's content hash: if the repo
  edits the file afterwards, it demotes itself).
- **Banks**: per-project scoping (hash of git remote / path) — project A's facts
  never surface in project B, and one project can never supersede or evict
  another's memories.
- **Index block**: budget-capped (default 4000 chars, headers/separators
  counted), pinned notes first, per-kind shares so run history can't starve
  lessons. Every served key is guaranteed to appear in the emitted block.
- **Lifecycle**: episodes auto-compact to an archive db past a **per-bank** cap
  (`max_episodes`, default 2000/bank). Quarantine hides a bad memory everywhere
  without touching your file. Project-scoped notes are written under
  `<project>/.ktisis/memory` by default — override via the `project_subdir`
  config key for a non-Ktisis layout.

## One layer, one interface

Mneme is the sole memory layer — research sources (Holographic, Hindsight,
Honcho, MEMORY.md, and Mneme's own lineage) are folded into ONE engine, not
routed between. The agent-facing API:

| verb | what it does |
|---|---|
| `remember(title, body, kind=..., project=...)` | durable write (lesson/fact/preference) |
| `recall(query, project)` | ranked retrieval, sub-ms, lexical + trust |
| `record_served(run_id, keys, tier)` | meter what entered context |
| `record_outcome(run_id, status, used_keys=[...])` | proof-coupled trust: proven outcomes of USED memories move rank |
| `promote(slug)` / `unpromote(slug)` | trust (content-hash-bound) / distrust a repo-suggested note |
| `quarantine(slug)` | operator veto — survives any db rebuild (canon blocklist) |
| `summarize_user_model(project)` | deterministic operator model from preferences |
| `export_memory(path)` / `import_memory(snapshot)` | readable MEMORY.md-style snapshot, round-trips |
| `compact()` | archive episode overflow — never silent deletion |
| `audit()` | store health: graveyard ratio, trust distribution |
| `explain_recall(query)` | WHY each hit ranked: lexical/jaccard/trust/decay components |
| `finalize_episode(run_id, goal, status, detail, project)` | replace an early checkpoint with a compact final summary |
| `find_conflicts(project)` | report likely competing active notes; never mutates canon |

`index_block(project, task_text, run_id)` remains the host-integration hook —
the guaranteed read path injected at every task start.

## Wire it into any agent — three calls

```python
from mneme import Mneme

mem = Mneme("mneme-data/mneme.db", "mneme-data/notes")
mem.reindex(project_dir)                      # cheap mtime scan

# 1. task start: inject memory into the model's context
block = mem.index_block(project_dir, task_text, run_id=run_id)
prompt = f"{block.text}\n\n{task_prompt}"

# 2. let the model search memory mid-task (optional tool)
hits = mem.recall("flask blueprint 404", project_dir)

# 3. close the loop with a PROVEN outcome ("done" | "rolled_back" | ...)
used = mem.served_keys(run_id, tier="recall")
mem.record_outcome(run_id, "done", used_keys=used or None)
#   record_outcome is idempotent per run_id — a retry/duplicate close is a no-op,
#   and only mechanical results or an explicit operator verdict should call it.
```

Writes: `mem.remember(title, body, kind=..., keywords=..., pinned=...)` — or just
drop a markdown file in the notes folder; the next reindex picks it up.

## CLI

```
python mneme.py --dir ./mneme-data add --title "..." [--kind lesson|fact|preference] [--body ...] [--keywords ...] [--pin] [--supersedes <slug>]
python mneme.py --dir ./mneme-data recall "search terms"
python mneme.py --dir ./mneme-data show ["query"]        # the exact block an agent would get
python mneme.py --dir ./mneme-data stats | audit | conflicts
python mneme.py --dir ./mneme-data reindex | compact
python mneme.py --dir ./mneme-data quarantine <slug> [--off]   # --off un-quarantines
python mneme.py --dir ./mneme-data export [--out snapshot.md] | import snapshot.md
```

## Requirements

Python 3.11+. Nothing else. (FTS5 ships in standard CPython builds; a LIKE+Jaccard
fallback engages automatically where it doesn't.)

## Optional: semantic recall (ADR-0004)

Paraphrase-robust recall via a local MiniLM ONNX encoder — opt-in, and the core stays zero-dependency.
Provision the sha256-pinned model files (~23MB; the library itself never downloads anything): `scripts/provision_embeddings.ps1` (Windows) or `scripts/provision_embeddings.sh` (POSIX).
Install the one optional wheel: `pip install "onnxruntime>=1.20"` (or `pip install "mneme-memory[embeddings]"`).
With both present (`models/` next to the store's db, or the `embed_model_dir` config key), embeddings activate automatically. A missing/broken encoder still degrades non-fatally to lexical-only, but `stats()["embeddings"]` now exposes the configured state, model directory, coverage, and exact error. Episodes are vectorized too; their existing time decay still controls final ranking.

## Provenance

Extracted verbatim from the memory layer of **Ktisis**, a proof-gated coding agent —
where it runs in production under adversarial conditions (self-modifying runs,
rollbacks, concurrent benchmark harnesses). Field data in the PRD. The integrated
version adds engine hooks (serve-metering into a proof ledger, a smoke check that
asserts a canary note reaches the live planner prompt, an optional post-run
distiller); this standalone core is the same code minus the host.
