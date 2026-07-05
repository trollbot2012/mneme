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
A memory's rank moves only on *proven outcomes* — served into a run that verifiably
succeeded, it rises; served into a failure, it sinks (Laplace-smoothed success
ratio, counters stored and auditable). The model's opinion of a memory never moves it.

## How it works

```
your-agent/
  mneme-data/
    notes/                 <- CANON: plain markdown, yours, git-friendly
      lessons/*.md            frontmatter: kind, tags, keywords, pinned, supersedes
      facts/*.md
      preferences/*.md
    mneme.db               <- DISPOSABLE: SQLite FTS5 index + episode rows
                              delete it, reindex, nothing is lost
```

- **Retrieval**: FTS5/BM25 with porter stemming → trigram-Jaccard re-rank →
  × trust × recency-decay (episodes only). Pure-Python LIKE fallback if your
  SQLite lacks FTS5.
- **Banks**: per-project scoping (hash of git remote / path) — project A's facts
  never surface in project B.
- **Index block**: budget-capped (default 4000 chars), pinned notes always
  included, per-kind shares so run history can't starve lessons.
- **Lifecycle**: episodes auto-compact to an archive db past a cap. Quarantine
  hides a bad memory everywhere without touching your file.

## One layer, one interface

Mneme is the sole memory layer — research sources (Holographic, Hindsight,
Honcho, MEMORY.md, and Mneme's own lineage) are folded into ONE engine, not
routed between. The agent-facing API:

| verb | what it does |
|---|---|
| `remember(title, body, kind=..., project=...)` | durable write (lesson/fact/preference) |
| `recall(query, project)` | ranked retrieval, sub-ms, lexical + trust |
| `record_served(run_id, keys, tier)` | meter what entered context |
| `record_outcome(run_id, status)` | proof-coupled trust: proven outcomes move rank |
| `summarize_user_model(project)` | deterministic operator model from preferences |
| `export_memory(path)` / `import_memory(snapshot)` | readable MEMORY.md-style snapshot, round-trips |
| `compact()` | archive episode overflow — never silent deletion |
| `audit()` | store health: graveyard ratio, trust distribution |
| `explain_recall(query)` | WHY each hit ranked: lexical/jaccard/trust/decay components |

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
mem.apply_outcome(run_id, "done")             # served memories earn trust
```

Writes: `mem.add_note("lesson", title, body, keywords=..., pinned=...)` — or just
drop a markdown file in the notes folder; the next reindex picks it up.

## CLI

```
python mneme.py --dir ./mneme-data add --kind lesson --title "..." [--body ...] [--pin]
python mneme.py --dir ./mneme-data recall "search terms"
python mneme.py --dir ./mneme-data show ["query"]     # the exact block an agent would get
python mneme.py --dir ./mneme-data stats | reindex | compact | quarantine <slug>
```

## Requirements

Python 3.11+. Nothing else. (FTS5 ships in standard CPython builds; a LIKE+Jaccard
fallback engages automatically where it doesn't.)

## Provenance

Extracted verbatim from the memory layer of **Ktisis**, a proof-gated coding agent —
where it runs in production under adversarial conditions (self-modifying runs,
rollbacks, concurrent benchmark harnesses). Field data in the PRD. The integrated
version adds engine hooks (serve-metering into a proof ledger, a smoke check that
asserts a canary note reaches the live planner prompt, an optional post-run
distiller); this standalone core is the same code minus the host.
