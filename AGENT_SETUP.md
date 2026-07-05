# AGENT_SETUP.md — install Mneme into a host agent (instructions FOR an AI agent)

You are an AI agent tasked with wiring Mneme (this repo's `mneme.py`) into a host
agent system. This document is your complete spec. Follow the acceptance checks —
your installation is NOT done until they pass. Do not claim success without them.

## What you are installing

A file-canon memory layer: markdown notes owned by the operator + a disposable
SQLite index. Your job is to give it a **guaranteed read path** in the host — the
single non-negotiable requirement. A memory layer that is only written to is a
failed installation, even if every write works.

## Step 0 — Survey the host (do this before writing any code)

Answer these by reading the host's code, not its docs (docs lie; read paths don't):

1. **Context assembly point**: find the ONE place the host builds the model's
   context/prompt at task or session start. If there are several, find the one
   ALL tasks flow through. This is where `index_block()` gets injected.
2. **Outcome signal**: find where the host learns a task/run/job truly succeeded
   or failed (exit codes, verification gates, job results). Map to statuses:
   proven success → `"done"`, proven failure → `"rolled_back"`. If the host has
   no proven-outcome signal, wire serves only — trust stays flat at 0.5 and
   retrieval still works; say so in your report.
3. **Tool surface**: how the model requests things mid-task (tool calls, slash
   commands). That's where `recall` gets exposed.
4. **State home**: where the host keeps per-install mutable state. Mneme's data
   dir goes there: `<state>/mneme/notes/` + `<state>/mneme/mneme.db`.
5. **Existing memory surfaces**: list every current memory-like store. Decide:
   mount as read-only canon (`extra_dirs=[...]` — zero migration, best default),
   or leave untouched. NEVER delete or migrate destructively.

## Step 1 — Install

Copy `mneme.py` into the host's source tree (vendor it — it is one stdlib-only
file; do not add a pip dependency unless the host already uses pip freely).
Python 3.11+ required.

## Step 2 — Wire the three calls

```python
from mneme import Mneme  # adjust import to where you vendored it

mem = Mneme(state_dir / "mneme" / "mneme.db", state_dir / "mneme" / "notes",
            extra_dirs=[<existing read-only note dirs>])

# (a) at EVERY task/session start — inside the context assembly point:
mem.reindex(project_dir)          # cheap mtime scan; project_dir may be None
ib = mem.index_block(project_dir, task_or_session_text, run_id=run_id)
context = ib.text + "\n\n" + existing_context     # ib.keys = what was served

# (b) at task/run end, ONLY on proven outcomes:
mem.apply_outcome(run_id, "done" | "rolled_back")   # anything else = neutral

# (c) as a model-callable tool (mid-task search):
hits = mem.recall(query, project_dir)   # list of {kind,title,body,trust,score}
if run_id: mem.record_served(run_id, [h["key"] for h in hits], "recall")
```

Rules:
- Wrap EVERY Mneme call in a non-fatal guard (try/except + host logging).
  Memory must never break a run. The reference host treats this as law.
- `run_id` is any stable per-run string the host already has.
- Do not put secrets in notes. Do not let the model edit note files directly —
  writes go through `add_note` (same-slug = operator-owned update) or the
  operator's own editor.

## Step 3 — Give the operator a write path

Expose `add_note` wherever the operator lives (a slash command, CLI, or UI form):
`mem.add_note(kind, title, body, keywords=..., tags=..., pinned=..., repo=...)`
with kind ∈ lesson|fact|preference. `pinned=True` = always in context.

## Step 4 — Acceptance checks (ALL must pass; show evidence)

1. **Smoke (the only proof that matters)**: `add_note` a canary lesson with a
   unique token → start a fresh task/session → capture the EXACT context the
   model received (request dump, debug log, or a capture hook) → assert the
   token appears in it → delete the canary → assert it's gone after reindex.
   Writes prove nothing. Only the token inside a real prompt proves memory.
2. **Bank isolation**: a note added with `repo=project_A` must NOT appear in
   `index_block(project_B, ...)`.
3. **Rebuild**: delete `mneme.db`, `reindex()`, and assert `index_block` output
   for the same query is identical (trust counters reset is expected and OK).
4. **Non-fatality**: point Mneme at an unwritable path and assert the host still
   completes a task (memory failure logged, run unharmed).
5. **Outcome coupling** (if Step 0.2 found a signal): run one succeeding and one
   failing task; assert the served memories' positive/negative counters moved
   (query `mem.stats()["most_served"]`).

Report to the operator: where each of the three calls landed (file:line), the
smoke evidence, and anything from Step 0.5 you mounted or deliberately left alone.

## Failure modes to avoid (learned in production)

- **Write-only graveyard**: wiring add/recall but not `index_block` at task
  start. The index block is the product; the rest is plumbing.
- **Fragile delivery**: any wiring that depends on how the host is *launched*
  (env inheritance, cwd) will rot. Resolve paths from the host's own config.
- **LLM-managed memory**: do not add a step where a model rewrites or deletes
  notes in place. Supersession and quarantine exist; silent rewrites are the
  documented failure mode of other systems.
- **Unpinned prose**: if the operator has rules that MUST hold, they belong in
  the host's enforcement (checks/hooks), not in a memory note. Notes are
  advisory context.

---

---

## Writing a host profile

Before handing this document to an installing agent, append a **host profile**
section for your specific system: the verified context-assembly point, state
home, outcome signal, tool surface, existing memory stores to mount or avoid,
and where the agent can capture a real prompt for the smoke check. The more
verified facts you pin, the fewer the agent has to rediscover.
