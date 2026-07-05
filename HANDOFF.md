# Mneme — try it in five minutes

You got this file because someone thinks your AI agent deserves memory that
actually gets read back. Mneme is a persistent memory layer for AI agents:
one Python file, zero dependencies, no server, no cloud, no embeddings
required. Your notes live as plain markdown you own; retrieval is
sub-millisecond; and memories earn or lose rank based on whether the runs
that used them *provably succeeded* — a mechanism no other memory layer has.

## Step 1 — install (human, one command)

macOS / Linux / Git Bash:

    curl -fsSL https://raw.githubusercontent.com/trollbot2012/mneme/master/install.sh | sh

Windows PowerShell:

    iwr -useb https://raw.githubusercontent.com/trollbot2012/mneme/master/install.ps1 | iex

That puts the engine at `~/.mneme/mneme.py`, a `mneme` CLI on your user bin,
runs a self-test, and drops the docs beside it. Needs Python 3.11+ and
nothing else. (Suspicious of piped installers? Reasonable — it's ~50 lines,
read it first: same URL without the `| sh`.)

Try it immediately:

    mneme --dir ~/.mneme/data add --kind lesson --title "Deploys fail without STAGING_BUCKET set"
    mneme --dir ~/.mneme/data recall "why did the deployment fail"
    mneme --dir ~/.mneme/data show          # the exact block an agent would be given

## Step 2 — wire it into your agent (your AGENT does this, not you)

Paste the following prompt to your coding agent (Claude Code, or anything
that can read files and edit your agent's source):

---

> **Task: install the Mneme memory layer into my agent system.**
>
> The engine is already on this machine at `~/.mneme/mneme.py` (if missing,
> fetch https://raw.githubusercontent.com/trollbot2012/mneme/master/mneme.py).
> Your complete installation spec is `~/.mneme/AGENT_SETUP.md` — read it
> FIRST and follow it exactly. Summary of what it requires of you:
>
> 1. Survey my agent's code before writing anything: find the ONE place it
>    assembles model context at task/session start, find where it learns a
>    task truly succeeded or failed, find its tool surface, find its state
>    directory, and inventory any existing memory stores (mount them
>    read-only via `extra_dirs`; never migrate destructively).
> 2. Vendor `mneme.py` into the agent's source tree and wire three calls:
>    `index_block()` injected at every task start (this is the non-negotiable
>    guaranteed read path), `record_served()`/`record_outcome()` on proven
>    outcomes, and `recall()` as a model-callable tool.
> 3. Wrap every Mneme call in a non-fatal guard — memory must never break a run.
> 4. You are NOT done until the acceptance checks in AGENT_SETUP.md pass,
>    especially the smoke check: a canary note's unique token must appear in
>    a REAL captured prompt/request, then vanish after deletion. Writes prove
>    nothing; show me the evidence.
>
> Report back: where each call landed (file:line), the smoke evidence, and
> what existing memory stores you found and how you handled them.

---

## What you get once it's wired

- **Guaranteed recall**: a budget-capped, relevance-ranked memory block in
  every task's context — lessons from failures, project facts, your
  preferences, run history. Never a raw dump.
- **Proof-coupled trust**: memories served into runs that verifiably succeed
  rise; ones that ride failures sink. The model's opinion never moves rank.
- **Project banks**: notes scoped to one repo never leak into another.
- **Your data, readable**: plain markdown files + `mneme export` for a
  single-document snapshot. Delete the SQLite index anytime; it rebuilds.
- **Auditability**: `audit()` shows what's actually being read (write-only
  memory is the #1 failure mode of agent memory — Mneme measures it), and
  `explain_recall()` shows exactly why any memory ranked where it did.

Honest limits, so you're not oversold: retrieval is lexical (BM25 + porter
stemming). Measured precision@3 on adversarially paraphrased queries is ~0.56
and stays flat from 100 to 2000 entries (`bench/scale_bench.py` in the repo) at
~1ms median recall — it does not degrade with scale, but synonym-only queries
that share no words with your notes are its ceiling; adding `keywords:` to notes
lifts them. Semantic embedding re-rank is on the roadmap as an optional flag.

Repo, PRD, research provenance, benchmarks: https://github.com/trollbot2012/mneme
