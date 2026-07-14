# Mneme domain language

- **Canon note** — operator-owned Markdown for a durable lesson, fact, or preference. The file is authoritative; SQLite rows and vectors are derived.
- **Bank** — a global or Git-project-scoped retrieval partition. Reads combine global canon with the current project bank; project banks never cross-contaminate.
- **Episode checkpoint** — the minimal, idempotent run record written at first serve so an unclean exit still leaves history.
- **Episode finalization** — the explicit update that replaces a checkpoint's thin detail with a compact session summary while preserving the run identity. Finalization is **monotonic on richness**: a later thin close cannot erase a richer progressive body.
- **Progressive episode** — mid-session upgrade (`session_progress`) written on each turn so unclean exits still leave structured user/assistant detail.
- **Thin episode** — checkpoint stub or sub-80-char body. Kept for audit, hard-demoted in retrieval, preferred for archive on compact overflow.
- **Evidence** — an external, auditable signal that a recalled memory was actually used in a proven outcome. Retrieval alone is not evidence of use.
- **Memory runtime** — the Hermes-specific policy module that maps session cwd to a bank, applies write scope, attributes recalled keys, and finalizes episodes over the host-agnostic engine.
- **Host package** — `host/hermes/` in the Mneme repo: source of truth for the Hermes MemoryProvider + runtime, deployed into a profile by `scripts/sync_hermes_bundle.py`.
- **Semantic health** — whether encoder files are configured, the encoder is loadable, vectors cover live rows, and any degradation reason is visible.
- **Episode health** — thin/rich/finalized ratios and mean body length over live episodes; high thin ratios are operational warnings, not silent success.
- **Conflict candidate** — two active notes in the same bank and kind whose titles are similar enough to require an explicit supersession decision. Detection never mutates canon.

