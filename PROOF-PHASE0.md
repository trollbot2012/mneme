# Phase 0 proof report — foundation promises made true

Date: 2026-07-05. Baseline: commit a190fdb (post-audit engine). Every claim
below is backed by a test in `test_phase0.py` that **fails on the old engine
and passes on the new one**, plus an old-API-compatible roundtrip probe for
the flagship promise.

## Headline before/after (identical script, both engines)

```
=== OLD ENGINE (a190fdb) ===
BEFORE rebuild: trust=(0, 1) quarantined=1
AFTER  rebuild: trust=None quarantined=0 vetoed_note_reinjected=True
VERDICT: LOST

=== NEW ENGINE (Phase 0) ===
BEFORE rebuild: trust=(0.0, 0.1) quarantined=1
AFTER  rebuild: trust=(0.0, 0.1) quarantined=1 vetoed_note_reinjected=False
VERDICT: PRESERVED
```

(The probe reports a `rolled_back` run served without a used signal: old
engine debits a full point, new engine a weak 0.1.) Two behavior changes in
one probe: (1) delete-the-db-and-reindex now preserves trust byte-for-byte
and keeps the security quarantine enforced — the old engine silently reset
trust and **re-injected an operator-vetoed note into the index block**;
(2) trust now measures usefulness, not retrievability — full-strength moves
require a `used_keys` attestation.

## Promise 1 — disposable DB is true

- Stable identity: frontmatter `id:` minted at `add_note` (preserved across
  update-in-place); hand-authored/repo notes get a deterministic
  content-derived id (`sha256(bank|kind|title-slug)`), so file renames, repo
  moves, and cross-host syncs keep their trust. Episodes keep `run:<id>`.
- `mem_stats` is keyed by `note_id`, never by filesystem path. Old DBs are
  migrated in place (stats merged through the id mapping).
- Canon sidecars under `global_dir/.mneme/` (operator-owned, unreachable from
  any repo): `trust.jsonl` (cumulative counters, appended on outcome,
  compacted at reindex, fill-only restore) and `blocklist.jsonl` (quarantine —
  restore always enforces, never silently lifts).
- Tests: `test_delete_db_roundtrip_preserves_trust_and_quarantine`,
  `test_rename_preserves_identity_hand_authored`,
  `test_rename_preserves_identity_frontmatter_id`.
  Old engine: trust=None, quarantine=0, vetoed note re-served. New: preserved.

## Promise 2 — repo notes cannot poison canon

- New `credibility` column: `operator` (global dir, own writes) vs `repo`
  (anything ingested from a repo's memory dir that isn't promoted).
- Unverified notes render ONLY under
  `Unverified notes suggested by this repo (verify before trusting):`,
  never inside the trusted kind sections; capped at 3 lines, last in budget.
- Structural guarantee: retrieval sorts credibility tier FIRST — an
  unverified note can never outrank trusted canon, regardless of lexical gap.
- Repo `pinned: true` is stripped at ingest (pinning is operator authority);
  repo notes cannot supersede operator notes; unverified notes never move
  trust, even if marked used.
- `promote(slug)` records id + **content hash** in the canon promoted
  registry: if the repo later edits the promoted file, the hash breaks and it
  demotes on the next reindex. Own `add_note(repo=...)` writes self-register.
- Tests: `test_repo_ingested_note_is_unverified`,
  `test_promotion_is_content_hash_bound`,
  `test_own_repo_writes_stay_trusted_across_reindex`.
  Old engine: the malicious note entered at trust 0.5, ranked FIRST for its
  query, and pinned itself into every task's block.

## Promise 3 — trust measures usefulness, not retrievability

- `apply_outcome(run_id, status, used_keys=None)`:
  - key in `used_keys` → full ±1.0 (host attests the memory was consulted);
  - with `used_keys` given, served-but-not-used → weak 0.1 negative on a
    proven win (retrieved constantly without helping is evidence against),
    nothing on a failure (takes no blame);
  - no `used_keys` → a WIN is NEUTRAL (a blanket weak-negative would erode
    all trust monotonically and invert ranking — adversarial verify F12);
    a FAILURE still nudges every served memory weakly down;
  - pinned notes never auto-credit; unverified notes never move until
    promoted; quarantined notes are FROZEN both directions; keys with no
    live mem row have no authority (no COALESCE-to-canon laundering);
  - credit lands once per NOTE (stable id), however many files served it.
- Tests: `test_served_but_unused_does_not_gain_trust`,
  `test_used_memory_gains_trust`, `test_pinned_memory_does_not_auto_credit`.
  Old engine: every served key got +1 on any successful run.

## Promise 4 — fleet-safe writes

- `PRAGMA busy_timeout=5000` + bounded exponential retry on residual
  `database is locked` for the hot write paths (record_served, apply_outcome,
  add_episode, quarantine, note indexing).
- Exactly-once outcomes cross-process: `BEGIN IMMEDIATE` +
  `INSERT OR IGNORE INTO run_outcome` rowcount claim — losers of the race
  see 0 rows and roll back cleanly instead of throwing or double-crediting.
- Test: `test_concurrency_hammer_exactly_once` — 6 processes × 20 runs racing
  on the SAME run ids: zero uncaught errors, positive == 20.0 exactly,
  120/120 episode writes survive, 20 outcome rows.
  Old engine: no busy_timeout (default 0), SELECT-then-INSERT guard.

## Adversarial verification round (post-implementation)

The Phase 0 diff was reviewed by a 4-lens adversarial workflow (25 agents:
migration/identity, sidecar security, trust semantics, concurrency), every
claim re-verified against the code: **20 confirmed findings, all fixed**,
1 rejected. The heavy ones: the migration's INNER JOIN silently destroyed
orphaned v0 quarantines (now LEFT JOIN + orphan preservation + blocklist
export); a torn migration became permanent (now one atomic BEGIN IMMEDIATE
with the guard re-checked inside); `apply_outcome` COALESCE'd a missing mem
row to trusted canon (now raw credibility, no authority on join-miss);
`unpromote()`'s zero-match path left an open transaction that crashed the
next outcome (now rolled back); weak-negative-on-wins inverted ranking for
hosts without `used_keys` (now neutral); repo notes leaked unlabelled through
`summarize_user_model` and the export/import round-trip (now credibility-
gated); frontmatter ids are bank-composed (a repo can't collide an operator
note's identity); duplicate-id files credit once; v0 archives are sidelined
instead of crashing compact forever. Deferred, documented: cross-process
sidecar file locking (appends are single-write batched, tmp names unique,
reader skips torn lines); derived-id reuse re-attaching an old veto to a
future same-titled note (inherent to content-derived identity — visible via
`explain_recall`/`audit`, operator lifts with `quarantine --off`).

Five regression tests cover the confirmed classes:
`test_migration_preserves_orphaned_quarantine`,
`test_unpromote_zero_match_leaves_no_open_transaction`,
`test_pruned_note_cannot_be_credited`, `test_quarantined_note_trust_is_frozen`,
`test_duplicate_note_id_credits_once`.

## Test matrix

| Suite | Old engine | New engine |
|---|---|---|
| test_phase0.py (15 tests) | **fails** (10/10 of the promise tests) | **15 passed** |
| test_mneme.py + test_unified.py + test_audit_fixes.py (39) | 39 passed* | **39 passed** |

*3 tests updated to the new trust contract (they now pass `used_keys` — same
intent, sharp signal): `test_trust_sinks_serially_failing_memory`,
`test_record_served_then_outcome_moves_trust`,
`test_apply_outcome_is_idempotent_per_run`.

## Semantics change hosts must know (upgrade note)

Without `used_keys`, successful runs no longer raise trust for served
memories — pass `used_keys` (the served keys the agent actually consulted or
acted on) at `record_outcome`/`apply_outcome` for full-strength signal.
Hermes: `mneme_feedback`/`apply_job_outcome` should forward the used set.
Ktisis: keys fetched through recall rounds are honest "used" evidence.
