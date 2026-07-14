# ADR 0005 — Semantic episodes and a deep host runtime

Status: **ACCEPTED — 2026-07-14**

## Context

The production Hermes adapter passed `repo=None` on every call, so the engine's
bank isolation never operated. It also wrote an episode checkpoint on first
serve and later attempted another `add_episode` at session end; the idempotent
first-write-wins rule discarded the closing detail. Episodes were excluded from
the vector cache, making paraphrased questions about prior sessions lexical-only.
Finally, encoder load failures flipped semantic recall off without exposing the
reason in structured health.

## Decision

1. Hermes owns a **memory runtime** module. It resolves the per-session cwd,
   finds the Git root, applies write scope, records explicit recall use, and
   treats outcomes without used keys as neutral.
2. The engine adds `finalize_episode`. Checkpoint creation stays idempotent;
   finalization updates the same run row, FTS entry, and derived vector.
3. Live episodes participate in the same disposable vector cache as canon
   notes. Existing episode decay remains in the final score, so semantic recall
   does not remove temporal ranking.
4. `stats().embeddings` reports configured/available/error/model_dir/coverage.
   Lexical fallback remains non-fatal, but degradation is no longer silent.
5. Conflict detection is read-only. Resolution remains explicit through the
   existing `supersedes` field; the engine never guesses which canon note wins.

## Consequences

- Project facts stop bleeding across repositories in Hermes.
- Future finalized sessions can answer paraphrased continuity questions without
  retaining raw transcripts.
- Vector coverage now includes episodes, so a first upgrade may backfill more
  rows than ADR-0004's non-episode-only rollout.
- Hosts that checkpoint early must call `finalize_episode`; hosts that do not
  checkpoint can continue using `add_episode` alone.

## Verification

- `test_memory_gaps.py` covers visible encoder failure, checkpoint finalization,
  conflict reporting, and evidence health.
- The full canonical suite remains green under the deployment interpreter.
- The Hermes runtime suite covers project reads, auto write scope, and neutral
  no-use outcomes.

