# ADR 0006 — Episode quality, progressive finalization, host layout

Status: **ACCEPTED — 2026-07-14**

## Context

ADR-0005 added `finalize_episode` and a Hermes runtime, but live production
data still showed:

- ~100% thin episode bodies (checkpoint stubs or `0 turns` closes)
- progressive mid-session detail never written on unclean exits
- run-segment rotation orphaning open checkpoints without a summary
- thin stubs competing in semantic/lexical recall with real notes
- dual-copy drift risk between the canonical repo and the Hermes plugin
  (engine only was verified; provider/runtime lived only under the profile)

## Decision

1. **Progressive finalization** — the Hermes provider upgrades the same run
   row on every `sync_turn` (`session_progress`) and closes segments on
   rotate (`session_segment`) before opening a new evidence run id.
2. **Monotonic richness** — `finalize_episode` never replaces a richer body
   with a thinner one. Terminal status may still promote on tags/title.
3. **Quality-weighted retrieval** — thin/checkpoint episodes keep an audit
   trail but score with a hard multiplier (down to 0.15).
4. **Visible episode health** — `stats()["episodes"]` reports thin/rich/
   finalized ratios and mean body length; Hermes health treats high thin
   ratios as warnings.
5. **Host package as source of truth** — `host/hermes/{provider,runtime,
   plugin.yaml}` lives in the Mneme repo. `scripts/sync_hermes_bundle.py`
   deploys engine + adapter into a Hermes profile; verify covers all three.

## Consequences

- New multi-turn sessions leave structured summaries even without a clean
  `on_session_end`.
- Historical thin episodes remain; they are demoted, not reconstructed.
- Compact prefers archiving thin stubs before rich summaries on overflow.
- Editing the active profile plugin without re-syncing is a drift error.

## Verification

- `test_memory_gaps.py` covers monotonic finalize + thin demotion + episode
  health stats.
- Host lifecycle tests cover progressive upgrade and segment close.
- `sync_hermes_bundle.py --check` is green after deploy.
