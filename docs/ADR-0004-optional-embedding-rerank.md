# ADR 0004 — Optional embedding rerank (UNION-then-rerank)

Status: **PROPOSED — decision document only, build deferred** (operator
directive 2026-07-05: Phase 0 safety first; semantic rerank decided behind an
ADR, not built yet). Revisits ADR 0003 (stdlib-only bundle).

## Context

The lexical ceiling is Mneme's one remaining value-limiting defect. Measured:
p@3 0.56–0.60 on the 25-query paraphrase bench (scale-invariant to 2000+
rows), i.e. ~40% top-3 miss when the agent's phrasing doesn't share tokens
with the note. Reproduced concretely: "switch which llm handles a request"
misses a seeded "LiteLLM aliases" fact and a "429 fallback" lesson entirely —
the co-occurrence rescue never fires because results weren't strictly empty.
Every stdlib-only mechanism was bake-off-tested (jaccard, kwboost,
co-occurrence expansion): none lifts paraphrase p@3. The 2026-07-05 design
review (4-lens adversarial panel) concluded semantic recall belongs in core,
as a cached embedding column — not a per-recall model call.

Prerequisites now in place: Phase 0 (stable ids, canon sidecars, provenance
gate, usefulness-coupled trust, fleet-safe writes) and honest ranking weights
(jaccard dropped on measurement; one blend constant).

## Decision (proposed)

1. **`mem_vec(note_id, vec BLOB, content_hash)`** — a DERIVED cache in the
   disposable DB, like `mem_fts`. Embedded at write (`_upsert_locked`), keyed
   by `sha256(title+body+keywords)`: reindexing an unchanged corpus performs
   ZERO encodes. Vectors are never canonical; disposability holds.
2. **Encoder**: local ONNX all-MiniLM-L6-v2 (384-dim, ~22MB), via one
   OPTIONAL wheel (onnxruntime). Gate `embeddings_available` mirrors
   `fts_available`; absent → lexical-only, byte-identical to today. The
   stdlib-only CORE promise is preserved; the encoder is a host layer.
3. **UNION-then-rerank** (load-bearing): candidates = BM25 top-64 ∪ cosine
   top-N (brute dot over the bank's cached vectors), THEN blend
   `w_lex·bm25n + w_sem·cos`, × trust × credibility × decay. Rerank-only-
   over-BM25 is rejected: it cannot recover a zero-lexical-overlap note —
   the exact reproduced miss. Query embedding encoded once per recall
   (~10ms CPU), LRU-cached on repeated task text.
4. **Co-occurrence expansion is DELETED once embeddings are enabled** — it is
   a worse, slower cosine, and its O(rows) rebuild is the store's biggest
   scaling landmine (measured 4.6s cold @100k, invalidated on every write).
5. **`w_sem` is fit, not picked**: grid 0.0–0.7 by measured p@3 on the
   widened paraphrase fixture. No hand-chosen blend (that mistake was made
   once at 0.65/0.35 and survived two audits before measurement killed it).

## Rejected alternatives

- **HRR / holographic vectors in retrieval** — category error: unbind-via-
  correlation injects crosstalk that degrades fuzzy nearest-neighbor recall.
  HRR is a compositional-binding tool. If relationship structure is ever
  wanted, it's a typed `mem_edge(src_id, rel, dst_id)` table, not vector
  algebra.
- **sqlite-vec / ANN index** — brute cosine over ≤~128 pre-filtered
  candidates plus one bank's vectors is sub-millisecond; ANN pays only if
  BM25 pre-filtering is dropped, and it would make vectors canonical.
- **Per-recall LLM rerank** — the Hindsight latency failure mode; violates
  the guaranteed-read-path budget.
- **Author-supplied keywords as the fix** — already supported, helps, but
  demands discipline the write path can't enforce; it's mitigation, not
  architecture.

## Acceptance gates (must ALL pass before merge)

1. Paraphrase fixture widened to ≥100 queries (current 25 → p@3 resolution
   0.04), including a zero-lexical-overlap class.
2. UNION-then-rerank MUST diverge from rerank-only on the zero-overlap class
   (if it doesn't, the union isn't wired).
3. p@3 uplift ≥ +0.10 over lexical-only on the paraphrase fixture; no
   regression on the keyword-identical fixture.
4. Latency budget, asserted in `bench/scale_bench.py`: recall p50 ≤ 2ms
   embeddings-off (1k scale), ≤ 20ms embeddings-on; a guard forbids any
   per-candidate encode at recall time.
5. Unchanged-corpus reindex performs 0 encodes (content-hash cache test).
6. `embeddings_available=False` path byte-identical to the pre-ADR engine
   (regression suite green with the wheel absent).
