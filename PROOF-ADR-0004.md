# PROOF — ADR 0004: optional embedding rerank (UNION-then-rerank)

Evidence ledger for [docs/ADR-0004-optional-embedding-rerank.md](docs/ADR-0004-optional-embedding-rerank.md).
Sections land in gate order; the baseline below is recorded in git strictly
before any engine change and is the fixed denominator for every uplift claim.

## Baseline (pre-implementation, engine commit 2c3e1df)

Measured with the engine at its pre-ADR head: `mneme.py` sha256
`98df68f5599bed646984571463a786892f57a568514c4292b3842cea08103d40` (byte-identical
to commit `2c3e1df`, the decision-doc-only commit; verified before and after the
run). Fixture and match rule are the widened ADR-0004 versions (commit `9417def`):
110 target queries (80 paraphrase + 30 zero-overlap), keyword class derived as
`query == title`, and the unified exact-title `is_hit` rule — one ruler for every
p@3 number in this document.

Command (deployment interpreter):

```
C:\Python314\python.exe bench\scale_bench.py --scales 1000 2000
```

Output:

```
provider=mneme | 110 target queries (80 paraphrase + 30 zero-overlap), precision@3 under distractor noise
 scale   seed_ms    p@3   para   zero     kw   p50_ms   max_ms
  1000    1606.3    0.4   0.55    0.0    1.0     0.51     3.34
  2000    3520.3    0.4   0.55    0.0    1.0     0.32     73.8
```

| metric                 | scale 1000 | scale 2000 |
|------------------------|-----------:|-----------:|
| p@3 overall            |      0.400 |      0.400 |
| p@3 paraphrase (80)    |      0.550 |      0.550 |
| p@3 zero-overlap (30)  |      0.000 |      0.000 |
| p@3 keyword (110)      |      1.000 |      1.000 |
| recall p50 (ms)        |       0.51 |       0.32 |

All four p@3 values reproduced exactly on a repeat run at scale 1000
(p50 varied 0.46–0.61 ms wall-clock; hit counts identical). These values are
committed as `BASELINE_P3_OVERALL` / `BASELINE_P3_PARAPHRASE` /
`BASELINE_P3_ZERO_OVERLAP` / `BASELINE_P3_KEYWORD` / `BASELINE_P50_MS` in
`bench/scale_bench.py`. **Gate 3's +0.10 uplift and the keyword non-regression
are defined against these constants, not against re-runs.** The historical
0.56–0.60 p@3 figures came from the old 25-query fixture under the retired
40-char-substring match rule and are explicitly NOT the baseline; the drop to
0.400 is the expected ceiling loss from adding 30 queries that lexical
retrieval structurally cannot reach (zero porter-stem overlap, enforced
per-entry by `test_adr0004.py` against the production fts5 tokenizer).

Companion artifacts already pinned at this state:

- `bench/golden_off_path.json` — off-path golden (220 recalls, 1609 pinned
  hits, scores to 9 places) captured from the same pre-ADR engine; gate 6's
  byte-identity net.
- `bench/scale_bench.py --assert` — gate-4 battery (embeddings-off p50 ≤ 2 ms,
  embeddings-on p50 ≤ 20 ms + per-text encode guard); fails loud if model
  files are absent, exits 0 today with the off-arm green against the pre-ADR
  engine (`[embeddings-off] scale=1000 p@3=0.400 p50=0.46ms ... ASSERT OK`).

## Gates (to be completed by later work packages)

- Gate 1 (fixture ≥100 incl. zero-overlap class): green at commit `9417def` —
  `uv run --python 3.11 --with pytest pytest test_adr0004.py -k "fixture or zero_overlap" -q` → `5 passed`.
- Gate 2 (union vs rerank-only divergence): pending engine work.
- Gate 3 (p@3 uplift ≥ +0.10, keyword non-regression): baseline above; fitted
  `w_sem` grid and uplift table pending.
- Gate 4 (p50 budgets, no per-candidate encode): assert-mode home landed;
  embeddings-on numbers pending.
- Gate 5 (zero encodes on unchanged reindex): pending.
- Gate 6 (embeddings-off byte-identical, wheel-absent suite green): golden pin
  landed; full battery pending.
