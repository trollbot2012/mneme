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

## Headline: before/after per class (deployment interpreter)

`C:\Python314\python.exe bench\engine_gate.py` (onnxruntime 1.26.0,
numpy 2.4.3), real `Mneme.recall`, per-class p@3, 2026-07-13:

```
110 fixture queries (80 paraphrase + 30 zero-overlap) + 110 keyword-identical | engine-level per-class p@3 (real Mneme.recall)
-- scale 1000 --
  lexical-only             p@3=0.400 para=0.550 zero=0.000 kw=1.000 p50= 0.38ms
  embeddings-on (union)    p@3=0.600 para=0.662 zero=0.433 kw=1.000 p50= 5.21ms  backfill: 1000 rows / 8 passes, per-pass p50 95ms max 398ms total 1.1s
  rerank-only (top_n=0)    p@3=0.418 para=0.575 zero=0.000 kw=1.000 p50= 4.99ms  backfill: 1000 rows / 8 passes, per-pass p50 98ms max 277ms total 1.0s
-- scale 2000 --
  lexical-only             p@3=0.400 para=0.550 zero=0.000 kw=1.000 p50= 0.20ms
  embeddings-on (union)    p@3=0.627 para=0.675 zero=0.500 kw=1.000 p50= 9.02ms  backfill: 2000 rows / 16 passes, per-pass p50 98ms max 180ms total 1.6s
  rerank-only (top_n=0)    p@3=0.436 para=0.600 zero=0.000 kw=1.000 p50= 8.19ms  backfill: 2000 rows / 16 passes, per-pass p50 100ms max 269ms total 1.8s
```

| class (scale 1000)     | baseline (commit A) | fitted default | delta  |
|------------------------|--------------------:|---------------:|-------:|
| p@3 overall            |               0.400 |          0.600 | +0.200 |
| p@3 paraphrase (80)    |               0.550 |          0.662 | +0.112 |
| p@3 zero-overlap (30)  |               0.000 |          0.433 | +0.433 |
| p@3 keyword (110)      |               1.000 |          1.000 |  0.000 |

The lexical-only row reproduces the commit-A baseline constants exactly (same
fixture, same `is_hit` rule, same interpreter) — the off path is behaviorally
frozen. Rerank-only scores 0.000 on the zero-overlap class *structurally*
(BM25 can never nominate a zero-stem-overlap note, and with `embed_top_n=0`
nothing else can either): the ADR's rejected variant is measurably rejected.

## w_sem grid (gate 3 fit — pre-registered rule, deployment interpreter)

Selection rule, committed in `bench/engine_gate.py` BEFORE the grid ran:
**maximize paraphrase p@3 subject to keyword-identical p@3 ≥
BASELINE_P3_KEYWORD (1.000); ties → smaller w_sem, then `clip_abs`.**

`C:\Python314\python.exe bench\engine_gate.py --wsem-grid`, engine at commit B
(both norm branches present), one 1000-row store, full 2×8 table verbatim:

```
backfill during build: 1000 rows / 8 passes, per-pass p50 96ms max 325ms total 1.0s
w_sem grid at scale 1000, norms=('clip_abs', 'minmax') (shipped-norm-only when the losing branch has been removed)
norm       w_sem     p@3   para   zero     kw   p50ms
clip_abs     0.0   0.427  0.550  0.100  1.000    5.43
clip_abs     0.1   0.500  0.588  0.267  1.000    5.02
clip_abs     0.2   0.500  0.588  0.267  1.000    5.16
clip_abs     0.3   0.518  0.600  0.300  1.000    5.37
clip_abs     0.4   0.527  0.613  0.300  1.000    5.06
clip_abs     0.5   0.527  0.613  0.300  1.000    4.49
clip_abs     0.6   0.527  0.613  0.300  1.000    5.23
clip_abs     0.7   0.536  0.625  0.300  1.000    5.34
minmax       0.0   0.427  0.550  0.100  1.000    5.40
minmax       0.1   0.500  0.588  0.267  1.000    5.27
minmax       0.2   0.518  0.600  0.300  1.000    5.22
minmax       0.3   0.536  0.613  0.333  1.000    5.17
minmax       0.4   0.545  0.625  0.333  1.000    5.41
minmax       0.5   0.564  0.637  0.367  1.000    5.08
minmax       0.6   0.600  0.662  0.433  1.000    5.53
minmax       0.7   0.600  0.662  0.433  1.000    5.18
FITTED (pre-registered rule: max paraphrase s.t. keyword >= 1.0; ties -> smaller w_sem, then clip_abs):
  w_sem=0.6 norm=minmax -> p@3=0.600 para=0.662 zero=0.433 kw=1.000 p50= 5.53ms
```

**Fitted: `w_sem = 0.6`, sem-norm = per-recall min-max.** The rule was
mechanical: max paraphrase is 0.662 (minmax at 0.6 and 0.7); the tie broke to
the smaller w_sem. Notably the MEASUREMENT overruled the hand argument — the
draft shipped `clip_abs` on the theory that per-set min-max launders relative
rank into absolute confidence, but minmax is strictly better at every
w_sem ≥ 0.2 and the keyword class never regresses, so the losing `clip_abs`
branch was **removed from `mneme.py` with this grid as the evidence** (the
0.65/0.35 lesson applied to a branch instead of a constant). The former
`_SEM_NORM` module switch is gone with it; `--wsem-grid` detects its absence
and sweeps only the shipped norm on post-fit builds.

### Blend rationale (shape of the formula)

`score = (w_lex·bm25n + w_jac·jaccard + w_sem·semn) × trust × credibility ×
decay`, in exactly one place (`Mneme._score_parts`, shared by `retrieve` and
`explain_recall`). `w_sem` applies to the min-max-normalized cosine over the
candidates that carry a vector; candidates without a vector contribute
`sem = 0.0` (identity). Off path (`embeddings_available == False`) `w_sem` is
forced to 0.0 and no `sem` key is ever attached — IEEE-exact identity with the
pre-ADR score.

### Truncation policy (baked into ENCODER_ID)

Embed text = `title + "\n" + keywords + "\n" + body`, hard-truncated to 256
wordpiece tokens including `[CLS]`/`[SEP]` (high-signal fields survive
truncation). Encoder identity string:
`all-MiniLM-L6-v2-quint8-avx2-rev1110a243-s256` — changing model, revision, or
policy changes the string, and the `mem_vec.encoder` mismatch triggers
re-encode.

## Latency, cold load, and first-backfill (gate 4 — measured, not estimated)

`C:\Python314\python.exe bench\scale_bench.py --assert --model-dir models` →
exit 0:

```
--assert: gate-4 latency + encode-guard battery at scale 1000
  [embeddings-off] scale=1000 p@3=0.400 p50=0.41ms max=2.20ms (budget 2.0ms)
  [embeddings-on] warmup recall (cold load, excluded from p50): 7.6ms
  [embeddings-on] scale=1000 p@3=0.582 p50=5.95ms max=9.05ms (budget 20.0ms)
ASSERT OK: all gate-4 budgets and encode guards held
```

- p50 embeddings-off 0.41ms ≤ 2ms; embeddings-on 5.95ms ≤ 20ms (engine_gate
  rows agree: 5.21ms @1k, 9.02ms @2k). Encode guard: ≤1 text encoded per
  recall, exactly 0 on an immediately repeated query (LRU) — per-TEXT counter,
  so a per-candidate encode smuggled into one batch call would trip it.
- **Cold encoder load** (ONNX session build + tokenizer + first encode,
  measured on a fresh Python314 process): **221ms**; warm query encode 0.7ms.
  The 7.6ms "warmup" above is small because the assert arm's backfill already
  paid the session build. Hosts pay the ~220ms once, at provider init via the
  synchronous warmup recall (WP8), never inside a turn's p50.
- **First-backfill stall, re-measured at `EMBED_BACKFILL_BATCH = 128`** (the
  ~1–2s-per-512 estimate in the ADR discussion was optimistic; these are the
  published numbers): per-128-row pass p50 **95–100ms**, max observed
  **398ms**; full-store backfill 1000 rows ≈ **1.1s**, 2000 rows ≈ **1.8s**
  (Python314, quint8 AVX2, single process). A single pass stays two orders of
  magnitude under the ~2s tuning threshold, so 128 stands; any one caller
  (e.g. provider init) stalls at most one pass, and the sweep amortizes the
  rest.

## Gate battery (all six, exact commands + decisive output)

Run matrix: **3.11 via `uv` = wheel-absent environment** (onnxruntime cannot
be imported — proves the off path and gate 6); **3.14 via `uv` with
onnxruntime = embeddings environment** (gates 2–5 pytest); **C:\Python314 =
deployment interpreter** (bench numbers, grid fit, gate 4 assert).

| gate | command | decisive output |
|------|---------|-----------------|
| 1 fixture | `uv run --python 3.11 --with pytest pytest test_adr0004.py -k "fixture or zero_overlap" -q` | `5 passed, 16 deselected` |
| 2 union≠rerank | `uv run --python 3.14 --with pytest --with onnxruntime pytest test_adr0004.py -k union_diverges -q` | `1 passed, 20 deselected` |
| 3 uplift | `uv run --python 3.14 --with pytest --with onnxruntime pytest test_adr0004.py -k uplift -q` | `2 passed, 19 deselected` (fitted-default test ACTIVE, no skip) |
| 4 latency+guards | `C:\Python314\python.exe bench\scale_bench.py --assert --model-dir models` | `ASSERT OK: all gate-4 budgets and encode guards held`, exit 0 |
| 5 zero encodes | `uv run --python 3.14 --with pytest --with onnxruntime pytest test_adr0004.py -k zero_encodes -q` | `3 passed, 18 deselected` (incl. edited-while-off heal + delete-the-DB rebuild) |
| 6 off-path identity | `uv run --python 3.11 --with pytest pytest -q` → `63 passed, 12 skipped`; `... -k golden -q` → `1 passed`; `... -k import_block -q` → `1 passed` | whole suite green wheel-absent; golden off-path replay exact to 9 places; onnxruntime AND numpy never enter `sys.modules` |

Full embeddings run: `uv run --python 3.14 --with pytest --with onnxruntime
pytest test_adr0004.py -q` → `21 passed`.

## Hosts must know

- **Pass `config=` to the constructor.** The Hermes provider passes its config
  dict wholesale with `embed_model_dir` defaulted to `<profile>/mneme/models`
  (WP8); unknown keys are dropped harmlessly. `''` resolves to
  `db_path.parent / "models"`.
- **Models dir**: `model_quint8_avx2.onnx` (23,046,789 bytes) + `vocab.txt`
  (231,508 bytes), provisioned by `scripts/provision_embeddings.{ps1,sh}`
  (sha256-pinned; the library itself ships zero network code; `models/` is
  gitignored). `embeddings: auto` activates iff both files exist; there is no
  forcing `on` — a missing model degrades silently to lexical-only (check the
  `stats()["embeddings"]` coverage block or the provider init log line).
- **First start on an existing store** pays the backfill: ~1s per 1000 notes
  (measured above), in 128-row passes — provider init stalls at most one pass;
  reindex/sweeps drain the rest. Cold encoder load ~220ms, paid once at init
  by the synchronous warmup recall.
- **Episodes are not vectorized in v1** (settled): paraphrased queries about
  past runs remain lexical-only until a decay-aware design exists.
- **`w_sem` is deliberately not exposed** in the provider config schema; it is
  a fitted engine constant. If live behavior disappoints, the honest fix is
  widening the fixture with sanitized real note/query pairs and re-running the
  grid — not nudging the constant.
- **Ktisis's vendored mneme copy remains divergent** (out of scope, separately
  flagged): the fleet knowingly runs mixed engines until that resync; do not
  record "all synced".
- Per-recall vector scan is O(bank rows) with no cache — comfortably inside
  budget at the asserted scales (5.2ms @1k, 9.0ms @2k); the sanctioned
  contingency (per-bank matrix keyed by `PRAGMA data_version`) is documented
  in a code comment and deliberately NOT built until a measurement fails.
