"""Engine-level p@3 gate for ranking changes (real Mneme.recall, not the
raw-sqlite mechanism harness).

ADR-0004 comparator rows (per-class p@3 on the widened 110-query fixture):
  - lexical-only   {'embeddings': 'off'}                 — the pre-ADR path
  - embeddings-on  {'embeddings': 'auto'}                — UNION-then-rerank,
                                                           shipped default w_sem
  - rerank-only    {'embeddings': 'auto', 'embed_top_n': 0}
                                                         — the ADR's REJECTED
    variant (cosine rescores BM25 candidates but adds none), expressible as a
    plain config row: gate 2's comparator, no monkeypatch.

Usage:
    python bench/engine_gate.py               # three comparator rows at 1000/2000
    python bench/engine_gate.py --wsem-grid   # gate-3 fit (run on the deployment
                                              # interpreter, C:\\Python314)

w_sem SELECTION RULE — PRE-REGISTERED, MECHANICAL (ADR-0004 point 5: fitted,
never hand-picked; committed here BEFORE the grid was run):

    maximize paraphrase p@3
    subject to keyword-identical p@3 >= BASELINE_P3_KEYWORD
    ties -> smaller w_sem, then sem-norm 'clip_abs'

The original grid was 2x8 (sem-norm in {clip_abs, minmax} x w_sem 0.0..0.7),
measured 2026-07-13 on C:\\Python314\\python.exe (onnxruntime 1.26.0) with the
engine at commit B; the full table is committed verbatim in PROOF-ADR-0004.md.
'minmax' won under the rule (paraphrase 0.662 at w_sem=0.6 vs clip_abs's best
0.625 — the hand-asserted clip_abs rationale lost to measurement, the
0.65/0.35 lesson again) and the losing clip_abs branch was removed from
mneme.py with that grid as evidence, so on the shipped engine the grid sweeps
w_sem over the one remaining norm (a mneme build that still carries the
module-level _SEM_NORM switch is swept over both).

Historical note: this file also gated the 2026-07-05 freebie round (df-aware
pruning must not change p@3 — recorded green in PROOF-PHASE0.md); the
lexical-only row remains that configuration.
"""
import argparse
import shutil
import statistics
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import mneme as mneme_module  # noqa: E402  (grid needs the module, not just the class)
from mneme import Mneme  # noqa: E402
from scale_bench import (  # noqa: E402
    BASELINE_P3_KEYWORD,
    KEYWORD_QUERIES,
    TARGETS,
    ZERO_OVERLAP_TARGETS,
    _PARAPHRASE_TARGETS,
    distractors,
    is_hit,
)

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
GRID_W_SEM = [round(0.1 * i, 1) for i in range(8)]  # 0.0 .. 0.7 step 0.1


def build(scale, **cfg):
    """Seeded store at `scale` rows. When the config activates embeddings the
    vector backfill is drained here (`while mem._sync_vectors(): pass`) and
    every pass is TIMED — the per-batch first-backfill stall is a published
    PROOF number, measured, not estimated. Returns (mem, tmpdir, passes)
    where passes is [(rows_encoded, ms), ...] (empty on the lexical path)."""
    tmp = Path(tempfile.mkdtemp(prefix="mneme-gate-"))
    mem = Mneme(tmp / "m.db", tmp / "notes", config=cfg or None)
    corpus = [(t, b) for t, b, _ in TARGETS] + distractors(max(0, scale - len(TARGETS)))
    # direct DB seed (bulk): same shape add_note produces, at bench speed
    now = time.time()
    with mem._lock:
        for i, (title, body) in enumerate(corpus):
            key = f"file:/bench/note_{i}.md"
            mem._upsert_locked(key, "fact", "global", title, body, "", "",
                               key, False, now, now)
        mem._conn.commit()
    passes = []
    if getattr(mem, "embeddings_available", False):
        while True:
            t0 = time.perf_counter()
            n = mem._sync_vectors()
            dt_ms = (time.perf_counter() - t0) * 1000
            if not n:
                break
            passes.append((n, dt_ms))
    return mem, tmp, passes


def run(mem):
    """Per-class p@3 over the fixture: paraphrase / zero-overlap by index
    split (zero-overlap entries are LAST in TARGETS), keyword-identical via
    the derived KEYWORD_QUERIES. Keyword latencies stay OUT of p50 so p50
    keeps meaning 'fixture queries' (scale_bench convention)."""
    n_para = len(_PARAPHRASE_TARGETS)
    hits = [0, 0]  # [paraphrase, zero-overlap]
    lat = []
    for idx, (title, _body, query) in enumerate(TARGETS):
        t0 = time.perf_counter()
        top = mem.recall(query, top_k=3)
        lat.append((time.perf_counter() - t0) * 1000)
        if is_hit(title, [h["title"] for h in top]):  # ONE match rule (scale_bench.is_hit)
            hits[0 if idx < n_para else 1] += 1
    kw = sum(1 for title, _body, query in KEYWORD_QUERIES
             if is_hit(title, [h["title"] for h in mem.recall(query, top_k=3)]))
    return {"overall": (hits[0] + hits[1]) / len(TARGETS),
            "paraphrase": hits[0] / n_para,
            "zero_overlap": hits[1] / len(ZERO_OVERLAP_TARGETS),
            "keyword": kw / len(KEYWORD_QUERIES),
            "p50_ms": statistics.median(lat)}


def _fmt(r):
    return (f"p@3={r['overall']:.3f} para={r['paraphrase']:.3f} "
            f"zero={r['zero_overlap']:.3f} kw={r['keyword']:.3f} "
            f"p50={r['p50_ms']:5.2f}ms")


def main():
    print(f"{len(TARGETS)} fixture queries ({len(_PARAPHRASE_TARGETS)} paraphrase"
          f" + {len(ZERO_OVERLAP_TARGETS)} zero-overlap) + {len(KEYWORD_QUERIES)}"
          f" keyword-identical | engine-level per-class p@3 (real Mneme.recall)")
    rows = (
        ("lexical-only", {"embeddings": "off"}),
        ("embeddings-on (union)", {"embeddings": "auto",
                                   "embed_model_dir": str(MODELS_DIR)}),
        ("rerank-only (top_n=0)", {"embeddings": "auto",
                                   "embed_model_dir": str(MODELS_DIR),
                                   "embed_top_n": 0}),
    )
    for scale in (1000, 2000):
        print(f"-- scale {scale} --")
        for label, cfg in rows:
            mem, tmp, passes = build(scale, **cfg)
            if cfg.get("embeddings") != "off" and not mem.embeddings_available:
                print(f"  {label:<24} SKIPPED: embeddings unavailable "
                      f"(models missing under {MODELS_DIR}?) — row would be "
                      f"a silent lexical-only false pass")
                mem.close()
                shutil.rmtree(tmp, ignore_errors=True)
                continue
            r = run(mem)
            mem.close()
            shutil.rmtree(tmp, ignore_errors=True)
            extra = ""
            if passes:
                per = [ms for _n, ms in passes]
                total_s = sum(per) / 1000
                extra = (f"  backfill: {sum(n for n, _ in passes)} rows / "
                         f"{len(passes)} passes, per-pass p50 "
                         f"{statistics.median(per):.0f}ms max {max(per):.0f}ms "
                         f"total {total_s:.1f}s")
            print(f"  {label:<24} {_fmt(r)}{extra}")


def wsem_grid() -> int:
    """Gate-3 fit. Builds ONE embeddings-on store at scale 1000 (stored
    vectors are independent of w_sem and sem-norm — both act at recall time),
    then sweeps the grid and applies the PRE-REGISTERED selection rule from
    the module docstring. Prints the full table plus the fitted pick."""
    has_norm_switch = hasattr(mneme_module, "_SEM_NORM")
    norms = ("clip_abs", "minmax") if has_norm_switch else ("minmax",)
    mem, tmp, passes = build(1000, embeddings="auto",
                             embed_model_dir=str(MODELS_DIR))
    try:
        if not mem.embeddings_available:
            print(f"GRID FAIL: embeddings unavailable (models under "
                  f"{MODELS_DIR}?) — a lexical-only grid would be meaningless")
            return 1
        per = [ms for _n, ms in passes]
        print(f"backfill during build: {sum(n for n, _ in passes)} rows / "
              f"{len(passes)} passes, per-pass p50 {statistics.median(per):.0f}ms "
              f"max {max(per):.0f}ms total {sum(per) / 1000:.1f}s")
        print(f"w_sem grid at scale 1000, norms={norms} "
              f"(shipped-norm-only when the losing branch has been removed)")
        print(f"{'norm':<10} {'w_sem':>5}  {'p@3':>6} {'para':>6} {'zero':>6} "
              f"{'kw':>6} {'p50ms':>7}")
        results = []
        saved = getattr(mneme_module, "_SEM_NORM", None)
        try:
            for norm in norms:
                if has_norm_switch:
                    mneme_module._SEM_NORM = norm  # harness-only monkeypatch
                for w in GRID_W_SEM:
                    mem.cfg["w_sem"] = w
                    r = run(mem)
                    results.append((norm, w, r))
                    print(f"{norm:<10} {w:>5.1f}  {r['overall']:>6.3f} "
                          f"{r['paraphrase']:>6.3f} {r['zero_overlap']:>6.3f} "
                          f"{r['keyword']:>6.3f} {r['p50_ms']:>7.2f}")
        finally:
            if has_norm_switch:
                mneme_module._SEM_NORM = saved
        eligible = [(norm, w, r) for norm, w, r in results
                    if r["keyword"] >= BASELINE_P3_KEYWORD - 1e-9]
        if not eligible:
            print("GRID FAIL: no cell satisfies the keyword non-regression "
                  "constraint — nothing to fit")
            return 1
        norm, w, r = sorted(
            eligible,
            key=lambda t: (-t[2]["paraphrase"], t[1],
                           0 if t[0] == "clip_abs" else 1))[0]
        print(f"FITTED (pre-registered rule: max paraphrase s.t. keyword >= "
              f"{BASELINE_P3_KEYWORD}; ties -> smaller w_sem, then clip_abs):")
        print(f"  w_sem={w:.1f} norm={norm} -> {_fmt(r)}")
        return 0
    finally:
        mem.close()
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--wsem-grid", action="store_true",
                    help="gate-3 fit: sweep w_sem (x sem-norm where the engine "
                         "still carries the switch) on one 1000-row store and "
                         "apply the pre-registered selection rule")
    args = ap.parse_args()
    sys.exit(wsem_grid()) if args.wsem_grid else main()
