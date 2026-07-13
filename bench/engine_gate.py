"""Engine-level p@3 gate for ranking changes (freebie round, 2026-07-05).

Runs the scale_bench paraphrase fixture through the REAL Mneme engine (not the
raw-sqlite mechanism harness) and reports precision@3 + recall latency for:
  - the shipped configuration (jaccard dropped in FTS mode),
  - df-aware pruning FORCED ACTIVE (df_prune_min_rows=0).
Gate: pruning must not change p@3; the shipped config must match the bakeoff's
bm25 row (jaccard was byte-identical, so dropping it must not move p@3).

    python bench/engine_gate.py
"""
import shutil
import statistics
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from mneme import Mneme  # noqa: E402
from scale_bench import TARGETS, distractors, is_hit  # noqa: E402


def build(scale, **cfg):
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
    return mem, tmp


def run(mem):
    hits3 = 0
    lat = []
    for title, _body, query in TARGETS:
        t0 = time.perf_counter()
        top = mem.recall(query, top_k=3)
        lat.append((time.perf_counter() - t0) * 1000)
        if is_hit(title, [h["title"] for h in top]):  # ONE match rule (scale_bench.is_hit)
            hits3 += 1
    return hits3 / len(TARGETS), statistics.median(lat)


def main():
    print(f"{len(TARGETS)} fixture queries | engine-level p@3 (real Mneme.recall)")
    print(f"{'config':<28} {'n=1000':>16} {'n=2000':>16}")
    for label, cfg in (("shipped (no jaccard)", {}),
                       ("df-pruning forced on", {"df_prune_min_rows": 0})):
        cells = []
        for scale in (1000, 2000):
            mem, tmp = build(scale, **cfg)
            p, ms = run(mem)
            mem.close()
            shutil.rmtree(tmp, ignore_errors=True)
            cells.append(f"p={p:.2f} t={ms:5.2f}ms")
        print(f"{label:<28} {cells[0]:>16} {cells[1]:>16}")


if __name__ == "__main__":
    main()
