"""Golden off-path pin for ADR-0004 gate 6 (embeddings-off byte-identity).

Captures the CURRENT engine's lexical read path over a fully deterministic
corpus (TARGETS + distractors(seed=42) at scale 1000, fixed timestamps,
synthetic dedupe keys) and pins it to bench/golden_off_path.json:

  - recall(top_k=8) for EVERY fixture query (paraphrase, zero-overlap, and the
    derived keyword-identical class): (key, title, round(score, 9)) per hit;
  - one full index_block text + its keys.

This file was generated ONCE at the pre-ADR-0004 engine head and committed
BEFORE any engine change — the ordering is load-bearing. After the ADR lands,
the embeddings-OFF path must reproduce it exactly (the gate-6 parity test
replays this JSON). Re-running the script against an unchanged off path must
reproduce the JSON byte-identically:

    C:\\Python314\\python.exe bench/capture_golden.py           # verify (or create)
    C:\\Python314\\python.exe bench/capture_golden.py --write   # overwrite (rebaseline)

Rebaselining is an ADR-level event: never --write casually.
"""
import argparse
import hashlib
import json
import shutil
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from mneme import Mneme  # noqa: E402
from scale_bench import (  # noqa: E402
    KEYWORD_QUERIES, TARGETS, ZERO_OVERLAP_TARGETS, _PARAPHRASE_TARGETS, distractors)

SCALE = 1000
TOP_K = 8
SEED_TS = 1_700_000_000.0  # fixed: facts don't decay, but determinism is free
OUT = Path(__file__).resolve().parent / "golden_off_path.json"
MNEME_PY = Path(__file__).resolve().parent.parent / "mneme.py"


def build():
    tmp = Path(tempfile.mkdtemp(prefix="mneme-golden-"))
    mem = Mneme(tmp / "m.db", tmp / "notes")
    if not mem.fts_available:
        raise SystemExit("FTS5 unavailable: a LIKE-fallback capture would pin the wrong path")
    corpus = [(t, b) for t, b, _ in TARGETS] + distractors(max(0, SCALE - len(TARGETS)))
    # Deterministic direct seed (engine_gate.build precedent): synthetic keys +
    # fixed timestamps, so keys/scores never depend on temp dirs or wall clock.
    with mem._lock:
        for i, (title, body) in enumerate(corpus):
            key = f"file:/bench/note_{i}.md"
            mem._upsert_locked(key, "fact", "global", title, body, "", "",
                               key, False, SEED_TS, SEED_TS)
        mem._conn.commit()
    return mem, tmp


def capture() -> str:
    mem, tmp = build()
    try:
        n_para = len(_PARAPHRASE_TARGETS)
        recalls = []
        classed = (
            [("paraphrase", TARGETS[i][2]) for i in range(n_para)]
            + [("zero_overlap", t[2]) for t in ZERO_OVERLAP_TARGETS]
            + [("keyword", q) for _, _, q in KEYWORD_QUERIES]
        )
        for cls, query in classed:
            hits = mem.recall(query, None, top_k=TOP_K)
            recalls.append({
                "class": cls,
                "query": query,
                "hits": [[h["key"], h["title"], round(h["score"], 9)] for h in hits],
            })
        ib_query = TARGETS[0][2]
        ib = mem.index_block(None, ib_query)
        doc = {
            "captured_from": "pre-ADR-0004 engine (embeddings-off path pin)",
            "engine_mneme_sha256": hashlib.sha256(MNEME_PY.read_bytes()).hexdigest(),
            "scale": SCALE,
            "top_k": TOP_K,
            "seed_ts": SEED_TS,
            "fixture": {
                "targets": len(TARGETS),
                "paraphrase": n_para,
                "zero_overlap": len(ZERO_OVERLAP_TARGETS),
                "keyword": len(KEYWORD_QUERIES),
            },
            "recalls": recalls,
            "index_block": {"query": ib_query, "text": ib.text, "keys": ib.keys},
        }
        return json.dumps(doc, indent=1, sort_keys=True, ensure_ascii=True) + "\n"
    finally:
        mem.close()
        shutil.rmtree(tmp, ignore_errors=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--write", action="store_true",
                    help="overwrite golden_off_path.json (rebaseline; ADR-level event)")
    args = ap.parse_args()
    text = capture()
    if OUT.exists() and not args.write:
        old = OUT.read_bytes()
        if old == text.encode("utf-8"):
            print(f"OK: byte-identical reproduction of {OUT.name} "
                  f"({len(old)} bytes, {sum(len(r['hits']) for r in json.loads(old)['recalls'])} pinned hits)")
            return 0
        print(f"MISMATCH: engine output diverges from {OUT.name}; "
              f"off-path behavior changed (gate 6). Use --write ONLY to rebaseline.")
        return 1
    OUT.write_bytes(text.encode("utf-8"))
    print(f"wrote {OUT} ({len(text)} bytes)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
