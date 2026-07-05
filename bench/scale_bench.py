"""Provider-agnostic relevance-at-scale benchmark: precision@3 under growing
distractor noise (the honest ranking test a 20-entry corpus can't give).

Usage:
    python bench/scale_bench.py                 # Mneme, scales 20..2000
    python bench/scale_bench.py --scales 100 1000

Plug another provider: subclass Provider, implement seed()/query(), add to
PROVIDERS. Queries are PARAPHRASED (synonym-shifted), not keyword-identical —
a provider that only echoes exact tokens will show it here.
"""
import argparse
import random
import shutil
import statistics
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# --- 25 target facts with paraphrased queries (expected in top-3) -------------
TARGETS = [
    ("Deploy script needs the staging bucket name",
     "The deploy script fails without STAGING_BUCKET set in the environment.",
     "what does the deployment tooling require to run"),
    ("Nightly backup runs at 03:15 local",
     "The cron backup fires at 03:15 and writes to the NAS share.",
     "when does the overnight archive job execute"),
    ("Postgres pool exhausts under load tests",
     "Load testing exhausts the connection pool at 100 concurrent workers.",
     "database connections run out during stress runs"),
    ("Operator prefers tabs over spaces in Go files",
     "Go sources use tabs; the formatter enforces it.",
     "indentation style preference for golang code"),
    ("The auth service rejects clock-skewed tokens",
     "JWTs older than 30s of skew are rejected by authsvc.",
     "why do login tokens fail when system time drifts"),
    ("Redis cache flushes on config reload",
     "Reloading config.yaml empties the redis cache unexpectedly.",
     "cache gets cleared after configuration changes"),
    ("Build machine GPU is an RTX card with 24GB",
     "The build host has a 24GB RTX GPU for model inference.",
     "how much video memory does the ci server have"),
    ("Vendor API rate limit is 50 requests per minute",
     "SupplierAPI throttles at 50 rpm; batch accordingly.",
     "third party api throttling threshold"),
    ("Log rotation keeps fourteen days of history",
     "Logs rotate daily and retain 14 days.",
     "how long are server logs kept before deletion"),
    ("The mobile app pins TLS certificates",
     "Cert pinning in the iOS/Android clients breaks on CA rotation.",
     "https certificate handling in the phone application"),
    ("Payments settle through the EU acquirer on Tuesdays",
     "Settlement batch to the EU acquirer runs weekly on Tuesday.",
     "when do card transactions get settled"),
    ("Feature flags live in flags.json not the database",
     "Feature toggles are file-based (flags.json), not DB rows.",
     "where are experiment toggles stored"),
    ("The scraper respects robots.txt with a 2s delay",
     "Crawler waits 2 seconds between fetches and honors robots.",
     "web harvesting politeness settings"),
    ("Kubernetes ingress strips the X-Request-Id header",
     "The ingress controller drops X-Request-Id; re-add via middleware.",
     "why does request tracing lose its correlation id"),
    ("Email sending is sandboxed in staging",
     "Staging routes all outbound mail to a sink mailbox.",
     "do test environments send real messages to users"),
    ("The ML model retrains every Sunday night",
     "Weekly retraining job runs Sunday 23:00 on fresh labels.",
     "recommendation engine refresh schedule"),
    ("Invoices are numbered per fiscal year",
     "Invoice sequence resets each fiscal year, prefix FY.",
     "billing document numbering scheme"),
    ("The ops runbook lives in the wiki under /oncall",
     "On-call procedures are documented at wiki /oncall.",
     "where to find incident response instructions"),
    ("Search reindexes take four hours end to end",
     "A full search reindex is a 4-hour operation.",
     "how long does rebuilding the search index take"),
    ("Customer PII is masked in analytics exports",
     "Analytics pipelines mask personally identifiable fields.",
     "is private user data visible in reporting dumps"),
    ("The legacy FTP import still feeds inventory",
     "Inventory sync still depends on the old FTP drop.",
     "does stock data still arrive over the ancient file transfer"),
    ("Terraform state is locked via DynamoDB",
     "TF state locking uses a DynamoDB table.",
     "infrastructure as code state locking mechanism"),
    ("The support bot escalates after two failed answers",
     "Helpdesk bot hands off to a human after 2 misses.",
     "when does the assistant give up and page a person"),
    ("Video encoding uses two-pass VP9 for uploads",
     "Uploads transcode with 2-pass VP9 at target 4Mbps.",
     "clip compression pipeline codec choice"),
    ("The office VPN blocks UDP except DNS",
     "Corporate VPN permits UDP only on port 53.",
     "why do voice calls fail on the company network"),
]

_TOPICS = ["renderer", "billing", "gateway", "scheduler", "exporter", "notifier",
           "importer", "dashboard", "archiver", "resizer", "tokenizer", "profiler"]
_VERBS = ["times out", "retries thrice", "caches results", "emits metrics",
          "requires auth", "runs hourly", "skips weekends", "uses gzip",
          "writes json", "reads yaml", "needs review", "supports batching"]
_QUALS = ["on the blue cluster", "for premium tenants", "since version nine",
          "unless disabled", "in dry-run mode", "behind the proxy",
          "during migrations", "when queues back up", "for large payloads"]


def distractors(n, seed=42):
    rng = random.Random(seed)
    out = []
    for i in range(n):
        t, v, q = rng.choice(_TOPICS), rng.choice(_VERBS), rng.choice(_QUALS)
        out.append((f"The {t} {v} {q} (note {i})",
                    f"Operational note {i}: the {t} component {v} {q}."))
    return out


class Provider:
    name = "base"

    def seed(self, entries):  # [(title, body)]
        raise NotImplementedError

    def query(self, q) -> list[str]:  # -> ranked titles/texts
        raise NotImplementedError

    def close(self):
        pass


class MnemeProvider(Provider):
    name = "mneme"

    def __init__(self):
        from mneme import Mneme
        self.dir = Path(tempfile.mkdtemp(prefix="scale-mneme-"))
        self.mem = Mneme(self.dir / "m.db", self.dir / "notes")

    def seed(self, entries):
        for title, body in entries:
            self.mem.add_note("fact", title[:120], body)

    def query(self, q):
        return [h["title"] for h in self.mem.recall(q, None, top_k=3)]

    def close(self):
        self.mem.close()
        shutil.rmtree(self.dir, ignore_errors=True)


PROVIDERS = {"mneme": MnemeProvider}  # add hindsight/holographic adapters here


def run(provider_cls, scale):
    p = provider_cls()
    try:
        corpus = list(TARGETS_TITLES_BODIES) + distractors(max(0, scale - len(TARGETS)))
        t0 = time.perf_counter()
        p.seed(corpus)
        seed_ms = (time.perf_counter() - t0) * 1000
        hits, lats = 0, []
        for title, _body, query in TARGETS:
            t0 = time.perf_counter()
            top3 = p.query(query)
            lats.append((time.perf_counter() - t0) * 1000)
            if any(title.lower()[:40] in t.lower() for t in top3):
                hits += 1
        return {"scale": scale, "seed_ms": round(seed_ms, 1),
                "p_at_3": round(hits / len(TARGETS), 3),
                "lat_p50_ms": round(statistics.median(lats), 2),
                "lat_max_ms": round(max(lats), 2)}
    finally:
        p.close()


TARGETS_TITLES_BODIES = [(t, b) for t, b, _q in TARGETS]

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--provider", default="mneme", choices=sorted(PROVIDERS))
    ap.add_argument("--scales", nargs="*", type=int, default=[20, 100, 500, 1000, 2000])
    args = ap.parse_args()
    print(f"provider={args.provider} | {len(TARGETS)} paraphrased target queries, "
          f"precision@3 under distractor noise")
    print(f"{'scale':>6} {'seed_ms':>9} {'p@3':>6} {'p50_ms':>8} {'max_ms':>8}")
    for s in args.scales:
        r = run(PROVIDERS[args.provider], s)
        print(f"{r['scale']:>6} {r['seed_ms']:>9} {r['p_at_3']:>6} "
              f"{r['lat_p50_ms']:>8} {r['lat_max_ms']:>8}")
