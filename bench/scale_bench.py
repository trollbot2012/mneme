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

# --- target facts with paraphrased queries (expected in top-3) ----------------
# ADR-0004 gate-1 fixture. Structure is load-bearing:
#   TARGETS = _PARAPHRASE_TARGETS + ZERO_OVERLAP_TARGETS  (zero-overlap LAST, so
#   mechanism_bakeoff's rid=idx+1 seeding rule survives), >=100 total.
#   _PARAPHRASE_TARGETS: synonym-shifted queries, partial token overlap allowed.
#   ZERO_OVERLAP_TARGETS: the query shares ZERO porter stems with title+body
#   (validated mechanically in test_adr0004.py against the production
#   fts5 'porter unicode61' tokenizer) — lexical retrieval cannot reach these.
# The first 25 entries are the original fixture, kept verbatim.
_PARAPHRASE_TARGETS = [
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
    # --- widened paraphrase set (ADR-0004, appended after the original 25) ----
    ("Grafana dashboards provision from git on merge",
     "Monitoring dashboards deploy from the grafana/ folder when a PR merges.",
     "how do the metric boards get provisioned"),
    ("The staging database resets each morning",
     "Staging DB restores from an anonymized snapshot at 06:00.",
     "when is test environment data refreshed"),
    ("CI caches node modules per branch",
     "The pipeline keys its npm cache on branch name plus lockfile hash.",
     "build dependency caching strategy"),
    ("Webhooks retry five times with backoff",
     "Failed webhook deliveries retry 5 times with exponential backoff.",
     "how are missed callback deliveries retried"),
    ("Image uploads lose EXIF metadata",
     "The resizer strips EXIF, including GPS, from uploaded photos.",
     "does picture processing remove location tags"),
    ("Sentry errors route to the platform channel",
     "Sentry alerts post into the #platform-alerts channel.",
     "where do exception notifications end up"),
    ("The API gateway caps request bodies at 1MB",
     "Requests over one megabyte get a 413 at the gateway.",
     "maximum payload size for inbound calls"),
    ("Session cookies expire after twelve hours",
     "Auth sessions last 12 hours; then users must re-login.",
     "how long does a signed-in user stay authenticated"),
    ("Queue consumers ack after the side effect",
     "Workers acknowledge messages only after the database commit.",
     "when are broker messages acknowledged"),
    ("Blue-green cutover shifts DNS weights",
     "Deploy cutover moves DNS weight from 10/90 to 90/10 over five minutes.",
     "how does traffic shift to a new release"),
    ("The billing ledger is append-only",
     "Ledger rows are never updated; corrections append reversal entries.",
     "can invoice records be edited after posting"),
    ("Feature branches deploy to ephemeral previews",
     "Every feature branch gets a preview environment torn down on merge.",
     "temporary deploy environments for pull requests"),
    ("The Android build signs with the upload key",
     "Play Store builds are signed with the upload key from the vault.",
     "mobile release signing arrangement"),
    ("Data warehouse loads run incrementally",
     "The warehouse ingests deltas hourly; full loads only on schema change.",
     "how often does the analytics store ingest"),
    ("The rate limiter is per-token not per-IP",
     "Throttling counts requests per API token, not per client address.",
     "is request limiting keyed to user or machine"),
    ("Standup notes archive to the wiki weekly",
     "Monday standup notes are archived under wiki/standups.",
     "where do daily meeting summaries get stored"),
    ("The CDN caches assets for one year",
     "Static assets carry immutable cache headers of 365 days.",
     "static file cache duration on the edge"),
    ("Passwords hash with argon2id",
     "User passwords use argon2id with 64MB memory cost.",
     "credential hashing algorithm in use"),
    ("The search service tolerates two node failures",
     "Search cluster quorum survives losing 2 of 5 nodes.",
     "how resilient is the query cluster"),
    ("Batch jobs checkpoint every ten thousand rows",
     "Long batch jobs checkpoint each 10k rows for resumability.",
     "can long-running imports resume midway"),
    ("The style guide mandates conventional commits",
     "Commit messages follow Conventional Commits, enforced by CI.",
     "required format for change messages"),
    ("Secrets live in Vault not env files",
     "Runtime secrets come from Vault; .env files are forbidden in prod.",
     "where are production credentials stored"),
    ("The scheduler skips public holidays",
     "Cron-driven business jobs pause on the holiday calendar.",
     "do automated jobs run on holidays"),
    ("Object storage lifecycle deletes after 90 days",
     "Temp bucket objects expire via lifecycle rule at 90 days.",
     "how long do uploaded files persist in the bucket"),
    ("The GraphQL API depth limit is eight",
     "Queries deeper than 8 levels are rejected.",
     "nesting limit for graph queries"),
    ("Canary releases get five percent of traffic",
     "Canaries receive 5% of requests for 30 minutes before promotion.",
     "what share of users hit the trial build"),
    ("The mail digest sends Monday mornings",
     "Weekly digest emails go out Mondays at 08:00 local.",
     "when does the newsletter get sent"),
    ("Thumbnails generate on first request",
     "Image thumbnails render lazily and cache after the first hit.",
     "are preview images made eagerly or on demand"),
    ("The audit log is immutable for seven years",
     "Audit entries are WORM-stored for a 7-year retention.",
     "how long are compliance records kept"),
    ("Feature flags refresh every thirty seconds",
     "Clients poll flags.json every 30s; no push channel.",
     "how quickly do toggles propagate"),
    ("The recommendation cache warms at deploy",
     "Post-deploy hook pre-warms the recommendation cache.",
     "is the suggestion store cold after a release"),
    ("Database migrations run in CI before deploy",
     "Migrations apply in the pipeline, never at app boot.",
     "when do schema changes get applied"),
    ("The SSO provider is Okta with SCIM",
     "Okta handles SSO; SCIM provisions and deprovisions accounts.",
     "which identity vendor manages logins"),
    ("Timeseries data downsamples after a week",
     "Metrics older than 7 days downsample to 5-minute resolution.",
     "granularity of old monitoring data"),
    ("The PDF renderer runs in a sandboxed pod",
     "PDF generation is isolated in a gVisor-sandboxed pod.",
     "is document rendering isolated from the main app"),
    ("Websocket connections cap at ten thousand",
     "Each gateway node holds at most 10k concurrent sockets.",
     "concurrent realtime connection ceiling"),
    ("The fraud model scores async post-checkout",
     "Fraud scoring runs asynchronously after checkout completes.",
     "does risk evaluation block the purchase flow"),
    ("Error budgets reset monthly per service",
     "SLO error budgets reset on the first of each month.",
     "when does reliability allowance replenish"),
    ("The iOS app requires TLS 1.3 minimum",
     "The Apple client refuses TLS below 1.3.",
     "minimum encryption protocol for the apple client"),
    ("Kafka topics retain three days of events",
     "Event topics keep 72 hours; compacted topics keep forever.",
     "message retention window for the event bus"),
    ("The dev proxy rewrites cookies to localhost",
     "Local proxy rewrites Secure cookies so localhost works.",
     "why do sessions work on the development machine"),
    ("Payment webhooks verify HMAC signatures",
     "Stripe callbacks are HMAC-verified before processing.",
     "are incoming billing callbacks authenticated"),
    ("The backup restore drill runs quarterly",
     "A restore-from-backup drill happens every quarter.",
     "how often is disaster recovery rehearsed"),
    ("Container images build distroless",
     "Production images are distroless; no shell inside.",
     "can you exec a shell in the prod containers"),
    ("The A/B framework buckets by account id",
     "Experiment bucketing hashes the account id, not the device.",
     "how are experiment groups assigned"),
    ("Slow queries log over two hundred ms",
     "Statements slower than 200ms land in the slow query log.",
     "threshold for logging sluggish database statements"),
    ("The translations sync from Lokalise nightly",
     "Locale strings pull from Lokalise every night at 01:00.",
     "when do updated translations reach the app"),
    ("Static analysis blocks merges on high severity",
     "The linter gate fails PRs with high-severity findings.",
     "can code with critical findings be merged"),
    ("The video CDN prefers AV1 when supported",
     "Players negotiate AV1 first, falling back to VP9.",
     "which codec does playback try first"),
    ("On-call handoff happens Fridays at noon",
     "Pager rotation flips every Friday 12:00.",
     "when does the pager switch owners"),
    ("The email queue drains at two hundred per minute",
     "Outbound mail sends at 200/min to protect sender reputation.",
     "outgoing message throughput cap"),
    ("Report exports stream as CSV chunks",
     "Large report exports stream chunked CSV, never buffered.",
     "do big downloads load fully into memory"),
    ("The geo service snaps to H3 resolution seven",
     "Locations index on H3 cells at resolution 7.",
     "spatial indexing scheme for coordinates"),
    ("Browser support starts at last two versions",
     "The frontend supports the last 2 versions of evergreen browsers.",
     "oldest browsers the web app supports"),
    ("The license scanner runs on dependency bumps",
     "Renovate PRs trigger the license compliance scan.",
     "when is open source compliance checked"),
]

# Queries with ZERO porter-stem overlap against their own title+body: only a
# semantic mechanism (or the cooc rescue, which dies once distractors match)
# can surface these. Kept LAST in TARGETS so index-split per-class scoring and
# mechanism_bakeoff's rid=idx+1 rule both hold. test_adr0004.py enforces the
# zero-overlap property per entry with the production tokenizer.
ZERO_OVERLAP_TARGETS = [
    ("Payroll disbursement lands Friday",
     "Wages reach employee accounts each Friday evening.",
     "when do staff get paid"),
    ("Credential rotation happens quarterly",
     "Secrets regenerate on a three-month cadence, enforced automatically.",
     "how often must api keys change"),
    ("Architecture writeups live under the wiki",
     "Design documents sit at wiki slash architecture.",
     "where are system diagrams kept"),
    ("Ledger reconciliation fires nightly at two",
     "Books balance against bank statements at 02:00.",
     "accounting sync schedule"),
    ("Fallback SMTP relay sits in Frankfurt",
     "Mail failover routes through a relay hosted near Frankfurt.",
     "backup email server location"),
    ("Deployment freezes cover December weekends",
     "No releases ship on Saturdays or Sundays during December.",
     "holiday change embargo window"),
    ("Meeting rooms unlock via badge reader",
     "Conference doors open when an employee badge taps the reader.",
     "office access control method"),
    ("Customer refunds settle within five days",
     "Reimbursements post back to cards inside five business days.",
     "how long until buyers see money returned"),
    ("Print service spools through a Windows host",
     "Office printing spools on the WIN-PRINT box.",
     "where do paper jobs queue"),
    ("Analytics events sample at ten percent",
     "Only one event in ten reaches the warehouse stream.",
     "telemetry collection rate"),
    ("Feature requests triage on Tuesdays",
     "Product reviews the suggestion backlog every Tuesday.",
     "when are user ideas evaluated"),
    ("Dark mode ships behind a beta toggle",
     "Night theme stays gated until the beta cohort validates it.",
     "is low-light appearance available yet"),
    ("Invoice dunning escalates after three attempts",
     "Unpaid bills trigger a final notice once three emails go unanswered.",
     "overdue payment reminder policy"),
    ("Warehouse robots recharge overnight",
     "Autonomous pickers dock at charging stations after the evening shift.",
     "when do fulfillment machines top up batteries"),
    ("Legal holds suspend document purges",
     "Litigation flags stop scheduled deletion for affected records.",
     "can files be removed during a lawsuit"),
    ("Conference budget approves up to two grand",
     "Each engineer may spend $2000 yearly on events.",
     "training trip expense ceiling"),
    ("Fleet laptops image with encrypted disks",
     "Company notebooks arrive with full-disk encryption preinstalled.",
     "are work computers protected against theft"),
    ("Primary outage promotes a standby automatically",
     "Failover elects a warm replica without operator action.",
     "what happens when the main database dies"),
    ("Office plants get watered by a service",
     "A vendor tends greenery every second week.",
     "who looks after indoor foliage"),
    ("Parking spots allocate by seniority",
     "Garage bays assign according to tenure at hiring date.",
     "who gets a reserved car space"),
    ("Interview loops cap at four sessions",
     "Candidates meet at most four panels before a decision.",
     "hiring process length"),
    ("Data retention purges inactive accounts yearly",
     "Profiles idle past twelve months delete automatically.",
     "how long before dormant users are erased"),
    ("The status page hosts on external infra",
     "status.example.com serves from a third-party platform, isolated from prod.",
     "where does uptime reporting run"),
    ("Expense reports need receipts over twenty five",
     "Claims above $25 must attach an itemized receipt.",
     "purchase reimbursement paperwork threshold"),
    ("The beta program invites power users first",
     "Early access rolls out to the most active accounts.",
     "who tries new functionality before launch"),
    ("Support tickets auto-close after a week idle",
     "Stale cases resolve automatically once seven days pass silently.",
     "when do help requests expire"),
    ("The demo tenant reseeds hourly",
     "Sandbox showcase data rebuilds every sixty minutes.",
     "how fresh is trial environment content"),
    ("Certificate renewal automates via ACME",
     "TLS certs renew through Let's Encrypt without manual steps.",
     "does https expiry need human action"),
    ("Growth experiments conclude at significance",
     "A/B tests stop once results reach statistical confidence.",
     "when do marketing trials wrap up"),
    ("The intranet search excludes draft pages",
     "Internal lookup skips unpublished wiki entries.",
     "why can I not find my unfinished doc"),
]

TARGETS = _PARAPHRASE_TARGETS + ZERO_OVERLAP_TARGETS

# Keyword-identical class (gate 3 non-regression): query == title, DERIVED so
# it can never drift from the fixture. Same 3-tuple shape as TARGETS.
KEYWORD_QUERIES = [(t, b, t) for t, b, _ in TARGETS]


# --- ADR-0004 gate-3 baseline (pre-implementation, DO NOT re-derive) ----------
# Measured on the PRE-ADR engine — mneme.py sha256
# 98df68f5599bed646984571463a786892f57a568514c4292b3842cea08103d40 (engine head
# 2c3e1df, the ADR-doc-only commit) — with C:\Python314\python.exe at scale 1000
# on THIS widened 110-query fixture under the unified exact-title is_hit rule.
# Identical p@3 values reproduced at scale 2000 and across repeat runs.
# Gate 3's +0.10 uplift and the keyword non-regression are defined against
# THESE constants, never against re-runs; the historical 0.56-0.60 figures
# (25-query fixture, 40-char-substring rule) are explicitly NOT the baseline.
BASELINE_P3_OVERALL = 0.400        # 44/110
BASELINE_P3_PARAPHRASE = 0.550     # 44/80
BASELINE_P3_ZERO_OVERLAP = 0.000   # 0/30 — lexical retrieval cannot reach these
BASELINE_P3_KEYWORD = 1.000        # 110/110 (query == title)
BASELINE_P50_MS = 0.51             # recall p50 at scale 1000 (0.46-0.61 across runs)


def is_hit(target_title: str, hit_titles) -> bool:
    """THE match rule for every p@3 number (scale_bench, engine_gate, tests).

    Exact title equality — one ruler, unified BEFORE any ADR-0004 uplift or
    baseline number exists. The old 40-char-substring rule could credit a hit
    whose title merely contained the target prefix, while engine_gate already
    used exact equality: two benches disagreed about what a 'hit' was.
    """
    return any(t == target_title for t in hit_titles)

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

    def __init__(self, cfg: dict | None = None):
        from mneme import Mneme
        self.dir = Path(tempfile.mkdtemp(prefix="scale-mneme-"))
        self.mem = Mneme(self.dir / "m.db", self.dir / "notes", config=cfg or None)

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
        n_para = len(_PARAPHRASE_TARGETS)
        hits, lats = 0, []
        class_hits = [0, 0]  # [paraphrase, zero-overlap] by fixture index split
        for idx, (title, _body, query) in enumerate(TARGETS):
            t0 = time.perf_counter()
            top3 = p.query(query)
            lats.append((time.perf_counter() - t0) * 1000)
            if is_hit(title, top3):
                hits += 1
                class_hits[0 if idx < n_para else 1] += 1
        # keyword-identical class (gate-3 non-regression probe): hit rate only,
        # its latencies stay OUT of p50 so p50 keeps meaning "fixture queries"
        kw_hits = sum(1 for title, _body, query in KEYWORD_QUERIES
                      if is_hit(title, p.query(query)))
        return {"scale": scale, "seed_ms": round(seed_ms, 1),
                "p_at_3": round(hits / len(TARGETS), 3),
                "p3_paraphrase": round(class_hits[0] / n_para, 3),
                "p3_zero_overlap": round(class_hits[1] / len(ZERO_OVERLAP_TARGETS), 3),
                "p3_keyword": round(kw_hits / len(KEYWORD_QUERIES), 3),
                "lat_p50_ms": round(statistics.median(lats), 2),
                "lat_max_ms": round(max(lats), 2)}
    finally:
        p.close()


TARGETS_TITLES_BODIES = [(t, b) for t, b, _q in TARGETS]


# --- ADR-0004 gate-4 assert mode ----------------------------------------------
ASSERT_SCALE = 1000
ASSERT_P50_OFF_MS = 2.0   # gate 4 budget, embeddings off (baseline p50 well under)
ASSERT_P50_ON_MS = 20.0   # gate 4 budget, embeddings on (cold load excluded)
MODEL_FILES = ("model_quint8_avx2.onnx", "vocab.txt")
DEFAULT_MODEL_DIR = Path(__file__).resolve().parent.parent / "models"


def _encode_calls(mem) -> int:
    """Read the PER-TEXT encode counter.

    WP3 contract: the lazily-built encoder lives at Mneme._encoder and its
    encode() does `self.calls += len(texts)` — texts, not batches, so a
    per-candidate encode smuggled into one batch call still moves the counter.
    Returns 0 when the engine predates ADR-0004 or the encoder is not loaded.
    """
    return int(getattr(getattr(mem, "_encoder", None), "calls", 0) or 0)


def _assert_arm(label, cfg, p50_budget_ms, encode_guard):
    """One --assert arm at ASSERT_SCALE. Returns a list of failure strings.

    All embeddings hooks are hasattr/getattr-guarded so the off-arm (and the
    whole command) runs against the pre-ADR engine today; once the engine
    grows `embeddings_available`, a False value with model files present is a
    LOUD failure, never a silent lexical-only pass.
    """
    failures = []
    p = MnemeProvider(cfg=cfg)
    try:
        corpus = list(TARGETS_TITLES_BODIES) + distractors(
            max(0, ASSERT_SCALE - len(TARGETS)))
        p.seed(corpus)
        mem = p.mem
        pre_adr = not hasattr(mem, "embeddings_available")
        if encode_guard:
            if hasattr(mem, "_sync_vectors"):
                while mem._sync_vectors():
                    pass
            if pre_adr:
                print(f"  [{label}] NOTE: engine predates ADR-0004 "
                      f"(no embeddings hooks); arm ran lexical-only")
            elif not mem.embeddings_available:
                failures.append(
                    f"[{label}] embeddings_available is False with model files "
                    f"present — the on-arm silently degraded to lexical-only")
            # one warmup recall: cold encoder load is paid (and printed) here,
            # never inside the p50 window
            t0 = time.perf_counter()
            p.query(TARGETS[0][2])
            print(f"  [{label}] warmup recall (cold load, excluded from p50): "
                  f"{(time.perf_counter() - t0) * 1000:.1f}ms")
        hits, lats = 0, []
        for title, _body, query in TARGETS:
            c0 = _encode_calls(mem)
            t0 = time.perf_counter()
            top3 = p.query(query)
            lats.append((time.perf_counter() - t0) * 1000)
            c1 = _encode_calls(mem)
            if is_hit(title, top3):
                hits += 1
            if encode_guard and not pre_adr:
                if c1 - c0 > 1:
                    failures.append(
                        f"[{label}] encode guard: {c1 - c0} texts encoded in one "
                        f"recall ({query!r}) — per-candidate encoding")
                p.query(query)  # immediately repeated: must hit the query-vec LRU
                c2 = _encode_calls(mem)
                if c2 != c1:
                    failures.append(
                        f"[{label}] LRU guard: repeated query {query!r} "
                        f"re-encoded ({c2 - c1} texts)")
        if encode_guard and not pre_adr and _encode_calls(mem) == 0:
            failures.append(
                f"[{label}] per-text counter never moved across "
                f"{len(TARGETS)} recalls — WP3 counter contract "
                f"(Mneme._encoder.calls) is broken, guards were vacuous")
        p50 = statistics.median(lats)
        print(f"  [{label}] scale={ASSERT_SCALE} p@3={hits / len(TARGETS):.3f} "
              f"p50={p50:.2f}ms max={max(lats):.2f}ms (budget {p50_budget_ms}ms)")
        if p50 > p50_budget_ms:
            failures.append(
                f"[{label}] lat_p50_ms {p50:.2f} > budget {p50_budget_ms}")
        return failures
    finally:
        p.close()


def run_assert(model_dir: Path) -> int:
    """Gate-4 battery (ADR-0004): returns nonzero on ANY breach.

    Two arms at scale 1000: embeddings-off (p50 <= 2ms), then embeddings-on
    (p50 <= 20ms after a full `_sync_vectors` fill and one warmup recall)
    with the per-TEXT encode guard: <=1 encode per recall, exactly 0 on an
    immediately repeated query (LRU proof). Missing model files fail LOUD —
    a gate runner must never silently skip the on-arm.
    """
    missing = [str(model_dir / f) for f in MODEL_FILES
               if not (model_dir / f).is_file()]
    if missing:
        print("ASSERT FAIL: embedding model files absent — the embeddings-on "
              "arm cannot run, and skipping it silently would void gate 4:")
        for m in missing:
            print(f"  missing: {m}")
        print("Provision via scripts/provision_embeddings.ps1 (or .sh), "
              "or pass --model-dir.")
        return 1
    print(f"--assert: gate-4 latency + encode-guard battery at scale {ASSERT_SCALE}")
    failures = _assert_arm("embeddings-off", {"embeddings": "off"},
                           ASSERT_P50_OFF_MS, encode_guard=False)
    failures += _assert_arm(
        "embeddings-on",
        {"embeddings": "auto", "embed_model_dir": str(model_dir)},
        ASSERT_P50_ON_MS, encode_guard=True)
    if failures:
        print("ASSERT FAIL:")
        for f in failures:
            print(f"  {f}")
        return 1
    print("ASSERT OK: all gate-4 budgets and encode guards held")
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--provider", default="mneme", choices=sorted(PROVIDERS))
    ap.add_argument("--scales", nargs="*", type=int, default=[20, 100, 500, 1000, 2000])
    ap.add_argument("--assert", dest="assert_mode", action="store_true",
                    help="gate-4 mode: run the two-arm latency + encode-guard "
                         "battery at scale 1000 and exit 1 on any breach "
                         "(plain runs stay print-only)")
    ap.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR,
                    help=f"embedding model directory (default {DEFAULT_MODEL_DIR})")
    args = ap.parse_args()
    if args.assert_mode:
        sys.exit(run_assert(args.model_dir))
    print(f"provider={args.provider} | {len(TARGETS)} target queries "
          f"({len(_PARAPHRASE_TARGETS)} paraphrase + {len(ZERO_OVERLAP_TARGETS)} "
          f"zero-overlap), precision@3 under distractor noise")
    print(f"{'scale':>6} {'seed_ms':>9} {'p@3':>6} {'para':>6} {'zero':>6} "
          f"{'kw':>6} {'p50_ms':>8} {'max_ms':>8}")
    for s in args.scales:
        r = run(PROVIDERS[args.provider], s)
        print(f"{r['scale']:>6} {r['seed_ms']:>9} {r['p_at_3']:>6} "
              f"{r['p3_paraphrase']:>6} {r['p3_zero_overlap']:>6} "
              f"{r['p3_keyword']:>6} {r['lat_p50_ms']:>8} {r['lat_max_ms']:>8}")
