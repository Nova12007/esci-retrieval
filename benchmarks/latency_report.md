# Latency report

Load test: `loadtest/search.js` (k6, open loop). Arrival rate ramps 5 to 50
requests/s over 1 minute, then holds 50 requests/s for 3 minutes. Each request
is `GET /search?k=10` for one test-fold query. "Peak" is the 3-minute hold.

Open loop matters: requests arrive on schedule whether or not earlier ones
have finished, so queueing shows up in the tail instead of slowing the test.

## Local container, 12 Oct 2026

Image `deploy/space/Dockerfile`, run with `--cpus=2 --memory=4g` (a 2-vCPU
instance), serving folder for `biencoder_v3` mounted read-only. k6 ran in a
second container on the same laptop.

| run | requests | failed | p50 | p95 | p99 at peak | cache hit rate |
|---|---|---|---|---|---|---|
| 2,000 queries drawn with replacement | 10,649 | 0% | 10.0 ms | 27.0 ms | 74.7 ms | 81% |
| **22,458 unique queries, no repeats** | **10,649** | **0%** | **16.3 ms** | **41.0 ms** | **78.2 ms** | **0%** |

The second run is the one to quote: every request misses the cache, the
worst case. The first is listed because it was run first and its script
wrongly described it as cache-miss dominated.

Server memory held at 2.56 GB of 4 GB. Mean encoder batch 1.02: at 50
requests/s and ~10 ms per request, requests rarely overlap, so opportunistic
batching (ADR-016) is idle at this load and costs nothing.

**Budget (p99 < 100 ms at 50 requests/s): met with no cache help.**

### Caveats

- k6 shared the laptop's remaining cores with Docker Desktop.
- The serving folder was a Windows bind mount into a Linux VM, which slows
  file reads; the first SQLite hydrate took 16 ms against ~1 ms warm. The
  per-stage timings (encode ~8 ms, HNSW ~1.6 ms) sum well below the p99, and
  slow first reads plus host contention are the likely tail.
- The deployed Space is the reference; its run is recorded below.

## Deployed Space

Pending.
