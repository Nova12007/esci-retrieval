# Architecture Decision Records

Each entry: the decision, the alternatives rejected, and the evidence.

---

## ADR-001 — Scope to the US locale, `large_version` subset

**Date:** 2026-09-22
**Status:** accepted

### Context

ESCI ships three locales (us / es / jp) and two task subsets. Observed
counts from `docs/data_profile.txt`:

- total examples: 2,621,288
- US + large_version: 1,818,825 (69.38% of all judgments)
- unique queries: 97,345   unique products: 1,215,851
- median judged products per query: 16.0 

### Decision

Use US locale, `large_version == 1`, with the official `split` column for
train/test. A dev set is carved from train by hashing `query_id`.

### Alternatives rejected

- **All three locales.** Multilingual retrieval is a materially different
  problem (tokenization, base model choice, per-locale evaluation). It
  roughly triples the work for no additional signal about the thing this
  project demonstrates.
- **`small_version` (Task 1) only.** Smaller and reranking-focused. The
  full corpus is the point here -- the scale is what forces the ANN
  index, the memory budget, and the latency work.

### Consequences

- Results are not comparable to multilingual ESCI leaderboard entries.
  Stated explicitly in the README.
- Corpus is large enough that the index will not fit naively in memory
  at fp32, which is the intended constraint.

---

## ADR-002 — Two evaluation modes, always reported together

**Date:** 2026-09-22
**Status:** accepted

### Context

ESCI judges a median of 16 products per query, not the full
catalogue. Retrieving top-100 from the full corpus and computing
"recall" against those judgments measures a lower bound, not true recall:
unjudged retrieved products are unknown, not wrong.

### Decision

Report two modes side by side, never blended:

- **Mode A -- reranking.** Rank only each query's judged candidate set.
  nDCG@10 with the KDD Cup gain mapping (E=1.0, S=0.1, C=0.01, I=0.0).
  Comparable to published Task 1 results.
- **Mode B -- full-corpus retrieval.** Retrieve top-k from all products.
  Recall@100 over judged-Exact items only, labelled a lower bound.

### Alternatives rejected

- **Mode B alone.** Would overstate what the recall number means.
- **Mode A alone.** Loses the scale, latency and memory story entirely,
  which is the engineering point of the project.

---

## ADR-003 -- Document text and BM25 configuration

**Date:** 2026-09-27
**Status:** accepted

### Context

BM25 is the baseline every later system is measured against, so its
configuration has to be defensible rather than tuned to flatter us.

Measured on the `test` fold, 22458 queries:

| document text          | nDCG@10 (Mode A) | Recall@100 (Mode B) | index build |
|------------------------|------------------|---------------------|-------------|
| title + brand + bullets| 0.751299693255098| 0.560099101305388   | 95.1s       |
| title only             | 0.751295405993130| 0.5513083481457026  | 22.9s       |

### Decision

Use tbb. BM25 parameters k1=1.2, b=0.75 -- standard defaults,
deliberately untuned.

### Alternatives rejected

- **Including `product_description`.** Long, frequently HTML, and
  46.9% missing. Inflates the index for unclear gain.
- **Tuning k1 and b on the test fold.** Would overstate the baseline's
  quality relative to a fair comparison, and tuning on test is tuning
  on test.

### Consequences

Every later result in this project is a delta against this row.