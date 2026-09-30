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
---

## ADR-004 -- Bi-encoder training setup (run v2)

**Date:** 2026-09-30
**Status:** accepted for the first run; each untested choice below is an
ablation, run one at a time

### Context

First fine-tuning run of `BAAI/bge-small-en-v1.5` on the train fold
(67,494 queries), on a 6 GB RTX 4050 Laptop GPU. Some settings were
measured, some were forced by memory, and some are unexamined defaults.
This record says which is which.

### Decision

| setting | value | basis |
|---|---|---|
| training unit | one example per query | **measured**: 94.7% of queries have several Exact products; one row per (query, Exact) pair put a query in a batch several times, training its own correct answers as negatives |
| document length | 256 tokens | **measured** (below) |
| batch | 16 queries, 96 sequences | **memory-bound** at 256 tokens without checkpointing |
| negatives | up to 4 S/C/I per query, never repeated | **measured**: median query has 3; 28.4% have none |
| learning rate | 2e-5, AdamW, weight decay 0.01 | default for BERT-size fine-tuning; not tuned |
| warmup | 10% linear, then linear decay to 0 | convention; not tuned |
| gradient accumulation | 4 (64 queries per optimiser step) | default; not tuned, and doubtful (below) |
| temperature | 0.05 | default; not tuned |
| epochs | 2 | default; not tuned |

Peak memory and speed for one training step on real batches, batch 16:

| doc length | gradient checkpointing | peak VRAM | s/batch |
|---|---|---|---|
| 128 | off | 2.64 GB | 0.18 |
| 192 | off | 3.71 GB | 0.26 |
| 256 | off | 4.81 GB | 0.37 |
| 256 | on | 1.22 GB | 0.49 |
| 384 | on | 1.62 GB | 0.73 |
| 512 | on | 1.98 GB | 0.94 |

Document tokens: p50 131, p90 330, p99 500. A 256 cap truncates 21.5% of
products; 128 would truncate 50.6%.

### Alternatives rejected

- **512 tokens without checkpointing.** Peaks at 9.4 GB. The Windows driver
  spills past VRAM into system RAM instead of raising OOM, and the overfit
  run crawled at 11.7 s/batch. The loop now raises when peak memory passes
  97% of VRAM.
- **512 tokens with checkpointing.** Fits (1.98 GB) but costs 2.5x the time
  per batch. Kept as an ablation, not the starting point.
- **Filling negative slots by repetition.** A repeated negative adds the same
  term to the softmax denominator again, weighting that product k-fold.
  Short rows are padded and masked instead.

### Known weaknesses

- **Few negatives per query.** Each query sees up to 15 in-batch negatives plus
  up to 4 of its own. bge was pretrained with batches in the thousands.
  Accumulation does not help: the similarity matrix is per micro-batch.
- **Accumulation mostly slows learning.** AdamW normalises update size, so
  accumulating 4 micro-batches gives 4x fewer steps of about the same size,
  not a cleaner step of the same count.
- **The cross-query relevance mask almost never fires at this batch size.**
  Over 3,000 random batches of 16 it set 14 of 768,000 cells (12 batches).
  16% of Exact products are Exact for more than one query, but two such
  queries rarely share a batch of 16 drawn from 67k. One-query-per-row
  batching does the real work; the mask is insurance that matters more at
  larger batches (collisions grow with batch size squared) and with mined
  negatives.

### Ablations owed, one variable each

1. Batch 48-64 with gradient checkpointing (more in-batch negatives)
2. `--accum 1`
3. 512-token documents
4. Substitutes: leave / mask (`--mask-threshold 0.1`) / weight by `1 - gain`
5. lr 1e-5, only if fine-tuned lands below zero-shot
