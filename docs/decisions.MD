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
**Status:** accepted; amended 2026-10-01 (Mode A comparability, see below)

### Context

ESCI judges a median of 16 products per query, not the full
catalogue. Retrieving top-100 from the full corpus and computing
"recall" against those judgments measures a lower bound, not true recall:
unjudged retrieved products are unknown, not wrong.

### Decision

Report two modes side by side, never blended:

- **Mode A -- reranking.** Rank only each query's judged candidate set.
  nDCG@10 with the KDD Cup gain mapping (E=1.0, S=0.1, C=0.01, I=0.0).
  ~~Comparable to published Task 1 results.~~

  **Correction (2026-10-01):** not comparable. KDD Cup Task 1 is scored on
  `small_version`, a controlled candidate set; this project uses
  `large_version` (ADR-001), whose judged sets are mostly Exact. Mode A is
  also saturated here: median 13 Exact in a judged set of 16, 64% of queries
  have at least 10 Exact, and 30.6% of queries score 1.0 under BM25 already.
  Mode A is reported overall and on the unsaturated slice (ADR-005), and
  compared only within this project.
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

---

## ADR-006 -- Exact search for evaluation runs on the GPU

**Date:** 2026-09-30
**Status:** accepted

### Context

The first dense evaluations took ~30 minutes per system for Mode B. Profiled
on 2,048 test queries against 1,215,854 products (384-d, fp32):

| exact search | time | queries/s |
|---|---|---|
| FAISS `IndexFlatIP`, CPU, 20 threads | 22.8 s | ~90 |
| FAISS, CPU, 1 thread | 29.1 s | ~70 |
| torch matmul + topk, CPU | 22.9 s | ~90 |
| torch matmul + topk, RTX 4050 | 0.7 s | ~2,900 |

Thread count barely matters and torch on the CPU is no faster, so FAISS is
not misconfigured: scoring 1.2M products and selecting the top 100 per query
is simply slow on this laptop's CPU. Chunking queries 256 at a time made it
worse (5-25 s per chunk, erratic).

### Decision

`DenseRanker` takes a backend. Evaluation uses `torch` on the GPU; `faiss`
stays the default and is what serving and the week 4 ANN indexes use.

Both are exact search. Measured agreement on the 2,048 queries: 99.97% top-100
overlap, maximum score difference 1.07e-6; the disagreements are ties at the
100th place. A unit test asserts identical rankings on a synthetic corpus.

### Consequences

- A full test-fold evaluation takes seconds after encoding, which makes the
  ADR-004 ablations affordable.
- Needs 1.9 GB of VRAM for the matrix plus 1.2 GB of scores per 256-query
  chunk: cannot run while training occupies the GPU.
- Serving latency must be measured on the FAISS path, not inferred from these
  numbers.

---

## Results -- week 3 gate, 2026-09-30

Test fold, 22,458 queries. 95% percentile bootstrap over queries, 1,000
resamples (`scripts/compare.py`). Differences are paired.

| system | Mode A nDCG@10 | Mode B Recall@100 |
|---|---|---|
| `bm25_tbb` | 0.7513 [0.7481, 0.7546] | 0.5601 [0.5554, 0.5648] |
| `bge_small_zeroshot` | 0.8694 [0.8671, 0.8719] | 0.5698 [0.5655, 0.5749] |
| `biencoder_v2` (ADR-004) | 0.8928 [0.8907, 0.8951] | 0.6273 [0.6229, 0.6319] |

- v2 vs BM25, Recall@100: **+0.0672 [+0.0637, +0.0704]**. Gate passed.
- v2 vs zero-shot, Recall@100: +0.0575 [+0.0553, +0.0598].
- Zero-shot vs BM25, Recall@100: +0.0097 [+0.0061, +0.0128]. Off-the-shelf bge
  reranks a judged set far better than BM25 (Mode A +0.118) but barely beats it
  at finding products in the full catalogue; fine-tuning is where Mode B moves.
- Fine-tuning's +5.8 points is below the +8 to +20 usually seen; ADR-004 lists
  the likely causes (16-query batches, accumulation cutting update count).

---

## ADR-007 -- Batch 64 with gradient checkpointing becomes the training default

**Date:** 2026-09-30
**Status:** accepted

### Context

ADR-004 ablation 1: does the number of in-batch negatives matter? v2 used 16
queries per micro-batch with 4-step accumulation. v3 uses 64 queries with no
accumulation. Both average gradients over 64 queries per optimiser step and
take 1,054 steps per epoch with the same schedule, so the only change is
negatives per query: up to 63 in-batch instead of 15. Gradient checkpointing
changes memory and speed, not the maths.

Measured at 256-token documents, checkpointing on:

| batch | peak VRAM | s/batch | min/epoch |
|---|---|---|---|
| 32 | 2.03 GB | 1.34 | ~47 |
| 48 | 2.82 GB | 1.71 | ~40 |
| 64 | 3.64 GB | 2.19 | ~38 |

### Results

Test fold, paired bootstrap, v3 minus v2:

| metric | v2 | v3 | difference |
|---|---|---|---|
| Mode B Recall@100 | 0.6273 | 0.6343 | +0.0070 [+0.0059, +0.0080] |
| Mode A nDCG@10 | 0.8928 | 0.8919 | -0.0009 [-0.0013, -0.0005] |

Dev zero-shot (epoch -1) reproduced exactly at 0.8850. The relevance mask set
99 cells over the run, where at batch 16 it almost never fired.

### Decision

Batch 64, `--accum 1`, `--grad-checkpoint` is the new default configuration.

### Consequences

- More negatives help full-corpus retrieval, but only by 0.7 points. The gap
  between our +5.8 fine-tuning gain and the usual +8 to +20 is mostly not batch
  size.
- The Mode A drop clears significance with 22k queries but is a thousandth of
  the scale on a saturated metric. Not a reason to prefer v2.
- ADR-004's accumulation ablation is superseded: at batch 64 there is nothing
  left to accumulate.

---

## ADR-005 -- Report Mode A on the unsaturated slice as well as overall

**Date:** 2026-10-01
**Status:** accepted

### Context

Mode A saturates on `large_version`. Test fold, 22,458 queries: median 16
judged products of which 13 are Exact; 64.19% of queries have at least 10
Exact; BM25 already scores exactly 1.0 on 30.63% of queries. When ten or more
Exact products exist, any ranking that fills the top ten with them scores
about 1.0, so those queries cannot separate a good ranker from a great one.
`bm25_t` and `bm25_tbb` agree to the sixth decimal for this reason.

### Decision

Every Mode A result is reported twice: over all queries, and over the
**unsaturated slice**, queries with fewer than 10 Exact judgements (8,043
queries, 35.8%). Defined in `eval/slices.py`; `scripts/compare.py` prints both.

### Evidence

nDCG@10 with 95% bootstrap CIs; differences paired:

| system | all queries | unsaturated slice |
|---|---|---|
| `bm25_tbb` | 0.7513 | 0.5384 [0.5338, 0.5432] |
| `bge_small_zeroshot` | 0.8694 | 0.7637 [0.7587, 0.7686] |
| `biencoder_v2` | 0.8928 | 0.8033 [0.7987, 0.8079] |
| `biencoder_v3` | 0.8919 | 0.8014 [0.7968, 0.8062] |

| comparison | all queries | unsaturated slice |
|---|---|---|
| v2 vs BM25 | +0.1416 | +0.2649 [+0.2589, +0.2707] |
| v2 vs zero-shot | +0.0234 | +0.0396 |
| v3 vs v2 | -0.0009 | -0.0019 [-0.0030, -0.0008] |

Every gap roughly doubles on the slice, which is the point.

### Alternatives rejected

- **Slice only.** Hides that most queries are easy, which a reader needs to know.
- **Switching Mode A to `small_version`.** A different dataset from Mode B
  (ADR-001) would break the pairing between the two modes.
- **A threshold other than 10.** Ten is where nDCG@10 can be maxed by Exacts
  alone; any other cut would be tuned to the results.

### Consequences

- Mode B Recall@100 stays the headline metric; the slice makes Mode A useful
  for reranking work in week 5, not a replacement.
- v3's small Mode A drop is consistent across both views: more in-batch
  negatives help retrieval and slightly hurt fine ranking within the judged set.

---

## ADR-008 -- Substitutes stay negatives in training

**Date:** 2026-10-06
**Status:** accepted

### Context

ESCI scores a Substitute 0.1 in nDCG but training pushed Substitutes away as
negatives: the objective and the metric disagree. v4 = v3 with
`--mask-threshold 0.1`, which removes Substitutes from both the in-batch block
and each query's own negatives. One variable. Real negatives per query fell
from about 0.60 to 0.28 of slots.

Prediction recorded before the run: Mode B flat or slightly down, Mode A
possibly up.

### Results

Test fold, paired bootstrap, v4 minus v3:

| metric | v3 | v4 | difference |
|---|---|---|---|
| Mode B Recall@100 | 0.6343 | 0.6100 | -0.0243 [-0.0259, -0.0228] |
| Mode A nDCG@10, all | 0.8919 | 0.8844 | -0.0075 [-0.0083, -0.0068] |
| Mode A nDCG@10, unsaturated | 0.8014 | 0.7871 | -0.0144 [-0.0163, -0.0125] |

The prediction was wrong on both counts: Mode B fell 2.4 points and Mode A
fell too.

### Decision

Keep `--mask-threshold 1.0`: only Exact products are masked, Substitutes are
negatives. The "weight negatives by 1 - gain" variant is dropped; it sits
between two settings and the gap between them favours hard negatives.

### Consequences

Human-labelled Substitutes are the most valuable negatives in the data.
Removing them cost about 3.5x what batch 64 gained (ADR-007). The metric's
partial credit for Substitutes does not make them good training targets: the
model has to learn what separates exact from nearly exact.

---

## ADR-009 -- Serving index: HNSW, M=16, efSearch=256

**Date:** 2026-10-06
**Status:** accepted

### Context

The served path needs single-query search over 1,215,854 products well
inside a 100 ms p99 budget on CPU. Exact search takes 508 ms p99 on one
thread. Swept on `biencoder_v3` embeddings, 2,000 seeded test queries
(`scripts/sweep_index.py`; full table in `benchmarks/index_sweep.md`, chart
in `benchmarks/figures/pareto.png`). Latency is one query per call on one
thread, as a request handler searches.

| index | recall vs exact | judged Recall@100 | size | p99 |
|---|---|---|---|---|
| Flat (exact) | 0.9998 | 0.6341 | 1,868 MB | 508 ms |
| HNSW M=16, efSearch=128 | 0.9195 | 0.6038 | 2,043 MB | 0.63 ms |
| **HNSW M=16, efSearch=256** | **0.9570** | **0.6133** | **2,043 MB** | **1.37 ms** |
| HNSW M=64, efSearch=256 | 0.9677 | 0.6181 | 2,510 MB | 6.28 ms |
| IVF-PQ nlist=1024, m=96, nprobe=64 | 0.8061 | 0.5845 | 128 MB | 27.4 ms |
| IVF-PQ nlist=4096, m=48, nprobe=64 | 0.6363 | 0.5442 | 75 MB | 5.6 ms |

### Decision

HNSW, M=16, efConstruction=200, efSearch=256. It gives up 2.1 points of judged
recall to exact search and answers in 1.4 ms p99.

### Alternatives rejected

- **IVF-PQ.** 16x smaller, but product quantisation at these code sizes loses
  5 to 9 points of judged recall, more than fine-tuning gained over zero-shot.
  The plan expected IVF-PQ to make deployment cheap; measured, the memory is
  not worth the recall.
- **Larger HNSW graphs.** M=64 buys 0.5 points for 4.6x the latency and 470 MB.
- **efSearch=128.** Halves latency that is already negligible, for 1 point.

### Consequences

- Needs ~2 GB resident for the index: the Cloud Run instance gets 4 GiB.
- If memory becomes the constraint, HNSW over 8-bit scalar-quantised vectors
  is the next thing to measure, not IVF-PQ.
- Recall at the served path is 0.6133 on this sample against 0.6341 exact;
  reported results state which path they come from.

---

## ADR-010 -- Mined negatives replacing ESCI labels: rejected

**Date:** 2026-10-06
**Status:** accepted

### Context

ANCE-style mining with v3: retrieve top-100 for each of 67,494 training
queries, drop judged-relevant products, skip the top 3 ranks, sample 12 of
the rest per query (`scripts/mine_negatives.py`). v5 = v3's configuration
with `--negatives mined`: the mined products replace the S/C/I negatives.

Mining statistics: 809,928 negatives, 12 per query; **98.25% were never
judged**, and 94.63% of the discarded top-3 products were unjudged too.

### Results

Test fold, paired bootstrap, v5 minus v3:

| metric | v3 | v5 | difference |
|---|---|---|---|
| Mode B Recall@100 | 0.6343 | 0.6248 | -0.0095 [-0.0117, -0.0075] |
| Mode A nDCG@10, all | 0.8919 | 0.8724 | -0.0196 [-0.0207, -0.0184] |
| Mode A nDCG@10, unsaturated | 0.8014 | 0.7717 | -0.0297 [-0.0325, -0.0268] |

Split by whether a test query has any S/C/I judgements: Mode B falls on both
(-0.0072 with none, -0.0103 with some); Mode A falls only where S/C/I exist
(-0.0256), since an all-Exact judged set scores the same in any order.

### Decision

Mined negatives do not replace human-labelled ones. ANCE rounds 2 and 3 are
not run on top of a step that loses; v6 tests whether mined negatives *add*
to the labels (`--negatives mixed`).

### Interpretation

Two causes overlap. Replacing the labels drops the Substitutes, which ADR-008
measured at -2.4 points on their own. But Mode B also falls on queries with no
S/C/I labels at all, where nothing was dropped, which points at false
negatives: nearly every mined product is unjudged, so any relevant ones among
them are trained as wrong answers. The three guards reduce this; they do not
remove it.

### Update 2026-10-08 -- v6, labels plus mined

v6 = v3's configuration with `--negatives mixed`: each query's S/C/I
negatives interleaved with its mined ones. Paired bootstrap, v6 minus v3:

| metric | v3 | v6 | difference |
|---|---|---|---|
| Mode B Recall@100 | 0.6343 | 0.6378 | +0.0035 [+0.0016, +0.0053] |
| Mode A nDCG@10, all | 0.8919 | 0.8794 | -0.0125 [-0.0135, -0.0114] |
| Mode A nDCG@10, unsaturated | 0.8014 | 0.7833 | -0.0182 [-0.0206, -0.0157] |

On top of the labels, mined negatives buy a third of a point of retrieval and
cost ranking quality about 3.5 times that: v5's false-negative pattern,
diluted. **v3 remains the model.** ANCE rounds 2 and 3 are not run: re-mining
with a stronger model surfaces harder negatives, which raises the unjudged
share rather than lowering it, and none of the three guards can tell a wrong
answer from an unjudged right one. Week 4's finding is that on ESCI the
human-labelled near-misses beat mining.

---

## ADR-011 -- Cross-encoder reranker: the quality ceiling, not the served ranker

**Date:** 2026-10-10
**Status:** accepted

### Context

`cross-encoder/ms-marco-MiniLM-L-6-v2` fine-tuned one epoch on 1,256,164
(query, product, gain) pairs from the train fold: pointwise binary
cross-entropy with the ESCI gain (E 1.0, S 0.1, C 0.01, I 0) as a soft target,
lr 2e-5, batch 64, 256 tokens for query and product together. The plan said
MSE; MSE through a sigmoid has vanishing gradients on confident mistakes,
while BCE's gradient on the logit is (p - target).

Dev nDCG@10 (2,000 queries): 0.9047 before ESCI training, 0.9148 at step
5,000, 0.9153 at the end of the epoch (step 19,627). The gain came in the
first quarter; a second epoch is not worth running.

### Results

Two-stage: `biencoder_v3` retrieves the top 100 by exact search, the
cross-encoder reorders them. Test fold, paired bootstrap against v3 alone:

| metric | v3 | v3 -> CE | difference |
|---|---|---|---|
| Mode B Recall@100 | 0.6343 | 0.6343 | +0.0000 (same candidates, as required) |
| Mode B nDCG@10 of the list | 0.3757 | 0.3987 | +0.0230 [+0.0205, +0.0252] |
| Mode A nDCG@10, all | 0.8919 | 0.9016 | +0.0096 [+0.0082, +0.0110] |
| Mode A nDCG@10, unsaturated | 0.8014 | 0.8161 | +0.0147 [+0.0113, +0.0179] |

Scoring one query's 100 candidates (256 tokens):

| device | p50 | p99 |
|---|---|---|
| RTX 4050, bf16 | 155 ms | 169 ms |
| CPU, 14 threads, fp32 | 7,371 ms | 7,859 ms |

### Decision

The cross-encoder is the reported quality ceiling and the teacher for
distillation (week 7). It is not served.

### Consequences

- The plan estimated 200-400 ms on CPU for 100 pairs. Measured, it is 7.4 s:
  100 candidates x ~256 tokens is ~25k tokens through the model per query.
  No CPU tuning closes a 70x gap to a 100 ms budget, so the served ranker
  must be distilled.
- Mode B nDCG@10 was added to scoring for this: Recall@100 cannot see
  reordering. It counts unjudged products as gain 0, so like recall it is a
  lower bound.

---

## ADR-012 -- Reciprocal rank fusion: a better candidate set, a worse ranking

**Date:** 2026-10-10
**Status:** accepted

### Context

RRF over BM25 and `biencoder_v3`, k=60 untuned, each contributing its top
100. Ranks rather than scores, because BM25 scores and cosine similarities
are on unrelated scales.

### Results

Test fold, paired bootstrap against v3 alone:

| metric | v3 | RRF | difference |
|---|---|---|---|
| Mode B Recall@100 | 0.6343 | 0.6497 | +0.0154 [+0.0137, +0.0174] |
| Mode B nDCG@10 of the list | 0.3757 | 0.3861 | +0.0104 [+0.0084, +0.0126] |
| Mode A nDCG@10, all | 0.8919 | 0.8444 | -0.0476 [-0.0492, -0.0459] |
| Mode A nDCG@10, unsaturated | 0.8014 | 0.7078 | -0.0936 [-0.0971, -0.0900] |

### Decision

Use RRF as a candidate generator only, followed by the cross-encoder;
evaluated as `ce_v1_on_rrf`. Never use it to order a judged set.

### Consequences

- +1.5 points of Recall@100 is the largest retrieval gain after fine-tuning:
  BM25 still finds products the dense model misses.
- In Mode A, fusing BM25's weak ordering (0.7513) drags down v3's: the
  9.4-point loss on the unsaturated slice is the price of averaging in a
  worse ranker.
- Serving it means BM25 in the request path. Its latency and memory have to
  be measured before it is adopted for serving, not assumed.

### Not done: Matryoshka embeddings

The plan's week 5 includes training nested 384/256/128-d embeddings to
shrink the index. Skipped: ADR-009 found memory is not the binding
constraint (the HNSW index fits a 4 GiB instance), so an 80-minute training
run would answer a question the deployment does not ask.

### Update 2026-10-10 -- RRF then cross-encoder (`ce_v1_on_rrf`)

RRF(BM25, v3) supplies the top 100, the cross-encoder orders them. Paired
bootstrap against v3 alone:

| metric | v3 | RRF -> CE | difference |
|---|---|---|---|
| Mode B Recall@100 | 0.6343 | 0.6497 | +0.0154 [+0.0137, +0.0174] |
| Mode B nDCG@10 of the list | 0.3757 | 0.4043 | +0.0286 [+0.0261, +0.0311] |

Recall equals RRF's exactly and Mode A equals `ce_v1`'s exactly, both as they
must. The two stages compose: RRF's extra recall adds 0.56 points of nDCG@10
on top of the cross-encoder over v3 alone (0.3987). This is the project's
quality ceiling; week 7 distils it into something servable.

---

## ADR-013 -- Week 6: significance, slices and a regression test that can fail

**Date:** 2026-10-10
**Status:** accepted

### Significance: paired permutation test beside the bootstrap

Every paired comparison in `scripts/compare.py` now carries a two-sided
paired permutation p-value (10,000 random sign flips of the per-query
differences, (count+1)/(n+1), so it bottoms out at ~0.0001). The bootstrap CI
answers "how uncertain is this difference"; the permutation test answers
"could it be chance" without a normality assumption, which nDCG and recall
violate: bounded, skewed, 30% of queries at exactly 1.0. A paired t-test was
rejected for that reason.

With 22,458 queries nearly every difference in this project is significant,
including v2's +0.0009 Mode A edge over v3. Significance here mostly says the
test fold is large; effect size decides. ADR-007 already acted on that.

### Slices

`eval/slices.py` defines query slices from the query text and judgements
only, never from any system's output: length (1-2, 3-4, 5-7, 8+ words),
contains a digit, names the brand of one of its Exact products (whole word,
case-insensitive, generic brands excluded), and saturation (<10 Exact).
`scripts/slice_report.py` writes `benchmarks/slices.md`. Mode B Recall@100,
test fold:

| slice | queries | BM25 | v3 | v3 - BM25 [95% CI] |
|---|---|---|---|---|
| has a number | 3,573 | 0.5739 | 0.6943 | +0.1205 [+0.1105, +0.1312] |
| no number | 18,885 | 0.5575 | 0.6229 | +0.0654 [+0.0620, +0.0688] |
| names a brand | 3,903 | 0.7840 | 0.8117 | +0.0277 [+0.0201, +0.0348] |
| names no brand | 18,555 | 0.5130 | 0.5970 | +0.0840 [+0.0803, +0.0874] |
| 1-2 words | 5,878 | 0.5090 | 0.5723 | +0.0634 [+0.0562, +0.0707] |

- Against the usual expectation, v3's lead is largest on queries with
  numbers (sizes, model numbers, counts). Zero-shot bge also beats BM25 there.
- Brand queries are BM25's strength (0.784) and where v3's lead is smallest.
  RRF's gain over v3 is largest on them: that is what BM25 contributes to
  fusion (ADR-012).
- v3 beats BM25 on average in every slice, so where dense fails has to be
  found per query: `docs/failure_analysis.md`.

### Regression test: pinned value, not a floor

`tests/regression/test_floor.py` encodes a committed fixture (20 unsaturated
test queries, 355 judged products, 0.09 MB, built by
`scripts/make_regression_fixture.py`) with zero-shot bge on CPU and asserts
Mode A nDCG@10 equals a pinned value (0.758770) within 0.003. A second test
pins v3 (0.772805) and skips when the checkpoint is absent, as in CI.

A floor was the plan and was rejected after measuring what it should catch.
On a 50-query fixture, dropping the bge query prefix cost 0.008 and mean
pooling 0.006, under any sensible floor margin. On the final 20-query fixture
dropping the prefix *raised* the score, to 0.768216 (+0.0094), and mean
pooling lowered it to 0.753921 (-0.0048). A floor cannot catch a bug that
scores higher on a small sample; a pinned value catches drift either way. The zero-shot model never changes and CPU
fp32 is deterministic, so any drift means the pipeline changed. The fixture
is drawn from unsaturated queries because saturated ones score ~1.0 under
almost any ranking and hide exactly these bugs.

---

## ADR-014 -- Interleaving, simulated: agrees with offline metrics in direction

**Date:** 2026-10-10
**Status:** accepted

### Context

Team-draft interleaving (Radlinski et al. 2008) over each pair's top 10, on
2,000 seeded test queries x 5 sessions. Clicks come from a cascade model
whose click and stop probabilities are assumptions keyed on ESCI labels
(`eval/interleave.py`). It demonstrates the method; it is not evidence
about users.

### Results (`benchmarks/interleaving.md`)

| A | B | preference for B [95% CI] | offline nDCG@10, B - A | agree |
|---|---|---|---|---|
| BM25 | v3 | +0.136 [+0.114, +0.159] | +0.0584 | yes |
| v3 | RRF | +0.020 [-0.003, +0.043] | +0.0134 | yes, not significant |
| v3 | RRF -> CE | +0.047 [+0.024, +0.070] | +0.0272 | yes |

### Consequences

- Interleaving orders the three comparisons as the offline metric does.
- It does not resolve v3 vs RRF at this volume: a 1.3-point offline gain is
  inside its noise. Small differences need more sessions.
- The intervals treat sessions as independent; the 5 sessions on one query
  are correlated, so they are somewhat too narrow. Resampling queries rather
  than sessions is the fix. No conclusion above would change: the two
  significant results are well clear of zero, and the third already is not
  significant.

---

## Week 6 gate -- 2026-10-10

Paired bootstrap CIs and permutation p-values on every comparison
(`scripts/compare.py`); Mode A reported overall and on the unsaturated
slice; query slices (ADR-013); failure analysis written by hand
(`docs/failure_analysis.md`); a pinned-value regression test running in CI;
interleaving demonstrated. `results.jsonl` rows are point estimates by
design (append-only log); every reported comparison carries an interval.

---

## ADR-015 -- Distillation with margin-MSE at weight 1 hurts retrieval

**Date:** 2026-10-11
**Status:** accepted; v8 (weight 0.1) pending

### Context

The cross-encoder cannot be served (ADR-011), so its ranking was distilled
into a bi-encoder student. Teacher: `ce_v1` logits for all 1,256,164 judged
train pairs (`scripts/score_teacher.py`). Student `biencoder_v7`: v3's
configuration plus margin-MSE (Hofstätter et al. 2020) on each query's
(positive, negative) pairs: loss = InfoNCE + 1.0 x margin-MSE, student
scores as cosine / 0.05. One variable against v3.

Teacher coverage was complete (`teacher_valid_rate` = `neg_valid_rate` =
0.59 throughout) and margin-MSE fell from 5.3 to 3.0.

### Results

Paired bootstrap, v7 minus v3, test fold:

| metric | v3 | v7 | difference |
|---|---|---|---|
| Mode B Recall@100 | 0.6343 | 0.5952 | -0.0391 [-0.0409, -0.0372] |
| Mode B nDCG@10 of the list | 0.3757 | 0.3569 | -0.0188 [-0.0203, -0.0173] |
| Mode A nDCG@10, all | 0.8919 | 0.8928 | +0.0008 [+0.0001, +0.0016] |
| Mode A nDCG@10, unsaturated | 0.8014 | 0.8030 | +0.0016 [-0.0003, +0.0034], p=0.089 |

The cross-encoder adds +0.0147 on the unsaturated slice; v7 recovers about
a tenth of it, not significantly, and loses 3.9 points of recall.

### Diagnosis

On 2,000 training triples (query, Exact, non-Exact), mean margin in logit
units: teacher 2.24, v3 2.62, v7 2.20; correlation with the teacher 0.688
for v3, 0.775 for v7. The student learned exactly what it was asked: its
margins match the teacher's. The scales were compatible, so a scale
mismatch is not the cause.

The weighting is. At the end of training margin-MSE was ~3.0 against
InfoNCE ~1.05, while v3 finished InfoNCE at 0.87. The margin term dominated
the gradient, and it only knows judged pairs: Exact against its near-misses.
Full-corpus recall depends on separating Exact products from the other 1.2M,
which only InfoNCE's in-batch negatives teach, and that got worse.

### Decision

Weight 1.0 was an unexamined default, chosen without comparing the two
losses' magnitudes. Re-run at 0.1 (v8), one variable. Recorded prediction:
Recall@100 within ~0.005 of v3, a small Mode A gain. If v8 does not beat v3,
distillation is closed and v3 is the served model.

---

## ADR-016 -- Served query encoder: ONNX fp32, opportunistic batching of 8

**Date:** 2026-10-11
**Status:** accepted

### Context

Each request encodes one query on a 2-vCPU CPU instance; product vectors are
computed offline. Model: `biencoder_v3`. All timings on an idle laptop CPU.

### Export and quantisation

`scripts/export_onnx.py` exports the query path (encoder, CLS pooling, L2
norm) and checks parity on 1,000 real test queries before anything is timed:
fp32 max |diff| 2.6e-7, cosine 1.000000. Dynamic INT8: mean cosine 0.970,
min 0.908.

Quality with the same product vectors, exact search (`bench_query_encoder.py`):

| query encoder | Recall@100 vs PyTorch, 3,000 queries | 1,000 queries |
|---|---|---|
| ONNX fp32 | +0.0000 (identical) | +0.0000 |
| ONNX INT8 | -0.0054 | -0.0073 |

One query per call:

| encoder | 1 thread p50 / p99 | 2 threads p50 / p99 |
|---|---|---|
| PyTorch fp32 | 14.4 / 19.3 ms | 9.5 / 11.9 ms |
| ONNX fp32 | 8.9 / 11.7 ms | 6.0 / 8.3 ms |
| ONNX INT8 | 3.7 / 6.3 ms | 2.9 / 4.1 ms |

**Decision: ONNX fp32.** INT8 saves ~4 ms of p99 inside a 100 ms budget and
costs half a point of recall and nDCG. A saving no user can perceive does not
pay for a measurable quality loss.

### Micro-batching

Closed-loop load against `serving/batcher.py`, ONNX fp32 at 2 threads
(`scripts/sweep_batcher.py`, `benchmarks/batcher_sweep.md`):

| setting | 1 client p50 / p99 | 8 clients q/s, p99 | 32 clients q/s |
|---|---|---|---|
| no batching | 6.2 / 7.9 ms | 163, 55 ms | 159 |
| batch 8, 2 ms window | 15.8 / 31.0 ms | 250, 43 ms | 249 |
| batch 8, opportunistic | 6.1 / 7.9 ms | 250, 43 ms | 248 |

Batching raises capacity ~53%, to ~250 queries/s; batch 16 or 32 adds no
throughput and worsens p99 (every query pads to the batch's longest). A
timed window costs a single client 10 ms because an asyncio wait on Windows
sleeps a ~15.6 ms timer tick. **Decision: max_batch 8, no window**: take the
first request plus whatever is already queued. Batches form on their own
under load, and an idle server adds no latency.

### Consequences

- Projected served path: ~8 ms encode + ~1.4 ms HNSW (ADR-009) plus HTTP.
  The week 8 load test measures it end to end; this does not replace it.
- 250 queries/s per instance is the encoder ceiling on 2 vCPU, well above the
  50 queries/s target.

### Update 2026-10-12 -- v8, margin weight 0.1: closed, v3 is served

v8 = v7 with margin-MSE weight 0.1. InfoNCE finished at 0.868 (v3: 0.867),
so the margin term no longer swamps it. Paired bootstrap against v3:

| metric | v3 | v8 | difference |
|---|---|---|---|
| Mode B Recall@100 | 0.6343 | 0.6268 | -0.0075 [-0.0086, -0.0064] |
| Mode B nDCG@10 of the list | 0.3757 | 0.3728 | -0.0029 [-0.0039, -0.0018] |
| Mode A nDCG@10, all | 0.8919 | 0.8937 | +0.0018 [+0.0013, +0.0023] |
| Mode A nDCG@10, unsaturated | 0.8014 | 0.8043 | +0.0029 [+0.0015, +0.0042] |

The recorded prediction (recall within 0.005 of v3) missed: -0.0075.

Recovery of the cross-encoder's gain on the unsaturated slice: v7 11% (not
significant), v8 20%. On the headline Mode B metrics, neither recovers any;
both lose.

**Decision:** distillation is closed as a negative result and `biencoder_v3`
is the served model. The teacher scored only judged pairs, so it can teach
"Exact against its near-misses" but never "Exact against the other 1.2M
products", which is what Recall@100 rewards; every margin weight tried
traded the second for the first. The fixes are known and out of budget:
teacher-scoring each batch's in-batch pairs is ~4,096 cross-encoder calls
per step (~2.6 s/step, an 80-minute run becomes ~90 hours), or a denoised
mining pipeline where the teacher relabels mined candidates.

---

## Week 7 gate -- 2026-10-12

Served path decided and measured component by component: `biencoder_v3`,
query encoding in ONNX fp32 (6.0 / 8.3 ms p50 / p99 on 2 threads, ADR-016),
opportunistic batching of 8 (~250 queries/s per instance), HNSW M=16
efSearch=256 (1.37 ms p99, ADR-009). The cross-encoder stays the reported
ceiling (ADR-011); distillation did not reach it (ADR-015). INT8 rejected
(ADR-016).
