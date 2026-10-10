# Failure analysis: where the dense retriever loses to BM25

Read by hand, 10 October 2026. System: `biencoder_v3` (fine-tuned bge-small),
compared with BM25 on Mode B Recall@100, test fold.

## How often, and which queries

| outcome per query | queries |
|---|---|
| dense better than BM25 | 9,709 |
| BM25 better than dense | 4,540 |
| **dense found none, BM25 found all** | **113** |
| dense found all, BM25 found none | 299 |

Complete failures are rare and outnumbered by the reverse. 62% of the 113
have exactly one Exact judgement, against 4.7% of all test queries: with one
judged answer, recall is all-or-nothing, so single-judgement queries dominate
any list of extreme cases. That shapes what follows.

## The 20 cases read

`scripts/failure_cases.py` lists the 20 largest losses. All 20 are tied at
dense 0.00 / BM25 1.00, and the tie-break by `query_id` makes them the first
20 of the 113 in roughly alphabetical order: a sample not chosen by content,
but not a random one either. Raw dump: `artifacts/failure_cases.md`.

### 1. Unknown tokens: a real weakness (9 of 20)

The query hinges on a string the encoder has rarely or never seen: a rare
brand, a misspelling, an abbreviation, another language, or an alphanumeric
code. WordPiece splits it into fragments, and the embedding follows the
fragments to lookalike products. BM25 matches the literal token.

| query | what dense retrieved instead |
|---|---|
| `arteckin` (brand) | Arteholic, VOSS Artesian water |
| `chassy powder` (Chassis brand, misspelled) | chai powder, chaste tree powder |
| `990k` (glove model) | KTM 990, Intel i9-9900K |
| `aeramax 290 small room` (air purifier) | Radeon R9 290 graphics card, PC cases |
| `blk cigars` (abbreviation) | chocolate cigars, Big Green Egg (BG) |
| `auto park sensor benzel` (bezel) | parking sensors, but not the bezel |
| `11.5 x 8 clear plastic bux` (box) | 8.5x11 document holders |
| `autocollant moto` (French: motorcycle sticker) | Moto Metal wheels, MotoTec |
| `chillx - 2 - 5 ton ... air handlers` (brand) | fans, compressors |

The query slices (ADR-013) say v3 beats BM25 on queries with numbers by +0.12
overall, and these cases do not contradict it: the failures are numbers glued
into identifiers (`990k`, `aeramax 290`), not sizes and quantities.

### 2. Judgement artefacts: not dense failures (7 of 20)

The single Exact judgement is doubtful, or dense returned something at least
as good that nobody judged.

- `10 succulent plants without plastic containers`: the only Exact product is
  an empty hanging planter. Dense returned live succulent packs (unjudged).
- `4moms pack and play`: the Exact is a mattress pad. Dense returned the
  actual 4moms Breeze playards, judged Substitute.
- `cake topper sword pirate`: the Exact is plain bamboo cocktail picks. Dense
  returned pirate cake toppers, judged Substitute.
- `chocolate banana flavored peanut spread`: dense's second result is the
  same Legendary Foods Chocolate Banana product under another listing,
  unjudged.
- `brown spot remover for skin`, `canvas storage tote`: dense's top five are
  all plausible matches, all unjudged.
- `above the ground pool 60`: dense returned above-ground pools; the "60" is
  ambiguous and the one Exact is a 24 ft pool.

These are the pooling limitation of ADR-002 seen from the other side: Mode B
recall counts unjudged products as misses, so a model that finds different
correct products is scored as wrong.

### 3. Ambiguous or underspecified queries (4 of 20)

- `blue stained glasses`: stained-glass sheets or blue-light eyeglasses.
  Dense chose eyeglasses.
- `black nvg mount`: night-vision helmet mounts, or the GoPro mounts that
  share the name. Dense's top results were the GoPro ones (judged S and C).
- `boutique shirts`: no constraint beyond "shirt"; one Exact product.
- `cheetah tassel earrings`: dense's first result is a cheetah tassel
  earring-and-necklace set, judged Substitute: a near miss on "set".

## What follows

1. **Dense's real weakness is unfamiliar strings.** That is exactly what BM25
   contributes, which is why RRF adds 1.5 points of Recall@100 and gains most
   on brand queries (ADR-012, ADR-013). Fusion is the fix already in place.
2. **Not tried, and in order of cost:** spelling correction before encoding
   (would cover `chassy`, `bux`, `benzel`, `blk`); exact-match boosting for
   tokens absent from the encoder's training queries.
3. **About a third of the worst "failures" are measurement, not model.**
   Single-judgement queries plus unjudged near-duplicates make per-query
   recall on them noisy. Aggregates over 22k queries absorb this; reading
   individual worst cases without knowing it would overstate the problem.
