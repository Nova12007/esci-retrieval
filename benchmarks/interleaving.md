Simulated team-draft interleaving, 2,000 test queries x 5 sessions, top 10. Clicks are simulated from ESCI labels (eval/interleave.py): a methodology demonstration, not user evidence.

| A | B | B wins | A wins | ties | preference for B [95% CI] | offline nDCG@10 B - A | agree |
|---|---|---|---|---|---|---|---|
| bm25 | biencoder_v3 | 4,091 | 3,109 | 2,800 | +0.136 [+0.114, +0.159] | +0.0584 | yes |
| biencoder_v3 | rrf | 3,710 | 3,563 | 2,727 | +0.020 [-0.003, +0.043] | +0.0134 | yes |
| biencoder_v3 | rrf_ce | 3,782 | 3,441 | 2,777 | +0.047 [+0.024, +0.070] | +0.0272 | yes |
