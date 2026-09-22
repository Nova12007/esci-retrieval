import polars as pl

df = (
    pl.scan_parquet("data/processed/judgements.parquet")
    .filter(pl.col("query_id") == 79706)
    .select(["query_id", "split"])
    .collect()
)

print(df.unique())
