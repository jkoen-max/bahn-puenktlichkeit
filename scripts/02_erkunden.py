"""Schneller Blick in eine Monatsdatei: Spalten, Zeilen, Zuggattungen.

Aufruf:  python scripts/02_erkunden.py data/raw/monthly_processed_data/data-2026-05.parquet
"""
import sys

import polars as pl

pfad = sys.argv[1]
lf = pl.scan_parquet(pfad)

print("Spalten und Typen:")
for name, typ in lf.collect_schema().items():
    print(f"  {name:28} {typ}")

print("\nZeilen gesamt:", lf.select(pl.len()).collect().item())

print("\nZuggattungen (Top 20):")
print(lf.group_by("train_type").len().sort("len", descending=True).head(20).collect())

print("\nBeispielzeilen Fernverkehr:")
with pl.Config(tbl_cols=-1,tbl_rows = 25, fmt_str_lengths=25):
    print(lf.filter(pl.col("train_type").is_in(["ICE", "IC", "EC"])).collect())