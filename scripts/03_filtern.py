"""Filtert die piebro-Monatsdateien auf den Fernverkehr und schreibt eine Parquet-Datei.

Aufruf:  python scripts/03_filtern.py --start 2026-05-01 --ende 2026-10-31
         python scripts/03_filtern.py --eva 8000207 8000105   (nur ausgewählte Bahnhöfe)

Verspätungslogik (geprüft gegen die offizielle DB-Statistik):
- Basis ist die ANKUNFTSverspätung = arrival_change_time - arrival_planned_time,
  wie bei der betrieblichen Pünktlichkeit der DB (Ankunft < 6 min).
- delay_in_min bleibt nur als Rohwert erhalten; die Spalte stimmt nur bei ~44 %
  der Halte mit der Ankunftsverspätung überein.
- time und line_number fallen weg (line_number im Fernverkehr immer leer).
"""
import argparse
from datetime import date
from pathlib import Path

import polars as pl

FERNVERKEHR = ["ICE", "IC", "EC"]

AUSREISSER_MIN = -5    # früher als 5 min vor Plan: vermutlich Datenfehler
AUSREISSER_MAX = 300   # mehr als 5 h Verspätung: prüfen, nicht stillschweigend löschen

SPALTEN = [
    "id",                      # eindeutige ID des Halts
    "eva",                     # Bahnhofsnummer -> Schlüssel zu Dim_Bahnhof / StaDa
    "station_name",
    "xml_station_name",
    "train_type",              # ICE, IC, EC
    "train_number",
    "train_line_ride_id",      # eindeutige Fahrt
    "train_line_station_num",  # Position des Halts im Laufweg
    "final_destination_station",
    "arrival_planned_time",
    "arrival_change_time",
    "departure_planned_time",
    "departure_change_time",
    "delay_in_min",            # Rohwert, nicht für Pünktlichkeit verwenden
    "arrival_is_canceled",
    "departure_is_canceled",
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Fernverkehr aus piebro-Daten filtern")
    parser.add_argument("--input", default="data/raw/monthly_processed_data")
    parser.add_argument("--output", default="data/processed/halte_fernverkehr.parquet")
    parser.add_argument("--start", help="erstes Soll-Datum, z. B. 2026-05-01")
    parser.add_argument("--ende", help="letztes Soll-Datum, z. B. 2026-10-31")
    parser.add_argument("--eva", nargs="*", help="optional: nur diese EVA-Nummern behalten")
    args = parser.parse_args()

    dateien = sorted(Path(args.input).glob("data-*.parquet"))
    if not dateien:
        raise SystemExit(f"Keine Dateien in {args.input} gefunden – erst 01_download.py ausführen.")
    print("Lese:", *[d.name for d in dateien], sep="\n  ")

    ank_verspaetung = (
        (pl.col("arrival_change_time") - pl.col("arrival_planned_time")).dt.total_minutes()
    )

    lf = (
        pl.scan_parquet(dateien)
        .select(SPALTEN)
        .filter(pl.col("train_type").is_in(FERNVERKEHR))
        .with_columns(
            pl.coalesce("arrival_planned_time", "departure_planned_time").alias("soll_zeit"),
            pl.coalesce("station_name", "xml_station_name").alias("bahnhof_name"),
            (
                pl.col("arrival_is_canceled").fill_null(False)
                | pl.col("departure_is_canceled").fill_null(False)
            ).alias("ist_ausgefallen"),
            pl.col("arrival_planned_time").is_not_null().alias("hat_ankunft"),
            ank_verspaetung.cast(pl.Int32).alias("ank_verspaetung"),
        )
        .with_columns(
            # verfrühte Ankünfte zählen als 0 Minuten (für Ø Verspätung)
            pl.col("ank_verspaetung").clip(lower_bound=0).alias("verspaetung_bereinigt"),
            (
                (pl.col("ank_verspaetung") < AUSREISSER_MIN)
                | (pl.col("ank_verspaetung") > AUSREISSER_MAX)
            ).fill_null(False).alias("ist_ausreisser"),
        )
        .drop("station_name", "xml_station_name")
    )

    if args.start:
        lf = lf.filter(pl.col("soll_zeit").dt.date() >= date.fromisoformat(args.start))
    if args.ende:
        lf = lf.filter(pl.col("soll_zeit").dt.date() <= date.fromisoformat(args.ende))
    if args.eva:
        lf = lf.filter(pl.col("eva").is_in(args.eva))

    df = lf.unique(subset="id", keep="first").sort("soll_zeit").collect()

    ziel = Path(args.output)
    ziel.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(ziel, compression="zstd")

    print(f"\nGespeichert: {ziel}  ({ziel.stat().st_size / 1e6:.1f} MB)")
    print(
        df.select(
            pl.len().alias("halte"),
            pl.col("soll_zeit").min().alias("von"),
            pl.col("soll_zeit").max().alias("bis"),
            pl.col("eva").n_unique().alias("bahnhoefe"),
            pl.col("ist_ausgefallen").mean().alias("ausfallquote"),
            pl.col("ist_ausreisser").sum().alias("ausreisser"),
        )
    )
    print(df.group_by("train_type").len().sort("len", descending=True))

    # Validierung: Pünktlichkeit nach DB-Definition je Monat
    # (nur Ankünfte, ohne Ausfälle, Ankunftsverspätung < 6 min)
    print("\nPünktlichkeit je Monat (mit offizieller DB-Statistik vergleichen):")
    print(
        df.filter(pl.col("hat_ankunft") & ~pl.col("ist_ausgefallen"))
        .group_by(pl.col("soll_zeit").dt.strftime("%Y-%m").alias("monat"))
        .agg(
            pl.len().alias("ankuenfte"),
            (pl.col("ank_verspaetung") < 6).mean().round(3).alias("puenktlichkeit"),
            pl.col("verspaetung_bereinigt").mean().round(1).alias("mittel_min"),
            pl.col("ank_verspaetung").median().alias("median_min"),
        )
        .sort("monat")
    )


if __name__ == "__main__":
    main()