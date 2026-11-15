"""Lädt nur ausgewählte Monatsdateien aus piebro/deutsche-bahn-data.

Aufruf:  python scripts/01_download.py 2026-04 2026-05 2026-06
         python scripts/01_download.py --liste     (zeigt verfügbare Monate)
"""
import argparse

from huggingface_hub import HfApi, hf_hub_download

REPO_ID = "piebro/deutsche-bahn-data"
ORDNER = "monthly_processed_data"


def verfuegbare_monate() -> list[str]:
    dateien = HfApi().list_repo_files(REPO_ID, repo_type="dataset")
    return sorted(
        d.removeprefix(f"{ORDNER}/data-").removesuffix(".parquet")
        for d in dateien
        if d.startswith(f"{ORDNER}/data-") and d.endswith(".parquet")
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("monate", nargs="*", help="Monate im Format YYYY-MM")
    parser.add_argument("--liste", action="store_true", help="verfügbare Monate anzeigen")
    parser.add_argument("--ziel", default="data/raw")
    args = parser.parse_args()

    verfuegbar = verfuegbare_monate()
    if args.liste or not args.monate:
        print("Verfügbare Monate:", ", ".join(verfuegbar))
        return

    for monat in args.monate:
        if monat not in verfuegbar:
            print(f"Übersprungen: {monat} gibt es nicht im Datensatz")
            continue
        pfad = hf_hub_download(
            repo_id=REPO_ID,
            repo_type="dataset",
            filename=f"{ORDNER}/data-{monat}.parquet",
            local_dir=args.ziel,  # -> data/raw/monthly_processed_data/data-YYYY-MM.parquet
        )
        print("OK:", pfad)


if __name__ == "__main__":
    main()