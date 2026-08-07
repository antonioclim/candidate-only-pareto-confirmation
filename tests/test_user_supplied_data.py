from pathlib import Path
import csv
import numpy as np

from pcpi_candidate_tas.opsd_full import extract_official_market_year

def test_official_extraction_contract_without_hash(tmp_path):
    path = tmp_path / "mini.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "utc_timestamp",
                "DK_1_price_day_ahead",
                "DK_1_load_actual_entsoe_transparency",
            ],
        )
        writer.writeheader()
        writer.writerows(
            [
                {
                    "utc_timestamp": "2019-01-01T00:00:00Z",
                    "DK_1_price_day_ahead": "42.0",
                    "DK_1_load_actual_entsoe_transparency": "2300",
                },
                {
                    "utc_timestamp": "2019-07-01T00:00:00Z",
                    "DK_1_price_day_ahead": "44.0",
                    "DK_1_load_actual_entsoe_transparency": "2100",
                },
                {
                    "utc_timestamp": "2020-01-01T00:00:00Z",
                    "DK_1_price_day_ahead": "50.0",
                    "DK_1_load_actual_entsoe_transparency": "2500",
                },
            ]
        )
    rows = extract_official_market_year(path, verify_hash=False)
    assert len(rows) == 2
    assert rows[0]["split"] == "development"
    assert rows[1]["split"] == "confirmation"
    assert np.isclose(rows[0]["price_eur_per_mwh"], 42.0)
