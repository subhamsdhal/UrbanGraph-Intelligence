"""Generate invented, deterministic energy demo fixtures using only the standard library.

No project dataset or research artifact is read. This file is itself included in the
checkpoint so a restored copy can independently regenerate and compare fixtures.
"""

from __future__ import annotations

import argparse
import csv
import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import cast

SCHEMA = "urbangraph-synthetic-demo-1.0"
START = datetime(2030, 1, 1, tzinfo=UTC)
HOURS = 2400
REGIONS = ("SYN_A", "SYN_B", "SYN_C")
ORIGIN_DAYS = tuple(range(90, 97))


def load(region: int, hour_index: int) -> float:
    """Invented integer MW curve; unrelated to any real energy series."""
    day, hour = divmod(hour_index, 24)
    daytime = max(0, 12 - abs(hour - 15))
    weekly = 7 if day % 7 < 5 else -5
    return float(180 + 100 * region + 3 * daytime + weekly + day // 10)


def timestamp(hour_index: int) -> str:
    return (START + timedelta(hours=hour_index)).strftime("%Y-%m-%dT%H:%M:%SZ")


def json_bytes(value: object) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n").encode("utf-8")


def write_exact(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != data:
            raise RuntimeError(f"Existing synthetic fixture differs: {path.name}")
        return
    with path.open("xb") as handle:
        handle.write(data)


def generate(destination: Path) -> None:
    if destination.is_symlink():
        raise RuntimeError("Refusing symlink destination")
    destination.mkdir(parents=True, exist_ok=True)
    rows = [
        [timestamp(t), *(f"{load(region, t):.1f}" for region in range(3))] for t in range(HOURS)
    ]
    from io import StringIO

    buffer = StringIO(newline="")
    writer = csv.writer(buffer, lineterminator="\n")
    writer.writerow(["utc_timestamp", *(f"{region}_load_MW" for region in REGIONS)])
    writer.writerows(rows)
    write_exact(destination / "hourly_energy.csv", buffer.getvalue().encode("utf-8"))

    edges = [
        {"from": "SYN_A", "to": "SYN_B", "weight": 0.7},
        {"from": "SYN_B", "to": "SYN_C", "weight": 0.6},
        {"from": "SYN_C", "to": "SYN_A", "weight": 0.4},
    ]
    graph = {
        "schema_version": SCHEMA,
        "synthetic": True,
        "coordinate_system": "invented unit square; not latitude/longitude",
        "regions": [
            {"id": "SYN_A", "label": "Fictional A", "x": 0.1, "y": 0.3},
            {"id": "SYN_B", "label": "Fictional B", "x": 0.7, "y": 0.2},
            {"id": "SYN_C", "label": "Fictional C", "x": 0.5, "y": 0.8},
        ],
        "edges": edges,
        "meaning": "display-only invented graph; no electrical or geographic inference",
    }
    write_exact(destination / "graph.json", json_bytes(graph))

    forecasts = []
    for day in ORIGIN_DAYS:
        origin_index = day * 24 - 1
        for region_index, region in enumerate(REGIONS):
            for horizon in range(1, 25):
                target_index = origin_index + horizon
                actual = load(region_index, target_index)
                prediction = load(region_index, target_index - 24) + (region_index - 1) * 2
                half_width = 8 + horizon / 4
                forecasts.append(
                    {
                        "origin_utc": timestamp(origin_index),
                        "target_utc": timestamp(target_index),
                        "region_id": region,
                        "horizon_hours": horizon,
                        "truth_MW": actual,
                        "point_MW": prediction,
                        "lower_MW": prediction - half_width,
                        "upper_MW": prediction + half_width,
                    }
                )
    write_exact(
        destination / "forecasts_and_intervals.json",
        json_bytes(
            {
                "schema_version": SCHEMA,
                "synthetic": True,
                "interval_label": "illustrative band; not conformal or calibrated",
                "forecast_rule": "previous-day same hour plus fixed regional offset",
                "records": forecasts,
            }
        ),
    )

    # A simple, explicitly fictional schedule. Round-trip efficiency is 0.90.
    # Four 5-MW charging hours and four 4.5-MW discharging hours return to 50 MWh.
    eta = 0.9**0.5
    soc = 50.0
    dispatch = []
    day = ORIGIN_DAYS[0]
    for hour in range(24):
        charge = 5.0 if 0 <= hour < 4 else 0.0
        discharge = 4.5 if 17 <= hour < 21 else 0.0
        price = 35 if hour < 6 else 110 if 17 <= hour < 22 else 65
        soc += eta * charge - discharge / eta
        load_mw = load(0, day * 24 + hour)
        dispatch.append(
            {
                "target_utc": timestamp(day * 24 + hour),
                "region_id": "SYN_A",
                "load_MW": load_mw,
                "price_synthetic_CU_per_MWh": price,
                "charge_MW": charge,
                "discharge_MW": discharge,
                "import_MW": load_mw + charge - discharge,
                "soc_MWh_end": round(soc, 9),
            }
        )
    assert abs(soc - 50) < 1e-9
    assert all(10 <= cast(float, row["soc_MWh_end"]) <= 100 for row in dispatch)
    write_exact(
        destination / "battery_scenario.json",
        json_bytes(
            {
                "schema_version": SCHEMA,
                "synthetic": True,
                "warning": "Illustrative schedule and CU tariff; no market or real-asset claim",
                "battery": {
                    "capacity_MWh": 100,
                    "power_limit_MW": 50,
                    "roundtrip_efficiency": 0.9,
                    "initial_terminal_soc_MWh": 50,
                },
                "records": dispatch,
            }
        ),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--destination", type=Path, required=True)
    arguments = parser.parse_args()
    generate(arguments.destination)
    print("Generated deterministic synthetic fixtures")
