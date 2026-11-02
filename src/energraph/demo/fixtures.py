"""Strict loader for the separately restored, approved synthetic checkpoint."""

from __future__ import annotations

import csv
import hashlib
import json
import math
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path, PurePosixPath
from typing import Any

EXPECTED_MANIFEST_SHA256 = "49ae5bf07146e8d61e3d726fe89c84f4f9643940baaf1b2b61b4151d50668b0c"
BUILTIN_MANIFEST_SHA256 = "b874f63bd9215a3defdf706691fdc703cf8c5147eaf7dce4415a237352ecf3bd"
EXPECTED_REGIONS = ("SYN_A", "SYN_B", "SYN_C")
SCHEMA = "eg006-demo-v1"


class FixtureError(ValueError):
    """An approved fixture is missing or fails its contract."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _utc(value: str) -> datetime:
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except (ValueError, TypeError) as exc:
        raise FixtureError("Invalid UTC timestamp") from exc
    if parsed.tzinfo is None or parsed.utcoffset() != timedelta(0):
        raise FixtureError("Timestamp must be UTC")
    return parsed.astimezone(UTC)


def _number(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise FixtureError("Expected numeric fixture value")
    result = float(value)
    if not math.isfinite(result):
        raise FixtureError("Nonfinite fixture value")
    return result


def _object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise FixtureError("Invalid JSON fixture") from exc
    if not isinstance(value, dict):
        raise FixtureError("Fixture root must be an object")
    return value


@dataclass(frozen=True)
class DemoData:
    manifest_sha256: str
    series: dict[str, tuple[dict[str, object], ...]]
    forecasts: dict[tuple[str, str], tuple[dict[str, object], ...]]
    graph: dict[str, Any]
    dispatch: dict[tuple[str, str], tuple[dict[str, object], ...]]
    battery: dict[str, Any]

    @property
    def origins(self) -> tuple[str, ...]:
        return tuple(sorted({origin for _, origin in self.forecasts}))


def _validate_manifest(root: Path) -> str:
    manifest_file = root / "manifest.json"
    if manifest_file.is_symlink() or not manifest_file.is_file():
        raise FixtureError("Missing or linked manifest")
    for path in root.rglob("*"):
        if path.is_symlink() or (hasattr(path, "is_junction") and path.is_junction()):
            raise FixtureError("Linked package entry is forbidden")
    digest = _sha256(manifest_file)
    if digest not in {EXPECTED_MANIFEST_SHA256, BUILTIN_MANIFEST_SHA256}:
        raise FixtureError("Approved manifest digest mismatch")
    manifest = _object(manifest_file)
    expected_schema, expected_count = (
        ("synthetic-recovery-manifest-1.0", 11)
        if digest == EXPECTED_MANIFEST_SHA256
        else ("eg006-demo-fixtures-1.0", 4)
    )
    if manifest.get("schema_version") != expected_schema:
        raise FixtureError("Unsupported manifest schema")
    items = manifest.get("files")
    if not isinstance(items, list) or len(items) != expected_count:
        raise FixtureError("Unexpected manifest inventory")
    seen: set[str] = set()
    total = 0
    for item in items:
        if not isinstance(item, dict):
            raise FixtureError("Invalid manifest record")
        name = item.get("relative_path")
        if not isinstance(name, str) or not name or "\\" in name:
            raise FixtureError("Unsafe manifest path")
        posix = PurePosixPath(name)
        if posix.is_absolute() or ".." in posix.parts or name in seen:
            raise FixtureError("Unsafe or duplicate manifest path")
        seen.add(name)
        path = root.joinpath(*posix.parts)
        if path.is_symlink() or not path.is_file():
            raise FixtureError("Missing or linked fixture")
        size = path.stat().st_size
        if size != item.get("size_bytes") or _sha256(path) != item.get("sha256"):
            raise FixtureError("Fixture digest or size mismatch")
        total += size
    found = {p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file()}
    if found != seen | {"manifest.json"} or manifest.get("file_count") != len(seen):
        raise FixtureError("Unexpected package files")
    expected_directories = {
        parent.as_posix()
        for name in seen
        for parent in PurePosixPath(name).parents
        if parent != PurePosixPath(".")
    }
    found_directories = {p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_dir()}
    if found_directories != expected_directories:
        raise FixtureError("Unexpected package directories")
    if manifest.get("total_bytes") != total:
        raise FixtureError("Manifest totals mismatch")
    return digest


def load_demo(root: Path) -> DemoData:
    """Load a complete synthetic package once; fail closed without fallback."""
    if root.is_symlink() or not root.is_dir():
        raise FixtureError("Demo package must be a real directory")
    managed_data = Path(__file__).resolve().parents[3] / "data"
    if root.resolve().is_relative_to(managed_data.resolve()):
        raise FixtureError("Demo package cannot be loaded from managed research data")
    digest = _validate_manifest(root)
    fixtures = root / "fixtures"
    graph = _object(fixtures / "graph.json")
    forecast_doc = _object(fixtures / "forecasts_and_intervals.json")
    battery_doc = _object(fixtures / "battery_scenario.json")
    for document in (graph, forecast_doc, battery_doc):
        if (
            document.get("synthetic") is not True
            or document.get("schema_version") != "urbangraph-synthetic-demo-1.0"
        ):
            raise FixtureError("Fixture lacks synthetic provenance")
    if graph.get("coordinate_system") != "invented unit square; not latitude/longitude":
        raise FixtureError("Graph must use fictional coordinates")
    nodes = graph.get("regions")
    edges = graph.get("edges")
    if not isinstance(nodes, list) or not isinstance(edges, list):
        raise FixtureError("Invalid graph")
    region_ids: list[str] = []
    for node in nodes:
        if not isinstance(node, dict) or not isinstance(node.get("id"), str):
            raise FixtureError("Invalid graph node identity")
        region_ids.append(node["id"])
    ids = tuple(region_ids)
    if tuple(sorted(ids)) != EXPECTED_REGIONS or len(ids) != len(nodes):
        raise FixtureError("Invalid region identities")
    for node in nodes:
        if not isinstance(node, dict) or not isinstance(node.get("label"), str):
            raise FixtureError("Invalid graph node")
        if not (0 <= _number(node.get("x")) <= 1 and 0 <= _number(node.get("y")) <= 1):
            raise FixtureError("Graph coordinates outside synthetic square")
    edge_keys: set[tuple[str, str]] = set()
    for edge in edges:
        if not isinstance(edge, dict) or edge.get("from") not in ids or edge.get("to") not in ids:
            raise FixtureError("Invalid graph edge")
        edge_key = (str(edge["from"]), str(edge["to"]))
        if edge_key[0] == edge_key[1] or edge_key in edge_keys:
            raise FixtureError("Duplicate or self-referential graph edge")
        edge_keys.add(edge_key)
        if not 0 <= _number(edge.get("weight")) <= 1:
            raise FixtureError("Invalid graph weight")

    series: dict[str, list[dict[str, object]]] = {region: [] for region in ids}
    with (fixtures / "hourly_energy.csv").open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        expected_columns = ["utc_timestamp", *(f"{region}_load_MW" for region in ids)]
        if reader.fieldnames != expected_columns:
            raise FixtureError("Unexpected series columns")
        previous: datetime | None = None
        for row in reader:
            instant = _utc(row["utc_timestamp"])
            if previous is not None and instant - previous != timedelta(hours=1):
                raise FixtureError("Non-hourly or unsorted series")
            previous = instant
            for region in ids:
                try:
                    load = _number(float(row[f"{region}_load_MW"]))
                except (ValueError, TypeError, KeyError) as exc:
                    raise FixtureError("Invalid numeric load") from exc
                if load < 0:
                    raise FixtureError("Negative load")
                series[region].append(
                    {"timestamp_utc": instant.isoformat().replace("+00:00", "Z"), "load_MW": load}
                )
    if any(len(rows) != 2400 for rows in series.values()):
        raise FixtureError("Incomplete series")
    truths = {
        region: {str(row["timestamp_utc"]): _number(row["load_MW"]) for row in rows}
        for region, rows in series.items()
    }

    records = forecast_doc.get("records")
    if not isinstance(records, list):
        raise FixtureError("Invalid forecast inventory")
    forecasts: dict[tuple[str, str], list[dict[str, object]]] = {}
    for row in records:
        if not isinstance(row, dict) or row.get("region_id") not in ids:
            raise FixtureError("Invalid forecast region")
        origin = _utc(row["origin_utc"])
        target = _utc(row["target_utc"])
        horizon = row.get("horizon_hours")
        if type(horizon) is not int or not 1 <= horizon <= 24:
            raise FixtureError("Invalid horizon")
        if target - origin != timedelta(hours=horizon):
            raise FixtureError("Forecast timestamp mismatch")
        lower = _number(row.get("lower_MW"))
        point = _number(row.get("point_MW"))
        upper = _number(row.get("upper_MW"))
        truth = _number(row.get("truth_MW"))
        if not 0 <= lower <= point <= upper or truth < 0:
            raise FixtureError("Invalid forecast band or truth")
        target_text = target.isoformat().replace("+00:00", "Z")
        if truths[str(row["region_id"])].get(target_text) != truth:
            raise FixtureError("Forecast truth differs from synthetic series")
        key = (row["region_id"], origin.isoformat().replace("+00:00", "Z"))
        forecasts.setdefault(key, []).append(
            {
                "horizon_hours": horizon,
                "target_utc": target.isoformat().replace("+00:00", "Z"),
                "point_MW": point,
                "truth_MW": truth,
                "lower_MW": lower,
                "upper_MW": upper,
            }
        )
    origins = {origin for _, origin in forecasts}
    if len(origins) != 7 or len(forecasts) != 21:
        raise FixtureError("Expected seven origins for every region")
    if any({origin for name, origin in forecasts if name == region} != origins for region in ids):
        raise FixtureError("Inconsistent regional forecast origins")
    for rows in forecasts.values():
        rows.sort(key=lambda row: _number(row["horizon_hours"]))
        if [row["horizon_hours"] for row in rows] != list(range(1, 25)):
            raise FixtureError("Missing or duplicate forecast horizon")

    battery = battery_doc.get("battery")
    dispatch_rows = battery_doc.get("records")
    if not isinstance(battery, dict) or not isinstance(dispatch_rows, list):
        raise FixtureError("Invalid battery fixture")
    capacity = _number(battery.get("capacity_MWh"))
    power = _number(battery.get("power_limit_MW"))
    initial = _number(battery.get("initial_terminal_soc_MWh"))
    efficiency = _number(battery.get("roundtrip_efficiency"))
    if not (capacity > 0 and power > 0 and 0 <= initial <= capacity and 0 < efficiency <= 1):
        raise FixtureError("Invalid battery parameters")
    dispatch: dict[tuple[str, str], list[dict[str, object]]] = {}
    for row in dispatch_rows:
        if not isinstance(row, dict) or row.get("region_id") not in ids:
            raise FixtureError("Invalid dispatch region")
        target = _utc(row["target_utc"])
        dispatch_origin = (
            (target.replace(hour=0) - timedelta(hours=1)).isoformat().replace("+00:00", "Z")
        )
        dispatch_key = (str(row["region_id"]), dispatch_origin)
        charge = _number(row.get("charge_MW"))
        discharge = _number(row.get("discharge_MW"))
        soc = _number(row.get("soc_MWh_end"))
        load = _number(row.get("load_MW"))
        imported = _number(row.get("import_MW"))
        price = _number(row.get("price_synthetic_CU_per_MWh"))
        if not (0 <= charge <= power and 0 <= discharge <= power and 0 <= soc <= capacity):
            raise FixtureError("Dispatch exceeds battery bounds")
        if charge > 0 and discharge > 0 or load < 0 or imported < 0 or price < 0:
            raise FixtureError("Invalid dispatch action")
        if abs(imported - (load + charge - discharge)) > 1e-6:
            raise FixtureError("Dispatch power balance mismatch")
        if truths[str(row["region_id"])].get(target.isoformat().replace("+00:00", "Z")) != load:
            raise FixtureError("Dispatch load differs from synthetic series")
        dispatch.setdefault(dispatch_key, []).append(
            {
                "target_utc": target.isoformat().replace("+00:00", "Z"),
                "load_MW": load,
                "import_MW": imported,
                "charge_MW": charge,
                "discharge_MW": discharge,
                "soc_MWh_end": soc,
                "price_synthetic_CU_per_MWh": price,
            }
        )
    if len(dispatch) != 1 or next(iter(dispatch)) not in forecasts:
        raise FixtureError("Unexpected illustrative dispatch population")
    for rows in dispatch.values():
        rows.sort(key=lambda row: str(row["target_utc"]))
        if len(rows) != 24:
            raise FixtureError("Incomplete dispatch example")
        previous_soc = initial
        for index, row in enumerate(rows):
            target = _utc(str(row["target_utc"]))
            if index and target - _utc(str(rows[index - 1]["target_utc"])) != timedelta(hours=1):
                raise FixtureError("Non-hourly dispatch example")
            expected_soc = (
                previous_soc
                + _number(row["charge_MW"]) * math.sqrt(efficiency)
                - _number(row["discharge_MW"]) / math.sqrt(efficiency)
            )
            if abs(_number(row["soc_MWh_end"]) - expected_soc) > 1e-5:
                raise FixtureError("Dispatch state mismatch")
            previous_soc = _number(row["soc_MWh_end"])
        if abs(previous_soc - initial) > 1e-5:
            raise FixtureError("Dispatch terminal state mismatch")
    return DemoData(
        manifest_sha256=digest,
        series={key: tuple(value) for key, value in series.items()},
        forecasts={key: tuple(value) for key, value in forecasts.items()},
        graph={"coordinate_system": "synthetic_unit_square", "regions": nodes, "edges": edges},
        dispatch={key: tuple(value) for key, value in dispatch.items()},
        battery=battery,
    )
