"""Recompute descriptive results from saved cases without model inference."""
from __future__ import annotations

import json
import random
import statistics
from pathlib import Path


def read_json(path: Path):
    return json.loads(path.read_text())


def cases_by_id(report: dict) -> dict:
    result = {}
    for row in report.get("cases", []):
        if row["id"] in result:
            raise ValueError(f"Duplicate case ID: {row['id']}")
        score = row["score"]
        if not isinstance(score, (float, int)) or not 0 <= score <= 1:
            raise ValueError(f"Invalid score for {row['id']}")
        result[row["id"]] = row
    return result


def paired_comparison(baseline: dict, candidate: dict, *, exclude: str | None = None) -> dict:
    """Use matched IDs only; partial results never qualify as complete."""
    base = cases_by_id(baseline)
    rows = [row for row in cases_by_id(candidate).values()
            if row["id"] in base and row["category"] != exclude]
    for row in rows:
        other = base[row["id"]]
        if any(row[key] != other[key] for key in ["expected", "category", "modality"]):
            raise ValueError(f"Mismatched paired case definition: {row['id']}")
    if not rows:
        raise ValueError("No paired cases")
    changes = [row["score"] - base[row["id"]]["score"] for row in rows]
    rng = random.Random(20261121)
    bootstrap = sorted(statistics.mean(rng.choice(changes) for _ in changes)
                       for _ in range(2000))
    complete = (exclude is None and len(rows) == 180
                and baseline.get("status") == candidate.get("status") == "complete")
    return {
        "candidate": candidate["candidate"],
        "paired_cases": len(rows),
        "difference": statistics.mean(changes),
        "bootstrap_interval_95": [bootstrap[49], bootstrap[1949]],
        "all_180_cases_completed": complete,
        "excluded_category": exclude,
        "post_hoc": exclude is not None,
    }


def summarize(directory: Path) -> dict:
    reports = {p.stem.removeprefix("heldout-"): read_json(p)
               for p in sorted((directory / "mac").glob("heldout-*.json"))}
    if "R64" not in reports:
        raise ValueError("Missing R64 baseline report")
    measurements = {}
    for name, report in reports.items():
        rows = list(cases_by_id(report).values())
        speeds = [row["generation_tps"] for row in rows if row.get("generation_tps")]
        measurements[name] = {
            "completed_cases": len(rows),
            "status": report["status"],
            "peak_mlx_gib": max((row["peak_mlx_bytes"] for row in rows), default=0) / 2**30,
            "median_generation_tps": statistics.median(speeds) if speeds else None,
            "warning_pressure_cases": sum(
                row.get("system_after", {}).get("kern.memorystatus_vm_pressure_level") == "2"
                for row in rows),
        }
    remote = read_json(directory / "leonardo/RESULTS.json")
    subset = read_json(directory / "leonardo/DEVELOPMENT_SUBSET.json")["ids"]
    source = [row for row in remote["reference_records"] if row["id"] in subset]
    if len(source) != len(subset) or {row["id"] for row in source} != set(subset):
        raise ValueError("Source likelihoods do not cover the fixed development subset")
    source_metrics = {}
    for modality in ["all", "text", "image"]:
        rows = [row for row in source if modality == "all" or row["modality"] == modality]
        tokens = sum(row["answer_tokens"] for row in rows)
        source_metrics[modality] = {
            "cases": len(rows), "answer_tokens": tokens,
            "nll_per_token": sum(row["nll_sum"] for row in rows) / tokens if tokens else None,
        }
    development = []
    for row in sorted(remote["complete_evaluations"], key=lambda row: row["candidate"]):
        if [case["id"] for case in row["records"]] != subset:
            raise ValueError(f"Development subset mismatch for {row['candidate']}")
        development.append({"candidate": row["candidate"], "weight_bytes": row["weight_bytes"],
                            "metrics": row["metrics"]})
    return {
        "scope": "Custom diagnostics; incomplete tests and uncontrolled memory pressure limit conclusions.",
        "measurements": measurements,
        "paired": [paired_comparison(reports["R64"], report)
                   for name, report in reports.items() if name != "R64"],
        "post_hoc_without_extraction": [
            paired_comparison(reports["R64"], report, exclude="extraction")
            for name, report in reports.items() if name != "R64"],
        "development": development,
        "source_reference_on_same_development_subset": source_metrics,
    }
