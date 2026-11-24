#!/usr/bin/env python3
"""Inspect archived JSON only; never load weights or initialize MLX/Metal."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
import math
from pathlib import Path
import statistics


def read(path):
    return json.loads(path.read_text())


def analyze(root):
    run = root / "results/series2"
    reports = {name: read(run / f"mac/heldout-{name}.json")
               for name in ["R64", "W128-E4", "W128-S4"]}
    result = {"analysis_date": "2026-11-23", "analysis_is_post_hoc": True,
              "original_scores_unchanged": True, "mac": {}, "paired_changes": {},
              "metadata_estimates": {}, "development": {}}
    baseline = {row["id"]: row for row in reports["R64"]["cases"]}
    for name, report in reports.items():
        rows = report["cases"]
        missing = [row for row in rows if not row["has_final"]]
        zero = [row for row in rows if row["score"] == 0]
        completed_wrong = [row for row in rows if row["has_final"] and row["score"] < 1]
        result["mac"][name] = {
            "completed_cases": len(rows), "zero_scored_cases": len(zero),
            "missing_final": len(missing),
            "missing_final_at_192_tokens": sum(row["generation_tokens"] == 192 for row in missing),
            "missing_final_categories": dict(Counter(row["category"] for row in missing)),
            "zero_scored_completed_answers": [
                {key: row[key] for key in ["id", "category", "expected", "answer", "score"]}
                for row in completed_wrong],
        }
        if name != "R64":
            matched = [row for row in rows if row["id"] in baseline]
            result["paired_changes"][name] = {
                "matched_cases": len(matched),
                "differences": [{"id": row["id"], "category": row["category"],
                                 "baseline_score": baseline[row["id"]]["score"],
                                 "candidate_score": row["score"],
                                 "baseline_has_final": baseline[row["id"]]["has_final"],
                                 "candidate_has_final": row["has_final"],
                                 "baseline_tokens": baseline[row["id"]]["generation_tokens"],
                                 "candidate_tokens": row["generation_tokens"]}
                                for row in matched if row["score"] != baseline[row["id"]]["score"]],
            }
    remote = read(run / "leonardo/RESULTS.json")
    for evaluation in remote["complete_evaluations"]:
        rows = evaluation["records"]
        grouped = defaultdict(list)
        for row in rows:
            if row["category"] != "extraction":
                grouped[row["category"]].append(row["nll_per_token"])
        tokens = sum(row["answer_tokens"] for row in rows)
        result["development"][evaluation["candidate"]] = {
            "cases": len(rows), "answer_tokens": tokens,
            "structured_cases": sum(row["category"] == "structured" for row in rows),
            "structured_token_share": sum(row["answer_tokens"] for row in rows
                                          if row["category"] == "structured") / tokens,
            "case_mean_nll": statistics.mean(row["nll_per_token"] for row in rows),
            "category_mean_nll_without_extraction": statistics.mean(
                statistics.mean(values) for values in grouped.values()),
            "metrics": evaluation["metrics"],
        }
    for name, path in [
        ("R64", root / "results/series1/CONVERSION_REPORT.json"),
        ("W128-E4", run / "leonardo/exports/W128-E4-CONVERSION_REPORT.json"),
        ("W128-S4", run / "leonardo/exports/W128-S4-CONVERSION_REPORT.json"),
    ]:
        conversion = read(path)
        tensors = [value for key, value in conversion["tensor_signatures"].items()
                   if ".mlp.experts." in key and key.endswith((".scales", ".biases"))
                   and value["dtype"] == "mlx.core.float32"]
        savings = 2 * sum(math.prod(value["shape"]) for value in tensors)
        result["metadata_estimates"][name] = {
            "fp32_expert_metadata_tensors": len(tensors),
            "estimated_savings_bytes_if_bf16": savings,
            "estimated_savings_gib_if_bf16": savings / 2**30,
            "estimated_weight_gib_if_bf16": (conversion["weight_bytes"] - savings) / 2**30,
            "is_measured_export": False,
            "caveat": "Metadata rounding changes weights. Estimate omits small header changes; quality, speed and memory fit are untested.",
        }
    result["interpretation_limits"] = [
        "No extra generation was performed; longer output caps have not been tested.",
        "Conditional accuracy on completed answers is selection-biased and is not a replacement benchmark.",
        "Some completed responses contain the expected answer plus words; manual examples do not rescore the frozen reports.",
        "Routing/loop sensitivity is a hypothesis; no per-pass divergence traces exist in this run.",
    ]
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = analyze(args.repository)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(f"Saved JSON-only retrospective to {args.output}")


if __name__ == "__main__":
    main()
