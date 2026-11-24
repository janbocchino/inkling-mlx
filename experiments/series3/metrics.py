"""Score explicit completed-answer events and retain right-censored traces."""
from __future__ import annotations
from collections import Counter
import json
import statistics


def correct(case, final):
    if not isinstance(final, str):
        return False
    if case["category"] == "structured":
        try:
            # Python equality treats True as 1. Canonical JSON preserves types.
            def unique_object(pairs):
                result = {}
                for key, value in pairs:
                    if key in result:
                        raise ValueError("Duplicate JSON field")
                    result[key] = value
                return result
            actual = json.loads(final, object_pairs_hook=unique_object)
            expected = json.loads(case["answer"])
            canonical = lambda value: json.dumps(value, sort_keys=True, allow_nan=False)
            return canonical(actual) == canonical(expected)
        except (ValueError, TypeError):
            return False
    return final.strip() == case["answer"].strip()


def score_trace(case, trace, budgets=(192, 512, 1536)):
    """final_end_token is a 1-based count through the closing delimiter/EOS."""
    count = trace["tokens_emitted"]
    end = trace.get("final_end_token")
    start = trace.get("final_start_token")
    if (type(count) is not int or count < 0
            or (end is not None and not (type(start) is int and type(end) is int and 0 <= start <= end <= count))):
        raise ValueError("Invalid completion event offsets")
    completed = end is not None
    answer_correct = completed and correct(case, trace.get("final_answer", ""))
    by_budget = {}
    for budget in budgets:
        # A trace cut at 192 cannot tell us whether it would finish at 512.
        known = completed or count >= budget or trace.get("stop_reason") == "eos"
        by_budget[str(budget)] = {
            "observed": known,
            "completed": bool(completed and end <= budget) if known else None,
            "correct": bool(answer_correct and end <= budget) if known else None}
    return {"id": case["id"], "category": case["category"],
            "tokens_emitted": count, "natural_completion_observed": completed,
            "right_censored": not completed and trace.get("stop_reason") in {"token_cap", "time_limit"},
            "stop_reason": trace.get("stop_reason"), "correct_final": bool(answer_correct),
            "at_budget": by_budget}


def summarize(rows):
    ids = [row["id"] for row in rows]
    if len(set(ids)) != len(ids):
        raise ValueError("Duplicate case ID; repeated seeds need separate clustered analysis")
    return {"cases": len(rows), "correct": sum(row["correct_final"] for row in rows),
            "completed": sum(row["natural_completion_observed"] for row in rows),
            "censored": sum(row["right_censored"] for row in rows),
            "errors_and_stops": dict(Counter(row["stop_reason"] for row in rows)),
            "mean_emitted_tokens_under_observed_limits": statistics.mean(row["tokens_emitted"] for row in rows) if rows else None,
            "warning": "Observed capped length is not the uncensored natural reasoning length"}

