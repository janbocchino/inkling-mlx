"""Small corruption and pairing checks; no MLX, model weights, or network."""
from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import struct
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from inkling_mlx.results import paired_comparison, summarize
from inkling_mlx.verify import verify_frozen, verify_model

spec = importlib.util.spec_from_file_location("portable_chat", ROOT / "scripts/ask_inkling.py")
chat = importlib.util.module_from_spec(spec)
spec.loader.exec_module(chat)


def record(identity, score, category="arithmetic"):
    return {"id": identity, "score": score, "category": category,
            "modality": "text", "expected": "42"}


def report(candidate, rows, status="incomplete"):
    return {"candidate": candidate, "cases": rows, "status": status}


class ComparisonTests(unittest.TestCase):
    def test_match_ids_not_positions_and_keep_partial_incomplete(self):
        base = report("R64", [record("a", 1), record("b", 0), record("c", 1)])
        other = report("W", [record("b", 1), record("a", 0), record("x", 1)])
        result = paired_comparison(base, other)
        self.assertEqual(result["paired_cases"], 2)
        self.assertEqual(result["difference"], 0)
        self.assertFalse(result["all_180_cases_completed"])

    def test_duplicate_and_mismatched_definition_rejected(self):
        base = report("R64", [record("a", 1)])
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            paired_comparison(base, report("W", [record("a", 1), record("a", 0)]))
        changed = record("a", 1)
        changed["expected"] = "43"
        with self.assertRaisesRegex(ValueError, "Mismatched"):
            paired_comparison(base, report("W", [changed]))

    def test_sensitivity_is_separate_and_labeled(self):
        base = report("R64", [record("a", 0, "extraction"), record("b", 1)])
        other = report("W", [record("a", 0, "extraction"), record("b", 0)])
        primary = paired_comparison(base, other)
        sensitivity = paired_comparison(base, other, exclude="extraction")
        self.assertEqual(primary["difference"], -.5)
        self.assertEqual(sensitivity["difference"], -1)
        self.assertTrue(sensitivity["post_hoc"])
        self.assertFalse(sensitivity["all_180_cases_completed"])

    def test_committed_evidence_reproduces_historical_paired_results(self):
        root = ROOT / "results/series2"
        actual = summarize(root)
        source = actual["source_reference_on_same_development_subset"]
        self.assertEqual(source["all"]["cases"], 30)
        self.assertEqual(source["image"]["answer_tokens"], 15)
        self.assertAlmostEqual(source["all"]["nll_per_token"], 0.6590592656816755)
        frozen = json.loads((root / "MAC_COMPARISON.json").read_text())
        by_name = {row["candidate"]: row for row in frozen["comparisons"]}
        for row in actual["paired"]:
            old = by_name[row["candidate"]]
            self.assertEqual(row["paired_cases"], old["paired_cases"])
            self.assertAlmostEqual(row["difference"], old["paired_accuracy_difference"])
            for observed, expected in zip(row["bootstrap_interval_95"], old["bootstrap_interval_95"]):
                self.assertAlmostEqual(observed, expected)


class VerificationTests(unittest.TestCase):
    def make_fixture(self, root):
        payload = struct.pack("<2f", 1, 2)
        header = {"weight": {"dtype": "F32", "shape": [2], "data_offsets": [0, 8]}}
        raw = json.dumps(header).encode()
        shard = root / "model-00001-of-00001.safetensors"
        shard.write_bytes(struct.pack("<Q", len(raw)) + raw + payload)
        signatures = {"weight": {"dtype": "mlx.core.float32", "shape": [2],
                                   "sha256": hashlib.sha256(payload).hexdigest()}}
        (root / "CONVERSION_REPORT.json").write_text(json.dumps({"tensor_signatures": signatures}))
        return shard

    def test_valid_then_corrupted_tensor(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            shard = self.make_fixture(root)
            self.assertEqual(verify_model(root)["verified_tensors"], 1)
            data = bytearray(shard.read_bytes())
            data[-1] ^= 1
            shard.write_bytes(data)
            with self.assertRaisesRegex(ValueError, "hash mismatch"):
                verify_model(root)

    def test_truncated_tensor_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            shard = self.make_fixture(root)
            shard.write_bytes(shard.read_bytes()[:-1])
            with self.assertRaisesRegex(ValueError, "range"):
                verify_model(root)

    def test_published_frozen_code_hashes(self):
        self.assertEqual(verify_frozen(ROOT)["status"], "passed")


class AnswerParsingTests(unittest.TestCase):
    def test_reasoning_and_unclosed_final_are_not_complete(self):
        self.assertEqual(chat.final_answer("reasoning only"), ("", False))
        self.assertEqual(chat.final_answer("<|content_text|>42"), ("42", False))
        self.assertEqual(chat.final_answer("reasoning<|content_text|>42<|end_message|>"), ("42", True))


if __name__ == "__main__":
    unittest.main(verbosity=2)
