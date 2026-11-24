"""Small protocol checks; no MLX, model loading, or GPU access."""
import unittest
from protocol import make_splits, CATEGORIES
from metrics import score_trace, summarize, correct
from reference_gate import mlx_name


class ProtocolTests(unittest.TestCase):
    def test_reduced_prefixes_have_the_same_category_balance(self):
        splits = make_splits()
        for rows in splits.values():
            for size in range(6, len(rows)+1, 6):
                counts = [sum(row['category'] == category for row in rows[:size]) for category in CATEGORIES]
                self.assertEqual(len(set(counts)), 1)

    def test_completion_at_larger_budget_does_not_repair_earlier_score(self):
        case = {'id': 'x', 'category': 'logic', 'answer': '42'}
        trace = {'tokens_emitted': 301, 'final_start_token': 299, 'final_end_token': 301,
                 'final_answer': '42', 'stop_reason': 'eos'}
        row = score_trace(case, trace)
        self.assertFalse(row['at_budget']['192']['correct'])
        self.assertTrue(row['at_budget']['512']['correct'])
        self.assertFalse(row['right_censored'])

    def test_incomplete_reasoning_is_not_a_correct_final(self):
        case = {'id': 'x', 'category': 'logic', 'answer': '42'}
        row = score_trace(case, {'tokens_emitted': 192, 'final_answer': '42', 'stop_reason': 'token_cap'})
        self.assertFalse(row['correct_final'])
        self.assertTrue(row['right_censored'])
        self.assertIsNone(row['at_budget']['512']['completed'])
        self.assertFalse(row['at_budget']['512']['observed'])

    def test_early_eos_without_final_is_a_failure_at_all_budgets(self):
        case = {'id': 'x', 'category': 'logic', 'answer': '42'}
        row = score_trace(case, {'tokens_emitted': 8, 'stop_reason': 'eos'})
        self.assertFalse(row['at_budget']['512']['completed'])
        self.assertTrue(row['at_budget']['512']['observed'])

    def test_structured_answers_keep_json_types_and_reject_duplicate_fields(self):
        case = {'category': 'structured', 'answer': '{"count":1,"code":"x"}'}
        self.assertTrue(correct(case, '{"code":"x", "count":1}'))
        for answer in ('{"count":true,"code":"x"}', '{"count":1.0,"code":"x"}',
                       '{"count":0,"count":1,"code":"x"}'):
            self.assertFalse(correct(case, answer))

    def test_reference_adapter_handles_nontrivial_parameter_layouts(self):
        self.assertEqual(mlx_name('model.embed_tokens.embed_norm.weight'),
                         'model.language_model.embed_norm.weight')
        self.assertEqual(mlx_name('model.layers.5.self_attn.o_proj.gate_proj.weight'),
                         'model.language_model.layers.5.self_attn.gate_proj.weight')
        self.assertEqual(mlx_name('model.layers.5.mlp.experts.down_proj'),
                         'model.language_model.layers.5.mlp.experts.down_proj.weight')

    def test_errors_are_separate_from_token_censoring(self):
        case = {'id': 'x', 'category': 'logic', 'answer': '42'}
        row = score_trace(case, {'tokens_emitted': 5, 'stop_reason': 'runtime_error'})
        self.assertFalse(row['right_censored'])
        self.assertEqual(summarize([row])['errors_and_stops']['runtime_error'], 1)

    def test_invalid_offsets_and_duplicate_cases_are_rejected(self):
        case = {'id': 'x', 'category': 'logic', 'answer': '42'}
        with self.assertRaises(ValueError):
            score_trace(case, {'tokens_emitted': 2, 'final_start_token': 1, 'final_end_token': 3})
        row = score_trace(case, {'tokens_emitted': 2, 'stop_reason': 'token_cap'})
        with self.assertRaises(ValueError):
            summarize([row, row])


if __name__ == '__main__':
    unittest.main(verbosity=2)
