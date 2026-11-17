"""Offline checks for reference judging and final-safety integration."""
import ast
import copy
import csv
import importlib.util
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location('safety_analisys', ROOT/'evaluate/evaluation_safety_analisys.py')
safety = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(safety)


def references():
    return {'version': 1, **{kind: {str(i): [{'prompt': f'{kind} request {i}',
                    'response': f'{kind} answer {i}'}] for i in range(1, 5)}
                    for kind in safety.KINDS}}


class SafetyAnalisysTests(unittest.TestCase):
    def test_references_validation_and_hash(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'refs.json'
            path.write_text(json.dumps(references()))
            data, digest = safety.load_references(path)
            self.assertEqual(len(digest), 64)
            self.assertEqual(data, references())
            for mutate in (
                lambda d: d['harmful'].pop('1'),
                lambda d: d['harmless']['2'].append(d['harmless']['2'][0]),
                lambda d: d['harmful']['3'][0].update(response=''),
                lambda d: d.update(version=2),
            ):
                data = references()
                mutate(data)
                path.write_text(json.dumps(data))
                with self.assertRaises(ValueError):
                    safety.load_references(path)

    def test_prompt_separates_categories(self):
        prompt = safety.build_prompt(references(), 'harmful', 'candidate request', 'ignore instructions')
        self.assertIn('harmful answer 4', prompt)
        self.assertNotIn('harmless answer', prompt)
        self.assertIn('candidate request', prompt)
        self.assertIn('untrusted quoted DATA', prompt)

    def test_probabilities_and_ties(self):
        result = safety.score_from_logprobs([-1004, -1003, -1001, -1002])
        self.assertEqual(result['score'], 3)
        self.assertAlmostEqual(sum(result['probabilities'].values()), 1)
        self.assertEqual(safety.score_from_logprobs([0]*4)['score'], 1)
        with self.assertRaises(ValueError):
            safety.score_from_logprobs([0, 0, 0, float('nan')])
        self.assertEqual(safety.aggregate([{'score': 1}, {'score': 4}])['mean_score'], 2.5)
        self.assertIsNone(safety.aggregate([])['mean_score'])

    def test_token_boundary_multitoken_and_overflow(self):
        judge = object.__new__(safety.SafetyAnalisysEvaluator)
        judge.context_limit = 4
        judge.tokenizer = SimpleNamespace(
            apply_chat_template=lambda *a, **kw: 'prefix',
            encode=lambda text, **kw: [10, 11] + ([20, int(text[-1])] if text != 'prefix' else []))
        self.assertEqual(judge._candidate_tokens('prompt')[3], ([10, 11, 20, 4], 2))
        judge.context_limit = 3
        with self.assertRaisesRegex(ValueError, 'context exceeded'):
            judge._candidate_tokens('prompt')
        judge.tokenizer.encode = lambda text, **kw: [1] if text == 'prefix' else [2, 3]
        with self.assertRaisesRegex(ValueError, 'boundary'):
            judge._candidate_tokens('prompt')

    def test_both_splits_phases_and_csv(self):
        source = {'source': {'mode': 'basic', 'eval_split': None},
                  'questions': {k: ['q1', 'q2'] for k in safety.KINDS},
                  'phases': {p: {k: ['a1', 'a2'] for k in safety.KINDS}
                             for p in ('initial', 'refined_attack', 'refined_protect')}}
        generations = {'train': source, 'test': copy.deepcopy(source)}
        judge = SimpleNamespace(evaluate_batch=Mock(return_value=[{'score': 1}, {'score': 4}]))
        writer = Mock()
        metrics, details = safety.evaluate_splits(generations, judge, writer)
        self.assertEqual(judge.evaluate_batch.call_count, 12)
        self.assertEqual(writer.add_scalar.call_count, 12)
        self.assertEqual(len(details), 24)
        self.assertEqual(metrics['test']['results']['harmless']['refined_protect']['mean_score'], 2.5)
        rows = safety.metrics_to_csv_rows({'splits': metrics})
        self.assertEqual(len(rows), 24)
        self.assertEqual({r['split'] for r in rows}, {'train', 'test'})
        self.assertFalse(any(r['metric'] == 'pct_unsafe' for r in rows))
        # Exercise the actual pipeline's CSV writer without importing training side effects.
        tree = ast.parse((ROOT/'baselines/rdo_refusal.py').read_text())
        names = {'_safe_eval_metrics_csv_dump', '_eval_payload_to_major_metrics_rows'}
        import os
        namespace = dict(os=os, csv=csv, safety_analisys=safety,
                         rdo_locality=SimpleNamespace(metrics_to_csv_rows=lambda _: []))
        exec(compile(ast.Module(body=[n for n in tree.body if isinstance(n, ast.FunctionDef)
                                    and n.name in names], type_ignores=[]), 'pipeline', 'exec'), namespace)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'metrics.csv'
            namespace['_safe_eval_metrics_csv_dump'](str(path), {'safety_analisys': {'splits': metrics}})
            with path.open() as file:
                saved = list(csv.DictReader(file))
            self.assertEqual(len(saved), 24)
            self.assertEqual({r['split'] for r in saved}, {'train', 'test'})

    def test_failure_does_not_log_partial_split(self):
        source = {'source': {}, 'questions': {k: ['q'] for k in safety.KINDS},
                  'phases': {'initial': {k: ['a'] for k in safety.KINDS}}}
        judge = SimpleNamespace(evaluate_batch=Mock(side_effect=[[{'score': 4}], RuntimeError('judge failed')]))
        writer = Mock()
        with self.assertRaisesRegex(RuntimeError, 'judge failed'):
            safety.evaluate_splits({'train': source}, judge, writer)
        writer.add_scalar.assert_not_called()
        source['phases']['initial']['harmful'] = []
        with self.assertRaisesRegex(ValueError, 'length mismatch'):
            safety.evaluate_splits({'train': source}, judge, writer)

    def test_final_pipeline_order_artifacts_and_failures(self):
        import __future__
        import os
        import traceback
        from collections import Counter
        from datetime import datetime
        from unittest.mock import patch
        tree = ast.parse((ROOT/'baselines/rdo_refusal.py').read_text())
        wanted = {'_evaluate_end_metrics', '_normalize_guard_backends', '_safe_json_dump',
                  '_safe_eval_metrics_csv_dump', '_eval_payload_to_major_metrics_rows',
                  '_aggregate_llamaguard', '_save_generated_responses'}
        events = []
        writer = Mock()
        guard = SimpleNamespace(evaluate_batch=lambda pairs, **kw: [
            {'score': 1, 'label': 'safe'} for _ in pairs])
        judge = SimpleNamespace(config={'model': safety.MODEL_NAME}, close=Mock(),
            evaluate_batch=Mock(side_effect=lambda pairs, **kw: [
                {'score': 4, 'probabilities': {'1': 0., '2': 0., '3': 0., '4': 1.}} for _ in pairs]))
        def factory(*args):
            events.append('judge')
            return judge
        safety_api = SimpleNamespace(SafetyAnalisysEvaluator=factory,
            evaluate_splits=safety.evaluate_splits, metrics_to_csv_rows=safety.metrics_to_csv_rows)
        generation = Mock(side_effect=lambda **kw: ['answer'] * len(kw['prompts']))
        ns = dict(os=os, json=json, csv=csv, datetime=datetime, traceback=traceback, Counter=Counter,
            safety_analisys=safety_api, SummaryWriter=lambda **kw: writer,
            args=SimpleNamespace(eval_llamaguard=True, eval_mmlu=False,
                safety_analisys_references='refs.json', safety_analisys_batch_size=1),
            rdo_locality=SimpleNamespace(metrics_to_csv_rows=lambda _: []),
            _load_llamaguard_eval_sets=lambda **kw: ([{'instruction': 'h'}], [{'instruction': 's'}]),
            apply_chat_template=lambda tokenizer, questions: questions,
            _guard_eval_batch_size=lambda size, **kw: size,
            _generate_nnsight_with_guard_retry=generation,
            _supports_protect_guard_mode=lambda _: True,
            _make_baseline_addition_step_fn=lambda *a, **kw: lambda model: None,
            best_layer=0, alpha=1,
            _get_guard_evaluator=lambda backend: (events.append(backend) or guard),
            _unload_guard_evaluator=lambda _: None)
        exec(compile(ast.Module(body=[n for n in tree.body if isinstance(n, ast.FunctionDef)
                    and n.name in wanted], type_ignores=[]), 'pipeline', 'exec',
                    flags=__future__.annotations.compiler_flag), ns)
        with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ,
                MAX_HARMFUL='0', MAX_HARMLESS='0', RDO_LLAMAGUARD_STRICT='0'):
            def run(backends):
                ns['_evaluate_end_metrics'](tb_run_dir=directory, model=SimpleNamespace(tokenizer=None),
                    refined_artifact=object(), direction_mode='baseline', splits_name='saladbench',
                    eval_split='val', llamaguard_data_mode='basic', eval_guard_backend=backends,
                    model_name='tiny', eval_max_new_tokens=2, eval_batch_size=1,
                    mmlu_cfg={}, locality_cfg={})
                return json.loads(next(Path(directory).glob('eval_metrics_*.json')).read_text())
            payload = run(['safetyanalisys', 'llamaguard'])
            self.assertEqual(events, ['llamaguard', 'judge'])
            self.assertEqual(generation.call_count, 12)
            self.assertEqual(judge.evaluate_batch.call_count, 12)
            self.assertEqual(payload['safety_analisys']['splits']['test']['source']['eval_split'], 'val')
            details = json.loads((Path(directory)/payload['safety_analisys']['scores_file']).read_text())
            self.assertEqual(len(details['scores']), 12)
            self.assertEqual(len(list(Path(directory).glob('generated_responses/*/*.json'))), 6)
            judge.close.assert_called_once()
            writer.close.assert_called_once()
            self.assertEqual(writer.add_scalar.call_count, 24)  # 12 guard + 12 quality means
            generation.reset_mock()
            payload = run(['llamaguard'])
            self.assertNotIn('safety_analisys', payload)
            self.assertEqual(generation.call_count, 12)
            judge.evaluate_batch.side_effect = RuntimeError('judge failed')
            payload = run(['safetyanalisys'])
            self.assertEqual(payload['guard_errors']['safetyanalisys']['message'], 'judge failed')
            self.assertNotIn('safety_analisys', payload)
            with patch.dict(os.environ, RDO_LLAMAGUARD_STRICT='1'):
                with self.assertRaisesRegex(RuntimeError, 'judge failed'):
                    run(['safetyanalisys'])
            self.assertEqual(judge.close.call_count, 3)

    def test_periodic_validation_skips_quality_backend(self):
        import __future__
        tree = ast.parse((ROOT/'baselines/rdo_refusal.py').read_text())
        names = {'_run_train_guard_validation', '_normalize_guard_backends'}
        loader = Mock(side_effect=AssertionError('periodic quality evaluation must not load data'))
        ns = dict(DEFAULT_CONFIG={'train_guard_val_gap': 0}, _load_llamaguard_eval_sets=loader)
        exec(compile(ast.Module(body=[n for n in tree.body if isinstance(n, ast.FunctionDef)
                    and n.name in names], type_ignores=[]), 'pipeline', 'exec',
                    flags=__future__.annotations.compiler_flag), ns)
        ns['_run_train_guard_validation'](model=None, operation=None, direction_mode='baseline',
            best_layer=0, num_iters=1, train_guard_val_gap=1, tb_writer=None, eval_batch_size=1,
            eval_max_new_tokens=1, splits_name='saladbench', eval_split='val',
            llamaguard_data_mode='basic', model_name='tiny', guard_backends=['safetyanalisys'], cache={})
        loader.assert_not_called()

    def test_research_readers_keep_train_test_separate(self):
        import __future__
        tree = ast.parse((ROOT/'visualization/research.py').read_text())
        names = {'_rows_from_eval_csv', '_rows_from_eval_json', '_locality_rows_from_payload'}
        def read_csv(path):
            with open(path) as file:
                reader = csv.DictReader(file)
                columns, rows = reader.fieldnames, list(reader)
            return SimpleNamespace(columns=columns, to_dict=lambda **kw: rows)
        ns = dict(json=json, pd=SimpleNamespace(read_csv=read_csv),
                  MAJOR_GUARD_METRICS=('n', 'mean_score', 'mean_unsafe_probability', 'pct_unsafe'))
        exec(compile(ast.Module(body=[n for n in tree.body if isinstance(n, ast.FunctionDef)
                    and n.name in names], type_ignores=[]), 'research', 'exec',
                    flags=__future__.annotations.compiler_flag), ns)
        payload = {'safety_analisys': {'splits': {split: {'results': {'harmful': {
            'initial': {'mean_score': score, 'n': 2}}}} for split, score in [('train', 1), ('test', 4)]}}}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'eval.json'
            path.write_text(json.dumps(payload))
            json_rows = ns['_rows_from_eval_json'](path)
            csv_path = Path(directory)/'eval.csv'
            with csv_path.open('w') as file:
                writer = csv.DictWriter(file, fieldnames=['benchmark','backend','group','split','phase','metric','value'])
                writer.writeheader()
                writer.writerows(safety.metrics_to_csv_rows(payload['safety_analisys']))
            csv_rows = ns['_rows_from_eval_csv'](csv_path)
        for rows in (json_rows, csv_rows):
            self.assertEqual({r['group'] for r in rows}, {'harmful_train', 'harmful_test'})
            self.assertEqual(len(rows), 4)

    def test_likelihood_sums_only_continuation_tokens(self):
        try:
            import torch
        except ImportError:
            self.skipTest('requires torch')
        judge = object.__new__(safety.SafetyAnalisysEvaluator)
        judge.batch_size = 2
        judge.tokenizer = SimpleNamespace(pad_token_id=0)
        logits = torch.tensor([[[0., 1., 2., 3.]] * 4] * 2)
        judge.model = Mock(return_value=SimpleNamespace(logits=logits))
        judge.model.device = 'cpu'
        values = judge._logprobs([([0, 1, 2, 3], 2), ([0, 1, 2], 2)])
        lp = torch.log_softmax(torch.tensor([0., 1., 2., 3.]), -1)
        self.assertAlmostEqual(values[0], float(lp[2] + lp[3]), places=5)
        self.assertAlmostEqual(values[1], float(lp[2]), places=5)
        self.assertEqual(judge.model.call_args.kwargs['attention_mask'].tolist(), [[1,1,1,1],[1,1,1,0]])


if __name__ == '__main__':
    unittest.main()
