from dataclasses import asdict
from pathlib import Path
import inspect
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import torch
from bigbrane import fitting
from bigbrane.artifacts import ROOT, freeze_sources, hash_file, read_json, verify_inventory, verify_sources, write_json
from bigbrane.decisions import advantage, bootstrap_interval, matched_report, stress_report, target_crossing
from bigbrane.evaluation import benchmark, predictor, summaries
from bigbrane.fitting import LearnerData, fit_candidate, load_checkpoint
from bigbrane.protocol import FitSpec, grid, selection_key
from bigbrane.runner import aggregate_cost, execute_candidate, freeze_selections, prepare_stage


torch.set_num_threads(1)


class HarnessTests(unittest.TestCase):
    def setUp(self):
        (ROOT/'results').mkdir(exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(prefix='unit-', dir=ROOT/'results')
        self.base = Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def data(self):
        r = np.random.default_rng(17)
        return LearnerData(r.integers(0, 8, (3, 4)), r.integers(0, 8, (2, 4)), {'train': 'fixture', 'validation': 'fixture'})

    def test_grid_and_selection_ties(self):
        matched = grid(20260930, 0); stress = grid(20260930, 1)
        self.assertEqual(len(matched), 384); self.assertEqual(len(stress), 192)
        self.assertEqual(len(grid(20260931, 0, True)), 3)
        self.assertEqual(len(set(s.key for s in matched)), 384)
        self.assertEqual(set(s.size_index for s in stress), {1, 3})
        records = [{'validation_nll': 1., 'checkpoint': 2, 'candidate': 0},
                   {'validation_nll': 1., 'checkpoint': 1, 'candidate': 1},
                   {'validation_nll': 1., 'checkpoint': 1, 'candidate': 0}]
        self.assertEqual(min(records, key=selection_key), records[2])
        self.assertEqual(FitSpec(1, 1, 1, 2, 3, 'hmm', 1, 1).seed(0), (1, 3, 1, 1, 2, 3, 0, 1, 1, 0))

    def test_exact_resume_reproducibility_and_accounting(self):
        for family in ('hmm', 'rnn', 'attention'):
            spec = FitSpec(20260931, 0, 0, 0, 0, family, 0, 0)
            data = self.data(); uninterrupted = self.base/f'{family}-a'; resumed = self.base/f'{family}-b'
            a = fit_candidate(spec, data, uninterrupted, cap=3)
            with self.assertRaises(InterruptedError):
                fit_candidate(spec, data, resumed, cap=3, interrupt_after=1)
            b = fit_candidate(spec, data, resumed, cap=3)
            for key in ('checkpoint', 'validation_nll', 'iterations_or_epochs', 'optimizer_steps', 'token_exposures', 'validation_evaluations'):
                self.assertEqual(a[key], b[key])
            self.assertEqual(read_json(uninterrupted/'trace.json'), read_json(resumed/'trace.json'))
            aa = load_checkpoint(uninterrupted/a['checkpoint_file']); bb = load_checkpoint(resumed/b['checkpoint_file'])
            if family == 'hmm':
                for key in ('pi', 'A', 'B'):
                    torch.testing.assert_close(aa[key], bb[key], rtol=0, atol=0)
                self.assertEqual(a['parameter_count'], 52); self.assertEqual(a['free_parameters'], 43)
                self.assertEqual(a['optimizer_tensor_bytes'], 0)
            else:
                for key, tensor in aa['model'].items():
                    torch.testing.assert_close(tensor, bb['model'][key], rtol=0, atol=0)
                self.assertGreater(a['optimizer_tensor_bytes'], a['parameter_tensor_bytes'])
                self.assertEqual(a['optimizer_steps'], 3)
            self.assertEqual(a['token_exposures'], 3*data.train.size)
            self.assertEqual(a['validation_evaluations'], 4)
            self.assertEqual(a['training_diagnostics_token_passes'], 4*data.train.size)
            self.assertGreaterEqual(a['peak_process_rss_bytes'], a['library_overhead_rss_bytes'])
            for timing in ('training_elapsed_s', 'validation_elapsed_s', 'checkpoint_io_s'):
                self.assertGreater(a[timing], 0)
            verify_inventory(uninterrupted, a['artifact_hashes'])
            # A complete candidate is returned unchanged rather than retrained.
            self.assertEqual(a, fit_candidate(spec, data, uninterrupted, cap=3))
            pred, _ = predictor(spec, uninterrupted/a['checkpoint_file'])
            log_probs = pred(data.validation)
            np.testing.assert_allclose(np.exp(log_probs).sum(-1), 1, atol=1e-10)
            self.assertAlmostEqual(-np.take_along_axis(log_probs, data.validation[..., None], -1).mean(), a['validation_nll'], places=12)
            costs = aggregate_cost([a, b]); self.assertEqual(costs['candidate_fits'], 2)
            self.assertEqual(costs['token_exposures'], 2*a['token_exposures'])

    def test_raw_best_includes_initial_and_small_improvements(self):
        spec = FitSpec(20260931, 0, 0, 0, 0, 'rnn', 0, 0)
        # Each checkpoint has training and validation diagnostics, in that order.
        scores = iter([2., 1., 2., 1.0001, 2., .9999999])
        with patch('bigbrane.fitting.sampled_nll', side_effect=lambda *_: next(scores)):
            record = fit_candidate(spec, self.data(), self.base/'small', cap=2)
        self.assertEqual(record['checkpoint'], 2)
        scores = iter([2., 1., 2., 1.1])
        with patch('bigbrane.fitting.sampled_nll', side_effect=lambda *_: next(scores)):
            record = fit_candidate(spec, self.data(), self.base/'initial', cap=1)
        self.assertEqual(record['checkpoint'], 0)
        state = load_checkpoint(self.base/'initial'/record['checkpoint_file'])
        self.assertEqual(state['index'], 0)

    def test_stopping_progress_is_separate_from_raw_best(self):
        spec = FitSpec(20260931, 0, 0, 0, 0, 'rnn', 0, 0)
        values = []
        for e in range(81):
            values.extend([2., 1.-e*1e-7])
        scores = iter(values)
        with patch('bigbrane.fitting.sampled_nll', side_effect=lambda *_: next(scores)):
            record = fit_candidate(spec, self.data(), self.base/'stop')
        self.assertEqual(record['iterations_or_epochs'], 20)
        self.assertEqual(record['checkpoint'], 20)
        self.assertEqual(record['validation_evaluations'], 21)

    def test_learner_artifact_access_and_counts(self):
        directory = self.base/'learner'; directory.mkdir()
        np.save(directory/'train.npy', np.zeros((256, 64), dtype=np.int64))
        np.save(directory/'validation.npy', np.ones((32, 64), dtype=np.int64))
        original = np.load; accessed = []
        def tracked(path, **kwargs):
            accessed.append(Path(path).name)
            return original(path, **kwargs)
        with patch('bigbrane.fitting.np.load', side_effect=tracked):
            data = LearnerData.load(directory, 1)
        self.assertEqual(accessed, ['train.npy', 'validation.npy'])
        self.assertEqual(data.train.size, 2048); self.assertEqual(data.validation.size, 2048)
        self.assertFalse(data.train.flags.writeable)
        source = inspect.getsource(fitting)
        self.assertNotIn('stress_filter', source); self.assertNotIn('oracle', source)
        self.assertNotIn("/'evaluator'", source)

    def test_frozen_selection_and_hashes(self):
        specs = [FitSpec(20260931, 0, 0, 0, 0, 'hmm', 0, c) for c in (0, 1)]
        records = [fit_candidate(s, self.data(), self.base/'candidates'/s.key, cap=1) for s in specs]
        selected = freeze_selections(self.base, 0, specs, records)
        self.assertEqual(len(selected), 1)
        winner = min(records, key=selection_key)
        self.assertEqual(selected[0]['spec']['candidate'], winner['candidate'])
        self.assertEqual(selected[0]['selection_cost']['candidate_fits'], 2)
        self.assertEqual(selected[0]['retained_candidate_cost']['candidate_fits'], 1)
        self.assertEqual(selected, freeze_selections(self.base, 0, specs, records))
        inventory = freeze_sources(self.base); verify_sources(self.base, inventory)
        copied = self.base/'frozen'/'EXPERIMENT.md'; copied.write_text('changed')
        with self.assertRaises(ValueError):
            verify_sources(self.base, inventory)
        (self.base/'candidates'/specs[0].key/'trace.json').write_text('changed')
        with self.assertRaises(ValueError):
            fit_candidate(specs[0], self.data(), self.base/'candidates'/specs[0].key, cap=1)

    def test_metrics_benchmarks_and_decisions(self):
        tokens = np.array([[0, 1, 0, 1], [1, 0, 1, 0]])
        oracle = np.broadcast_to([.3, .7], (2, 4, 2))
        metrics = summaries(oracle, np.log(oracle), tokens)
        self.assertEqual(metrics['all']['kl'], 0.)
        self.assertEqual(len(metrics['per_sequence']), 2)
        calls = []
        result = benchmark(lambda x, batch: calls.append((x.shape, batch)), tokens, 1)
        self.assertEqual(len(calls), 12); self.assertEqual(len(result['elapsed_s']), 10)
        self.assertEqual(result['scored_tokens'], 8)
        self.assertTrue(advantage(np.full((2, 4), .03))['passed'])
        self.assertFalse(advantage(np.full((2, 4), .019))['passed'])
        self.assertTrue(advantage(np.full((2, 4), -.03))['directional_claim_contradicted'])
        self.assertEqual(bootstrap_interval(np.arange(8)/100), bootstrap_interval(np.arange(8)/100))
        self.assertEqual(target_crossing([.04, .06, .03, .02])['tokens'], 8192)
        self.assertEqual(target_crossing([.03]*4)['display'], '<=512')
        self.assertEqual(target_crossing([.06]*4)['display'], '>16384')
        evaluations = []
        for s in grid(20260930, 0)+grid(20260930, 1):
            if s.candidate != 0:
                continue
            error = .01 if s.family == 'hmm' else (.05 if s.stage == 0 else .001)
            evaluations.append({'selection': {'spec': asdict(s)}, 'metrics': {'all': {'kl': error}}})
        matched = [r for r in evaluations if r['selection']['spec']['stage'] == 0]
        stress = [r for r in evaluations if r['selection']['spec']['stage'] == 1]
        report = matched_report(matched)
        self.assertTrue(report['matched_advantage_passed']); self.assertEqual(report['stress_stage'], 'triggered')
        for result in stress_report(matched, stress):
            self.assertEqual(result['controls']['rnn']['label'], 'reversed')

    def test_learning_rate_pair_initial_tensors(self):
        for family in ('rnn', 'attention'):
            specs = [FitSpec(20260931, 0, 0, 0, 0, family, 1, c) for c in (0, 1)]
            states = []
            for spec in specs:
                directory = self.base/spec.key
                fit_candidate(spec, self.data(), directory, cap=0)
                states.append(load_checkpoint(directory/'checkpoint-000.pt')['model'])
            for name, value in states[0].items():
                torch.testing.assert_close(value, states[1][name], atol=0, rtol=0)

    def test_saved_data_splits_and_candidate_subprocess(self):
        stage = prepare_stage(self.base, 20260931, 0, True)
        verify_inventory(self.base, stage['hashes'])
        base = self.base/'data'/'s0-r0-g0'
        for directory, name, count in (('learner', 'train', 256), ('learner', 'validation', 32), ('evaluator', 'test', 128)):
            data = np.load(base/directory/f'{name}.npy')
            self.assertEqual(data.shape, (count, 64))
        self.assertEqual(stage, prepare_stage(self.base, 20260931, 0, True))
        stress = prepare_stage(self.base, 20260931, 1, True)
        self.assertNotEqual(stage['hashes'], stress['hashes'])
        # Exercise a real fresh worker, rather than an in-process fit, with unchanged protocol caps.
        spec = FitSpec(20260931, 0, 0, 0, 0, 'hmm', 0, 0)
        manifest = {'attempts': [], 'completed_candidates': []}
        record = execute_candidate(self.base, manifest, spec)
        import os
        self.assertNotEqual(record['worker_pid'], os.getpid())
        self.assertEqual(manifest['completed_candidates'], [spec.key])
        self.assertEqual(manifest['attempts'][0]['status'], 'complete')
        self.assertGreater(manifest['attempts'][0]['elapsed_s'], 0)
        self.assertEqual(record['training_tokens'], 512)
        self.assertEqual(record['total_accessible_tokens'], 2560)
        job = read_json(self.base/'candidates'/spec.key/'job.json')
        self.assertEqual(set(job), {'spec', 'learner_directory', 'data_hashes', 'candidate_directory'})
        self.assertNotIn('evaluator', job['learner_directory'])
        # Recover the driver bookkeeping if a worker finished just before driver interruption.
        manifest['completed_candidates'] = []
        manifest['attempts'][0]['status'] = 'running'
        self.assertEqual(record, execute_candidate(self.base, manifest, spec))
        self.assertEqual(manifest['completed_candidates'], [spec.key])
        self.assertEqual(manifest['attempts'][0]['status'], 'complete-recovered')


if __name__ == '__main__':
    unittest.main()
