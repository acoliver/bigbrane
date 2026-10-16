from dataclasses import asdict
from pathlib import Path
import os
import signal
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import patch
import numpy as np
import torch
from bigbrane.artifacts import ROOT, freeze_sources, hash_file, read_json, write_json
from bigbrane.fitting import LearnerData, fit_candidate, load_checkpoint
from bigbrane.protocol import FitSpec
from bigbrane.runner import execute_candidate, freeze_selections, prepare_stage, run, verify_boundary
from bigbrane.evaluation import evaluate_selected


class IntegrityTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='integrity-', dir=ROOT/'results')
        self.out = Path(self.temp.name)
        self.spec = FitSpec(20260931, 0, 0, 0, 0, 'hmm', 0, 0)

    def tearDown(self):
        self.temp.cleanup()

    def fixture(self):
        stage = prepare_stage(self.out, 20260931, 0, True)
        manifest = {'source_hashes': freeze_sources(self.out), 'data_stages': {'0': stage},
                    'attempts': [], 'completed_candidates': [], 'candidate_commits': {}}
        write_json(self.out/'manifest.json', manifest)
        return manifest

    def tiny(self):
        return LearnerData(np.zeros((3, 1), dtype=np.int64), np.zeros((2, 1), dtype=np.int64),
                           {'train': 'fixture', 'validation': 'fixture'})

    def test_changed_earlier_best_checkpoint_rejected_on_resume(self):
        spec = FitSpec(20260931, 0, 0, 0, 0, 'rnn', 0, 0)
        directory = self.out/'candidates'/spec.key
        scores = iter([2., 1., 2., 2.])
        with patch('bigbrane.fitting.sampled_nll', side_effect=lambda *_: next(scores)):
            with self.assertRaises(InterruptedError):
                fit_candidate(spec, self.tiny(), directory, cap=2, interrupt_after=1)
        state = load_checkpoint(directory/'checkpoint-000.pt')
        state['model']['by'][0] += 5
        torch.save(state, directory/'checkpoint-000.pt')
        with self.assertRaisesRegex(ValueError, 'hash mismatch'):
            fit_candidate(spec, self.tiny(), directory, cap=2)
        self.assertFalse((directory/'record.json').exists())

    def test_completed_record_mutations_fail_before_selection(self):
        directory = self.out/'candidates'/self.spec.key
        record = fit_candidate(self.spec, self.tiny(), directory, cap=1)
        original = (directory/'record.json').read_bytes()
        for field, value in [('validation_nll', -100.), ('checkpoint', 0), ('candidate', 1),
                             ('checkpoint_file', 'checkpoint-999.pt'), ('root', 1)]:
            with self.subTest(field=field):
                changed = dict(record)
                changed[field] = value if record[field] != value else 999
                write_json(directory/'record.json', changed)
                with self.assertRaises(ValueError):
                    freeze_selections(self.out, 0, [self.spec], [changed])
                self.assertFalse((self.out/'selections-stage-0.json').exists())
                (directory/'record.json').write_bytes(original)
        write_json(directory/'identity.json', {'spec': asdict(self.spec), 'data_hashes': {}})
        with self.assertRaises(ValueError):
            freeze_selections(self.out, 0, [self.spec], [record])

    def test_completion_commit_interruption_recovers_from_trusted_state(self):
        directory = self.out/'candidates'/self.spec.key
        def interrupted(path, value):
            if Path(path).name == 'completion.json':
                raise InterruptedError('before completion pointer')
            write_json(path, value)
        with patch('bigbrane.fitting.write_json', side_effect=interrupted):
            with self.assertRaises(InterruptedError):
                fit_candidate(self.spec, self.tiny(), directory, cap=1)
        state = load_checkpoint(directory/'checkpoint-001.pt')
        record = read_json(directory/'record.json')
        write_json(directory/'record.json', dict(record, validation_nll=-100.))
        recovered = fit_candidate(self.spec, self.tiny(), directory, cap=1)
        self.assertEqual(recovered['validation_nll'], state['best_score'])
        self.assertEqual(recovered['checkpoint'], state['best_index'])
        self.assertEqual(recovered['iterations_or_epochs'], 1)

    def test_completed_record_is_bound_in_driver_before_reuse(self):
        manifest = self.fixture()
        record = execute_candidate(self.out, manifest, self.spec)
        directory = self.out/'candidates'/self.spec.key
        owner = read_json(directory/'worker-owner.json')
        self.assertEqual(owner['source_directory'], str(self.out/'frozen'))
        changed = dict(record, validation_nll=-100.)
        write_json(directory/'record.json', changed)
        with self.assertRaises(ValueError):
            execute_candidate(self.out, manifest, self.spec)
        self.assertIn(self.spec.key, manifest['candidate_commits'])

    def test_dispatch_rejects_data_and_frozen_source_drift(self):
        manifest = self.fixture()
        paths = [self.out/'data'/'s0-r0-g0'/'learner'/f'{name}.npy'
                 for name in ('train', 'validation')]
        paths.append(self.out/'frozen'/'bigbrane'/'core.py')
        for path in paths:
            with self.subTest(path=str(path)):
                original = path.read_bytes(); path.write_bytes(original+b'changed')
                with patch('bigbrane.runner.subprocess.Popen') as launch:
                    with self.assertRaises(ValueError):
                        execute_candidate(self.out, manifest, self.spec)
                    launch.assert_not_called()
                path.write_bytes(original)
        with patch('bigbrane.artifacts.source_files', return_value=[]):
            with self.assertRaises(ValueError):
                execute_candidate(self.out, manifest, self.spec)

    def test_selection_evaluation_and_finalization_reject_drift(self):
        manifest = self.fixture()
        record = execute_candidate(self.out, manifest, self.spec)
        selections = freeze_selections(self.out, 0, [self.spec], [record], manifest=manifest)
        manifest['selections'] = {'0': {'path': 'selections-stage-0.json',
                                       'hash': hash_file(self.out/'selections-stage-0.json')}}
        paths = [self.out/'data'/'s0-r0-g0'/'evaluator'/name
                 for name in ('test.npy', 'generator.npz')]
        paths += [self.out/'data'/'s0-r0-g0'/'learner'/'validation.npy',
                  self.out/'frozen'/'bigbrane'/'evaluation.py']
        for path in paths:
            with self.subTest(path=str(path)):
                original = path.read_bytes(); path.write_bytes(original+b'changed')
                with self.assertRaises(ValueError):
                    freeze_selections(self.out, 0, [self.spec], [record], manifest=manifest)
                with patch('bigbrane.evaluation.predictor') as predict:
                    with self.assertRaises(ValueError):
                        evaluate_selected(self.out, selections[0], manifest=manifest)
                    predict.assert_not_called()
                with self.assertRaises(ValueError):
                    verify_boundary(self.out, manifest)
                path.write_bytes(original)

    def test_run_finalization_rejects_source_drift_after_last_evaluation(self):
        manifest = self.fixture()
        manifest.update({'root': 20260931, 'pilot': True, 'status': 'frozen', 'scientific_limitations': []})
        records = []
        evaluations = []
        for family in ('hmm', 'rnn', 'attention'):
            spec = FitSpec(20260931, 0, 0, 0, 0, family, 0, 0)
            records.append(fit_candidate(spec, self.tiny(), self.out/'candidates'/spec.key, cap=0))
            evaluations.append({'selection': {'spec': asdict(spec)}, 'metrics': {'all': {}}})
        def evaluated(*_):
            (self.out/'frozen'/'bigbrane'/'evaluation.py').write_text('changed after evaluation')
            return evaluations, records
        with patch('bigbrane.runner.initialize', return_value=(self.out, manifest)):
            with patch('bigbrane.runner.run_stage', side_effect=evaluated):
                with self.assertRaisesRegex(ValueError, 'frozen source changed'):
                    run(20260931, True, self.out, resume=True)
        self.assertFalse((self.out/'report.json').exists())
        self.assertNotEqual(read_json(self.out/'manifest.json').get('status'), 'complete')

    def test_live_source_content_drift_rejected_at_dispatch(self):
        manifest = self.fixture()
        live = self.out/'live-tree'
        for name in manifest['source_hashes']:
            target = live/name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes((ROOT/name).read_bytes())
        (live/'bigbrane'/'core.py').write_text('changed source')
        with patch('bigbrane.artifacts.ROOT', live):
            with patch('bigbrane.runner.subprocess.Popen') as launch:
                with self.assertRaises(ValueError):
                    execute_candidate(self.out, manifest, self.spec)
                launch.assert_not_called()

    def test_manifest_completion_digest_cannot_be_replaced(self):
        manifest = self.fixture()
        record = execute_candidate(self.out, manifest, self.spec)
        directory = self.out/'candidates'/self.spec.key
        write_json(directory/'record.json', dict(record, validation_nll=-100.))
        completion = read_json(directory/'completion.json')
        completion['hashes']['record.json'] = hash_file(directory/'record.json')
        write_json(directory/'completion.json', completion)
        with self.assertRaisesRegex(ValueError, 'hash mismatch'):
            execute_candidate(self.out, manifest, self.spec)

    def test_scaling_drift_rejected_before_evaluation_access(self):
        manifest = self.fixture()
        path = self.out/'scaling'/'r0-g0-l16.npy'
        path.parent.mkdir()
        np.save(path, np.zeros((32, 16), dtype=np.int64))
        manifest['data_stages']['0']['hashes'][str(path.relative_to(self.out))] = hash_file(path)
        write_json(self.out/'data-stage-0.json', manifest['data_stages']['0'])
        record = execute_candidate(self.out, manifest, self.spec)
        selection = freeze_selections(self.out, 0, [self.spec], [record], manifest=manifest)[0]
        manifest['selections'] = {'0': {'path': 'selections-stage-0.json',
                                       'hash': hash_file(self.out/'selections-stage-0.json')}}
        path.write_bytes(path.read_bytes()+b'changed')
        with patch('bigbrane.evaluation.predictor') as predict:
            with self.assertRaises(ValueError):
                evaluate_selected(self.out, selection, scaling=True, manifest=manifest)
            predict.assert_not_called()

    def wait_for(self, predicate):
        deadline = time.monotonic()+40
        while time.monotonic() < deadline:
            if predicate():
                return
            time.sleep(.05)
        self.fail('process fixture did not reach its synchronization point')

    def test_sigkill_driver_surviving_worker_and_simultaneous_driver(self):
        manifest = self.fixture()
        gate = self.out/'worker-ready'
        wrapper = self.out/'paused_worker.py'
        wrapper.write_text("import os, signal, sys\n"
                           f"sys.path.insert(0, {str(self.out/'frozen')!r})\n"
                           "from bigbrane import cli, fitting\n"
                           "original = fitting.persist\n"
                           "def persist(directory, state, index):\n"
                           "    result = original(directory, state, index)\n"
                           "    if index == 0:\n"
                           f"        with open({str(gate)!r}, 'w') as stream: stream.write(str(os.getpid()))\n"
                           "        os.kill(os.getpid(), signal.SIGSTOP)\n"
                           "    return result\n"
                           "fitting.persist = persist\n"
                           "sys.argv = ['bigbrane.cli', '--worker', sys.argv[1]]\n"
                           "cli.main()\n")
        driver = self.out/'driver.py'
        driver.write_text("from pathlib import Path\nimport subprocess\n"
                          "from bigbrane.runner import execute_candidate\n"
                          "from bigbrane.artifacts import read_json\nfrom bigbrane.protocol import FitSpec\n"
                          "original = subprocess.Popen\n"
                          "def launch(command, **kwargs):\n"
                          f"    return original([command[0], {str(wrapper)!r}, command[-1]], **kwargs)\n"
                          "subprocess.Popen = launch\n"
                          f"out = Path({str(self.out)!r})\n"
                          f"execute_candidate(out, read_json(out/'manifest.json'), FitSpec(**{asdict(self.spec)!r}))\n")
        env = dict(os.environ, PYTHONPATH=str(ROOT))
        log = (self.out/'driver.log').open('w')
        process = subprocess.Popen([sys.executable, str(driver)], cwd=ROOT, env=env, stdout=log, stderr=log)
        worker_pid = None
        try:
            self.wait_for(gate.exists)
            worker_pid = int(gate.read_text())
            self.wait_for(lambda: len(read_json(self.out/'manifest.json')['attempts']) == 1
                          and 'worker_identity' in read_json(self.out/'manifest.json')['attempts'][0])
            current = read_json(self.out/'manifest.json')
            competing = subprocess.run([sys.executable, '-m', 'bigbrane.cli', '--pilot', '--root', '20260931',
                                        '--output', str(self.out), '--resume'], cwd=ROOT, env=env,
                                       capture_output=True, text=True, timeout=20)
            self.assertNotEqual(competing.returncode, 0)
            self.assertIn('owned', competing.stderr)
            with self.assertRaisesRegex(RuntimeError, 'owned'):
                execute_candidate(self.out, current, self.spec)
            process.kill(); process.wait(timeout=10)
            resume = subprocess.run([sys.executable, '-m', 'bigbrane.cli', '--pilot', '--root', '20260931',
                                     '--output', str(self.out), '--resume'], cwd=ROOT, env=env,
                                    capture_output=True, text=True, timeout=20)
            self.assertNotEqual(resume.returncode, 0)
            self.assertIn('owned', resume.stderr)
            for family in ('hmm', 'rnn'):
                spec = FitSpec(20260931, 0, 0, 0, 0, family, 0, 0)
                with self.assertRaisesRegex(RuntimeError, 'owned'):
                    execute_candidate(self.out, read_json(self.out/'manifest.json'), spec)
            self.assertEqual(len(read_json(self.out/'manifest.json')['attempts']), 1)
            self.assertFalse((self.out/'candidates'/self.spec.key/'record.json').exists())
            os.kill(worker_pid, signal.SIGCONT)
            completion = self.out/'candidates'/self.spec.key/'completion.json'
            self.wait_for(completion.exists)
            def released():
                try:
                    execute_candidate(self.out, read_json(self.out/'manifest.json'), self.spec)
                except RuntimeError:
                    return False
                return True
            self.wait_for(released)
            final = read_json(self.out/'manifest.json')
            self.assertEqual(len(final['attempts']), 1)
            self.assertEqual(final['attempts'][0]['status'], 'complete-recovered')
            self.assertIsNone(final['attempts'][0]['elapsed_s'])
            owner = read_json(self.out/'candidates'/self.spec.key/'worker-owner.json')
            self.assertEqual(owner['token'], final['attempts'][0]['worker_identity']['token'])
            self.assertEqual(owner['pid'], worker_pid)
        finally:
            if process.poll() is None:
                process.kill(); process.wait(timeout=10)
            if worker_pid is not None and not (self.out/'candidates'/self.spec.key/'completion.json').exists():
                os.kill(worker_pid, signal.SIGKILL)
            log.close()


if __name__ == '__main__':
    unittest.main()
