"""No-allocation tests for the server's scheduler and container boundaries."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


class ServerLauncherTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.path = Path(self.temporary.name).resolve()
        self.record = self.path / 'record.json'
        self.image = self.path / 'sail.sif'
        self.image.touch()
        for name in ['sbatch', 'srun', 'cards', 'apptainer', 'fake-python']:
            script = self.path / name
            script.write_text('#!/usr/bin/env python3\nimport json,os,sys\njson.dump({"cmd":os.path.basename(sys.argv[0]),"args":sys.argv[1:],"mask":os.getenv("APPTAINERENV_CUDA_VISIBLE_DEVICES")},open(os.environ["TEST_RECORD"],"w"))\n')
            script.chmod(0o755)
        self.env = dict(os.environ, PATH=f'{self.path}:{os.environ["PATH"]}', TEST_RECORD=str(self.record),
                        RDO_PLATFORM='ppu', RDO_PYTHON=str(self.path / 'fake-python'),
                        RDO_PPU_IMAGE=str(self.image))
        for key in ['SLURM_JOB_ID', 'SLURM_SUBMIT_DIR', 'RDO_REPO_ROOT', 'RDO_PPU_CONTAINER', 'RDO_PPU_PROFILE', 'RDO_DEVICE']:
            self.env.pop(key, None)

    def launch(self, entry, *args, env=None, cwd=ROOT):
        result = subprocess.run(['bash', str(entry), *args], env={**self.env, **(env or {})},
                                cwd=cwd, text=True, capture_output=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        return json.loads(self.record.read_text())

    def test_rdo_submits_labs_pool_whole_and_preserves_arguments(self):
        result = self.launch(ROOT / 'scripts/run_rdo_refusal.sh', '--lr', '0.00002')
        self.assertEqual(result['cmd'], 'sbatch')
        for flag in ['--partition=labs-pool', '--qos=pool-batch', '--comment=ppu=whole', f'--chdir={ROOT}']:
            self.assertIn(flag, result['args'])
        self.assertEqual(result['args'][-3:], ['rdo', '--lr', '0.00002'])

    def test_dim_submits_same_pool_with_model_argument(self):
        # Exercise the shared DIM boundary without requiring /dev/alixpu locally.
        result = self.launch(ROOT / 'scripts/launch_ppu.sh', 'dim', '--model_path', 'mistralai/Magistral-Small-2506')
        self.assertEqual(result['cmd'], 'sbatch')
        self.assertEqual(result['args'][-3:], ['dim', '--model_path', 'mistralai/Magistral-Small-2506'])

    def test_backfill_python_submits_and_preserves_paths_from_other_cwd(self):
        result = subprocess.run([
            sys.executable, str(ROOT / 'scripts/backfill_safety_metrics.py'),
            '--exp-dirs', 'first exp', 'second exp', '--metrics', 'safetyanalisys',
            '--safety-analisys-references', 'refs with spaces.json',
            '--safety-analisys-batch-size', '2'],
            env=self.env, cwd=self.path, text=True, capture_output=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        record = json.loads(self.record.read_text())
        self.assertEqual(record['cmd'], 'sbatch')
        for flag in ['--partition=labs-pool', '--qos=pool-batch', '--comment=ppu=whole',
                     '--job-name=backfill-ppu', f'--chdir={ROOT}']:
            self.assertIn(flag, record['args'])
        forwarded = record['args'][record['args'].index('backfill') + 1:]
        self.assertEqual(forwarded, [
            '--exp-dirs', str(self.path / 'first exp'), str(self.path / 'second exp'),
            '--metrics', 'safetyanalisys', '--safety-analisys-references',
            str(self.path / 'refs with spaces.json'), '--safety-analisys-batch-size', '2'])

    def test_backfill_inside_job_uses_container_python_and_preserves_mask(self):
        venv = self.prepared_venv()
        result = self.launch(ROOT / 'scripts/launch_ppu.sh', 'backfill',
                             '--exp-dirs', '/data/exp one', '/data/exp two', env={
            'SLURM_JOB_ID': 'test', 'CUDA_VISIBLE_DEVICES': 'MIG-assigned',
            'APPTAINERENV_CUDA_VISIBLE_DEVICES': 'MIG-assigned', 'RDO_PPU_VENV': str(venv)})
        self.assertEqual(result['cmd'], 'apptainer')
        self.assertEqual(result['mask'], 'MIG-assigned')
        self.assertEqual(result['args'], ['exec', '--cleanenv', str(self.image),
                         str(venv / 'bin/python'), '-u', str(ROOT / 'scripts/backfill_safety_metrics.py'),
                         '--exp-dirs', '/data/exp one', '/data/exp two'])

    def test_backfill_help_does_not_allocate(self):
        result = subprocess.run([sys.executable, str(ROOT / 'scripts/backfill_safety_metrics.py'), '--help'],
                                env=self.env, cwd=self.path, text=True, capture_output=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertFalse(self.record.exists())

    def test_setup_and_preflight_use_short_allocations(self):
        for flag in ['--setup-only', '--check_environment', '--smoke-test']:
            result = self.launch(ROOT / 'scripts/run_rdo_refusal.sh', flag)
            self.assertEqual(result['cmd'], 'srun')
            self.assertIn('--immediate=30', result['args'])

    def test_resources_does_not_submit(self):
        result = self.launch(ROOT / 'scripts/run_rdo_refusal.sh', '--resources')
        self.assertEqual(result, {'cmd': 'cards', 'args': ['status'], 'mask': None})

    def prepared_venv(self):
        venv = self.path / 'venv'
        venv.mkdir()
        (venv / 'pyvenv.cfg').touch()
        (venv / '.rdo-requirements.sha256').write_text(''.join(
            f'{hashlib.sha256((ROOT / name).read_bytes()).hexdigest()}  {name}\n'
            for name in ['requirements-ppu.txt', 'scripts/setup_rdo_ppu.sh']))
        return venv

    def test_batch_spool_copy_resolves_repo_and_preserves_mig_mask(self):
        spool = self.path / 'slurm_script'
        spool.write_text((ROOT / 'scripts/launch_ppu.sh').read_text())
        venv = self.prepared_venv()
        result = self.launch(spool, 'dim', '--check_environment', cwd=self.path, env={
            'SLURM_JOB_ID': 'test', 'SLURM_SUBMIT_DIR': str(ROOT), 'CUDA_VISIBLE_DEVICES': 'MIG-assigned',
            'APPTAINERENV_CUDA_VISIBLE_DEVICES': 'MIG-assigned', 'RDO_PPU_VENV': str(venv)})
        self.assertEqual(result['cmd'], 'apptainer')
        self.assertEqual(result['mask'], 'MIG-assigned')
        self.assertEqual(result['args'][-2:], [str(ROOT / 'scripts/run_dim_pipeline.sh'), '--check_environment'])

    def test_setup_cannot_run_outside_training_pool(self):
        result = subprocess.run(['bash', str(ROOT / 'scripts/setup_rdo_ppu.sh')],
                                env=self.env, capture_output=True, text=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('inside a Slurm labs-pool job', result.stderr)


if __name__ == '__main__':
    unittest.main()
