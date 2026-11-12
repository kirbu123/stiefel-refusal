"""No-allocation tests for the server's scheduler and container boundaries."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


class ServerLauncherTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.path = Path(self.temporary.name)
        self.record = self.path / 'record.json'
        for name in ['sbatch', 'srun', 'cards', 'apptainer', 'fake-python']:
            script = self.path / name
            script.write_text('#!/usr/bin/env python3\nimport json,os,sys\njson.dump({"cmd":os.path.basename(sys.argv[0]),"args":sys.argv[1:],"mask":os.getenv("APPTAINERENV_CUDA_VISIBLE_DEVICES")},open(os.environ["TEST_RECORD"],"w"))\n')
            script.chmod(0o755)
        self.env = dict(os.environ, PATH=f'{self.path}:{os.environ["PATH"]}', TEST_RECORD=str(self.record),
                        RDO_PLATFORM='ppu', RDO_PYTHON=str(self.path / 'fake-python'))
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
        result = self.launch(ROOT / 'scripts/run_dim_pipeline.sh', '--model_path', 'mistralai/Magistral-Small-2506')
        self.assertEqual(result['cmd'], 'sbatch')
        self.assertEqual(result['args'][-3:], ['dim', '--model_path', 'mistralai/Magistral-Small-2506'])

    def test_setup_and_preflight_use_short_allocations(self):
        for flag in ['--setup-only', '--check_environment', '--smoke-test']:
            result = self.launch(ROOT / 'scripts/run_rdo_refusal.sh', flag)
            self.assertEqual(result['cmd'], 'srun')
            self.assertIn('--immediate=30', result['args'])

    def test_resources_does_not_submit(self):
        result = self.launch(ROOT / 'scripts/run_rdo_refusal.sh', '--resources')
        self.assertEqual(result, {'cmd': 'cards', 'args': ['status'], 'mask': None})

    def test_batch_spool_copy_resolves_repo_and_preserves_mig_mask(self):
        spool = self.path / 'slurm_script'
        spool.write_text((ROOT / 'scripts/launch_ppu.sh').read_text())
        venv = self.path / 'venv'
        venv.mkdir()
        (venv / 'pyvenv.cfg').touch()
        (venv / '.rdo-requirements.sha256').write_text(''.join(
            f'{hashlib.sha256((ROOT / name).read_bytes()).hexdigest()}  {name}\n'
            for name in ['requirements-ppu.txt', 'scripts/setup_rdo_ppu.sh']))
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
