"""Device selection and small runtime probes, without loading model weights."""
import argparse
import os
from importlib.metadata import version
from pathlib import Path

import torch


def resolve_device(requested='auto'):
    if requested == 'auto':
        requested = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = torch.device(requested)
    if device.type not in ('cpu', 'cuda'):
        raise ValueError('Supported devices: auto, cpu, cuda, cuda:N')
    if device.type == 'cuda':
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA requested but unavailable to this PyTorch process. '
                               'Use --device cpu (RDO_DEVICE=cpu), or provide an accessible accelerator and compatible runtime.')
        if device.index is not None and device.index >= torch.cuda.device_count():
            raise ValueError(f'{device} is outside the {torch.cuda.device_count()} visible CUDA devices')
    # Only rank devices visible within this process's scheduler allocation.
    if device.type == 'cuda' and device.index is None and os.getenv('RDO_PLATFORM') == 'ppu':
        devices = [(torch.cuda.mem_get_info(i)[0], i) for i in range(torch.cuda.device_count())]
        free, index = max(devices)
        device = torch.device(f'cuda:{index}')
        print(f'[runtime] Selected allocated PPU {index}: {free / 2**30:.1f} GiB free', flush=True)
    return device


def runtime_dtype(device, dtype):
    if device.type == 'cpu' and dtype == torch.float16:
        print('[runtime] CPU float16 requested; using float32 for training compatibility', flush=True)
        return torch.float32
    if device.type == 'cuda' and dtype == torch.bfloat16:
        with torch.cuda.device(device):
            if not torch.cuda.is_bf16_supported():
                print('[runtime] CUDA device lacks bfloat16 support; using float32', flush=True)
                return torch.float32
    return dtype


def configure_runtime(requested='auto', dtype=torch.bfloat16, cpu_threads=None):
    device = resolve_device(requested)
    if device.type == 'cpu':
        available = len(os.sched_getaffinity(0)) if hasattr(os, 'sched_getaffinity') else (os.cpu_count() or 1)
        threads = cpu_threads if cpu_threads is not None else min(16, available)
        if threads < 1:
            raise ValueError('cpu_threads must be positive')
        torch.set_num_threads(min(threads, available))
        print('[runtime] No CUDA selected; CPU training and evaluation can be slow.', flush=True)
        vendor_device = Path('/dev/alixpu')
        if vendor_device.exists():
            try:
                fd = os.open(vendor_device, os.O_RDWR)
                os.close(fd)
                status = 'present; not selected by this PyTorch runtime'
            except OSError as error:
                status = f'inaccessible: {error.strerror}'
            print(f'[runtime] {vendor_device}: {status}', flush=True)
    dtype = runtime_dtype(device, dtype)
    # Exercise the kernels used by steering training before generating targets.
    try:
        x = torch.eye(4, device=device, dtype=dtype).requires_grad_()
        (x @ x).float().square().sum().backward()
    except RuntimeError:
        if device.type != 'cpu' or dtype == torch.float32:
            raise
        dtype = torch.float32
        x = torch.eye(4, device=device, dtype=dtype).requires_grad_()
        (x @ x).square().sum().backward()
        print('[runtime] CPU dtype probe failed; using float32', flush=True)
    torch.linalg.qr(torch.eye(4, device=device, dtype=torch.float32))
    import psutil
    print('[runtime] ' + ' '.join(f'{name}={version(name)}' for name in ('transformers', 'nnsight', 'numpy')), flush=True)
    print(f'[runtime] available RAM={psutil.virtual_memory().available / 2**30:.1f} GiB', flush=True)
    print(f'[runtime] torch={torch.__version__} device={device} dtype={dtype} '
          f'CUDA devices={torch.cuda.device_count()} CPU threads={torch.get_num_threads()} '
          'matmul/backward/QR=OK', flush=True)
    return device, dtype


def check_data_runtime():
    """Detect Arrow/NumPy ABI failures before loading model weights."""
    import numpy as np
    import pyarrow as pa
    import datasets
    assert np.array_equal(pa.array(np.array([0, 1])).to_numpy(), [0, 1])
    assert datasets.Dataset.from_dict({'answer': [0]})[0]['answer'] == 0
    print(f'[runtime] Data: numpy={np.__version__} pyarrow={pa.__version__} datasets={datasets.__version__}', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', default=os.getenv('RDO_DEVICE', 'auto'))
    parser.add_argument('--cpu_threads', type=int)
    args = parser.parse_args()
    configure_runtime(args.device, cpu_threads=args.cpu_threads)
