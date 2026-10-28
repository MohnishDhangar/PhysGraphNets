"""Optional, scoped CPU runtime settings for the PyTorch baseline."""
from contextlib import contextmanager

import torch


def runtime_metadata(threads: int = 1) -> dict:
    if isinstance(threads, bool) or not isinstance(threads, int) or threads < 1:
        raise ValueError('torch_threads must be a positive integer')
    return {'torch_version': str(torch.__version__), 'device': 'cpu', 'dtype': 'float64',
            'threads': threads, 'deterministic_algorithms': True,
            'adam_step_dtype': 'float64', 'foreach': False, 'fused': False}


@contextmanager
def cpu_runtime(threads: int = 1):
    """Restore caller settings even on failure; no global RNG is consumed."""
    runtime_metadata(threads)
    previous_threads = torch.get_num_threads()
    previous_deterministic = torch.are_deterministic_algorithms_enabled()
    previous_warn = torch.is_deterministic_algorithms_warn_only_enabled()
    try:
        torch.set_num_threads(threads)
        torch.use_deterministic_algorithms(True)
        yield
    finally:
        torch.use_deterministic_algorithms(previous_deterministic, warn_only=previous_warn)
        torch.set_num_threads(previous_threads)
