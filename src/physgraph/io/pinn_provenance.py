"""Conservative scientific compatibility fingerprint, independent of repo layout."""
import hashlib
from pathlib import Path


def training_fingerprint():
    root = Path(__file__).resolve().parents[1]
    digest = hashlib.sha256()
    # Include all training semantics; exclude tests, plots and evaluation settings.
    for folder in ('physics', 'models', 'losses', 'training', 'core'):
        for path in sorted((root/folder).glob('*.py')):
            digest.update(str(path.relative_to(root)).encode()+b'\0'+path.read_bytes()+b'\0')
    return digest.hexdigest()
