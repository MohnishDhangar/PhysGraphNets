"""Experiment provenance shared by local and cloud entry points."""
from datetime import datetime, timezone
import hashlib
from importlib.metadata import PackageNotFoundError, version
import json
from pathlib import Path
import platform
import subprocess
import sys


def environment_manifest() -> dict:
    root = Path(__file__).resolve().parents[3]

    def git(*args):
        try:
            result = subprocess.run(["git", "-C", str(root), *args],
                                    capture_output=True, text=True, timeout=5)
        except (OSError, subprocess.TimeoutExpired):
            return None
        return result.stdout.strip() if result.returncode == 0 else None

    revision = git("rev-parse", "HEAD")
    status = git("status", "--porcelain")
    source_files = sorted({*root.glob("src/**/*.py"), *root.glob("scripts/**/*.py"),
                           *root.glob("tests/**/*.py"), *root.glob("configs/**/*.json"),
                           *root.glob("configs/**/*.yml"), *root.glob("configs/**/*.yaml"), *root.glob("configs/**/*.txt"),
                           *root.glob(".github/workflows/*.yml"),
                           *root.glob("pyproject.toml")})
    digest = hashlib.sha256()
    for path in source_files:
        digest.update(str(path.relative_to(root)).encode() + b"\0" + path.read_bytes() + b"\0")
    dependencies = {}
    for name in ("physgraph", "numpy", "scipy", "matplotlib", "pytest", "torch"):
        try:
            dependencies[name] = version(name)
        except PackageNotFoundError:
            dependencies[name] = None
    return {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "python": platform.python_version(), "executable": sys.executable,
        "platform": platform.platform(), "machine": platform.machine(),
        "device": "cpu", "precision": "float64", "dependencies": dependencies,
        "code_revision": revision, "git_dirty": None if status is None else bool(status),
        "revision_note": None if revision else "Git revision unavailable; see source_sha256",
        "source_sha256": digest.hexdigest() if source_files else None,
        "source_hash_files": [str(p.relative_to(root)) for p in source_files],
    }


def write_manifest(path: str | Path, config: dict, metrics: dict | list,
                   elapsed_seconds: float) -> dict:
    manifest = {**environment_manifest(), "config": config,
                "metrics": metrics, "elapsed_seconds": elapsed_seconds}
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n")
    return manifest
