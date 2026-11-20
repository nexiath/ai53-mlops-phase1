from __future__ import annotations

import hashlib
import json
import platform
import subprocess
from importlib.metadata import version
from pathlib import Path

import torch


def training_data_identity(
    train_manifest: Path = Path("data/processed/manifests/train.csv"),
    validation_manifest: Path = Path("data/processed/manifests/validation.csv"),
) -> str:
    """Return a deterministic identity for the training + validation manifests.

    This helper is provided because the final evaluator uses the same identity
    to verify that the selected MLflow Run matches the current prepared data.
    Do not hard-code the value.
    """
    paths = (train_manifest, validation_manifest)
    if any(not path.exists() for path in paths):
        return "unavailable"

    digest = hashlib.sha256()
    for path in paths:
        digest.update(path.name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def run_git(*args: str) -> str | None:
    """Run a git command and return its output, or None if it fails."""
    try:
        result = subprocess.run(
            ["git", *args], capture_output=True, text=True, check=False
        )
    except OSError:
        return None
    if result.returncode != 0:
        return None
    return result.stdout.strip()


def git_state() -> dict:
    commit = run_git("rev-parse", "HEAD")
    # reports/ and dvc.lock are rewritten by the pipeline itself (DVC removes
    # the old outputs before a stage runs), so they are not counted as changes.
    status = run_git(
        "status", "--porcelain", "--", ".", ":(exclude)reports", ":(exclude)dvc.lock"
    )
    return {
        "commit": commit or "unavailable",
        "dirty": None if status is None else status != "",
    }


def build_run_manifest(params: dict, device: str, mlflow_run_id: str) -> dict:
    """Build experiment provenance metadata for one training run."""
    train = params["train"]
    augmentation = params["augmentation"]

    return {
        "git": git_state(),
        "data": {
            "training_data_identity": training_data_identity(),
        },
        "environment": {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "torch": version("torch"),
            "torchvision": version("torchvision"),
            "dvc": version("dvc"),
            "mlflow": version("mlflow"),
            "cuda": torch.version.cuda if torch.cuda.is_available() else None,
        },
        "configuration": {
            "image_size": params["data"]["image_size"],
            "batch_size": train["batch_size"],
            "epochs": train["epochs"],
            "learning_rate": train["learning_rate"],
            "dropout": train["dropout"],
            "augmentation_enabled": augmentation["enabled"],
            "horizontal_flip": augmentation["horizontal_flip"],
            "rotation_degrees": augmentation["rotation_degrees"],
        },
        "execution": {
            "device": device,
            "random_state": train["random_state"],
            "mlflow_run_id": mlflow_run_id,
        },
    }


def write_run_manifest(params: dict, device: str, mlflow_run_id: str, path: Path) -> dict:
    manifest = build_run_manifest(params, device, mlflow_run_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    return manifest
