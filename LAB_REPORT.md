# AI53 - MLOps Project Phase 1 - Lab report

## Milestone 0 - Starter audit

| # | File or setting | Problem | Fix |
|---|---|---|---|
| 1 | `src/reproducibility.py` | Only Python `random` is seeded, not NumPy or PyTorch. | Seed NumPy, PyTorch CPU and CUDA. |
| 2 | `src/reproducibility.py` | The DataLoader generator is not seeded and `seed_worker()` is empty, so the shuffle order changes. | Seed the generator and the workers. |
| 3 | `data/raw/`, `.gitignore` | The images are ignored by Git and not tracked anywhere. | Track them with DVC and commit `data/raw.dvc`. |
| 4 | `src/validate_data.py` | `validate_dataset()` is not implemented, so bad data is not detected. | Check splits, classes, counts and that images can be decoded. |
| 5 | no `dvc.yaml` | Stages are run by hand, with no dependencies between them. | Write a DVC pipeline with the four stages. |
| 6 | `src/provenance.py` | `build_run_manifest()` is not implemented: no commit SHA, data identity or versions saved. | Write `reports/run_manifest.json` for each run. |
| 7 | `src/train.py` | The MLflow run is empty: no params, metrics, artifacts or model. | Complete the logging TODOs. |
| 8 | `models/model.pt` | Overwritten at each training and not linked to a run. | Log each model in MLflow with its Run ID. |
| 9 | `params.yaml`, `selection.run_id` | No model is selected and nothing protects the test set. | Select on validation only, then test once. |

### Engineering checkpoint 1

A `model.pt` only shows that one training finished. It does not say which code, data, seed, settings or library versions were used, so nobody can rebuild it and get the same result. In the starter it is even overwritten at each run.

## Milestone 1 - Repeatable training

- `pyproject.toml`: lists the packages the project needs, with version ranges.
- `uv.lock`: the exact versions resolved from it, so everyone installs the same ones.
- `.venv/`: the local folder where uv installs these packages. It is not in Git, it is rebuilt with `uv sync --locked`.

First `pytest -q` on the starter: several tests fail (reproducibility, validation, provenance) because of the TODOs.

After completing `src/reproducibility.py`, `tests/test_reproducibility.py` passes (3 passed).

### Engineering checkpoint 2

No. The seed only fixes the random numbers. Results can still differ between CPUs, GPUs or library versions, because floating-point operations are not computed in the same order and some CUDA operations are not deterministic.

## Milestone 2 - Data versioning

I used option A (the included archives). The three checksums are OK and `check_dataset.py` finds 1034 / 133 / 128 images.

`data/raw.dvc` is a small text file with the hash of the folder (`2b62d5a2...dir`, about 180 MB). Git tracks this file, not the images.

### Engineering checkpoint 3

- The Git remote (GitHub) stores the code, the settings and the small `.dvc` pointer files.
- The DVC remote (`../ai53-phase1-dvc-storage`) stores the real image files, named by their hash.
- No, `git push` only sends the pointer. The images are sent with `dvc push`.
