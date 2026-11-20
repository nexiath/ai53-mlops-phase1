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
