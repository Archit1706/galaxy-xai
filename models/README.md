# Model weights

This directory holds trained model checkpoints. `*.pth` / `*.pt` files are **git-ignored**
(they're versioned via DVC / the MLflow registry, not committed) — see the root README.

## Expected file

The service and registry bootstrap look for:

```
models/resnet18_galaxy_best.pth
```

This is the trained ResNet-18 Smooth/Featured classifier (~96% test accuracy on
Galaxy10 DECaLS) produced by the research notebooks in `research/`.

### How to provide it

- **Drop in your trained checkpoint:** copy `resnet18_galaxy_best.pth` from your
  training run (originally saved to Google Drive `CS517_SRAI_Project/models/`) into
  this directory. It must be a `state_dict` matching the architecture in
  [`src/model.py`](../src/model.py) (`build_model`): ResNet-18 with
  `fc = Sequential(Dropout(0.3), Linear(512, 2))`.
- **Or train fresh:** `python -m src.train` (Phase 2) downloads Galaxy10 DECaLS,
  trains, and writes the checkpoint here while logging to MLflow.

The smoke test (`tests/test_model.py`) runs **without** weights present (random init),
so CI is green before a real checkpoint is available. Real predictions need the file.
