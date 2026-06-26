# GalaxyServe

**Production MLOps loop for a galaxy morphology classifier.**

GalaxyServe takes a trained ResNet-18 that classifies galaxies as **Smooth** vs
**Featured** (~96% test accuracy on Galaxy10 DECaLS, from the research in
[`research/`](research/)) and runs it the way a model is run *in production*:
served behind an API, tracked in a model registry, monitored for data/prediction
drift, and automatically retrained + promoted through an evaluation gate.

> One project, the full loop: **model serving · experiment tracking · model
> registry · containerization · drift detection · monitoring · CI/CD · automated
> retraining · champion/challenger promotion.**

---

## Architecture

```
                 ┌─────────────── GitHub Actions (CI/CD) ───────────────┐
                 │  lint → test → train-smoke → EVAL GATE → build image  │
                 └───────────────────────┬───────────────────────────────┘
                                          ▼
  training data ──► train.py ──► MLflow (tracking + model registry) ──► model artifact
       ▲                                   │  champion/challenger promote
       │ retrain trigger                   ▼
  ┌────┴─────────┐              FastAPI service ──► /predict /predict_batch /health /metrics
  │ retrain.py   │◄── drift > threshold        │
  └──────────────┘                             ├─► Prometheus ──► Grafana (latency, QPS, errors)
       ▲                                        └─► prediction + input log
       │                                              │
       └──────────── Evidently (PSI/KS drift) ◄───────┘
```

## Status (built phase by phase)

| Phase | Scope | State |
|------|-------|-------|
| 0 | Scaffold, model module, smoke test, CLI predict | ✅ done |
| 1 | FastAPI service (`/predict` `/predict_batch` `/health` `/metrics`) + Docker | ✅ done |
| 2 | MLflow tracking + registry (serve from registry) | ✅ done |
| 3 | Evidently drift + Prometheus + Grafana | ✅ done |
| 4 | CI/CD eval gate + scheduled drift→retrain→promote | ✅ done |
| 5 | Load test + polish + deploy | ⏳ |

## Quickstart

### Local (Python)

```bash
python -m venv .venv && .venv/Scripts/activate      # Windows
pip install -e ".[serve,dev]"

# Place the trained checkpoint at models/resnet18_galaxy_best.pth (see models/README.md)
python -m src.predict --demo                         # smoke prediction, no data needed
uvicorn src.service:app --port 8000                  # run the API
```

### Docker

```bash
docker compose up --build      # starts MLflow (:5000) + the inference service (:8000)
curl -F "file=@your_galaxy.png" http://localhost:8000/predict
```

Interactive API docs: <http://localhost:8000/docs> · Metrics: <http://localhost:8000/metrics> ·
MLflow UI: <http://localhost:5000>

### MLflow registry (serve from the registry)

The service can load its model from the **MLflow Model Registry** instead of a
file (`GALAXYSERVE_USE_REGISTRY=true`, the default in `docker-compose.yml`). Seed
the registry from the existing trained checkpoint and promote it to Production:

```bash
# with an MLflow server running at :5000
python -m src.register_model --tracking-uri http://localhost:5000 --promote --stage Production
```

`/health` then reports `model_stage: "Production"` and the registry version it is
serving. Retraining (`python -m src.train`) logs runs and registers new versions;
if the registry is empty or unreachable the service falls back to the local
weights file automatically.

### Monitoring & drift demo ("watch it spike")

`docker compose up` also starts **Prometheus** (:9090) and **Grafana** (:3000,
admin/admin) with a provisioned *Serving & Drift* dashboard. The service logs an
interpretable feature vector (brightness, contrast, per-channel colour stats) +
the prediction for every request; **Evidently** (PSI/KS) compares the live log to
a reference built from the training distribution.

```bash
python -m src.build_reference --source synthetic --n 300   # baseline distribution
python -m src.simulate_traffic --clean 60 --trigger        # -> drift_score ~0.0
python -m src.simulate_traffic --shifted 60 --trigger      # -> drift_score spikes to ~0.9
```

`POST /drift/check` recomputes on demand; `GET /drift/status` returns the last
summary. Drift is exported as Prometheus gauges (`galaxyserve_drift_score`,
`galaxyserve_dataset_drift`, `galaxyserve_prediction_drift_detected`, …) and the
Grafana panels turn red when a survey shift is detected. Point `build_reference`
at Galaxy10 (`--source galaxy10`) and feed Galaxy Zoo Evo images to demo drift on
the real cross-survey distribution shift.

### CI/CD and the retraining loop

Three GitHub Actions workflows in [`.github/workflows`](.github/workflows):

- **`ci.yml`** (PRs + master): ruff lint → pytest → smoke-train (pipeline runs) →
  **eval gate** (`src.eval_gate` fails the build if accuracy is below the floor).
  A change that breaks the model can't be merged.
- **`build.yml`** (master): build the image and push it to GHCR.
- **`drift-retrain.yml`** (scheduled + manual): drift check → `src.retrain` trains
  a challenger → `src.evaluate` → `src.promote` promotes it **only if it clears the
  floor and beats the champion**, otherwise it is parked in Staging.

```bash
python -m src.eval_gate --floor 0.96                       # the gate
python -m src.retrain --tracking-uri http://localhost:5000 # drift -> train -> promote
python -m src.promote --challenger-version 7 --floor 0.96  # champion/challenger only
```

The champion/challenger rule means the service's Production model never regresses:
a new version is served only when it is provably better.

## API

| Method | Path | Description |
|---|---|---|
| `POST` | `/predict` | Single image (multipart `file`) → class + confidence + probabilities |
| `POST` | `/predict_batch` | Multiple images (multipart `files`) → per-item results (partial failures OK) |
| `GET` | `/health` | Liveness + model/weights status (`503` if no model) |
| `GET` | `/ready` | Readiness probe |
| `GET` | `/metrics` | Prometheus exposition |

Example:

```bash
$ curl -s -F "file=@galaxy.png" http://localhost:8000/predict
{
  "class_id": 1, "label": "Featured", "confidence": 0.93,
  "probabilities": {"Smooth": 0.07, "Featured": 0.93},
  "filename": "galaxy.png", "inference_ms": 25.9, "model_version": "local"
}
```

## Tech stack

Serving **FastAPI** · Tracking/registry **MLflow** · Drift **Evidently** ·
Monitoring **Prometheus + Grafana** · Containers **Docker Compose** · CI/CD
**GitHub Actions** · Load testing **Locust**.

> The serving layer is FastAPI for robustness and speed; **BentoML** is a
> documented future alternative for an even cleaner auto-containerization story.

## Repository layout

```
src/         train · evaluate · promote · service · monitoring · retrain
tests/       API contract tests + eval-gate test
monitoring/  prometheus config + grafana dashboards
load/        locust load test
research/    original notebooks, report, and research README
```

## Authors

Archit Rathod · Gargi Sathe — University of Illinois Chicago.
