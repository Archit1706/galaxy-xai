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
| 2 | MLflow tracking + registry (serve from registry) | ⏳ |
| 3 | Evidently drift + Prometheus + Grafana | ⏳ |
| 4 | CI/CD eval gate + scheduled drift→retrain→promote | ⏳ |
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
docker compose up --build
curl -F "file=@your_galaxy.png" http://localhost:8000/predict
```

Interactive API docs: <http://localhost:8000/docs> · Metrics: <http://localhost:8000/metrics>

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
