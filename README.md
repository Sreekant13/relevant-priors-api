# Relevant Priors API

HTTP API for the New Lantern `relevant-priors-v1` challenge.

## Endpoint

The app exposes both:

- `POST /`
- `POST /predict`

Both accept the challenge JSON payload and return:

```json
{
  "predictions": [
    {
      "case_id": "1001016",
      "study_id": "2453245",
      "predicted_is_relevant": true
    }
  ]
}
```

## Local run

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

uvicorn app.main:app --host 0.0.0.0 --port 8000
```

Then submit:

```text
http://localhost:8000/predict
```

## Train model

```bash
python scripts/train.py --data data/relevant_priors_public.json --out models/relevant_priors_model.joblib
```

The submitted API uses the saved `models/relevant_priors_model.joblib`.

## Deploy on Render

Use these settings:

- Environment: Python
- Build command: `pip install -r requirements.txt`
- Start command: `uvicorn app.main:app --host 0.0.0.0 --port $PORT`
- Health check path: `/health`

Submit the deployed `/predict` URL.
