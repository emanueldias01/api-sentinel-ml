# api-sentinel-ml

## Dataset

The dataset used by this project was obtained from the [Cisco Ariel Uni API Security Challenge repository](https://github.com/ArielCyber/Cisco_Ariel_Uni_API_security_challenge).

## Model Training

Once the project environment is set up, run the following command from the repository root:

```bash
uv run python -m src.train
```

The training process loads `data/dataset_1_train.json`, applies the feature engineering steps defined in the notebook, and evaluates a decision tree classifier using grid search and stratified cross-validation.

To save the trained pipeline (preprocessing + model), specify an output path:

```bash
uv run python -m src.train --model-output models/decision_tree.pkl
```

## Traffic Classification

The inference module loads the trained pipeline and accepts either a single JSON request object or a list of objects following the same nested structure used in the dataset.

Missing input fields are treated as absent, and the label field, if provided, is ignored.

```bash
uv run python -m src.inference --input request.json

uv run python -m src.inference \
  --model models/decision_tree.pkl \
  --input requests.json \
  --output predictions.json
```

Each result contains the predicted label (`prediction`), its confidence score (`confidence`), and the class probabilities (`probabilities`).

By default, the module looks for `models/decision_tree.pkl`.

Serialized models loaded with joblib can execute Python code. Only use trusted artifacts that are compatible with the dependency versions used during training.

## Prediction API

### Docker Compose

The `trainer` job runs before the API, trains the model from the dataset, and
persists the resulting pipeline in the `api-sentinel-ml-models` named volume.
The API starts only after training succeeds and mounts the model read-only.

Build, train, and start the API:

```bash
docker compose up -d --build
```

Check readiness:

```bash
docker compose up -d api
curl http://localhost:8000/ready
```

The trainer runs again when the Compose application is recreated, which makes
the model artifact reflect the current dataset and code. The named volume keeps
the artifact across container recreation. To remove the trained artifact as
well, use `docker compose down -v`.

Start the HTTP API from the repository root:

```bash
uv run uvicorn src.api:app --host 0.0.0.0 --port 8000
```

By default, the API loads `models/decision_tree.pkl` once at startup. Set
`API_MODEL_PATH` to use another trained pipeline. If the model cannot be loaded,
the application fails to start rather than serving unsuccessful predictions.

`POST /predict` accepts one nested request record. `POST /predict/batch`
accepts a JSON array of 1 to 100 records. Both return the predicted attack type,
its confidence score, and the probabilities per class. Interactive API
documentation is available at `/docs`; `/health` is a liveness probe and
`/ready` reports whether the model has loaded.

```bash
curl -X POST http://localhost:8000/predict \
  -H 'Content-Type: application/json' \
  -d '{"request":{"method":"GET","url":"/"}}'
```

For production deployment, run behind a TLS-terminating reverse proxy and
configure authentication, request-size limits, and rate limiting at the
gateway. Each Uvicorn worker loads its own copy of the model.
