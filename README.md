# api-sentinel-ml

## Model Training

Once the project environment is set up, run the following command from the repository root:

```bash
uv run python -m src.train
```

The training process loads `data/dataset_1_train.json`, applies the feature engineering steps defined in the notebook, and evaluates a decision tree classifier using grid search and stratified cross-validation.

To save the trained pipeline (preprocessing + model), specify an output path:

```bash
uv run python -m src.train --model-output models/decision_tree.joblib
```

## Traffic Classification

The inference module loads the trained pipeline and accepts either a single JSON request object or a list of objects following the same nested structure used in the dataset.

Missing input fields are treated as absent, and the label field, if provided, is ignored.

```bash
uv run python -m src.inference --input request.json

uv run python -m src.inference \
  --model models/decision_tree.joblib \
  --input requests.json \
  --output predictions.json
```

Each result contains the predicted label (`prediction`), its confidence score (`confidence`), and the class probabilities (`probabilities`).

By default, the module looks for `models/decision_tree.pkl`.

Serialized models loaded with joblib can execute Python code. Only use trusted artifacts that are compatible with the dependency versions used during training.