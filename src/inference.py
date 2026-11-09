from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from sklearn.pipeline import Pipeline

from src.preprocessing import (
    REQUIRED_COLUMNS,
    TARGET_COLUMN,
    get_feature_columns,
    preprocess_data,
)

DEFAULT_MODEL_PATH = (
    Path(__file__).resolve().parents[1] / "models" / "decision_tree.pkl"
)


def load_model(model_path: str | Path = DEFAULT_MODEL_PATH) -> Pipeline:
    """Load a trusted, previously trained scikit-learn pipeline."""
    path = Path(model_path)
    if not path.is_file():
        raise FileNotFoundError(f"Trained model not found: {path.resolve()}")

    model = joblib.load(path)
    if not isinstance(model, Pipeline):
        raise TypeError(
            f"Expected a scikit-learn Pipeline in {path.resolve()}, "
            f"got {type(model).__name__}."
        )
    if not callable(getattr(model, "predict_proba", None)):
        raise TypeError("The trained pipeline must implement predict_proba().")
    if getattr(model, "classes_", None) is None:
        raise ValueError("The trained pipeline has no fitted classes.")
    return model


def read_records(input_path: str | Path) -> list[Mapping[str, Any]]:
    """Read one nested JSON request or a list of request records."""
    path = Path(input_path)
    with path.open("r", encoding="utf-8") as file:
        payload: Any = json.load(file)

    if isinstance(payload, Mapping):
        records = [payload]
    elif isinstance(payload, list):
        records = payload
    else:
        raise ValueError("The input JSON must contain a request object or a list.")

    for index, record in enumerate(records):
        if not isinstance(record, Mapping):
            raise ValueError(f"Request at index {index} must be a JSON object.")
    return records


def predict_records(
    records: Sequence[Mapping[str, Any]],
    model: Pipeline,
) -> list[dict[str, Any]]:
    """Predict an attack label and per-class probabilities for each request."""
    if not records:
        return []
    probability_predictor = getattr(model, "predict_proba", None)
    if not callable(probability_predictor):
        raise TypeError("The trained pipeline must implement predict_proba().")
    classes = getattr(model, "classes_", None)
    if classes is None:
        raise ValueError("The trained pipeline has no fitted classes.")

    for index, record in enumerate(records):
        if not isinstance(record, Mapping):
            raise ValueError(f"Request at index {index} must be a JSON object.")

    data = pd.json_normalize(records, max_level=2)
    # Labels in an incoming record must never become model features.
    data = data.drop(columns=[TARGET_COLUMN], errors="ignore")

    # Many API records omit headers or response metadata. These fields are
    # feature-engineering inputs, not mandatory inference-time payload fields.
    required_feature_sources = REQUIRED_COLUMNS.difference({TARGET_COLUMN})
    for column in required_feature_sources:
        if column not in data:
            data[column] = None

    processed = preprocess_data(data, require_target=False)
    features = processed.loc[:, get_feature_columns()]

    predictions = model.predict(features)
    probabilities = probability_predictor(features)
    classes = list(classes)
    if len(predictions) != len(records) or len(probabilities) != len(records):
        raise ValueError(
            "The model returned a different number of results than inputs."
        )

    results: list[dict[str, Any]] = []
    for prediction, row_probabilities in zip(predictions, probabilities, strict=True):
        if len(row_probabilities) != len(classes):
            raise ValueError("The model returned probabilities with an invalid shape.")
        if not isinstance(prediction, str):
            scalar_value = getattr(prediction, "item", None)
            if callable(scalar_value):
                prediction = scalar_value()
        if not isinstance(prediction, str):
            raise TypeError("The model's predicted attack labels must be strings.")

        class_probabilities: dict[str, float] = {}
        for label, probability in zip(classes, row_probabilities, strict=True):
            if not isinstance(label, str):
                scalar_value = getattr(label, "item", None)
                if callable(scalar_value):
                    label = scalar_value()
            if not isinstance(label, str):
                raise TypeError("The model's attack classes must be strings.")
            class_probabilities[label] = float(probability)
        if prediction not in class_probabilities:
            raise ValueError(
                f"Predicted label {prediction!r} is not in model.classes_."
            )

        results.append(
            {
                "prediction": prediction,
                "confidence": class_probabilities[prediction],
                "probabilities": class_probabilities,
            }
        )
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Path to a JSON request object or a list of request objects.",
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=DEFAULT_MODEL_PATH,
        help=f"Path to the trained model pipeline (default: {DEFAULT_MODEL_PATH}).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional path for writing predictions as JSON.",
    )
    args = parser.parse_args()

    predictions = predict_records(read_records(args.input), load_model(args.model))
    output = json.dumps(predictions, ensure_ascii=False, indent=2)
    if args.output is None:
        print(output)
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(f"{output}\n", encoding="utf-8")


if __name__ == "__main__":
    main()
