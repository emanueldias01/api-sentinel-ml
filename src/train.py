"""Train and evaluate the notebook's tuned Decision Tree classifier.

Run from the repository root with ``python -m src.train``.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    make_scorer,
    precision_score,
    recall_score,
)
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier

from src.preprocessing import (
    DEFAULT_DATA_PATH,
    ENGINEERED_FEATURES,
    LOG4J_LABEL,
    RANDOM_STATE,
    TEST_SIZE,
    build_classifier_pipeline,
    get_feature_columns,
    load_dataset,
    preprocess_data,
)

DECISION_TREE_PARAM_GRID = {
    "classifier__criterion": ("gini", "entropy"),
    "classifier__max_depth": (None, 5, 10, 20),
    "classifier__min_samples_split": (2, 5),
    "classifier__min_samples_leaf": (1, 2),
}

SCORING = {
    "accuracy": "accuracy",
    "precision_macro": make_scorer(precision_score, average="macro", zero_division=0),
    "recall_macro": make_scorer(recall_score, average="macro", zero_division=0),
    "f1_macro": make_scorer(f1_score, average="macro", zero_division=0),
    "log4j_recall": make_scorer(
        recall_score,
        labels=[LOG4J_LABEL],
        average="macro",
        zero_division=0,
    ),
}


def split_data(
    features: pd.DataFrame,
    target: pd.Series,
    *,
    test_size: float = TEST_SIZE,
    random_state: int = RANDOM_STATE,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Make the notebook's stratified holdout split."""
    return train_test_split(
        features,
        target,
        test_size=test_size,
        random_state=random_state,
        stratify=target,
    )


def tune_decision_tree(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    additional_features: tuple[str, ...] = ENGINEERED_FEATURES,
) -> GridSearchCV:
    """Fit the notebook's five-fold grid search, selecting by macro F1."""
    pipeline = build_classifier_pipeline(
        DecisionTreeClassifier(random_state=RANDOM_STATE),
        additional_features,
    )
    search = GridSearchCV(
        estimator=pipeline,
        param_grid=DECISION_TREE_PARAM_GRID,
        scoring=SCORING,
        refit="f1_macro",
        cv=5,
        n_jobs=-1,
        verbose=1,
        return_train_score=False,
    )
    return search.fit(X_train, y_train)


def evaluate_model(
    model: Pipeline | BaseEstimator,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> dict[str, float]:
    """Compute holdout metrics, including the notebook's Log4j metrics."""
    predictions = model.predict(X_test)
    report = classification_report(
        y_test,
        predictions,
        output_dict=True,
        zero_division=0,
    )
    log4j_metrics = report.get(LOG4J_LABEL, {})
    return {
        "accuracy": accuracy_score(y_test, predictions),
        "macro_precision": precision_score(
            y_test, predictions, average="macro", zero_division=0
        ),
        "macro_recall": recall_score(
            y_test, predictions, average="macro", zero_division=0
        ),
        "macro_f1": f1_score(y_test, predictions, average="macro", zero_division=0),
        "log4j_precision": log4j_metrics.get("precision", 0.0),
        "log4j_recall": log4j_metrics.get("recall", 0.0),
        "log4j_f1": log4j_metrics.get("f1-score", 0.0),
        "log4j_support": float(log4j_metrics.get("support", 0)),
    }


def train_and_evaluate(
    dataset_path: str | Path = DEFAULT_DATA_PATH,
    model_output: str | Path | None = None,
) -> tuple[GridSearchCV, dict[str, float]]:
    """Load data, engineer features, tune on training data, and evaluate holdout."""
    data = preprocess_data(load_dataset(dataset_path))
    X = data.loc[:, get_feature_columns(ENGINEERED_FEATURES)]
    y = data["request.Attack_Tag"]
    X_train, X_test, y_train, y_test = split_data(X, y)

    search = tune_decision_tree(X_train, y_train)
    metrics = evaluate_model(search.best_estimator_, X_test, y_test)

    if model_output is not None:
        output_path = Path(model_output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(search.best_estimator_, output_path)

    print(f"Best parameters: {search.best_params_}")
    print(f"Best cross-validation macro F1: {search.best_score_:.4f}")
    print("Holdout metrics:")
    for name, value in metrics.items():
        print(f"  {name}: {value:.4f}")
    print("\nHoldout classification report:")
    print(
        classification_report(
            y_test,
            search.best_estimator_.predict(X_test),
            digits=4,
            zero_division=0,
        )
    )
    if model_output is not None:
        print(f"Saved model pipeline to {Path(model_output).resolve()}")
    return search, metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data",
        type=Path,
        default=DEFAULT_DATA_PATH,
        help="Path to the nested JSON dataset.",
    )
    parser.add_argument(
        "--model-output",
        type=Path,
        help="Optional path for saving the fitted preprocessing and model pipeline.",
    )
    args = parser.parse_args()
    train_and_evaluate(args.data, args.model_output)


if __name__ == "__main__":
    main()
