from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

RANDOM_STATE = 42
TEST_SIZE = 0.20
TARGET_COLUMN = "request.Attack_Tag"
NORMAL_LABEL = "Normal Traffic"
LOG4J_LABEL = "LOG4J"
DEFAULT_DATA_PATH = (
    Path(__file__).resolve().parents[1] / "data" / "dataset_1_train.json"
)

BASE_CATEGORICAL_FEATURES = (
    "request.method",
    "response.status_code",
    "response.headers.Content-Type",
)
BASE_NUMERIC_FEATURES = (
    "response.headers.Content-Length",
    "has_cookie",
    "has_content_length",
    "sec_fetch_site_none",
)
TEMPLATE_FEATURES = ("has_template",)
JNDI_FEATURES = (
    "accept_encoding_has_jndi",
    "sec_fetch_dest_has_jndi",
    "set_cookie_has_jndi",
    "has_jndi_anywhere",
)
URL_FEATURES = ("url_has_traversal", "url_has_newline_encoded")
ENGINEERED_FEATURES = TEMPLATE_FEATURES + JNDI_FEATURES + URL_FEATURES

FEATURE_CONFIGURATIONS = {
    "Baseline": (),
    "Baseline + template": TEMPLATE_FEATURES,
    "Baseline + JNDI": JNDI_FEATURES,
    "Full engineered set": ENGINEERED_FEATURES,
}

JNDI_PATTERN = r"\$\{.*?jndi\s*:"
JNDI_SCAN_COLUMNS = (
    "request.headers.Accept-Encoding",
    "request.headers.Sec-Fetch-Dest",
    "request.headers.Set-Cookie",
    "request.headers.Accept-Language",
    "request.headers.Host",
    "request.headers.Accept",
    "request.headers.Connection",
    "request.headers.Sec-Fetch-Mode",
    "request.headers.Sec-Fetch-User",
    "request.headers.User-Agent",
    "request.url",
    "request.body",
)
JNDI_FEATURE_MAP = {
    "accept_encoding_has_jndi": "request.headers.Accept-Encoding",
    "sec_fetch_dest_has_jndi": "request.headers.Sec-Fetch-Dest",
    "set_cookie_has_jndi": "request.headers.Set-Cookie",
}

REQUIRED_COLUMNS = frozenset(
    {
        TARGET_COLUMN,
        "request.method",
        "request.url",
        "request.body",
        "request.headers.Cookie",
        "request.headers.Content-Length",
        "request.headers.Sec-Fetch-Site",
        "response.status_code",
        "response.headers.Content-Type",
        "response.headers.Content-Length",
        *JNDI_SCAN_COLUMNS,
    }
)


def load_dataset(path: str | Path = DEFAULT_DATA_PATH) -> pd.DataFrame:
    """Load nested request records from JSON and flatten them like the notebook."""
    dataset_path = Path(path)
    if not dataset_path.is_file():
        raise FileNotFoundError(f"Dataset not found: {dataset_path.resolve()}")

    with dataset_path.open("r", encoding="utf-8") as file:
        records: Any = json.load(file)
    if not isinstance(records, list):
        raise ValueError("The dataset JSON must contain a list of request records.")

    return pd.json_normalize(records, max_level=2)


def _validate_columns(data: pd.DataFrame, *, require_target: bool = True) -> None:
    required_columns = (
        REQUIRED_COLUMNS
        if require_target
        else REQUIRED_COLUMNS.difference({TARGET_COLUMN})
    )
    missing_columns = required_columns.difference(data.columns)
    if missing_columns:
        raise ValueError(
            f"The dataset is missing required columns: {sorted(missing_columns)}"
        )


def preprocess_data(
    data: pd.DataFrame,
    *,
    require_target: bool = True,
) -> pd.DataFrame:
    """Normalize labels and add the notebook's request/response indicators.

    The input is expected to be flattened by :func:`load_dataset` or
    ``pandas.json_normalize``. A copy is returned; the caller's frame is unchanged.
    Set ``require_target=False`` for unlabeled inference records.
    """
    _validate_columns(data, require_target=require_target)
    features = data.copy()
    if TARGET_COLUMN in features:
        features[TARGET_COLUMN] = (
            features[TARGET_COLUMN].fillna(NORMAL_LABEL).astype(str)
        )

    request_columns = [
        column
        for column in features.columns
        if column.startswith("request.") and column != TARGET_COLUMN
    ]
    template_matches = pd.DataFrame(
        {
            column: features[column]
            .fillna("")
            .astype("string")
            .str.contains(r"\$\{", case=False, regex=True, na=False)
            for column in request_columns
        },
        index=features.index,
    )
    has_template = template_matches.any(axis=1).astype("int8")

    jndi_matches = pd.DataFrame(
        {
            column: features[column]
            .fillna("")
            .astype("string")
            .str.contains(JNDI_PATTERN, case=False, regex=True, na=False)
            for column in JNDI_SCAN_COLUMNS
        },
        index=features.index,
    )
    jndi_coverage = jndi_matches.any(axis=1)

    features["has_cookie"] = (
        features["request.headers.Cookie"]
        .fillna("")
        .astype("string")
        .str.len()
        .gt(0)
        .astype("int8")
    )
    features["has_content_length"] = (
        features["request.headers.Content-Length"].notna().astype("int8")
    )
    features["sec_fetch_site_none"] = (
        features["request.headers.Sec-Fetch-Site"].fillna("").eq("none").astype("int8")
    )
    features["response.headers.Content-Length"] = pd.to_numeric(
        features["response.headers.Content-Length"],
        errors="coerce",
    )
    features["has_template"] = has_template

    for feature_name, source_column in JNDI_FEATURE_MAP.items():
        features[feature_name] = (
            features[source_column]
            .fillna("")
            .astype("string")
            .str.contains(JNDI_PATTERN, case=False, regex=True, na=False)
            .astype("int8")
        )
    features["has_jndi_anywhere"] = jndi_coverage.astype("int8")

    url = features["request.url"].fillna("").astype("string")
    features["url_has_traversal"] = url.str.contains(
        r"\.\./", regex=True, na=False
    ).astype("int8")
    features["url_has_newline_encoded"] = url.str.contains(
        r"%0a|%0d", case=False, regex=True, na=False
    ).astype("int8")

    return features


def get_feature_columns(
    additional_features: tuple[str, ...] = ENGINEERED_FEATURES,
) -> list[str]:
    """Return baseline columns plus the requested engineered features."""
    return [
        *BASE_CATEGORICAL_FEATURES,
        *BASE_NUMERIC_FEATURES,
        *additional_features,
    ]


def build_preprocessor(
    categorical_features: tuple[str, ...],
    numeric_features: tuple[str, ...],
) -> ColumnTransformer:
    """Build imputers and one-hot encoding for model input columns."""
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("one_hot_encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    numeric_pipeline = Pipeline(steps=[("imputer", SimpleImputer(strategy="median"))])
    return ColumnTransformer(
        transformers=[
            ("categorical", categorical_pipeline, list(categorical_features)),
            ("numeric", numeric_pipeline, list(numeric_features)),
        ],
        remainder="drop",
    )


def build_classifier_pipeline(
    classifier: BaseEstimator,
    additional_features: tuple[str, ...] = ENGINEERED_FEATURES,
) -> Pipeline:
    """Create a leakage-safe pipeline with preprocessing fitted during training."""
    numeric_features = (*BASE_NUMERIC_FEATURES, *additional_features)
    return Pipeline(
        steps=[
            (
                "preprocessor",
                build_preprocessor(BASE_CATEGORICAL_FEATURES, numeric_features),
            ),
            ("classifier", classifier),
        ]
    )
