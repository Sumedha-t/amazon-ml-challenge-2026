import pickle
import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, recall_score, fbeta_score
from sklearn.model_selection import train_test_split

from src.features.pair_features import build_pair_features


FEATURE_COLUMNS = [
    "name_char_similarity",
    "name_token_similarity",
    "name_edit_similarity",
    "address_char_similarity",
    "address_token_similarity",
    "address_edit_similarity",
    "address_number_overlap",
    "country_same",
    "name_a_missing",
    "name_b_missing",
    "address_a_missing",
    "address_b_missing",
]


def feature_vector(record_a, record_b):
    features = build_pair_features(record_a, record_b)
    return [features[col] for col in FEATURE_COLUMNS]


def train_model(X, y):
    model = LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        random_state=42,
    )

    model.fit(X, y)

    return model


def evaluate_model(model, X, y, threshold=0.5):
    probabilities = model.predict_proba(X)[:, 1]
    predictions = (probabilities >= threshold).astype(int)

    precision = precision_score(y, predictions, zero_division=0)
    recall = recall_score(y, predictions, zero_division=0)
    f05 = fbeta_score(y, predictions, beta=0.5, zero_division=0)

    return {
        "precision": precision,
        "recall": recall,
        "f0.5": f05,
        "threshold": threshold,
    }


def save_model(model, path):
    with open(path, "wb") as f:
        pickle.dump(
            {
                "model": model,
                "feature_columns": FEATURE_COLUMNS,
            },
            f,
        )


def load_model(path):
    with open(path, "rb") as f:
        bundle = pickle.load(f)

    return bundle["model"], bundle["feature_columns"]


def predict_pair_probability(model, record_a, record_b):
    X = np.array(
        [feature_vector(record_a, record_b)],
        dtype=float,
    )

    return float(model.predict_proba(X)[0, 1])