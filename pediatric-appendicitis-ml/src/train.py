"""Train KNN and MLP classifiers for the reconstructed project.

The source PDF contains cropped notebook screenshots. See README.md for
reconstruction notes and assumptions before using this script.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder, StandardScaler

TARGET = "DiagnosisByCriteria"


def make_pipeline(model):
    """Impute and encode within each fitted model to avoid train/test leakage."""
    numeric = Pipeline(
        [("imputer", SimpleImputer(strategy="median")), ("scale", StandardScaler())]
    )
    categorical = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="constant", fill_value="Unknown")),
            (
                "encode",
                OrdinalEncoder(
                    handle_unknown="use_encoded_value", unknown_value=-1
                ),
            ),
        ]
    )
    prep = ColumnTransformer(
        [("numeric", numeric, make_column_selector(dtype_include=np.number)),
         ("categorical", categorical, make_column_selector(dtype_exclude=np.number))]
    )
    return Pipeline([("preprocess", prep), ("model", model)])


def evaluate(name, estimator, X_test, y_test):
    predicted = estimator.predict(X_test)
    print(f"\n{name}")
    print(f"Accuracy:  {accuracy_score(y_test, predicted) * 100:.2f}%")
    print(f"Precision: {precision_score(y_test, predicted, average='macro', zero_division=0) * 100:.2f}%")
    print(f"Recall:    {recall_score(y_test, predicted, average='macro', zero_division=0) * 100:.2f}%")
    print(f"F1 score:  {f1_score(y_test, predicted, average='macro', zero_division=0) * 100:.2f}%")
    print("Confusion matrix (rows=true, columns=predicted):")
    print(confusion_matrix(y_test, predicted))
    print("Classification report:")
    print(classification_report(y_test, predicted, zero_division=0))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("D-7 APPENDICITIS.csv"))
    parser.add_argument("--target", default=TARGET)
    parser.add_argument("--output-dir", type=Path, default=Path("models"))
    parser.add_argument("--predict", type=Path, help="Optional separate CSV to score")
    parser.add_argument("--predictions-out", type=Path, default=Path("predictions.csv"))
    args = parser.parse_args()

    data = pd.read_csv(args.data)
    if args.target not in data.columns:
        raise SystemExit(f"Target column {args.target!r} was not found in {args.data}.")
    data = data.dropna(subset=[args.target])
    X = data.drop(columns=[args.target])
    y = data[args.target].astype(str)
    if y.nunique() < 2:
        raise SystemExit("Training requires at least two target classes.")

    # The PDF shows a 20% holdout split. A fixed seed makes this reconstruction repeatable.
    stratify = y if y.value_counts().min() >= 2 else None
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=stratify
    )

    models = {
        "KNN": make_pipeline(KNeighborsClassifier(n_neighbors=4, algorithm="ball_tree")),
        "MLP": make_pipeline(MLPClassifier(hidden_layer_sizes=(100,), max_iter=300, random_state=42)),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for name, estimator in models.items():
        estimator.fit(X_train, y_train)
        evaluate(name, estimator, X_test, y_test)
        model_path = args.output_dir / f"{name.lower()}.joblib"
        joblib.dump(estimator, model_path)
        print(f"Saved {name} pipeline to {model_path}")

    if args.predict:
        external = pd.read_csv(args.predict)
        if args.target in external.columns:
            external = external.drop(columns=[args.target])
        # Use the saved MLP flow shown in the notebook's separate test section.
        predictions = models["MLP"].predict(external)
        result = pd.read_csv(args.predict)
        result["predict"] = predictions
        result.to_csv(args.predictions_out, index=False)
        print(f"Saved {len(result)} predictions to {args.predictions_out}")


if __name__ == "__main__":
    main()
