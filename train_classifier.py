import argparse
import sqlite3

import joblib
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
import warnings


def load_labeled_jobs(db_path, min_class_size, exclude_other):
    connection = sqlite3.connect(db_path)
    try:
        dataframe = pd.read_sql_query(
            """
            SELECT title, description_clean, role_label
            FROM jobs
            WHERE role_label IS NOT NULL
              AND TRIM(role_label) != ''
            """,
            connection,
        )
    finally:
        connection.close()

    if dataframe.empty:
        raise ValueError(
            "No labeled rows found in jobs table. Fill role_label and import labels first."
        )

    if exclude_other:
        dataframe = dataframe[dataframe["role_label"] != "Other"].copy()

    if dataframe.empty:
        raise ValueError(
            "No labeled rows left after filtering. Try training without --exclude-other."
        )

    dataframe["text"] = (
        dataframe["title"].fillna("") + " " + dataframe["description_clean"].fillna("")
    ).str.strip()

    class_counts = dataframe["role_label"].value_counts()
    valid_labels = class_counts[class_counts >= min_class_size].index
    filtered_dataframe = dataframe[dataframe["role_label"].isin(valid_labels)].copy()

    dropped_labels = class_counts[class_counts < min_class_size]
    if not dropped_labels.empty:
        print("Dropped rare classes:")
        for label_name, count in dropped_labels.items():
            print(f"  {label_name}: {count}")
        print()

    if filtered_dataframe["role_label"].nunique() < 2:
        raise ValueError(
            "Need at least 2 classes with enough labeled rows to train a classifier."
        )

    return filtered_dataframe


def build_model(max_features, class_weight):
    return Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    lowercase=True,
                    stop_words="english",
                    ngram_range=(1, 2),
                    max_features=max_features,
                ),
            ),
            (
                "clf",
                LogisticRegression(
                    max_iter=2000,
                    class_weight=class_weight,
                ),
            ),
        ]
    )


def train_and_evaluate(dataframe, test_size, max_features, class_weight):
    features = dataframe["text"]
    labels = dataframe["role_label"]

    x_train, x_test, y_train, y_test = train_test_split(
        features,
        labels,
        test_size=test_size,
        random_state=42,
        stratify=labels,
    )

    model = build_model(max_features=max_features, class_weight=class_weight)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        model.fit(x_train, y_train)

    predictions = model.predict(x_test)
    accuracy = accuracy_score(y_test, predictions)
    report = classification_report(y_test, predictions, zero_division=0)

    return model, accuracy, report, y_train, y_test


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", default="jobs.db")
    parser.add_argument("--model-out", default="job_classifier.pkl")
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--min-class-size", type=int, default=2)
    parser.add_argument("--max-features", type=int, default=5000)
    parser.add_argument("--exclude-other", action="store_true")
    parser.add_argument("--class-weight", default="balanced")
    args = parser.parse_args()

    dataframe = load_labeled_jobs(
        db_path=args.db,
        min_class_size=args.min_class_size,
        exclude_other=args.exclude_other,
    )

    print(f"Rows used for training: {len(dataframe)}")
    print(f"Using class_weight: {args.class_weight}")
    print(f"Exclude Other: {args.exclude_other}")
    print("Class counts:")
    for label_name, count in dataframe["role_label"].value_counts().items():
        print(f"  {label_name}: {count}")
    print()

    model, accuracy, report, y_train, y_test = train_and_evaluate(
        dataframe=dataframe,
        test_size=args.test_size,
        max_features=args.max_features,
        class_weight=args.class_weight,
    )

    print(f"Train rows: {len(y_train)}")
    print(f"Test rows: {len(y_test)}")
    print(f"Accuracy: {accuracy:.4f}")
    print()
    print(report)

    joblib.dump(model, args.model_out)
    print(f"Model saved to {args.model_out}")


if __name__ == "__main__":
    main()
