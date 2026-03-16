import argparse
import sqlite3
from pathlib import Path

import joblib
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
import warnings


TOP_CONFUSION_LIMIT = 10


def load_labeled_jobs(db_path, min_class_size, exclude_other):
    connection = sqlite3.connect(db_path)
    try:
        dataframe = pd.read_sql_query(
            """
            SELECT job_id, company, source, title, description_clean, role_label
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


def get_prediction_scores(model, features):
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(features)
        return probabilities.max(axis=1)
    return None


def build_prediction_frame(split_name, dataframe, predicted_labels, predicted_scores):
    prediction_frame = dataframe[
        ["job_id", "company", "source", "title", "description_clean", "role_label"]
    ].copy()
    prediction_frame.insert(0, "split", split_name)
    prediction_frame = prediction_frame.rename(columns={"role_label": "true_label"})
    prediction_frame["predicted_label"] = predicted_labels
    prediction_frame["is_correct"] = (
        prediction_frame["true_label"] == prediction_frame["predicted_label"]
    )

    if predicted_scores is not None:
        prediction_frame["predicted_score"] = predicted_scores.round(6)

    return prediction_frame


def build_per_class_metrics(report_dict, label_order):
    per_class_metrics = (
        pd.DataFrame(report_dict)
        .transpose()
        .loc[label_order, ["precision", "recall", "f1-score", "support"]]
        .rename(columns={"f1-score": "f1_score"})
    )
    per_class_metrics["support"] = per_class_metrics["support"].astype(int)
    per_class_metrics.index.name = "label"
    return per_class_metrics


def build_confusion_frame(y_true, predictions, label_order):
    confusion = pd.DataFrame(
        confusion_matrix(y_true, predictions, labels=label_order),
        index=label_order,
        columns=label_order,
    )
    confusion.index.name = "true_label"
    confusion.columns.name = "predicted_label"
    return confusion


def find_top_confusion_pairs(confusion_frame, limit=TOP_CONFUSION_LIMIT):
    pairs = []

    for true_label in confusion_frame.index:
        for predicted_label in confusion_frame.columns:
            if true_label == predicted_label:
                continue

            count = int(confusion_frame.loc[true_label, predicted_label])
            if count > 0:
                pairs.append(
                    {
                        "true_label": true_label,
                        "predicted_label": predicted_label,
                        "count": count,
                    }
                )

    if not pairs:
        return pd.DataFrame(columns=["true_label", "predicted_label", "count"])

    return pd.DataFrame(pairs).sort_values(
        by=["count", "true_label", "predicted_label"],
        ascending=[False, True, True],
    ).head(limit)


def format_float_columns(dataframe, columns):
    printable = dataframe.copy()
    for column_name in columns:
        if column_name in printable.columns:
            printable[column_name] = printable[column_name].map(
                lambda value: f"{value:.4f}"
            )
    return printable


def print_section(title):
    print(title)
    print("-" * len(title))


def print_summary(summary_metrics):
    print_section("Evaluation Summary")
    print(f"Train rows   : {summary_metrics['train_rows']}")
    print(f"Test rows    : {summary_metrics['test_rows']}")
    print(f"Accuracy     : {summary_metrics['accuracy']:.4f}")
    print(f"Macro F1     : {summary_metrics['macro_f1']:.4f}")
    print(f"Weighted F1  : {summary_metrics['weighted_f1']:.4f}")
    print()


def print_per_class_metrics(per_class_metrics):
    print_section("Per-Class Metrics")
    printable = format_float_columns(
        per_class_metrics.reset_index(),
        ["precision", "recall", "f1_score"],
    )
    print(printable.to_string(index=False))
    print()


def print_confusion_matrix(confusion_frame):
    print_section("Confusion Matrix")
    print("Rows = true labels, columns = predicted labels")
    print(confusion_frame.to_string())
    print()


def print_top_confusions(top_confusions):
    print_section("Top Confusion Pairs")
    if top_confusions.empty:
        print("No off-diagonal confusions found in the test split.")
    else:
        print(top_confusions.to_string(index=False))
    print()


def resolve_artifact_paths(model_out, artifacts_dir):
    model_path = Path(model_out)
    output_dir = Path(artifacts_dir) if artifacts_dir else model_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = model_path.stem

    return {
        "train_predictions": output_dir / f"{stem}_train_predictions.csv",
        "test_predictions": output_dir / f"{stem}_test_predictions.csv",
        "per_class_metrics": output_dir / f"{stem}_per_class_metrics.csv",
        "confusion_matrix": output_dir / f"{stem}_confusion_matrix.csv",
        "top_confusions": output_dir / f"{stem}_top_confusion_pairs.csv",
        "summary_metrics": output_dir / f"{stem}_summary_metrics.csv",
    }


def save_artifacts(evaluation_result, artifact_paths):
    evaluation_result["train_predictions"].to_csv(
        artifact_paths["train_predictions"], index=False
    )
    evaluation_result["test_predictions"].to_csv(
        artifact_paths["test_predictions"], index=False
    )
    evaluation_result["per_class_metrics"].to_csv(artifact_paths["per_class_metrics"])
    evaluation_result["confusion_matrix"].to_csv(artifact_paths["confusion_matrix"])
    evaluation_result["top_confusions"].to_csv(
        artifact_paths["top_confusions"], index=False
    )
    pd.DataFrame([evaluation_result["summary_metrics"]]).to_csv(
        artifact_paths["summary_metrics"], index=False
    )


def train_and_evaluate(dataframe, test_size, max_features, class_weight):
    train_frame, test_frame = train_test_split(
        dataframe,
        test_size=test_size,
        random_state=42,
        stratify=dataframe["role_label"],
    )

    x_train = train_frame["text"]
    y_train = train_frame["role_label"]
    x_test = test_frame["text"]
    y_test = test_frame["role_label"]

    model = build_model(max_features=max_features, class_weight=class_weight)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        model.fit(x_train, y_train)

    train_predictions = model.predict(x_train)
    test_predictions = model.predict(x_test)
    label_order = list(model.classes_)

    report_dict = classification_report(
        y_test,
        test_predictions,
        labels=label_order,
        output_dict=True,
        zero_division=0,
    )
    per_class_metrics = build_per_class_metrics(report_dict, label_order)
    confusion_frame = build_confusion_frame(y_test, test_predictions, label_order)
    top_confusions = find_top_confusion_pairs(confusion_frame)

    train_scores = get_prediction_scores(model, x_train)
    test_scores = get_prediction_scores(model, x_test)

    summary_metrics = {
        "train_rows": len(train_frame),
        "test_rows": len(test_frame),
        "accuracy": accuracy_score(y_test, test_predictions),
        "macro_f1": f1_score(y_test, test_predictions, average="macro"),
        "weighted_f1": f1_score(y_test, test_predictions, average="weighted"),
    }

    return {
        "model": model,
        "summary_metrics": summary_metrics,
        "per_class_metrics": per_class_metrics,
        "confusion_matrix": confusion_frame,
        "top_confusions": top_confusions,
        "train_predictions": build_prediction_frame(
            "train", train_frame, train_predictions, train_scores
        ),
        "test_predictions": build_prediction_frame(
            "test", test_frame, test_predictions, test_scores
        ),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", default="jobs.db")
    parser.add_argument("--model-out", default="job_classifier.pkl")
    parser.add_argument("--artifacts-dir", default=None)
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

    evaluation_result = train_and_evaluate(
        dataframe=dataframe,
        test_size=args.test_size,
        max_features=args.max_features,
        class_weight=args.class_weight,
    )

    print_summary(evaluation_result["summary_metrics"])
    print_per_class_metrics(evaluation_result["per_class_metrics"])
    print_confusion_matrix(evaluation_result["confusion_matrix"])
    print_top_confusions(evaluation_result["top_confusions"])

    artifact_paths = resolve_artifact_paths(args.model_out, args.artifacts_dir)
    save_artifacts(evaluation_result, artifact_paths)

    Path(args.model_out).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(evaluation_result["model"], args.model_out)
    print(f"Model saved to {args.model_out}")
    print("Saved artifacts:")
    for artifact_name, artifact_path in artifact_paths.items():
        print(f"  {artifact_name}: {artifact_path}")


if __name__ == "__main__":
    main()
