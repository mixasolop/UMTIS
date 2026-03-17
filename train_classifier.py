import argparse
import sqlite3
import warnings

import joblib
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from scraper import build_model_text


TEST_SIZE = 0.2
MIN_CLASS_SIZE = 2
MAX_FEATURES = 5000
CLASS_WEIGHT = "balanced"
TOP_CONFUSION_LIMIT = 10


def get_target_info(target):
    if target == "role":
        return {
            "label_column": "role_label",
            "guess_column": "role_guess",
            "model_path": "job_classifier.pkl",
        }

    return {
        "label_column": "seniority_label",
        "guess_column": "seniority_guess",
        "model_path": "job_seniority_classifier.pkl",
    }


def load_data(db_path, target, allow_guess_fallback):
    target_info = get_target_info(target)
    connection = sqlite3.connect(db_path)
    try:
        dataframe = pd.read_sql_query(
            f"""
            SELECT job_id, company, source, title, description_clean,
                   {target_info['label_column']} AS label,
                   {target_info['guess_column']} AS guess_label
            FROM jobs
            """,
            connection,
        )
    finally:
        connection.close()

    dataframe["label"] = dataframe["label"].fillna("").str.strip()
    dataframe["guess_label"] = dataframe["guess_label"].fillna("").str.strip()

    labeled = dataframe[dataframe["label"] != ""].copy()
    if labeled.empty:
        if not allow_guess_fallback:
            raise ValueError(
                f"No rows found in {target_info['label_column']}. "
                f"Import labels first or run with --allow-guess-fallback."
            )

        labeled = dataframe[dataframe["guess_label"] != ""].copy()
        if labeled.empty:
            raise ValueError(f"No rows found in {target_info['guess_column']} either.")

        print(
            f"Warning: training {target} from {target_info['guess_column']} because "
            f"{target_info['label_column']} is empty."
        )
        labeled["label"] = labeled["guess_label"]

    labeled["text"] = [
        build_model_text(title, description_clean)
        for title, description_clean in zip(
            labeled["title"].fillna(""),
            labeled["description_clean"].fillna(""),
        )
    ]

    class_counts = labeled["label"].value_counts()
    valid_labels = class_counts[class_counts >= MIN_CLASS_SIZE].index
    filtered = labeled[labeled["label"].isin(valid_labels)].copy()

    dropped = class_counts[class_counts < MIN_CLASS_SIZE]
    if not dropped.empty:
        print("Dropped rare classes:")
        for label_name, count in dropped.items():
            print(f"  {label_name}: {count}")
        print()

    if filtered["label"].nunique() < 2:
        raise ValueError("Need at least 2 classes to train.")

    return filtered, target_info["model_path"]


def build_model():
    return Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    lowercase=True,
                    stop_words="english",
                    ngram_range=(1, 2),
                    max_features=MAX_FEATURES,
                ),
            ),
            (
                "clf",
                LogisticRegression(
                    max_iter=2000,
                    class_weight=CLASS_WEIGHT,
                ),
            ),
        ]
    )


def save_results(model_path, train_frame, test_frame, train_pred, test_pred, train_scores, test_scores, summary, per_class, confusion, top_confusions):
    stem = model_path.replace(".pkl", "")

    train_predictions = train_frame[
        ["job_id", "company", "source", "title", "description_clean", "label"]
    ].copy()
    train_predictions["split"] = "train"
    train_predictions = train_predictions.rename(columns={"label": "true_label"})
    train_predictions["predicted_label"] = train_pred
    train_predictions["is_correct"] = (
        train_predictions["true_label"] == train_predictions["predicted_label"]
    )
    train_predictions["predicted_score"] = train_scores.round(6)

    test_predictions = test_frame[
        ["job_id", "company", "source", "title", "description_clean", "label"]
    ].copy()
    test_predictions["split"] = "test"
    test_predictions = test_predictions.rename(columns={"label": "true_label"})
    test_predictions["predicted_label"] = test_pred
    test_predictions["is_correct"] = (
        test_predictions["true_label"] == test_predictions["predicted_label"]
    )
    test_predictions["predicted_score"] = test_scores.round(6)

    test_errors = test_predictions[~test_predictions["is_correct"]].copy()
    test_errors = test_errors.sort_values(
        by=["predicted_score", "true_label", "predicted_label"],
        ascending=[False, True, True],
    )

    train_predictions.to_csv(f"{stem}_train_predictions.csv", index=False)
    test_predictions.to_csv(f"{stem}_test_predictions.csv", index=False)
    test_errors.to_csv(f"{stem}_test_errors.csv", index=False)
    per_class.to_csv(f"{stem}_per_class_metrics.csv")
    confusion.to_csv(f"{stem}_confusion_matrix.csv")
    top_confusions.to_csv(f"{stem}_top_confusion_pairs.csv", index=False)
    pd.DataFrame([summary]).to_csv(f"{stem}_summary_metrics.csv", index=False)


def train_and_save(dataframe, model_path):
    train_frame, test_frame = train_test_split(
        dataframe,
        test_size=TEST_SIZE,
        random_state=42,
        stratify=dataframe["label"],
    )

    model = build_model()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        model.fit(train_frame["text"], train_frame["label"])

    train_pred = model.predict(train_frame["text"])
    test_pred = model.predict(test_frame["text"])
    train_scores = model.predict_proba(train_frame["text"]).max(axis=1)
    test_scores = model.predict_proba(test_frame["text"]).max(axis=1)

    labels = list(model.classes_)
    report = classification_report(
        test_frame["label"],
        test_pred,
        labels=labels,
        output_dict=True,
        zero_division=0,
    )

    per_class = (
        pd.DataFrame(report)
        .transpose()
        .loc[labels, ["precision", "recall", "f1-score", "support"]]
        .rename(columns={"f1-score": "f1_score"})
    )
    per_class["support"] = per_class["support"].astype(int)
    per_class.index.name = "label"

    confusion = pd.DataFrame(
        confusion_matrix(test_frame["label"], test_pred, labels=labels),
        index=labels,
        columns=labels,
    )
    confusion.index.name = "true_label"
    confusion.columns.name = "predicted_label"

    pairs = []
    for true_label in labels:
        for predicted_label in labels:
            if true_label == predicted_label:
                continue
            count = int(confusion.loc[true_label, predicted_label])
            if count > 0:
                pairs.append(
                    {
                        "true_label": true_label,
                        "predicted_label": predicted_label,
                        "count": count,
                    }
                )
    top_confusions = pd.DataFrame(pairs)
    if not top_confusions.empty:
        top_confusions = top_confusions.sort_values(
            by=["count", "true_label", "predicted_label"],
            ascending=[False, True, True],
        ).head(TOP_CONFUSION_LIMIT)

    summary = {
        "train_rows": len(train_frame),
        "test_rows": len(test_frame),
        "accuracy": accuracy_score(test_frame["label"], test_pred),
        "macro_f1": f1_score(test_frame["label"], test_pred, average="macro"),
        "weighted_f1": f1_score(test_frame["label"], test_pred, average="weighted"),
    }

    print("Evaluation Summary")
    print("------------------")
    print(f"Train rows   : {summary['train_rows']}")
    print(f"Test rows    : {summary['test_rows']}")
    print(f"Accuracy     : {summary['accuracy']:.4f}")
    print(f"Macro F1     : {summary['macro_f1']:.4f}")
    print(f"Weighted F1  : {summary['weighted_f1']:.4f}")
    print()

    print("Per-Class Metrics")
    print("-----------------")
    printable = per_class.reset_index().copy()
    for column_name in ["precision", "recall", "f1_score"]:
        printable[column_name] = printable[column_name].map(lambda value: f"{value:.4f}")
    print(printable.to_string(index=False))
    print()

    print("Confusion Matrix")
    print("----------------")
    print("Rows = true labels, columns = predicted labels")
    print(confusion.to_string())
    print()

    print("Top Confusion Pairs")
    print("-------------------")
    if top_confusions.empty:
        print("No off-diagonal confusions found in the test split.")
    else:
        print(top_confusions.to_string(index=False))
    print()

    joblib.dump(model, model_path)
    save_results(
        model_path,
        train_frame,
        test_frame,
        train_pred,
        test_pred,
        train_scores,
        test_scores,
        summary,
        per_class,
        confusion,
        top_confusions,
    )
    print(f"Model saved to {model_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", default="jobs.db")
    parser.add_argument("--target", choices=["role", "seniority"], default="role")
    parser.add_argument("--allow-guess-fallback", action="store_true")
    args = parser.parse_args()

    dataframe, model_path = load_data(
        db_path=args.db,
        target=args.target,
        allow_guess_fallback=args.allow_guess_fallback,
    )

    print(f"Training target: {args.target}")
    print(f"Rows used for training: {len(dataframe)}")
    print("Class counts:")
    for label_name, count in dataframe["label"].value_counts().items():
        print(f"  {label_name}: {count}")
    print()

    train_and_save(dataframe, model_path)


if __name__ == "__main__":
    main()
