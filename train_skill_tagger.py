import argparse
import sqlite3
from collections import Counter

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MultiLabelBinarizer

from scraper import build_model_text


TEST_SIZE = 0.2
MIN_SKILL_SUPPORT = 10
MAX_FEATURES = 8000
THRESHOLD = 0.35
TOP_K = 5
MODEL_PATH = "job_skill_tagger.pkl"


def load_data(db_path):
    connection = sqlite3.connect(db_path)
    try:
        jobs = pd.read_sql_query(
            """
            SELECT job_id, company, source, title, description_clean
            FROM jobs
            """,
            connection,
        )
        skill_rows = pd.read_sql_query(
            """
            SELECT job_id, skill
            FROM job_skills
            """,
            connection,
        )
    finally:
        connection.close()

    if jobs.empty:
        raise ValueError("No jobs found in jobs table.")
    if skill_rows.empty:
        raise ValueError("No skills found in job_skills table. Run `python scraper.py extract-skills` first.")

    skill_counts = Counter(skill_rows["skill"].tolist())
    valid_skills = {
        skill_name
        for skill_name, count in skill_counts.items()
        if count >= MIN_SKILL_SUPPORT
    }
    if not valid_skills:
        raise ValueError("No skills left after filtering rare skills.")

    filtered_skill_rows = skill_rows[skill_rows["skill"].isin(valid_skills)].copy()
    grouped_skills = filtered_skill_rows.groupby("job_id")["skill"].apply(
        lambda values: sorted(set(values))
    )

    jobs["skills"] = jobs["job_id"].map(grouped_skills).apply(
        lambda value: value if isinstance(value, list) else []
    )
    jobs["text"] = [
        build_model_text(title, description_clean)
        for title, description_clean in zip(
            jobs["title"].fillna(""),
            jobs["description_clean"].fillna(""),
        )
    ]

    jobs = jobs[jobs["skills"].map(bool)].copy()
    if jobs.empty:
        raise ValueError("No jobs left with valid skill labels.")

    dropped_skills = sorted(skill_counts.keys() - valid_skills)
    return jobs, skill_counts, dropped_skills


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
                OneVsRestClassifier(
                    LogisticRegression(
                        max_iter=2000,
                        class_weight="balanced",
                    )
                ),
            ),
        ]
    )


def precision_at_k(y_true, probabilities, k):
    total = 0.0
    for true_row, prob_row in zip(y_true, probabilities):
        top_indices = np.argsort(prob_row)[::-1][:k]
        total += true_row[top_indices].sum() / k
    return total / len(y_true)


def recall_at_k(y_true, probabilities, k):
    total = 0.0
    rows = 0
    for true_row, prob_row in zip(y_true, probabilities):
        positives = int(true_row.sum())
        if positives == 0:
            continue
        top_indices = np.argsort(prob_row)[::-1][:k]
        total += true_row[top_indices].sum() / positives
        rows += 1
    return total / rows if rows else 0.0


def make_prediction_frame(split_name, frame, y_true, y_pred, probabilities, skill_names):
    rows = []
    for i, (_, row) in enumerate(frame.iterrows()):
        true_skills = [skill_names[j] for j, value in enumerate(y_true[i]) if value == 1]
        predicted_skills = [skill_names[j] for j, value in enumerate(y_pred[i]) if value == 1]
        top_skills = [skill_names[j] for j in np.argsort(probabilities[i])[::-1][:TOP_K]]

        rows.append(
            {
                "split": split_name,
                "job_id": row["job_id"],
                "company": row["company"],
                "source": row["source"],
                "title": row["title"],
                "true_skills": "|".join(true_skills),
                "predicted_skills": "|".join(predicted_skills),
                "top_k_skills": "|".join(top_skills),
            }
        )

    return pd.DataFrame(rows)


def train_and_save(dataframe):
    train_frame, test_frame = train_test_split(
        dataframe,
        test_size=TEST_SIZE,
        random_state=42,
    )

    mlb = MultiLabelBinarizer()
    y_train = mlb.fit_transform(train_frame["skills"])
    y_test = mlb.transform(test_frame["skills"])

    model = build_model()
    model.fit(train_frame["text"], y_train)

    train_prob = model.predict_proba(train_frame["text"])
    test_prob = model.predict_proba(test_frame["text"])
    train_pred = (train_prob >= THRESHOLD).astype(int)
    test_pred = (test_prob >= THRESHOLD).astype(int)

    skill_names = list(mlb.classes_)
    summary = {
        "train_rows": len(train_frame),
        "test_rows": len(test_frame),
        "skills": len(skill_names),
        "micro_f1": f1_score(y_test, test_pred, average="micro", zero_division=0),
        "macro_f1": f1_score(y_test, test_pred, average="macro", zero_division=0),
        "precision_at_5": precision_at_k(y_test, test_prob, TOP_K),
        "recall_at_5": recall_at_k(y_test, test_prob, TOP_K),
        "threshold": THRESHOLD,
    }

    report = classification_report(
        y_test,
        test_pred,
        target_names=skill_names,
        output_dict=True,
        zero_division=0,
    )
    per_skill = (
        pd.DataFrame(report)
        .transpose()
        .loc[skill_names, ["precision", "recall", "f1-score", "support"]]
        .rename(columns={"f1-score": "f1_score"})
    )
    per_skill["support"] = per_skill["support"].astype(int)
    per_skill.index.name = "skill"

    print("Skill Tagging Summary")
    print("---------------------")
    print(f"Train rows     : {summary['train_rows']}")
    print(f"Test rows      : {summary['test_rows']}")
    print(f"Skill labels   : {summary['skills']}")
    print(f"Micro F1       : {summary['micro_f1']:.4f}")
    print(f"Macro F1       : {summary['macro_f1']:.4f}")
    print(f"Precision@5    : {summary['precision_at_5']:.4f}")
    print(f"Recall@5       : {summary['recall_at_5']:.4f}")
    print(f"Threshold      : {summary['threshold']:.2f}")
    print()

    bundle = {
        "model": model,
        "mlb": mlb,
        "threshold": THRESHOLD,
        "top_k": TOP_K,
    }
    joblib.dump(bundle, MODEL_PATH)

    train_predictions = make_prediction_frame(
        "train", train_frame, y_train, train_pred, train_prob, skill_names
    )
    test_predictions = make_prediction_frame(
        "test", test_frame, y_test, test_pred, test_prob, skill_names
    )
    train_predictions.to_csv("job_skill_tagger_train_predictions.csv", index=False)
    test_predictions.to_csv("job_skill_tagger_test_predictions.csv", index=False)
    per_skill.to_csv("job_skill_tagger_per_skill_metrics.csv")
    pd.DataFrame([summary]).to_csv("job_skill_tagger_summary_metrics.csv", index=False)

    print(f"Model saved to {MODEL_PATH}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", default="jobs.db")
    args = parser.parse_args()

    dataframe, skill_counts, dropped_skills = load_data(args.db)

    print(f"Rows used for training: {len(dataframe)}")
    print("Top skill counts:")
    for skill_name, count in skill_counts.most_common(10):
        print(f"  {skill_name}: {count}")
    if dropped_skills:
        print(f"Dropped rare skills: {len(dropped_skills)}")
    print()

    train_and_save(dataframe)


if __name__ == "__main__":
    main()
