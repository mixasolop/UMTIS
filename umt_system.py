import argparse
import json
import sqlite3
from pathlib import Path

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

import train_classifier
import train_skill_tagger
from scraper import (
    backfill_existing_jobs,
    build_model_text,
    clean_text,
    create_tables,
    delete_irrelevant_jobs,
    extract_skills,
    fill_job_skills,
    guess_role,
    guess_seniority,
    normalize_legacy_job_ids,
)


ROLE_MODEL = "job_classifier.pkl"
SENIORITY_MODEL = "job_seniority_classifier.pkl"
SKILL_MODEL = "job_skill_tagger.pkl"
DEFAULT_CANDIDATE_FILE = "sample_candidate.json"

SENIORITY_ORDER = [
    "Intern",
    "Junior",
    "Mid",
    "Senior",
    "Lead/Staff",
    "Manager/Director",
]


def prepare_database(db_path):
    connection = sqlite3.connect(db_path)
    try:
        create_tables(connection)
        normalize_legacy_job_ids(connection)
        backfill_existing_jobs(connection)

        removed_jobs = delete_irrelevant_jobs(connection)
        if removed_jobs:
            print(f"Removed irrelevant jobs: {removed_jobs}")

        skill_count = connection.execute("SELECT COUNT(*) FROM job_skills").fetchone()[0]
        if skill_count == 0:
            processed_jobs, total_skills = fill_job_skills(connection)
            print(f"Extracted skills for {processed_jobs} jobs ({total_skills} rows).")
    finally:
        connection.close()


def train_everything(db_path, allow_guess_fallback):
    prepare_database(db_path)

    print("\n=== ROLE MODEL ===")
    role_data, role_model_path = train_classifier.load_data(
        db_path=db_path,
        target="role",
        allow_guess_fallback=allow_guess_fallback,
    )
    train_classifier.train_and_save(role_data, role_model_path)

    print("\n=== SENIORITY MODEL ===")
    seniority_data, seniority_model_path = train_classifier.load_data(
        db_path=db_path,
        target="seniority",
        allow_guess_fallback=allow_guess_fallback,
    )
    train_classifier.train_and_save(seniority_data, seniority_model_path)

    print("\n=== SKILL MODEL ===")
    skill_data, _, _ = train_skill_tagger.load_data(db_path)
    train_skill_tagger.train_and_save(skill_data)


def load_candidate(candidate_json):
    file_path = candidate_json or DEFAULT_CANDIDATE_FILE
    with open(file_path, encoding="utf-8") as file_handle:
        return json.load(file_handle)


def predict_label(model, text, fallback_label):
    probabilities = model.predict_proba([text])[0]
    ranked = sorted(zip(model.classes_, probabilities), key=lambda item: item[1], reverse=True)
    label = ranked[0][0]

    if fallback_label and fallback_label != "Other" and ranked[0][1] < 0.40:
        label = fallback_label

    return label, ranked


def predict_skills(model_bundle, text):
    probabilities = model_bundle["model"].predict_proba([text])[0]
    skill_names = list(model_bundle["mlb"].classes_)
    threshold = model_bundle["threshold"]

    predicted = []
    for i, probability in enumerate(probabilities):
        if probability >= threshold:
            predicted.append(skill_names[i])

    top_skills = []
    for i in probabilities.argsort()[::-1][: model_bundle["top_k"]]:
        top_skills.append(skill_names[i])

    return predicted, top_skills


def load_jobs(db_path):
    connection = sqlite3.connect(db_path)
    try:
        jobs = pd.read_sql_query(
            """
            SELECT job_id, title, company, location, source, description_clean,
                   role_label, role_guess, seniority_label, seniority_guess
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

    grouped_skills = skill_rows.groupby("job_id")["skill"].apply(
        lambda values: sorted(set(values))
    )

    jobs["skills"] = jobs["job_id"].map(grouped_skills).apply(
        lambda value: value if isinstance(value, list) else []
    )
    jobs["role"] = jobs["role_label"].fillna("").str.strip()
    jobs.loc[jobs["role"] == "", "role"] = jobs["role_guess"].fillna("").str.strip()
    jobs["seniority"] = jobs["seniority_label"].fillna("").str.strip()
    jobs.loc[jobs["seniority"] == "", "seniority"] = jobs["seniority_guess"].fillna("").str.strip()
    jobs["text"] = [
        build_model_text(title, description_clean)
        for title, description_clean in zip(
            jobs["title"].fillna(""),
            jobs["description_clean"].fillna(""),
        )
    ]
    return jobs


def seniority_score(candidate_seniority, job_seniority):
    if not candidate_seniority or not job_seniority:
        return 0.0
    if candidate_seniority == job_seniority:
        return 1.0

    try:
        candidate_index = SENIORITY_ORDER.index(candidate_seniority)
        job_index = SENIORITY_ORDER.index(job_seniority)
    except ValueError:
        return 0.0

    gap = abs(candidate_index - job_index)
    if gap == 1:
        return 0.6
    if gap == 2:
        return 0.25
    return 0.0


def rank_jobs(candidate, jobs, top_n):
    vectorizer = TfidfVectorizer(
        lowercase=True,
        stop_words="english",
        ngram_range=(1, 2),
        max_features=12000,
    )
    job_matrix = vectorizer.fit_transform(jobs["text"])
    candidate_vector = vectorizer.transform([candidate["text"]])
    similarities = linear_kernel(candidate_vector, job_matrix).ravel()

    jobs = jobs.copy()
    jobs["text_similarity"] = similarities
    jobs = jobs.sort_values("text_similarity", ascending=False).head(50).copy()

    final_scores = []
    matched_skills_list = []
    missing_skills_list = []
    role_scores = []
    seniority_scores = []
    skill_scores = []

    candidate_skills = set(candidate["skills_predicted"])
    for _, row in jobs.iterrows():
        role_match = 1.0 if row["role"] == candidate["role"] else 0.0
        seniority_match = seniority_score(candidate["seniority"], row["seniority"])

        job_skills = set(row["skills"])
        overlap = sorted(candidate_skills & job_skills)
        missing = sorted(job_skills - candidate_skills)
        if candidate_skills and job_skills:
            skill_match = (2 * len(overlap)) / (len(candidate_skills) + len(job_skills))
        else:
            skill_match = 0.0

        final_score = (
            0.50 * row["text_similarity"]
            + 0.20 * role_match
            + 0.15 * seniority_match
            + 0.15 * skill_match
        )

        final_scores.append(final_score)
        matched_skills_list.append(overlap)
        missing_skills_list.append(missing)
        role_scores.append(role_match)
        seniority_scores.append(seniority_match)
        skill_scores.append(skill_match)

    jobs["final_score"] = final_scores
    jobs["matched_skills"] = matched_skills_list
    jobs["missing_skills"] = missing_skills_list
    jobs["role_score"] = role_scores
    jobs["seniority_score"] = seniority_scores
    jobs["skill_score"] = skill_scores

    return jobs.sort_values("final_score", ascending=False).head(top_n)


def run_demo(db_path, candidate_json, top_n):
    prepare_database(db_path)

    if not Path(ROLE_MODEL).exists() or not Path(SENIORITY_MODEL).exists() or not Path(SKILL_MODEL).exists():
        print("Models not found. Training everything first...")
        train_everything(db_path, allow_guess_fallback=False)

    candidate = load_candidate(candidate_json)
    extra_text = " ".join(
        [
            candidate.get("summary", ""),
            candidate.get("projects", ""),
            " ".join(candidate.get("skills", [])),
        ]
    ).strip()
    cleaned_text = clean_text(extra_text)
    candidate_text = build_model_text(candidate.get("title", ""), cleaned_text)

    role_model = joblib.load(ROLE_MODEL)
    seniority_model = joblib.load(SENIORITY_MODEL)
    skill_model = joblib.load(SKILL_MODEL)

    role, role_probs = predict_label(
        role_model,
        candidate_text,
        guess_role(candidate.get("title", ""), cleaned_text),
    )
    seniority, seniority_probs = predict_label(
        seniority_model,
        candidate_text,
        guess_seniority(candidate.get("title", ""), cleaned_text),
    )

    ml_skills, top_ml_skills = predict_skills(skill_model, candidate_text)
    regex_skills = extract_skills(candidate_text)
    all_skills = sorted(set(ml_skills) | set(regex_skills))

    candidate["text"] = candidate_text
    candidate["role"] = role
    candidate["seniority"] = seniority
    candidate["skills_predicted"] = all_skills

    jobs = load_jobs(db_path)
    top_jobs = rank_jobs(candidate, jobs, top_n)

    print("\nCandidate")
    print("---------")
    print(f"Title               : {candidate['title']}")
    print(f"Predicted role      : {role}")
    print(f"Predicted seniority : {seniority}")
    print(f"Predicted skills    : {', '.join(all_skills[:12]) or 'None'}")
    print(f"Top ML skills       : {', '.join(top_ml_skills[:8]) or 'None'}")
    print(f"Regex skills        : {', '.join(regex_skills[:8]) or 'None'}")
    print()

    print("Top role classes:")
    for label, score in role_probs[:3]:
        print(f"  {label}: {score:.3f}")
    print("Top seniority classes:")
    for label, score in seniority_probs[:3]:
        print(f"  {label}: {score:.3f}")
    print()

    print("Top Matches")
    print("-----------")
    for i, (_, row) in enumerate(top_jobs.iterrows(), start=1):
        print(f"{i}. {row['title']} | {row['company']} | {row['location'] or 'Unknown'}")
        print(f"   final_score     : {row['final_score']:.4f}")
        print(f"   text_similarity : {row['text_similarity']:.4f}")
        print(f"   role_score      : {row['role_score']:.2f} ({role} vs {row['role']})")
        print(f"   seniority_score : {row['seniority_score']:.2f} ({seniority} vs {row['seniority']})")
        print(f"   skill_score     : {row['skill_score']:.2f}")
        print(f"   matched_skills  : {', '.join(row['matched_skills'][:8]) or 'None'}")
        print(f"   missing_skills  : {', '.join(row['missing_skills'][:8]) or 'None'}")
        print()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", nargs="?", default="demo", choices=["demo", "train"])
    parser.add_argument("--db", default="jobs.db")
    parser.add_argument("--candidate-json", default=None)
    parser.add_argument("--top-n", type=int, default=5)
    parser.add_argument("--allow-guess-fallback", action="store_true")
    args = parser.parse_args()

    if args.command == "train":
        train_everything(args.db, allow_guess_fallback=args.allow_guess_fallback)
    else:
        run_demo(args.db, args.candidate_json, args.top_n)


if __name__ == "__main__":
    main()
