## Universal Multimodal Talent Intelligence System

ML project for job clarifications and candidate-job matching.

The project includes:

- role classification
- seniority prediction
- multi-label skill tagging
- retrieval and ranking

## What it Consist of

### 1. Role Classification

- input: job title + cleaned description
- model: TF-IDF + Logistic Regression
- output: role label

### 2. Seniority Prediction

- input: job title + cleaned description
- model: TF-IDF + Logistic Regression
- output: seniority label

### 3. Skill Tagging

- input: job title + cleaned description
- task: multi-label prediction
- model: TF-IDF + OneVsRest Logistic Regression
- metrics: micro F1, macro F1, precision@k, recall@k

### 4. Candidate-Job Matching

- candidate profile -> predicted role
- candidate profile -> predicted seniority
- candidate profile -> predicted skills
- top-N retrieval by TF-IDF similarity
- final hybrid ranking with:
  - text similarity
  - role match
  - seniority match
  - skill overlap

## How to launch

Run:

```powershell
python umt_system.py
```

This runs the program with the following plan:

1. loads the sample candidate
2. predicts role, seniority, and skills
3. retrieves similar jobs
4. re-ranks them with a score
5. prints the top matches with explaML-first portfolio project for analyzing job postings and matching candidate profiles to relevant roles.

This project is built to show practical ML engineering skills, not just a generic AI wrapper.  
It combines:

- supervised NLP
- multi-label classification
- skill extraction and tagging
- retrieval
- hybrid ranking
- explainable matching

## Project Goal

The system takes job postings, learns structured signals from them, and then uses those signals to match a candidate profile against many jobs.

The current pipeline does four main things:

1. predicts the job role
2. predicts the seniority level
3. predicts multiple required skills
4. retrieves and re-ranks the best matching jobs for a candidate

## ML Part

### 1. Role Classification
Predicts a single role label from job title + cleaned description.

Examples:
- Backend
- Data Analyst
- Data Scientist
- ML Engineer
- DevOps
- QA
- Other

Model:
- TF-IDF
- Logistic Regression

### 2. Seniority Prediction
Predicts a single seniority label from job title + cleaned description.

Examples:
- Intern
- Junior
- Mid
- Senior
- Lead/Staff
- Manager/Director

Model:
- TF-IDF
- Logistic Regression

### 3. Skill Tagging
Predicts multiple skills from one job posting.

Examples:
- Python
- SQL
- Docker
- AWS
- PyTorch
- scikit-learn
- NLP

Task type:
- multi-label classification

Model:
- TF-IDF
- One-vs-Rest Logistic Regression

Metrics:
- Micro F1
- Macro F1
- Precision@5
- Recall@5

### 4. Candidate-Job Matching
Matches one candidate profile against all jobs in the database.

The ranking uses:
- text similarity
- predicted role match
- predicted seniority match
- predicted skill overlap

This makes the system more realistic than a plain cosine similarity demo.

## Current Project Acting Plan

The pipeline works like this:

1. scrape and clean job postings
2. store them in SQLite
3. label role and seniority
4. extract skills into a `job_skills` table
5. train role model
6. train seniority model
7. train multi-label skill tagger
8. build a candidate profile
9. predict candidate role, seniority, and skills
10. retrieve similar jobs
11. re-rank them with a hybrid score
12. show top matches with explanations

## Tech Stack

- Python
- pandas
- numpy
- scikit-learn
- SQLite

## Main Files

- `scraper.py`  
  scraping, cleaning, skill extraction, DB

- `train_classifier.py`  
  trains role and seniority classifiers

- `train_skill_tagger.py`  
  trains the skill tagging model

- `umt_system.py`  
  full training file

- `trained_model.py`  
  quick single-example prediction example

- `jobs.db`  
  SQLite database with jobs and skills

## How to Run

To scrape jobs
```powershell
python scraper.py scrape
```
## Full training

To retrain everything:

```powershell
python umt_system.py train
```

This trains:

- `job_classifier.pkl`
- `job_seniority_classifier.pkl`
- `job_skill_tagger.pkl`


## What it shows

- NLP
- multi-label classification
- class imbalance handling
- evaluation with LR/OneVsRest LR
- retrieval and ranking
- matchings
