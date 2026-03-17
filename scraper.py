import argparse
import csv
import html
import json
import re
import sqlite3
import sys
import urllib.request

REMOTEOK_API_URL = "https://remoteok.com/api"
GREENHOUSE_API_TEMPLATE = "https://boards-api.greenhouse.io/v1/boards/{board}/jobs?content=true"
DEFAULT_GREENHOUSE_BOARDS = [
    "stripe",
    "cloudflare",
    "datadog",
    "mongodb",
    "fivetran",
    "coinbase",
]

ROLE_CLASSES = [
    "Backend",
    "Data Analyst",
    "Data Scientist",
    "ML Engineer",
    "DevOps",
    "QA",
    "Other",
]

SENIORITY_CLASSES = [
    "Intern",
    "Junior",
    "Mid",
    "Senior",
    "Lead/Staff",
    "Manager/Director",
    "Other",
]

ROLE_RULES = {
    "ML Engineer": [
        r"\bml engineer\b",
        r"\bmachine learning engineer\b",
        r"\bmlops\b",
        r"\bllm\b",
        r"\bpytorch\b",
        r"\btensorflow\b",
        r"\bmodel serving\b",
        r"\bretrieval\b",
        r"\bcomputer vision\b",
        r"\bnlp\b",
    ],
    "Data Scientist": [
        r"\bdata scientist\b",
        r"\bapplied scientist\b",
        r"\bexperimentation\b",
        r"\bstatistical\b",
        r"\bcausal inference\b",
        r"\bpredictive modeling\b",
    ],
    "Data Analyst": [
        r"\bdata analyst\b",
        r"\bbusiness intelligence\b",
        r"\bbi analyst\b",
        r"\btableau\b",
        r"\bpower bi\b",
        r"\bdashboard\b",
        r"\banalytics\b",
    ],
    "DevOps": [
        r"\bdevops\b",
        r"\bsite reliability\b",
        r"\bsre\b",
        r"\bplatform engineer\b",
        r"\binfrastructure\b",
        r"\bkubernetes\b",
        r"\bterraform\b",
        r"\bci/cd\b",
        r"\bobservability\b",
    ],
    "QA": [
        r"\bqa\b",
        r"\bquality assurance\b",
        r"\btest automation\b",
        r"\bautomation engineer\b",
        r"\bselenium\b",
        r"\bcypress\b",
        r"\bplaywright\b",
    ],
    "Backend": [
        r"\bbackend\b",
        r"\bback[- ]end\b",
        r"\bsoftware engineer\b",
        r"\bsoftware developer\b",
        r"\bpython developer\b",
        r"\bjava developer\b",
        r"\bnode(?:\.js)? developer\b",
        r"\bfull[- ]stack\b",
        r"\bapi\b",
        r"\bserver[- ]side\b",
    ],
}

SENIORITY_RULES = {
    "Intern": [
        r"\bintern\b",
        r"\binternship\b",
        r"\bworking student\b",
        r"\bco-op\b",
    ],
    "Junior": [
        r"\bjunior\b",
        r"\bjr\b",
        r"\bgraduate\b",
        r"\bnew grad\b",
        r"\bentry level\b",
        r"\bassociate\b",
    ],
    "Senior": [
        r"\bsenior\b",
        r"\bsr\b",
    ],
    "Lead/Staff": [
        r"\blead\b",
        r"\bstaff\b",
        r"\bprincipal\b",
        r"\barchitect\b",
    ],
    "Manager/Director": [
        r"\bmanager\b",
        r"\bdirector\b",
        r"\bhead\b",
        r"\bvice president\b",
        r"\bvp\b",
    ],
}

SKILL_PATTERNS = {
    "Python": [r"\bpython\b"],
    "SQL": [r"\bsql\b"],
    "JavaScript": [r"\bjavascript\b", r"\bjs\b"],
    "TypeScript": [r"\btypescript\b", r"\bts\b"],
    "Java": [r"\bjava\b"],
    "C++": [r"\bc\+\+\b"],
    "C#": [r"\bc#\b", r"\bcsharp\b"],
    "Go": [r"\bgolang\b", r"\bgo\b"],
    "Rust": [r"\brust\b"],
    "PostgreSQL": [r"\bpostgres\b", r"\bpostgresql\b"],
    "MySQL": [r"\bmysql\b"],
    "MongoDB": [r"\bmongodb\b", r"\bmongo\b"],
    "Redis": [r"\bredis\b"],
    "Docker": [r"\bdocker\b"],
    "Kubernetes": [r"\bkubernetes\b", r"\bk8s\b"],
    "AWS": [r"\baws\b", r"\bamazon web services\b"],
    "GCP": [r"\bgcp\b", r"\bgoogle cloud\b"],
    "Azure": [r"\bazure\b"],
    "Git": [r"\bgit\b"],
    "Linux": [r"\blinux\b"],
    "Bash": [r"\bbash\b", r"\bshell scripting\b"],
    "Pandas": [r"\bpandas\b"],
    "NumPy": [r"\bnumpy\b"],
    "scikit-learn": [r"\bsklearn\b", r"\bscikit-learn\b"],
    "TensorFlow": [r"\btensorflow\b"],
    "PyTorch": [r"\bpytorch\b"],
    "Spark": [r"\bspark\b", r"\bapache spark\b"],
    "Airflow": [r"\bairflow\b", r"\bapache airflow\b"],
    "dbt": [r"\bdbt\b"],
    "Tableau": [r"\btableau\b"],
    "Power BI": [r"\bpower bi\b"],
    "Excel": [r"\bexcel\b"],
    "Statistics": [r"\bstatistics\b", r"\bstatistical\b"],
    "Machine Learning": [r"\bmachine learning\b", r"\bml\b"],
    "Deep Learning": [r"\bdeep learning\b"],
    "NLP": [r"\bnlp\b", r"\bnatural language processing\b"],
    "Computer Vision": [r"\bcomputer vision\b"],
    "MLOps": [r"\bmlops\b"],
    "CI/CD": [r"\bci/cd\b", r"\bcontinuous integration\b"],
    "Terraform": [r"\bterraform\b"],
    "Django": [r"\bdjango\b"],
    "Flask": [r"\bflask\b"],
    "FastAPI": [r"\bfastapi\b"],
    "React": [r"\breact\b"],
    "Node.js": [r"\bnode\.js\b", r"\bnodejs\b"],
    "Kafka": [r"\bkafka\b", r"\bapache kafka\b"],
}

BOILERPLATE_MARKERS = [
    "Pay Transparency Notice",
    "Global Data Privacy Notice",
    "Commitment to Equal Opportunity",
    "Equal Employment Opportunity",
    "Equal Opportunity Employer",
    "To request a reasonable accommodation",
    "We are committed to ensuring that all candidates have an equal opportunity",
    "The target annual base salary for this position can range",
    "MongoDB's base salary range",
    "MongoDB’s base salary range",
]


def create_tables(connection):
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS jobs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            job_id TEXT UNIQUE,
            title TEXT NOT NULL,
            company TEXT NOT NULL,
            location TEXT,
            description TEXT,
            date TEXT,
            source TEXT NOT NULL
        )
        """
    )

    ensure_column(connection, "jobs", "job_url", "TEXT")
    ensure_column(connection, "jobs", "description_clean", "TEXT")
    ensure_column(connection, "jobs", "role_guess", "TEXT")
    ensure_column(connection, "jobs", "role_label", "TEXT")
    ensure_column(connection, "jobs", "seniority_guess", "TEXT")
    ensure_column(connection, "jobs", "seniority_label", "TEXT")

    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS job_skills (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            job_id TEXT NOT NULL,
            skill TEXT NOT NULL,
            UNIQUE(job_id, skill)
        )
        """
    )
    connection.commit()


def ensure_column(connection, table_name, column_name, column_type):
    columns = {
        row[1]
        for row in connection.execute(f"PRAGMA table_info({table_name})").fetchall()
    }
    if column_name not in columns:
        connection.execute(
            f"ALTER TABLE {table_name} ADD COLUMN {column_name} {column_type}"
        )


def normalize_legacy_job_ids(connection):
    connection.execute(
        """
        UPDATE jobs
        SET job_id = 'remoteok:' || job_id
        WHERE source = 'remoteok'
          AND job_id NOT LIKE 'remoteok:%'
        """
    )
    connection.commit()


def backfill_existing_jobs(connection):
    rows = connection.execute(
        """
        SELECT job_id, title, description
        FROM jobs
        WHERE description_clean IS NULL
           OR role_guess IS NULL
           OR seniority_guess IS NULL
           OR TRIM(COALESCE(description_clean, '')) = ''
           OR TRIM(COALESCE(role_guess, '')) = ''
           OR TRIM(COALESCE(seniority_guess, '')) = ''
        """
    ).fetchall()

    for job_id, title, description in rows:
        description_clean = clean_text(description or "")
        role_guess = guess_role(title or "", description_clean)
        seniority_guess = guess_seniority(title or "", description_clean)
        connection.execute(
            """
            UPDATE jobs
            SET description_clean = ?, role_guess = ?, seniority_guess = ?
            WHERE job_id = ?
            """,
            (description_clean, role_guess, seniority_guess, job_id),
        )

    connection.commit()


def delete_irrelevant_jobs(connection):
    rows = connection.execute(
        """
        SELECT job_id
        FROM jobs
        WHERE role_guess IS NULL
           OR TRIM(role_guess) = ''
           OR role_guess = 'Other'
        """
    ).fetchall()
    irrelevant_ids = [row[0] for row in rows]

    if not irrelevant_ids:
        return 0

    for chunk_start in range(0, len(irrelevant_ids), 500):
        chunk = irrelevant_ids[chunk_start : chunk_start + 500]
        placeholders = ",".join("?" for _ in chunk)
        connection.execute(
            f"DELETE FROM job_skills WHERE job_id IN ({placeholders})",
            chunk,
        )
        connection.execute(
            f"DELETE FROM jobs WHERE job_id IN ({placeholders})",
            chunk,
        )

    connection.commit()
    return len(irrelevant_ids)


def request_json(url):
    request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(request, timeout=30) as response:
        return json.load(response)


def fetch_remoteok_jobs():
    data = request_json(REMOTEOK_API_URL)
    jobs = []

    for item in data[1:]:
        title = (item.get("position") or "").strip()
        raw_description = item.get("description") or ""
        description_clean = clean_text(raw_description)
        role_guess = guess_role(title, description_clean)
        seniority_guess = guess_seniority(title, description_clean)

        if role_guess == "Other":
            continue

        jobs.append(
            {
                "job_id": f"remoteok:{item.get('id')}",
                "title": title,
                "company": (item.get("company") or "").strip() or "Unknown",
                "location": (item.get("location") or "").strip() or None,
                "description": raw_description,
                "description_clean": description_clean,
                "date": item.get("date") or "",
                "source": "remoteok",
                "job_url": item.get("url") or item.get("apply_url") or "",
                "role_guess": role_guess,
                "seniority_guess": seniority_guess,
            }
        )

    return jobs


def fetch_greenhouse_jobs(board):
    payload = request_json(GREENHOUSE_API_TEMPLATE.format(board=board))
    jobs = []

    for item in payload.get("jobs", []):
        title = (item.get("title") or "").strip()
        raw_description = item.get("content") or ""
        description_clean = clean_text(raw_description)
        role_guess = guess_role(title, description_clean)
        seniority_guess = guess_seniority(title, description_clean)

        if role_guess == "Other":
            continue

        location = None
        if isinstance(item.get("location"), dict):
            location = (item["location"].get("name") or "").strip() or None

        jobs.append(
            {
                "job_id": f"greenhouse:{board}:{item.get('id')}",
                "title": title,
                "company": (item.get("company_name") or board).strip(),
                "location": location,
                "description": raw_description,
                "description_clean": description_clean,
                "date": item.get("updated_at") or item.get("first_published") or "",
                "source": "greenhouse",
                "job_url": item.get("absolute_url") or "",
                "role_guess": role_guess,
                "seniority_guess": seniority_guess,
            }
        )

    return jobs


def guess_role(title, description_clean):
    text = f"{title}\n{description_clean}".lower()
    best_role = "Other"
    best_score = 0

    for role_name, patterns in ROLE_RULES.items():
        score = 0
        for pattern in patterns:
            if re.search(pattern, text):
                score += 1
        if score > best_score:
            best_role = role_name
            best_score = score

    return best_role


def guess_seniority(title, description_clean):
    title_text = (title or "").lower()
    description_text = (description_clean or "").lower()

    for seniority_name in [
        "Manager/Director",
        "Lead/Staff",
        "Senior",
        "Junior",
        "Intern",
    ]:
        for pattern in SENIORITY_RULES[seniority_name]:
            if re.search(pattern, title_text):
                return seniority_name

    for seniority_name in ["Intern", "Junior"]:
        for pattern in SENIORITY_RULES[seniority_name]:
            if re.search(pattern, description_text):
                return seniority_name

    return "Mid"


def trim_boilerplate_sections(text):
    earliest_index = None
    lowercase_text = text.lower()

    for marker in BOILERPLATE_MARKERS:
        marker_index = lowercase_text.find(marker.lower())
        if marker_index == -1 or marker_index < 500:
            continue
        if earliest_index is None or marker_index < earliest_index:
            earliest_index = marker_index

    if earliest_index is not None:
        return text[:earliest_index]

    return text


def clean_text(text):
    text = html.unescape(text or "")
    text = text.replace("\xa0", " ")
    text = re.sub(r"<[^>]+>", " ", text)
    text = trim_boilerplate_sections(text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def deduplicate_jobs(jobs):
    unique_jobs = {}
    for job in jobs:
        unique_jobs[job["job_id"]] = job
    return list(unique_jobs.values())


def save_jobs(connection, jobs):
    existing_ids = get_existing_job_ids(connection, [job["job_id"] for job in jobs])
    inserted = 0
    updated = 0

    for job in jobs:
        if job["job_id"] in existing_ids:
            updated += 1
        else:
            inserted += 1

        connection.execute(
            """
            INSERT INTO jobs (
                job_id, title, company, location, description, description_clean,
                date, source, job_url, role_guess, seniority_guess
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(job_id) DO UPDATE SET
                title = excluded.title,
                company = excluded.company,
                location = excluded.location,
                description = excluded.description,
                description_clean = excluded.description_clean,
                date = excluded.date,
                source = excluded.source,
                job_url = excluded.job_url,
                role_guess = excluded.role_guess,
                seniority_guess = excluded.seniority_guess
            """,
            (
                job["job_id"],
                job["title"],
                job["company"],
                job["location"],
                job["description"],
                job["description_clean"],
                job["date"],
                job["source"],
                job["job_url"],
                job["role_guess"],
                job["seniority_guess"],
            ),
        )

    connection.commit()
    return inserted, updated


def get_existing_job_ids(connection, job_ids):
    existing_ids = set()
    if not job_ids:
        return existing_ids

    for chunk_start in range(0, len(job_ids), 500):
        chunk = job_ids[chunk_start : chunk_start + 500]
        placeholders = ",".join("?" for _ in chunk)
        rows = connection.execute(
            f"SELECT job_id FROM jobs WHERE job_id IN ({placeholders})",
            chunk,
        ).fetchall()
        for row in rows:
            existing_ids.add(row[0])

    return existing_ids


def export_labels(connection, csv_path, limit, include_labeled):
    if include_labeled:
        rows = connection.execute(
            """
            SELECT job_id, title, company, location, date, source, role_guess, role_label,
                   seniority_guess, seniority_label, description_clean
            FROM jobs
            ORDER BY id DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()
    else:
        rows = connection.execute(
            """
            SELECT job_id, title, company, location, date, source, role_guess, role_label,
                   seniority_guess, seniority_label, description_clean
            FROM jobs
            WHERE role_label IS NULL OR TRIM(role_label) = ''
               OR seniority_label IS NULL OR TRIM(seniority_label) = ''
            ORDER BY id DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()

    with open(csv_path, "w", newline="", encoding="utf-8") as file_handle:
        writer = csv.writer(file_handle)
        writer.writerow(
            [
                "job_id",
                "title",
                "company",
                "location",
                "date",
                "source",
                "role_guess",
                "role_label",
                "seniority_guess",
                "seniority_label",
                "description_clean",
            ]
        )
        writer.writerows(rows)

    return len(rows)


def import_labels(connection, csv_path):
    updated = 0

    with open(csv_path, newline="", encoding="utf-8") as file_handle:
        reader = csv.DictReader(file_handle)
        for row in reader:
            role_label = (row.get("role_label") or "").strip()
            seniority_label = (row.get("seniority_label") or "").strip()

            if not role_label and not seniority_label:
                continue

            if role_label and role_label not in ROLE_CLASSES:
                continue

            if seniority_label and seniority_label not in SENIORITY_CLASSES:
                continue

            connection.execute(
                """
                UPDATE jobs
                SET role_label = COALESCE(NULLIF(?, ''), role_label),
                    seniority_label = COALESCE(NULLIF(?, ''), seniority_label)
                WHERE job_id = ?
                """,
                (role_label, seniority_label, row["job_id"]),
            )
            updated += 1

    connection.commit()
    return updated


def extract_skills(text):
    text = text.lower()
    found_skills = []

    for canonical_skill, patterns in SKILL_PATTERNS.items():
        for pattern in patterns:
            if re.search(pattern, text):
                found_skills.append(canonical_skill)
                break

    return sorted(found_skills)


def fill_job_skills(connection, limit=None):
    query = """
        SELECT job_id, title, description_clean
        FROM jobs
        ORDER BY id DESC
    """
    params = ()

    if limit is not None:
        query += " LIMIT ?"
        params = (limit,)

    rows = connection.execute(query, params).fetchall()
    processed_jobs = 0
    total_skills = 0

    for job_id, title, description_clean in rows:
        text = f"{title}\n{description_clean or ''}"
        skills = extract_skills(text)

        connection.execute("DELETE FROM job_skills WHERE job_id = ?", (job_id,))
        for skill in skills:
            connection.execute(
                "INSERT OR IGNORE INTO job_skills (job_id, skill) VALUES (?, ?)",
                (job_id, skill),
            )
        processed_jobs += 1
        total_skills += len(skills)

    connection.commit()
    return processed_jobs, total_skills


def show_stats(connection):
    job_count = connection.execute("SELECT COUNT(*) FROM jobs").fetchone()[0]
    skill_count = connection.execute("SELECT COUNT(*) FROM job_skills").fetchone()[0]

    print(f"Jobs in DB: {job_count}")
    print(f"Extracted skill rows: {skill_count}")
    print()

    print("Jobs by source:")
    for source, count in connection.execute(
        "SELECT source, COUNT(*) FROM jobs GROUP BY source ORDER BY COUNT(*) DESC"
    ).fetchall():
        print(f"  {source}: {count}")

    print()
    print("Jobs by role guess:")
    for role_name, count in connection.execute(
        """
        SELECT role_guess, COUNT(*)
        FROM jobs
        GROUP BY role_guess
        ORDER BY COUNT(*) DESC
        """
    ).fetchall():
        print(f"  {role_name}: {count}")

    seniority_rows = connection.execute(
        """
        SELECT seniority_guess, COUNT(*)
        FROM jobs
        WHERE seniority_guess IS NOT NULL AND TRIM(seniority_guess) != ''
        GROUP BY seniority_guess
        ORDER BY COUNT(*) DESC
        """
    ).fetchall()
    if seniority_rows:
        print()
        print("Jobs by seniority guess:")
        for seniority_name, count in seniority_rows:
            print(f"  {seniority_name}: {count}")

    print()
    print("Top skills:")
    for skill, count in connection.execute(
        """
        SELECT skill, COUNT(*)
        FROM job_skills
        GROUP BY skill
        ORDER BY COUNT(*) DESC, skill ASC
        LIMIT 15
        """
    ).fetchall():
        print(f"  {skill}: {count}")


def build_parser():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command")

    scrape_parser = subparsers.add_parser("scrape")
    scrape_parser.add_argument("--db", default="jobs.db")
    scrape_parser.add_argument("--limit", type=int, default=None)
    scrape_parser.add_argument("--skip-remoteok", action="store_true")
    scrape_parser.add_argument("--boards", nargs="*", default=DEFAULT_GREENHOUSE_BOARDS)

    export_parser = subparsers.add_parser("export-labels")
    export_parser.add_argument("--db", default="jobs.db")
    export_parser.add_argument("--csv", default="labels_to_annotate.csv")
    export_parser.add_argument("--limit", type=int, default=150)
    export_parser.add_argument("--include-labeled", action="store_true")

    import_parser = subparsers.add_parser("import-labels")
    import_parser.add_argument("--db", default="jobs.db")
    import_parser.add_argument("--csv", default="labels_to_annotate.csv")

    skills_parser = subparsers.add_parser("extract-skills")
    skills_parser.add_argument("--db", default="jobs.db")
    skills_parser.add_argument("--limit", type=int, default=None)

    stats_parser = subparsers.add_parser("stats")
    stats_parser.add_argument("--db", default="jobs.db")

    return parser


def scrape_command(connection, args):
    all_jobs = []

    if not args.skip_remoteok:
        remoteok_jobs = fetch_remoteok_jobs()
        all_jobs.extend(remoteok_jobs)
        print(f"RemoteOK relevant jobs: {len(remoteok_jobs)}")

    for board in args.boards:
        board_jobs = fetch_greenhouse_jobs(board)
        all_jobs.extend(board_jobs)
        print(f"Greenhouse {board} relevant jobs: {len(board_jobs)}")

    all_jobs = deduplicate_jobs(all_jobs)

    if args.limit is not None:
        all_jobs = all_jobs[: args.limit]

    inserted, updated = save_jobs(connection, all_jobs)
    print()
    print(f"Fetched relevant jobs: {len(all_jobs)}")
    print(f"Inserted: {inserted}")
    print(f"Updated: {updated}")
    print(f"Database: {args.db}")


def main():
    parser = build_parser()
    argv = sys.argv[1:]

    if not argv or argv[0].startswith("-"):
        argv = ["scrape", *argv]

    args = parser.parse_args(argv)
    connection = sqlite3.connect(args.db)

    try:
        create_tables(connection)
        normalize_legacy_job_ids(connection)
        backfill_existing_jobs(connection)
        removed_jobs = delete_irrelevant_jobs(connection)
        if removed_jobs:
            print(f"Removed irrelevant jobs: {removed_jobs}")

        if args.command == "scrape":
            scrape_command(connection, args)
        elif args.command == "export-labels":
            exported = export_labels(
                connection,
                csv_path=args.csv,
                limit=args.limit,
                include_labeled=args.include_labeled,
            )
            print(f"Exported rows: {exported}")
            print(f"CSV file: {args.csv}")
            print("Allowed role labels:")
            print(", ".join(ROLE_CLASSES))
            print("Allowed seniority labels:")
            print(", ".join(SENIORITY_CLASSES))
        elif args.command == "import-labels":
            updated = import_labels(connection, csv_path=args.csv)
            print(f"Imported labels: {updated}")
            print(f"CSV file: {args.csv}")
        elif args.command == "extract-skills":
            processed_jobs, total_skills = fill_job_skills(connection, limit=args.limit)
            print(f"Processed jobs: {processed_jobs}")
            print(f"Saved skill rows: {total_skills}")
        elif args.command == "stats":
            show_stats(connection)
    finally:
        connection.close()


if __name__ == "__main__":
    main()
