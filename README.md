# Simple Job Data Pipeline

Один файл без классов и без лишней структуры, но уже для первых 5 шагов проекта:

- сбор вакансий из `RemoteOK` и нескольких `Greenhouse` boards
- сохранение в `SQLite`
- `description_clean` для чистого текста
- `role_guess` и `role_label` для будущей классификации
- отдельная таблица `job_skills`
- экспорт CSV для ручной разметки

## 1. Собрать вакансии

```bash
python scraper.py scrape --limit 600
```

Можно и по-старому:

```bash
python scraper.py --limit 600
```

## 2. Выгрузить вакансии на ручную разметку ролей

```bash
python scraper.py export-labels --limit 150
```

После этого появится `labels_to_annotate.csv`. Заполняй колонку `role_label` значениями:

- `Backend`
- `Data Analyst`
- `Data Scientist`
- `ML Engineer`
- `DevOps`
- `QA`
- `Other`

## 3. Импортировать разметку обратно в БД

```bash
python scraper.py import-labels --csv labels_to_annotate.csv
```

## 4. Извлечь skills в отдельную таблицу

```bash
python scraper.py extract-skills
```

## 5. Посмотреть краткую статистику

```bash
python scraper.py stats
```

## 6. Train baseline classifier

Install ML packages first:

```bash
pip install pandas scikit-learn joblib
```

Train on labeled rows from `jobs.db`:

```bash
python train_classifier.py
```

If you want a different DB or model filename:

```bash
python train_classifier.py --db jobs.db --model-out job_classifier.pkl
```
