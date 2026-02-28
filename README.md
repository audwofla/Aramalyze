# Aramalyze

End-to-end ARAM data and ML pipeline for League of Legends.

This project ingests champion metadata and ARAM modifiers, crawls ARAM match history, stores normalized records in PostgreSQL, builds a team-level dataset, and trains a win-probability model.

## What It Does

- Ingests champion and patch data from Data Dragon and Fandom
- Builds canonical patch snapshots in `data/canonical/*.json`
- Loads champion metadata/modifiers into PostgreSQL
- Discovers ARAM match IDs from seeded accounts and queues them for ingestion
- Ingests match, participant, and item-level stats into PostgreSQL
- Builds `aram_team_dataset.csv` for modeling
- Trains a logistic regression model and predicts win probability for a 5-champion team

## Tech Stack

- Python
- PostgreSQL
- Riot Match-V5 API
- Fandom (MediaWiki) API
- scikit-learn, pandas, psycopg2, requests

## Project Structure

- `main.py` - Runs the champion ingestion/parsing/merge/load pipeline
- `src/pipeline/run.py` - Main pipeline orchestration
- `scripts/insert_seeds.py` - Inserts seed PUUID accounts
- `scripts/discover_matches.py` - Enqueues ARAM match IDs into `match_queue`
- `scripts/ingest_matches.py` - Ingests queued matches into relational tables
- `scripts/run_crawl_cycle.py` - Continuous discovery + ingestion cycle
- `src/datasets/build_team_dataset.py` - Builds team-level training CSV from DB
- `src/ml/train.py` - Trains and saves ML model artifact
- `src/ml/predict.py` - Predicts win probability for a given team composition
- `sql/schema/schema.sql` - Core schema
- `sql/schema/match_queue.sql` - Match queue schema

## Prerequisites

- Python 3.10+
- PostgreSQL 13+
- Riot API key

## Setup

1. Create and activate a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

2. Install dependencies:

```bash
pip install -r requirements.txt
pip install pandas scikit-learn joblib psycopg2-binary python-dotenv
```

3. Create `.env` in project root:

```env
DATABASE_URL=postgresql://<user>:<password>@localhost:5432/aramalyze
CANONICAL_DIR=data/canonical
API_KEY=<your_riot_api_key>
```

4. Initialize database schema:

```bash
psql -h localhost -U <user> -d aramalyze -f sql/schema/schema.sql
psql -h localhost -U <user> -d aramalyze -f sql/schema/match_queue.sql
```

## Usage

### 1) Build champion canonical data + load to DB

```bash
python3 main.py
```

### 2) Seed initial accounts

```bash
python3 -m scripts.insert_seeds
```

### 3) Discover and ingest ARAM matches

One-off runs:

```bash
python3 -m scripts.discover_matches
python3 -m scripts.ingest_matches
```

Continuous crawl cycle:

```bash
python3 -m scripts.run_crawl_cycle
```

### 4) Build ML dataset

```bash
python3 -m src.datasets.build_team_dataset
```

This creates `aram_team_dataset.csv`.

### 5) Train model

```bash
python3 -m src.ml.train --csv aram_team_dataset.csv --out models/aram_lr.joblib
```

### 6) Predict team win probability

```bash
python3 -m src.ml.predict \
  --model models/aram_lr.joblib \
  --champs "[57,63,233,245,555]" \
  --tag_counts "{'Mage':2,'Tank':1,'Fighter':1,'Assassin':1}"
```


- Patch version strings are normalized to `major.minor` (for example, `16.4.1 -> 16.4`) for consistency across tables and artifacts.
