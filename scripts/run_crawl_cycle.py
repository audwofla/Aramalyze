import os
import time
import psycopg2
from dotenv import load_dotenv

from src.crawling.discovery import discover_for_active_accounts
from scripts.ingest_matches import main as ingest_main

load_dotenv()

DATABASE_URL = os.environ["DATABASE_URL"]
API_KEY = os.environ["API_KEY"]


def run_cycle(*, limit_accounts: int = 25, per_account_count: int = 100, ingest_batch_size: int = 10) -> None:
    print("=== DISCOVERY START ===", flush=True)

    with psycopg2.connect(DATABASE_URL) as conn:
        discover_for_active_accounts(
            conn,
            region="americas",
            api_key=API_KEY,
            per_account_count=per_account_count,
            limit_accounts=limit_accounts,
            sleep_seconds=0.10,
        )

    print("=== DISCOVERY DONE ===", flush=True)
    print("=== INGESTION START ===", flush=True)

    ingest_main(batch_size=ingest_batch_size)

    print("=== INGESTION DONE ===", flush=True)


if __name__ == "__main__":
    while True:
        try:
            print("=== CRAWL CYCLE START ===", flush=True)
            run_cycle(limit_accounts=5, per_account_count=100, ingest_batch_size=10)
            print("=== CRAWL CYCLE COMPLETE ===", flush=True)
        except Exception as e:
            print(f"Fatal error in crawl loop: {e}", flush=True)

        print("Sleeping 60 seconds before next cycle...", flush=True)
        time.sleep(60)