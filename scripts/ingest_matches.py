import os
import time
import datetime
import requests
import psycopg2
from dotenv import load_dotenv

from src.utils.versioning import patch_mm

load_dotenv()

API_KEY = os.environ["API_KEY"]
DATABASE_URL = os.environ["DATABASE_URL"]

HEADERS = {"X-Riot-Token": API_KEY}
ARAM_QUEUE_ID = 450


def fetch_match(match_id):
    url = f"https://americas.api.riotgames.com/lol/match/v5/matches/{match_id}"
    while True:
        r = requests.get(url, headers=HEADERS, timeout=20)
        if r.status_code == 429:
            ra = int(r.headers.get("Retry-After", "2"))
            print(f"429 rate limited; sleeping {ra}s", flush=True)
            time.sleep(ra)
            continue
        else:
            r.raise_for_status()
            return r.json()


def main(batch_size: int = 10):
    with psycopg2.connect(DATABASE_URL) as conn:
        while True:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT match_id
                    FROM match_queue
                    WHERE status='pending'
                    ORDER BY discovered_at
                    LIMIT %s;
                    """,
                    (batch_size,),
                )
                rows = cur.fetchall()

            if not rows:
                print("No pending matches left.")
                return

            for (match_id,) in rows:
                try:
                    with conn.cursor() as cur:
                        cur.execute(
                            "UPDATE match_queue SET status='processing' WHERE match_id=%s;",
                            (match_id,),
                        )
                    conn.commit()

                    data = fetch_match(match_id)
                    info = data["info"]

                    if info.get("queueId") != ARAM_QUEUE_ID:
                        with conn.cursor() as cur:
                            cur.execute(
                                "UPDATE match_queue SET status='done', fetched_at=CURRENT_TIMESTAMP WHERE match_id=%s;",
                                (match_id,),
                            )
                        conn.commit()
                        continue

                    patch = patch_mm(info.get("gameVersion", ""))
                    game_dt = datetime.datetime.fromtimestamp(
                        info["gameStartTimestamp"] / 1000.0, tz=datetime.timezone.utc
                    )

                    participants = info["participants"]

                    with conn.cursor() as cur:
                        cur.execute(
                            """
                            INSERT INTO matches(match_id, patch, queue_id, game_datetime)
                            VALUES (%s, %s, %s, %s)
                            ON CONFLICT (match_id) DO NOTHING;
                            """,
                            (match_id, patch, info["queueId"], game_dt),
                        )

                        for p in participants:
                            cur.execute(
                                """
                                INSERT INTO accounts(puuid)
                                VALUES (%s)
                                ON CONFLICT (puuid) DO NOTHING;
                                """,
                                (p["puuid"],),
                            )

                        for p in participants:
                            cur.execute(
                                """
                                INSERT INTO participants(
                                  match_id, puuid, champion_id, team_id, win,
                                  total_damage_dealt, physical_damage_dealt, magic_damage_dealt, true_damage_dealt,
                                  damage_taken, gold_earned, heals, shields,
                                  kills, deaths, assists
                                )
                                VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
                                ON CONFLICT (match_id, puuid) DO NOTHING;
                                """,
                                (
                                    match_id,
                                    p["puuid"],
                                    p["championId"],
                                    p["teamId"],
                                    p["win"],
                                    p.get("totalDamageDealtToChampions"),
                                    p.get("physicalDamageDealtToChampions"),
                                    p.get("magicDamageDealtToChampions"),
                                    p.get("trueDamageDealtToChampions"),
                                    p.get("totalDamageTaken"),
                                    p.get("goldEarned"),
                                    p.get("totalHeal"),
                                    p.get("totalDamageShieldedOnTeammates"),
                                    p.get("kills"),
                                    p.get("deaths"),
                                    p.get("assists"),
                                ),
                            )

                            # 4) items (slots 0..6)
                            for slot in range(7):
                                item_id = p.get(f"item{slot}")
                                if not item_id:
                                    continue
                                cur.execute(
                                    """
                                    INSERT INTO participant_items(match_id, puuid, item_id, slot)
                                    VALUES (%s,%s,%s,%s)
                                    ON CONFLICT (match_id, puuid, slot) DO NOTHING;
                                    """,
                                    (match_id, p["puuid"], item_id, slot),
                                )

                        cur.execute(
                            """
                            UPDATE match_queue
                            SET status='done', fetched_at=CURRENT_TIMESTAMP, last_error=NULL
                            WHERE match_id=%s;
                            """,
                            (match_id,),
                        )

                    conn.commit()
                    #print(f"done {match_id}")

                except Exception as e:
                    conn.rollback()
                    with conn.cursor() as cur:
                        cur.execute(
                            """
                            UPDATE match_queue
                            SET status='error',
                                retry_count=retry_count+1,
                                last_error=%s
                            WHERE match_id=%s;
                            """,
                            (str(e), match_id),
                        )
                    conn.commit()
                    print(f"error {match_id}: {e}")


if __name__ == "__main__":
    main(batch_size=5)
