import json
import os
from pathlib import Path

import psycopg
from dotenv import load_dotenv

load_dotenv()

DB_URL = os.environ["DATABASE_URL"]
CANONICAL_DIR = Path(os.environ["CANONICAL_DIR"])


def extract_champions_map(payload: dict) -> dict:
    """
    Supports canonical shapes:
      A) {"432": {...}, "200": {...}}
      B) {"patch": "...", "champions": {"432": {...}}}
      C) {"data": {"champions": {...}}}
      D) {"data": {...}} where data itself is the champ map
    Returns dict keyed by champ-id strings.
    """
    if not isinstance(payload, dict):
        raise ValueError("Canonical JSON must be a dict/object at top-level.")

    if payload and all(isinstance(v, dict) and "id" in v for v in payload.values()):
        return payload

    champs = payload.get("champions")
    if isinstance(champs, dict) and champs and all(isinstance(v, dict) and "id" in v for v in champs.values()):
        return champs

    data = payload.get("data")
    if isinstance(data, dict) and isinstance(data.get("champions"), dict):
        champs = data["champions"]
        if champs and all(isinstance(v, dict) and "id" in v for v in champs.values()):
            return champs

    if isinstance(data, dict) and data and all(isinstance(v, dict) and "id" in v for v in data.values()):
        return data

    raise ValueError(f"Couldn't find champions map. Top-level keys: {list(payload.keys())[:30]}")


UPSERT_CHAMPION = """
INSERT INTO champions (id, key, name)
VALUES (%s, %s, %s)
ON CONFLICT (id) DO UPDATE
SET key = EXCLUDED.key,
    name = EXCLUDED.name;
"""

UPSERT_TAG = """
INSERT INTO champion_tag (champion_id, tag)
VALUES (%s, %s)
ON CONFLICT (champion_id, tag) DO NOTHING;
"""

UPSERT_ARAM_MODS = """
INSERT INTO champion_aram_mods (
  champion_id, patch,
  ability_haste, dmg_dealt, dmg_taken, healing, shielding, tenacity,
  attack_speed, energy_regen
)
VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
ON CONFLICT (champion_id, patch) DO UPDATE
SET
  ability_haste = EXCLUDED.ability_haste,
  dmg_dealt     = EXCLUDED.dmg_dealt,
  dmg_taken     = EXCLUDED.dmg_taken,
  healing       = EXCLUDED.healing,
  shielding     = EXCLUDED.shielding,
  tenacity      = EXCLUDED.tenacity,
  attack_speed  = EXCLUDED.attack_speed,
  energy_regen  = EXCLUDED.energy_regen;
"""

UPSERT_SPELL_CHANGE = """
INSERT INTO champion_spell_changes (champion_id, patch, spell_key, idx, change_text)
VALUES (%s,%s,%s,%s,%s)
ON CONFLICT (champion_id, patch, spell_key, idx) DO UPDATE
SET change_text = EXCLUDED.change_text;
"""


def load_champions_for_patch(patch: str, canonical_path: Path | None = None) -> dict:
    """
    Callable entrypoint for run.py.

    Loads champions/tags/aram_mods/spell_changes for `patch` into Postgres.

    Returns a dict of counts:
      {"patch": patch, "champions": X, "tags": Y, "mods_rows": Z, "spell_rows": W}
    """
    if canonical_path is None:
        canonical_path = CANONICAL_DIR / f"{patch}.json"

    payload = json.loads(canonical_path.read_text(encoding="utf-8"))
    champions_map = extract_champions_map(payload)

    champs_upserted = 0
    tags_inserted = 0
    mods_rows_upserted = 0
    spell_rows_upserted = 0

    with psycopg.connect(DB_URL) as conn:
        with conn.cursor() as cur:
            for champ in champions_map.values():
                if not isinstance(champ, dict):
                    continue

                champ_id = int(champ["id"])
                key = champ["key"]
                name = champ["name"]

                cur.execute(UPSERT_CHAMPION, (champ_id, key, name))
                champs_upserted += 1

                for tag in champ.get("tags") or []:
                    cur.execute(UPSERT_TAG, (champ_id, tag))
                    tags_inserted += 1

                aram_mods = champ.get("aram_mods") or {}
                if isinstance(aram_mods, dict) and aram_mods:
                    cur.execute(
                        UPSERT_ARAM_MODS,
                        (
                            champ_id,
                            patch,
                            aram_mods.get("ability_haste"), 
                            aram_mods.get("dmg_dealt"),
                            aram_mods.get("dmg_taken"),
                            aram_mods.get("healing"),
                            aram_mods.get("shielding"),
                            aram_mods.get("tenacity"),
                            aram_mods.get("attack_speed"),
                            aram_mods.get("energy_regen"),
                        ),
                    )
                    mods_rows_upserted += 1

                spell_changes = champ.get("spell_changes") or {}
                if isinstance(spell_changes, dict) and spell_changes:
                    for spell_key, lines in spell_changes.items():
                        if not lines:
                            continue
                        for i, line in enumerate(lines):
                            cur.execute(UPSERT_SPELL_CHANGE, (champ_id, patch, spell_key, i, line))
                            spell_rows_upserted += 1

        conn.commit()

    return {
        "patch": patch,
        "champions": champs_upserted,
        "tags": tags_inserted,
        "mods_rows": mods_rows_upserted,
        "spell_rows": spell_rows_upserted,
        "canonical_path": str(canonical_path),
    }


def main():
    files = list(CANONICAL_DIR.glob("*.json"))
    if not files:
        raise FileNotFoundError(f"No *.json files found in {CANONICAL_DIR}")

    def ver_tuple(path: Path):
        return tuple(int(x) for x in path.stem.split("."))

    best = max(files, key=ver_tuple)
    patch = best.stem
    stats = load_champions_for_patch(patch, canonical_path=best)
    print("Loaded:", stats)


if __name__ == "__main__":
    main()