import json
import requests
import shutil
from pathlib import Path
from src.config.paths import DATA_DIR

PATCH_VERSION_URL = "https://ddragon.leagueoflegends.com/api/versions.json"
CHAMPION_DATA_URL = "https://ddragon.leagueoflegends.com/cdn/{patch}/data/en_US/champion.json"
CHAMPION_ICON_URL = "https://ddragon.leagueoflegends.com/cdn/{patch}/img/champion/{filename}"
CHAMPION_SPLASH_URL = "https://ddragon.leagueoflegends.com/cdn/img/champion/splash/{key}_0.jpg"

DDDRAGON_DIR = DATA_DIR / "ddragon"
ICON_DIR = DDDRAGON_DIR / "icons"
SPLASH_DIR = DDDRAGON_DIR / "splashes"
RAW_DIR = DDDRAGON_DIR / "raw"


def fetch_latest_patch() -> str:
    r = requests.get(PATCH_VERSION_URL, timeout=10)
    r.raise_for_status()
    return r.json()[0]


def fetch_champion_json(patch: str) -> dict:
    url = CHAMPION_DATA_URL.format(patch=patch)
    r = requests.get(url, timeout=10)
    r.raise_for_status()
    return r.json()


def save_raw_json(patch: str, data: dict) -> None:
    path = RAW_DIR / patch / "ddragon_champions.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def download_file(url: str, filepath: Path) -> None:
    filepath.parent.mkdir(parents=True, exist_ok=True)
    if filepath.exists():
        return

    with requests.get(url, stream=True, timeout=20) as r:
        r.raise_for_status()
        with open(filepath, "wb") as f:
            for chunk in r.iter_content(chunk_size=1024 * 256):
                if chunk:
                    f.write(chunk)


def download_champion_icon(patch: str, icon_dir: Path, filename: str) -> None:
    url = CHAMPION_ICON_URL.format(patch=patch, filename=filename)
    filepath = icon_dir / filename
    download_file(url, filepath)


def download_champion_splash(splash_dir: Path, champ_key: str) -> None:
    url = CHAMPION_SPLASH_URL.format(key=champ_key)
    filepath = splash_dir / f"{champ_key}_0.jpg"
    download_file(url, filepath)


def _delete_other_patch_folders(root: Path, keep_patch: str) -> None:
    if not root.exists():
        return
    for p in root.iterdir():
        if p.is_dir() and p.name != keep_patch:
            shutil.rmtree(p)


def update_ddragon(keep_only_latest: bool = True) -> str:
    ICON_DIR.mkdir(parents=True, exist_ok=True)
    SPLASH_DIR.mkdir(parents=True, exist_ok=True)
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    patch = fetch_latest_patch()

    if keep_only_latest:
        _delete_other_patch_folders(ICON_DIR, patch)
        _delete_other_patch_folders(SPLASH_DIR, patch)
        _delete_other_patch_folders(RAW_DIR, patch)

    patch_icon_dir = ICON_DIR / patch
    patch_splash_dir = SPLASH_DIR / patch
    patch_icon_dir.mkdir(parents=True, exist_ok=True)
    patch_splash_dir.mkdir(parents=True, exist_ok=True)

    champ_json = fetch_champion_json(patch)
    save_raw_json(patch, champ_json)

    for champ_data in champ_json["data"].values():
        icon_filename = champ_data["image"]["full"] 
        champ_key = champ_data["id"]             

        download_champion_icon(patch, patch_icon_dir, icon_filename)
        download_champion_splash(patch_splash_dir, champ_key)

    return patch