from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
INGESTION_DIR = PROJECT_ROOT / "src" / "ingestion"
PARSING_DIR = PROJECT_ROOT / "src" / "parsing"
MERGING_DIR = PROJECT_ROOT / "src" / "merging"
PIPELINE_DIR = PROJECT_ROOT / "src" / "pipeline"
DATA_DIR = PROJECT_ROOT / "data"
