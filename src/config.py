"""
Central place for filesystem paths so every module agrees on where
raw/processed data lives, regardless of what directory a script is run from.
"""
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
MODELS_DIR = ROOT_DIR / "models"

# Big Data Bowl 2025 ships tracking data split into one file per week.
NUM_TRACKING_WEEKS = 9
