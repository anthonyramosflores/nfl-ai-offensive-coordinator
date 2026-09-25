"""
Stage A: loads raw Big Data Bowl CSVs into DataFrames.

BDB 2025 files under data/raw/ (as unzipped from the Kaggle competition):
    games.csv
    plays.csv
    players.csv
    player_play.csv
    tracking_week_1.csv ... tracking_week_9.csv

BDB 2026 Analytics files under data/raw/ (currently the active dataset —
see src/features_2026.py / src/labels_2026.py):
    supplementary_data.csv       - one row per pass play, PFF-labeled
    train/input_2023_w01.csv ... - per-frame tracking, not needed for the
                                    shell/success-rate pipeline since
                                    supplementary_data.csv already has the
                                    shell columns (box count, coverage).
"""
import pandas as pd

from src.config import RAW_DIR, NUM_TRACKING_WEEKS


def load_supplementary() -> pd.DataFrame:
    """BDB 2026: one row per pass play with PFF shell/outcome columns."""
    return pd.read_csv(RAW_DIR / "supplementary_data.csv", low_memory=False)


def load_week_input(week: int, season: int = 2023) -> pd.DataFrame:
    """BDB 2026: per-frame tracking for one week (only needed if you want
    to go beyond the shell columns already in supplementary_data.csv)."""
    path = RAW_DIR / "train" / f"input_{season}_w{week:02d}.csv"
    return pd.read_csv(path)


def load_games() -> pd.DataFrame:
    return pd.read_csv(RAW_DIR / "games.csv")


def load_plays() -> pd.DataFrame:
    return pd.read_csv(RAW_DIR / "plays.csv")


def load_players() -> pd.DataFrame:
    return pd.read_csv(RAW_DIR / "players.csv")


def load_player_play() -> pd.DataFrame:
    """Per-player-per-play data: motion flags, routes, block/rush assignments."""
    return pd.read_csv(RAW_DIR / "player_play.csv")


def load_tracking_week(week: int) -> pd.DataFrame:
    path = RAW_DIR / f"tracking_week_{week}.csv"
    return pd.read_csv(path)


def load_all_tracking(weeks=None) -> pd.DataFrame:
    """
    Concatenate tracking data across weeks. Tracking files are large
    (millions of rows/week), so callers doing a full-season build should
    prefer processing week-by-week and appending to data/processed/
    rather than holding all weeks in memory via this function.
    """
    weeks = weeks or range(1, NUM_TRACKING_WEEKS + 1)
    return pd.concat([load_tracking_week(w) for w in weeks], ignore_index=True)
