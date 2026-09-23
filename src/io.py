"""
Stage A: loads the raw Big Data Bowl 2025 CSVs into DataFrames.

Expected files under data/raw/ (as unzipped from the Kaggle competition):
    games.csv
    plays.csv
    players.csv
    player_play.csv
    tracking_week_1.csv ... tracking_week_9.csv
"""
import pandas as pd

from src.config import RAW_DIR, NUM_TRACKING_WEEKS


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
