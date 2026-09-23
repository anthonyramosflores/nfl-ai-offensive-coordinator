"""
Unit tests for the empirical success-rate lookup table in src/evaluate.py
— this is the actual "coordinator" recommendation logic.
"""
import pandas as pd

from src.evaluate import (
    bucket_box_count,
    bucket_safety_count,
    build_success_rate_table,
    recommend_play,
)


def _labeled_row(box_count, num_high_safeties, defensive_personnel,
                  motion_flag, play_family, success, epa):
    return {
        "box_count": box_count, "num_high_safeties": num_high_safeties,
        "defensive_personnel": defensive_personnel, "motion_flag": motion_flag,
        "play_family": play_family, "success": success, "epa": epa,
    }


def test_bucket_box_count_edges():
    assert bucket_box_count(5) == "light"
    assert bucket_box_count(6) == "standard"
    assert bucket_box_count(7) == "heavy"
    assert bucket_box_count(9) == "stacked"


def test_bucket_safety_count():
    assert bucket_safety_count(1) == "1-high"
    assert bucket_safety_count(2) == "2-high"
    assert bucket_safety_count(4) == "3+-high"


def test_recommend_play_picks_higher_success_rate():
    # Same shell (light box, 2-high, Base, no motion): inside_zone succeeds
    # 8/10 times, quick_pass only 3/10 -> inside_zone should be recommended.
    rows = []
    rows += [_labeled_row(5, 2, "Base", False, "inside_zone", s, 0.5 if s else -0.3)
             for s in [True] * 8 + [False] * 2]
    rows += [_labeled_row(5, 2, "Base", False, "quick_pass", s, 0.5 if s else -0.3)
             for s in [True] * 3 + [False] * 7]
    labeled_df = pd.DataFrame(rows)

    table = build_success_rate_table(labeled_df, min_sample_size=5)
    result = recommend_play(
        table, safety_shell="2-high", defensive_personnel="Base",
        box_bucket="light", motion_flag=False, min_sample_size=5,
    )

    assert result is not None
    assert result["play_family"] == "inside_zone"
    assert result["success_rate"] == 0.8
    assert result["match_level"] == "exact_shell"


def test_recommend_play_falls_back_when_shell_too_rare():
    # Only 2 plays match the exact shell (below min_sample_size), but
    # plenty of data exists for the same safety_shell more broadly.
    rows = []
    rows += [_labeled_row(5, 1, "Base", False, "outside_zone", True, 0.5)] * 2
    rows += [_labeled_row(6, 1, "Nickel", True, "outside_zone", s, 0.5 if s else -0.2)
             for s in [True] * 9 + [False] * 1]
    labeled_df = pd.DataFrame(rows)

    table = build_success_rate_table(labeled_df, min_sample_size=5)
    result = recommend_play(
        table, safety_shell="1-high", defensive_personnel="Base",
        box_bucket="light", motion_flag=False, min_sample_size=5,
    )

    assert result is not None
    assert result["match_level"] != "exact_shell"


def test_recommend_play_returns_none_without_any_data():
    empty_table = build_success_rate_table(pd.DataFrame(columns=[
        "box_count", "num_high_safeties", "defensive_personnel",
        "motion_flag", "play_family", "success", "epa",
    ]), min_sample_size=5)
    result = recommend_play(
        empty_table, safety_shell="2-high", defensive_personnel="Base",
        box_bucket="light", motion_flag=False, min_sample_size=5,
    )
    assert result is None
