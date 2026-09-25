"""
Tests for the BDB 2026 pipeline (src/features_2026.py, src/labels_2026.py).
Uses small hand-built rows mirroring supplementary_data.csv's real schema
(column names/values verified against the actual downloaded file).
"""
import pandas as pd

from src.features_2026 import bucket_coverage_shell, build_feature_table
from src.labels_2026 import assign_play_family, build_labeled_dataset
from src.evaluate import build_success_rate_table, recommend_play


def _play(pass_length, dropback_type="TRADITIONAL", team_coverage_type="COVER_3_ZONE",
          team_coverage_man_zone="ZONE_COVERAGE", defenders_in_the_box=6,
          epa=0.2, nullified=False):
    return {
        "game_id": 1, "play_id": 1, "pass_length": pass_length,
        "dropback_type": dropback_type, "team_coverage_type": team_coverage_type,
        "team_coverage_man_zone": team_coverage_man_zone,
        "defenders_in_the_box": defenders_in_the_box,
        "expected_points_added": epa, "play_nullified_by_penalty": nullified,
    }


def test_bucket_coverage_shell():
    assert bucket_coverage_shell("COVER_0_MAN") == "0-high"
    assert bucket_coverage_shell("COVER_1_MAN") == "1-high"
    assert bucket_coverage_shell("COVER_3_ZONE") == "1-high"
    assert bucket_coverage_shell("COVER_2_ZONE") == "2-high"
    assert bucket_coverage_shell("COVER_4_ZONE") == "2-high"
    assert bucket_coverage_shell("COVER_6_ZONE") == "2-high"
    assert bucket_coverage_shell("PREVENT") == "prevent"
    assert bucket_coverage_shell(None) == "unknown"


def test_play_family_priorities_rollout_over_depth():
    df = pd.DataFrame([_play(pass_length=25, dropback_type="DESIGNED_ROLLOUT_RIGHT")])
    labeled = assign_play_family(df)
    assert labeled.iloc[0]["play_family"] == "rollout_pass"


def test_play_family_depth_buckets():
    df = pd.DataFrame([
        _play(pass_length=-3),
        _play(pass_length=5),
        _play(pass_length=15),
        _play(pass_length=25),
    ])
    labeled = assign_play_family(df)
    assert list(labeled["play_family"]) == [
        "screen_pass", "quick_pass", "intermediate_pass", "deep_pass",
    ]


def test_build_labeled_dataset_drops_scrambles_and_penalties():
    df = pd.DataFrame([
        _play(pass_length=10, dropback_type="SCRAMBLE"),
        _play(pass_length=10, nullified=True),
        _play(pass_length=10),
    ])
    labeled = build_labeled_dataset(df)
    assert len(labeled) == 1


def test_feature_table_and_recommend_end_to_end():
    rows = []
    # Light box, 2-high, zone: deep_pass wins historically
    rows += [_play(pass_length=25, team_coverage_type="COVER_2_ZONE",
                    defenders_in_the_box=5, epa=0.6 if s else -0.4)
             for s in [True] * 8 + [False] * 2]
    rows += [_play(pass_length=5, team_coverage_type="COVER_2_ZONE",
                    defenders_in_the_box=5, epa=0.6 if s else -0.4)
             for s in [True] * 2 + [False] * 8]
    df = pd.DataFrame(rows)

    labeled = build_labeled_dataset(df)
    featured = build_feature_table(labeled)

    shell_key_cols = ["coverage_shell", "man_zone", "box_bucket"]
    table = build_success_rate_table(featured, shell_key_cols=shell_key_cols, min_sample_size=5)
    result = recommend_play(
        table,
        shell={"coverage_shell": "2-high", "man_zone": "ZONE_COVERAGE", "box_bucket": "light"},
        shell_key_cols=shell_key_cols, min_sample_size=5,
    )

    assert result is not None
    assert result["play_family"] == "deep_pass"
    assert result["success_rate"] == 0.8
