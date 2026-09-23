"""
Unit tests for the feature-engineering geometry in src/features.py.
These use small hand-built DataFrames (not real tracking data) so the
expected box count / safety count / LB spacing can be computed by hand
and checked exactly.
"""
import pandas as pd
import pytest

from src.features import (
    get_snap_frame,
    compute_box_count,
    compute_safety_count,
    compute_lb_spacing,
    compute_defensive_personnel,
    detect_motion,
    build_feature_table,
)

GAME_ID, PLAY_ID = 1, 1
LOS_X = 60.0  # absoluteYardlineNumber
FIELD_CENTER_Y = 26.65


def _row(nfl_id, club, position, x, y, frame_type="SNAP"):
    return {
        "gameId": GAME_ID, "playId": PLAY_ID, "nflId": nfl_id,
        "club": club, "frameType": frame_type, "x": x, "y": y,
        "_position": position,  # merged in separately via players_df in real data
    }


@pytest.fixture
def snap_tracking_and_players():
    rows = [
        # Offensive line: y spans 21-29 -> box width bounds [20, 30]
        _row(1, "AAA", "C", 59.5, 25),
        _row(2, "AAA", "G", 59.5, 23),
        _row(3, "AAA", "G", 59.5, 27),
        _row(4, "AAA", "T", 59.5, 21),
        _row(5, "AAA", "T", 59.5, 29),
        _row(6, "AAA", "QB", 58.0, 25),
        # Defensive line: 1 yard deep, inside the tackle box -> in the box
        _row(11, "BBB", "DE", 61.0, 27),
        _row(12, "BBB", "DT", 61.0, 25),
        _row(13, "BBB", "DE", 61.0, 23),
        # Linebackers: 4 yards deep, at the box edges (y=20 and y=30) -> in the box
        _row(14, "BBB", "ILB", 64.0, 20),
        _row(15, "BBB", "OLB", 64.0, 30),
        # Corners: shallow but split out wide -> NOT in the box
        _row(16, "BBB", "CB", 63.0, 5),
        _row(17, "BBB", "CB", 63.0, 48),
        # Safeties: 12 yards deep, near the middle -> high safeties, NOT in the box
        _row(18, "BBB", "FS", 72.0, 20),
        _row(19, "BBB", "SS", 72.0, 30),
        # A distractor pre-snap frame that get_snap_frame must filter out
        _row(1, "AAA", "C", 59.5, 25, frame_type="BEFORE_SNAP"),
    ]
    tracking_df = pd.DataFrame(rows).drop(columns=["_position"])
    players_df = pd.DataFrame(
        [{"nflId": r["nflId"], "position": r["_position"]} for r in rows]
    ).drop_duplicates("nflId")
    return tracking_df, players_df


@pytest.fixture
def plays_df():
    return pd.DataFrame([{
        "gameId": GAME_ID, "playId": PLAY_ID,
        "possessionTeam": "AAA", "defensiveTeam": "BBB",
        "absoluteYardlineNumber": LOS_X, "playDirection": "right",
        "offenseFormation": "SHOTGUN", "receiverAlignment": "2x2",
        "isDropback": False, "pff_runConceptPrimary": "INSIDE ZONE",
        "dropbackType": None, "timeToThrow": None,
        "qbKneel": 0, "qbSpike": 0,
        "expectedPointsAdded": 0.4, "playNullifiedByPenalty": False,
    }])


@pytest.fixture
def player_play_df():
    return pd.DataFrame([
        {"gameId": GAME_ID, "playId": PLAY_ID, "nflId": 1,
         "motionSinceLineset": False, "inMotionAtBallSnap": False},
        {"gameId": GAME_ID, "playId": PLAY_ID, "nflId": 6,
         "motionSinceLineset": True, "inMotionAtBallSnap": False},
    ])


def test_get_snap_frame_drops_other_frames(snap_tracking_and_players):
    tracking_df, _ = snap_tracking_and_players
    snap_frame = get_snap_frame(tracking_df)
    assert (snap_frame["frameType"] == "SNAP").all()
    assert len(snap_frame) == len(tracking_df) - 1


def test_box_count(snap_tracking_and_players, plays_df):
    tracking_df, players_df = snap_tracking_and_players
    snap_frame = get_snap_frame(tracking_df)
    result = compute_box_count(snap_frame, plays_df, players_df)
    assert result[(GAME_ID, PLAY_ID)] == 5  # 3 DL + 2 LBs


def test_safety_count(snap_tracking_and_players, plays_df):
    tracking_df, players_df = snap_tracking_and_players
    snap_frame = get_snap_frame(tracking_df)
    result = compute_safety_count(snap_frame, plays_df, players_df)
    assert result[(GAME_ID, PLAY_ID)] == 2


def test_lb_spacing(snap_tracking_and_players, plays_df):
    tracking_df, players_df = snap_tracking_and_players
    snap_frame = get_snap_frame(tracking_df)
    result = compute_lb_spacing(snap_frame, plays_df, players_df)
    assert result[(GAME_ID, PLAY_ID)] == 10  # y=20 to y=30


def test_defensive_personnel(snap_tracking_and_players, plays_df):
    tracking_df, players_df = snap_tracking_and_players
    snap_frame = get_snap_frame(tracking_df)
    result = compute_defensive_personnel(snap_frame, plays_df, players_df)
    assert result[(GAME_ID, PLAY_ID)] == "Base"  # 2 CBs + 2 safeties = 4 DBs


def test_detect_motion(player_play_df):
    result = detect_motion(player_play_df)
    assert result[(GAME_ID, PLAY_ID)] is True or result[(GAME_ID, PLAY_ID)]


def test_build_feature_table_end_to_end(
    snap_tracking_and_players, plays_df, player_play_df
):
    tracking_df, players_df = snap_tracking_and_players
    table = build_feature_table(tracking_df, plays_df, players_df, player_play_df)

    assert len(table) == 1
    row = table.iloc[0]
    assert row["box_count"] == 5
    assert row["num_high_safeties"] == 2
    assert row["lb_spacing"] == 10
    assert row["defensive_personnel"] == "Base"
    assert row["motion_flag"] == True  # noqa: E712
    assert row["offenseFormation"] == "SHOTGUN"
