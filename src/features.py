"""
Turns raw per-player-per-frame tracking data into one row per play,
describing the defensive pre-snap look.

Output of build_feature_table: one row per (gameId, playId) with columns:
    box_count, num_high_safeties, lb_spacing, defensive_personnel,
    motion_flag, offense_formation, receiver_alignment

Coordinate system reminder (NFL Next Gen Stats convention):
    x: 0-120, length of the field including end zones.
    y: 0-53.3, width of the field (sideline to sideline).
    playDirection tells you whether the offense is driving toward
    increasing or decreasing x, which determines which side of the
    line of scrimmage (LOS) the defense is lined up on.
"""
import numpy as np
import pandas as pd

BOX_DEPTH_YARDS = 5.0
BOX_WIDTH_BUFFER_YARDS = 1.0
SAFETY_DEPTH_YARDS = 8.0
SAFETY_MIDDLE_HALF_WIDTH_YARDS = 15.0
FIELD_WIDTH_YARDS = 53.3

OL_POSITIONS = {"C", "G", "T"}
LB_POSITIONS = {"ILB", "OLB", "MLB", "LB"}
DL_POSITIONS = {"DE", "DT", "NT"}
DB_POSITIONS = {"CB", "FS", "SS", "DB", "S"}


def get_snap_frame(tracking_df):
    """Filter tracking data down to just the frame at the snap, per play."""
    if "frameType" in tracking_df.columns:
        return tracking_df[tracking_df["frameType"] == "SNAP"].copy()
    return tracking_df[tracking_df["event"] == "ball_snap"].copy()


def _attach_play_context(snap_frame_df, plays_df, players_df):
    """
    Join snap-frame rows with the play-level context (who's on offense/
    defense, where the LOS is, which way the play is headed) and each
    player's position, so the geometry functions below have everything
    they need in one frame.
    """
    ctx = plays_df[[
        "gameId", "playId", "possessionTeam", "defensiveTeam",
        "absoluteYardlineNumber", "playDirection",
        "offenseFormation", "receiverAlignment",
    ]]
    merged = snap_frame_df.merge(ctx, on=["gameId", "playId"], how="inner")
    merged = merged.merge(
        players_df[["nflId", "position"]], on="nflId", how="left"
    )
    merged["is_defense"] = merged["club"] == merged["defensiveTeam"]
    return merged


def _defense_depth_from_los(merged_df):
    """
    Signed depth of each defender past the LOS, always positive toward
    the defense's side regardless of which way the play is headed.
    """
    sign = np.where(merged_df["playDirection"] == "right", 1.0, -1.0)
    return sign * (merged_df["x"] - merged_df["absoluteYardlineNumber"])


def compute_box_count(snap_frame_df, plays_df, players_df):
    """
    Count defenders within ~5 yards of the LOS, horizontally between the
    offensive tackles (plus a small buffer for edge defenders lined up
    just outside the tackle).
    """
    merged = _attach_play_context(snap_frame_df, plays_df, players_df)
    merged["depth"] = _defense_depth_from_los(merged)

    ol = merged[merged["position"].isin(OL_POSITIONS) & ~merged["is_defense"]]
    ol_bounds = ol.groupby(["gameId", "playId"])["y"].agg(
        y_min="min", y_max="max"
    )

    defenders = merged[merged["is_defense"]].merge(
        ol_bounds, on=["gameId", "playId"], how="inner"
    )
    in_box = (
        (defenders["depth"] >= -0.5)
        & (defenders["depth"] <= BOX_DEPTH_YARDS)
        & (defenders["y"] >= defenders["y_min"] - BOX_WIDTH_BUFFER_YARDS)
        & (defenders["y"] <= defenders["y_max"] + BOX_WIDTH_BUFFER_YARDS)
    )
    box_count = (
        defenders[in_box]
        .groupby(["gameId", "playId"])
        .size()
        .rename("box_count")
    )
    return box_count


def compute_safety_count(snap_frame_df, plays_df, players_df):
    """
    Count defenders aligned >8 yards deep and near the middle of the
    field — i.e. players deep enough and central enough to be reading
    the deep middle, regardless of what position they're listed at.
    """
    merged = _attach_play_context(snap_frame_df, plays_df, players_df)
    merged["depth"] = _defense_depth_from_los(merged)

    field_center = FIELD_WIDTH_YARDS / 2
    is_deep = merged["depth"] > SAFETY_DEPTH_YARDS
    is_central = (
        (merged["y"] - field_center).abs() <= SAFETY_MIDDLE_HALF_WIDTH_YARDS
    )
    high_safeties = merged[merged["is_defense"] & is_deep & is_central]
    safety_count = (
        high_safeties.groupby(["gameId", "playId"])
        .size()
        .rename("num_high_safeties")
    )
    return safety_count


def compute_lb_spacing(snap_frame_df, plays_df, players_df):
    """Horizontal spread (max y - min y) of linebacker alignment."""
    merged = _attach_play_context(snap_frame_df, plays_df, players_df)
    lbs = merged[merged["is_defense"] & merged["position"].isin(LB_POSITIONS)]
    spacing = lbs.groupby(["gameId", "playId"])["y"].agg(
        lambda s: s.max() - s.min() if len(s) > 1 else np.nan
    )
    return spacing.rename("lb_spacing")


def compute_defensive_personnel(snap_frame_df, plays_df, players_df):
    """
    Classify the defensive sub-package by DB count on the field at snap:
    4 DBs -> Base, 5 -> Nickel, 6 -> Dime, 7+ -> Quarter/Prevent,
    <4 -> Heavy (goal-line/short-yardage front).
    """
    merged = _attach_play_context(snap_frame_df, plays_df, players_df)
    db_counts = (
        merged[merged["is_defense"] & merged["position"].isin(DB_POSITIONS)]
        .groupby(["gameId", "playId"])
        .size()
    )

    def label(n):
        if n <= 3:
            return "Heavy"
        if n == 4:
            return "Base"
        if n == 5:
            return "Nickel"
        if n == 6:
            return "Dime"
        return "Quarter"

    return db_counts.apply(label).rename("defensive_personnel")


def detect_motion(player_play_df):
    """
    Flag whether any offensive player went in motion pre-snap.

    player_play.csv already records this per player (motionSinceLineset /
    inMotionAtBallSnap), which is more reliable than re-deriving it from
    raw speed thresholds in tracking data, so we use it directly instead
    of touching the frame-by-frame positions.
    """
    motion_cols = [
        c for c in ("motionSinceLineset", "inMotionAtBallSnap")
        if c in player_play_df.columns
    ]
    any_motion = player_play_df[motion_cols].fillna(False).any(axis=1)
    flag = (
        player_play_df.assign(any_motion=any_motion)
        .groupby(["gameId", "playId"])["any_motion"]
        .any()
        .rename("motion_flag")
    )
    return flag


def build_feature_table(tracking_df, plays_df, players_df, player_play_df):
    """Combine the above into one row per play."""
    snap_frame = get_snap_frame(tracking_df)

    feature_table = pd.concat(
        [
            compute_box_count(snap_frame, plays_df, players_df),
            compute_safety_count(snap_frame, plays_df, players_df),
            compute_lb_spacing(snap_frame, plays_df, players_df),
            compute_defensive_personnel(snap_frame, plays_df, players_df),
            detect_motion(player_play_df),
        ],
        axis=1,
    ).reset_index()

    context_cols = plays_df[
        ["gameId", "playId", "offenseFormation", "receiverAlignment"]
    ]
    feature_table = feature_table.merge(
        context_cols, on=["gameId", "playId"], how="left"
    )
    feature_table["motion_flag"] = feature_table["motion_flag"].fillna(False)
    return feature_table
