"""
Derives what we're trying to predict:
    - play_family (categorical): inside_zone, outside_zone, gap_scheme,
      draw, quick_pass, dropback_pass, rollout_pass, screen, kneel, spike
    - success (binary), from expectedPointsAdded (EPA)

play_family is built from plays.csv's pff_runConceptPrimary / dropbackType /
timeToThrow rather than parsing playDescription text, since the structured
PFF columns already exist in the Big Data Bowl 2025 data. The exact string
values in pff_runConceptPrimary should be checked against the data once
downloaded (`plays_df["pff_runConceptPrimary"].value_counts()`) and the
keyword lists below adjusted if PFF's vocabulary differs — that's a lookup
change here, not a rewrite of the classification logic.
"""
import numpy as np

SUCCESS_EPA_THRESHOLD = 0.0

RUN_CONCEPT_KEYWORDS = [
    ("outside_zone", ["OUTSIDE ZONE", "STRETCH", "SWEEP"]),
    ("inside_zone", ["ZONE"]),  # catch-all zone match after outside_zone
    ("draw", ["DRAW"]),
    ("gap_scheme", ["POWER", "COUNTER", "TRAP", "PULL", "DUO"]),
]

QUICK_PASS_MAX_TIME_TO_THROW = 2.5


def _classify_run(concept):
    if not isinstance(concept, str):
        return "other_run"
    concept_upper = concept.upper()
    for family, keywords in RUN_CONCEPT_KEYWORDS:
        if any(kw in concept_upper for kw in keywords):
            return family
    return "other_run"


def _classify_pass(row):
    dropback_type = row.get("dropbackType")
    if isinstance(dropback_type, str) and "ROLLOUT" in dropback_type.upper():
        return "rollout_pass"

    time_to_throw = row.get("timeToThrow")
    if pd_notna(time_to_throw) and time_to_throw <= QUICK_PASS_MAX_TIME_TO_THROW:
        return "quick_pass"

    return "dropback_pass"


def pd_notna(value):
    return value is not None and not (isinstance(value, float) and np.isnan(value))


def assign_play_family(plays_df):
    """Map each play to a play_family label based on structured PFF columns."""
    plays_df = plays_df.copy()

    def classify(row):
        if row.get("qbKneel") == 1:
            return "kneel"
        if row.get("qbSpike") == 1:
            return "spike"
        if row.get("isDropback"):
            return _classify_pass(row)
        return _classify_run(row.get("pff_runConceptPrimary"))

    plays_df["play_family"] = plays_df.apply(classify, axis=1)
    return plays_df


def merge_outcomes(feature_df, plays_df, success_epa_threshold=SUCCESS_EPA_THRESHOLD):
    """Join engineered defensive-shell features with play_family + success/EPA."""
    labeled_plays = assign_play_family(plays_df)
    outcome_cols = labeled_plays[[
        "gameId", "playId", "play_family", "expectedPointsAdded",
        "playNullifiedByPenalty",
    ]].rename(columns={"expectedPointsAdded": "epa"})

    merged = feature_df.merge(outcome_cols, on=["gameId", "playId"], how="inner")
    merged = merged[merged["playNullifiedByPenalty"] != True]  # noqa: E712
    merged["success"] = merged["epa"] > success_epa_threshold
    return merged.drop(columns=["playNullifiedByPenalty"])
