"""
play_family and success/EPA labels for the Big Data Bowl 2026 Analytics
data. This dataset is pass plays only (verified against real data: every
row in supplementary_data.csv has a pass_result of C/I/IN, no sacks and
no run plays), so play_family here is a pass-depth taxonomy rather than
the run/pass split used for BDB 2025.

Categories, in priority order:
    rollout_pass  - dropback_type is a designed rollout
    screen_pass   - pass_length < 0 (thrown behind the LOS)
    quick_pass    - pass_length 0-9
    intermediate_pass - pass_length 10-19
    deep_pass     - pass_length 20+

Scrambles (dropback_type SCRAMBLE / SCRAMBLE_ROLLOUT_*) are QB
improvisation, not a called play, so they're dropped rather than labeled
as a recommendable family.
"""
SUCCESS_EPA_THRESHOLD = 0.0

SCRAMBLE_DROPBACK_TYPES = {
    "SCRAMBLE", "SCRAMBLE_ROLLOUT_RIGHT", "SCRAMBLE_ROLLOUT_LEFT",
}


def _classify_pass_family(row):
    dropback_type = row.get("dropback_type")
    if isinstance(dropback_type, str) and "ROLLOUT" in dropback_type.upper() \
            and dropback_type.upper() not in SCRAMBLE_DROPBACK_TYPES:
        return "rollout_pass"

    pass_length = row.get("pass_length")
    if pass_length < 0:
        return "screen_pass"
    if pass_length <= 9:
        return "quick_pass"
    if pass_length <= 19:
        return "intermediate_pass"
    return "deep_pass"


def assign_play_family(supplementary_df):
    df = supplementary_df.copy()
    df["play_family"] = df.apply(_classify_pass_family, axis=1)
    return df


def build_labeled_dataset(supplementary_df, success_epa_threshold=SUCCESS_EPA_THRESHOLD):
    """Filter to called pass plays and attach play_family + success/EPA."""
    df = supplementary_df[
        supplementary_df["play_nullified_by_penalty"] != True  # noqa: E712
    ].copy()
    df = df[~df["dropback_type"].isin(SCRAMBLE_DROPBACK_TYPES)]

    df = assign_play_family(df)
    df["epa"] = df["expected_points_added"]
    df["success"] = df["epa"] > success_epa_threshold
    return df
