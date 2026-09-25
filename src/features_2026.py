"""
Defensive shell features for the Big Data Bowl 2026 Analytics data.

Unlike BDB 2025, this dataset has no isolated pre-snap tracking frame
(no frameType/event marker in the per-frame files), and supplementary_data.csv
covers pass plays only. So there's no geometry to derive here: box count and
coverage shell are already PFF-labeled columns in supplementary_data.csv —
this module just buckets them into the categories a user would pick in the
app. Safety count / LB spacing / motion aren't derivable from this data at
all (no run plays, no tagged pre-snap frame), so they're dropped for this
dataset rather than approximated.
"""
from src.evaluate import bucket_box_count

COVERAGE_SHELL_KEYWORDS = [
    ("0-high", ["COVER_0"]),
    ("1-high", ["COVER_1", "COVER_3"]),
    ("2-high", ["COVER_2", "COVER_4", "COVER_6"]),
    ("prevent", ["PREVENT"]),
]


def bucket_coverage_shell(team_coverage_type):
    """Map a PFF coverage label (e.g. 'COVER_3_ZONE') to a safety-shell
    bucket a coordinator would actually talk about ('1-high', '2-high')."""
    if not isinstance(team_coverage_type, str):
        return "unknown"
    coverage_upper = team_coverage_type.upper()
    for shell, keywords in COVERAGE_SHELL_KEYWORDS:
        if any(kw in coverage_upper for kw in keywords):
            return shell
    return "unknown"


def build_feature_table(supplementary_df):
    """One row per play, with the defensive shell already bucketed."""
    df = supplementary_df.copy()
    df["box_bucket"] = df["defenders_in_the_box"].apply(bucket_box_count)
    df["coverage_shell"] = df["team_coverage_type"].apply(bucket_coverage_shell)
    df["man_zone"] = df["team_coverage_man_zone"].fillna("unknown")
    return df
