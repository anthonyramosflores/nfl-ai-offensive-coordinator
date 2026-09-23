"""
The actual "coordinator" logic: given a defensive shell, what has
historically worked against it?

This is a groupby lookup table (empirical success rate per play_family,
per shell bucket), not a model. It's the baseline the classifier
(Stage B/C) needs to beat, and it's also the thing that ultimately
answers the app's question: "what beats THIS look, and how often has
it worked?" A model earns its place later by generalizing to shells
that are too rare in the data for this table to have a confident
answer for.
"""
MIN_SAMPLE_SIZE = 10

BOX_BUCKET_EDGES = [(5, "light"), (6, "standard"), (7, "heavy")]
BOX_BUCKET_DEFAULT = "stacked"

SHELL_KEY_COLS = ["safety_shell", "defensive_personnel", "box_bucket", "motion_flag"]


def bucket_box_count(box_count):
    for max_count, label in BOX_BUCKET_EDGES:
        if box_count <= max_count:
            return label
    return BOX_BUCKET_DEFAULT


def bucket_safety_count(num_high_safeties):
    if num_high_safeties >= 3:
        return "3+-high"
    return f"{num_high_safeties}-high"


def bucket_features(labeled_df):
    """Discretize continuous shell features into the categorical shell a
    user would actually pick in the app (e.g. '2-high', 'Base', 'light')."""
    df = labeled_df.copy()
    df["box_bucket"] = df["box_count"].apply(bucket_box_count)
    df["safety_shell"] = df["num_high_safeties"].apply(bucket_safety_count)
    return df


def build_success_rate_table(labeled_df, min_sample_size=MIN_SAMPLE_SIZE):
    """
    One row per (shell, play_family): how often that play family has
    succeeded historically against that shell.
    """
    bucketed = bucket_features(labeled_df)
    table = (
        bucketed.groupby(SHELL_KEY_COLS + ["play_family"])
        .agg(n_plays=("success", "size"), success_rate=("success", "mean"),
             avg_epa=("epa", "mean"))
        .reset_index()
    )
    table["reliable"] = table["n_plays"] >= min_sample_size
    return table


def _relaxation_stages(shell):
    """
    Progressively drop constraints (most to least specific) so a rare
    shell still gets an answer, from the most specific match down to
    "what generally works." Each stage is a dict of filters to apply.
    """
    full = dict(shell)
    yield "exact_shell", dict(full)

    no_motion = dict(full)
    no_motion.pop("motion_flag", None)
    yield "ignore_motion", no_motion

    no_box = dict(no_motion)
    no_box.pop("box_bucket", None)
    yield "ignore_motion_and_box", no_box

    safety_only = {"safety_shell": full["safety_shell"]}
    yield "safety_shell_only", safety_only

    yield "league_wide", {}


def recommend_play(success_table, safety_shell, defensive_personnel,
                    box_bucket, motion_flag, min_sample_size=MIN_SAMPLE_SIZE):
    """
    Return the play_family with the best historical success rate against
    the given shell, relaxing the match if the exact shell is too rare
    to trust. Returns None if the table has no data at all.
    """
    shell = {
        "safety_shell": safety_shell,
        "defensive_personnel": defensive_personnel,
        "box_bucket": box_bucket,
        "motion_flag": motion_flag,
    }

    for match_level, filters in _relaxation_stages(shell):
        subset = success_table
        for col, value in filters.items():
            subset = subset[subset[col] == value]
        reliable = subset[subset["n_plays"] >= min_sample_size]
        if not reliable.empty:
            best = reliable.sort_values("success_rate", ascending=False).iloc[0]
            return {
                "play_family": best["play_family"],
                "success_rate": float(best["success_rate"]),
                "avg_epa": float(best["avg_epa"]),
                "n_plays": int(best["n_plays"]),
                "match_level": match_level,
            }

    return None


def top_alternatives(success_table, safety_shell, defensive_personnel,
                      box_bucket, motion_flag, n=3, min_sample_size=MIN_SAMPLE_SIZE):
    """The top-n play families for an exact shell match, for comparison."""
    subset = success_table[
        (success_table["safety_shell"] == safety_shell)
        & (success_table["defensive_personnel"] == defensive_personnel)
        & (success_table["box_bucket"] == box_bucket)
        & (success_table["motion_flag"] == motion_flag)
        & (success_table["n_plays"] >= min_sample_size)
    ]
    return subset.sort_values("success_rate", ascending=False).head(n)
