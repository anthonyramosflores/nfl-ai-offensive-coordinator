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


def build_success_rate_table(labeled_df, shell_key_cols=None, min_sample_size=MIN_SAMPLE_SIZE):
    """
    One row per (shell, play_family): how often that play family has
    succeeded historically against that shell.

    shell_key_cols lets this work for any dataset's shell definition (e.g.
    BDB 2025's [safety_shell, defensive_personnel, box_bucket, motion_flag]
    vs. BDB 2026's [coverage_shell, man_zone, box_bucket]) — the labeled_df
    passed in must already have those columns bucketed (see bucket_features
    for 2025, or src/features_2026.py for 2026).
    """
    shell_key_cols = shell_key_cols or SHELL_KEY_COLS
    if not set(shell_key_cols) <= set(labeled_df.columns) and "box_count" in labeled_df.columns:
        labeled_df = bucket_features(labeled_df)
    table = (
        labeled_df.groupby(shell_key_cols + ["play_family"])
        .agg(n_plays=("success", "size"), success_rate=("success", "mean"),
             avg_epa=("epa", "mean"))
        .reset_index()
    )
    table["reliable"] = table["n_plays"] >= min_sample_size
    return table


def _relaxation_stages(shell, shell_key_cols):
    """
    Progressively drop constraints, least-specific first, so a rare shell
    still gets an answer: exact match -> drop the last key -> drop the
    last two keys -> ... -> just the first (most important) key -> whatever
    generally works league-wide. Each stage is a dict of filters to apply.
    """
    yield "exact_shell", dict(shell)

    for keep_n in range(len(shell_key_cols) - 1, 0, -1):
        kept_cols = shell_key_cols[:keep_n]
        stage_name = "partial_" + "_".join(kept_cols)
        yield stage_name, {col: shell[col] for col in kept_cols}

    yield "league_wide", {}


def recommend_play(success_table, shell, shell_key_cols=None, min_sample_size=MIN_SAMPLE_SIZE):
    """
    Return the play_family with the best historical success rate against
    the given shell (a dict of {shell_key_col: value}), relaxing the match
    if the exact shell is too rare to trust. Returns None if the table has
    no data at all.

    shell_key_cols should list the shell's dimensions in priority order
    (most important to keep first) — see build_success_rate_table.
    """
    shell_key_cols = shell_key_cols or SHELL_KEY_COLS

    for match_level, filters in _relaxation_stages(shell, shell_key_cols):
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


def top_alternatives(success_table, shell, shell_key_cols=None, n=3,
                      min_sample_size=MIN_SAMPLE_SIZE):
    """The top-n play families for an exact shell match, for comparison."""
    shell_key_cols = shell_key_cols or SHELL_KEY_COLS
    subset = success_table
    for col in shell_key_cols:
        subset = subset[subset[col] == shell[col]]
    subset = subset[subset["n_plays"] >= min_sample_size]
    return subset.sort_values("success_rate", ascending=False).head(n)
