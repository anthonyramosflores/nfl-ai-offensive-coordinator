# nfl-ai-offensive-coordinator

Reads a defense's pre-snap shell (coverage, box count, man/zone) and
recommends the offensive play family with the best historical success rate
against that specific look.

## Setup

```bash
pip install -r requirements.txt
```

## Data

**Active dataset: [Big Data Bowl 2026 Analytics](https://www.kaggle.com/competitions/nfl-big-data-bowl-2026-analytics/data).**
Download from Kaggle (accept the competition rules first) and place under
`data/raw/` so it contains:

```
data/raw/supplementary_data.csv        # one row per pass play, PFF-labeled
data/raw/train/input_2023_w01.csv ...  # per-frame tracking (not used by the
                                        # shell/success-rate pipeline — see below)
```

```bash
kaggle competitions download -c nfl-big-data-bowl-2026-analytics -p data/raw
unzip data/raw/nfl-big-data-bowl-2026-analytics.zip -d data/raw
```

**Scope note:** BDB 2026's data covers pass plays only (verified against the
real data — every row has a pass result, no run plays, no sacks) and has no
tagged pre-snap tracking frame. So `play_family` here is a pass-depth
taxonomy (quick/intermediate/deep/screen/rollout), and the defensive shell is
built entirely from PFF columns already in `supplementary_data.csv`
(`team_coverage_type`, `team_coverage_man_zone`, `defenders_in_the_box`) —
no tracking geometry needed. See [src/features_2026.py](src/features_2026.py)
and [src/labels_2026.py](src/labels_2026.py).

There's also a dormant pipeline for **Big Data Bowl 2025** (pre-snap
tracking, run + pass, motion flags) in [src/io.py](src/io.py) /
[src/features.py](src/features.py) / [src/labels.py](src/labels.py) — fully
built and tested, just not wired to real data yet. Revisit it if the project
wants to cover run plays or true pre-snap geometry (safety depth, LB
spacing, motion) later.

## Pipeline stages (BDB 2026, active)

- **Features & labels** ([src/features_2026.py](src/features_2026.py), [src/labels_2026.py](src/labels_2026.py)):
  `supplementary_data.csv` -> one row per play with the defensive shell
  bucketed (`coverage_shell`, `man_zone`, `box_bucket`) and outcome labeled
  (`play_family`, `success`, `epa`). Scrambles and penalized plays are
  dropped as not-a-called-play.
- **Success-rate table** ([src/evaluate.py](src/evaluate.py)): a groupby
  lookup of historical success rate per (shell, play_family), with a
  fallback ladder that relaxes the shell match when a combination is too
  rare to trust (`min_sample_size`, default 10). This is the actual
  recommendation engine.
- **Stage B — sklearn baseline** ([src/baseline.py](src/baseline.py)): sanity
  check that features/labels are predictive before touching PyTorch.
- **Stage C — PyTorch classifier** ([src/model.py](src/model.py), [src/dataset.py](src/dataset.py), [src/train.py](src/train.py)):
  generalizes to shells too rare for the lookup table to trust.
- **Stage D — app** ([src/explain.py](src/explain.py), [app/app.py](app/app.py)):
  interactive Streamlit UI — pick a defensive shell, get a recommended play
  family, its historical success rate, and a plain-English reason.

## Quick check

```python
from src.io import load_supplementary
from src.labels_2026 import build_labeled_dataset
from src.features_2026 import build_feature_table
from src.evaluate import build_success_rate_table, recommend_play

featured = build_feature_table(build_labeled_dataset(load_supplementary()))
table = build_success_rate_table(
    featured, shell_key_cols=["coverage_shell", "man_zone", "box_bucket"]
)
print(recommend_play(
    table,
    shell={"coverage_shell": "2-high", "man_zone": "ZONE_COVERAGE", "box_bucket": "light"},
    shell_key_cols=["coverage_shell", "man_zone", "box_bucket"],
))
```

## Tests

```bash
pytest tests/
```
