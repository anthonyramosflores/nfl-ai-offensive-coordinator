# nfl-ai-offensive-coordinator

Reads a defense's pre-snap alignment (personnel, box count, safety shell, LB
spacing, motion) and recommends the offensive play family with the best
historical success rate against that specific look.

## Setup

```bash
pip install -r requirements.txt
```

## Data

Uses [Big Data Bowl 2025](https://www.kaggle.com/competitions/nfl-big-data-bowl-2025/data)
(pre-snap player behavior). Download and unzip into `data/raw/` so it contains:

```
data/raw/games.csv
data/raw/plays.csv
data/raw/players.csv
data/raw/player_play.csv
data/raw/tracking_week_1.csv ... tracking_week_9.csv
```

```bash
kaggle competitions download -c nfl-big-data-bowl-2025 -p data/raw
unzip data/raw/nfl-big-data-bowl-2025.zip -d data/raw
```

## Pipeline stages

- **Stage A — ingestion & features** ([src/io.py](src/io.py), [src/features.py](src/features.py), [src/labels.py](src/labels.py)):
  raw tracking/play CSVs -> one row per play describing the defensive shell,
  joined with play_family + success/EPA outcome.
- **Stage A.5 — success-rate table** ([src/evaluate.py](src/evaluate.py)): a
  groupby lookup of historical success rate per (shell, play_family). This is
  the actual recommendation engine and the baseline any model needs to beat.
- **Stage B — sklearn baseline** ([src/baseline.py](src/baseline.py)): sanity
  check that features/labels are predictive before touching PyTorch.
- **Stage C — PyTorch classifier** ([src/model.py](src/model.py), [src/dataset.py](src/dataset.py), [src/train.py](src/train.py)):
  generalizes to shells too rare for the lookup table to trust.
- **Stage D — app** ([src/explain.py](src/explain.py), [app/app.py](app/app.py)):
  interactive Streamlit UI — pick a defensive shell, get a recommended play
  family, its historical success rate, and a plain-English reason.

## Tests

```bash
pytest tests/
```
