# aac-recsys

<a target="_blank" href="https://cookiecutter-data-science.drivendata.org/">
    <img src="https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter" />
</a>

This project implements a **context-aware recommendation system for Augmentative and Alternative Communication (AAC)**, with a focus on **personalized pictogram recommendation per user**.
The pipeline is designed to be **modular, reproducible, and extensible**, allowing independent or integrated execution of **pre-processing**, **exploratory visualization**, and **model evaluation** stages.

## Project Organization

```
├── LICENSE            <- Open-source license if one is chosen
├── Makefile           <- Makefile with convenience commands like `make data` or `make train`
├── README.md          <- The top-level README for developers using this project.
├── data
│   ├── external       <- Data from third party sources.
│   ├── interim        <- Intermediate data that has been transformed.
│   ├── processed      <- The final, canonical data sets for modeling.
│   └── raw            <- The original, immutable data dump.
│
├── docs               <- A default mkdocs project; see www.mkdocs.org for details
│
├── models             <- Trained and serialized models, model predictions, or model summaries
│
├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
│                         the creator's initials, and a short `-` delimited description, e.g.
│                         `1.0-jqp-initial-data-exploration`.
│
├── pyproject.toml     <- Project configuration file with package metadata for
│                         aac_recsys and configuration for tools like black
│
├── references         <- Data dictionaries, manuals, and all other explanatory materials.
│
├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
│   └── figures        <- Generated graphics and figures to be used in reporting
│
├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
│                         generated with `pip freeze > requirements.txt`
│
├── setup.cfg          <- Configuration file for flake8
│
└── aac_recsys   <- Source code for use in this project.
    │
    ├── __init__.py             <- Makes aac_recsys a Python module
    │
    ├── config.py               <- Store useful variables and configuration
    │
    ├── dataset.py              <- Scripts to download or generate data
    │
    ├── features.py             <- Code to create features for modeling
    │
    ├── modeling
    │   ├── __init__.py
    │   ├── predict.py          <- Code to run model inference with trained models
    │   └── train.py            <- Code to train models
    │
    └── plots.py                <- Code to create visualizations
```

--------

## Usage Pipeline

The overall workflow of the project is composed of three main stages:

1. **Data Pre-processing**
2. **Visualization (optional)**
3. **Recommendation System Evaluation**

Each stage can be executed independently or orchestrated through the main entry point.

---

## 1. Data Pre-processing

In this stage, raw interaction logs are filtered, enriched with temporal and spatial context, and transformed into datasets ready for use in the recommendation system.

### The pre-processing stage includes:
- Cleaning invalid or incomplete records
- Temporal context extraction (hour, week, fuzzy time-of-day)
- Minimum activity filtering per user
- Per-user spatial clustering (DBSCAN)
- Generation of intermediate and final datasets

### Generated datasets:
- `df_filtered.parquet` — base filtered dataset
- `df_baseline.parquet` — minimal dataset for the baseline model
- `data_for_visualization.parquet` — dataset used for plotting
- `user_{i}/processed.parquet` — processed data per user

### Basic command

Run from the **project root**:

```bash
python -m aac_recsys.pre_processing
```

**Available arguments:**

| Argument           | Description                                         |
| :----------------- | :-------------------------------------------------- |
| `--user-idx`       | Process only a specific user (user_{idx})           |
| `--force`          | Force re-processing even if artifacts already exist |
| `--plots`          | Generate plots at the end of pre-processing         |
| `--plots-per-user` | Generate individual plots per user                  |

**Examples**

Run full pre-processing **(all users)**:
```bash
python -m aac_recsys.pre_processing
```

Force full re-processing:
```bash
python -m aac_recsys.pre_processing --force
```

Process a single user:
```bash
python -m aac_recsys.pre_processing --user-idx 3
```

## 2. Data Visualization (Plots)

This module generates spatial visualizations from the `data_for_visualization.parquet` dataset, enabling inspection of mobility patterns and interaction density.

### Available visualizations:
- Per-user heatmaps (hexbin)
- Per-user spatial cluster scatter plots
- Global scatter plot (users distinguished by color)
- Global density heatmap
- Visualizations restricted to the Top-N most active users

### Generated datasets:
- `df_filtered.parquet` — base filtered dataset
- `df_baseline.parquet` — minimal dataset for the baseline model
- `data_for_visualization.parquet` — dataset used for plotting
- `user_{i}/processed.parquet` — processed data per user

### Basic command

Run from the **project root**:

```bash
python -m aac_recsys.plots
```

**Available arguments:**

| Argument             | Description                                               |
| :------------------- | :-------------------------------------------------------- |
| `--clear`            | Clear the entire output directory before generating plots |
| `--per-user`         | Generate individual plots per user                        |
| `--clear-per-user`   | Clear only the per-user plots subdirectory                |
| `--top-n`            | Number of most active users used in global plots          |
| `--viz-path`         | Path to data_for_visualization.parquet                    |

**Examples**

Generate only global plots **(Top-15 users)**:
```bash
python -m aac_recsys.plots
```

Generate per-user plots and clear only that folder:
```bash
python -m aac_recsys.plots --per-user --clear-per-user
```

Clear everything and regenerate all plots:
```bash
python -m aac_recsys.plots --clear --per-user
```

## 2. Recommendation System Evaluation

The evaluation uses a rolling / expanding window strategy, ensuring that no future information is used during training.

### Computed metrics:
- Accuracy@1
- Macro F1@1
- Recall@K
- MRR@K

### Metrics are:
- computed per user and per fold
- aggregated by temporal evaluation window
- weighted by the number of test interactions

### Basic command

Run from the **project root**:

```bash
python -m aac_recsys.main --model baseline
```

**Main arguments:**

| Argument             | Description                                             |
| :------------------- | :------------------------------------------------------ |
| `--preprocess`       | Run pre-processing before evaluation                    |
| `--preprocess-force` | Force pre-processing                                    |
| `--preprocess-plots` | Generate plots during pre-processing                    |
| `--model`            | Model chosen (e.g., baseline, two-tower, random-forest) |
| `--predict-plots`    | Generate evaluation plots from prediction results       |
| `--min-train-days`   | Minimum training window size                            |
| `--max-train-days`   | Maximum training window size                            |
| `--test-days`        | Test window size                                        |
| `--step-days`        | Step size between windows                               |
| `--rank-k`           | Maximum number of ranked items                          |
| `--ks`               | K values for metrics (e.g., 1,3,5)                      |
| `--half-life-days`   | Half-life (in days) for exponential decay in ranker     |

**Full example (integrated pipeline)**

```bash
python -m aac_recsys.main \
  --preprocess \
  --model baseline \
  --min-train-days 60 \
  --max-train-days 180 \
  --test-days 7 \
  --step-days 7 \
  --rank-k 60 \
  --ks 1,3,5
```

**Main arguments:**

| Path / File                                     | Description                                         |
| :---------------------------------------------- | :-------------------------------------------------- |
| `data/processed/df_filtered.parquet`            | Base filtered dataset                               |
| `data/processed/df_baseline.parquet`            | Minimal dataset for baseline                        |
| `data/processed/user_{i}/processed.parquet`     | Per-user processed data                             |
| `data/processed/data_for_visualization.parquet` | Visualization dataset                               |
| `reports/predict_metrics_<model>.csv`           | Per-user, per-fold metrics                          |
| `reports/predict_user_summary_<model>.csv`      | Per-user aggregation across folds                   |
| `reports/figures/cluster_heatmaps/`             | Spatial visualizations (global + optional per-user) |
