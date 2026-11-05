# Spotify Genre Classification 🎧

Machine learning pipeline for **multiclass music genre classification** using Spotify-derived audio features. The project compares CatBoost, LightGBM and ensembles, and explores the difference between optimizing **macro F1** and **MAP@3**.

**Authors:** Federico Del Cuadro, Bautista Goya and Timo Doeswijk  
**Course:** Tecnología Digital VI — Inteligencia Artificial (2026)

[**View the full notebook**](./TP2_TD6.ipynb)

## Project overview

Predict one of six genres — **acoustic, classical, electronic, hip-hop, jazz or rock** — from track characteristics such as energy, danceability, acousticness, instrumentalness, speechiness and tempo. The project was developed for an internal Kaggle competition, with an additional evaluation of top-three ranked predictions.

| Data split | Tracks |
| --- | ---: |
| Labeled training data | 4,200 |
| Unlabeled competition test data | 1,800 |
| **Total** | **6,000** |

The competition dataset is derived from the [Spotify Tracks Dataset](https://www.kaggle.com/datasets/maharshipandya/-spotify-tracks-dataset). The competition CSVs are not redistributed in this repository.

## Approach

1. **Exploratory analysis:** class balance, genre-specific audio-feature distributions, correlations and track duration outliers.
2. **Preprocessing:** remove the ID from predictors, encode target genres, treat `key` and `time_signature` as categorical variables, and use model-specific encoders/imputation. Preprocessing is fitted within the relevant training fold.
3. **Validation:** stratified 85/15 split into a development set (3,570 tracks) and a local holdout (630 tracks), followed by stratified cross-validation on development.
4. **Modeling and tuning:** CatBoost and LightGBM, with Optuna hyperparameter searches optimizing macro F1.
5. **Ensembling:** weighted soft voting plus a neighbor-based adjustment in standardized feature space.
6. **Business-oriented evaluation:** compare macro F1 against MAP@3, explore probability calibration, and reselect ensemble parameters for MAP@3.
7. **Final predictions:** train on all labeled examples and generate a Kaggle-formatted `submission.csv`.

## Results

### Five-fold cross-validation — development set

| Model | Macro F1 | MAP@3 |
| --- | ---: | ---: |
| CatBoost | 0.82994 | 0.89398 |
| LightGBM | 0.83055 | 0.89491 |
| Soft voting | 0.83308 | 0.89659 |
| **Final ensemble (voting + neighbors)** | **0.83511** | **0.89771** |

### Local holdout — 630 tracks

| Model | Macro F1 | MAP@3 |
| --- | ---: | ---: |
| CatBoost | 0.82570 | 0.89339 |
| LightGBM | 0.82068 | 0.89233 |
| Soft voting | 0.82572 | 0.89339 |
| **Final F1-oriented ensemble** | **0.83191** | **0.89735** |

A separate ensemble configuration selected for **MAP@3** achieved **0.89841 MAP@3** and **0.83356 macro F1** on the local holdout.

> These metrics come from the notebook's **local evaluation**, not from a Kaggle leaderboard or an independent production test.

## Why evaluate two metrics?

- **Macro F1:** appropriate when every track must receive one genre label; each genre contributes equally.
- **MAP@3:** evaluates a ranked list of up to three predicted genres. With one true label per track, a match at rank 1, 2 or 3 receives 1, 1/2 or 1/3, respectively.

MAP@3 can be informative for ranked genre suggestions. It does **not** directly measure recommendation relevance or user satisfaction.

## Tools

Python · pandas · NumPy · scikit-learn · CatBoost · LightGBM · Optuna · matplotlib · seaborn · Jupyter

## Run locally

1. Clone the repository:

   ```bash
   git clone https://github.com/fededelcuadro/spotify-genre-classification.git
   cd spotify-genre-classification
   ```

2. Create an environment and install dependencies:

   ```bash
   python -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```

   On Windows activate using `.venv\Scripts\activate`.

3. Place the course competition files `train.csv` and `test.csv` alongside `TP2_TD6.ipynb`. The public source dataset alone is **not** a direct substitute for these exact competition files.

4. Run `jupyter lab` and execute the notebook in order. Generated reports and the submission file are saved in `resultados_entrega/`.

**Reproducibility note:** The notebook defaults to `repetir_busqueda = False`, reusing recorded Optuna results. In this mode it checks the SHA-256 checksum of the original `train.csv`. With a different training file, set `repetir_busqueda = True` to repeat the searches; this can require substantial computation.

## Repository contents

```text
spotify-genre-classification/
├── TP2_TD6.ipynb      # Complete analysis and modeling workflow
├── README.md           # Project documentation
├── requirements.txt    # Python dependencies
└── .gitignore          # Excluded local and generated files
```

## Credits

Academic group project by **Federico Del Cuadro, Bautista Goya and Timo Doeswijk**.
