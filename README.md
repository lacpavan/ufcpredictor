# UFC Winner Predictor

 Machine Learning system for analyzing and predicting UFC fight outcomes based on historical data.

<p align="center">
  <img src="https://media.giphy.com/media/v1.Y2lkPTc5MGI3NjExMHBlMm1mcmI0dWR4cHJkaWk2a2poN2ZvNDN4aDM2eHI4Y2czZG03eCZlcD12MV9naWZzX3NlYXJjaCZjdD1n/N6o559ZkDsozTnQDcE/giphy.gif" width="600"/>
</p>

<p align="center">
  Predicting UFC fight outcomes using Machine Learning
</p>

<p align="center">
  <a href="https://ufc-predictor.streamlit.app/">Live App</a> •
  <a href="https://github.com/lacpavan/ufcpredictor">GitHub Repository</a>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10+-blue.svg"/>
  <img src="https://img.shields.io/badge/Machine%20Learning-Scikit--Learn-orange.svg"/>
  <img src="https://img.shields.io/badge/App-Streamlit-red.svg"/>
  <img src="https://img.shields.io/badge/Status-Active-success.svg"/>
</p>

---

## About

The **UFC Winner Predictor** estimates which fighter is more likely to win based on pre-fight information from both athletes.

The project was built to demonstrate an end-to-end ML workflow, from data preparation to model comparison, explainability, and interactive deployment.

The project includes:

- Data cleaning and preprocessing
- Comparative feature engineering
- Temporal train/test split
- Classification model training
- Performance evaluation
- Interactive Streamlit app

---

## Live Demo

- Streamlit app: [https://ufc-predictor.streamlit.app/](https://ufc-predictor.streamlit.app/)
- Repository: [https://github.com/lacpavan/ufcpredictor](https://github.com/lacpavan/ufcpredictor)

---

## API

The project also includes a production-style API built with `FastAPI`.

Main endpoints:

- `GET /health`: checks if the model is loaded correctly
- `POST /predict`: returns the predicted winner, probabilities, and optional SHAP-based explanations

Run locally:

```bash
uvicorn api.main:app --reload
```

Interactive docs:

- `http://127.0.0.1:8000/docs`

Example request:

```json
{
  "red_corner": {
    "name": "Alex Pereira",
    "age": 38,
    "height_cm": 193,
    "reach_cm": 201,
    "wins": 12,
    "losses": 3,
    "sig_str_acc": 0.62,
    "takedown_acc": 0.50,
    "stance": "Orthodox"
  },
  "blue_corner": {
    "name": "Ciryl Gane",
    "age": 35,
    "height_cm": 193,
    "reach_cm": 206,
    "wins": 13,
    "losses": 2,
    "sig_str_acc": 0.61,
    "takedown_acc": 0.25,
    "stance": "Orthodox"
  },
  "include_explanations": true
}
```

---

## Features

The model compares the red corner and blue corner using variables such as:

- Age
- Height
- Reach
- Wins and losses
- Significant strike accuracy
- Takedown accuracy
- Fighting stance

Instead of analyzing each fighter in isolation, the model focuses on the matchup difference between both sides.

---

## Model

The current classifier is a `HistGradientBoostingClassifier` from `scikit-learn`, wrapped in a pipeline with median imputation for missing values.

Why this model:

- Strong baseline for tabular data
- Handles non-linear relationships well
- Works well with mixed numerical feature sets after preprocessing

Current training setup:

- Imputation: `SimpleImputer(strategy="median")`
- Classifier: `HistGradientBoostingClassifier`
- Validation strategy: temporal train/test split

The model implementation lives in `src/ufc_predictor/modeling.py`.

---

## Current performance

Using the current Kaggle-based dataset pipeline, the latest training report shows:

- Selected Model: `HistGradientBoostingClassifier`
- Accuracy: `0.7466`
- ROC AUC: `0.8242`
- Log Loss: `0.5115`

These metrics are generated after training with a temporal split, so the model is evaluated on more recent fights instead of random shuffled rows.

---

## Baseline comparison

The project also compares the selected model against simpler baselines to make the result easier to interpret.

Latest comparison:

| Model | Accuracy | ROC AUC | Log Loss |
|---|---:|---:|---:|
| DummyClassifier (most frequent) | `0.5641` | `0.5000` | `6.0221` |
| Logistic Regression | `0.6984` | `0.8200` | `0.5771` |
| Random Forest | `0.7363` | `0.8305` | `0.5348` |
| HistGradientBoostingClassifier | `0.7466` | `0.8242` | `0.5115` |

This helps show that the current model is not only producing a raw score, but also performing competitively against simpler alternatives.

---

## Dataset

The repository includes the current dataset and trained artifacts used for the live demo.

Main data files:

- `data/raw/UFC.csv`
- `data/processed/modeling_dataset.csv`
- `models/ufc_winner_model.joblib`

If you want to reproduce the pipeline from scratch:

1. Access the dataset on Kaggle: [UFC DATASETS [1994-2025]](https://www.kaggle.com/datasets/neelagiriaditya/ufc-datasets-1994-2025)
2. Download and extract the `UFC.csv` file
3. Place the file at `data/raw/UFC.csv`

> You will need a free Kaggle account to download the dataset.

---

## How to run?

Install dependencies:

```bash
pip install -r requirements.txt
```

Train the model:

```bash
python -m src.ufc_predictor.train
```

Launch the application:

```bash
streamlit run app.py
```

---

## Project structure

```text
.
|-- app.py
|-- api
|-- README.md
|-- requirements.txt
|-- data
|-- docs
|-- notebooks
`-- src
    `-- ufc_predictor
```

Main files:

- `src/ufc_predictor/data.py`: Data loading and transformation
- `src/ufc_predictor/features.py`: fFature engineering
- `src/ufc_predictor/modeling.py`: Training and evaluation
- `src/ufc_predictor/train.py`: End-to-end pipeline
- `app.py`: Streamlit interface
- `api/main.py`: FastAPI application
- `notebooks/01_eda_ufc.ipynb`: Exploratory data analysis walkthrough

---

## Design decisions

- **Temporal data split**: training and test sets are separated by date, reducing information leakage risk.
- **Comparative features**: the model learns the difference between red and blue corner attributes.
- **Pre-fight variables only**: features are based on information available before the fight.

---

## Modeling Decisions

- Selected `HistGradientBoostingClassifier` because it performs well on tabular data, captures non-linear patterns, and delivered the best overall balance among the tested models.
- Feature engineering focused on matchup-based comparisons such as age difference, reach difference, win rate difference, and striking/takedown efficiency between red and blue corners.
- Trade-offs considered included interpretability vs. performance, historical realism vs. dataset availability, and keeping the model strong enough for prediction while still understandable through SHAP explanations and baseline comparisons.

---

## System Design

- Data pipeline: loads the UFC dataset, standardizes columns, prepares the target, and generates the modeling dataset.
- Model training: builds engineered features, applies temporal validation, compares baselines, trains the main classifier, and stores artifacts for inference.
- API layer: exposes prediction endpoints through FastAPI so the trained model can be consumed outside the Streamlit interface.

---

## Limitations

- The current dataset pipeline uses fighter statistics from `UFC.csv` without fully rebuilding them fight by fight from historical records.
- Because of that, some features may not perfectly represent the exact information that would have been available before each fight.
- The temporal split improves evaluation realism, but it does not fully eliminate this dataset limitation by itself.
- The next major improvement for the project is to reconstruct rolling fighter stats using fight history only up to each event date.

This limitation does not make the project useless, but it is important to state it clearly in a technical portfolio.

---

## App output

The app allows you to:

- Compare two fighters
- Generate a model prediction
- Inspect estimated win probability for both sides
- Inspect SHAP-based feature contributions for each prediction
- Browse the processed dataset
- View the training report

---

## Exploratory analysis

To make the decision process more visible, the repository now includes an EDA notebook:

- `notebooks/01_eda_ufc.ipynb`

The notebook is meant to show:

- Dataset shape and missing values
- Target distribution
- Core numeric summaries
- First checks before feature engineering and modeling

This helps make the project more transparent for technical reviewers and recruiters.

---

## Next steps

- Improve feature engineering with historical rolling stats
- Tune hyperparameters
- Expand SHAP analysis across multiple fighter profiles and matchup scenarios
- Deploy the FastAPI service separately as a standalone backend
