# UFC Winner Predictor

End-to-end machine learning project that predicts UFC fight winners using pre-fight fighter data, baseline comparison, SHAP explainability, and a deployed Streamlit app.

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

- data cleaning and preprocessing
- comparative feature engineering
- temporal train/test split
- classification model training
- performance evaluation
- interactive Streamlit app

---

## Live Demo

- Streamlit app: [https://ufc-predictor.streamlit.app/](https://ufc-predictor.streamlit.app/)
- Repository: [https://github.com/lacpavan/ufcpredictor](https://github.com/lacpavan/ufcpredictor)

---

## Features

The model compares the red corner and blue corner using variables such as:

- age
- height
- reach
- wins and losses
- significant strike accuracy
- takedown accuracy
- fighting stance

Instead of analyzing each fighter in isolation, the model focuses on the matchup difference between both sides.

---

## Model

The current classifier is a `HistGradientBoostingClassifier` from `scikit-learn`, wrapped in a pipeline with median imputation for missing values.

Why this model:

- strong baseline for tabular data
- handles non-linear relationships well
- works well with mixed numerical feature sets after preprocessing

Current training setup:

- imputation: `SimpleImputer(strategy="median")`
- classifier: `HistGradientBoostingClassifier`
- validation strategy: temporal train/test split

The model implementation lives in `src/ufc_predictor/modeling.py`.

---

## Current performance

Using the current Kaggle-based dataset pipeline, the latest training report shows:

- selected model: `HistGradientBoostingClassifier`
- accuracy: `0.7466`
- ROC AUC: `0.8242`
- log loss: `0.5115`

These metrics are generated after training with a temporal split, so the model is evaluated on more recent fights instead of random shuffled rows.

---

## Baseline comparison

The project also compares the selected model against simpler baselines to make the result easier to interpret.

Latest comparison:

| Model | Accuracy | ROC AUC | Log loss |
|---|---:|---:|---:|
| DummyClassifier (most frequent) | `0.5641` | `0.5000` | `6.0221` |
| Logistic Regression | `0.6984` | `0.8200` | `0.5771` |
| Random Forest | `0.7363` | `0.8305` | `0.5348` |
| HistGradientBoostingClassifier | `0.7466` | `0.8242` | `0.5115` |

This helps show that the current model is not only producing a raw score, but also performing competitively against simpler alternatives.

---

## Dataset

The original dataset is not included in this repository because of file size.

To reproduce the project:

1. Access the dataset on Kaggle: [UFC DATASETS [1994-2025]](https://www.kaggle.com/datasets/neelagiriaditya/ufc-datasets-1994-2025)
2. Download and extract the `UFC.csv` file
3. Place the file in one of these locations:
   - `data/raw/UFC.csv`
   - `C:/Users/laris/Downloads/archive/UFC.csv`

> You will need a free Kaggle account to download the dataset.

---

## How to run

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
|-- README.md
|-- requirements.txt
|-- data
|-- docs
|-- notebooks
`-- src
    `-- ufc_predictor
```

Main files:

- `src/ufc_predictor/data.py`: data loading and transformation
- `src/ufc_predictor/features.py`: feature engineering
- `src/ufc_predictor/modeling.py`: training and evaluation
- `src/ufc_predictor/train.py`: end-to-end pipeline
- `app.py`: Streamlit interface
- `notebooks/01_eda_ufc.ipynb`: exploratory data analysis walkthrough

---

## Design decisions

- **Temporal data split**: training and test sets are separated by date, reducing information leakage risk.
- **Comparative features**: the model learns the difference between red and blue corner attributes.
- **Pre-fight variables only**: features are based on information available before the fight.

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

- compare two fighters
- generate a model prediction
- inspect estimated win probability for both sides
- inspect SHAP-based feature contributions for each prediction
- browse the processed dataset
- view the training report

---

## Exploratory analysis

To make the decision process more visible, the repository now includes an EDA notebook:

- `notebooks/01_eda_ufc.ipynb`

The notebook is meant to show:

- dataset shape and missing values
- target distribution
- core numeric summaries
- first checks before feature engineering and modeling

This helps make the project more transparent for technical reviewers and recruiters.

---

## Next steps

- improve feature engineering with historical rolling stats
- test additional models
- tune hyperparameters
- compare SHAP explanations across different fighter profiles
- deploy a production-ready API
